"""Exercise cached full text, alternate locations, provider ordering and credential boundaries."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

from radar.adapters.openalex import OpenAlexAdapter
from radar.common.paper_evidence import PaperEvidenceService
from radar.common.config import ConfigError, load_settings
from radar.common.pdf import _tei_text, download_pdf_text, download_tei_text
from radar.openalex.collect import OpenAlexClient, normalize_work


class OpenAlexFullTextTests(unittest.TestCase):
    """Protect identity-matched alternate sources and parsed full-text evidence."""

    def test_normalization_retains_cached_content_and_all_pdf_locations(self):
        """Keep repository URLs distinct from cache URLs and reject cache links for another work."""
        work = normalize_work({'id': 'https://openalex.org/W123', 'title': 'Paper',
            'primary_location': {'pdf_url': 'https://publisher.test/blocked.pdf'},
            'locations': [{'pdf_url': 'https://repository.test/paper.pdf'}, {'pdf_url': 'https://publisher.test/blocked.pdf'}],
            'has_content': {'pdf': True, 'grobid_xml': True},
            'content_urls': {'pdf': 'https://content.openalex.org/works/W123.pdf?api_key=not-to-be-stored',
                'grobid_xml': 'https://content.openalex.org/works/W999.grobid-xml'}})
        self.assertEqual(work['pdf_urls'], ['https://publisher.test/blocked.pdf', 'https://repository.test/paper.pdf'])
        self.assertEqual(work['content_urls'], {'pdf': 'https://content.openalex.org/works/W123.pdf'})

    def test_openalex_doi_lookup_keeps_the_whole_identifier(self):
        """External DOIs must not be reduced to their final slash-delimited component."""
        client = OpenAlexClient(request_pause_seconds=0)
        with patch.object(client, 'get', return_value={'id': 'https://openalex.org/W123'}) as get:
            client.work('https://doi.org/10.1234/article/supplement')
        self.assertEqual(get.call_args.args[0], '/works/doi:10.1234/article/supplement')

    def test_openalex_enrichment_advertises_and_resolves_evidence(self):
        """Use a known work's metadata for cached and alternate sources without downloading in the adapter."""
        client = Mock()
        client.work.return_value = {'id': 'https://openalex.org/W123', 'doi': 'https://doi.org/10.1/same',
            'content_urls': {'pdf': 'https://content.openalex.org/works/W123.pdf'}}
        adapter = OpenAlexAdapter({}, Path(), client)
        result = adapter.enrich({'doi': '10.1/same'})
        self.assertIn('evidence', adapter.descriptor.capabilities)
        self.assertEqual(result['content_urls']['pdf'], 'https://content.openalex.org/works/W123.pdf')
        self.assertEqual(result['evidence_native_id'], 'https://openalex.org/W123')

    def test_openalex_enrichment_rejects_a_conflicting_paper(self):
        """A successful HTTP response must still identify the candidate before supplying its full text."""
        client = Mock()
        client.work.return_value = {'id': 'https://openalex.org/W999', 'doi': 'https://doi.org/10.1/other'}
        result = OpenAlexAdapter({}, Path(), client).enrich({'doi': '10.1/requested'})
        self.assertEqual(result['evidence_lookup_attempts'][-1]['status'], 'identity_conflict')
        self.assertNotIn('content_urls', result)

    def test_provider_chain_prefers_cached_xml_before_attempting_a_pdf(self):
        """Structured full text takes priority even when the candidate already has a PDF URL."""
        openalex, semantic = Mock(), Mock()
        openalex.enrich.return_value = {'content_urls': {'pdf': 'https://content.openalex.org/works/W123.pdf',
            'grobid_xml': 'https://content.openalex.org/works/W123.grobid-xml'}, 'evidence_provider': 'openalex'}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True,
            'full_text_lookup_providers': ['openalex', 'semantic_scholar']})
        failure = {'full_text_status': 'download_failed', 'full_text_error': 'HTTPError', 'full_text_error_detail': 'HTTP 403'}
        with patch('radar.adapters.registry.create_adapter', side_effect=[openalex, semantic]) as factory, patch(
                'radar.common.paper_evidence.download_pdf_text', side_effect=AssertionError('PDF used before TEI')), patch(
                'radar.common.paper_evidence.download_tei_text', return_value={'full_text': 'Recovered article body', 'full_text_format': 'tei_xml'}) as xml:
            result = service.prepare({'doi': '10.1/paper', 'pdf_url': 'https://publisher.test/blocked.pdf'})
        self.assertEqual(result['full_text'], 'Recovered article body')
        self.assertEqual(result['full_text_format'], 'tei_xml')
        self.assertEqual(factory.call_count, 1)
        semantic.enrich.assert_not_called()
        xml.assert_called_once()
        self.assertEqual([entry['stage'] for entry in json.loads(result['full_text_attempts_json'])], ['lookup', 'download'])

    def test_provider_chain_continues_to_semantic_scholar(self):
        """OpenAlex lookup exhaustion must not hide a second provider's working PDF."""
        openalex, semantic = Mock(), Mock()
        openalex.enrich.return_value = {}
        semantic.enrich.return_value = {'full_text_url': 'https://repository.test/paper.pdf', 'evidence_provider': 'semantic_scholar'}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True})
        with patch('radar.adapters.registry.create_adapter', side_effect=[openalex, semantic]) as factory, patch(
                'radar.common.paper_evidence.download_pdf_text', return_value={'full_text': 'Article text'}):
            first, second = service.prepare({'doi': '10.1/paper'}), service.prepare({'doi': '10.1/paper'})
        self.assertEqual(first, second)
        self.assertEqual([call.args[0] for call in factory.call_args_list], ['openalex', 'semantic_scholar'])
        openalex.enrich.assert_called_once()
        semantic.enrich.assert_called_once()
        self.assertEqual(first['full_text'], 'Article text')

    def test_structured_format_priority_precedes_pdf_only_provider_order(self):
        """A structured-capable provider takes priority over a PDF-only provider."""
        semantic = Mock()
        semantic.enrich.return_value = {'full_text': 'Resolved by selected provider'}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True,
            'full_text_lookup_providers': ['semantic_scholar', 'openalex']})
        with patch('radar.adapters.registry.create_adapter', return_value=semantic) as factory:
            service.prepare({'doi': '10.1/paper'})
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(factory.call_args.args[0], 'openalex')

    def test_failed_expansion_keeps_the_existing_text_source(self):
        """Failed alternate acquisition must not relabel already usable partial evidence."""
        service = PaperEvidenceService({'enrich_uncertain_with_full_text': True}, Mock())
        service.enricher.enrich.return_value = {'full_text_url': 'https://other.test/unavailable.pdf'}
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text_status': 'download_failed'}):
            result = service.enrich({'full_text': 'Existing partial text', 'full_text_complete': False,
                'full_text_url': 'https://original.test/paper.pdf', 'full_text_status': 'downloaded'})
        self.assertNotIn('full_text_url', result)
        self.assertNotIn('full_text_status', result)
        self.assertIn('full_text_attempts_json', result)

    def test_tei_extracts_article_body_and_preserves_inline_mathematics(self):
        """Modern TEI keeps inline words and operators while excluding bibliography content."""
        data = ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><profileDesc><abstract><p>Abstract text.</p></abstract></profileDesc></teiHeader>'
            '<text><body><div><head>Method</head><p>We estimate <hi>treatment</hi> effects at α ≤ 0.05.</p>'
            '<formula>x − y = 1</formula></div></body><back><div type="references"><listBibl><p>Cited paper text.</p></listBibl></div><div type="annex"><p>Appendix proof.</p></div></back></text></TEI>').encode()
        text = _tei_text(data)
        self.assertIn('Abstract text.', text)
        self.assertIn('We estimate treatment effects at α ≤ 0.05.', text)
        self.assertIn('x − y = 1', text)
        self.assertNotIn('Cited paper text.', text)
        self.assertIn('Appendix proof.', text)

    def test_historical_wrapped_tei_and_invalid_documents(self):
        """Support the documented older shape without accepting a landing page or entity expansion."""
        self.assertEqual(_tei_text(b'<html><body><tei><text><div><p>Article text.</p></div></text></tei></body></html>'), 'Article text.')
        for data in (b'<html><body>Publisher landing page</body></html>', b'<TEI><teiHeader><abstract>Abstract only</abstract></teiHeader></TEI>',
                b'<!DOCTYPE TEI [<!ENTITY x "text">]><TEI><text><body><p>&x;</p></body></text></TEI>'):
            with self.subTest(data=data), self.assertRaises(ValueError):
                _tei_text(data)

    def test_tei_download_handles_gzip_and_bounds_extracted_evidence(self):
        """Preserve source provenance while clipping only the evidence window."""
        data = b'<TEI><text><body><div><p>Long article body.</p></div></body></text></TEI>'
        with patch('radar.common.pdf._download_bytes', return_value=gzip.compress(data)):
            result = download_tei_text('https://content.openalex.org/works/W123.grobid-xml', 7)
        self.assertEqual(result['full_text'], 'Long ar')
        self.assertFalse(result['full_text_complete'])
        self.assertEqual(result['full_text_format'], 'tei_xml')
        self.assertTrue(result['full_text_source_sha256'])

    def test_cached_content_authentication_never_leaks_to_publisher_requests_or_audits(self):
        """Send the existing key to the content host only and keep saved URLs credential-free."""
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [b'%PDF-test']
        document = Mock(pages=[Mock(extract_text=Mock(return_value='Source text'))])
        with patch.dict(os.environ, {'OPENALEX_API_KEY': 'test-private-key'}), patch('radar.common.pdf.requests.get', return_value=response) as get, patch(
                'pypdf.PdfReader', return_value=document):
            result = download_pdf_text('https://content.openalex.org/works/W123.pdf', 100)
            self.assertEqual(get.call_args.kwargs['params'], {'api_key': 'test-private-key'})
            download_pdf_text('https://publisher.test/paper.pdf', 100)
            self.assertIsNone(get.call_args.kwargs['params'])
        self.assertNotIn('test-private-key', json.dumps(result))
        with patch.dict(os.environ, {}, clear=True), patch('radar.common.pdf.requests.get') as get:
            result = download_pdf_text('https://content.openalex.org/works/W123.pdf', 100)
        get.assert_not_called()
        self.assertEqual(result['full_text_status'], 'api_key_required')


    def test_provider_configuration_rejects_duplicates_and_unsupported_capabilities(self):
        """Validate the full-text chain before any provider or document request can run."""
        original = Path(__file__).resolve().parents[1] / 'config'
        for providers in ('["openalex", "openalex"]', '["github"]', '["unregistered"]'):
            with self.subTest(providers=providers), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                shutil.copytree(original, root / 'config')
                study = root / 'config/studies/conformal_treatment_effects'
                pipeline = study / 'pipeline.toml'
                pipeline.write_text(pipeline.read_text().replace('["openalex", "semantic_scholar"]', providers))
                with self.assertRaises(ConfigError):
                    load_settings(study / 'study.toml')

    def test_gzip_xml_limits_and_head_only_parses_are_not_usable_full_text(self):
        """Reject decompression bombs and header-only parses before they reach screening."""
        with patch('radar.common.pdf._download_bytes', return_value=gzip.compress(b'x' * 100)), patch('radar.common.pdf.MAX_CONTENT_BYTES', 50):
            result = download_tei_text('https://content.openalex.org/works/W123.grobid-xml', 120000)
        self.assertEqual(result['full_text_status'], 'download_failed')
        self.assertIn('exceeds', result['full_text_error_detail'])
        with self.assertRaises(ValueError):
            _tei_text(b'<TEI><text><body><div><head>Title only</head></div></body></text></TEI>')
