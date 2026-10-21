"""Exercise bounded alternate full-text acquisition without network or model inference."""
from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

from radar.adapters.profiles import paper_identifiers
from radar.adapters.semantic_scholar import SemanticScholarAdapter
from radar.common.paper_evidence import PaperEvidenceService
from radar.common.evidence import merge_new_evidence
from radar.common.pdf import download_pdf_text


class PaperEvidenceTests(unittest.TestCase):
    """Protect missing-text lookup, PDF fallback identity, caching and forensic details."""

    def test_abstract_does_not_suppress_missing_full_text_lookup(self):
        """Recover a Chen-like candidate with an abstract and no original PDF URL."""
        work = {'doi': '10.1145/example', 'title': 'A conformal paper', 'abstract': 'Existing abstract.'}
        enricher = Mock()
        enricher.enrich.return_value = {'full_text_url': 'https://example.org/alternate.pdf', 'evidence_native_id': 'same-paper'}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True}, enricher)
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text': 'Paper text', 'full_text_status': 'downloaded'}) as download:
            first, second = service.prepare(work), service.prepare(work)
        self.assertEqual(first, second)
        self.assertEqual(first['full_text'], 'Paper text')
        self.assertEqual(first['full_text_url'], enricher.enrich.return_value['full_text_url'])
        enricher.enrich.assert_called_once_with(work)
        download.assert_called_once()
        self.assertEqual([entry['stage'] for entry in json.loads(first['full_text_attempts_json'])], ['lookup', 'download'])

    def test_failed_primary_pdf_uses_an_alternate_and_keeps_both_attempts(self):
        """Resolve an alternate after HTTP failure and clear stale failure fields on success."""
        work = {'doi': '10.1/fixture', 'pdf_url': 'https://example.org/blocked.pdf', 'full_text_error': 'HTTPError'}
        enricher = Mock()
        enricher.enrich.return_value = {'full_text_url': 'https://example.org/accepted.pdf'}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True}, enricher)
        failure = {'full_text_status': 'download_failed', 'full_text_error': 'HTTPError', 'full_text_error_detail': 'HTTP 403'}
        with patch('radar.common.paper_evidence.download_pdf_text', side_effect=[failure, {'full_text': 'Actual text', 'full_text_status': 'downloaded'}]) as download:
            result = service.prepare(work)
        self.assertEqual(download.call_count, 2)
        work.update(result)
        self.assertEqual(work['full_text_error'], '')
        audit = json.loads(result['full_text_attempts_json'])
        self.assertEqual([entry['stage'] for entry in audit], ['download', 'lookup', 'download'])
        self.assertEqual(audit[0]['detail'], 'HTTP 403')
        self.assertEqual(result['full_text_url'], 'https://example.org/accepted.pdf')

    def test_empty_pdf_is_not_a_success_and_duplicate_urls_are_not_retried(self):
        """An empty extraction can trigger lookup without fetching the same URL again."""
        url = 'https://example.org/empty.pdf'
        enricher = Mock()
        enricher.enrich.return_value = {'full_text_url': url}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True}, enricher)
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text': '   ', 'full_text_status': 'empty_pdf'}) as download:
            result = service.prepare({'pdf_url': url})
        download.assert_called_once()
        self.assertNotIn('full_text', result)
        self.assertEqual(result['full_text_status'], 'empty_pdf')
        enricher.enrich.assert_called_once()

    def test_arxiv_identifier_can_supply_a_pdf_without_external_lookup(self):
        """Use this paper's own arXiv DOI rather than any cited or benchmark identifier."""
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True}, Mock())
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text': 'Arxiv text'}) as download:
            result = service.prepare({'doi': '10.48550/arXiv.2408.01582'})
        self.assertEqual(result['full_text'], 'Arxiv text')
        download.assert_called_once_with('https://arxiv.org/pdf/2408.01582', 120000, 30)
        service.enricher.enrich.assert_not_called()

    def test_lookup_arxiv_identity_supplies_a_pdf_even_without_open_access_pdf(self):
        """Recover an arXiv manifestation returned by identity-matched metadata lookup."""
        enricher = Mock()
        enricher.enrich.return_value = {'evidence_identifiers': {'arxiv': '2408.01582'}, 'evidence_native_id': 'same-paper'}
        service = PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True}, enricher)
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text': 'Recovered text'}) as download:
            result = service.prepare({'doi': '10.1/published', 'abstract': 'An abstract'})
        self.assertEqual(result['full_text'], 'Recovered text')
        download.assert_called_once_with('https://arxiv.org/pdf/2408.01582', 120000, 30)

    def test_existing_text_and_disabled_fetch_do_not_trigger_fallback(self):
        """Respect usable evidence and acquisition policy without hidden network calls."""
        for config, work in [({'fetch_full_text': True, 'enrich_missing_full_text': True}, {'full_text': 'Available'}),
                ({'fetch_full_text': False, 'enrich_missing_full_text': True}, {'doi': '10.1/paper'}),
                ({'fetch_full_text': True, 'enrich_missing_full_text': False}, {'doi': '10.1/paper'})]:
            with self.subTest(config=config, work=work):
                enricher = Mock()
                service = PaperEvidenceService(config, enricher)
                with patch('radar.common.paper_evidence.download_pdf_text', side_effect=AssertionError('unexpected PDF download')):
                    service.prepare(work)
                enricher.enrich.assert_not_called()

    def test_uncertain_only_policy_still_fetches_full_text(self):
        """Allow configured uncertainty escalation while initial PDF fetching is disabled."""
        enricher = Mock()
        enricher.enrich.return_value = {'full_text_url': 'https://example.org/paper.pdf'}
        service = PaperEvidenceService({'fetch_full_text': False, 'enrich_uncertain_with_full_text': True}, enricher)
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text': 'Text'}) as download:
            result = service.enrich({'abstract': 'Abstract'})
        download.assert_called_once()
        self.assertEqual(result['full_text'], 'Text')

    def test_semantic_scholar_lookup_rejects_conflicting_doi_title_match(self):
        """A same-title paper with a conflicting DOI must not provide screening evidence."""
        client = Mock()
        client.get.side_effect = [requests.HTTPError('not found'), {'data': [{'paperId': 'wrong', 'title': 'Same title',
            'externalIds': {'DOI': '10.2/wrong'}, 'openAccessPdf': {'url': 'https://example.org/wrong.pdf'}}]}]
        adapter = SemanticScholarAdapter({'max_full_text_chars': 0}, Path(), client)
        result = adapter.enrich({'doi': '10.1/requested', 'title': 'Same title'})
        self.assertEqual(result['evidence_lookup_attempts'][-1]['status'], 'identity_conflict')
        self.assertNotIn('full_text_url', result)

    def test_semantic_scholar_uses_arxiv_identity_after_doi_lookup_failure(self):
        """Try supported strong IDs before falling back to title search."""
        client = Mock()
        client.get.side_effect = [requests.HTTPError('not found'), {'paperId': 'same', 'title': 'Paper',
            'externalIds': {'ArXiv': '2408.01582'}, 'openAccessPdf': {'url': 'https://example.org/paper.pdf'}}]
        adapter = SemanticScholarAdapter({'max_full_text_chars': 0}, Path(), client)
        with patch('radar.adapters.semantic_scholar.download_pdf_text', side_effect=AssertionError('adapter downloaded PDF')):
            result = adapter.enrich({'doi': '10.48550/arxiv.2408.01582'})
        self.assertEqual(client.get.call_count, 2)
        self.assertIn('ARXIV:2408.01582', client.get.call_args.args[0])
        self.assertEqual(result['evidence_identifiers']['arxiv'], '2408.01582')

    def test_pdf_errors_keep_http_status_and_non_pdf_details(self):
        """Make failed-source audits distinguish HTTP refusal from invalid returned content."""
        response = Mock(status_code=403)
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.raise_for_status.side_effect = requests.HTTPError(response=response)
        with patch('radar.common.pdf.requests.get', return_value=response):
            result = download_pdf_text('https://example.org/paper.pdf', 100)
        self.assertEqual(result['full_text_error_detail'], 'HTTP 403')
        response.raise_for_status.side_effect = None
        response.iter_content.return_value = [b'<html>landing page</html>']
        with patch('radar.common.pdf.requests.get', return_value=response):
            result = download_pdf_text('https://example.org/paper.pdf', 100)
        self.assertIn('non-PDF', result['full_text_error_detail'])

    def test_identifier_resolution_is_limited_to_the_candidate_itself(self):
        """Never turn cited-paper IDs or arbitrary URLs into evidence for this candidate."""
        self.assertEqual(paper_identifiers({'abstract': 'See arxiv 2408.01582', 'referenced_works': ['2408.01582']}), {})
        self.assertEqual(paper_identifiers({'primary_url': 'https://arxiv.org/abs/2408.01582v2'})['arxiv'], '2408.01582')
        with self.assertRaises(ValueError):
            paper_identifiers({'identifiers': {'arxiv': '../another-paper'}})


    def test_failed_retrieval_audit_is_saved_without_becoming_new_usable_evidence(self):
        """Recording a failed source must not restart an otherwise unchanged LLM assessment."""
        work = {'abstract': 'Existing abstract', 'full_text_error': 'HTTPError'}
        audit = {'full_text_status': 'download_failed', 'full_text_attempts_json': '[{"status":"failed"}]'}
        self.assertFalse(merge_new_evidence(work, audit))
        self.assertEqual(work['full_text_attempts_json'], audit['full_text_attempts_json'])
        self.assertTrue(merge_new_evidence(work, {'full_text': 'New paper text', 'full_text_error': ''}))
        self.assertEqual(work['full_text_error'], '')
