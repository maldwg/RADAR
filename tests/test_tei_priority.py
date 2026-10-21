"""Exercise shared structured-first retrieval independently of discovery provider."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from radar.adapters.contracts import ProviderDescriptor
from radar.adapters.openalex import OpenAlexAdapter
from radar.adapters.registry import PROVIDERS, ProviderRegistration, register_provider
from radar.common.evidence_blocks import EvidenceBlockIndex
from radar.common.paper_evidence import FULL_TEXT_RETRIEVAL_POLICY, PaperEvidenceService
from radar.common.pdf import ContentKeyRequired
from radar.common.study_consensus import ensemble_study
from radar.common.study_pipeline import classify_study
from tests.test_study_pipeline import FakeModel, TEXT

XML_URL = 'https://content.openalex.org/works/W123.grobid-xml'
PDF_URL = 'https://repository.test/paper.pdf'
TEI = b'<TEI><text><body><div><head>Methods</head><p>First paragraph.</p><p>Separate evidence.</p><formula>x = 1</formula></div></body></text></TEI>'
ROOT = Path(__file__).resolve().parents[1]


class TEIPriorityTests(unittest.TestCase):
    """Protect TEI preference, identity-matched cross-provider lookup and PDF fallback."""

    def test_semantic_scholar_candidate_resolves_openalex_tei_before_its_working_pdf(self):
        """A Semantic Scholar source uses OpenAlex's structured parse through the shared service."""
        client = Mock()
        client.work.return_value = {'id':'https://openalex.org/W123','doi':'https://doi.org/10.1/same',
            'title':'Paper','content_urls':{'grobid_xml':XML_URL}}
        adapter = OpenAlexAdapter({},Path(),client)
        service = PaperEvidenceService({'fetch_full_text':True,'enrich_missing_full_text':True})
        service.enrichers['openalex'] = adapter
        work = {'paper_id':'doi:10.1/same','doi':'10.1/same','title':'Paper','pdf_url':PDF_URL,
            'source_memberships':[{'provider':'semantic_scholar','native_id':'s2-paper'}]}
        with patch('radar.common.pdf._download_bytes',return_value=TEI), patch(
                'radar.common.paper_evidence.download_pdf_text',side_effect=AssertionError('working PDF should remain unused')), patch(
                'radar.adapters.registry.create_adapter',side_effect=AssertionError('unneeded provider request')):
            result = service.prepare(work)
        client.work.assert_called_once_with('doi:10.1/same')
        self.assertEqual(result['full_text_format'],'tei_xml')
        self.assertEqual(result['full_text_url'],XML_URL)
        self.assertEqual(result['full_text_source_sha256'],hashlib.sha256(TEI).hexdigest())
        index = EvidenceBlockIndex.from_record(result)
        self.assertEqual([b.kind for b in index.blocks],['head','paragraph','paragraph','formula'])
        self.assertTrue(all(b.section == 'Methods' for b in index.blocks))
        self.assertTrue(all(result['full_text'][b.start:b.end] == b.text for b in index.blocks))

    def test_supplied_tei_requires_no_metadata_lookup_and_no_pdf_extraction(self):
        """Use already supplied structured content before resolving any other provider."""
        service = PaperEvidenceService({'fetch_full_text':True,'enrich_missing_full_text':True})
        with patch('radar.common.pdf._download_bytes',return_value=TEI), patch(
                'radar.adapters.registry.create_adapter',side_effect=AssertionError('unneeded lookup')), patch(
                'radar.common.paper_evidence.download_pdf_text',side_effect=AssertionError('unneeded PDF')):
            result = service.prepare({'paper_id':'native','content_urls':{'pdf':PDF_URL,'grobid_xml':XML_URL}})
        self.assertEqual(result['full_text_format'],'tei_xml')
        self.assertEqual([a['stage'] for a in json.loads(result['full_text_attempts_json'])],['download'])

    def test_unavailable_tei_falls_back_to_supplied_pdf_without_duplicate_attempts(self):
        """A missing content key does not prevent use of a public PDF for the same work."""
        oa = Mock()
        oa.enrich.return_value = {'content_urls':{'grobid_xml':XML_URL}}
        service = PaperEvidenceService({'fetch_full_text':True,'enrich_missing_full_text':True})
        service.enrichers['openalex'] = oa
        with patch('radar.common.pdf._download_bytes',side_effect=ContentKeyRequired('OpenAlex cached content requires OPENALEX_API_KEY')), patch(
                'radar.common.paper_evidence.download_pdf_text',return_value={'full_text':'PDF fallback','full_text_format':'pdf','full_text_status':'downloaded'}) as pdf, patch(
                'radar.adapters.registry.create_adapter',side_effect=AssertionError('unneeded Semantic Scholar request')):
            result = service.prepare({'paper_id':'one','pdf_url':PDF_URL,'content_urls':{'grobid_xml':XML_URL}})
        pdf.assert_called_once()
        attempts = json.loads(result['full_text_attempts_json'])
        self.assertEqual(sum(a.get('url') == XML_URL for a in attempts),1)
        self.assertEqual(attempts[0]['status'],'api_key_required')
        self.assertEqual(result['full_text_format'],'pdf')
        self.assertEqual(result['full_text_failure_reason'],'')

    def test_malformed_or_abstract_only_xml_can_use_pdf_without_external_lookup(self):
        """Reject unusable structured responses before invoking the shared pypdf fallback."""
        for xml in (b'<html>Publisher page</html>', b'<TEI><teiHeader><abstract>Only abstract</abstract></teiHeader></TEI>'):
            with self.subTest(xml=xml):
                service = PaperEvidenceService({'fetch_full_text':True,'full_text_lookup_providers':[]})
                with patch('radar.common.pdf._download_bytes',return_value=xml), patch(
                        'radar.common.paper_evidence.download_pdf_text',return_value={'full_text':'PDF body','full_text_format':'pdf'}) as pdf:
                    result = service.prepare({'content_urls':{'grobid_xml':XML_URL},'pdf_url':PDF_URL})
                pdf.assert_called_once()
                self.assertEqual(result['full_text_format'],'pdf')
                self.assertEqual(json.loads(result['full_text_attempts_json'])[0]['status'],'download_failed')

    def test_pdf_only_provider_remains_lazy_after_a_successful_known_pdf(self):
        """Checking for structured text does not add unnecessary Semantic Scholar requests."""
        oa = Mock()
        oa.enrich.return_value = {'evidence_native_id':'W123'}
        service = PaperEvidenceService({'fetch_full_text':True,'enrich_missing_full_text':True})
        service.enrichers['openalex'] = oa
        with patch('radar.common.paper_evidence.download_pdf_text',return_value={'full_text':'PDF body','full_text_format':'pdf'}), patch(
                'radar.adapters.registry.create_adapter',side_effect=AssertionError('unneeded provider')):
            result = service.prepare({'paper_id':'one','pdf_url':PDF_URL})
        oa.enrich.assert_called_once()
        self.assertEqual(result['full_text_format'],'pdf')

    def test_new_structured_provider_uses_capability_metadata_instead_of_provider_name(self):
        """A registered TEI provider participates in the same priority tier as OpenAlex."""
        descriptor = ProviderDescriptor('other_archive',('paper',),('evidence',),evidence_formats=('tei_xml',))
        provider = Mock()
        provider.enrich.return_value = {'content_urls':{'grobid_xml':XML_URL}}
        with patch.dict(PROVIDERS,{'other_archive':ProviderRegistration(descriptor,Mock())}):
            service = PaperEvidenceService({'fetch_full_text':True,'enrich_missing_full_text':True,
                'full_text_lookup_providers':['semantic_scholar','other_archive']})
            with patch('radar.adapters.registry.create_adapter',return_value=provider) as factory, patch(
                    'radar.common.pdf._download_bytes',return_value=TEI), patch(
                    'radar.common.paper_evidence.download_pdf_text',side_effect=AssertionError('PDF before TEI')):
                result = service.prepare({'pdf_url':PDF_URL})
        self.assertEqual(factory.call_count,1)
        self.assertEqual(factory.call_args.args[0],'other_archive')
        self.assertEqual(result['full_text_format'],'tei_xml')

    def test_registry_rejects_unsupported_or_inconsistent_format_declarations(self):
        """Capabilities must not promise document formats the shared extractor cannot handle."""
        for formats,caps in ((('tei_xml',),('discovery',)), (('other',),('evidence',)), (('pdf','pdf'),('evidence',))):
            with self.subTest(formats=formats), self.assertRaises(ValueError):
                register_provider(ProviderDescriptor('invalid-format',('paper',),caps,evidence_formats=formats),Mock())

    def test_review_checkpoints_and_ensembles_pin_the_retrieval_policy(self):
        """Completed results from different text-selection policies cannot form one ensemble."""
        protocol = ROOT/'config/studies/conformal_treatment_effects/screening.toml'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'input.json'
            source.write_text(json.dumps([{'paper_id':'one','full_text':TEXT}]))
            outputs = {'one':root/'one.csv','two':root/'two.csv'}
            for model,path in outputs.items():
                classify_study(protocol,source,path,{'model':model},client=FakeModel())
                data = json.loads(path.with_suffix('.assessments.json').read_text())
                self.assertEqual(data['evidence_retrieval_policy'],FULL_TEXT_RETRIEVAL_POLICY)
            self.assertEqual(ensemble_study(protocol,outputs,root/'ensemble.csv',1)[0]['decision'],'accept')
            checkpoint = outputs['two'].with_suffix('.assessments.json')
            data = json.loads(checkpoint.read_text())
            data['evidence_retrieval_policy'] = 'pdf-first'
            checkpoint.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'retrieval policy'):
                ensemble_study(protocol,outputs,root/'mixed.csv',1)
