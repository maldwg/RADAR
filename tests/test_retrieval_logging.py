"""Distinguish unavailable text, provider failures and document responses in logs."""
import json
import logging
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests

from radar.adapters.openalex import OpenAlexAdapter
from radar.adapters.semantic_scholar import SemanticScholarAdapter
from radar.common.paper_evidence import PaperEvidenceService
from radar.common.evidence import merge_new_evidence
from radar.common.retrieval_diagnostics import lookup_failure, safe_url


def http_error(status):
    """Build a real HTTP failure without credentials in its diagnostic fields."""
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError('request failed https://api.test/?api_key=PRIVATE', response=response)


class RetrievalLoggingTests(unittest.TestCase):
    """Exercise logging and audits through offline provider/document boundaries."""

    def service(self, result=None):
        """Inject an offline metadata resolver behind the shared evidence service."""
        provider = Mock()
        provider.enrich.return_value = result or {}
        return PaperEvidenceService({'fetch_full_text': True, 'enrich_missing_full_text': True}, provider)

    def test_missing_links_are_not_logged_as_api_failures_and_doi_less_papers_are_identified(self):
        """A matched record without document links yields an informative non-error summary."""
        service = self.service({'evidence_native_id':'W123', 'abstract':'Available abstract',
            'evidence_lookup_attempts':[{'method':'identifier','status':'matched'}]})
        with self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            result = service.prepare({'openalex_id':'https://openalex.org/W123','title':'A title\nwith a newline'})
        logs = '\n'.join(captured.output)
        self.assertIn('outcome=record_found_no_document', logs)
        self.assertIn('outcome=no_document_link', logs)
        self.assertIn('paper_id="https://openalex.org/W123"', logs)
        self.assertIn('doi="-"', logs)
        self.assertIn('title="A title\\nwith a newline"', logs)
        self.assertIn('lookup_failures=0', logs)
        self.assertTrue(all(record.levelno == logging.INFO for record in captured.records))
        self.assertEqual(result['full_text_status'], 'no_pdf_url')

    def test_download_failure_and_fallback_success_both_name_the_paper_and_source(self):
        """HTTP refusal remains visible when a second source successfully supplies the text."""
        service = self.service({'full_text_url':'https://repository.test/paper.pdf'})
        failure = {'full_text_status':'download_failed','full_text_error':'HTTPError',
            'full_text_error_detail':'HTTP 403','full_text_http_status':403,'full_text_content_type':'text/html',
            'full_text_failure_reason':'http_error'}
        success = {'full_text_status':'downloaded','full_text':'Article body','full_text_format':'pdf','full_text_http_status':200}
        with patch('radar.common.paper_evidence.download_pdf_text',side_effect=[failure,success]), self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            result = service.prepare({'paper_id':'paper-1','doi':'10.1/fixture','title':'Paper','pdf_url':'https://publisher.test/blocked.pdf'})
        download_logs = [line for line in captured.output if 'download complete' in line]
        self.assertEqual(len(download_logs),2)
        self.assertTrue(all('paper_id="paper-1"' in line for line in download_logs))
        self.assertIn('provider=candidate',download_logs[0])
        self.assertIn('http_status=403',download_logs[0])
        self.assertIn('url=https://publisher.test/blocked.pdf',download_logs[0])
        summary = captured.output[-1]
        self.assertIn('outcome=retrieved',summary)
        self.assertIn('downloads=2 download_failures=1',summary)
        self.assertIn('url=https://repository.test/paper.pdf',summary)
        self.assertEqual(result['full_text_error'],'')
        self.assertEqual(result['full_text_failure_reason'],'')
        prior = {'full_text_failure_reason':'http_error'}
        self.assertTrue(merge_new_evidence(prior,result))
        self.assertEqual(prior['full_text_failure_reason'],'')

    def test_wrapped_api_errors_are_distinct_from_no_match_and_keep_http_status(self):
        """OpenAlex's RuntimeError wrapper cannot hide a rate-limit response."""
        failure = RuntimeError('OpenAlex wrapper')
        failure.__cause__ = http_error(429)
        service = self.service()
        service.enricher.enrich.side_effect = failure
        with self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            result = service.prepare({'paper_id':'rate-limited','title':'Paper'})
        logs = '\n'.join(captured.output)
        self.assertIn('request failed',logs)
        self.assertIn('http_status=429',logs)
        self.assertIn('outcome=lookup_failed',logs)
        self.assertIn('outcome=lookup_incomplete',logs)
        self.assertNotIn('PRIVATE',logs)
        self.assertEqual(json.loads(result['full_text_attempts_json'])[0]['attempts'][0]['http_status'],429)

    def test_http_404_does_not_become_a_network_failure(self):
        """An absent indexed record is a normal unsuccessful lookup, not an API outage."""
        service = self.service()
        service.enricher.enrich.side_effect = http_error(404)
        with self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            result = service.prepare({'paper_id':'not-indexed','title':'Paper'})
        logs = '\n'.join(captured.output)
        self.assertIn('outcome=no_match',logs)
        self.assertIn('lookup_failures=0',logs)
        self.assertNotIn('WARNING',logs)
        self.assertEqual(json.loads(result['full_text_attempts_json'])[0]['attempts'][0]['status'],'not_found')

    def test_non_pdf_response_logs_media_redirect_and_redacted_url(self):
        """A landing page reports HTTP 200/text-html without exposing tokens or page bodies."""
        response = Mock(status_code=200,headers={'Content-Type':'text/html; charset=utf-8'},
            url='https://publisher.test/login?token=PRIVATE_REDIRECT')
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [b'<html>PRIVATE_PAGE_BODY</html>']
        service = PaperEvidenceService({'fetch_full_text':True})
        work = {'paper_id':'html-response','pdf_url':'https://user:PRIVATE_PASSWORD@publisher.test/paper.pdf?api_key=PRIVATE_QUERY'}
        with patch('radar.common.pdf.requests.get',return_value=response), self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            result = service.prepare(work)
        logs = '\n'.join(captured.output)
        self.assertIn('http_status=200',logs)
        self.assertIn('content_type=text/html',logs)
        self.assertIn('reason=non_pdf_content',logs)
        self.assertIn('url=https://publisher.test/paper.pdf',logs)
        self.assertIn('final_url=https://publisher.test/login',logs)
        self.assertIn('outcome=download_failed',logs)
        self.assertNotIn('PRIVATE',logs)
        self.assertNotIn('PRIVATE',result['full_text_attempts_json'])
        self.assertEqual(result['full_text_content_type'],'text/html; charset=utf-8')

    def test_cached_attempts_are_identified_without_changing_saved_evidence(self):
        """Cache hits are visible while elapsed time/cache flags do not alter persisted content."""
        service = self.service({'full_text_url':'https://repository.test/paper.pdf'})
        with patch('radar.common.paper_evidence.download_pdf_text',return_value={'full_text':'Text','full_text_status':'downloaded'}) as download:
            first = service.prepare({'paper_id':'cached','doi':'10.1/paper'})
            with self.assertLogs('radar.paper_evidence',level='INFO') as captured:
                second = service.prepare({'paper_id':'cached','doi':'10.1/paper'})
        self.assertEqual(first,second)
        self.assertEqual(download.call_count,1)
        self.assertEqual(service.enricher.enrich.call_count,1)
        self.assertEqual(sum('cached=true' in line for line in captured.output),2)

    def test_existing_or_empty_text_have_distinct_final_outcomes(self):
        """Do not describe an empty extraction or retained prior text as a new successful download."""
        service = PaperEvidenceService({'fetch_full_text':True})
        with self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            service.prepare({'paper_id':'existing','full_text':'Already available'})
        self.assertIn('outcome=already_available',captured.output[-1])
        with patch('radar.common.paper_evidence.download_pdf_text',return_value={'full_text':'   ','full_text_status':'empty_document'}), self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            service.prepare({'paper_id':'empty','pdf_url':'https://repository.test/empty.pdf'})
        self.assertIn('outcome=no_extractable_text',captured.output[-1])

    def test_adapters_preserve_failed_identity_routes_even_after_successful_fallback(self):
        """Provider adapters must not erase API errors before the shared service sees them."""
        oa_client = Mock()
        oa_client.work.side_effect = http_error(429)
        oa_client.get.return_value = {'results':[{'id':'https://openalex.org/W123','title':'Paper'}]}
        oa = OpenAlexAdapter({},Path(),oa_client).enrich({'doi':'10.1/paper','title':'Paper'})
        self.assertEqual(oa['evidence_lookup_attempts'][0]['http_status'],429)
        self.assertEqual(oa['evidence_lookup_attempts'][-1]['status'],'matched')
        self.assertNotIn('evidence_lookup_error',oa)
        with self.assertLogs('radar.paper_evidence',level='INFO') as captured:
            self.service(oa).prepare({'paper_id':'recovered-api','doi':'10.1/paper','title':'Paper'})
        self.assertIn('outcome=no_document_link',captured.output[-1])
        self.assertIn('lookup_failures=0 lookup_request_failures=1',captured.output[-1])
        s2_client = Mock()
        s2_client.get.side_effect = [http_error(503), {'data':[]}]
        s2 = SemanticScholarAdapter({'max_full_text_chars':0},Path(),s2_client).enrich({'doi':'10.1/paper','title':'Paper'})
        self.assertEqual(s2['evidence_lookup_attempts'][0]['http_status'],503)
        self.assertEqual(s2['evidence_lookup_error'],'HTTPError')

    def test_safe_diagnostics_do_not_echo_credential_bearing_exception_strings(self):
        """Logs retain actionable status/class information and redact request secrets."""
        self.assertEqual(safe_url('https://u:secret@example.test/p.pdf?api_key=secret#fragment'),'https://example.test/p.pdf')
        self.assertNotIn('PRIVATE',json.dumps(lookup_failure(http_error(403))))
        self.assertEqual(lookup_failure(requests.Timeout('PRIVATE URL'))['detail'],'Timeout')


if __name__ == '__main__':
    unittest.main()
