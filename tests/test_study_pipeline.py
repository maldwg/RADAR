"""Exercise blind discovery, model evidence validation and resumable study runs."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ids_meta.common.study_pipeline import classify_study, collect_study

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/studies/conformal_treatment_effects"
TEXT = "We propose an original conformal treatment-effect method. The method requires SUTVA, ignorability and overlap."


def answer() -> dict:
    """Construct a complete, evidence-backed model response fixture."""
    assessment = lambda value: {"value": value, "quote": TEXT if value != "unknown" else "", "source": "full_text" if value != "unknown" else ""}
    return {"criteria": {**{f"F{i}": assessment("yes") for i in range(1, 6)},
        **{f"Q{i}": assessment("yes" if i <= 5 else "unknown") for i in range(1, 10)}},
        "fields": {field: assessment("required") for field in ("sutva", "strong_ignorability", "overlap")}}


class FakeModel:
    def __init__(self, response=None):
        """Initialize the adapter or fixture state."""
        self.calls = []
        self.response = response or answer()

    def generate(self, prompt, schema):
        """Return a recorded model response for the test."""
        self.calls.append((prompt, schema))
        return copy.deepcopy(self.response)


class FakeSearch:
    select = "id,title"

    def __init__(self):
        """Initialize the adapter or fixture state."""
        self.calls = []

    def get(self, path, params):
        """Return a controlled discovery page for the test."""
        self.calls.append((path, params))
        ids = ["W2", "W1"] if 'individual OR patient' in params['search.title_and_abstract.exact'] else ["W1", "W3"]
        return {"meta": {"count": 3, "next_cursor": "next"}, "results": [
            {"id": f"https://openalex.org/{key}", "title": f"Paper {key}", "publication_year": 2024} for key in ids]}


class StudyPipelineTests(unittest.TestCase):
    def test_collection_deduplicates_without_reference_seeding_and_records_caps(self):
        """Verify collection deduplicates without reference seeding and records caps."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidates.json"
            client = FakeSearch()
            rows = collect_study(PROFILE / "search/openalex.toml", output, limit=2, per_query_limit=2, client=client)
            self.assertEqual([r['paper_id'] for r in rows], ['https://openalex.org/W2', 'https://openalex.org/W1'])
            self.assertEqual(rows[1]['discovery_queries'], ['picoc', 'core_concepts'])
            audit = json.loads(output.with_suffix('.collection.json').read_text())
            self.assertEqual(audit['pool_size'], 3)
            self.assertTrue(all(q['coverage'] == 'bounded' and q['scanned'] == 2 for q in audit['queries']))
            self.assertTrue(all('from_publication_date:2015-01-01,to_publication_date:2025-09-25' == c[1]['filter'] for c in client.calls))

    def test_model_is_blind_to_reference_fields_and_resume_avoids_inference(self):
        """Verify model is blind to reference fields and resume avoids inference."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'candidates.json', root / 'classifications.csv'
            candidate = {'paper_id': 'one', 'title': 'A method', 'full_text': TEXT,
                'expected_decision': 'GOLD_MARKER_MUST_NOT_REACH_MODEL', 'sutva': 'not_required'}
            source.write_text(json.dumps([candidate]))
            client = FakeModel()
            rows = classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, client=client)
            self.assertEqual(rows[0]['decision'], 'accept')
            self.assertEqual(rows[0]['sutva'], 'required')
            self.assertEqual(len(client.calls), 1)
            self.assertNotIn('GOLD_MARKER', client.calls[0][0])
            self.assertEqual(len(client.calls[0][1]['properties']['criteria']['required']), 14)
            with patch('ids_meta.common.study_pipeline.OllamaClient', side_effect=AssertionError('backend initialized')):
                resumed = classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, resume=True)
            self.assertEqual(resumed[0]['decision'], 'accept')
            candidate['full_text'] += ' New evidence.'
            source.write_text(json.dumps([candidate]))
            classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, resume=True, client=client)
            self.assertEqual(len(client.calls), 2)

    def test_fabricated_model_quotes_are_errors_not_accepted_decisions(self):
        """Verify fabricated model quotes are errors not accepted decisions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'title': 'A method', 'full_text': TEXT}]))
            for section, key in [('criteria', 'F1'), ('fields', 'sutva')]:
                invalid = answer()
                invalid[section][key]['quote'] = 'invented evidence'
                rows = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'}, client=FakeModel(invalid))
                self.assertEqual(rows[0]['decision'], 'error')
                self.assertEqual(rows[0]['sutva'], 'unknown')
                self.assertTrue(rows[0]['error'])
                self.assertIn('invented evidence', rows[0]['model_response_json'])
                self.assertIn('quot', rows[0]['assessment_error_detail'])
                self.assertEqual(len(json.loads(rows[0]['assessment_attempts_json'])), 2)

    def test_one_correction_can_recover_valid_evidence_without_reference_labels(self):
        """Verify one correction can recover valid evidence without reference labels."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT,
                'expected_decision': 'GOLD_MUST_STAY_PRIVATE'}]))
            invalid = answer()
            invalid['criteria']['F1']['quote'] = 'invented evidence'
            client = FakeModel()
            client.generate = Mock(side_effect=[invalid, answer()])
            rows = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'}, client=client)
            self.assertEqual(rows[0]['decision'], 'accept')
            self.assertEqual(client.generate.call_count, 2)
            correction_prompt = client.generate.call_args_list[1].args[0]
            self.assertIn('criteria.F1', correction_prompt)
            self.assertNotIn('GOLD_MUST_STAY_PRIVATE', correction_prompt)
            attempts = json.loads(rows[0]['assessment_attempts_json'])
            self.assertTrue(attempts[0]['validation_error'])
            self.assertFalse(attempts[1]['validation_error'])

    def test_malformed_unknown_field_evidence_is_an_error(self):
        """Verify malformed unknown field evidence is an error."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}]))
            invalid = answer()
            invalid['fields']['sutva'] = {'value': 'unknown', 'quote': 123, 'source': ''}
            rows = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'}, client=FakeModel(invalid))
            self.assertEqual(rows[0]['decision'], 'error')

    def test_missing_text_and_dry_run_make_no_http_or_model_calls(self):
        """Verify missing text and dry run make no http or model calls."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'title': 'Missing text', 'pdf_url': 'https://example.org/paper.pdf'}]))
            with patch('ids_meta.common.study_pipeline.OllamaClient', side_effect=AssertionError('model call')), patch(
                'ids_meta.common.study_pipeline.download_study_text', side_effect=AssertionError('HTTP call')):
                missing = classify_study(PROFILE / 'screening.toml', source, root / 'missing.csv', {})
                preview = classify_study(PROFILE / 'screening.toml', source, root / 'preview.csv', {}, dry_run=True, fetch_full_text=True)
            self.assertEqual(missing[0]['decision'], 'uncertain')
            self.assertEqual(missing[0]['assessment_status'], 'missing_full_text')
            self.assertEqual(preview[0]['assessment_status'], 'dry_run')

    def test_bad_identities_fail_before_models_or_output_writes(self):
        """Verify bad identities fail before models or output writes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / 'candidates.json', root / 'out.csv'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}] * 2))
            client = FakeModel()
            with self.assertRaises(ValueError):
                classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, client=client)
            self.assertEqual(client.calls, [])
            self.assertFalse(output.exists())
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}, {'paper_id': 'two', 'full_text': None}]))
            with self.assertRaises(ValueError):
                classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, client=client)
            self.assertEqual(client.calls, [])
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
