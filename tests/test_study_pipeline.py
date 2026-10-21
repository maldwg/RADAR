"""Exercise blind discovery, model evidence validation and resumable study runs."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from radar.common.study_pipeline import classify_study, collect_study
from radar.common.evidence_blocks import EvidenceBlockIndex, EVIDENCE_BLOCK_VERSION
from radar.common.study_consensus import ensemble_study

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/studies/conformal_treatment_effects"
TEXT = "We propose an original conformal treatment-effect method. The method requires SUTVA, ignorability and overlap."


def answer(text=TEXT) -> dict:
    """Construct judgments referencing the actual frozen extraction."""
    block = EvidenceBlockIndex(text).blocks[0].id
    assessment = lambda value: {"value": value, "evidence_ids": [block] if value != "unknown" else []}
    return {"criteria": {**{f"F{i}": assessment("yes") for i in range(1, 6)},
        **{f"Q{i}": assessment("yes" if i <= 5 else "unknown") for i in range(1, 10)}},
        "fields": {field: assessment("required") for field in ("sutva", "strong_ignorability", "overlap")}}


class FakeModel:
    def __init__(self, response=None):
        """Initialize the adapter or fixture state."""
        self.calls = []
        self.response = response

    def generate(self, prompt, schema):
        """Return a recorded model response for the test."""
        self.calls.append((prompt, schema))
        if self.response is None:
            evidence = json.loads(prompt.split('UNTRUSTED EVIDENCE BLOCKS\n', 1)[1])
            result = answer()
            for group in result.values():
                for judgment in group.values():
                    if judgment['evidence_ids']:
                        judgment['evidence_ids'] = [evidence['evidence_blocks'][0]['id']]
            return result
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
            rows = collect_study(PROFILE / "search/openalex.toml", output, limit=2,
                source_config={"query_limit_factor": 1}, client=client)
            self.assertEqual([r['paper_id'] for r in rows], ['openalex:https://openalex.org/W2', 'openalex:https://openalex.org/W1'])
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
            with patch('radar.common.study_pipeline.OllamaClient', side_effect=AssertionError('backend initialized')):
                resumed = classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, resume=True)
            self.assertEqual(resumed[0]['decision'], 'accept')
            candidate['full_text'] += ' New evidence.'
            source.write_text(json.dumps([candidate]))
            classify_study(PROFILE / 'screening.toml', source, output, {'model': 'test'}, resume=True, client=client)
            self.assertEqual(len(client.calls), 2)

    def test_unknown_block_ids_abstain_only_on_the_affected_judgment(self):
        """A bad reference is audited once; other supported judgments remain usable."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}]))
            for section, key in [('criteria', 'F1'), ('fields', 'sutva')]:
                invalid = answer()
                invalid[section][key]['evidence_ids'] = ['B000000000000-00001']
                model = FakeModel(invalid)
                rows = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'}, client=model)
                self.assertEqual(rows[0]['decision'], 'uncertain' if section == 'criteria' else 'accept')
                self.assertEqual(len(model.calls), 1)
                warning = json.loads(rows[0]['assessment_evidence_warnings_json'])[0]
                self.assertEqual(warning['field'], f'{section}.{key}')
                self.assertEqual(warning['judgment']['evidence_ids'], ['B000000000000-00001'])
                self.assertEqual(len(json.loads(rows[0]['assessment_attempts_json'])), 1)

    def test_multiple_passages_are_copied_separately_with_math_and_typos_intact(self):
        """Evidence can span distant blocks without generated quotations or omission stitching."""
        text = TEXT + '\n\nThe calibartion uses α ≤ 0.05.\n\nAnother section: treat-\nment effects and E fficient SCI-\nENCE.'
        index = EvidenceBlockIndex(text)
        response = answer(text)
        ids = [index.blocks[2].id, index.blocks[0].id]
        response['criteria']['F5']['evidence_ids'] = ids
        response['fields']['sutva']['evidence_ids'] = [index.blocks[1].id]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': text}]))
            model = FakeModel(response)
            row = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'}, client=model)[0]
            self.assertEqual(row['decision'], 'accept')
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(json.loads(row['criteria_json'])['F5']['evidence_ids'], ids)
            manifest = json.loads(row['evidence_passages_json'])
            self.assertEqual(len(manifest['passages']), 3)
            for passage in manifest['passages']:
                self.assertEqual(text[passage['start']:passage['end']], passage['text'])
            self.assertIn('calibartion', manifest['passages'][1]['text'])
            self.assertIn('treat-\nment', manifest['passages'][2]['text'])
            schema = model.calls[0][1]['properties']['criteria']['properties']['F5']
            self.assertEqual(schema['required'], ['value', 'evidence_ids'])
            self.assertNotIn('quote', schema['properties'])
            self.assertNotIn('assessment_quote_repairs_json', row)

    def test_ensemble_retains_reference_audits_and_rejects_mixed_segmentation(self):
        """Voting retains IDs and model attribution, and requires the same block policy."""
        supported, unsupported = answer(), answer()
        unsupported['criteria']['Q8'] = {'value': 'yes', 'evidence_ids': ['other-document']}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}]))
            inputs = {'one': root / 'one.csv', 'two': root / 'two.csv'}
            for model, response in [('one', supported), ('two', unsupported)]:
                classify_study(PROFILE / 'screening.toml', source, inputs[model], {'model': model}, client=FakeModel(response))
            result = ensemble_study(PROFILE / 'screening.toml', inputs, root / 'ensemble.csv', 1)[0]
            self.assertEqual(result['decision'], 'accept')
            self.assertEqual({e['model'] for e in json.loads(result['assessment_evidence_warnings_json'])}, {'two'})
            self.assertEqual(json.loads(result['evidence_passages_json'])['passages'][0]['text'], TEXT)
            checkpoint = inputs['two'].with_suffix('.assessments.json')
            data = json.loads(checkpoint.read_text())
            data['evidence_block_version'] = 'different-policy'
            checkpoint.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'evidence block version'):
                ensemble_study(PROFILE / 'screening.toml', inputs, root / 'mixed.csv', 1)

    def test_repeated_assessments_never_add_evidence_correction_calls(self):
        """Invalid IDs cost exactly the configured assessments and remain audited."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT,
                'expected_decision': 'GOLD_MUST_STAY_PRIVATE'}]))
            invalid = answer()
            invalid['criteria']['F1']['evidence_ids'] = ['invented-id']
            client = FakeModel(invalid)
            row = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'},
                client=client, consecutive_runs=5, consensus_threshold=.8)[0]
            self.assertEqual(row['decision'], 'uncertain')
            self.assertEqual(len(client.calls), 5)
            for prompt, _schema in client.calls:
                self.assertNotIn('GOLD_MUST_STAY_PRIVATE', prompt)
                self.assertNotIn('CORRECTION REQUEST', prompt)
            self.assertEqual(len(json.loads(row['assessment_evidence_warnings_json'])), 5)

    def test_malformed_response_is_an_error_without_correction(self):
        """Keep malformed responses as auditable errors and never ask the model to fix their schema."""
        unknown_with_reference = answer()
        unknown_with_reference['fields']['sutva'] = {'value': 'unknown', 'evidence_ids': [EvidenceBlockIndex(TEXT).blocks[0].id]}
        invalid_type = answer()
        invalid_type['fields']['sutva'] = {'value': 'unknown', 'evidence_ids': 123}
        for invalid in (None, {}, {'criteria': [], 'fields': {}}, unknown_with_reference, invalid_type):
            with self.subTest(response=invalid), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / 'candidates.json'
                source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}]))
                model = Mock()
                model.generate.side_effect = [invalid, AssertionError('unexpected schema correction call')]
                rows = classify_study(PROFILE / 'screening.toml', source, root / 'out.csv', {'model': 'test'}, client=model)
                self.assertEqual(model.generate.call_count, 1)
                self.assertEqual(rows[0]['decision'], 'error')
                attempts = json.loads(rows[0]['assessment_attempts_json'])
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]['response'], invalid)
                self.assertTrue(attempts[0]['validation_error'])

    def test_missing_text_and_dry_run_make_no_http_or_model_calls(self):
        """Verify missing text and dry run make no http or model calls."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'candidates.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'title': 'Missing text', 'pdf_url': 'https://example.org/paper.pdf'}]))
            with patch('radar.common.study_pipeline.OllamaClient', side_effect=AssertionError('model call')), patch(
                'radar.common.paper_evidence.download_pdf_text', side_effect=AssertionError('HTTP call')):
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
