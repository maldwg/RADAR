"""Verify shared discovery limits and evidence-backed review voting offline."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from radar.cli import main, _execution_paths, _variant_paths
from radar.common.verification import verify_directory
from radar.common.config import ConfigError, load_settings
from radar.common.csvio import read_csv
from radar.common.study_consensus import ensemble_study
from radar.common.study_pipeline import collect_study, classify_study
from tests.test_study_pipeline import answer, TEXT

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'config/studies/conformal_treatment_effects'


class PagedSearch:
    """Serve separate query streams with real cursor pagination and exhaustion."""

    select = 'id,title'

    def __init__(self, count=40):
        """Set the result count for each query and record request budgets."""
        self.count = count
        self.calls = []

    def get(self, path, params):
        """Honor page size and return the next cursor only while results remain."""
        self.calls.append(copy.deepcopy(params))
        offset = 0 if params['cursor'] == '*' else int(params['cursor'])
        stop = min(self.count, offset + params['per-page'])
        prefix = 'A' if 'individual OR patient' in params['search.title_and_abstract.exact'] else 'B'
        return {'meta': {'count': self.count, 'next_cursor': str(stop) if stop < self.count else None},
            'results': [{'id': f'https://openalex.org/{prefix}{index}', 'title': f'Paper {prefix}{index}'} for index in range(offset, stop)]}


class SequenceModel:
    """Supply controlled per-run answers or failures without inference."""

    def __init__(self, responses):
        """Copy a reusable response sequence and reset its call counter."""
        self.responses = responses
        self.calls = 0

    def generate(self, prompt, schema):
        """Cycle through responses so a fresh resumed configuration can be tested."""
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)


class ConformalAlignmentTests(unittest.TestCase):
    """Exercise query caps, configured voting, compatible ensembles and validation."""

    def _config(self, root):
        """Copy profiles into an isolated project root with fresh outputs."""
        shutil.copytree(ROOT / 'config', root / 'config')
        return root / 'config/studies/conformal_treatment_effects/study.toml'

    def _candidates(self, root, count=1):
        """Write a frozen cohort with full text independent of reference labels."""
        path = root / 'candidates.json'
        path.write_text(json.dumps([{'paper_id': str(i), 'title': 'Fixture', 'full_text': TEXT} for i in range(count)]))
        return path

    def test_per_query_caps_follow_shared_factor_and_larger_numeric_limit(self):
        """Retrieve a broader pool while retaining only the configured cohort size."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for classification, cap in [('all', 6), (5, 15)]:
                client = PagedSearch()
                output = root / f'{classification}.json'
                rows = collect_study(PROFILE / 'search/openalex.toml', output, limit=2, client=client,
                    source_config={'query_limit_factor': 3.0, 'classification_limit': classification, 'per_page': 4})
                audit = json.loads(output.with_suffix('.collection.json').read_text())
                self.assertEqual(len(rows), 2)
                self.assertEqual(audit['per_query_limit'], cap)
                self.assertTrue(all(query['scanned'] == cap and query['coverage'] == 'bounded' for query in audit['queries']))

    def test_unlimited_collection_exhausts_queries_and_preserves_finite_classification_cap(self):
        """All removes application caps only when both stage limits are unbounded."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'all.json'
            rows = collect_study(PROFILE / 'search/openalex.toml', output, limit='all', client=PagedSearch(7),
                source_config={'classification_limit': 'all', 'per_page': 3})
            audit = json.loads(output.with_suffix('.collection.json').read_text())
            self.assertEqual(len(rows), 14)
            self.assertIsNone(audit['per_query_limit'])
            self.assertTrue(all(query['coverage'] == 'complete' for query in audit['queries']))
            rows = collect_study(PROFILE / 'search/openalex.toml', output, limit=None, client=PagedSearch(7),
                source_config={'classification_limit': 1})
            self.assertEqual(len(rows), 6)
            self.assertEqual(json.loads(output.with_suffix('.collection.json').read_text())['per_query_limit'], 3)

    def test_cli_limits_apply_to_collection_and_classification_together(self):
        """A classification override changes retrieval capacity before any inference."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._config(root)
            output = root / 'collected.json'
            with patch('radar.adapters.openalex.OpenAlexClient', return_value=PagedSearch()), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(manifest), 'collect', '--collect-limit', '4', '--classify-limit', '6',
                    '--output', str(output)]), 0)
            audit = json.loads(output.with_suffix('.collection.json').read_text())
            self.assertEqual(audit['collection_limit'], 4)
            self.assertEqual(audit['classification_limit'], 6)
            self.assertEqual(audit['per_query_limit'], 18)

    def test_repeated_assessments_vote_retain_evidence_and_resume_without_inference(self):
        """Four of five judgments pass at 0.8 while a higher cutoff abstains."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._candidates(root)
            rejected = answer()
            rejected['criteria']['F1']['value'] = 'no'
            model = SequenceModel([answer()] * 4 + [rejected])
            output = root / 'classifications.csv'
            rows = classify_study(PROFILE / 'screening.toml', source, output, {'model': 'one'}, client=model,
                consecutive_runs=5, consensus_threshold=.8)
            self.assertEqual(model.calls, 5)
            self.assertEqual(rows[0]['decision'], 'accept')
            self.assertEqual(json.loads(rows[0]['evidence_passages_json'])['passages'][0]['text'], TEXT)
            self.assertTrue(json.loads(rows[0]['criteria_json'])['F1']['evidence_ids'])
            checkpoint = json.loads(output.with_suffix('.assessments.json').read_text())
            self.assertTrue(checkpoint['completed'])
            self.assertEqual(len(json.loads(checkpoint['records'][0]['assessment_votes_json'])), 5)
            with patch('radar.common.study_pipeline.OllamaClient', side_effect=AssertionError('inference')):
                self.assertEqual(classify_study(PROFILE / 'screening.toml', source, output, {'model': 'one'},
                    consecutive_runs=5, consensus_threshold=.8, resume=True)[0]['decision'], 'accept')
            rows = classify_study(PROFILE / 'screening.toml', source, output, {'model': 'one'}, client=model,
                consecutive_runs=5, consensus_threshold=1, resume=True)
            self.assertEqual(model.calls, 10)
            self.assertEqual(rows[0]['decision'], 'uncertain')

    def test_voted_criteria_cannot_invent_acceptance_without_disposition_support(self):
        """Reject a coherent-looking majority tuple when every run excluded the paper."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            votes = []
            for criterion in range(1, 6):
                vote = answer()
                vote['criteria'][f'F{criterion}']['value'] = 'no'
                votes.append(vote)
            rows = classify_study(PROFILE / 'screening.toml', self._candidates(root), root / 'out.csv',
                {'model': 'one'}, client=SequenceModel(votes), consecutive_runs=5)
            self.assertEqual(rows[0]['decision'], 'uncertain')
            self.assertEqual(rows[0]['consensus_validation_flags'], 'decision_criteria_conflict')

    def test_failed_assessments_count_in_the_voting_denominator(self):
        """One successful response cannot satisfy a five-voter 0.8 threshold."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = classify_study(PROFILE / 'screening.toml', self._candidates(root), root / 'out.csv',
                {'model': 'one'}, client=SequenceModel([answer()] + [RuntimeError('offline')] * 4), consecutive_runs=5)
            self.assertEqual(rows[0]['decision'], 'uncertain')
            self.assertEqual(rows[0]['assessment_status'], 'partial_error')

    def test_ensemble_votes_fields_and_abstains_on_tied_decisions(self):
        """Compare actual judgments without transplanting IDS paper-role validators."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._candidates(root)
            other = answer()
            other['fields']['sutva']['value'] = 'not_required'
            inputs = {'first': root / 'first.csv', 'second': root / 'second.csv'}
            for model, response in [('first', answer()), ('second', other)]:
                classify_study(PROFILE / 'screening.toml', source, inputs[model], {'model': model}, client=SequenceModel([response]))
            rows = ensemble_study(PROFILE / 'screening.toml', inputs, root / 'ensemble.csv', 1)
            self.assertEqual(rows[0]['decision'], 'accept')
            self.assertEqual(rows[0]['sutva'], 'unknown')
            other['criteria']['F1']['value'] = 'no'
            classify_study(PROFILE / 'screening.toml', source, inputs['second'], {'model': 'second'}, client=SequenceModel([other]))
            self.assertEqual(ensemble_study(PROFILE / 'screening.toml', inputs, root / 'ensemble.csv', .5)[0]['decision'], 'uncertain')

    def test_ensemble_rejects_different_cohorts_and_incomplete_checkpoints(self):
        """A fresh ensemble cannot accidentally combine partial or different inputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._candidates(root, 2)
            inputs = {'first': root / 'first.csv', 'second': root / 'second.csv'}
            for model, limit in [('first', None), ('second', 1)]:
                classify_study(PROFILE / 'screening.toml', source, inputs[model], {'model': model}, limit=limit, client=SequenceModel([answer()]))
            with self.assertRaisesRegex(ValueError, 'incompatible cohorts'):
                ensemble_study(PROFILE / 'screening.toml', inputs, root / 'ensemble.csv', 1)
            checkpoint = inputs['first'].with_suffix('.assessments.json')
            data = json.loads(checkpoint.read_text())
            data['completed'] = False
            checkpoint.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'completed'):
                ensemble_study(PROFILE / 'screening.toml', inputs, root / 'ensemble.csv', 1)

    def test_cli_classification_runs_repetitions_and_writes_ensemble(self):
        """Study controls reach the actual model calls and isolated ensemble output."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._config(root)
            model = SequenceModel([answer()])
            with patch('radar.common.study_pipeline.OllamaClient', return_value=model), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(manifest), 'classify', '--input', str(self._candidates(root))]), 0)
            settings = load_settings(manifest)
            self.assertEqual(model.calls, 5 * len(settings.section('ollama')['models']))
            ensemble = _variant_paths(settings, 'cohort', Path('ensemble'))['classifications']
            self.assertEqual(read_csv(ensemble)[0]['decision'], 'accept')

    def test_unsupported_and_obsolete_controls_are_rejected(self):
        """Repository-only controls and obsolete fixed query caps are rejected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._config(root)
            path = manifest.parent / 'pipeline.toml'
            original = path.read_text()
            for setting in ('minimum_stars = 100', 'per_query_limit = 300'):
                path.write_text(original.replace('[collection]', '[collection]\n' + setting))
                with self.assertRaises(ConfigError):
                    load_settings(manifest)

    def test_empty_discovery_classification_and_ensemble_remain_empty(self):
        """An exhausted empty search creates explicit completed, rowless outputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'candidates.json'
            self.assertEqual(collect_study(PROFILE / 'search/openalex.toml', source, client=PagedSearch(0)), [])
            inputs = {'first': root / 'first.csv', 'second': root / 'second.csv'}
            for model, output in inputs.items():
                self.assertEqual(classify_study(PROFILE / 'screening.toml', source, output, {'model': model}), [])
            self.assertEqual(ensemble_study(PROFILE / 'screening.toml', inputs, root / 'ensemble.csv', 1), [])

    def test_empty_full_run_reports_reference_misses_without_inference(self):
        """Unlimited but empty discovery completes all configured report variants."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._config(root)
            reporting = manifest.parent / 'reporting.toml'
            reporting.write_text(reporting.read_text().replace('["svg", "pdf"]', '["svg"]'))
            with patch('radar.adapters.openalex.OpenAlexClient', return_value=PagedSearch(0)), patch(
                'radar.common.study_pipeline.OllamaClient', side_effect=AssertionError('inference')), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(manifest), 'run', '--collect-all', '--classify-all']), 0)
            settings = load_settings(manifest)
            reports = [paths['plots'] for paths in _execution_paths(settings, 'cohort', settings.section('ollama')['models']).values()]
            reports.append(_variant_paths(settings, 'cohort', Path('ensemble'))['plots'])
            for report in reports:
                metrics = json.loads((report / 'evaluation/metrics.json').read_text())
                self.assertEqual(metrics['operational']['fn'], 11)
                self.assertEqual(metrics['classification_coverage'], 0)

    def test_full_run_reports_and_verifies_models_and_ensemble(self):
        """A synthetic positive exercises classification, evaluation and figure audits."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._config(root)
            reporting = manifest.parent / 'reporting.toml'
            reporting.write_text(reporting.read_text().replace('["svg", "pdf"]', '["svg"]'))
            source = self._candidates(root)
            records = json.loads(source.read_text())
            with (PROFILE / 'evaluation/references.toml').open('rb') as handle:
                records[0]['title'] = tomllib.load(handle)['entities'][0]['title']
            source.write_text(json.dumps(records))
            with patch('radar.common.study_pipeline.OllamaClient', return_value=SequenceModel([answer()])), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(manifest), 'run', '--input', str(source)]), 0)
            settings = load_settings(manifest)
            reports = [paths['plots'] for paths in _execution_paths(settings, 'cohort', settings.section('ollama')['models']).values()]
            reports.append(_variant_paths(settings, 'cohort', Path('ensemble'))['plots'])
            for report in reports:
                metrics = json.loads((report / 'evaluation/metrics.json').read_text())
                self.assertEqual(metrics['operational']['tp'], 1)
                self.assertTrue(verify_directory(report)['ok'])
