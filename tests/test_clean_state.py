"""Verify clean-state contracts without old-format repairs or model calls."""
from pathlib import Path
import json
import tempfile
import unittest

from radar.cli import _configured_models, _execution_paths, build_parser
from radar.common.cache import PipelineCache
from radar.common.config import ConfigError, Settings
from radar.common.csvio import atomic_write_csv
from radar.common.evaluation import evaluate_results, load_rating_profile
from radar.github.pipeline import prefilter as github_prefilter
from radar.openalex.pipeline import prefilter as openalex_prefilter
from radar.plotting.compare import build_classifiers


class CleanStateTests(unittest.TestCase):
    """Require explicit current configuration, collections and report fingerprints."""

    def test_old_limit_option_and_scalar_model_configuration_are_rejected(self):
        """Keep one model configuration form and two independent stage budgets."""
        with self.assertRaises(SystemExit):
            build_parser().parse_args(['github', 'run', '--limit', '5'])
        with self.assertRaises(ConfigError):
            _configured_models(Settings(Path('.'), {'ollama': {'model': 'old'}}))
        self.assertEqual(_configured_models(Settings(Path('.'), {'ollama': {'models': ['current']}})), ['current'])

    def test_one_model_has_the_same_directory_layout_as_several(self):
        """A clean single-model run uses its explicit model directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = Settings(root, {'paths': {name: str(root / name) for name in
                ('interim_dir', 'processed_dir', 'plots_dir', 'evaluation_dir', 'github_cache')}})
            path = _execution_paths(settings, 'github', ['only'])['only']['classifications']
            self.assertEqual(path, root / 'interim_dir/github/models/only/classifications.csv')

    def test_missing_collection_fails_and_empty_collection_stays_empty(self):
        """An explicit empty selection cannot expand to unrelated cached records."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, prefilter, records, ids in (
                ('github', github_prefilter, 'github.repository', 'repository_ids'),
                ('openalex', openalex_prefilter, 'openalex.work', 'work_ids'),
            ):
                with self.subTest(source=source), PipelineCache(root / (source+'.sqlite3')) as cache:
                    cache.put_record(records, 'unselected', {}, '2026-01-01T00:00:00+00:00')
                    output = root / (source+'.csv')
                    with self.assertRaisesRegex(ValueError, 'completed collection'):
                        prefilter({}, {}, cache, output, 'fixture')
                    self.assertFalse(output.exists())
                    cache.put_record(source+'.collection', 'latest', {ids: [], 'selected_at': '2026-01-01T00:00:00+00:00'}, 'now')
                    self.assertEqual(prefilter({}, {}, cache, output, 'fixture'), [])

    def test_two_unpinned_reports_are_rejected_even_at_the_same_reference_path(self):
        """Missing fingerprints cannot pass comparison merely because paths agree."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluations = {}
            for name in ('a', 'b'):
                path = root / name
                path.mkdir()
                (path/'metrics.json').write_text(json.dumps({'ratings_path': '/same/reference', 'evaluation_unit': 'source_record'}))
                evaluations[name] = path
            with self.assertRaisesRegex(ValueError, 'requires reference_fingerprint'):
                build_classifiers('github', {}, evaluations, root/'comparison', {'formats': ['svg']})

    def test_empty_current_format_classifications_can_be_evaluated(self):
        """An empty clean collection records no attempts and no taxonomy capability."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ratings = root/'ratings.toml'
            ratings.write_text('schema_version="1.0"\nsource="github"\nclassification_fields=["role"]\n[[entities]]\nid="one"\ndecision="accept"\nrole="native_detector"\n')
            classifications, candidates = root/'classifications.csv', root/'candidates.csv'
            atomic_write_csv(classifications, [], ['github_repo_id','decision','classifier_supported_fields'])
            atomic_write_csv(candidates, [], ['github_repo_id'])
            result = evaluate_results(load_rating_profile(ratings,'github'), classifications, root/'report', candidates)
            self.assertEqual(result.metrics['operational']['fn'], 1)
            self.assertEqual(result.metrics['classification_coverage'], 0)
            self.assertIsNone(result.metrics['classification_field_accuracy'])
