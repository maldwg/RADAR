"""Exercise portable reporting, immutable provenance and independent verification."""
from __future__ import annotations

import json
import io
import shutil
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stdout
from pathlib import Path

from radar.common.csvio import read_csv
from radar.cli import main
from radar.common.cache import PipelineCache
from radar.common.evaluation import evaluate_results, load_rating_profile
from radar.common.provenance import freeze_database, freeze_file
from radar.common.reporting import ReportingDatabase
from radar.common.sources import LocalSourceAdapter, entity_identifier, load_source_contract
from radar.common.verification import verify_database, verify_directory, verify_manifest
from radar.plotting.compare import build_classifiers
from radar.plotting.generic import build_report

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / 'examples/reporting'
PLOTS = {'formats': ['svg'], 'dpi': 72, 'show_subcaptions': False}


class ReportingTests(unittest.TestCase):
    """Verify counts, identity boundaries, unavailable stages and corruption detection."""

    def test_repository_and_paper_end_to_end(self) -> None:
        """Evaluate both entity types without GitHub/OpenAlex-specific branches."""
        with tempfile.TemporaryDirectory() as temporary:
            for profile, records, ratings in [
                ('repositories.toml', 'repositories.csv', 'repository_references.toml'),
                ('papers.toml', 'papers.json', 'paper_references.toml'),
            ]:
                output = Path(temporary) / profile
                build_report(EXAMPLES / profile, EXAMPLES / records, output, PLOTS,
                    candidates_path=EXAMPLES / records, ratings_path=EXAMPLES / ratings)
                result = verify_directory(output)
                self.assertTrue(result['ok'], result)
                self.assertTrue(all(f['verification'] == 'recomputed' for f in result['figures']))
                result = verify_database(output / 'reporting.sqlite3')
                self.assertTrue(result['ok'], result)
                metrics = json.loads((output / 'evaluation/metrics.json').read_text())
                self.assertEqual(metrics['operational']['tp'], 1)
                self.assertEqual(metrics['operational']['fn'], 1)
                self.assertEqual(metrics['actual_entities'], 3)
                stages = {r['stage']: r for r in read_csv(output / 'figures/data/02_pipeline_stages.csv')}
                self.assertEqual(stages['classification']['count'], '3.0')
                self.assertEqual(stages['export']['basis'], 'not_observed')
                if profile == 'repositories.toml':
                    self.assertEqual(metrics['operational']['tn'], 2)
                    self.assertEqual(metrics['strict']['fp'], 2)
                    self.assertEqual(metrics['taxonomy_field_metrics']['role']['exact_accuracy'], .25)
                    availability = read_csv(output / 'figures/data/plot_availability.csv')
                    self.assertTrue(all(r['available'] == 'False' for r in availability))

    def test_identity_rejects_duplicates_and_distinguishes_provider_and_kind(self) -> None:
        """Prevent silent title-based merging and ambiguous native identifiers."""
        contract = load_source_contract(EXAMPLES / 'repositories.toml')
        adapter = LocalSourceAdapter(contract)
        for rows in [[{'repo_id': 'same'}, {'repo_id': 'same'}], [{'repository': 'missing'}], [{'repo_id': None}]]:
            with self.assertRaises(ValueError):
                adapter.normalize(rows)
        keys = {entity_identifier(contract, 'same'), entity_identifier(replace(contract, name='another'), 'same'),
            entity_identifier(replace(contract, entity_kind='paper'), 'same')}
        self.assertEqual(len(keys), 3)
        self.assertEqual(adapter.normalize([{'repo_id': 'x', 'decision': 'included', 'classifier_supported_fields': '*'}])[0]['decision'], 'accept')

    def test_metadata_only_inputs_have_no_classification_count(self) -> None:
        """Do not pretend that imported unclassified metadata constitutes decisions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / 'records.json'
            records.write_text(json.dumps([{'doi': 'metadata:one', 'title': 'Metadata'}]))
            build_report(EXAMPLES / 'papers.toml', records, root / 'report', PLOTS)
            stages = {r['stage']: r for r in read_csv(root / 'report/figures/data/02_pipeline_stages.csv')}
            self.assertEqual(stages['classification']['basis'], 'not_observed')
            self.assertEqual(stages['inclusion']['basis'], 'not_observed')
            self.assertTrue(verify_directory(root / 'report')['ok'])

    def test_frozen_bundle_survives_input_changes_and_detects_wrong_counts(self) -> None:
        """Replay frozen inputs and catch a fabricated table even with updated hashes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / 'records.csv'
            shutil.copy(EXAMPLES / 'repositories.csv', records)
            output = root / 'report'
            build_report(EXAMPLES / 'repositories.toml', records, output, PLOTS)
            records.write_text('changed live input\n')
            self.assertTrue(verify_directory(output)['ok'])
            manifest_path = output / 'figures/01_decisions.manifest.json'
            manifest = json.loads(manifest_path.read_text())
            table = Path(manifest['tables'][0]['path'])
            table.write_text(table.read_text().replace('accept,1', 'accept,9'))
            manifest['tables'][0].update(freeze_file(table, output))
            manifest_path.write_text(json.dumps(manifest))
            result = verify_manifest(manifest_path)
            self.assertFalse(result['ok'])
            self.assertTrue(any('recomputed counts' in e for e in result['errors']), result)

    def test_metrics_and_database_tampering_are_detected(self) -> None:
        """Catch incorrect confusion counts and source payloads with intact SQLite structure."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'report'
            build_report(EXAMPLES / 'repositories.toml', EXAMPLES / 'repositories.csv', output, PLOTS,
                ratings_path=EXAMPLES / 'repository_references.toml')
            path = output / 'evaluation/metrics.json'
            metrics = json.loads(path.read_text())
            metrics['operational']['fn'] = 0
            path.write_text(json.dumps(metrics))
            self.assertFalse(verify_directory(output)['ok'])
            with sqlite3.connect(output / 'reporting.sqlite3') as db:
                db.execute("UPDATE source_records SET payload_json='{}'")
            result = verify_database(output / 'reporting.sqlite3')
            self.assertFalse(result['ok'])
            self.assertTrue(any('source record differs' in e for e in result['errors']))

    def test_reference_content_and_candidate_cohort_control_comparability(self) -> None:
        """Reject same-path reference changes while allowing identical profile copies."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = load_source_contract(EXAMPLES / 'repositories.toml')
            build_report(EXAMPLES / 'repositories.toml', EXAMPLES / 'repositories.csv', root / 'initial', PLOTS)
            classifications = root / 'initial/normalized/classifications.csv'
            ratings = root / 'ratings.toml'
            shutil.copy(EXAMPLES / 'repository_references.toml', ratings)
            first = root / 'first'
            evaluate_results(load_rating_profile(ratings, contract.name, contract=contract), classifications, first, classifications)
            copy = root / 'copy.toml'
            shutil.copy(ratings, copy)
            second = root / 'second'
            evaluate_results(load_rating_profile(copy, contract.name, contract=contract), classifications, second, classifications)
            a, b = (json.loads((p / 'metrics.json').read_text()) for p in (first, second))
            self.assertEqual(a['reference_fingerprint'], b['reference_fingerprint'])
            build_classifiers(contract.name, {'A': classifications, 'B': classifications}, {'A': first, 'B': second}, root / 'comparison', PLOTS)
            ratings.write_text(ratings.read_text().replace('role = "dataset"', 'role = "tool"', 1))
            third = root / 'third'
            evaluate_results(load_rating_profile(ratings, contract.name, contract=contract), classifications, third, classifications)
            with self.assertRaisesRegex(ValueError, 'reference_fingerprint'):
                build_classifiers(contract.name, {}, {'A': first, 'B': third}, root / 'rejected', PLOTS)

    def test_import_requires_fingerprints_and_separate_cache_schema(self) -> None:
        """Reject unpinned reports and refuse to reuse inference cache schemas."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = load_source_contract(EXAMPLES / 'repositories.toml')
            build_report(EXAMPLES / 'repositories.toml', EXAMPLES / 'repositories.csv', root / 'report', PLOTS,
                ratings_path=EXAMPLES / 'repository_references.toml')
            evaluation = root / 'report/evaluation'
            path = evaluation / 'metrics.json'
            metrics = json.loads(path.read_text())
            metrics.pop('input_fingerprints')
            path.write_text(json.dumps(metrics))
            with ReportingDatabase(root / 'unpinned.sqlite3') as db:
                with self.assertRaisesRegex(ValueError, 'fingerprints'):
                    db.ingest_evaluation(evaluation, contract)
            with PipelineCache(root / 'cache.sqlite3') as cache:
                cache.put_record('fixture', 'one', {'value': 1}, '2026-10-12T00:00:00Z')
                frozen = freeze_database(root / 'cache.sqlite3', root / 'report')
                cache.put_record('fixture', 'one', {'value': 2}, '2026-10-12T00:00:01Z')
                with PipelineCache(frozen, read_only=True) as snapshot:
                    self.assertEqual(snapshot.get_record('fixture', 'one'), {'value': 1})
                self.assertEqual(cache.get_record('fixture', 'one'), {'value': 2})

    def test_relocated_bundle_and_database_remain_verifiable(self) -> None:
        """Resolve archived objects relative to a moved bundle, independent of native paths."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_report(EXAMPLES / 'repositories.toml', EXAMPLES / 'repositories.csv', root / 'old', PLOTS,
                ratings_path=EXAMPLES / 'repository_references.toml')
            shutil.move(root / 'old', root / 'moved')
            result = verify_directory(root / 'moved')
            self.assertTrue(result['ok'], result)
            result = verify_database(root / 'moved/reporting.sqlite3')
            self.assertTrue(result['ok'], result)

    def test_malformed_audits_fail_and_exports_require_acceptance(self) -> None:
        """Reject invalid JSON, incomplete databases and exports outside accepted decisions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'invalid.manifest.json'
            path.write_text('{')
            self.assertFalse(verify_manifest(path)['ok'])
            self.assertFalse(verify_database(root / 'missing.sqlite3')['ok'])
            with self.assertRaisesRegex(ValueError, 'export contains'):
                build_report(EXAMPLES / 'repositories.toml', EXAMPLES / 'repositories.csv', root / 'rejected', PLOTS,
                    exports_path=EXAMPLES / 'repositories.csv')

    def test_verification_cli_returns_parseable_json_and_failure_code(self) -> None:
        """Expose a machine-readable failed audit without extra stdout logging."""
        with tempfile.TemporaryDirectory() as temporary:
            stream = io.StringIO()
            with redirect_stdout(stream):
                result = main(['verify', '--database', str(Path(temporary) / 'absent.sqlite3')])
            self.assertEqual(result, 2)
            self.assertFalse(json.loads(stream.getvalue())['ok'])


if __name__ == '__main__':
    unittest.main()
