"""Check shared paper adapters and operational schemas across two scientific studies."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

from radar.adapters.openalex import OpenAlexAdapter
from radar.cli import main, _execution_paths
from radar.common.cache import PipelineCache
from radar.common.config import ConfigError, load_settings, load_term_profile
from radar.common.discovery import DiscoverySource, collect_sources
from radar.common.paper_collection import collect_papers, eligible_papers, prefilter_paper
from radar.common.paper_evidence import PaperEvidenceService
from radar.common.study_pipeline import classify_study
from tests.test_study_pipeline import answer, TEXT

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / 'config/studies/ids_datasets'
CONFORMAL = ROOT / 'config/studies/conformal_treatment_effects'


class PaperApi:
    """Serve direct papers and survey references through real adapter methods."""
    select = 'id,title,type,referenced_works'

    def __init__(self):
        """Record observed native query parameters."""
        self.calls = []

    def get(self, path, params):
        """Return complete native pages for a survey and a direct study."""
        self.calls.append(params)
        text = params.get('search', params.get('search.title_and_abstract.exact', ''))
        items = ([{'id': 'https://openalex.org/S', 'title': 'Survey of conformal treatment effects',
            'type': 'review', 'referenced_works': ['https://openalex.org/R']}] if text == 'survey' else
            [{'id': 'https://openalex.org/W', 'title': 'Original conformal treatment effect dataset method'}])
        return {'results': items, 'meta': {'count': len(items), 'next_cursor': None}}

    def works_by_ids(self, identifiers):
        """Resolve referenced original studies without contacting any API."""
        for identifier in identifiers:
            yield {'id': identifier, 'title': 'Referenced conformal treatment effect study'}


class TextEnricher:
    """Supply complete synthetic full text and count evidence retrievals."""
    def __init__(self):
        """Initialize the request counter."""
        self.calls = 0

    def enrich(self, work):
        """Return literal evidence appropriate for the fixture criterion assessments."""
        self.calls += 1
        return {'full_text': TEXT, 'full_text_complete': True, 'abstract': 'Conformal treatment effect method'}


class SharedPaperStudyTests(unittest.TestCase):
    """Prevent provider and configuration drift while preserving scientific differences."""
    def test_paper_pipeline_and_search_schemas_match(self):
        """Both studies explicitly declare identical operational sections, keys and search structure."""
        dataset, conformal = [tomllib.loads((root / 'pipeline.toml').read_text()) for root in (DATASETS, CONFORMAL)]
        self.assertEqual(list(dataset), list(conformal))
        for stage in ('collection', 'evidence', 'classification', 'reconciliation'):
            self.assertEqual(list(dataset[stage]), list(conformal[stage]))
        self.assertEqual(dataset['classification']['ensemble'].keys(), conformal['classification']['ensemble'].keys())
        searches = [tomllib.loads((root / 'search/openalex.toml').read_text()) for root in (DATASETS, CONFORMAL)]
        self.assertEqual(set(searches[0]), set(searches[1]))
        self.assertEqual(set(searches[0]['queries'][0]), set(searches[1]['queries'][0]))
        self.assertEqual(dataset['collection']['collection_mode'], 'ranked')
        self.assertEqual(conformal['collection']['collection_mode'], 'query_position')
        settings = [load_settings(root / 'study.toml') for root in (DATASETS, CONFORMAL)]
        self.assertEqual(settings[0].section('study')['sources'], settings[1].section('study')['sources'])
        self.assertEqual(settings[0].section('openalex')['canonical_role_scores']['original_release'], 60)
        self.assertNotIn('canonical_role_scores', dataset['reconciliation'])

    def test_both_workflows_call_the_same_collector_with_the_same_adapter(self):
        """Provider changes are shared because both routes execute one scholarly collection function."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = load_settings(DATASETS / 'study.toml')
            conformal = load_settings(CONFORMAL / 'study.toml')
            first_config = {**dataset.section('openalex'), 'max_surveys': 0, 'snowball_max_depth': 0,
                'survey_reference_fraction': 0}
            second_config = dict(conformal.section('openalex'))
            first_api, second_api = PaperApi(), PaperApi()
            first = OpenAlexAdapter(first_config, DATASETS / 'search/openalex.toml', first_api)
            second = OpenAlexAdapter(second_config, CONFORMAL / 'search/openalex.toml', second_api)
            spy = Mock(wraps=collect_papers)
            with patch('radar.adapters.openalex.collect_papers', spy), patch('radar.common.paper_collection.collect_papers', spy):
                with PipelineCache(root / 'datasets.sqlite3') as cache:
                    first.collect_catalogue(first_config, load_term_profile(dataset, 'openalex'), cache, 2)
                collect_sources([DiscoverySource('openalex', second.profile_path, second_config, second,
                    root / 'conformal.sqlite3')], root / 'cohort.json', limit=2, entity_kind='paper')
            self.assertEqual(spy.call_count, 2)
            self.assertIs(spy.call_args_list[0].args[-1], first)
            self.assertIs(spy.call_args_list[1].args[-1], second)
            self.assertIn('search', first_api.calls[0])
            self.assertIn('search.title_and_abstract.exact', second_api.calls[0])
            self.assertIn('from_publication_date:2015-01-01', second_api.calls[0]['filter'])

    def test_conformal_survey_and_snowball_settings_execute(self):
        """Enabling the shared knobs retrieves reference papers rather than silently ignoring them."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'search.toml'
            path.write_text((CONFORMAL / 'search/openalex.toml').read_text().replace('survey_queries = []', 'survey_queries = ["survey"]'))
            config = {**load_settings(CONFORMAL / 'study.toml').section('openalex'), 'max_surveys': 1,
                'snowball_max_depth': 1, 'survey_reference_fraction': .5}
            adapter = OpenAlexAdapter(config, path, PaperApi())
            rows = collect_sources([DiscoverySource('openalex', path, config, adapter)], root / 'cohort.json', limit=2)
            self.assertEqual({row['openalex_id'] for row in rows}, {'https://openalex.org/W', 'https://openalex.org/R'})
            audit = json.loads((root / 'cohort.collection.json').read_text())
            self.assertEqual(audit['sources'][0]['counts']['snowball_reached_depth'], 1)
            self.assertEqual(audit['sources'][0]['counts']['survey_sources'], 1)
            self.assertEqual(audit['sources'][0]['selection']['lookup_responses'][0]['native_id'], 'https://openalex.org/R')

    def test_shared_prefilter_and_borderline_gate_have_identical_semantics(self):
        """Study values determine bands while the scoring and shortlist rules are shared."""
        work = {'title': 'An original study', 'full_text': 'A conformal treatment effect method'}
        settings = {'candidate_threshold': 7, 'borderline_threshold': 4, 'survey_consensus_threshold': 2}
        result = prefilter_paper(work, {'positive': {}}, settings, survey_sources=2)
        self.assertEqual(result['prefilter_band'], 'borderline')
        self.assertIn('survey_consensus:2', result['prefilter_signals'])
        rows = [{**result, 'paper_id': 'one'}, {'paper_id': 'two', 'prefilter_band': 'candidate'}]
        self.assertEqual([row['paper_id'] for row in eligible_papers(rows, False)], ['two'])
        self.assertEqual(len(eligible_papers(rows, True)), 2)

    def test_shared_pdf_policy_downloads_once_per_evidence_cohort(self):
        """Repeated model preparation reuses one frozen PDF result and the declared limits."""
        service = PaperEvidenceService({'fetch_full_text': True, 'max_full_text_chars': 100,
            'evidence_lookup_timeout_seconds': 7})
        work = {'title': 'Paper', 'pdf_url': 'https://example.org/paper.pdf'}
        with patch('radar.common.paper_evidence.download_pdf_text', return_value={'full_text': 'Evidence'}) as download:
            self.assertEqual(service.prepare(work), service.prepare(work))
        download.assert_called_once_with(work['pdf_url'], 100, 7)

    def test_uncertain_review_restarts_assessments_with_shared_enrichment(self):
        """Unavailable full text can be recovered by the same evidence service used by datasets."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'title': 'Conformal method', 'full_text': ''}]))
            enricher = TextEnricher()
            service = PaperEvidenceService({'enrich_uncertain_with_full_text': True}, enricher)
            model = Mock()
            model.generate.return_value = answer()
            rows = classify_study(CONFORMAL / 'screening.toml', source, root / 'classified.csv', {'model': 'fixture'},
                client=model, enrich_uncertain_with_full_text=True, evidence_service=service)
            self.assertEqual(rows[0]['decision'], 'accept')
            self.assertEqual(enricher.calls, 1)
            self.assertTrue(rows[0]['evidence_expansion_changed'])
            self.assertEqual(model.generate.call_count, 1)

    def test_review_ensemble_freezes_shared_evidence_before_model_calls(self):
        """Both models assess the same retrieved text when conditional enrichment is configured."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'config', root / 'config')
            manifest = root / 'config/studies/conformal_treatment_effects/study.toml'
            pipeline = manifest.parent / 'pipeline.toml'
            pipeline.write_text(pipeline.read_text().replace('enrich_uncertain_with_full_text = false', 'enrich_uncertain_with_full_text = true'))
            source = root / 'input.json'
            source.write_text(json.dumps([{'paper_id': 'one', 'title': 'Conformal treatment effect study', 'full_text': ''}]))
            enricher = TextEnricher()
            service = PaperEvidenceService(load_settings(manifest).section('pipeline')['evidence'], enricher)
            model = Mock()
            model.generate.return_value = answer()
            with patch('radar.common.paper_evidence.PaperEvidenceService', return_value=service), patch(
                'radar.common.study_pipeline.OllamaClient', return_value=model), redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(manifest), 'classify', '--input', str(source)]), 0)
            self.assertEqual(enricher.calls, 1)
            self.assertEqual(model.generate.call_count, 10)
            settings = load_settings(manifest)
            for paths in _execution_paths(settings, 'cohort', settings.section('ollama')['models']).values():
                record = json.loads(paths['classifications'].with_suffix('.assessments.json').read_text())['records'][0]
                self.assertEqual(record['full_text'], TEXT)

    def test_incomplete_schema_and_unsupported_reference_provider_fail_validation(self):
        """Missing shared controls and unsupported enabled features fail before requests."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(ROOT / 'config', root / 'config')
            manifest = root / 'config/studies/conformal_treatment_effects/study.toml'
            pipeline = manifest.parent / 'pipeline.toml'
            original = pipeline.read_text()
            pipeline.write_text(original.replace('max_surveys = 0\n', ''))
            with self.assertRaisesRegex(ConfigError, 'must declare exactly'):
                load_settings(manifest)
            pipeline.write_text(original.replace('snowball_max_depth = 0', 'snowball_max_depth = 1'))
            manifest.write_text(manifest.read_text().replace('name = "openalex"', 'name = "semantic_scholar"').replace(
                'sources/openalex.toml', 'sources/semantic_scholar.toml').replace('search/openalex.toml', 'search/semantic_scholar.toml'))
            with self.assertRaisesRegex(ConfigError, 'does not support survey/reference traversal'):
                load_settings(manifest)
