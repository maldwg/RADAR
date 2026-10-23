"""Exercise TOML-owned quality instructions, limits, scoring and cache provenance."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from radar.common.config import load_settings, load_prompt_profile, load_system_type_profile
from radar.common.csvio import atomic_write_csv
from radar.common.evidence_blocks import EvidenceBlockIndex
from radar.common.nli import load_nli_runtime_config
from radar.common.ollama import EvidenceSection
from radar.common.plain_llm import classify_plain_llm, plain_llm_task
from radar.common.provenance import file_hash
from radar.common.quality import assess_llm_quality, assess_nli_quality, quality_schema
from radar.common.quality_config import load_quality_config
from radar.common.study_consensus import ensemble_study
from radar.common.study_pipeline import classify_study, _assessment_prompt, load_assessment_fields, load_assessment_template, _assessment_errors
from radar.common.study_screening import load_study_protocol
from radar.github.pipeline import prompt as repository_prompt
from radar.openalex.pipeline import prompt as paper_prompt
from tests.test_plain_llm import completion, schema_answer
from tests.test_quality import QualityNLIBackend
from tests.test_study_pipeline import TEXT, FakeModel
from radar.common.nli import NLIScore

ROOT = Path(__file__).resolve().parents[1]


class QualityConfigurationTests(unittest.TestCase):
    """Use real relocated study manifests and fake model backends only."""
    def _copy(self, root):
        """Create a complete portable configuration tree without any runtime artifacts."""
        shutil.copytree(ROOT / 'config', root / 'config')
        return root / 'config/shared/quality.toml'

    def _settings(self, root, study):
        """Resolve the selected study and keep fake generation settings deterministic."""
        settings = load_settings(root / f'config/studies/{study}/study.toml')
        settings.values['ollama']['models'] = ['fixture:1']
        return settings

    def test_shared_profile_is_explicit_validated_and_fingerprinted(self):
        """Missing references, malformed templates and invalid numeric settings fail before inference."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._copy(root)
            settings = self._settings(root, 'ids_repositories')
            self.assertEqual(settings.path('quality'), shared)
            self.assertEqual(settings.section('study')['input_fingerprints'][str(shared)], file_hash(shared))
            manifest = root / 'config/studies/ids_repositories/study.toml'
            manifest.write_text(manifest.read_text().replace('quality = "../../shared/quality.toml"\n', ''))
            with self.assertRaisesRegex(ValueError, 'quality'):
                load_settings(manifest)
            original = shared.read_text()
            for before, after in (('max_quotation_passages = 4', 'max_quotation_passages = 0'),
                ('max_block_passages = 0', 'max_block_passages = -1'),
                ('entailment_threshold = 0.5', 'entailment_threshold = nan'),
                ('label_margin = 0.1', 'label_margin = true'),
                ('${criterion}', '${unknown}'), ('${instructions}', '${missing}')):
                shared.write_text(original.replace(before, after))
                with self.subTest(setting=after), self.assertRaises(ValueError):
                    load_quality_config(shared)

    def test_editing_shared_instructions_updates_every_llm_without_editing_class_templates(self):
        """RADAR and plain LLM receive the same quality wording in both evidence formats."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._copy(root)
            templates = {path: file_hash(path) for path in (root / 'config/studies').glob('*/prompts/classification.toml')}
            shared.write_text(shared.read_text().replace('Assess these content-quality', 'SHARED_MARKER Assess these content-quality'))
            for study, source in (('ids_repositories', 'github'), ('ids_datasets', 'openalex'), ('conformal_treatment_effects', 'cohort')):
                settings = self._settings(root, study)
                payload = {'paper_id': 'one', 'title': 'Paper', 'full_text': TEXT}
                plain, _ = plain_llm_task(settings, source, payload)
                if source == 'cohort':
                    protocol = load_study_protocol(settings.path('screening'))
                    radar, _ = _assessment_prompt(protocol, load_assessment_fields(protocol), payload,
                        load_assessment_template(protocol)[0], EvidenceBlockIndex(TEXT))
                else:
                    profile = load_prompt_profile(settings, source)
                    sections = [EvidenceSection('README' if source == 'github' else 'Abstract', 'Literal ${instructions} text')]
                    radar = repository_prompt(sections, profile, load_system_type_profile(settings, source)) if source == 'github' else paper_prompt(sections, profile)
                    self.assertIn('Literal ${instructions} text', radar)
                self.assertIn('SHARED_MARKER', radar)
                self.assertIn('SHARED_MARKER', plain)
                self.assertEqual(radar.count('SHARED_MARKER'), 1)
                self.assertEqual(plain.count('SHARED_MARKER'), 1)
            self.assertTrue(all(file_hash(path) == digest for path, digest in templates.items()))

    def test_evidence_limits_control_both_schemas_and_local_validation(self):
        """Configured quotation and block limits have the same meaning in requests and validation."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._copy(root)
            shared.write_text(shared.read_text().replace('max_quotation_passages = 4', 'max_quotation_passages = 1').replace('max_block_passages = 0', 'max_block_passages = 1'))
            settings = self._settings(root, 'ids_repositories')
            rules = settings.section('screening')
            schema = quality_schema({'properties': {}, 'required': []}, rules)
            self.assertEqual(schema['properties']['quality']['properties']['Q1']['properties']['evidence']['maxItems'], 1)
            item = {'value': 'yes', 'evidence': [{'source': 'README', 'quote': 'Source text'}] * 2}
            row = {'decision': 'accept'}
            assess_llm_quality(row, {key: item for key in rules['quality']}, [EvidenceSection('README', 'Source text')], rules, grounded=True)
            self.assertEqual((row['decision'], row['quality_status']), ('accept', 'error'))
            protocol = load_study_protocol(root / 'config/studies/conformal_treatment_effects/screening.toml')
            text = 'First source paragraph.\n\nSecond source paragraph.'
            index = EvidenceBlockIndex(text)
            prompt, schema = _assessment_prompt(protocol, load_assessment_fields(protocol), {'full_text': text}, load_assessment_template(protocol)[0], index)
            self.assertEqual(schema['properties']['criteria']['properties']['Q1']['properties']['evidence_ids']['maxItems'], 1)
            self.assertNotIn('maxItems', schema['properties']['criteria']['properties']['F1']['properties']['evidence_ids'])
            response = {'criteria': {key: {'value': 'unknown', 'evidence_ids': []} for key in [*protocol.filters, *protocol.quality]},
                'fields': {key: {'value': 'unknown', 'evidence_ids': []} for key in protocol.contract.fields}}
            response['criteria']['Q1'] = {'value': 'yes', 'evidence_ids': [block.id for block in index.blocks]}
            self.assertEqual(set(_assessment_errors(response, protocol, load_assessment_fields(protocol), index)), {'criteria.Q1'})

    def test_nli_uses_shared_cutoffs_and_templates_with_independent_relevance_cutoffs(self):
        """Quality scoring uses the selected NLI backend and records the effective shared configuration."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._copy(root)
            shared.write_text(shared.read_text().replace('entailment_threshold = 0.5', 'entailment_threshold = 0.9').replace('"${criterion}"', '"NLI_MARKER ${criterion}"'))
            settings = self._settings(root, 'ids_repositories')
            config = load_nli_runtime_config(settings)
            self.assertEqual((config.entailment_threshold, config.quality_entailment_threshold), (.5, .9))
            rules = settings.section('screening')
            backend = QualityNLIBackend([NLIScore(.8, .1, .1)] * len(rules['quality']))
            row = {'decision': 'accept'}
            assess_nli_quality(row, [EvidenceSection('README', 'Source text')], rules, backend, config, 'nli')
            self.assertEqual((row['decision'], row['quality_status']), ('accept', 'unknown'))
            self.assertEqual(row['quality_config_hash'], file_hash(shared))
            self.assertTrue(all(hypothesis.startswith('NLI_MARKER ') for _, hypothesis in backend.calls[0]))

    def test_shared_changes_invalidate_plain_and_radar_caches_and_old_ensemble_members(self):
        """A shared quality edit cannot silently reuse an assessment or mix old/new model judgments."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = self._copy(root)
            settings = self._settings(root, 'ids_repositories')
            candidate = {'github_repo_id': 'one', 'full_name': 'fixture/repo'}
            source = root / 'candidates.csv'
            atomic_write_csv(source, [candidate], list(candidate))
            _, schema = plain_llm_task(settings, 'github', {})
            response = schema_answer(schema)
            response['decision'] = 'accept'
            with patch('radar.common.ollama.requests.post', return_value=completion(response)) as post:
                first = classify_plain_llm(settings, 'github', source, root / 'plain.csv', 'fixture:1', 'd1', [(candidate, {})], resume=True)[0]
                classify_plain_llm(settings, 'github', source, root / 'plain.csv', 'fixture:1', 'd1', [(candidate, {})], resume=True)
                self.assertEqual(post.call_count, 1)
                protocol = root / 'config/studies/conformal_treatment_effects/screening.toml'
                review_source = root / 'review.json'
                review_source.write_text(json.dumps([{'paper_id': 'one', 'full_text': TEXT}]))
                model = FakeModel()
                classify_study(protocol, review_source, root / 'a.csv', {'model': 'a'}, client=model, resume=True)
                classify_study(protocol, review_source, root / 'a.csv', {'model': 'a'}, client=model, resume=True)
                self.assertEqual(len(model.calls), 1)
                shared.write_text(shared.read_text().replace('Assess these content-quality', 'REVISED Assess these content-quality'))
                settings = self._settings(root, 'ids_repositories')
                second = classify_plain_llm(settings, 'github', source, root / 'plain.csv', 'fixture:1', 'd1', [(candidate, {})], resume=True)[0]
                self.assertEqual(post.call_count, 2)
                self.assertNotEqual(first['classifier_profile_hash'], second['classifier_profile_hash'])
                classify_study(protocol, review_source, root / 'b.csv', {'model': 'b'}, client=FakeModel())
                with self.assertRaisesRegex(ValueError, 'quality configuration|incompatible'):
                    ensemble_study(protocol, {'a': root / 'a.csv', 'b': root / 'b.csv'}, root / 'ensemble.csv', .8)
                classify_study(protocol, review_source, root / 'a.csv', {'model': 'a'}, client=model, resume=True)
                self.assertEqual(len(model.calls), 2)
                checkpoint = json.loads((root / 'a.assessments.json').read_text())
                self.assertEqual(checkpoint['quality_config_sha256'], file_hash(shared))
