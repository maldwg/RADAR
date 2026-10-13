from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from tests.fixtures import SYSTEM_TYPE_PROFILE
from unittest.mock import patch

import requests

from ids_meta.cli import _classifier_entries, _execute_baselines, build_parser, main
from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError, Settings, load_settings
from ids_meta.common.csvio import read_csv
from tests.fixtures import write_csv as atomic_write_csv
from ids_meta.common.evaluation import evaluate_results, load_rating_profile
from ids_meta.common.llm_nli import LLMNLIConfig, NLIScoringError, OllamaNLIBackend, augmented_profile, load_llm_nli_config, parse_class_scores
from ids_meta.common.nli import NLIField, NLIProfile, NLIRuntimeConfig, TransformersNLIBackend, classify_nli_baseline, rescore_nli_rows
from ids_meta.common.ollama import EvidenceSection
from ids_meta.common.text import slug
from ids_meta.plotting.evaluation import build as plot_evaluation

ROOT = Path(__file__).resolve().parents[1]


def scored_response(values=(.8, .15, .05), *, prefix=False):
    """Construct a native Ollama response with auditable three-class alternatives."""
    tokens = [{"token": code, "logprob": math.log(value)} for code, value in zip("ABC", values)]
    winner = tokens[max(range(3), key=lambda i: values[i])]
    entries = [{**winner, "top_logprobs": tokens}]
    if prefix:
        entries.insert(0, {"token": "\n", "logprob": -1, "top_logprobs": []})
    return {"message": {"content": "\n" + winner["token"]}, "logprobs": entries, "prompt_eval_count": 20, "eval_count": 1}


class WordTokenizer:
    """Reversible tokenizer fixture independent of downloaded models."""
    def encode(self, text, **kwargs):
        """Treat words as fixture token IDs."""
        return text.split()

    def decode(self, values, **kwargs):
        """Recover fixture text from word tokens."""
        return " ".join(values)


class FakeSession:
    def __init__(self):
        """Initialize the request audit and optional score-completeness failure."""
        self.headers = {}
        self.calls = []
        self.incomplete = False
        self.digest = 'digest-1'

    def request(self, method, url, json=None, timeout=None):
        """Serve pinned metadata and deterministic class scores without networking."""
        self.calls.append((method, url, json))
        if url.endswith('/version'):
            data = {"version": "fixture-version"}
        elif url.endswith('/tags'):
            data = {"models": [{"name": "fixture:1", "digest": self.digest}]}
        elif url.endswith('/show'):
            data = {"capabilities": ["completion"]}
        else:
            prompt = json["messages"][0]["content"]
            data = scored_response((.8, .15, .05) if "MATCH" in prompt else (.1, .8, .1))
            if self.incomplete:
                data["logprobs"][0]["top_logprobs"] = data["logprobs"][0]["top_logprobs"][:2]
        class Response:
            def raise_for_status(self):
                """Represent a successful HTTP response."""
                pass
            def json(self):
                """Return the selected response fixture."""
                return data
        return Response()


def reference_config():
    """Return a small shared NLI evidence and decision policy."""
    return NLIRuntimeConfig("fixture/tokenizer", "tokenizer-revision", "cpu", 2, 128, 32, 4, 2, 1, .5, .1)


class LLMNLITests(unittest.TestCase):
    def setUp(self):
        """Use the production prompt with a deterministic fake model."""
        self.config = load_llm_nli_config(load_settings())
        self.config = replace(self.config, models=("fixture:1",), retries=0)

    def backend(self, cache=None, session=None):
        """Build a real scorer with injected HTTP and tokenizer fixtures."""
        return OllamaNLIBackend("fixture:1", self.config, reference_config(), {}, cache, session=session or FakeSession(), tokenizer=WordTokenizer())

    def test_scores_are_conditional_and_preserve_original_class_mass(self):
        """Normalize only class codes while preserving their original probability mass."""
        result, audit = parse_class_scores(scored_response((.4, .1, .05), prefix=True))
        self.assertAlmostEqual(result.entailment, .4 / .55)
        self.assertAlmostEqual(result.entailment + result.neutral + result.contradiction, 1)
        self.assertAlmostEqual(audit["label_mass"], .55)
        self.assertEqual(audit["position"], 1)
        # Different single-token whitespace verbalizers represent the same class.
        response = scored_response((.4, .1, .05))
        response["logprobs"][0]["top_logprobs"].append({"token": " A", "logprob": math.log(.1)})
        result, audit = parse_class_scores(response)
        self.assertAlmostEqual(result.entailment, .5 / .65)

    def test_invalid_or_incomplete_scores_never_become_neutral_or_zero(self):
        """Keep missing scores and malformed answers separate from semantic neutral."""
        mutations = [
            lambda r: r.pop("logprobs"),
            lambda r: r["logprobs"][0]["top_logprobs"].pop(),
            lambda r: r["message"].update(content="A because it is supported"),
            lambda r: r["message"].update(thinking="reasoning"),
            lambda r: r["logprobs"][0]["top_logprobs"][1].update(logprob=float('nan')),
            lambda r: r["logprobs"][0]["top_logprobs"].append({"token": "B", "logprob": -.1}),
            lambda r: r["logprobs"][0].update(token="Answer: A"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                response = scored_response()
                mutate(response)
                with self.assertRaises(NLIScoringError):
                    parse_class_scores(response)
        with self.assertRaises(NLIScoringError):
            parse_class_scores(scored_response((.1, .05, .01)), minimum_mass=.5)

    def test_chunking_matches_deberta_without_loading_its_classifier(self):
        """Keep premise texts identical across the two scoring backends."""
        backend = self.backend()
        reference = TransformersNLIBackend.__new__(TransformersNLIBackend)
        reference.tokenizer = WordTokenizer()
        sections = [EvidenceSection("README", "one two three four five six"), EvidenceSection("Docs", "seven eight nine")]
        self.assertEqual(backend.chunk_sections(sections, 3, 1), reference.chunk_sections(sections, 3, 1))

    def test_pair_cache_survives_threshold_changes_and_request_preserves_scores(self):
        """Reuse pair probabilities independently of cutoffs and avoid grammar masking."""
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / "cache.sqlite3") as cache:
            session = FakeSession()
            backend = self.backend(cache, session)
            first = backend.score([("evidence", "MATCH hypothesis")], 2)
            calls = len(session.calls)
            resumed = backend.score([("evidence", "MATCH hypothesis")], 2)
            self.assertEqual(len(session.calls), calls)
            self.assertEqual(first, resumed)
            changed = replace(self.config, entailment_threshold=.9)
            self.assertEqual(changed.scoring_hash(reference_config()), self.config.scoring_hash(reference_config()))
            request = session.calls[-1][2]
            self.assertTrue(request["logprobs"])
            self.assertFalse(request["think"])
            self.assertFalse(request["truncate"])
            self.assertNotIn("format", request)
            self.assertEqual(backend.metadata["model_digest"], "digest-1")
            self.assertNotEqual(replace(self.config, template=self.config.template + "changed").scoring_hash(reference_config()), self.config.scoring_hash(reference_config()))
            replacement_session = FakeSession()
            replacement_session.digest = 'digest-2'
            replacement = self.backend(cache, replacement_session)
            replacement.score([('evidence', 'MATCH hypothesis')], 2)
            self.assertEqual(len(replacement_session.calls), 4)

    def test_transport_failure_keeps_pair_audit_and_reuses_completed_prefix(self):
        """Retry failed pairs without repeating successfully cached prefix requests."""
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / 'cache.sqlite3') as cache:
            session = FakeSession()
            backend = self.backend(cache, session)
            original_request = session.request

            def request(method, url, **kwargs):
                """Fail one pair's transport after the previous pair was successfully scored."""
                if url.endswith('/chat') and 'FAIL' in kwargs['json']['messages'][0]['content']:
                    raise requests.ConnectionError('fixture transport failure')
                return original_request(method, url, **kwargs)

            session.request = request
            pairs = [('evidence', 'MATCH'), ('evidence', 'FAIL')]
            with self.assertRaises(NLIScoringError):
                backend.score(pairs, 1)
            self.assertEqual(len(backend.audit), 2)
            self.assertEqual(backend.audit[-1]['hypothesis'], 'FAIL')
            self.assertIn('ConnectionError', backend.audit[-1]['error'])
            session.request = original_request
            backend.start_record()
            backend.score(pairs, 1)
            self.assertTrue(backend.audit[0]['cache_hit'])
            self.assertFalse(backend.audit[1]['cache_hit'])

    def test_both_sources_evaluate_plot_and_resume_independent_llm_nli_outputs(self):
        """Classify, resume and plot both source contracts through shared NLI inference."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, identity, field in [('github', 'github_repo_id', 'role'), ('openalex', 'openalex_id', 'paper_role')]:
                with self.subTest(source=source), PipelineCache(root / f'{source}.sqlite3') as cache:
                    profile = NLIProfile(source, (NLIField(field, "exclusive", "uncertain", {"out_of_scope" if source == 'github' else "survey": "MATCH excluded", "native_detector" if source == 'github' else "original_release": "neutral"}),), {"fixture": source}, root / 'profile.toml')
                    backend = self.backend(cache)
                    profile = augmented_profile(profile, self.config, reference_config(), backend.metadata)
                    candidates = root / f'{source}-candidates.csv'
                    output = root / source / 'classifications.csv'
                    atomic_write_csv(candidates, [{identity: '1', 'evidence_hash': 'candidate', 'description': 'README evidence', 'title': 'paper'}], [identity, 'evidence_hash', 'description', 'title'])
                    result = classify_nli_baseline(source, candidates, output, cache, profile, backend.runtime_config(), backend=backend, method='llm_nli', progress_every=1, adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
                    self.assertEqual(result[0]['decision'], 'exclude')
                    self.assertEqual(result[0]['classifier_method'], 'llm_nli')
                    self.assertEqual(result[0]['classifier_supported_fields'], 'decision;' + field)
                    self.assertTrue(json.loads(read_csv(output)[0]['llm_nli_pair_scores_json']))
                    count = len(backend.session.calls)
                    resumed = classify_nli_baseline(source, candidates, output, cache, profile, backend.runtime_config(), backend=backend, method='llm_nli', adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
                    self.assertEqual(len(backend.session.calls), count)
                    self.assertEqual(resumed[0]['classified_at'], result[0]['classified_at'])
                    rating = root / f'{source}-ratings.toml'
                    rating.write_text(f'schema_version="1.0"\nsource="{source}"\nclassification_fields=["{field}"]\n[[entities]]\nid="1"\ndecision="exclude"\n')
                    report = root / source / 'evaluation'
                    evaluation = evaluate_results(load_rating_profile(rating, source), output, report, candidates)
                    self.assertEqual(evaluation.metrics['classifier_method'], 'llm_nli')
                    self.assertEqual(evaluation.metrics['selective']['tn'], 1)
                    plot_evaluation(report, {'formats': ['svg'], 'dpi': 72})
                    self.assertTrue((report / 'figures/08_reference_uncertainty.svg').exists())
                    replay = rescore_nli_rows(source, result, profile, .9, .1, adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
                    self.assertEqual(replay[0]['decision'], 'uncertain')
                    self.assertEqual(replay[0]['classifier_supported_fields'], 'decision;' + field)

    def test_scoring_failure_is_an_audited_error_not_an_abstention(self):
        """Persist incomplete alternatives as an error with the failed pair audit."""
        with tempfile.TemporaryDirectory() as directory, PipelineCache(Path(directory) / 'cache.sqlite3') as cache:
            root = Path(directory)
            backend = self.backend(cache)
            backend.session.incomplete = True
            profile = NLIProfile('github', (NLIField('role', 'exclusive', 'uncertain', {'out_of_scope': 'MATCH', 'native_detector': 'neutral'}),), {}, root / 'profile.toml')
            profile = augmented_profile(profile, self.config, reference_config(), backend.metadata)
            candidates = root / 'candidates.csv'
            atomic_write_csv(candidates, [{'github_repo_id': '1', 'evidence_hash': 'x'}], ['github_repo_id', 'evidence_hash'])
            result = classify_nli_baseline('github', candidates, root / 'output.csv', cache, profile, backend.runtime_config(), backend=backend, method='llm_nli', adapter_config={'system_type_profile': SYSTEM_TYPE_PROFILE})
            self.assertEqual(result[0]['decision'], 'error')
            self.assertEqual(result[0]['nli_scores_json'], '{}')
            self.assertIn('Incomplete', result[0]['error'])
            self.assertIn('error', json.loads(result[0]['llm_nli_pair_scores_json'])[0])

    def test_cli_capability_check_uses_no_dataset_and_pilots_do_not_enter_discovery(self):
        """Keep capability checks corpus-free and partial results out of comparisons."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = copy.deepcopy(load_settings().values)
            for key in ('interim_dir', 'processed_dir', 'plots_dir', 'runs_dir', 'evaluation_dir'):
                values['paths'][key] = str(root / key)
            values['paths']['llm_nli_prompt'] = str(ROOT / 'config/studies/ids_repositories/baselines/llm_nli.toml')
            values['baselines']['llm_nli']['models'] = ['fixture:1']
            settings = Settings(ROOT, values)
            args = build_parser().parse_args(['github', 'baseline', '--methods', 'llm_nli', '--check-only'])
            with patch('ids_meta.common.llm_nli.requests.Session', return_value=FakeSession()):
                result = _execute_baselines(args, settings, 'github')
            self.assertEqual(result['baseline_comparison_files'], 0)
            self.assertFalse(list(root.glob('**/classifications.csv')))
            interim = Path(values['paths']['interim_dir']) / 'github/baselines/llm_nli/fixture-1'
            atomic_write_csv(interim / 'classifications.csv', [], ['github_repo_id'])
            atomic_write_csv(interim / 'pilots/role-5/classifications.csv', [], ['github_repo_id'])
            entries = _classifier_entries(settings, 'github')
            self.assertEqual(len(entries), 1)
            self.assertIn('NLI - fixture-1', entries)

    def test_cli_pilots_for_both_sources_rescore_offline_and_reject_prompt_changes(self):
        """Exercise isolated CLI pilots, cutoff replay and fingerprint validation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for source, identity, field in [('github', 'github_repo_id', 'role'), ('openalex', 'openalex_id', 'paper_role')]:
                with self.subTest(source=source):
                    study = 'ids_repositories' if source == 'github' else 'ids_datasets'
                    values = copy.deepcopy(load_settings(ROOT / f'config/studies/{study}/study.toml').values)
                    for key in ('raw_dir', 'interim_dir', 'processed_dir', 'plots_dir', 'runs_dir', 'evaluation_dir'):
                        values['paths'][key] = str(root / source / key)
                    values['baselines']['llm_nli']['models'] = ['fixture:1']
                    values['baselines']['llm_nli']['retries'] = 0
                    values['baselines']['nli']['sensitivity']['enabled'] = False
                    values['plots'].update(formats=['svg'], dpi=72)
                    settings = Settings(ROOT, values)
                    profile = NLIProfile(source, (NLIField(field, 'exclusive', 'uncertain', {
                        'out_of_scope' if source == 'github' else 'survey': 'MATCH excluded',
                        'native_detector' if source == 'github' else 'original_release': 'neutral',
                    }),), {'fixture': source}, root / 'profile')
                    candidates = root / f'{source}-candidates.csv'
                    atomic_write_csv(candidates, [{identity: '1', 'description': 'evidence', 'evidence_hash': 'one'}, {identity: '2', 'evidence_hash': 'two'}], [identity, 'description', 'evidence_hash'])
                    rating = root / f'{source}-ratings.toml'
                    rating.write_text(f'schema_version="1.0"\nsource="{source}"\nclassification_fields=["{field}"]\n[[entities]]\nid="1"\ndecision="exclude"\n')
                    command = [source, 'baseline', '--methods', 'llm_nli', '--fields', field, '--limit', '1', '--input', str(candidates), '--ratings', str(rating)]
                    session = FakeSession()
                    def factory(model, config, reference, ollama, cache):
                        """Inject local fixtures while retaining production backend behavior."""
                        return OllamaNLIBackend(model, config, reference, ollama, cache, session=session, tokenizer=WordTokenizer())
                    with patch('ids_meta.cli.load_nli_profile', return_value=profile), patch('ids_meta.common.llm_nli.OllamaNLIBackend', side_effect=factory):
                        _execute_baselines(build_parser().parse_args(command), settings, source)
                    output = Path(values['paths']['interim_dir']) / source / f'baselines/llm_nli/fixture-1/pilots/{slug(field)}-1/classifications.csv'
                    rows = read_csv(output)
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['decision'], 'exclude')
                    self.assertEqual(_classifier_entries(settings, source), {})
                    values['baselines']['llm_nli']['entailment_threshold'] = .9
                    with patch('ids_meta.cli.load_nli_profile', return_value=profile), patch('ids_meta.common.llm_nli.OllamaNLIBackend', side_effect=AssertionError('Offline replay called Ollama')):
                        _execute_baselines(build_parser().parse_args(command + ['--evaluate-only']), settings, source)
                        self.assertEqual(read_csv(output)[0]['decision'], 'uncertain')
                        _execute_baselines(build_parser().parse_args(command + ['--evaluate-only']), settings, source)
                    changed = replace(profile, raw={'fixture': 'changed'})
                    with patch('ids_meta.cli.load_nli_profile', return_value=changed), self.assertRaisesRegex(ConfigError, 'hypotheses/chunking'):
                        _execute_baselines(build_parser().parse_args(command + ['--evaluate-only']), settings, source)

    def test_invalid_configuration_is_rejected_before_inference(self):
        """Reject unsupported score limits and unknown pilot field names."""
        settings = load_settings()
        values = copy.deepcopy(settings.values)
        values['baselines']['llm_nli']['top_logprobs'] = 21
        with self.assertRaises(ConfigError):
            load_llm_nli_config(Settings(settings.root, values))
        backend = self.backend()
        profile = NLIProfile('github', (NLIField('role', 'exclusive', 'uncertain', {'native_detector': 'MATCH'}),), {}, Path('profile'))
        with self.assertRaises(ConfigError):
            augmented_profile(profile, self.config, reference_config(), backend.metadata, ['imaginary'])

    def test_check_and_pilot_commands_do_not_rebuild_full_corpus_funnels(self):
        """Keep top-level CLI postprocessing from reading full results during small checks."""
        for source, role in [('github', 'role'), ('openalex', 'paper_role')]:
            for options in [['--check-only'], ['--fields', role, '--limit', '1']]:
                with self.subTest(source=source, options=options), patch('ids_meta.cli._execute_baselines', return_value={}), patch('ids_meta.funnel.build', side_effect=AssertionError('Full funnel accessed')):
                    study = 'ids_repositories' if source == 'github' else 'ids_datasets'
                    self.assertEqual(main(['--study', study, 'baseline', '--methods', 'llm_nli', *options]), 0)


if __name__ == '__main__':
    unittest.main()
