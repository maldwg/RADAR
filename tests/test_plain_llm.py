"""Verify single-pass baselines use frozen inputs and preserve model failures and decisions."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

import requests

from radar.cli import _classifier_entries, _execute_baselines, _execute_review, build_parser
from radar.common.cache import PipelineCache
from radar.common.config import ConfigError, load_settings
from radar.common.csvio import atomic_write_csv, read_csv
from radar.common.plain_llm import (classify_plain_llm, load_plain_llm_results, plain_llm_config,
    plain_llm_inputs, plain_llm_models, plain_llm_task)
from radar.common.provenance import file_hash

ROOT = Path(__file__).resolve().parents[1]


def schema_answer(schema):
    """Supply a structurally valid reply without relying on RADAR's scientific rules."""
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema.get("type")
    if kind == "object":
        return {name: schema_answer(schema["properties"][name]) for name in schema.get("required", [])}
    if kind == "array":
        return [schema_answer(schema["items"]) for _ in range(schema.get("minItems", 0))]
    if kind == "boolean":
        return False
    return "Model's own assessment"


def completion(answer, *, reason="stop", text=None, model="fixture:1"):
    """Return the actual Ollama envelope, including unmodified generated text and usage."""
    raw = json.dumps(answer) if text is None else text
    body = {"model": model, "response": raw, "done": True, "done_reason": reason,
        "prompt_eval_count": 101, "eval_count": 23, "total_duration": 1000000000}
    response = Mock()
    response.text = json.dumps(body)
    response.json.return_value = body
    return response


class PlainLLMTests(unittest.TestCase):
    """Exercise real request handling with an offline model server fixture."""
    def _settings(self, root, study="ids_repositories"):
        """Load the real study definitions with temporary output and one fixture model."""
        shutil.copytree(ROOT / "config", root / "config")
        settings = load_settings(root / f"config/studies/{study}/study.toml")
        settings.values["ollama"]["models"] = ["fixture:1"]
        settings.values["baselines"]["methods"] = ["plain_llm"]
        settings.values["baselines"]["llm_nli"]["enabled"] = False
        settings.values["ensemble"]["enabled"] = False
        settings.values["ensemble"]["models"] = []
        settings.values["plots"].update(formats=["svg"], dpi=60, top_n=3)
        return settings

    def _catalogue(self, settings, source, cache):
        """Save originals and a differently enriched working copy for a ranked candidate."""
        identity = "github_repo_id" if source == "github" else "openalex_id"
        namespace = "github.repository" if source == "github" else "openalex.work"
        path = settings.path("interim_dir") / source / ("candidates.csv" if source == "github" else "paper_candidates.csv")
        payload = ({"repository": {"id": 1, "full_name": "fixture/detector"}, "readme_text": "ORIGINAL_EVIDENCE_MARKER"}
            if source == "github" else {"work": {"native_id": "W1", "title": "Paper", "abstract": "ORIGINAL_EVIDENCE_MARKER"}})
        candidate = {identity: "1", "full_name": "fixture/detector", "title": "Paper", "evidence_hash": "candidate",
            "prefilter_band": "candidate", "prefilter_score": "9"}
        atomic_write_csv(path, [candidate], list(candidate))
        cache.put_collected_record(namespace, "1", payload, "now")
        enriched = copy.deepcopy(payload)
        if source == "github":
            enriched["readme_text"] = "ENRICHED_MUST_NOT_REACH_MODEL"
        else:
            enriched["work"]["full_text"] = "ENRICHED_MUST_NOT_REACH_MODEL"
        cache.put_record(namespace, "1", enriched, "later")
        return path, payload

    def _review(self, settings):
        """Preserve an audited original JSON cohort and a separate expanded prefilter file."""
        original = settings.path("interim_dir") / "cohort/candidates.json"
        original.parent.mkdir(parents=True)
        with Path(settings.section("evaluation")["references"]).open("rb") as handle:
            title = tomllib.load(handle)["entities"][0]["title"]
        payload = {"paper_id": "doi:10.1234/example", "title": title, "abstract": "ORIGINAL_EVIDENCE_MARKER",
            "full_text": "", "full_text_complete": False, "expected_decision": "GOLD_MUST_NOT_REACH_MODEL"}
        original.write_text(json.dumps([payload]))
        original.with_suffix(".collection.json").write_text(json.dumps({"candidate_sha256": file_hash(original), "providers": ["openalex"]}))
        enriched = {**payload, "full_text": "ENRICHED_MUST_NOT_REACH_MODEL", "prefilter_band": "candidate"}
        prefiltered = original.parent / "prefiltered.json"
        prefiltered.write_text(json.dumps([enriched]))
        return original, prefiltered, payload

    def _args(self, *, evaluate_only=False, limit=None):
        """Supply the same options as an explicit baseline CLI invocation."""
        return argparse.Namespace(action="baseline", input=None, output=None, methods="plain_llm", ratings=None,
            resume=False, evaluate_only=evaluate_only, check_only=False, fields=None, limit=limit, source=None)

    def test_configuration_inherits_models_and_forbids_pipeline_options(self):
        """Plain models can differ while retry and reconciliation settings cannot be enabled."""
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            self.assertEqual(plain_llm_models(settings), ["fixture:1"])
            self.assertEqual(plain_llm_config(settings, "fixture:1")["retries"], 0)
            self.assertEqual(plain_llm_config(settings, "fixture:1")["max_output_tokens"], settings.section("ollama")["max_output_tokens"])
            settings.values["baselines"]["plain_llm"] = {"models": ["other:2"], "temperature": 0.2}
            self.assertEqual(plain_llm_models(settings), ["other:2"])
            for invalid in ({"retries": 1}, {"context_length": -1}, {"models": ["x:y", "x-y"]}):
                settings.values["baselines"]["plain_llm"] = invalid
                with self.assertRaises(ConfigError):
                    plain_llm_models(settings)
            self.assertEqual(build_parser().parse_args(["baseline", "--methods", "plain_llm", "--limit", "2"]).limit, 2)

    def test_catalogue_original_evidence_and_model_decision_are_preserved(self):
        """Both catalogue workflows call once and retain contradictory but well-formed answers."""
        for source, study in (("github", "ids_repositories"), ("openalex", "ids_datasets")):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = self._settings(root, study)
                with PipelineCache(root / "cache.sqlite3") as cache:
                    path, payload = self._catalogue(settings, source, cache)
                    inputs = plain_llm_inputs(settings, source, path, cache)
                    prompt, schema = plain_llm_task(settings, source, payload)
                    answer = schema_answer(schema)
                    answer["decision"] = "accept"
                    if source == "github":
                        answer["role"] = "out_of_scope"
                    else:
                        answer["paper_role"], answer["relevance_tier"] = "unrelated", "excluded"
                    output = root / "plain.csv"
                    with patch("radar.common.ollama.requests.post", return_value=completion(answer)) as post, patch(
                        "radar.common.paper_evidence.PaperEvidenceService.prepare", side_effect=AssertionError("enrichment")), patch(
                        "radar.common.ollama.verify_evidence", side_effect=AssertionError("grounding")):
                        rows = classify_plain_llm(settings, source, path, output, "fixture:1", "digest-1", inputs)
                    self.assertEqual(post.call_count, 1)
                    request = post.call_args.kwargs["json"]
                    self.assertIn("ORIGINAL_EVIDENCE_MARKER", request["prompt"])
                    self.assertNotIn("ENRICHED_MUST_NOT_REACH_MODEL", request["prompt"])
                    self.assertEqual(rows[0]["decision"], "accept")
                    self.assertEqual(rows[0]["plain_llm_request_count"], "1")
                    self.assertEqual(json.loads(rows[0]["plain_llm_response_metadata_json"])["eval_count"], 23)
                    self.assertFalse(request["think"])
                    self.assertEqual(read_csv(output)[0]["decision"], "accept")
                    with patch("radar.common.ollama.requests.post", side_effect=AssertionError("inference")):
                        resumed = classify_plain_llm(settings, source, path, output, "fixture:1", "digest-1", inputs, resume=True)
                        self.assertEqual(resumed[0]["plain_llm_cache_reused"], "true")
                        self.assertEqual(load_plain_llm_results(settings, output, "fixture:1")[0]["decision"], "accept")
                    settings.values["ollama"]["temperature"] = 0.4
                    with self.assertRaisesRegex(ConfigError, "current configuration"):
                        load_plain_llm_results(settings, output, "fixture:1")

    def test_truncation_malformed_schema_and_http_errors_never_retry(self):
        """Every attempted candidate remains an error even when the transport has retries configured."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root)
            with PipelineCache(root / "cache.sqlite3") as cache:
                path, payload = self._catalogue(settings, "github", cache)
                inputs = plain_llm_inputs(settings, "github", path, cache)
                _, schema = plain_llm_task(settings, "github", payload)
                answer = schema_answer(schema)
                invalid = {**answer, "role": "made-up-role"}
                failures = [completion(answer, reason="length"), completion(answer, text='{"role":'),
                    completion(invalid), requests.ConnectionError("offline")]
                for index, failure in enumerate(failures):
                    with self.subTest(failure=index), patch("radar.common.ollama.requests.post", side_effect=[failure]) as post:
                        output = root / f"failure-{index}.csv"
                        rows = classify_plain_llm(settings, "github", path, output, "fixture:1", "digest-1", inputs)
                        self.assertEqual(post.call_count, 1)
                        self.assertEqual(rows[0]["decision"], "error")
                        self.assertEqual(len(read_csv(output)), 1)
                    with patch("radar.common.ollama.requests.post", side_effect=AssertionError("failed result retried")):
                        resumed = classify_plain_llm(settings, "github", path, output, "fixture:1", "digest-1", inputs, resume=True)
                        self.assertEqual(resumed[0]["decision"], "error")

    def test_review_uses_original_abstract_and_does_not_replay_criterion_rules(self):
        """Abstract-only originals are assessed once and no scientific postprocessor overrides the decision."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root, "conformal_treatment_effects")
            original, path, payload = self._review(settings)
            inputs = plain_llm_inputs(settings, "cohort", path, None, original_path=original)
            _, schema = plain_llm_task(settings, "cohort", payload)
            answer = schema_answer(schema)
            answer["decision"] = "accept"
            answer["criteria"]["F1"]["value"] = "no"
            with patch("radar.common.ollama.requests.post", return_value=completion(answer)) as post, patch(
                "radar.common.study_screening.StudyScreeningAdapter.normalize", side_effect=AssertionError("rule replay")):
                rows = classify_plain_llm(settings, "cohort", path, root / "plain.csv", "fixture:1", "digest-1", inputs)
            self.assertEqual(post.call_count, 1)
            self.assertIn("ORIGINAL_EVIDENCE_MARKER", post.call_args.kwargs["json"]["prompt"])
            self.assertNotIn("ENRICHED_MUST_NOT_REACH_MODEL", post.call_args.kwargs["json"]["prompt"])
            self.assertNotIn("GOLD_MUST_NOT_REACH_MODEL", post.call_args.kwargs["json"]["prompt"])
            self.assertEqual(rows[0]["decision"], "accept")
            self.assertEqual(json.loads(rows[0]["criteria_json"])["F1"]["value"], "no")
            original.write_text(original.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "collection audit"):
                plain_llm_inputs(settings, "cohort", path, None, original_path=original)

    def test_cache_invalidates_for_weights_inputs_and_scientific_instructions(self):
        """Changing model weights, original evidence or prompts requires a new single inference."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root)
            with PipelineCache(root / "cache.sqlite3") as cache:
                path, payload = self._catalogue(settings, "github", cache)
                inputs = plain_llm_inputs(settings, "github", path, cache)
                _, schema = plain_llm_task(settings, "github", payload)
                answer = schema_answer(schema)
                output = root / "plain.csv"
                with patch("radar.common.ollama.requests.post", return_value=completion(answer)) as post:
                    classify_plain_llm(settings, "github", path, output, "fixture:1", "d1", inputs)
                    classify_plain_llm(settings, "github", path, output, "fixture:1", "d2", inputs, resume=True)
                    payload["readme_text"] += " New original evidence."
                    cache.put_collected_record("github.repository", "1", payload, "new")
                    inputs = plain_llm_inputs(settings, "github", path, cache)
                    classify_plain_llm(settings, "github", path, output, "fixture:1", "d2", inputs, resume=True)
                    prompt = settings.path("prompts")
                    prompt.write_text(prompt.read_text().replace("You are", "You are carefully"))
                    classify_plain_llm(settings, "github", path, output, "fixture:1", "d2", inputs, resume=True)
                self.assertEqual(post.call_count, 4)

    def test_cli_runs_models_separately_and_evaluate_only_is_offline(self):
        """The baseline command writes isolated outputs and makes no inference or metadata calls offline."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root)
            settings.values["evaluation"]["enabled"] = False
            settings.values["baselines"]["plain_llm"] = {"models": ["fixture:1", "other:2"]}
            with PipelineCache(settings.path("github_cache")) as cache:
                _, payload = self._catalogue(settings, "github", cache)
            _, schema = plain_llm_task(settings, "github", payload)
            with patch("radar.common.plain_llm.requests.get") as get, patch(
                "radar.common.ollama.requests.post", side_effect=lambda *args, **kwargs: completion(schema_answer(schema), model=kwargs["json"]["model"])) as post:
                get.return_value.json.return_value = {"models": [{"name": name, "digest": name + "digest"} for name in ("fixture:1", "other:2")]}
                counts = _execute_baselines(self._args(), settings, "github")
            self.assertEqual(get.call_count, 1)
            self.assertEqual(post.call_count, 2)
            self.assertEqual(counts["plain_llm_fixture-1_rows"], 1)
            self.assertEqual(counts["plain_llm_fixture-1_prompt_tokens"], 101)
            self.assertEqual(counts["plain_llm_fixture-1_completion_tokens"], 23)
            self.assertEqual(len([name for name in _classifier_entries(settings, "github") if name.startswith("Plain LLM")]), 2)
            with patch("radar.common.plain_llm.requests.get", side_effect=AssertionError("metadata")), patch(
                "radar.common.ollama.requests.post", side_effect=AssertionError("inference")):
                _execute_baselines(self._args(evaluate_only=True), settings, "github")
            with patch("radar.common.plain_llm.requests.get", side_effect=AssertionError("metadata before snapshot validation")):
                with PipelineCache(settings.path("github_cache")) as cache:
                    cache.connection.execute("DELETE FROM records WHERE namespace='github.repository.collected'")
                    cache.connection.commit()
                with self.assertRaisesRegex(ValueError, "missing collected"):
                    _execute_baselines(self._args(), settings, "github")

    def test_review_baseline_cli_reports_errors_in_evaluation_and_is_offline_on_re_evaluation(self):
        """Real review reporting retains failed model attempts in the classification denominator."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root, "conformal_treatment_effects")
            self._review(settings)
            with patch("radar.common.plain_llm.requests.get") as get, patch(
                "radar.common.ollama.requests.post", side_effect=requests.ConnectionError("offline")) as post:
                get.return_value.json.return_value = {"models": [{"name": "fixture:1", "digest": "d1"}]}
                counts = _execute_review(self._args(), settings)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(counts["plain_llm_fixture-1_errors"], 1)
            metrics = settings.path("plots_dir") / "cohort/baselines/plain_llm/fixture-1/evaluation/metrics.json"
            self.assertTrue(metrics.is_file())
            self.assertEqual(json.loads(metrics.read_text())["operational"]["fn"], 11)
            instances = read_csv(metrics.parent / "instances.csv")
            self.assertEqual(sum(row["status"] == "error" for row in instances), 1)
            output = settings.path("interim_dir") / "cohort/baselines/plain_llm/fixture-1/classifications.csv"
            self.assertEqual(read_csv(output)[0]["decision"], "error")
            with patch("radar.common.plain_llm.requests.get", side_effect=AssertionError("metadata")), patch(
                "radar.common.ollama.requests.post", side_effect=AssertionError("inference")):
                self.assertEqual(_execute_review(self._args(evaluate_only=True), settings)["plain_llm_fixture-1_errors"], 1)

    def test_shortlist_matches_configured_radar_limits_and_pilots_are_separate(self):
        """Reject-band rows are excluded and isolated pilots never publish a full comparison input."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root)
            settings.values["evaluation"]["enabled"] = False
            settings.values["pipeline"]["classification"]["classification_limit"] = 1
            with PipelineCache(settings.path("github_cache")) as cache:
                path, payload = self._catalogue(settings, "github", cache)
                rows = read_csv(path)
                rows.extend([{**rows[0], "github_repo_id": "2", "prefilter_score": "99", "prefilter_band": "reject"},
                    {**rows[0], "github_repo_id": "3", "prefilter_score": "1"}])
                for key in ("2", "3"):
                    cache.put_collected_record("github.repository", key, payload, "now")
                atomic_write_csv(path, rows, list(rows[0]))
                inputs = plain_llm_inputs(settings, "github", path, cache)
                self.assertEqual([row["github_repo_id"] for row, _ in inputs], ["1"])
            _, schema = plain_llm_task(settings, "github", payload)
            with patch("radar.common.plain_llm.requests.get") as get, patch(
                "radar.common.ollama.requests.post", return_value=completion(schema_answer(schema))):
                get.return_value.json.return_value = {"models": [{"name": "fixture:1", "digest": "d1"}]}
                _execute_baselines(self._args(limit=1), settings, "github")
            self.assertFalse(_classifier_entries(settings, "github"))
            self.assertTrue((settings.path("interim_dir") / "github/baselines/plain_llm/fixture-1/pilots/1/classifications.csv").is_file())


    def test_both_catalogues_evaluate_plain_decisions_with_the_existing_evaluator(self):
        """Plain outputs reach the same reference evaluator without scientific normalization."""
        for source, study, field, value in (("github", "ids_repositories", "role", "native_detector"),
                ("openalex", "ids_datasets", "paper_role", "original_release")):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = self._settings(root, study)
                ratings = root / "ratings.toml"
                ratings.write_text(f'schema_version = "1.0"\nsource = "{source}"\nevaluation_unit = "source_record"\n'
                    f'classification_fields = ["{field}"]\n[[entities]]\nid = "1"\ndecision = "accept"\n{field} = "{value}"\n')
                settings.values["evaluation"]["reference_sets"] = [{"name": "fixture", "source": source, "profile": str(ratings)}]
                with PipelineCache(settings.path(f"{source}_cache")) as cache:
                    _, payload = self._catalogue(settings, source, cache)
                _, schema = plain_llm_task(settings, source, payload)
                with patch("radar.common.plain_llm.requests.get") as get, patch(
                    "radar.common.ollama.requests.post", return_value=completion(schema_answer(schema))):
                    get.return_value.json.return_value = {"models": [{"name": "fixture:1", "digest": "d1"}]}
                    counts = _execute_baselines(self._args(), settings, source)
                self.assertEqual(counts["plain_llm_fixture-1_rows"], 1)
                directory = settings.path("evaluation_dir") / source / "classifiers/plain_llm/fixture-1"
                metrics = json.loads((directory / "metrics.json").read_text())
                self.assertEqual(metrics["operational"]["tp"], 1)
                self.assertEqual(metrics["classifier_method"], "plain_llm")

    def test_saved_radar_ids_override_configured_cap_without_using_predictions(self):
        """A prior run's actual shortlist is reused even if its CLI cap differed from configuration."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root)
            settings.values["pipeline"]["classification"]["classification_limit"] = 1
            with PipelineCache(root / "cache.sqlite3") as cache:
                path, payload = self._catalogue(settings, "github", cache)
                rows = read_csv(path)
                rows.append({**rows[0], "github_repo_id": "2", "prefilter_score": "1"})
                cache.put_collected_record("github.repository", "2", payload, "now")
                atomic_write_csv(path, rows, list(rows[0]))
                saved = settings.path("interim_dir") / "github/models/fixture-1/classifications.csv"
                atomic_write_csv(saved, [{"github_repo_id": "2", "decision": "GOLD_OR_MODEL_DECISION_MUST_NOT_LEAK"}], ["github_repo_id", "decision"])
                inputs = plain_llm_inputs(settings, "github", path, cache)
                self.assertEqual([row["github_repo_id"] for row, _ in inputs], ["2"])
                self.assertNotIn("GOLD_OR_MODEL_DECISION", plain_llm_task(settings, "github", inputs[0][1])[0])
                settings.values["ollama"]["models"].append("second:2")
                other = saved.parents[1] / "second-2/classifications.csv"
                atomic_write_csv(other, [{"github_repo_id": "1"}], ["github_repo_id"])
                with self.assertRaisesRegex(ValueError, "different candidate cohorts"):
                    plain_llm_inputs(settings, "github", path, cache)

    def test_full_review_run_freezes_initial_input_and_compares_plain_and_radar(self):
        """Automatic baselines preserve initial evidence and use normalized evaluated comparison inputs."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root, "conformal_treatment_effects")
            original, _, payload = self._review(settings)
            supplied = root / "provided.json"
            supplied.write_text(original.read_text())
            original.unlink()
            original.with_suffix(".collection.json").unlink()
            args = self._args()
            args.action, args.input, args.dry_run = "run", supplied, False
            args.collect_limit = args.classify_limit = None
            args.collect_all = args.classify_all = False
            _, schema = plain_llm_task(settings, "cohort", payload)
            with patch("radar.common.paper_evidence.PaperEvidenceService.prepare", return_value={}), patch(
                "radar.common.plain_llm.requests.get") as get, patch(
                "radar.common.ollama.requests.post", return_value=completion(schema_answer(schema))) as post, redirect_stdout(io.StringIO()):
                get.return_value.json.return_value = {"models": [{"name": "fixture:1", "digest": "d1"}]}
                counts = _execute_review(args, settings)
            self.assertEqual(post.call_count, 1)
            self.assertEqual(counts["plain_llm_fixture-1_rows"], 1)
            self.assertEqual(json.loads(original.read_text())[0]["abstract"], "ORIGINAL_EVIDENCE_MARKER")
            self.assertEqual(json.loads(original.with_suffix(".collection.json").read_text())["origin"], "explicit_run_input")
            comparison = settings.path("evaluation_dir") / "cohort/comparison/data/00_metric_overview.csv"
            self.assertTrue(comparison.is_file())
            labels = {row["classifier"] for row in read_csv(comparison)}
            self.assertTrue(any(label.startswith("Plain LLM") for label in labels))
            self.assertTrue(any(label.startswith("RADAR LLM") for label in labels))


if __name__ == "__main__":
    unittest.main()
