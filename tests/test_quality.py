"""Keep content quality independent, backend-specific and scientifically auditable."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from radar.common.cache import PipelineCache
from radar.common.config import load_prompt_profile, load_settings, load_system_type_profile
from radar.common.csvio import atomic_write_csv, read_csv
from radar.common.evaluation import evaluate_results, load_rating_profile
from radar.common.nli import EvidenceChunk, NLIRuntimeConfig, NLIScore
from radar.common.ollama import EvidenceSection
from radar.common.plain_llm import classify_plain_llm, plain_llm_task
from radar.common.provenance import file_hash
from radar.common.quality import (assess_llm_quality, assess_nli_quality, finish_quality, quality_consensus,
    quality_prompt, quality_schema)
from radar.common.screening import load_screening_profile
from radar.common.quality_config import load_quality_config
from radar.common.study_screening import StudyScreeningAdapter, load_study_protocol
from radar.common.verification import verify_database, verify_directory
from radar.github.pipeline import classify as classify_github
from radar.plotting.quality import build_quality_reports
from tests.test_github_pipeline import FakeGitHubModel
from tests.test_plain_llm import completion, schema_answer
from tests.test_study_pipeline import TEXT, answer, FakeModel
from radar.common.study_pipeline import classify_study
from radar.common.study_consensus import ensemble_study

ROOT = Path(__file__).resolve().parents[1]
RULES = {"assessment": load_quality_config(ROOT / "config/shared/quality.toml"), "quality": {"Q1": "The documentation explains the artifact's purpose.",
    "Q2": "The documentation describes an execution example."},
    "decision": {"quality_minimum": 2, "quality_required": ["Q1"]}}
SECTIONS = [EvidenceSection("README", "This artifact detects intrusions.\n\nRun python detector.py.")]


class QualityNLIBackend:
    """Serve native NLI scores while recording exact quality hypotheses and evidence."""
    def __init__(self, scores):
        """Bind a controlled batch of scores, or a deliberate inference error."""
        self.scores = scores
        self.calls = []

    def chunk_sections(self, sections, premise_tokens, overlap):
        """Keep separate source passages in the backend's ordinary chunk interface."""
        return [EvidenceChunk(index, section.source, section.text) for index, section in enumerate(sections)]

    def score(self, pairs, batch_size):
        """Use only this NLI backend; no generative model is available to the fixture."""
        self.calls.append(list(pairs))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


class QualityTests(unittest.TestCase):
    """Exercise policy, uncertainty, original evidence, voting and independent reporting."""
    def _settings(self, root, study="ids_repositories"):
        """Load actual scientific definitions into a fresh project root."""
        shutil.copytree(ROOT / "config", root / "config")
        settings = load_settings(root / f"config/studies/{study}/study.toml")
        settings.values["ollama"]["models"] = ["fixture:1"]
        return settings

    def _judgments(self, value="yes"):
        """Supply explicit source quotations for both independently named criteria."""
        return {"Q1": {"value": value, "evidence": [{"source": "README", "quote": "This artifact detects intrusions."}]},
            "Q2": {"value": value, "evidence": [{"source": "README", "quote": "Run python detector.py."}]}}

    def test_all_studies_define_quality_in_screening_and_validate_mandatory_ids(self):
        """Every workflow owns quality content separately from taxonomy and class templates."""
        for study in ("ids_repositories", "ids_datasets", "conformal_treatment_effects"):
            profile = load_screening_profile(ROOT / f"config/studies/{study}/screening.toml")
            self.assertTrue(profile.quality)
            self.assertGreater(profile.quality_minimum, 0)
        with tempfile.TemporaryDirectory() as tmp:
            settings = self._settings(Path(tmp))
            path = settings.path("screening")
            path.write_text(path.read_text().replace('["Q1", "Q4"]', '["Q1", "Q99"]'))
            with self.assertRaisesRegex(ValueError, "quality_required"):
                load_screening_profile(path)

    def test_empty_quality_configuration_and_plain_requests_for_every_study(self):
        """Missing or empty quality disables assessment across studies, including explicit zero thresholds."""
        for study, source in (("ids_repositories", "github"), ("ids_datasets", "openalex"),
                              ("conformal_treatment_effects", "cohort")):
            with self.subTest(study=study), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = self._settings(root, study)
                path = settings.path("screening")
                prefix, tail = path.read_text().split("[quality]", 1)
                decision = tail.split("[decision]", 1)[1]
                decision = "\n".join(line for line in decision.splitlines()
                    if not line.startswith(("quality_minimum", "quality_required")))
                for section in ("", "[quality]\n\n"):
                    for cutoff in ("", "\nquality_minimum = 0\nquality_required = []\n"):
                        path.write_text(prefix + section + "[decision]" + cutoff + decision)
                        settings = load_settings(root / f"config/studies/{study}/study.toml")
                        profile = load_screening_profile(path)
                        self.assertEqual((profile.quality, profile.quality_minimum, profile.quality_required), ({}, 0, ()))
                        payload = {"paper_id": "one", "title": "Paper", "full_text": TEXT} if source == "cohort" else {}
                        prompt, schema = plain_llm_task(settings, source, payload)
                        if source == "cohort":
                            self.assertEqual(set(schema["properties"]["criteria"]["properties"]), set(profile.eligibility))
                            self.assertNotIn("Assess quality separately", prompt)
                        else:
                            self.assertNotIn("quality", schema["properties"])
                            self.assertNotIn("INDEPENDENT CONTENT QUALITY", prompt)
                identity = "paper_id" if source == "cohort" else "github_repo_id" if source == "github" else "openalex_id"
                candidate = {identity: "one", "title": "Paper"}
                candidates = root / "candidates.csv"
                atomic_write_csv(candidates, [candidate], list(candidate))
                response = schema_answer(schema)
                response["decision"] = "accept"
                with patch("radar.common.ollama.requests.post", return_value=completion(response)) as post:
                    rows = classify_plain_llm(settings, source, candidates, root / "plain.csv", "fixture:1", "d1", [(candidate, payload)])
                self.assertEqual(post.call_count, 1)
                self.assertEqual((rows[0]["quality_status"], rows[0]["quality_qualified_decision"]), ("not_configured", "accept"))
                if source == "cohort":
                    review_input = root / "candidates.json"
                    review_input.write_text(json.dumps([payload]))
                    response = answer()
                    response["criteria"] = {key: value for key, value in response["criteria"].items() if key in profile.eligibility}
                    model = FakeModel(response)
                    result = classify_study(path, review_input, root / "radar.csv", {"model": "fixture:1"}, client=model)[0]
                    self.assertEqual(len(model.calls), 1)
                    self.assertEqual((result["decision"], result["quality_status"], result["quality_qualified_decision"]),
                        ("accept", "not_configured", "accept"))
                for invalid in ("1", "-1", "true", "0.5"):
                    path.write_text(prefix + "[quality]\n\n[decision]" + f"\nquality_minimum = {invalid}\n" + decision)
                    with self.assertRaisesRegex(ValueError, "quality_minimum"):
                        load_screening_profile(path)
                path.write_text(prefix + "[quality]\n\n[decision]" + '\nquality_required = ["Q1"]\n' + decision)
                with self.assertRaisesRegex(ValueError, "quality_required"):
                    load_screening_profile(path)

    def test_disabled_quality_skips_scoring_and_preserves_relevance_and_failures(self):
        """Disabled quality produces no fabricated quality error, quota or extra model calls."""
        rules = {"quality": {}, "decision": {"quality_minimum": 0, "quality_required": []}}
        for decision in ("accept", "accept_adjacent", "exclude", "uncertain", "error", "dry_run"):
            for backend in ("keyword", "radar_llm", "plain_llm"):
                row = finish_quality({"decision": decision}, rules, {}, backend, error="classification_failed")
                self.assertEqual((row["quality_status"], row["quality_error"], row["quality_qualified_decision"]),
                    ("not_configured", "", decision))
                self.assertEqual(json.loads(row["quality_assessments_json"]), {})
            row = {"decision": decision}
            backend = QualityNLIBackend(AssertionError("disabled quality must not score"))
            assess_nli_quality(row, SECTIONS, rules, backend, None, "nli")
            self.assertFalse(backend.calls)
            self.assertEqual((row["quality_status"], row["quality_qualified_decision"]), ("not_configured", decision))
            assess_llm_quality(row, None, SECTIONS, rules, grounded=True)
            quality_consensus(row, {"a": row.copy(), "b": row.copy()}, .8)
            self.assertEqual((row["quality_status"], row["quality_qualified_decision"]), ("not_configured", decision))

    def test_thresholds_preserve_relevance_and_distinguish_missing_from_failure(self):
        """Unobserved content is unknown; explicit mandatory failure affects only the joint view."""
        row = {"decision": "accept"}
        finish_quality(row, RULES, {"Q1": {"value": "yes"}}, "fixture")
        self.assertEqual((row["decision"], row["relevance_decision"], row["quality_status"], row["quality_qualified_decision"]),
            ("accept", "accept", "unknown", "uncertain"))
        finish_quality(row, RULES, {"Q1": {"value": "no"}, "Q2": {"value": "yes"}}, "fixture")
        self.assertEqual((row["decision"], row["quality_status"], row["quality_qualified_decision"]), ("accept", "fail", "exclude"))
        finish_quality(row, RULES, {}, "keyword", supported=[])
        self.assertEqual(row["quality_status"], "not_assessed")
        self.assertEqual(row["quality_no"], 0)
        self.assertEqual(row["quality_not_assessed"], 2)
        finish_quality(row, RULES, {}, "fixture", error="bad_quality_json")
        self.assertEqual((row["decision"], row["quality_status"]), ("accept", "error"))

    def test_llm_grounding_and_plain_claims_are_explicit_and_never_corrected(self):
        """RADAR verifies source presence; plain inference retains the same unverified claim."""
        raw = self._judgments()
        raw["Q2"]["evidence"][0]["quote"] = "An invented execution example."
        grounded, plain = {"decision": "accept"}, {"decision": "accept"}
        with patch("radar.common.ollama.OllamaClient", side_effect=AssertionError("extra LLM")):
            assess_llm_quality(grounded, raw, SECTIONS, RULES, grounded=True)
            assess_llm_quality(plain, raw, SECTIONS, RULES, grounded=False, backend="plain_llm")
        self.assertEqual(grounded["quality_status"], "unknown")
        self.assertEqual(plain["quality_status"], "pass")
        self.assertEqual(grounded["decision"], plain["decision"])
        self.assertFalse(json.loads(plain["quality_assessments_json"])["Q2"]["evidence_verified"])
        bad = self._judgments()
        bad["Q1"]["value"] = ["yes"]
        assess_llm_quality(plain, bad, SECTIONS, RULES, grounded=False)
        self.assertEqual((plain["decision"], plain["quality_status"]), ("accept", "error"))
        schema = quality_schema({"properties": {}, "required": []}, RULES)
        self.assertEqual(set(schema["properties"]["quality"]["properties"]), {"Q1", "Q2"})
        self.assertIn("Quality must not change", quality_prompt("class instructions", RULES))

    def test_nli_uses_its_own_scores_and_neutrality_is_not_a_negative(self):
        """Quality has separate cutoffs and never invokes Ollama to complete an NLI judgment."""
        config = NLIRuntimeConfig("fixture", "d1", "cpu", 8, 512, 384, 48, 6, 2, .95, .1, .5, .1)
        backend = QualityNLIBackend([NLIScore(.8, .1, .1), NLIScore(.01, .98, .01)])
        row = {"decision": "accept"}
        with patch("radar.common.ollama.OllamaClient", side_effect=AssertionError("hidden LLM")):
            assess_nli_quality(row, SECTIONS, RULES, backend, config, "nli")
        self.assertEqual(row["quality_status"], "unknown")
        self.assertEqual(json.loads(row["quality_assessments_json"])["Q2"]["value"], "unknown")
        self.assertEqual([pair[1] for pair in backend.calls[0]], list(RULES["quality"].values()))
        backend.scores = [NLIScore(.8, .1, .1), NLIScore(.01, .04, .95)]
        assess_nli_quality(row, SECTIONS, RULES, backend, config, "nli")
        self.assertEqual((row["decision"], row["quality_status"]), ("accept", "fail"))
        failing = QualityNLIBackend(RuntimeError("quality scoring failed"))
        assess_nli_quality(row, SECTIONS, RULES, failing, config, "nli")
        self.assertEqual((row["decision"], row["quality_status"]), ("accept", "error"))
        with patch.object(failing, "chunk_sections", side_effect=ValueError("bad chunks")):
            assess_nli_quality(row, SECTIONS, RULES, failing, config, "nli")
        self.assertEqual((row["decision"], row["quality_status"]), ("accept", "error"))
        absent = QualityNLIBackend(AssertionError("metadata-only quality inference"))
        assess_nli_quality(row, [EvidenceSection("OpenAlex metadata", "A title")], RULES, absent, config, "nli")
        self.assertEqual(row["quality_status"], "unknown")
        self.assertFalse(absent.calls)

    def test_nli_conflicting_passages_remain_unknown(self):
        """A contradictory paragraph cannot be hidden by selecting the most positive passage."""
        backend = QualityNLIBackend([NLIScore(.95, .04, .01), NLIScore(.01, .04, .95)] * 2)
        config = NLIRuntimeConfig("fixture", "d1", "cpu", 8, 512, 384, 48, 6, 2, .5, .1, .5, .1)
        row = {"decision": "accept"}
        assess_nli_quality(row, [*SECTIONS, EvidenceSection("Documentation", "A different passage")], RULES, backend, config, "nli")
        self.assertEqual(row["quality_status"], "unknown")

    def test_quality_consensus_counts_missing_voters_and_requires_matched_evidence(self):
        """Four-model quality voting cannot inherit a passing representative's assessment."""
        positive = finish_quality({"decision": "accept"}, RULES, self._judgments(), "radar_llm", evidence_hash="same")
        negative = finish_quality({"decision": "accept"}, RULES, self._judgments("no"), "radar_llm", evidence_hash="same")
        row = {"decision": "accept"}
        quality_consensus(row, {"a": positive, "b": positive, "c": negative, "d": {}}, .75)
        self.assertEqual((row["decision"], row["quality_status"]), ("accept", "unknown"))
        self.assertEqual(json.loads(row["quality_votes_json"])["Q1"]["denominator"], 4)
        quality_consensus(row, {"a": positive, "b": positive, "c": positive, "d": {}}, .75)
        self.assertEqual(row["quality_status"], "pass")
        changed = {**positive, "quality_evidence_hash": "different"}
        quality_consensus(row, {"a": positive, "b": changed}, .5)
        self.assertEqual((row["decision"], row["quality_status"]), ("accept", "error"))

    def test_plain_llm_retains_valid_classification_when_quality_is_malformed(self):
        """Joint prompting stays one request and a quality format error stays out of relevance."""
        for study, source in (("ids_repositories", "github"), ("conformal_treatment_effects", "cohort")):
            with self.subTest(study=study), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = self._settings(root, study)
                payload = {"repository": {"full_name": "fixture/detector"}, "readme_text": SECTIONS[0].text} if source == "github" else {"paper_id": "one", "title": "A paper", "full_text": TEXT}
                identity = "github_repo_id" if source == "github" else "paper_id"
                candidate = {identity: "one", "evidence_hash": "e"}
                path = root / "candidates.csv"
                atomic_write_csv(path, [candidate], list(candidate))
                _, schema = plain_llm_task(settings, source, payload)
                response = schema_answer(schema)
                response["decision"] = "accept"
                if source == "github":
                    response.pop("quality")
                else:
                    for key in settings.section("screening")["quality"]:
                        response["criteria"].pop(key)
                with patch("radar.common.ollama.requests.post", return_value=completion(response)) as post:
                    rows = classify_plain_llm(settings, source, path, root / "plain.csv", "fixture:1", "d1", [(candidate, payload)])
                self.assertEqual(post.call_count, 1)
                self.assertEqual((rows[0]["decision"], rows[0]["relevance_decision"], rows[0]["quality_status"]), ("accept", "accept", "error"))

    def test_radar_votes_quality_in_the_same_calls_without_changing_role(self):
        """Consecutive class/quality responses use separate votes and no quality-only correction calls."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = self._settings(root)
            prompt = load_prompt_profile(settings, "github")
            prompt["screening"] = {**prompt["screening"], **RULES}
            taxonomy = load_system_type_profile(settings, "github")
            response = FakeGitHubModel().generate("", {})
            response["evidence"] = [{"source": "README", "quote": "This artifact detects intrusions."}]
            response["quality"] = self._judgments()
            other = copy.deepcopy(response)
            other["quality"]["Q2"] = {"value": "unknown", "evidence": []}
            client = SimpleNamespace(model="fixture:1", generate=Mock(side_effect=[response, other]))
            candidate = {"github_repo_id": "1", "full_name": "fixture/detector", "prefilter_band": "candidate", "prefilter_score": "9", "evidence_hash": "e"}
            path = root / "candidates.csv"
            atomic_write_csv(path, [candidate], list(candidate))
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("github.repository", "1", {"repository": {"full_name": "fixture/detector"}, "readme_text": SECTIONS[0].text}, "now")
                rows = classify_github(path, root / "classified.csv", cache, {"model": "fixture:1"}, generator=client,
                    prompt_config=prompt, system_type_profile=taxonomy, consecutive_runs=2, consensus_threshold=.75)
            self.assertEqual(client.generate.call_count, 2)
            self.assertEqual((rows[0]["decision"], rows[0]["quality_status"]), ("accept", "unknown"))
            self.assertIn("quality", client.generate.call_args.args[1]["properties"])

    def test_review_quality_failure_does_not_change_relevance_evaluation_or_database_audit(self):
        """An eligible paper can fail quality while retaining an independently correct relevance prediction."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = load_study_protocol(ROOT / "config/studies/conformal_treatment_effects/screening.toml")
            response = answer()
            for key in protocol.quality:
                response["criteria"][key] = {"value": "no", "evidence_ids": response["criteria"]["F1"]["evidence_ids"]}
            row = StudyScreeningAdapter(protocol).normalize([{"paper_id": "one", "title": "Paper", "full_text": TEXT, "criteria": response["criteria"]}])[0]
            self.assertEqual((row["decision"], row["relevance_decision"], row["quality_status"], row["quality_qualified_decision"]), ("exclude", "accept", "fail", "exclude"))
            row["classifier_supported_fields"] = "*"
            source = root / "classified.csv"
            atomic_write_csv(source, [row], list(row))
            references = root / "references.toml"
            references.write_text(f'schema_version="1.0"\nsource="{protocol.contract.name}"\nevaluation_unit="source_record"\nclassification_fields=[]\n[[entities]]\nid="one"\ndecision="accept"\n')
            profile = load_rating_profile(references, protocol.contract.name, contract=protocol.contract)
            relevance = evaluate_results(profile, source, root / "relevance", decision_column="relevance_decision")
            qualified = evaluate_results(profile, source, root / "qualified", decision_column="quality_qualified_decision")
            self.assertEqual(relevance.metrics["operational"]["tp"], 1)
            self.assertEqual(qualified.metrics["operational"]["fn"], 1)
            self.assertEqual(relevance.metrics["decision_column"], "relevance_decision")
            self.assertTrue(verify_directory(root / "relevance")["ok"])
            for view in ("relevance", "qualified"):
                audit = verify_database(root / view / "reporting.sqlite3")
                self.assertTrue(audit["ok"], audit)
            evaluated = read_csv(Path(relevance.metrics["classifications_path"]))[0]
            self.assertEqual((evaluated["decision"], evaluated["pipeline_decision"]), ("accept", "exclude"))
            self.assertEqual(read_csv(source)[0]["decision"], "exclude")

    def test_review_quality_errors_and_unsupported_ids_remain_independent_in_ensembles(self):
        """Malformed quality is an error; unsupported source IDs are unknown, with relevance intact."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = ROOT / "config/studies/conformal_treatment_effects/screening.toml"
            source = root / "candidates.json"
            source.write_text(json.dumps([{"paper_id": "one", "title": "Paper", "full_text": TEXT}]))
            for malformed in (False, True):
                response = answer()
                for key in load_study_protocol(protocol).quality:
                    response["criteria"][key] = None if malformed else {"value": "yes", "evidence_ids": ["B000000000000-00001"]}
                inputs = {}
                for model in ("a", "b"):
                    inputs[model] = root / f"{malformed}-{model}.csv"
                    client = FakeModel(response)
                    rows = classify_study(protocol, source, inputs[model], {"model": model}, client=client)
                    self.assertEqual(len(client.calls), 1)
                    self.assertEqual((rows[0]["relevance_decision"], rows[0]["quality_status"]),
                        ("accept", "error" if malformed else "unknown"))
                combined = ensemble_study(protocol, inputs, root / f"{malformed}-ensemble.csv", 1)[0]
                self.assertEqual((combined["relevance_decision"], combined["quality_status"]),
                    ("accept", "error" if malformed else "unknown"))

    def test_quality_reports_have_no_accuracy_claim_and_do_not_overwrite_predictions(self):
        """Keyword N/A, unknowns and errors remain distinct from quality rejection in offline reports."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, before = {}, {}
            for label, backend in (("RADAR", "radar_llm"), ("Keyword", "keyword")):
                row = {"github_repo_id": "1", "decision": "accept"}
                if backend == "keyword":
                    finish_quality(row, RULES, {}, backend, supported=[])
                else:
                    finish_quality(row, RULES, self._judgments("no"), backend)
                path = root / f"{label}.csv"
                atomic_write_csv(path, [row], list(row))
                paths[label], before[label] = path, file_hash(path)
            outputs = build_quality_reports(paths, root / "quality", {"formats": ["svg"], "dpi": 60})
            self.assertTrue(outputs)
            summaries = {row["classifier"]: row for row in read_csv(root / "quality/summary.csv")}
            self.assertEqual(summaries["Keyword"]["not_assessed"], "1")
            self.assertEqual(summaries["Keyword"]["quality_accuracy"], "")
            self.assertEqual(summaries["RADAR"]["fail"], "1")
            self.assertEqual(read_csv(root / "quality/radar/quality_qualified.csv")[0]["decision"], "exclude")
            verification = verify_directory(root / "quality")
            self.assertTrue(verification["ok"], verification)
            for label, path in paths.items():
                self.assertEqual(file_hash(path), before[label])
