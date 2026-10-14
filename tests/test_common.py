from __future__ import annotations

import ast
import argparse
import tempfile
import unittest
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
from tests.fixtures import SYSTEM_TYPE_PROFILE
from unittest.mock import patch

from ids_meta.common.cache import PipelineCache
from ids_meta.architecture import build as build_architecture
from ids_meta.common.csvio import read_csv
from tests.fixtures import write_csv as atomic_write_csv
from ids_meta.common.ensemble import consensus_classification, ensemble_classifications
from ids_meta.common.config import Settings
from ids_meta.common.config import load_prompt_profile, load_settings, load_system_type_profile, load_term_profile
from ids_meta.common.ollama import EvidenceSection, OllamaClient, decision_profile, verify_evidence
from ids_meta.common.scoring import score_evidence
from ids_meta.common.scoring import shortlist_candidates
from ids_meta.common.text import contains_concept, normalize_text, stable_hash, unicode_safe
from ids_meta.common.logging import configure_logging, logger
from ids_meta.cli import _configured_models, _ensemble_config, _execution_paths, _pipeline_limits, build_parser
from ids_meta.github.pipeline import prompt as github_prompt
from ids_meta.github.schema import CLASSIFICATION_COLUMNS as GITHUB_CLASSIFICATION_COLUMNS
from ids_meta.openalex.pipeline import prompt as openalex_prompt
from ids_meta.openalex.schema import CLASSIFICATION_COLUMNS as OPENALEX_CLASSIFICATION_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


class CommonTests(unittest.TestCase):
    def test_every_function_and_method_has_a_docstring(self) -> None:
        """Prevent undocumented production and test callables from being introduced."""
        missing: list[str] = []
        for directory in (ROOT / "src" / "ids_meta", ROOT / "tests"):
            for path in sorted(directory.rglob("*.py")):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node) is None:
                        missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [], "undocumented callables:\n" + "\n".join(missing))

    def test_logging_is_timestamped_and_written_to_stdout(self) -> None:
        """Verify CLI logs contain timestamps and are emitted to standard output."""
        output = io.StringIO()
        with redirect_stdout(output):
            configure_logging("INFO")
            logger("fixture").info("progress 2/10")
        value = output.getvalue()
        configure_logging("WARNING")
        self.assertIn("INFO", value)
        self.assertIn("ids_meta.fixture: progress 2/10", value)

    def test_normalization_and_typo_tolerance(self) -> None:
        """Verify text normalization and conservative misspelling matching."""
        self.assertEqual(normalize_text("Host‑Based", "IDS"), "host based ids")
        self.assertTrue(contains_concept("an intruson detction system", "intrusion detection system"))
        self.assertFalse(contains_concept("a recipe collection", "intrusion detection system"))

    def test_invalid_unicode_is_safe_at_cache_csv_and_hash_boundaries(self) -> None:
        """Replace unpaired surrogates without changing storage or export schemas."""
        broken = "abstract before \ud800 evidence \udfff after"
        cleaned = "abstract before \ufffd evidence \ufffd after"
        self.assertEqual(unicode_safe({"evidence": [broken]}), {"evidence": [cleaned]})
        self.assertEqual(stable_hash(broken), stable_hash(cleaned))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with PipelineCache(root / "cache.sqlite3") as cache:
                cache.put_record("openalex.work", "W1", {"abstract": broken}, "now")
                cache.put_decision("openalex", "W1", "profile", "evidence", {"reason": broken}, "now")
                cache.put_coverage("openalex", "query", "all", "complete", 1, {"note": broken}, "now")
                self.assertEqual(cache.get_record("openalex.work", "W1")["abstract"], cleaned)
                self.assertEqual(cache.get_decision("openalex", "W1", "profile", "evidence")["reason"], cleaned)
                self.assertIn(cleaned, cache.coverage_rows("openalex")[0]["details_json"])
            csv_path = root / "classification.csv"
            atomic_write_csv(csv_path, [{"openalex_id": "W1", "abstract": broken}], ["openalex_id", "abstract"])
            self.assertEqual(read_csv(csv_path)[0]["abstract"], cleaned)

    def test_grouped_score_counts_each_group_once(self) -> None:
        """Verify repeated terms cannot multiply one concept group's score."""
        profile = {"positive": {"scope": {"weight": 3, "terms": ["intrusion", "attack"]}}}
        result = score_evidence("intrusion attack intrusion", profile, 3, 1)
        self.assertEqual(result.score, 3)
        self.assertEqual(result.band, "candidate")

    def test_candidate_shortlist_is_ranked_and_bounded(self) -> None:
        """Verify shortlist limits select the strongest rows deterministically."""
        rows = [
            {"id": "b", "prefilter_band": "candidate", "prefilter_score": "8"},
            {"id": "a", "prefilter_band": "candidate", "prefilter_score": "8"},
            {"id": "c", "prefilter_band": "candidate", "prefilter_score": "11"},
            {"id": "d", "prefilter_band": "borderline", "prefilter_score": "6"},
            {"id": "e", "prefilter_band": "reject", "prefilter_score": "99"},
        ]
        selected = shortlist_candidates(rows, "id", 3)
        self.assertEqual([row["id"] for row in selected], ["c", "a", "b"])

    def test_candidate_shortlist_uses_meaningful_secondary_score_for_ties(self) -> None:
        """Verify a source-specific score resolves otherwise equal prefilter rows."""
        rows = [
            {"id": "a", "prefilter_band": "candidate", "prefilter_score": "11", "retrieval_rank_score": "12.5"},
            {"id": "b", "prefilter_band": "candidate", "prefilter_score": "11", "retrieval_rank_score": "16.5"},
            {"id": "c", "prefilter_band": "candidate", "prefilter_score": "10", "retrieval_rank_score": "20"},
        ]
        selected = shortlist_candidates(rows, "id", None, "retrieval_rank_score")
        self.assertEqual([row["id"] for row in selected], ["b", "a", "c"])

    def test_pipeline_limits_have_independent_defaults_and_overrides(self) -> None:
        """Verify collection and classification budgets resolve independently."""
        config = {"collection_limit": 1000, "classification_limit": 100}
        defaults = argparse.Namespace(action="run", limit=None, collect_limit=None, classify_limit=None, classify_all=False)
        self.assertEqual(_pipeline_limits(defaults, config), (1000, 100))
        explicit = argparse.Namespace(action="run", limit=None, collect_limit=750, classify_limit=80, classify_all=False)
        self.assertEqual(_pipeline_limits(explicit, config), (750, 80))
        classify_all = argparse.Namespace(action="run", limit=None, collect_limit=250, classify_limit=None, classify_all=True)
        self.assertEqual(_pipeline_limits(classify_all, config), (250, None))
        configured_all = argparse.Namespace(action="run", limit=None, collect_limit=None, classify_limit=None, classify_all=False)
        self.assertEqual(_pipeline_limits(configured_all, {"collection_limit": 500, "classification_limit": "all"}), (500, None))

    def test_classify_all_cli_is_mutually_exclusive_with_a_numeric_cap(self) -> None:
        """Verify the CLI exposes an unbounded eligible mode without accepting two classification modes."""
        parser = build_parser()
        args = parser.parse_args(["github", "run", "--collect-limit", "250", "--classify-all"])
        self.assertTrue(args.classify_all)
        self.assertEqual(_pipeline_limits(args, {"collection_limit": 1000, "classification_limit": 100}), (250, None))
        with self.assertRaises(SystemExit):
            parser.parse_args(["openalex", "classify", "--classify-all", "--classify-limit", "25"])

    def test_architecture_emits_detailed_pipeline_and_system_views(self) -> None:
        """Keep the sequential method and deployed components as separate figures."""
        with tempfile.TemporaryDirectory() as directory:
            outputs = build_architecture(Path(directory))
            self.assertEqual(
                {path.name for path in outputs},
                {
                    "pipeline_architecture.svg", "pipeline_architecture.pdf",
                    "system_architecture.svg", "system_architecture.pdf",
                },
            )
            pipeline = (Path(directory) / "pipeline_architecture.svg").read_text(encoding="utf-8")
            system = (Path(directory) / "system_architecture.svg").read_text(encoding="utf-8")
        for label in (
            "Repository search", "Direct paper search", "Diverse survey seeds",
            "Backward snowballing", "Deterministic prefilter", "Structured prompts", "Keyword + NLI baselines",
            "Repository families", "Publication families", "Repository catalogue",
            "Dataset catalogue", "GitHub-specific", "OpenAlex-specific",
            "Shared processing", "Evaluation only", "Reporting container",
            "(i)", "(ii)", "(iii)", "(iv)", "(v)",
        ):
            self.assertIn(label, pipeline)
        for label in (
            "GitHub", "OpenAlex", "GitHub crawler", "OpenAlex crawler",
            "Evidence + LLM results", "Evidence enrichment", "Adjudication client",
            "Ollama server", "Qwen", "gpt-oss", "Gemma",
            "Repository families", "Publication families",
            "GitHub path", "OpenAlex path", "Shared state", "Shared processing",
        ):
            self.assertIn(label, system)
        self.assertNotIn("Keyword baseline", system)
        self.assertNotIn("(i)", system)
        self.assertNotIn("structured request", system)
        self.assertNotIn(">serves<", system)
        self.assertNotIn("IDS meta-study", pipeline)
        self.assertNotIn("IDS meta-study", system)
        self.assertNotIn(">Python<", system)
        self.assertNotIn(">Record families<", system)
        self.assertEqual(system.count(">Outputs<"), 1)
        for asset in ("sqlite.svg", "python.svg", "ollama.svg", "qwen.svg", "openai.svg", "gemma.png"):
            self.assertTrue((ROOT / "assets" / "architecture" / asset).is_file())

    def test_architecture_is_an_independent_cli_command(self) -> None:
        """Expose architecture generation outside both source plotting commands."""
        args = build_parser().parse_args(["architecture", "--output", "figures"])
        self.assertEqual(args.command, "architecture")
        self.assertEqual(args.output, Path("figures"))
        self.assertEqual(args.formats, "svg,pdf")

    def test_profiles_contain_generic_query_types(self) -> None:
        """Verify both source profiles expose generic discovery queries."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        datasets = load_settings(ROOT / "config/studies/ids_datasets/study.toml")
        self.assertTrue(load_term_profile(settings, "github")["search_queries"])
        self.assertTrue(load_term_profile(datasets, "openalex")["queries"])
        self.assertTrue(load_prompt_profile(settings, "github")["template"])
        self.assertTrue(load_prompt_profile(datasets, "openalex")["template"])
        self.assertIn("NIDS", load_system_type_profile(settings, "github")["types"])
        self.assertGreaterEqual(int(settings.section("github")["collection_limit"]), 1)
        self.assertEqual(settings.section("github")["classification_limit"], "all")
        self.assertGreaterEqual(int(datasets.section("openalex")["collection_limit"]), 1)
        self.assertEqual(datasets.section("openalex")["classification_limit"], "all")
        self.assertEqual(settings.section("plots")["formats"], ["svg", "pdf"])

    def test_configured_prompts_describe_taxonomies_and_include_evidence(self) -> None:
        """Verify both source prompts render configurable taxonomy descriptions and supplied evidence."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        github_config = load_prompt_profile(settings, "github")
        openalex_config = load_prompt_profile(load_settings(ROOT / "config/studies/ids_datasets/study.toml"), "openalex")
        github = github_prompt([EvidenceSection("README", "repository fixture evidence")], github_config, system_type_profile=SYSTEM_TYPE_PROFILE)
        openalex = openalex_prompt([EvidenceSection("Abstract", "paper fixture evidence")], openalex_config)
        self.assertIn("native_detector", github)
        self.assertIn("repository fixture evidence", github)
        self.assertIn("dataset_and_generator", openalex)
        self.assertIn("paper fixture evidence", openalex)

    def test_missing_prompt_configuration_fails_without_python_fallbacks(self) -> None:
        """Verify classifiers reject absent templates and taxonomy descriptions."""
        sections = [EvidenceSection("README", "fixture evidence")]
        with self.assertRaisesRegex(ValueError, "roles"):
            github_prompt(sections, None, system_type_profile=SYSTEM_TYPE_PROFILE)
        incomplete = dict(load_prompt_profile(load_settings(ROOT / "config/studies/ids_repositories/study.toml"), "github"))
        incomplete.pop("methods")
        with self.assertRaisesRegex(ValueError, "methods"):
            github_prompt(sections, incomplete, system_type_profile=SYSTEM_TYPE_PROFILE)
        with self.assertRaisesRegex(ValueError, "roles"):
            openalex_prompt([EvidenceSection("Abstract", "fixture evidence")], None)

    def test_prompt_configuration_changes_decision_fingerprint(self) -> None:
        """Verify prompt edits invalidate otherwise matching cached classifications."""
        model_config = {"temperature": 0, "seed": 42}
        first = decision_profile("model", "version", model_config, {"template": "first ${evidence}"})
        second = decision_profile("model", "version", model_config, {"template": "second ${evidence}"})
        self.assertNotEqual(first, second)

    def test_ollama_runtime_controls_do_not_invalidate_cached_decisions(self) -> None:
        """Send runtime controls without treating capacity or residency as semantics."""
        config = {
            "model": "fixture-model", "context_length": 262144,
            "keep_alive": "30m", "max_output_tokens": 12, "retries": 0,
        }
        with patch("ids_meta.common.ollama.requests.post") as post:
            post.return_value.json.return_value = {"response": "{}"}
            result = OllamaClient(config).generate("fixture prompt", {"type": "object"})
        self.assertEqual(result, {})
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["num_ctx"], 262144)
        self.assertEqual(payload["keep_alive"], "30m")
        first = decision_profile("fixture-model", "version", config)
        runtime_changed = decision_profile(
            "fixture-model", "version",
            {**config, "context_length": 32768, "keep_alive": "5m"},
        )
        semantic_changed = decision_profile(
            "fixture-model", "version", {**config, "temperature": 0.5},
        )
        self.assertEqual(first, runtime_changed)
        self.assertNotEqual(first, semantic_changed)

    def test_evidence_quotes_are_source_bound(self) -> None:
        """Verify evidence quotes must occur in their claimed source section."""
        sections = [EvidenceSection("README", "This repository detects intrusions in AT&amp;T network traffic.")]
        verified, ok = verify_evidence([{"source": "readme", "quote": "detects intrusions in AT&T"}], sections)
        self.assertTrue(ok)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["source"], "README")
        _, wrong = verify_evidence([{"source": "Abstract", "quote": "detects intrusions"}], sections)
        self.assertFalse(wrong)

    def test_evidence_quotes_allow_ordered_ellipsis_fragments(self) -> None:
        """Verify truncated model quotes match all fragments in source order."""
        sections = [EvidenceSection("Abstract", "First release statement. Intervening details. Final access statement.")]
        evidence = [{"source": "Abstract", "quote": "First release statement ... Final access statement"}]
        verified, ok = verify_evidence(evidence, sections)
        self.assertTrue(ok)
        self.assertEqual(verified, evidence)
        _, reversed_fragments = verify_evidence(
            [{"source": "Abstract", "quote": "Final access statement … First release statement"}], sections,
        )
        self.assertFalse(reversed_fragments)
        _, missing_fragment = verify_evidence(
            [{"source": "Abstract", "quote": "First release statement...unsupported claim...Final access statement"}], sections,
        )
        self.assertFalse(missing_fragment)

    def test_cache_decisions_are_fingerprint_specific(self) -> None:
        """Verify cached decisions are isolated by evidence and profile hashes."""
        with tempfile.TemporaryDirectory() as directory:
            with PipelineCache(Path(directory) / "cache.sqlite3") as cache:
                cache.put_decision("x", "1", "profile-a", "evidence-a", {"decision": "accept"}, "now")
                self.assertIsNotNone(cache.get_decision("x", "1", "profile-a", "evidence-a"))
                self.assertIsNone(cache.get_decision("x", "1", "profile-a", "evidence-b"))
                self.assertIsNone(cache.get_decision("x", "1", "another-profile", "evidence-a"))

    def test_multi_model_configuration_and_paths(self) -> None:
        """Verify model lists, ensemble subsets, and isolated filesystem paths."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "paths": {
                    "interim_dir": "interim", "processed_dir": "processed", "plots_dir": "plots",
                    "github_cache": "github.sqlite3", "openalex_cache": "openalex.sqlite3",
                 'evaluation_dir': str(root / 'evaluation')},
                "ollama": {"models": ["model/a", "model:b", "model/a"]},
                "ensemble": {"enabled": True, "models": ["model:b"], "threshold": .75},
            }
            settings = Settings(root, values)
            models = _configured_models(settings)
            self.assertEqual(models, ["model/a", "model:b"])
            self.assertEqual(_ensemble_config(settings, models), (True, ["model:b"], .75))
            paths = _execution_paths(settings, "github", models)
            self.assertEqual(paths["model/a"]["classifications"], root / "interim/github/models/model-a/classifications.csv")
            self.assertNotEqual(paths["model/a"]["classifications"], paths["model:b"]["classifications"])

    def test_openalex_consensus_rejects_conflicting_resource_names(self) -> None:
        """Do not silently select one resource identity when repeated runs disagree."""
        votes = {}
        for label, artifact_name in (("run-1", "CICIDS2017"), ("run-2", "CSE-CIC-IDS2018")):
            votes[label] = {
                "decision": "accept", "paper_role": "original_release", "relevance_tier": "core",
                "artifact_type": "dataset", "artifact_name": artifact_name,
                "artifact_display_name": artifact_name, "direct_release_evidence": "true",
                "evidence_quote_verified": "true",
            }
        result = consensus_classification("openalex", votes, .5, system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertEqual(result["decision"], "uncertain")
        self.assertIn("artifact_identity_consensus_conflict", result["consensus_validation_flags"])
        self.assertEqual(json.loads(result["consensus_field_audit"])["artifact_identity"]["status"], "conflict")

    def test_ensemble_votes_include_exclude_uncertain_and_missing(self) -> None:
        """Verify majority votes, ties, thresholds, and missing results are auditable."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decisions = {
                "alpha": {"1": "accept", "2": "accept"},
                "beta": {"1": "accept", "2": "exclude", "3": "exclude"},
                "gamma": {"1": "exclude", "2": "uncertain", "3": "exclude"},
            }
            inputs: dict[str, Path] = {}
            for model, model_decisions in decisions.items():
                path = root / f"{model}.csv"
                rows = []
                for identifier, decision in model_decisions.items():
                    row = {column: "" for column in GITHUB_CLASSIFICATION_COLUMNS}
                    row.update({"github_repo_id": identifier, "full_name": f"fixture/{identifier}", "decision": decision, "llm_model": model, 'classifier_supported_fields': '*'})
                    rows.append(row)
                atomic_write_csv(path, rows, GITHUB_CLASSIFICATION_COLUMNS)
                inputs[model] = path
            output = root / "ensemble.csv"
            rows = ensemble_classifications("github", inputs, output, .5, GITHUB_CLASSIFICATION_COLUMNS)
            by_id = {str(row["github_repo_id"]): row for row in rows}
            self.assertEqual(by_id["1"]["decision"], "accept")
            self.assertEqual(by_id["1"]["ensemble_agreement"], "0.667")
            strict = ensemble_classifications("github", inputs, root / "strict.csv", .7, GITHUB_CLASSIFICATION_COLUMNS)
            strict_by_id = {str(row["github_repo_id"]): row for row in strict}
            self.assertEqual(strict_by_id["1"]["decision"], "uncertain")
            self.assertEqual(strict_by_id["1"]["ensemble_threshold"], "0.700")
            self.assertEqual(by_id["2"]["decision"], "uncertain")
            self.assertEqual(by_id["2"]["ensemble_vote_status"], "below_threshold")
            self.assertEqual(by_id["3"]["decision"], "exclude")
            self.assertEqual(by_id["3"]["ensemble_unavailable_score"], "0.333")
            self.assertEqual(json.loads(str(by_id["3"]["ensemble_decision_counts"]))["missing"], 1)
            self.assertEqual(json.loads(str(by_id["3"]["ensemble_model_decisions"]))["alpha"], "missing")
            tied = ensemble_classifications("github", {key: inputs[key] for key in ("alpha", "gamma")}, root / "tie.csv", .5, GITHUB_CLASSIFICATION_COLUMNS)
            tied_by_id = {str(row["github_repo_id"]): row for row in tied}
            self.assertEqual(tied_by_id["1"]["decision"], "uncertain")
            self.assertEqual(tied_by_id["1"]["ensemble_vote_status"], "tie")
            openalex_inputs: dict[str, Path] = {}
            for model, decision, tier in (("core-model", "accept", "core"), ("adjacent-model", "accept_adjacent", "adjacent")):
                path = root / f"{model}-openalex.csv"
                row = {column: "" for column in OPENALEX_CLASSIFICATION_COLUMNS}
                row.update({"openalex_id": "W1", "title": "Fixture release", "decision": decision, "relevance_tier": tier, "llm_model": model, 'classifier_supported_fields': '*'})
                atomic_write_csv(path, [row], OPENALEX_CLASSIFICATION_COLUMNS)
                openalex_inputs[model] = path
            openalex_rows = ensemble_classifications("openalex", openalex_inputs, root / "openalex-ensemble.csv", .5, OPENALEX_CLASSIFICATION_COLUMNS)
            self.assertEqual(openalex_rows[0]["decision"], "accept_adjacent")
            self.assertEqual(openalex_rows[0]["relevance_tier"], "adjacent")
            self.assertEqual(openalex_rows[0]["ensemble_representative_model"], "adjacent-model")

            conflict_inputs: dict[str, Path] = {}
            for model, artifact_name in (("cic-model", "CICIDS2017"), ("cse-model", "CSE-CIC-IDS2018")):
                path = root / f"{model}.csv"
                row = {column: "" for column in OPENALEX_CLASSIFICATION_COLUMNS}
                row.update({
                    "openalex_id": "W-conflict", "title": "A new intrusion dataset",
                    "decision": "accept", "relevance_tier": "core", "paper_role": "original_release",
                    "artifact_name": artifact_name, "artifact_display_name": artifact_name, "llm_model": model,
                 'classifier_supported_fields': '*'})
                atomic_write_csv(path, [row], OPENALEX_CLASSIFICATION_COLUMNS)
                conflict_inputs[model] = path
            conflict = ensemble_classifications("openalex", conflict_inputs, root / "openalex-conflict.csv", .5, OPENALEX_CLASSIFICATION_COLUMNS)[0]
            self.assertEqual(conflict["decision"], "uncertain")
            self.assertEqual(conflict["ensemble_vote_status"], "artifact_identity_conflict")
            self.assertIn("cross_model_artifact_identity_conflict", conflict["consensus_validation_flags"])


if __name__ == "__main__":
    unittest.main()
