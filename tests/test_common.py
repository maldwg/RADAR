from __future__ import annotations

import ast
import argparse
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import load_prompt_profile, load_settings, load_system_type_profile, load_term_profile
from ids_meta.common.ollama import EvidenceSection, decision_profile, verify_evidence
from ids_meta.common.scoring import score_evidence
from ids_meta.common.scoring import shortlist_candidates
from ids_meta.common.text import contains_concept, normalize_text
from ids_meta.common.logging import configure_logging, logger
from ids_meta.cli import _pipeline_limits, build_parser
from ids_meta.github.pipeline import prompt as github_prompt
from ids_meta.openalex.pipeline import prompt as openalex_prompt


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
        legacy = argparse.Namespace(action="run", limit=12, collect_limit=None, classify_limit=None, classify_all=False)
        self.assertEqual(_pipeline_limits(legacy, config), (12, 12))
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

    def test_profiles_contain_generic_query_types(self) -> None:
        """Verify both source profiles expose generic discovery queries."""
        settings = load_settings(ROOT / "config/default.toml")
        self.assertTrue(load_term_profile(settings, "github")["search_queries"])
        self.assertTrue(load_term_profile(settings, "openalex")["direct_queries"])
        self.assertTrue(load_prompt_profile(settings, "github")["template"])
        self.assertTrue(load_prompt_profile(settings, "openalex")["template"])
        self.assertIn("NIDS", load_system_type_profile(settings, "github")["types"])
        self.assertEqual(settings.section("github")["collection_limit"], 1000)
        self.assertEqual(settings.section("github")["classification_limit"], 100)
        self.assertEqual(settings.section("openalex")["collection_limit"], 1000)
        self.assertEqual(settings.section("openalex")["classification_limit"], 100)

    def test_configured_prompts_describe_taxonomies_and_include_evidence(self) -> None:
        """Verify both source prompts render configurable taxonomy descriptions and supplied evidence."""
        settings = load_settings(ROOT / "config/default.toml")
        github_config = load_prompt_profile(settings, "github")
        openalex_config = load_prompt_profile(settings, "openalex")
        github = github_prompt([EvidenceSection("README", "repository fixture evidence")], github_config)
        openalex = openalex_prompt([EvidenceSection("Abstract", "paper fixture evidence")], openalex_config)
        self.assertIn("standalone_detector", github)
        self.assertIn("repository fixture evidence", github)
        self.assertIn("dataset_and_generator", openalex)
        self.assertIn("paper fixture evidence", openalex)

    def test_missing_prompt_configuration_fails_without_python_fallbacks(self) -> None:
        """Verify classifiers reject absent templates and taxonomy descriptions."""
        sections = [EvidenceSection("README", "fixture evidence")]
        with self.assertRaisesRegex(ValueError, "roles"):
            github_prompt(sections, None)
        incomplete = dict(load_prompt_profile(load_settings(ROOT / "config/default.toml"), "github"))
        incomplete.pop("methods")
        with self.assertRaisesRegex(ValueError, "methods"):
            github_prompt(sections, incomplete)
        with self.assertRaisesRegex(ValueError, "roles"):
            openalex_prompt([EvidenceSection("Abstract", "fixture evidence")], None)

    def test_prompt_configuration_changes_decision_fingerprint(self) -> None:
        """Verify prompt edits invalidate otherwise matching cached classifications."""
        model_config = {"temperature": 0, "seed": 42}
        first = decision_profile("model", "version", model_config, {"template": "first ${evidence}"})
        second = decision_profile("model", "version", model_config, {"template": "second ${evidence}"})
        self.assertNotEqual(first, second)

    def test_evidence_quotes_are_source_bound(self) -> None:
        """Verify evidence quotes must occur in their claimed source section."""
        sections = [EvidenceSection("README", "This repository detects intrusions in AT&amp;T network traffic.")]
        verified, ok = verify_evidence([{"source": "readme", "quote": "detects intrusions in AT&T"}], sections)
        self.assertTrue(ok)
        self.assertEqual(len(verified), 1)
        self.assertEqual(verified[0]["source"], "README")
        _, wrong = verify_evidence([{"source": "Abstract", "quote": "detects intrusions"}], sections)
        self.assertFalse(wrong)

    def test_cache_decisions_are_fingerprint_specific(self) -> None:
        """Verify cached decisions are isolated by evidence and profile hashes."""
        with tempfile.TemporaryDirectory() as directory:
            with PipelineCache(Path(directory) / "cache.sqlite3") as cache:
                cache.put_decision("x", "1", "profile-a", "evidence-a", {"decision": "accept"}, "now")
                self.assertIsNotNone(cache.get_decision("x", "1", "profile-a", "evidence-a"))
                self.assertIsNone(cache.get_decision("x", "1", "profile-a", "evidence-b"))


if __name__ == "__main__":
    unittest.main()
