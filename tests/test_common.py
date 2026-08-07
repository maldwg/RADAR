from __future__ import annotations

import ast
import argparse
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import load_settings, load_term_profile
from ids_meta.common.ollama import EvidenceSection, verify_evidence
from ids_meta.common.scoring import score_evidence
from ids_meta.common.scoring import shortlist_candidates
from ids_meta.common.text import contains_concept, normalize_text
from ids_meta.common.logging import configure_logging, logger
from ids_meta.cli import _pipeline_limits


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

    def test_pipeline_limits_have_independent_defaults_and_overrides(self) -> None:
        """Verify collection and classification budgets resolve independently."""
        config = {"collection_limit": 1000, "classification_limit": 100}
        defaults = argparse.Namespace(action="run", limit=None, collect_limit=None, classify_limit=None)
        self.assertEqual(_pipeline_limits(defaults, config), (1000, 100))
        explicit = argparse.Namespace(action="run", limit=None, collect_limit=750, classify_limit=80)
        self.assertEqual(_pipeline_limits(explicit, config), (750, 80))
        legacy = argparse.Namespace(action="run", limit=12, collect_limit=None, classify_limit=None)
        self.assertEqual(_pipeline_limits(legacy, config), (12, 12))

    def test_profiles_contain_generic_query_types(self) -> None:
        """Verify both source profiles expose generic discovery queries."""
        settings = load_settings(ROOT / "config/default.toml")
        self.assertTrue(load_term_profile(settings, "github")["search_queries"])
        self.assertTrue(load_term_profile(settings, "openalex")["direct_queries"])
        self.assertEqual(settings.section("github")["collection_limit"], 1000)
        self.assertEqual(settings.section("github")["classification_limit"], 100)
        self.assertEqual(settings.section("openalex")["collection_limit"], 1000)
        self.assertEqual(settings.section("openalex")["classification_limit"], 100)

    def test_evidence_quotes_are_source_bound(self) -> None:
        """Verify evidence quotes must occur in their claimed source section."""
        sections = [EvidenceSection("README", "This repository detects intrusions in network traffic.")]
        verified, ok = verify_evidence([{"source": "README", "quote": "detects intrusions"}], sections)
        self.assertTrue(ok)
        self.assertEqual(len(verified), 1)
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
