from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from ids_meta.cli import _execute_github, build_parser
from ids_meta.common.baselines import (
    KeywordBaselineConfig,
    classify_keyword_baseline,
    configured_baseline_methods,
    infer_keyword_fields,
    load_keyword_baseline_config,
    load_keyword_taxonomy_profile,
)
from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError, Settings
from ids_meta.common.csvio import atomic_write_csv, read_csv
from ids_meta.common.evaluation import evaluate_results, load_rating_profile
from ids_meta.github.schema import CANDIDATE_COLUMNS
from ids_meta.openalex.schema import CANDIDATE_COLUMNS as OPENALEX_CANDIDATE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


class BaselineTests(unittest.TestCase):
    """Exercise generic classifier metadata and the deterministic keyword baseline."""

    def _settings(self, root: Path) -> Settings:
        """Build a minimal baseline-only configuration rooted in a temporary directory."""
        terms = root / "github-terms.toml"
        terms.write_text(
            '[positive.strong]\nweight = 7\nterms = ["alpha detector"]\n\n'
            '[positive.weak]\nweight = 5\nterms = ["beta research"]\n',
            encoding="utf-8",
        )
        ratings = root / "github-ratings.toml"
        ratings.write_text(
            'schema_version = "1.0"\nsource = "github"\n'
            'classification_fields = ["role", "observation_scopes"]\n\n'
            '[[entities]]\nid = "1"\ndecision = "accept"\n'
            'role = "native_detector"\nobservation_scopes = ["network"]\n',
            encoding="utf-8",
        )
        values = {
            "paths": {
                "interim_dir": str(root / "interim"),
                "processed_dir": str(root / "processed"),
                "plots_dir": str(root / "plots"),
                "runs_dir": str(root / "runs"),
                "evaluation_dir": str(root / "evaluation"),
                "github_cache": str(root / "github.sqlite3"),
                "github_terms": str(terms),
                "github_keyword": str(ROOT / "config/baselines/github_keyword.toml"),
                "openalex_keyword": str(ROOT / "config/baselines/openalex_keyword.toml"),
                "github_system_types": str(ROOT / "config/taxonomy/github_system_types.toml"),
            },
            "logging": {"progress_every": 1},
            "evaluation": {
                "enabled": True,
                "reference_sets": [
                    {"name": "fixture", "source": "github", "profile": str(ratings)},
                ],
            },
            "baselines": {
                "enabled": True,
                "methods": ["keyword"],
                "keyword": {"github": {"accept_threshold": 7, "uncertain_threshold": 4}},
            },
            "plots": {"formats": ["svg"], "dpi": 72},
        }
        return Settings(root, values)

    def test_baseline_configuration_validates_methods_and_keyword_thresholds(self) -> None:
        """Validate implemented method selection and independent keyword thresholds."""
        with tempfile.TemporaryDirectory() as directory:
            settings = self._settings(Path(directory))
            self.assertEqual(configured_baseline_methods(settings), ["keyword"])
            self.assertEqual(
                load_keyword_baseline_config(settings, "github"), KeywordBaselineConfig(7.0, 4.0),
            )
            self.assertEqual(configured_baseline_methods(settings, "nli"), ["nli"])
            keyword_profile = load_keyword_taxonomy_profile(settings, "github")
            self.assertEqual(
                [field.name for field in keyword_profile.fields],
                ["centralization", "role", "observation_scopes", "observation_objects", "capabilities", "detection_methods", "system_types"],
            )
            with self.assertRaisesRegex(ConfigError, "unknown baseline"):
                configured_baseline_methods(settings, "imaginary")
            invalid = dict(settings.values)
            invalid["baselines"] = {
                "keyword": {"github": {"accept_threshold": 4, "uncertain_threshold": 4}},
            }
            with self.assertRaisesRegex(ConfigError, "greater than"):
                load_keyword_baseline_config(Settings(settings.root, invalid), "github")

    def test_keyword_profile_distinguishes_malware_file_scanning_from_intrusion_detection(self) -> None:
        """Recognize antivirus input and method without fabricating an IDS type."""
        with tempfile.TemporaryDirectory() as directory:
            profile = load_keyword_taxonomy_profile(self._settings(Path(directory)), "github")
            predictions, _, _ = infer_keyword_fields(
                "Antivirus malware scanner. Scan uploaded files using signature matching. Run the CLI.", profile,
            )
            self.assertEqual(predictions["role"], "antivirus")
            self.assertEqual(predictions["system_types"], ["AV"])
            self.assertIn("file_analysis", predictions["observation_objects"])
            self.assertIn("detection", predictions["capabilities"])
            self.assertIn("signature_based", predictions["detection_methods"])
            rules, _, _ = infer_keyword_fields("YARA rules and signature collection", profile)
            self.assertEqual(rules["role"], "rules_content")

    def test_keyword_baseline_scores_all_bands_and_invalidates_cache_by_profile(self) -> None:
        """Classify candidate, boundary, and rejected evidence with fingerprinted decisions."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidates.csv"
            output = root / "classifications.csv"
            rows = []
            for identifier, text, discovery_band in (
                ("1", "alpha detector", "candidate"),
                ("2", "beta research", "borderline"),
                ("3", "unrelated material", "reject"),
            ):
                row = {column: "" for column in CANDIDATE_COLUMNS}
                row.update({
                    "github_repo_id": identifier,
                    "full_name": f"fixture/repo-{identifier}",
                    "prefilter_band": discovery_band,
                    "evidence_hash": "stale-candidate-hash",
                })
                rows.append(row)
            atomic_write_csv(candidates, rows, CANDIDATE_COLUMNS)
            profile = {
                "positive": {
                    "strong": {"weight": 7, "terms": ["alpha detector"]},
                    "weak": {"weight": 5, "terms": ["beta research"]},
                },
            }
            with PipelineCache(root / "cache.sqlite3") as cache:
                for identifier, text, _band in (
                    ("1", "alpha detector", "candidate"),
                    ("2", "beta research", "borderline"),
                    ("3", "unrelated material", "reject"),
                ):
                    role_evidence = {
                        "1": " intrusion detection system network traffic attack detection",
                        "2": " research code network traffic intrusion detection",
                        "3": "",
                    }[identifier]
                    cache.put_record(
                        "github.repository", identifier,
                        {"repository": {"full_name": f"fixture/repo-{identifier}"},
                         "readme_text": text + role_evidence},
                        "now",
                    )
                taxonomy = load_keyword_taxonomy_profile(self._settings(root), "github")
                first = classify_keyword_baseline(
                    "github", candidates, output, cache, profile, KeywordBaselineConfig(7, 4), taxonomy,
                )
                resumed = classify_keyword_baseline(
                    "github", candidates, output, cache, profile, KeywordBaselineConfig(7, 4), taxonomy,
                )
                changed = classify_keyword_baseline(
                    "github", candidates, output, cache, profile, KeywordBaselineConfig(8, 4), taxonomy,
                )
            self.assertEqual([row["decision"] for row in first], ["accept", "uncertain", "exclude"])
            self.assertEqual(first[0]["classified_at"], resumed[0]["classified_at"])
            self.assertNotEqual(first[0]["classifier_profile_hash"], changed[0]["classifier_profile_hash"])
            self.assertEqual(changed[0]["decision"], "uncertain")
            self.assertEqual(first[0]["classifier_method"], "keyword")
            self.assertIn("role", first[0]["classifier_supported_fields"])
            self.assertEqual(first[0]["role"], "native_detector")
            self.assertEqual(first[0]["observation_scopes"], "network")
            self.assertTrue(first[0]["keyword_taxonomy_scores_json"])
            self.assertNotEqual(first[0]["evidence_hash"], "stale-candidate-hash")

    def test_keyword_evaluation_compares_supported_taxonomy(self) -> None:
        """Include deterministic keyword taxonomy fields in the common evaluation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            classifications = root / "keyword.csv"
            atomic_write_csv(classifications, [{
                "github_repo_id": "1", "decision": "accept",
                "classifier_method": "keyword", "classifier_id": "keyword-score",
                "classifier_profile_hash": "profile",
                "classifier_supported_fields": "decision;role;observation_scopes",
                "role": "native_detector", "observation_scopes": "network",
            }], [
                "github_repo_id", "decision", "classifier_method", "classifier_id",
                "classifier_profile_hash", "classifier_supported_fields", "role", "observation_scopes",
            ])
            profile_path = Path(settings.section("evaluation")["reference_sets"][0]["profile"])
            profile = load_rating_profile(profile_path, "github")
            result = evaluate_results(profile, classifications, root / "evaluation")
            self.assertEqual(result.metrics["decision_accuracy"], 1.0)
            self.assertEqual(result.metrics["operational"]["DR"], 1.0)
            self.assertIsNone(result.metrics["operational"]["ACC"])  # This fixture contains only positives.
            self.assertEqual(result.metrics["classification_field_accuracy"], 1.0)
            self.assertEqual(result.metrics["classification_field_comparisons"], 2)
            self.assertTrue(all(row["supported"] == "true" for row in result.field_metrics))

    def test_keyword_taxonomy_abstains_on_exclusive_ties(self) -> None:
        """Do not break equal-scoring category ties using declaration order."""
        with tempfile.TemporaryDirectory() as directory:
            profile = load_keyword_taxonomy_profile(self._settings(Path(directory)), "github")
            predictions, scores, signals = infer_keyword_fields(
                "extensible framework and security platform", profile,
            )
            self.assertEqual(predictions["role"], "uncertain")
            self.assertEqual(scores["role"]["detection_framework"], scores["role"]["detection_platform"])
            self.assertTrue(signals["role"]["detection_framework"])

    def test_keyword_baseline_audits_candidate_evidence_fallback(self) -> None:
        """Continue transparently from an OpenAlex audit when its raw cache is unavailable."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "papers.csv"
            output = root / "keyword.csv"
            row = {column: "" for column in OPENALEX_CANDIDATE_COLUMNS}
            row.update({
                "openalex_id": "https://openalex.org/W1", "title": "Released corpus",
                "abstract": "We release a dataset named alpha detector.", "evidence_hash": "old",
            })
            atomic_write_csv(candidates, [row], OPENALEX_CANDIDATE_COLUMNS)
            profile = {"positive": {"release": {"weight": 7, "terms": ["release"]}}}
            taxonomy = load_keyword_taxonomy_profile(self._settings(root), "openalex")
            with PipelineCache(root / "empty.sqlite3") as cache:
                results = classify_keyword_baseline(
                    "openalex", candidates, output, cache, profile, KeywordBaselineConfig(7, 4),
                    taxonomy,
                )
            self.assertEqual(results[0]["decision"], "accept")
            self.assertEqual(results[0]["classifier_evidence_source"], "candidate_csv")

    def test_baseline_cli_runs_keyword_classification_and_evaluation(self) -> None:
        """Run the baseline command without loading Ollama configuration or invoking an LLM."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            candidate_path = root / "interim/github/candidates.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update({
                "github_repo_id": "1", "full_name": "fixture/repo-1",
                "prefilter_band": "candidate", "evidence_hash": "candidate-hash",
            })
            atomic_write_csv(candidate_path, [row], CANDIDATE_COLUMNS)
            with PipelineCache(root / "github.sqlite3") as cache:
                cache.put_record(
                    "github.repository", "1",
                    {"repository": {"full_name": "fixture/repo-1"},
                     "readme_text": "alpha detector intrusion detection system"},
                    "now",
                )
            args = argparse.Namespace(
                action="baseline", input=None, output=None, methods="keyword", ratings=None,
                resume=True, dry_run=False, limit=None, collect_limit=None, classify_limit=None,
                classify_all=False,
            )
            counts = _execute_github(args, settings)
            output = root / "interim/github/baselines/keyword/classifications.csv"
            self.assertEqual(counts["baseline_keyword_accept"], 1)
            self.assertEqual(read_csv(output)[0]["decision"], "accept")
            self.assertTrue((root / "evaluation/github/classifiers/keyword/metrics.json").is_file())
            parsed = build_parser().parse_args(["github", "baseline", "--methods", "keyword"])
            self.assertEqual(parsed.methods, "keyword")
            offline = build_parser().parse_args([
                "openalex", "baseline", "--methods", "nli", "--evaluate-only",
            ])
            self.assertTrue(offline.evaluate_only)


if __name__ == "__main__":
    unittest.main()
