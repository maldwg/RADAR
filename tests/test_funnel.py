from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ids_meta.cli import build_parser
from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import Settings
from ids_meta.common.csvio import atomic_write_csv
from ids_meta.funnel import build


class FunnelTests(unittest.TestCase):
    def _settings(self, root: Path, formats: list[str] | None = None) -> Settings:
        """Create a compact source configuration suitable for offline funnel tests."""
        terms = root / "github_terms.toml"
        terms.write_text(
            'schema_version = "1.0"\nsearch_queries = ["query one", "query two"]\n',
            encoding="utf-8",
        )
        openalex_terms = root / "openalex_terms.toml"
        openalex_terms.write_text(
            'schema_version = "1.0"\ndirect_queries = ["direct"]\n'
            'survey_queries = ["survey"]\n[survey_publishers]\n',
            encoding="utf-8",
        )
        return Settings(root, {
            "paths": {
                "interim_dir": "interim", "processed_dir": "processed", "plots_dir": "plots",
                "github_cache": "raw/github.sqlite3", "openalex_cache": "raw/openalex.sqlite3",
                "github_terms": "github_terms.toml", "openalex_terms": "openalex_terms.toml",
            },
            "github": {"query_sample_size": 1000, "collection_limit": 10},
            "openalex": {
                "query_sample_size": 100, "survey_query_sample_size": 50,
                "reference_sample_size": 25, "collection_limit": 80,
            },
            "plots": {"formats": formats or ["svg"], "dpi": 72},
        })

    def test_funnel_cli_is_an_independent_source_command(self) -> None:
        """Expose database-aware funnel generation beside the architecture command."""
        args = build_parser().parse_args(["funnel", "github", "--output", "figures"])
        self.assertEqual(args.command, "funnel")
        self.assertEqual(args.source, "github")
        self.assertEqual(args.output, Path("figures"))

    def test_configuration_only_funnel_marks_capacities_as_unobserved(self) -> None:
        """Use TOML capacities without presenting them as observed sample counts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = build(self._settings(root, ["svg", "pdf"]), "github")
            table = pd.read_csv(root / "plots/github/pipeline_funnel/data/00_pipeline_funnel.csv")
        self.assertTrue(any(path.name == "00_pipeline_funnel.svg" for path in outputs))
        self.assertTrue(any(path.name == "00_pipeline_funnel.pdf" for path in outputs))
        self.assertEqual(list(table["count"]), [2000, 10])
        self.assertEqual(set(table["basis"]), {"configured_capacity"})

    def test_observed_funnel_tracks_deduplication_filtering_and_classifier_losses(self) -> None:
        """Build exact observed stages and transition losses from cache and CSV artifacts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            cache_path = root / "raw/github.sqlite3"
            with PipelineCache(cache_path) as cache:
                cache.put_record("github.search_candidate", "1", {
                    "search_strategy": "fixture", "matched_queries": ["query one", "query two"],
                }, "2026-01-01T00:00:00+00:00")
                cache.put_record("github.search_candidate", "2", {
                    "search_strategy": "fixture", "matched_queries": ["query one"],
                }, "2026-01-01T00:00:00+00:00")
                cache.put_record("github.collection", "latest", {
                    "repository_ids": ["1", "2"], "search_strategy": "fixture",
                    "selected_at": "2026-01-01T00:00:00+00:00", "collection_limit": 10,
                }, "2026-01-01T00:00:00+00:00")
                for identifier in ("1", "2"):
                    cache.put_record("github.repository", identifier, {
                        "repository": {"id": identifier, "full_name": f"fixture/{identifier}"},
                    }, "2026-01-01T00:00:00+00:00")
            candidates = root / "interim/github/candidates.csv"
            atomic_write_csv(candidates, [
                {"github_repo_id": "1", "prefilter_band": "candidate"},
                {"github_repo_id": "2", "prefilter_band": "reject"},
            ], ["github_repo_id", "prefilter_band"])
            classifications = root / "interim/github/models/model-a/classifications.csv"
            atomic_write_csv(classifications, [
                {"github_repo_id": "1", "decision": "accept", "classifier_id": "model-a"},
            ], ["github_repo_id", "decision", "classifier_id"])
            catalogue = root / "processed/github/models/model-a/repositories.csv"
            atomic_write_csv(catalogue, [
                {"github_repo_id": "1", "decision": "accept"},
            ], ["github_repo_id", "decision"])

            outputs = build(settings, "github")
            table_path = root / "plots/github/pipeline_funnel/data/01_pipeline_funnel_model-a.csv"
            table = pd.read_csv(table_path)

        self.assertTrue(any(path.name == "01_pipeline_funnel_model-a.svg" for path in outputs))
        by_stage = table.set_index("stage_key")
        self.assertEqual(int(by_stage.loc["initial_results", "count"]), 3)
        self.assertEqual(int(by_stage.loc["deduplicated", "count"]), 2)
        self.assertEqual(int(by_stage.loc["prefilter_eligible", "count"]), 1)
        self.assertEqual(int(by_stage.loc["included", "count"]), 1)
        self.assertEqual(int(by_stage.loc["deduplicated", "removed_from_previous"]), 1)
        self.assertEqual(by_stage.loc["prefilter_eligible", "removed_label"], "Rejected by deterministic prefilter")
        self.assertEqual(set(table["basis"]), {"observed"})


if __name__ == "__main__":
    unittest.main()
