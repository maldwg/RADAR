from __future__ import annotations

import tempfile
import unittest
import os
from pathlib import Path

import pandas as pd

from ids_meta.cli import build_parser
from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import Settings
from tests.fixtures import write_csv as atomic_write_csv
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
            'schema_version = "1.0"\nqueries = [{name="direct", search="direct"}]\n'
            'survey_queries = ["survey"]\n[survey_publishers]\n',
            encoding="utf-8",
        )
        return Settings(root, {
            "paths": {
                "interim_dir": "interim", "processed_dir": "processed", "plots_dir": "plots",
                "github_cache": "raw/github.sqlite3", "openalex_cache": "raw/openalex.sqlite3",
                "github_terms": "github_terms.toml", "openalex_terms": "openalex_terms.toml",
             'evaluation_dir': str(root / 'evaluation')},
            "github": {"query_limit_factor": 3, "collection_limit": 10},
            "openalex": {
                "query_limit_factor": 3, "collection_limit": 80,
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
            context = pd.read_csv(root / "plots/github/pipeline_funnel/data/discovery_context.csv")
        self.assertTrue(any(path.name == "00_pipeline_funnel.svg" for path in outputs))
        self.assertTrue(any(path.name == "00_pipeline_funnel.pdf" for path in outputs))
        self.assertEqual(int(table.iloc[0]["count"]), 60)
        self.assertEqual(int(table.iloc[1]["count"]), 10)
        self.assertTrue(table.iloc[2:]["count"].isna().all())
        self.assertEqual(list(table["stage_label"]), [
            "Crawler capacity",
            "Candidate selection", "Enrichment / prefilter", "Classification", "Inclusion", "Export",
        ])
        self.assertEqual(int(context.set_index("context_key").loc["initial_results", "count"]), 60)
        self.assertEqual(set(table["basis"]), {"configured_capacity", "not_observed"})
        self.assertTrue(table.iloc[1:]["removed_from_previous"].isna().all())

    def test_unlimited_funnels_do_not_invent_numeric_capacities(self) -> None:
        """Render both source funnels without a hidden default or a zero capacity."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            for source in ("github", "openalex"):
                settings.section(source)["collection_limit"] = "all"
                settings.section(source)["classification_limit"] = "all"
                build(settings, source)
                table = pd.read_csv(root / f"plots/{source}/pipeline_funnel/data/00_pipeline_funnel.csv")
                self.assertTrue(table["count"].isna().all())
                self.assertIn("unlimited", table.iloc[1]["basis_detail"])

    def test_funnel_prefers_saved_unlimited_cap_to_current_config(self) -> None:
        """Keep observed run bounds intact after configuration changes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            for source, ids_key in (("github", "repository_ids"), ("openalex", "work_ids")):
                with PipelineCache(settings.path(f"{source}_cache")) as cache:
                    cache.put_record(f"{source}.collection", "latest", {
                        ids_key: ["1"], "collection_limit": None, "sample_size_per_query": None,
                        "selected_at": "2026-01-01T00:00:00+00:00",
                    }, "now")
                build(settings, source)
                table = pd.read_csv(root / f"plots/{source}/pipeline_funnel/data/00_pipeline_funnel.csv")
                self.assertIn("unlimited", table.iloc[1]["basis_detail"])

    def test_observed_funnel_aligns_main_stages_and_preserves_discovery_context(self) -> None:
        """Keep deduplication outside common candidate-to-export loss accounting."""
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
                {"github_repo_id": "1", "decision": "accept", "classifier_id": "model-a", 'classifier_supported_fields': '*'},
            ], ["github_repo_id", "decision", "classifier_id"])
            catalogue = root / "processed/github/models/model-a/repositories.csv"
            atomic_write_csv(catalogue, [
                {"github_repo_id": "1", "decision": "accept", 'classifier_supported_fields': '*'},
            ], ["github_repo_id", "decision"])

            outputs = build(settings, "github")
            table_path = root / "plots/github/pipeline_funnel/data/01_pipeline_funnel_model-a.csv"
            table = pd.read_csv(table_path)
            context = pd.read_csv(root / "plots/github/pipeline_funnel/data/discovery_context.csv")
            svg = (root / "plots/github/pipeline_funnel/01_pipeline_funnel_model-a.svg").read_text()
            # An old export is unavailable for this run, not an observed zero.
            os.utime(catalogue, (1, 1))
            build(settings, "github")
            stale = pd.read_csv(table_path).set_index("stage_key")

        self.assertTrue(any(path.name == "01_pipeline_funnel_model-a.svg" for path in outputs))
        by_stage = table.set_index("stage_key")
        self.assertEqual(list(table["stage_label"]), [
            "Crawler hits (overlap)",
            "Candidate selection", "Enrichment / prefilter", "Classification", "Inclusion", "Export",
        ])
        self.assertEqual(list(table["count"]), [3, 2, 1, 1, 1, 1])
        self.assertTrue(pd.isna(by_stage.loc["candidate_selection", "removed_from_previous"]))
        self.assertTrue(pd.isna(by_stage.loc["candidate_selection", "removed_percent_of_previous"]))
        self.assertEqual(int(by_stage.loc["enrichment_prefilter", "count"]), 1)
        self.assertEqual(int(by_stage.loc["included", "count"]), 1)
        self.assertEqual(int(by_stage.loc["enrichment_prefilter", "removed_from_previous"]), 1)
        self.assertEqual(by_stage.loc["enrichment_prefilter", "removed_label"], "Rejected by deterministic prefilter")
        discovery = context.set_index("context_key")
        self.assertEqual(int(discovery.loc["initial_results", "count"]), 3)
        self.assertEqual(int(discovery.loc["deduplicated", "count"]), 2)
        self.assertEqual(int(discovery.loc["duplicate_matches", "count"]), 1)
        self.assertEqual(int(discovery.loc["evidence_cached", "count"]), 2)
        self.assertIn("Crawler hits (overlap)", svg)
        self.assertNotIn("Discovery inputs", svg)
        self.assertNotIn("Unique search matches", svg)
        self.assertNotIn("Deduplication", svg)
        self.assertNotIn("<!-- GitHub pipeline funnel — model-a -->", svg)
        self.assertEqual(set(table["basis"]), {"observed"})
        self.assertTrue(pd.isna(stale.loc["final_catalogue", "count"]))
        self.assertTrue(pd.isna(stale.loc["final_catalogue", "removed_from_previous"]))
        self.assertEqual(stale.loc["final_catalogue", "basis"], "not_observed")

    def test_openalex_discovery_hits_are_independent_of_candidate_and_classification_limits(self) -> None:
        """Keep overlapping discovery hits distinct from retained and classified papers."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            settings.section("openalex")["classification_limit"] = 1
            timestamp = "2026-01-01T00:00:00+00:00"
            with PipelineCache(root / "raw/openalex.sqlite3") as cache:
                cache.put_record("openalex.collection", "latest", {
                    "work_ids": ["paper-1", "paper-2"],
                    "survey_ids": ["paper-2", "survey-1", "survey-2"],
                    "selected_at": timestamp, "collection_limit": 2,
                    "snowball_reached_depth": 1,
                }, timestamp)
                for query, shard, count in [
                    ("direct", "direct_cursor", 100),
                    ("survey", "survey_cursor", 50),
                    ("snowball_depth:1", "backward_references", 25),
                    ("unrelated old query", "direct_cursor", 999),
                    ("snowball_depth:2", "backward_references", 999),
                ]:
                    cache.put_coverage("openalex", query, shard, "bounded", count, {}, timestamp)
            candidates = root / "interim/openalex/paper_candidates.csv"
            atomic_write_csv(candidates, [
                {"openalex_id": "paper-1", "prefilter_band": "candidate"},
                {"openalex_id": "paper-2", "prefilter_band": "borderline"},
            ], ["openalex_id", "prefilter_band"])
            atomic_write_csv(root / "interim/openalex/models/model-a/classifications.csv", [
                {"openalex_id": "paper-1", "decision": "accept", "classifier_id": "model-a", 'classifier_supported_fields': '*'},
            ], ["openalex_id", "decision", "classifier_id"])
            atomic_write_csv(root / "processed/openalex/models/model-a/base_papers.csv", [
                {"openalex_id": "paper-1"},
            ], ["openalex_id"])

            build(settings, "openalex")
            table = pd.read_csv(root / "plots/openalex/pipeline_funnel/data/01_pipeline_funnel_model-a.csv")
            readme = (root / "plots/openalex/pipeline_funnel/README.md").read_text(encoding="utf-8")
            context = pd.read_csv(root / "plots/openalex/pipeline_funnel/data/discovery_context.csv")
            svg = (root / "plots/openalex/pipeline_funnel/01_pipeline_funnel_model-a.svg").read_text()

        by_stage = table.set_index("stage_key")
        self.assertEqual(list(table["stage_label"]), [
            "Crawler hits (overlap)",
            "Candidate selection", "Enrichment / prefilter", "Classification", "Inclusion", "Export",
        ])
        self.assertEqual(list(table["count"]), [175, 2, 2, 1, 1, 1])
        self.assertTrue(pd.isna(by_stage.loc["candidate_selection", "removed_from_previous"]))
        self.assertTrue(pd.isna(by_stage.loc["candidate_selection", "removed_percent_of_previous"]))
        self.assertEqual(int(by_stage.loc["candidate_selection", "count"]), 2)
        self.assertEqual(int(by_stage.loc["enrichment_prefilter", "count"]), 2)
        self.assertEqual(int(by_stage.loc["classified", "count"]), 1)
        self.assertEqual(int(by_stage.loc["classified", "removed_from_previous"]), 1)
        discovery = context.set_index("context_key")
        self.assertEqual(int(discovery.loc["initial_results", "count"]), 175)
        self.assertTrue(discovery.loc["initial_results", "visible_in_figure"])
        self.assertIn("Crawler hits (overlap)", svg)
        self.assertIn("175", svg)
        self.assertEqual(int(discovery.loc["unique_retained_with_surveys", "count"]), 4)
        self.assertEqual(int(discovery.loc["auxiliary_only_surveys", "count"]), 2)
        self.assertEqual(int(discovery.loc["survey_sources", "count"]), 3)
        self.assertTrue(pd.isna(discovery.loc["unique_discovery_pool", "count"]))
        self.assertEqual(discovery.loc["unique_discovery_pool", "basis"], "not_recorded")
        self.assertNotIn("Auxiliary-only surveys", svg)
        self.assertNotIn("Not recorded", svg)
        self.assertNotIn("Discovery inputs", svg)
        self.assertNotIn("Retention + deduplication", svg)
        self.assertNotIn("removed_percent_of_previous", context.columns)
        self.assertIn("Active collection limit recorded by the snapshot: `2`", readme)


if __name__ == "__main__":
    unittest.main()
