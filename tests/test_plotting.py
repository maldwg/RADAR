from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.github.schema import CATALOGUE_COLUMNS as GITHUB_COLUMNS
from ids_meta.openalex.schema import CATALOGUE_COLUMNS as OPENALEX_COLUMNS
from ids_meta.plotting.github import OPERATIONAL_ROLES, _github_why_not_included, build as build_github
from ids_meta.plotting.openalex import _cumulative_dimension, _openalex_why_not_included, build as build_openalex, spread_label_coordinates
from ids_meta.plotting.compare import build as build_compare
from ids_meta.plotting.compare import build_models
from ids_meta.architecture import build as build_architecture


class PlottingTests(unittest.TestCase):
    def test_timeline_label_positions_are_collision_spread_and_bounded(self) -> None:
        """Keep dense timeline callouts separated without disconnecting their ordering."""
        positions = spread_label_coordinates([1.0, 1.0, 1.02, 1.5], 0.0, 2.0, .25)
        self.assertTrue(all(0.0 <= value <= 2.0 for value in positions))
        ordered = sorted(positions)
        self.assertTrue(all(right - left >= .249 for left, right in zip(ordered, ordered[1:])))

    def test_operational_reporting_roles_are_explicit(self) -> None:
        """Keep reporting limited to genuine operational approaches without hiding other roles."""
        self.assertEqual(OPERATIONAL_ROLES, {"native_detector", "classifier_engine", "detection_framework"})

    def test_why_not_included_labels_are_plain_and_auditable(self) -> None:
        """Use reader-facing reasons while retaining how each reason was assigned."""
        repository_category, repository_basis, repository_explanation = _github_why_not_included({
            "decision": "exclude", "primary_reason": "This is a tutorial for configuring an existing detector.",
        })
        self.assertEqual(repository_category, "Tutorial or lab only — no reusable implementation")
        self.assertEqual(repository_basis, "Model explanation contains: tutorial")
        self.assertIn("without a reusable IDS contribution", repository_explanation)
        paper_category, paper_basis, paper_explanation = _openalex_why_not_included({
            "decision": "exclude", "paper_role": "survey",
        })
        self.assertEqual(paper_category, "Survey only — no new dataset or generator")
        self.assertEqual(paper_basis, "Structured paper role: survey")
        self.assertIn("reviews existing resources", paper_explanation)
        uncertain_category, uncertain_basis, uncertain_explanation = _github_why_not_included({
            "decision": "uncertain", "classification_validation_flags": "framework_architecture_missing;verified_evidence_missing",
        })
        self.assertEqual(uncertain_category, "Could not verify a framework architecture")
        self.assertEqual(uncertain_basis, "Automatic validation check: framework_architecture_missing")
        self.assertIn("cohesive agent-manager", uncertain_explanation)

    def test_cumulative_timeline_adds_categories_over_time(self) -> None:
        """Cumulative type and domain series must sum to the catalogue total each year."""
        dated = pd.DataFrame([
            {"publication_year": 2020, "artifact_type": "dataset", "domain": "general_network"},
            {"publication_year": 2022, "artifact_type": "dataset_and_generator", "domain": "iot_iiot"},
        ])
        type_table, years, categories, cumulative = _cumulative_dimension(
            dated, "artifact_type", ["dataset", "dataset_and_generator"],
        )
        self.assertEqual(years, [2020, 2021, 2022])
        self.assertEqual(categories, ["dataset", "dataset_and_generator"])
        self.assertEqual(int(cumulative.loc[2021].sum()), 1)
        self.assertEqual(int(cumulative.loc[2022].sum()), 2)
        self.assertEqual(set(type_table["dimension"]), {"resource_type"})
        domain_table, _, _, domain_cumulative = _cumulative_dimension(
            dated, "domain", ["general_network", "iot_iiot"],
        )
        self.assertEqual(int(domain_cumulative.loc[2022].sum()), 2)
        self.assertEqual(set(domain_table["dimension"]), {"security_domain"})

    def test_offline_plot_smoke_and_plot_data(self) -> None:
        """Verify both plot bundles render offline with matching data tables."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_csv = root / "github.csv"
            github_row = {column: "" for column in GITHUB_COLUMNS}
            github_row.update({"schema_version": "1.0", "run_id": "run", "github_repo_id": "1", "project_id": "p1", "canonical_repo_id": "1", "repository_relationship": "canonical", "full_name": "example/repo", "role": "native_detector", "observation_scopes": "network", "observation_objects": "packet_analysis", "capabilities": "detection", "detection_methods": "signature_based", "primary_system_type": "NIDS", "system_types": "NIDS", "system_type_names": "Network intrusion detection system", "language": "Python", "stars": "5", "forks": "1", "pushed_at": "2025-01-01T00:00:00Z", "html_url": "https://example.invalid", "evidence_json": "[]", "prefilter_band": "candidate", "prefilter_score": "10", "decision": "accept", "evidence_expansion_required": "true", "evidence_expansion_attempted": "true", "evidence_expansion_changed": "true", "evidence_expansion_improved": "true", "evidence_expansion_resolved": "true"})
            analyzer_row = {**github_row, "github_repo_id": "2", "project_id": "p2", "canonical_repo_id": "2",
                            "full_name": "example/analyzer", "role": "traffic_analyzer", "capabilities": "monitoring",
                            "detection_methods": "unknown", "primary_system_type": "NSM", "system_types": "NSM",
                            "system_type_names": "Network security monitoring / traffic analysis system"}
            atomic_write_csv(github_csv, [github_row, analyzer_row], GITHUB_COLUMNS)
            outputs = build_github(github_csv, root / "github-plots", {"formats": ["svg", "pdf"], "dpi": 72, "top_n": 5}, classifications_path=github_csv)
            self.assertTrue(outputs)
            self.assertTrue((root / "github-plots/data/01_repository_timeline.csv").is_file())
            overview = root / "github-plots/data/00_repository_overview.csv"
            self.assertTrue(overview.is_file())
            self.assertIn("last_commit_at", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("primary_system_type", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "github-plots/data/00_system_type_definitions.csv").is_file())
            self.assertTrue((root / "github-plots/data/02_role_type_matrix.csv").is_file())
            role_matrix = (root / "github-plots/data/02_role_type_matrix.csv").read_text(encoding="utf-8")
            self.assertIn("traffic_analyzer", role_matrix)
            self.assertIn("NSM", role_matrix)
            self.assertTrue((root / "github-plots/00_repository_overview.svg").is_file())
            self.assertTrue((root / "github-plots/00_repository_overview.pdf").is_file())
            self.assertTrue((root / "github-plots/10_evidence_escalation.svg").is_file())
            self.assertIn("denominator_scope", (root / "github-plots/data/10_evidence_escalation.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "github-plots/11_keyword_score_vs_decision.svg").is_file())
            self.assertTrue((root / "github-plots/12_why_not_included.svg").is_file())
            self.assertTrue((root / "github-plots/data/12_why_not_included_records.csv").is_file())
            github_reason_header = (root / "github-plots/data/12_why_not_included_records.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("primary_reason", github_reason_header)
            self.assertIn("how_reason_was_assigned", github_reason_header)
            github_score_header = (root / "github-plots/data/11_keyword_score_vs_decision.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("score_median", github_score_header)
            self.assertIn("lower_whisker", github_score_header)
            self.assertIn("upper_whisker", github_score_header)
            self.assertIn("outlier_count", github_score_header)
            self.assertIn("llm_decision", (root / "github-plots/data/11_keyword_score_vs_decision_records.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse((root / "github-plots/00_repository_overview_p01.svg").exists())
            comparison = build_compare("github", github_csv, root / "comparison", {"formats": ["svg"], "dpi": 72})
            self.assertTrue(comparison)
            second_csv = root / "github-second.csv"
            second_row = dict(github_row)
            second_row["decision"] = "exclude"
            atomic_write_csv(second_csv, [second_row], GITHUB_COLUMNS)
            model_outputs = build_models("github", {"model-a": github_csv, "model-b": second_csv}, root / "model-comparison", {"formats": ["svg"], "dpi": 72, "top_n": 5})
            self.assertTrue(model_outputs)
            self.assertTrue((root / "model-comparison/data/02_pairwise_agreement.csv").is_file())
            self.assertTrue((root / "model-comparison/data/03_disagreement_records.csv").is_file())

            openalex_csv = root / "openalex.csv"
            paper = {column: "" for column in OPENALEX_COLUMNS}
            paper.update({"schema_version": "1.0", "run_id": "run", "openalex_id": "W1", "artifact_id": "a1", "artifact_name": "Fixture trace corpus", "artifact_display_name": "FTC", "publication_family_id": "f1", "canonical_openalex_id": "W1", "paper_relationship": "canonical_base", "artifact_type": "dataset", "publication_year": "2024", "domain": "general_network", "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled", "cited_by_count": "42", "citation_as_of": "2026-08-07", "discovery_relations": "direct_search;survey_reference:W9", "prefilter_score": "11", "prefilter_band": "candidate", "decision": "accept", "paper_role": "original_release", "evidence_expansion_required": "true", "evidence_expansion_attempted": "true", "evidence_expansion_changed": "true", "evidence_expansion_improved": "true", "evidence_expansion_resolved": "true"})
            atomic_write_csv(openalex_csv, [paper], OPENALEX_COLUMNS)
            stale = root / "openalex-plots/08_publication_timeline.svg"
            stale.parent.mkdir(parents=True)
            stale.write_text("stale", encoding="utf-8")
            outputs = build_openalex(openalex_csv, None, root / "openalex-plots", {"formats": ["svg"], "dpi": 72}, openalex_csv, openalex_csv)
            self.assertTrue(outputs)
            timeline = root / "openalex-plots/data/00_publication_timeline.csv"
            provenance = root / "openalex-plots/data/02_discovery_provenance.csv"
            self.assertTrue(timeline.is_file())
            self.assertTrue(provenance.is_file())
            provenance_header = provenance.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("citing_surveys", provenance_header)
            self.assertNotIn("survey_source", provenance_header)
            contribution = (root / "openalex-plots/data/10_discovery_source_contribution.csv").read_text(encoding="utf-8")
            self.assertNotIn("Survey source", contribution)
            self.assertNotIn("Other", contribution)
            citation_table = root / "openalex-plots/data/01_citation_landscape.csv"
            self.assertIn("citations", citation_table.read_text(encoding="utf-8").splitlines()[0])
            timeline_header = timeline.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("resource_name", timeline_header)
            self.assertIn("citation_coordinate", timeline_header)
            self.assertIn("timeline_x_offset", timeline_header)
            self.assertIn("label_coordinate", timeline_header)
            self.assertIn("survey_snowball", provenance.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "openalex-plots/00_publication_timeline.svg").is_file())
            timeline_svg = (root / "openalex-plots/00_publication_timeline.svg").read_text(encoding="utf-8")
            self.assertIn("Citations (log-scaled)", timeline_svg)
            self.assertIn("Dataset domain (color)", timeline_svg)
            self.assertIn("Dataset / generator (shape)", timeline_svg)
            self.assertTrue((root / "openalex-plots/01_citation_landscape.svg").is_file())
            self.assertTrue((root / "openalex-plots/03_cumulative_growth_by_type_and_domain.svg").is_file())
            cumulative_data = (root / "openalex-plots/data/03_cumulative_growth_by_type_and_domain.csv").read_text(encoding="utf-8")
            self.assertIn("cumulative_all_resources", cumulative_data.splitlines()[0])
            self.assertIn("resource_type", cumulative_data)
            self.assertIn("security_domain", cumulative_data)
            self.assertTrue((root / "openalex-plots/07_classification_audit.svg").is_file())
            self.assertTrue((root / "openalex-plots/08_survey_sources.svg").is_file())
            self.assertTrue((root / "openalex-plots/09_evidence_escalation.svg").is_file())
            self.assertIn("denominator_scope", (root / "openalex-plots/data/09_evidence_escalation.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "openalex-plots/11_keyword_score_vs_decision.svg").is_file())
            self.assertTrue((root / "openalex-plots/12_why_not_included.svg").is_file())
            self.assertTrue((root / "openalex-plots/data/12_why_not_included_records.csv").is_file())
            paper_reason_header = (root / "openalex-plots/data/12_why_not_included_records.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("primary_reason", paper_reason_header)
            self.assertIn("how_reason_was_assigned", paper_reason_header)
            paper_score_header = (root / "openalex-plots/data/11_keyword_score_vs_decision.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("score_median", paper_score_header)
            self.assertIn("lower_whisker", paper_score_header)
            self.assertIn("upper_whisker", paper_score_header)
            self.assertIn("outlier_count", paper_score_header)
            self.assertIn("llm_decision", (root / "openalex-plots/data/11_keyword_score_vs_decision_records.csv").read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(stale.exists())
            architecture = build_architecture(root, ["svg"], 72)
            self.assertTrue(architecture)
            self.assertTrue((root / "pipeline_architecture.svg").is_file())
            self.assertTrue((root / "system_architecture.svg").is_file())
            self.assertFalse((root / "github-plots/pipeline_sequence.svg").exists())
            self.assertFalse((root / "openalex-plots/pipeline_sequence.svg").exists())


if __name__ == "__main__":
    unittest.main()
