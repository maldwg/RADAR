from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.github.schema import CATALOGUE_COLUMNS as GITHUB_COLUMNS
from ids_meta.openalex.schema import CATALOGUE_COLUMNS as OPENALEX_COLUMNS
from ids_meta.plotting.github import build as build_github
from ids_meta.plotting.openalex import build as build_openalex
from ids_meta.plotting.compare import build as build_compare
from ids_meta.plotting.compare import build_models
from ids_meta.architecture import build as build_architecture


class PlottingTests(unittest.TestCase):
    def test_offline_plot_smoke_and_plot_data(self) -> None:
        """Verify both plot bundles render offline with matching data tables."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_csv = root / "github.csv"
            github_row = {column: "" for column in GITHUB_COLUMNS}
            github_row.update({"schema_version": "1.0", "run_id": "run", "github_repo_id": "1", "project_id": "p1", "canonical_repo_id": "1", "repository_relationship": "canonical", "full_name": "example/repo", "role": "standalone_detector", "observation_scopes": "network", "capabilities": "detection", "detection_methods": "signature_based", "primary_system_type": "NIDS", "system_types": "NIDS", "system_type_names": "Network intrusion detection system", "language": "Python", "stars": "5", "forks": "1", "pushed_at": "2025-01-01T00:00:00Z", "html_url": "https://example.invalid", "evidence_json": "[]", "prefilter_band": "candidate", "prefilter_score": "10", "decision": "accept"})
            atomic_write_csv(github_csv, [github_row], GITHUB_COLUMNS)
            outputs = build_github(github_csv, root / "github-plots", {"formats": ["svg"], "dpi": 72, "top_n": 5})
            self.assertTrue(outputs)
            self.assertTrue((root / "github-plots/data/01_repository_timeline.csv").is_file())
            overview = root / "github-plots/data/00_repository_overview.csv"
            self.assertTrue(overview.is_file())
            self.assertIn("last_commit_at", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("primary_system_type", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "github-plots/data/00_system_type_definitions.csv").is_file())
            self.assertTrue((root / "github-plots/data/02_role_type_matrix.csv").is_file())
            self.assertTrue((root / "github-plots/00_repository_overview.svg").is_file())
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
            paper.update({"schema_version": "1.0", "run_id": "run", "openalex_id": "W1", "artifact_id": "a1", "artifact_name": "Fixture trace corpus", "artifact_display_name": "FTC", "publication_family_id": "f1", "canonical_openalex_id": "W1", "paper_relationship": "canonical_base", "artifact_type": "dataset", "publication_year": "2024", "domain": "general_network", "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled", "cited_by_count": "42", "citation_as_of": "2026-08-07", "discovery_relations": "direct_search;survey_reference:W9", "prefilter_score": "11", "prefilter_band": "candidate", "decision": "accept", "paper_role": "original_release"})
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
            self.assertIn("resource_name", timeline.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("survey_snowball", provenance.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "openalex-plots/00_publication_timeline.svg").is_file())
            self.assertTrue((root / "openalex-plots/01_citation_landscape.svg").is_file())
            self.assertTrue((root / "openalex-plots/07_classification_audit.svg").is_file())
            self.assertTrue((root / "openalex-plots/08_survey_sources.svg").is_file())
            self.assertFalse(stale.exists())
            architecture = build_architecture(root, ["svg"], 72)
            self.assertTrue(architecture)
            self.assertTrue((root / "pipeline_architecture.svg").is_file())
            self.assertFalse((root / "github-plots/pipeline_sequence.svg").exists())
            self.assertFalse((root / "openalex-plots/pipeline_sequence.svg").exists())


if __name__ == "__main__":
    unittest.main()
