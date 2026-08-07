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


class PlottingTests(unittest.TestCase):
    def test_offline_plot_smoke_and_plot_data(self) -> None:
        """Verify both plot bundles render offline with matching data tables."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            github_csv = root / "github.csv"
            github_row = {column: "" for column in GITHUB_COLUMNS}
            github_row.update({"schema_version": "1.0", "run_id": "run", "github_repo_id": "1", "project_id": "p1", "canonical_repo_id": "1", "repository_relationship": "canonical", "full_name": "example/repo", "role": "standalone_detector", "observation_scopes": "network", "capabilities": "detection", "detection_methods": "signature_based", "language": "Python", "stars": "5", "forks": "1", "pushed_at": "2025-01-01T00:00:00Z", "html_url": "https://example.invalid", "evidence_json": "[]", "prefilter_band": "candidate", "prefilter_score": "10", "decision": "accept"})
            atomic_write_csv(github_csv, [github_row], GITHUB_COLUMNS)
            outputs = build_github(github_csv, root / "github-plots", {"formats": ["svg"], "dpi": 72, "top_n": 5})
            self.assertTrue(outputs)
            self.assertTrue((root / "github-plots/data/01_role_counts.csv").is_file())
            overview = root / "github-plots/data/00_repository_overview.csv"
            self.assertTrue(overview.is_file())
            self.assertIn("last_commit_at", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "github-plots/00_repository_overview.svg").is_file())
            self.assertFalse((root / "github-plots/00_repository_overview_p01.svg").exists())
            comparison = build_compare("github", github_csv, root / "comparison", {"formats": ["svg"], "dpi": 72})
            self.assertTrue(comparison)

            openalex_csv = root / "openalex.csv"
            paper = {column: "" for column in OPENALEX_COLUMNS}
            paper.update({"schema_version": "1.0", "run_id": "run", "openalex_id": "W1", "artifact_id": "a1", "publication_family_id": "f1", "canonical_openalex_id": "W1", "paper_relationship": "canonical_base", "artifact_type": "dataset", "publication_year": "2024", "domain": "general_network", "data_form": "packet", "capture_environment": "synthetic_lab", "labeling": "labeled"})
            atomic_write_csv(openalex_csv, [paper], OPENALEX_COLUMNS)
            outputs = build_openalex(openalex_csv, None, root / "openalex-plots", {"formats": ["svg"], "dpi": 72})
            self.assertTrue(outputs)
            self.assertTrue((root / "openalex-plots/data/08_publication_timeline.csv").is_file())
            overview = root / "openalex-plots/data/00_artifact_timeline_overview.csv"
            self.assertTrue(overview.is_file())
            self.assertIn("artifact_type", overview.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue((root / "openalex-plots/00_artifact_timeline_overview.svg").is_file())


if __name__ == "__main__":
    unittest.main()
