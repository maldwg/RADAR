from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from radar.cli import _execute_github, _execute_openalex, _execution_paths, _variant_paths
from radar.common.cache import PipelineCache
from radar.common.config import Settings
from radar.common.csvio import read_csv
from tests.fixtures import write_csv as atomic_write_csv
from radar.github.schema import CANDIDATE_COLUMNS as GITHUB_CANDIDATE_COLUMNS
from radar.openalex.schema import CANDIDATE_COLUMNS as OPENALEX_CANDIDATE_COLUMNS


ROOT = Path(__file__).resolve().parents[1]


class MultiModelPipelineTests(unittest.TestCase):
    def _settings(self, root: Path) -> Settings:
        """Build a two-model offline configuration rooted in a temporary directory."""
        values = {
            "paths": {
                "raw_dir": str(root / "raw"), "interim_dir": str(root / "interim"),
                "processed_dir": str(root / "processed"), "plots_dir": str(root / "plots"),
                "runs_dir": str(root / "runs"), "github_cache": str(root / "github.sqlite3"),
                "openalex_cache": str(root / "openalex.sqlite3"),
                "github_terms": str(ROOT / "config/studies/ids_repositories/search/github.toml"),
                "openalex_terms": str(ROOT / "config/studies/ids_datasets/search/openalex.toml"),
                "github_prompts": str(ROOT / "config/studies/ids_repositories/prompts/classification.toml"),
                "github_screening": str(ROOT / "config/studies/ids_repositories/screening.toml"),
                "openalex_screening": str(ROOT / "config/studies/ids_datasets/screening.toml"),
                "github_taxonomy": str(ROOT / "config/studies/ids_repositories/taxonomy.toml"),
                "openalex_taxonomy": str(ROOT / "config/studies/ids_datasets/taxonomy.toml"),
                "openalex_prompts": str(ROOT / "config/studies/ids_datasets/prompts/classification.toml"),
                "github_system_types": str(ROOT / "config/studies/ids_repositories/taxonomy.toml"),
             'evaluation_dir': str(root / 'evaluation')},
            "logging": {"progress_every": 25},
            "ollama": {"models": ["model-alpha", "model-beta"], "temperature": 0.0, "seed": 42},
            "ensemble": {"enabled": True, "models": [], "threshold": .5},
            "github": {"collection_limit": 10, "classification_limit": 10, "borderline_threshold": 4, "reconcile_with_llm": False},
            "openalex": {"collection_limit": 10, "classification_limit": 10, "borderline_threshold": 4, "reconcile_with_llm": False, "enrich_missing_abstracts": False},
            "plots": {"formats": ["svg"], "dpi": 72, "top_n": 5},
        }
        return Settings(root, values)

    def _args(self) -> argparse.Namespace:
        """Return arguments for a network-free multi-model classification stage."""
        return argparse.Namespace(
            action="classify", limit=None, collect_limit=None, classify_limit=None,
            classify_all=False, resume=False, dry_run=True, input=None, output=None,
        )

    def test_both_sources_write_per_model_and_ensemble_csvs(self) -> None:
        """Verify GitHub and OpenAlex isolate model outputs and emit an ensemble audit."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = self._settings(root)
            models = ["model-alpha", "model-beta"]

            github_paths = _execution_paths(settings, "github", models)
            github_candidates = root / "interim/github/candidates.csv"
            github_row = {column: "" for column in GITHUB_CANDIDATE_COLUMNS}
            github_row.update({"github_repo_id": "1", "full_name": "fixture/detector", "prefilter_band": "candidate", "prefilter_score": "9", "evidence_hash": "github-evidence"})
            atomic_write_csv(github_candidates, [github_row], GITHUB_CANDIDATE_COLUMNS)
            with PipelineCache(root / "github.sqlite3") as cache:
                cache.put_record("github.repository", "1", {"repository": {"full_name": "fixture/detector", "description": "Intrusion detector"}, "readme_text": "Runs an intrusion detector."}, "now")
            _execute_github(self._args(), settings)
            for model in models:
                rows = read_csv(github_paths[model]["classifications"])
                self.assertEqual(rows[0]["llm_model"], model)
            github_ensemble = read_csv(_variant_paths(settings, "github", Path("ensemble"))["classifications"])
            self.assertEqual(github_ensemble[0]["decision"], "uncertain")
            self.assertEqual(github_ensemble[0]["ensemble_unavailable_score"], "1.000")

            openalex_paths = _execution_paths(settings, "openalex", models)
            openalex_candidates = root / "interim/openalex/paper_candidates.csv"
            openalex_row = {column: "" for column in OPENALEX_CANDIDATE_COLUMNS}
            openalex_row.update({"openalex_id": "W1", "title": "Fixture dataset", "prefilter_band": "candidate", "prefilter_score": "9", "retrieval_rank_score": "8", "evidence_hash": "openalex-evidence"})
            atomic_write_csv(openalex_candidates, [openalex_row], OPENALEX_CANDIDATE_COLUMNS)
            with PipelineCache(root / "openalex.sqlite3") as cache:
                cache.put_record("openalex.work", "W1", {"work": {"title": "Fixture dataset", "abstract": "We release a network dataset."}}, "now")
            _execute_openalex(self._args(), settings)
            for model in models:
                rows = read_csv(openalex_paths[model]["classifications"])
                self.assertEqual(rows[0]["llm_model"], model)
            openalex_ensemble = read_csv(_variant_paths(settings, "openalex", Path("ensemble"))["classifications"])
            self.assertEqual(openalex_ensemble[0]["decision"], "uncertain")
            self.assertEqual(openalex_ensemble[0]["ensemble_vote_status"], "below_threshold")


if __name__ == "__main__":
    unittest.main()
