"""Exercise study isolation, portable manifests and the shared CLI stages."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from ids_meta.cli import _execution_paths, main
from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError, load_prompt_profile, load_settings
from ids_meta.common.csvio import atomic_write_csv, read_csv
from ids_meta.github.schema import CANDIDATE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


class StudyConfigurationTests(unittest.TestCase):
    """Run fresh study configurations without network or model calls."""

    def _copy_config(self, root: Path) -> Path:
        """Copy the complete configuration tree to test relative file references."""
        shutil.copytree(ROOT / "config", root / "config")
        return root / "config/studies"

    def test_all_studies_resolve_from_their_manifest_and_keep_outputs_separate(self):
        """Moving the tree preserves references without depending on the cwd."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = self._copy_config(root)
            outputs = set()
            for name in ("ids_repositories", "ids_datasets", "conformal_treatment_effects"):
                settings = load_settings(studies / name / "study.toml")
                self.assertEqual(settings.root, root)
                outputs.add(settings.section("study")["output_root"])
                for value in settings.section("study")["input_fingerprints"]:
                    self.assertTrue(Path(value).is_relative_to(root))
                self.assertTrue(settings.path("interim_dir").is_relative_to(root / "artifacts/studies" / name))
            self.assertEqual(len(outputs), 3)

    def test_different_study_ids_use_the_same_workflow_without_cli_changes(self):
        """A configured use case does not need a hardcoded study-name branch."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = self._copy_config(root)
            custom = studies / "other_security_repositories"
            shutil.copytree(studies / "ids_repositories", custom)
            manifest = custom / "study.toml"
            manifest.write_text(manifest.read_text().replace("ids_repositories", "other_security_repositories"))
            settings = load_settings(manifest)
            self.assertEqual(settings.section("study")["id"], "other_security_repositories")
            self.assertEqual(settings.section("study")["workflow"], "repository_catalogue")
            self.assertIn("roles", load_prompt_profile(settings, "github"))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(manifest), "validate"]), 0)

    def test_old_configuration_and_ambiguous_settings_are_rejected(self):
        """Require a current manifest and one explicit owner for each setting."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old.toml"
            old.write_text('schema_version = "1.0"\n[ollama]\nmodels = ["test"]\n')
            with self.assertRaisesRegex(ConfigError, "study manifest"):
                load_settings(old)
            studies = self._copy_config(root)
            runtime = root / "config/shared/runtime.toml"
            runtime.write_text(runtime.read_text() + '\nmodels = ["ambiguous"]\n')
            with self.assertRaisesRegex(ConfigError, "ollama.models.*more than one"):
                load_settings(studies / "ids_repositories/study.toml")

    def test_unselected_and_unimplemented_sources_fail_before_execution(self):
        """Provider selection cannot silently reuse another study's rules."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        with self.assertRaisesRegex(ConfigError, "does not select"):
            settings.require_source("openalex")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = self._copy_config(root)
            manifest = studies / "ids_datasets/study.toml"
            manifest.write_text(manifest.read_text().replace('name = "openalex"', 'name = "unknown_provider"'))
            with self.assertRaisesRegex(ConfigError, "unsupported"):
                load_settings(manifest)

    def test_study_scientific_settings_do_not_mutate_other_studies(self):
        """Editing one baseline threshold leaves another use case independent."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = self._copy_config(root)
            baseline = studies / "ids_repositories/baselines/settings.toml"
            baseline.write_text(baseline.read_text().replace("accept_threshold = 7.0", "accept_threshold = 9.0"))
            repo = load_settings(studies / "ids_repositories/study.toml")
            datasets = load_settings(studies / "ids_datasets/study.toml")
            self.assertEqual(repo.section("baselines")["keyword"]["github"]["accept_threshold"], 9)
            self.assertEqual(datasets.section("baselines")["keyword"]["openalex"]["accept_threshold"], 7)

    def test_generic_repository_stage_writes_model_outputs_and_configuration_provenance(self):
        """Execute the existing classification engine through the study command."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = self._copy_config(root)
            manifest = studies / "ids_repositories/study.toml"
            settings = load_settings(manifest)
            candidates = root / "candidates.csv"
            row = {column: "" for column in CANDIDATE_COLUMNS}
            row.update(github_repo_id="one", full_name="fixture/detector", prefilter_band="candidate",
                prefilter_score="9", evidence_hash="fixture")
            atomic_write_csv(candidates, [row], CANDIDATE_COLUMNS)
            with PipelineCache(settings.path("github_cache")) as cache:
                cache.put_record("github.repository", "one", {"repository": {"full_name": "fixture/detector"},
                    "readme_text": "A repository fixture."}, "now")
            with patch("requests.Session.request", side_effect=AssertionError("network call")), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(manifest), "classify", "--input", str(candidates), "--dry-run"]), 0)
            paths = _execution_paths(settings, "github", settings.section("ollama")["models"])
            for model_paths in paths.values():
                self.assertEqual(len(read_csv(model_paths["classifications"])), 1)
            saved = list(settings.path("runs_dir").glob("*.json"))
            self.assertEqual(len(saved), 1)
            run = json.loads(saved[0].read_text())
            self.assertEqual(run["study"]["id"], "ids_repositories")
            self.assertIn(str(manifest), run["study"]["input_fingerprints"])

    def test_generic_review_classification_uses_study_prompt_and_taxonomy_offline(self):
        """Run a generic review preview with isolated model outputs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            studies = self._copy_config(root)
            manifest = studies / "conformal_treatment_effects/study.toml"
            settings = load_settings(manifest)
            candidates = root / "candidates.json"
            candidates.write_text(json.dumps([{"paper_id": "one", "title": "Fixture", "full_text": "evidence"}]))
            with patch("requests.Session.request", side_effect=AssertionError("network call")), redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--config", str(manifest), "classify", "--input", str(candidates), "--dry-run"]), 0)
            for model in settings.section("ollama")["models"]:
                output = _execution_paths(settings, "openalex", [model])[model]["classifications"]
                row = read_csv(output)[0]
                self.assertEqual(row["decision"], "uncertain")
                self.assertEqual(row["sutva"], "unknown")
                checkpoint = json.loads(output.with_suffix(".assessments.json").read_text())
                self.assertEqual(len(checkpoint["prompt_sha256"]), 64)

