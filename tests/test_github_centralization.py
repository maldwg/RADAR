from __future__ import annotations

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from tests.fixtures import SYSTEM_TYPE_PROFILE

import pandas as pd

from ids_meta.common.baselines import infer_keyword_fields, load_keyword_taxonomy_profile
from ids_meta.common.config import load_prompt_profile, load_settings
from tests.fixtures import write_csv as atomic_write_csv
from ids_meta.common.ensemble import _github_field_consensus, ensemble_classifications
from ids_meta.common.evaluation import evaluate_results, load_rating_profile
from ids_meta.common.nli import NLIProfile, decide_nli_fields, load_nli_profile
from ids_meta.github.baseline import normalize_keyword_result, normalize_nli_result
from ids_meta.github.pipeline import prompt
from ids_meta.github.schema import CENTRALIZATION, CLASSIFICATION_COLUMNS, RESPONSE_SCHEMA, normalize_response
from ids_meta.openalex.schema import RESPONSE_SCHEMA as OPENALEX_SCHEMA
from ids_meta.plotting.common import configure_style
from ids_meta.plotting.compare import build_classifiers
from ids_meta.plotting.evaluation import _field_accuracy
from ids_meta.plotting.github import _centralization_plots

from ids_meta.common.provenance import audited_bundle
_field_accuracy = audited_bundle(_field_accuracy)
_centralization_plots = audited_bundle(_centralization_plots)



ROOT = Path(__file__).resolve().parents[1]


class GitHubCentralizationTests(unittest.TestCase):
    """Verify GitHub-only architecture classification, scoring, and presentation."""

    def test_schema_prompt_and_normalizers(self) -> None:
        """Require a direct model field and keep it separate from inclusion and OpenAlex."""
        settings = load_settings(ROOT / "config/studies/ids_repositories/study.toml")
        self.assertEqual(RESPONSE_SCHEMA["properties"]["centralization"]["enum"], list(CENTRALIZATION))
        self.assertIn("centralization", RESPONSE_SCHEMA["required"])
        self.assertIn("centralization", CLASSIFICATION_COLUMNS)
        self.assertNotIn("centralization", OPENALEX_SCHEMA["properties"])
        rendered = prompt([], load_prompt_profile(settings, "github"), system_type_profile=SYSTEM_TYPE_PROFILE)
        self.assertIn("Collaborative IDS / CIDS", rendered)
        self.assertIn("distributed IDS / DIDS", rendered)
        self.assertNotIn("${centralization}", rendered)
        nli_field = next(field for field in load_nli_profile(settings, "github").fields if field.name == "centralization")
        self.assertEqual(set(nli_field.hypotheses), set(CENTRALIZATION) - {"unknown"})
        only_topology = NLIProfile("github", (nli_field,), {}, settings.path("github_nli"))
        audit = {"centralization": {label: {"entailment": .9 if label == "hierarchical" else .1,
                                             "contradiction": .05}
                                    for label in nli_field.hypotheses}}
        self.assertEqual(decide_nli_fields(audit, only_topology, .5, .1)["centralization"], "hierarchical")
        for label in (*CENTRALIZATION, "invalid"):
            raw = {"role": "out_of_scope", "centralization": label}
            expected = label if label in CENTRALIZATION else "unknown"
            self.assertEqual(normalize_response(raw, [], {"require_verified_quote": False}, False, system_type_profile=SYSTEM_TYPE_PROFILE)["centralization"], expected)
            for normalizer in (normalize_keyword_result, normalize_nli_result):
                result = normalizer(raw, {}, {"system_type_profile": SYSTEM_TYPE_PROFILE})
                self.assertEqual(result["centralization"], expected)
                self.assertEqual(result["decision"], "exclude")
                self.assertIn("centralization", result["classifier_supported_fields"].split(";"))

    def test_architecture_keywords_and_abstention(self) -> None:
        """Honor central/CIDS/DIDS vocabulary without matching DDoS or choosing tied classes."""
        profile = load_keyword_taxonomy_profile(load_settings(ROOT / "config/studies/ids_repositories/study.toml"), "github")
        examples = {
            "central analysis engine": "centralized",
            "single-node detector": "centralized",
            "single node detector": "centralized",
            "centralised IDS": "centralized",
            "collaborative IDS": "hierarchical",
            "CIDS": "hierarchical",
            "hierarchical detection": "hierarchical",
            "DIDS": "fully_distributed",
            "distributed IDS": "fully_distributed",
            "distributed intrusion detection": "fully_distributed",
            "peer-to-peer detection": "fully_distributed",
            "distributed denial of service detection": "unknown",
            "NIDS detects attacks": "unknown",
            "central CIDS": "unknown",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                predictions, scores, signals = infer_keyword_fields(text, profile)
                self.assertEqual(predictions["centralization"], expected)
                self.assertIn("centralization", scores)
                self.assertIn("centralization", signals)

    def test_consensus_votes_topology_independently(self) -> None:
        """Vote architecture rather than inheriting it from the representative model."""
        votes = {
            "a": {"role": "uncertain", "centralization": "centralized"},
            "b": {"role": "uncertain", "centralization": "hierarchical"},
            "c": {"role": "uncertain", "centralization": "hierarchical"},
        }
        result, audit = _github_field_consensus(dict(votes["a"]), votes, .5)
        self.assertEqual(result["centralization"], "hierarchical")
        self.assertIn("centralization", audit)
        result, _audit = _github_field_consensus({}, votes, .8)
        self.assertEqual(result["centralization"], "unknown")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {}
            for model, vote in votes.items():
                path = root / f"{model}.csv"
                atomic_write_csv(path, [{"github_repo_id": "1", "decision": "uncertain", **vote, 'classifier_supported_fields': '*'}],
                                 ["github_repo_id", "decision", "role", "centralization"])
                inputs[model] = path
            rows = ensemble_classifications("github", inputs, root / "ensemble.csv", .5, CLASSIFICATION_COLUMNS)
            self.assertEqual(rows[0]["centralization"], "hierarchical")
            rows = ensemble_classifications("github", inputs, root / "strict.csv", .8, CLASSIFICATION_COLUMNS)
            self.assertEqual(rows[0]["centralization"], "unknown")

    def test_reference_scoring_and_svg_pdf_plots(self) -> None:
        """Score supported topology only and render it in evaluation and general figures."""
        profile = load_rating_profile(ROOT / "config/studies/ids_repositories/evaluation/references.toml", "github")
        counts = Counter(entity["centralization"] for entity in profile.entities.values())
        self.assertEqual(sum(counts.values()), 120)
        self.assertGreater(counts["centralized"], 0)
        self.assertGreater(counts["hierarchical"], 0)
        for entity in profile.entities.values():
            self.assertIn(entity["centralization"], CENTRALIZATION)
            if entity["centralization"] != "unknown":
                self.assertTrue(entity["centralization_sources"])
                self.assertTrue(entity["centralization_note"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"github_repo_id": str(index), "full_name": entity["repository"],
                     "decision": entity["decision"], "centralization": entity["centralization"],
                     "classifier_supported_fields": "centralization"}
                    for index, entity in enumerate(profile.entities.values())]
            # An abstention on a known reference architecture remains an error.
            known = next(row for row in rows if row["centralization"] != "unknown")
            known["centralization"] = "unknown"
            actual = root / "actual.csv"
            atomic_write_csv(actual, rows, list(rows[0]))
            evaluation = evaluate_results(profile, actual, root / "evaluation")
            field = evaluation.metrics["taxonomy_field_metrics"]["centralization"]
            supported = 120 - counts["unknown"]
            self.assertEqual(field["comparisons"], supported)
            self.assertEqual(field["exact_matches"], supported - 1)
            self.assertEqual(field["per_label"]["centralized"]["fn"], 1)
            configure_style()
            outputs = _field_accuracy(root / "evaluation", evaluation.metrics, root / "figures", ["svg", "pdf"], 72)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertIn("IDS centralization", (root / "figures/03_taxonomy_fields.svg").read_text())
            outputs = build_classifiers("github", {label: actual for label in ("Keyword", "NLI", "LLM", "Ensemble")},
                                       {label: root / "evaluation" for label in ("Keyword", "NLI", "LLM", "Ensemble")},
                                       root / "comparison", {"formats": ["svg", "pdf"], "dpi": 72})
            self.assertTrue(all(path.is_file() for path in outputs))
            accuracy_plot = next(path for path in outputs if path.name == "01_exact_accuracy_by_field.svg")
            self.assertIn("IDS centralization", accuracy_plot.read_text())
            for label in ("Keyword", "NLI", "LLM", "Ensemble"):
                self.assertIn(label, accuracy_plot.read_text())
            frame = pd.DataFrame([
                {"github_repo_id": "1", "full_name": "test/central", "project_id": "p1", "role": "native_detector", "centralization": "centralized", "repository_relationship": "canonical"},
                {"github_repo_id": "2", "full_name": "test/hier", "project_id": "p2", "role": "detection_framework", "centralization": "hierarchical", "repository_relationship": "canonical"},
                {"github_repo_id": "3", "full_name": "test/fork", "project_id": "p1", "role": "native_detector", "centralization": "centralized", "repository_relationship": "fork"},
            ])
            outputs = _centralization_plots(frame, root / "general", ["svg", "pdf"], 72)
            self.assertEqual(len(outputs), 4)
            self.assertTrue(all(path.is_file() for path in outputs))
            table = pd.read_csv(root / "general/data/13_ids_centralization.csv")
            self.assertEqual(table["count"].sum(), 2)
            outputs = _centralization_plots(frame.iloc[:0], root / "empty", ["svg"], 72)
            self.assertTrue(all(path.is_file() for path in outputs))


if __name__ == "__main__":
    unittest.main()
