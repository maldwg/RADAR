"""Verify cross-field screening rules and their portable evaluation integration."""
from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ids_meta.cli import main
from ids_meta.common.csvio import read_csv
from tests.fixtures import write_csv as atomic_write_csv
from ids_meta.common.evaluation import load_rating_profile
from ids_meta.common.study_screening import StudyScreeningAdapter, load_study_protocol, screen_study
from ids_meta.common.verification import verify_database, verify_directory
from ids_meta.plotting.generic import build_report

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config/verification/memmesheimer"
EXAMPLE = ROOT / "examples/verification/memmesheimer/assessments.json"


class StudyScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        """Load the study protocol and screening fixtures."""
        self.protocol = load_study_protocol(PROFILE / "protocol.toml")
        self.adapter = StudyScreeningAdapter(self.protocol)
        self.example = json.loads(EXAMPLE.read_text())

    def assessment(self, value: str) -> dict[str, str]:
        """Construct a quoted fixture judgment."""
        return {"value": value, "quote": "synthetic fixture text", "source": "full_text"}

    def test_threshold_and_partial_information(self) -> None:
        """Verify threshold and partial information."""
        row = copy.deepcopy(self.example[0])
        self.assertEqual(self.adapter.normalize([row])[0]["decision"], "accept")
        row["criteria"].pop("Q5")
        result = self.adapter.normalize([row])[0]
        self.assertEqual(result["decision"], "uncertain")
        self.assertEqual(result["decision_basis"], "quality_incomplete")
        row["criteria"].update({f"Q{i}": self.assessment("no") for i in range(5, 10)})
        result = self.adapter.normalize([row])[0]
        self.assertEqual(result["decision"], "exclude")
        self.assertEqual(result["decision_basis"], "quality_threshold_unreachable")
        row["criteria"].pop("F1")
        self.assertEqual(self.adapter.normalize([row])[0]["decision"], "exclude")

    def test_eligibility_and_missing_evidence_are_distinct(self) -> None:
        """Verify eligibility and missing evidence are distinct."""
        rows = self.adapter.normalize(self.example)
        self.assertEqual([r["decision"] for r in rows], ["accept", "exclude", "uncertain"])
        self.assertEqual(rows[1]["failed_filters"], "F3")
        self.assertEqual(rows[2]["quality_unknown"], 9)
        row = copy.deepcopy(self.example[0])
        row["criteria"].pop("F2")
        self.assertEqual(self.adapter.normalize([row])[0]["decision_basis"], "eligibility_incomplete")

    def test_pdf_ligatures_are_equivalent_but_reworded_quotes_are_rejected(self) -> None:
        """Verify pdf ligatures are equivalent but reworded quotes are rejected."""
        row = copy.deepcopy(self.example[0])
        row['full_text'] += ' The treatment eﬀects are estimated.'
        row['criteria']['F1']['quote'] = 'The treatment effects are estimated.'
        result = self.adapter.normalize([row])[0]
        self.assertEqual(result['decision'], 'accept')
        self.assertEqual(result['evidence_normalization'], 'whitespace-and-latin-ligatures-v1')
        self.assertEqual(json.loads(result['criteria_json'])['F1']['quote'], 'The treatment effects are estimated.')
        row['criteria']['F1']['quote'] = 'The treatment effects are predicted.'
        with self.assertRaises(ValueError):
            self.adapter.normalize([row])

    def test_invalid_evidence_ids_and_decisions_fail_before_export(self) -> None:
        """Verify invalid evidence ids and decisions fail before export."""
        for change in (
            lambda r: r["criteria"]["F1"].update(quote="fabricated quotation"),
            lambda r: r["criteria"]["F1"].update(source="abstract"),
            lambda r: r["criteria"]["F1"].update(value=True),
            lambda r: r["criteria"].update(F99={"value": "unknown"}),
            lambda r: r.update(decision="accept"),
        ):
            row = copy.deepcopy(self.example[0])
            change(row)
            with self.assertRaises(ValueError):
                self.adapter.normalize([row])
        with self.assertRaises(ValueError):
            self.adapter.normalize([self.example[0], self.example[0]])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bad = root / "bad.json"
            bad.write_text(json.dumps([{"paper_id": "one", "criteria": {"F1": {"value": "yes"}}}]))
            out = root / "out.csv"
            out.write_text("existing output")
            with self.assertRaises(ValueError):
                screen_study(PROFILE / "protocol.toml", bad, out)
            self.assertEqual(out.read_text(), "existing output")

    def test_cli_export_keeps_provenance_and_assessments(self) -> None:
        """Verify cli export keeps provenance and assessments."""
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "classification.csv"
            with redirect_stdout(io.StringIO()):
                status = main(["study-screen", "--protocol", str(PROFILE / "protocol.toml"),
                    "--input", str(EXAMPLE), "--output", str(output)])
            self.assertEqual(status, 0)
            rows = read_csv(output)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(json.loads(rows[0]["criteria_json"])), 14)
            self.assertTrue(all(len(r["assessment_input_sha256"]) == 64 for r in rows))
            self.assertEqual(rows[0]["protocol_sha256"], self.protocol.sha256)

    def test_reference_crosswalk_and_offline_verification(self) -> None:
        """Verify reference crosswalk and offline verification."""
        contract = self.protocol.contract
        reference = load_rating_profile(PROFILE / "references.toml", contract.name, contract=contract)
        entities = list(reference.entities.values())
        self.assertEqual(len(entities), 11)
        self.assertEqual([r["survey_row"] for r in entities], list(range(1, 12)))
        self.assertTrue(all(r["decision"] == "accept" for r in entities))
        self.assertEqual([r["survey_row"] for r in entities if r["strong_ignorability"] == "not_required"], [2, 6])
        # Handwritten toy predictions test evaluation mechanics, not model performance.
        rows = [{"paper_id": f"test:{i}", "title": entities[i]["title"], "decision": "accept",
            "sutva": "required", "overlap": "required", "strong_ignorability": "required", 'classifier_supported_fields': '*'} for i in range(2)]
        candidates = [*rows, {"paper_id": "test:2", "title": entities[2]["title"]}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            classifications, candidate_csv = root / "predictions.csv", root / "candidates.csv"
            columns = ["paper_id", "title", "decision", *contract.fields]
            atomic_write_csv(classifications, rows, columns)
            atomic_write_csv(candidate_csv, candidates, columns)
            output = root / "report"
            build_report(PROFILE / "source.toml", classifications, output,
                {"formats": ["svg"], "dpi": 72, "show_subcaptions": False},
                candidates_path=candidate_csv, ratings_path=PROFILE / "references.toml")
            metrics = json.loads((output / "evaluation/metrics.json").read_text())
            self.assertEqual(metrics["operational"]["tp"], 2)
            self.assertEqual(metrics["operational"]["fn"], 9)
            self.assertIsNone(metrics["operational"]["precision"])
            self.assertIsNone(metrics["operational"]["negative_predictive_value"])
            self.assertAlmostEqual(metrics["taxonomy_field_metrics"]["strong_ignorability"]["exact_accuracy"], 1 / 11)
            instances = read_csv(output / "evaluation/instances.csv")
            self.assertEqual(instances[2]["status"], "not_classified")
            self.assertEqual(instances[3]["status"], "not_retrieved")
            for result in (verify_directory(output), verify_database(output / "reporting.sqlite3")):
                self.assertTrue(result["ok"], result)
            metrics["operational"]["precision"] = 1.0
            (output / "evaluation/metrics.json").write_text(json.dumps(metrics))
            self.assertFalse(verify_directory(output)["ok"])


if __name__ == "__main__":
    unittest.main()
