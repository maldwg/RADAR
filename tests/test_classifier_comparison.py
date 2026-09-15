from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ids_meta.common.csvio import atomic_write_csv, read_csv
from ids_meta.plotting.compare import build_classifiers


class ClassifierComparisonTests(unittest.TestCase):
    """Exercise unified keyword, NLI, LLM, and ensemble performance plots."""

    def test_every_metric_places_all_approaches_in_one_table_and_plot(self) -> None:
        """Keep classifier families together instead of emitting incomparable plot trees."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classifications = {}
            evaluations = {}
            approaches = (
                ("Keyword", "keyword", .50), ("NLI · fixture", "nli", .65),
                ("LLM · fixture", "llm", .75), ("Ensemble", "ensemble", .80),
            )
            for index, (label, method, value) in enumerate(approaches):
                csv_path = root / f"classifier-{index}.csv"
                atomic_write_csv(csv_path, [{
                    "github_repo_id": "g1", "full_name": "fixture/repo",
                    "decision": "accept" if index % 2 else "exclude",
                }], ["github_repo_id", "full_name", "decision"])
                evaluation = root / f"evaluation-{index}"
                evaluation.mkdir()
                metrics = {
                    "ratings_path": "/same/profile.toml", "evaluation_unit": "source_record",
                    "classifier_method": method, "rated_entities": 20,
                    "not_estimable_metrics": ["false_discovery_rate"] if index == 0 else [],
                    "definitive_coverage": value,
                    "abstention_rate": 1 - value, "classification_field_accuracy": value,
                    "classification_field_macro_f1": value - .05,
                    "operational": {"DR": value, "FNR": 1 - value,
                                    "FPR": 1 - value, "FDR": 1 - value, "ACC": value,
                                    "PREC": value, "f1": value, "tp": 8, "fp": 2, "tn": 8, "fn": 2},
                }
                (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                atomic_write_csv(evaluation / "field_metrics.csv", [{
                    "field": "role", "supported": "true", "comparisons": 20,
                    "exact_matches": int(value * 20), "exact_accuracy": value,
                    "label_f1": "", "macro_f1": value - .05,
                    "per_label_metrics_json": "{}",
                }], [
                    "field", "supported", "comparisons", "exact_matches", "exact_accuracy",
                    "label_f1", "macro_f1", "per_label_metrics_json",
                ])
                atomic_write_csv(evaluation / "field_label_metrics.csv", [{
                    "field": "role", "label": "native_detector", "supported": "true",
                    "tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0,
                    "f1": value,
                }], [
                    "field", "label", "supported", "tp", "fp", "fn",
                    "precision", "recall", "f1",
                ])
                classifications[label] = csv_path
                evaluations[label] = evaluation
            outputs = build_classifiers(
                "github", classifications, evaluations, root / "plots",
                {"formats": ["svg"], "dpi": 72, "top_n": 5},
            )
            self.assertTrue(outputs)
            overview = root / "plots/data/00_metric_overview.csv"
            self.assertEqual(len(overview.read_text(encoding="utf-8").splitlines()), 45)
            self.assertTrue((root / "plots/00_metric_overview.svg").is_file())
            self.assertIn(
                "N/A", (root / "plots/00_metric_overview.svg").read_text(encoding="utf-8"),
            )
            for number, metric in enumerate((
                "detection_rate", "false_positive_rate", "accuracy", "precision", "f1",
                "definitive_coverage", "abstention_rate", "field_accuracy", "field_macro_f1",
                "false_discovery_rate",
            ), 1):
                table = root / "plots/metrics/data" / f"{number:02d}_{metric}.csv"
                self.assertEqual(len(table.read_text(encoding="utf-8").splitlines()), 5)
                self.assertTrue((root / "plots/metrics" / f"{number:02d}_{metric}.svg").is_file())
            fdr_rows = read_csv(root / "plots/metrics/data/10_false_discovery_rate.csv")
            self.assertEqual(fdr_rows[0]["value"], "")
            self.assertEqual(fdr_rows[0]["ci_low"], "")
            self.assertEqual(fdr_rows[0]["ci_high"], "")
            self.assertTrue((root / "plots/taxonomy/01_exact_accuracy_by_field.svg").is_file())
            self.assertTrue((root / "plots/taxonomy/02_macro_f1_by_field.svg").is_file())
            self.assertTrue((root / "plots/taxonomy/03_category_f1.svg").is_file())
            self.assertTrue((root / "plots/agreement/data/02_pairwise_agreement.csv").is_file())
            self.assertTrue((root / "plots/README.md").is_file())
            detection_svg = (root / "plots/metrics/01_detection_rate.svg").read_text(encoding="utf-8")
            self.assertIn("95% Wilson interval", detection_svg)
            self.assertIn("bar height=observed metric", detection_svg)
            self.assertNotIn("65.0%", detection_svg)


if __name__ == "__main__":
    unittest.main()
