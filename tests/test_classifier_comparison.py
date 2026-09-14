from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ids_meta.common.csvio import atomic_write_csv
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
                    "not_estimable_metrics": [], "definitive_coverage": value,
                    "abstention_rate": 1 - value, "classification_field_accuracy": value,
                    "operational": {"DR": value, "FPR": 1 - value, "ACC": value,
                                    "PREC": value, "f1": value, "tp": 8, "fp": 2, "tn": 8, "fn": 2},
                }
                (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                classifications[label] = csv_path
                evaluations[label] = evaluation
            outputs = build_classifiers(
                "github", classifications, evaluations, root / "plots",
                {"formats": ["svg"], "dpi": 72, "top_n": 5},
            )
            self.assertTrue(outputs)
            for number, metric in enumerate((
                "detection_rate", "false_positive_rate", "accuracy", "precision", "f1",
                "definitive_coverage", "abstention_rate", "field_accuracy",
            ), 1):
                table = root / "plots/data" / f"{number:02d}_{metric}.csv"
                self.assertEqual(len(table.read_text(encoding="utf-8").splitlines()), 5)
                self.assertTrue((root / "plots" / f"{number:02d}_{metric}.svg").is_file())
            self.assertTrue((root / "plots/decision_agreement/data/02_pairwise_agreement.csv").is_file())


if __name__ == "__main__":
    unittest.main()
