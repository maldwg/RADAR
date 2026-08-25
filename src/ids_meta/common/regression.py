from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from ids_meta.common.ensemble import decision_disposition


@dataclass(frozen=True)
class RegressionResult:
    """Metrics and entity-level comparisons for one rated regression set."""

    metrics: dict[str, Any]
    instances: list[dict[str, Any]]


def _values(value: object) -> set[str]:
    """Normalize scalar or multi-label classification values for comparison."""
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().casefold() for item in value if str(item).strip()}
    return {item.strip().casefold() for item in str(value or "").replace(",", ";").split(";") if item.strip()}


def evaluate_regression(
    source: str, expected: dict[str, dict[str, Any]], rated_entities: dict[str, dict[str, Any]],
    classification_fields: list[str] | None = None,
) -> RegressionResult:
    """Compare dict-defined expectations with system-rated OpenAlex or GitHub entities.

    Decision accuracy is exact disposition accuracy. Detection rate is recall for
    expected inclusions. Classification accuracy is exact set equality across the
    requested non-decision fields, calculated only where an expected value exists.
    """
    if source not in {"openalex", "github"}:
        raise ValueError(f"unsupported regression source: {source}")
    default_field = "paper_role" if source == "openalex" else "role"
    fields = classification_fields or [default_field]
    comparisons: list[dict[str, Any]] = []
    decision_correct = true_positive = false_negative = 0
    field_correct = field_total = 0
    confusion: Counter[str] = Counter()
    for entity_id, truth in expected.items():
        actual = rated_entities.get(entity_id, {})
        expected_disposition = decision_disposition(truth.get("decision"))
        actual_disposition = decision_disposition(actual.get("decision"))
        correct = expected_disposition == actual_disposition
        decision_correct += int(correct)
        if expected_disposition == "include":
            true_positive += int(actual_disposition == "include")
            false_negative += int(actual_disposition != "include")
        confusion[f"{expected_disposition}->{actual_disposition}"] += 1
        row: dict[str, Any] = {
            "entity_id": entity_id, "expected_decision": truth.get("decision", ""),
            "actual_decision": actual.get("decision", "missing"), "decision_correct": correct,
        }
        for field in fields:
            if field not in truth:
                continue
            matches = _values(truth[field]) == _values(actual.get(field))
            row[f"{field}_correct"] = matches
            field_correct += int(matches)
            field_total += 1
        comparisons.append(row)
    total = len(expected)
    metrics = {
        "source": source, "instances": total, "rated_instances": sum(key in rated_entities for key in expected),
        "accuracy": decision_correct / total if total else 0.0,
        "detection_rate": true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0,
        "classification_accuracy": field_correct / field_total if field_total else 0.0,
        "classification_comparisons": field_total, "confusion": dict(sorted(confusion.items())),
    }
    return RegressionResult(metrics, comparisons)


def evaluate_github_regression(
    expected: dict[str, dict[str, Any]], rated_entities: dict[str, dict[str, Any]],
    classification_fields: list[str] | None = None,
) -> RegressionResult:
    """Evaluate a dict-defined GitHub regression set against rated repositories."""
    return evaluate_regression("github", expected, rated_entities, classification_fields)


def evaluate_openalex_regression(
    expected: dict[str, dict[str, Any]], rated_entities: dict[str, dict[str, Any]],
    classification_fields: list[str] | None = None,
) -> RegressionResult:
    """Evaluate a dict-defined OpenAlex regression set against rated works."""
    return evaluate_regression("openalex", expected, rated_entities, classification_fields)
