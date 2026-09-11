from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ids_meta.common.csvio import atomic_write_csv, read_csv
from ids_meta.common.ensemble import decision_disposition


MULTI_FIELDS = {
    "github": {"observation_scopes", "capabilities", "detection_methods", "system_types"},
    "openalex": {"artifact_aliases"},
}
DEFAULT_FIELDS = {
    "github": ["role", "observation_scopes", "capabilities", "detection_methods"],
    "openalex": ["paper_role", "relevance_tier", "artifact_type", "domain", "data_form", "capture_environment", "labeling"],
}
IDENTITY_FIELDS = {"github": "github_repo_id", "openalex": "openalex_id"}
SELECTOR_FIELDS = {
    "github": {"repository": "full_name", "id": "github_repo_id"},
    "openalex": {"openalex_url": "openalex_id", "title": "title", "id": "openalex_id"},
}


@dataclass(frozen=True)
class RatingProfile:
    """Validated centrally configured reference classifications for one source."""

    source: str
    classification_fields: list[str]
    entities: dict[str, dict[str, Any]]
    path: Path


@dataclass(frozen=True)
class EvaluationResult:
    """Computed metrics and auditable tables for one pipeline result file."""

    metrics: dict[str, Any]
    instances: list[dict[str, Any]]
    field_metrics: list[dict[str, Any]]
    confusion: list[dict[str, Any]]
    outputs: list[Path]


def _normalize_match_value(selector: str, value: object) -> str:
    """Normalize a configured human-friendly entity selector for exact matching."""
    normalized = " ".join(str(value or "").strip().split())
    if selector == "openalex_url":
        normalized = normalized.rstrip("/")
    return normalized if selector == "id" else normalized.casefold()


def load_rating_profile(path: Path, source: str) -> RatingProfile:
    """Load and validate one source-specific central rating profile."""
    if source not in IDENTITY_FIELDS:
        raise ValueError(f"unsupported evaluation source: {source}")
    if not path.is_file():
        raise ValueError(f"evaluation rating profile does not exist: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if str(raw.get("schema_version", "")) != "1.0":
        raise ValueError(f"evaluation profile requires schema_version 1.0: {path}")
    if str(raw.get("source", "")) != source:
        raise ValueError(f"evaluation profile source must be {source!r}: {path}")
    fields = raw.get("classification_fields", DEFAULT_FIELDS[source])
    if not isinstance(fields, list) or any(not isinstance(field, str) or not field.strip() for field in fields):
        raise ValueError("classification_fields must be an array of non-empty strings")
    fields = list(dict.fromkeys(field.strip() for field in fields))
    rows = raw.get("entities", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("evaluation entities must be an array of tables")
    entities: dict[str, dict[str, Any]] = {}
    selectors = SELECTOR_FIELDS[source]
    for index, row in enumerate(rows, 1):
        configured = [
            (selector, str(row.get(selector, "")).strip())
            for selector in selectors if str(row.get(selector, "")).strip()
        ]
        if len(configured) != 1:
            choices = ", ".join(selectors)
            raise ValueError(f"evaluation entity {index} requires exactly one non-empty selector: {choices}")
        selector, value = configured[0]
        key = f"{selector}:{_normalize_match_value(selector, value)}"
        if key in entities:
            raise ValueError(f"duplicate evaluation entity selector: {selector}={value}")
        disposition = decision_disposition(row.get("decision"))
        if disposition == "unavailable":
            decision_value = row.get("decision")
            raise ValueError(f"evaluation entity {selector}={value} has an unsupported decision: {decision_value!r}")
        expected = dict(row)
        expected["_evaluation_selector"] = selector
        expected["_evaluation_match_value"] = value
        entities[key] = expected
    return RatingProfile(source, fields, entities, path)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    """Divide safely, returning zero for an empty denominator."""
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    """Return the harmonic mean of precision and recall."""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _values(value: object) -> set[str]:
    """Normalize scalar, array, or semicolon-delimited classification values."""
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = str(value or "").replace(",", ";").split(";")
    return {str(item).strip().casefold() for item in values if str(item).strip()}


def _index_rows(path: Path, identity: str) -> dict[str, dict[str, str]]:
    """Index a CSV by stable source identity while rejecting duplicate rows."""
    rows = read_csv(path)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        key = str(row.get(identity, "")).strip()
        if not key:
            continue
        if key in indexed:
            raise ValueError(f"duplicate {identity} in evaluation input {path}: {key}")
        indexed[key] = row
    return indexed


def _matching_row(
    rows: dict[str, dict[str, str]], source: str, selector: str, value: str, path: Path,
) -> tuple[str | None, dict[str, str] | None]:
    """Resolve one configured selector to at most one result row."""
    result_field = SELECTOR_FIELDS[source][selector]
    target = _normalize_match_value(selector, value)
    matches = [
        (identity, row) for identity, row in rows.items()
        if _normalize_match_value(selector, row.get(result_field)) == target
    ]
    if len(matches) > 1:
        identities = ", ".join(identity for identity, _row in matches)
        raise ValueError(
            f"ambiguous evaluation selector {selector}={value!r} in {path}: {identities}"
        )
    return matches[0] if matches else (None, None)


def _status(actual: dict[str, str] | None, candidate_present: bool, candidates_supplied: bool) -> str:
    """Describe how far one rated entity progressed through the pipeline."""
    if actual is not None:
        decision = str(actual.get("decision", "")).strip().casefold()
        if decision == "error" or actual.get("error"):
            return "error"
        disposition = decision_disposition(decision)
        if disposition == "uncertain":
            return "uncertain"
        if disposition in {"include", "exclude"}:
            return "classified"
        return "unavailable"
    if candidates_supplied:
        return "not_classified" if candidate_present else "not_retrieved"
    return "unavailable"


def _binary_metrics(tp: int, fp: int, tn: int, fn: int, total: int) -> dict[str, float | int]:
    """Calculate standard binary screening metrics from one confusion matrix."""
    recall = _ratio(tp, tp + fn)
    precision = _ratio(tp, tp + fp)
    specificity = _ratio(tn, tn + fp)
    false_positive_rate = _ratio(fp, fp + tn)
    false_negative_rate = _ratio(fn, tp + fn)
    accuracy = _ratio(tp + tn, total)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "detection_rate": recall, "recall": recall, "sensitivity": recall,
        "false_positive_rate": false_positive_rate, "false_negative_rate": false_negative_rate,
        "precision": precision, "specificity": specificity,
        "negative_predictive_value": _ratio(tn, tn + fn),
        "accuracy": accuracy, "f1": _f1(precision, recall),
        "balanced_accuracy": (recall + specificity) / 2 if total else 0.0,
        "DR": recall, "FPR": false_positive_rate, "FNR": false_negative_rate,
        "ACC": accuracy, "PREC": precision,
    }


def evaluate_results(
    profile: RatingProfile, classifications_path: Path, output_dir: Path,
    candidates_path: Path | None = None,
) -> EvaluationResult:
    """Compare actual classifications with central ratings and write evaluation artifacts."""
    identity = IDENTITY_FIELDS[profile.source]
    actual = _index_rows(classifications_path, identity)
    candidate_rows = _index_rows(candidates_path, identity) if candidates_path and candidates_path.is_file() else {}
    candidates_supplied = bool(candidates_path and candidates_path.is_file())
    confusion_counts: Counter[tuple[str, str]] = Counter()
    status_counts: Counter[str] = Counter()
    instances: list[dict[str, Any]] = []
    operational = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    strict = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    selective = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    binary_total = definitive_total = decision_correct = 0
    classified_count = 0
    matched_actual_ids: set[str] = set()
    field_state: dict[str, dict[str, Any]] = {
        field: {"comparisons": 0, "exact": 0, "tp": 0, "fp": 0, "fn": 0, "multilabel": field in MULTI_FIELDS[profile.source]}
        for field in profile.classification_fields
    }

    for _profile_key, expected in profile.entities.items():
        selector = str(expected["_evaluation_selector"])
        match_value = str(expected["_evaluation_match_value"])
        actual_key, row = _matching_row(actual, profile.source, selector, match_value, classifications_path)
        candidate_key, candidate_row = _matching_row(
            candidate_rows, profile.source, selector, match_value, candidates_path or classifications_path,
        )
        candidate_present = candidate_row is not None
        if actual_key:
            matched_actual_ids.add(actual_key)
        classified_count += int(row is not None)
        entity_id = actual_key or candidate_key or match_value
        public_expected = {key: value for key, value in expected.items() if not key.startswith("_evaluation_")}
        display_name = expected.get("name") or expected.get("repository") or expected.get("title") or match_value
        expected_disposition = decision_disposition(expected.get("decision"))
        actual_disposition = decision_disposition((row or {}).get("decision"))
        status = _status(row, candidate_present, candidates_supplied)
        status_counts[status] += 1
        confusion_counts[(expected_disposition, actual_disposition)] += 1
        decision_correct += int(expected_disposition == actual_disposition)
        if expected_disposition in {"include", "exclude"}:
            binary_total += 1
            predicted_include = actual_disposition == "include"
            if expected_disposition == "include":
                operational["tp" if predicted_include else "fn"] += 1
                strict["tp" if predicted_include else "fn"] += 1
            else:
                operational["fp" if predicted_include else "tn"] += 1
                if actual_disposition == "exclude":
                    strict["tn"] += 1
                else:
                    strict["fp"] += 1
            if actual_disposition in {"include", "exclude"}:
                definitive_total += 1
                if expected_disposition == "include":
                    selective["tp" if actual_disposition == "include" else "fn"] += 1
                else:
                    selective["fp" if actual_disposition == "include" else "tn"] += 1
        field_results: dict[str, bool] = {}
        for field in profile.classification_fields:
            if field not in expected:
                continue
            expected_values = _values(expected[field])
            actual_values = _values((row or {}).get(field))
            exact = expected_values == actual_values
            field_results[field] = exact
            state = field_state[field]
            state["comparisons"] += 1
            state["exact"] += int(exact)
            if field in MULTI_FIELDS[profile.source] or isinstance(expected[field], list):
                state["multilabel"] = True
                state["tp"] += len(expected_values & actual_values)
                state["fp"] += len(actual_values - expected_values)
                state["fn"] += len(expected_values - actual_values)
        instances.append({
            "entity_id": entity_id, "name": display_name, "match_field": selector, "match_value": match_value, "status": status,
            "candidate_present": str(candidate_present).lower(), "classification_present": str(row is not None).lower(),
            "expected_decision": expected.get("decision", ""), "expected_disposition": expected_disposition,
            "actual_decision": (row or {}).get("decision", "missing"), "actual_disposition": actual_disposition,
            "decision_correct": str(expected_disposition == actual_disposition).lower(),
            "field_results_json": json.dumps(field_results, sort_keys=True),
            "expected_json": json.dumps(public_expected, ensure_ascii=False, sort_keys=True),
            "actual_json": json.dumps(row or {}, ensure_ascii=False, sort_keys=True),
        })

    field_metrics: list[dict[str, Any]] = []
    total_field_comparisons = total_field_exact = 0
    for field, state in field_state.items():
        comparisons = int(state["comparisons"])
        total_field_comparisons += comparisons
        total_field_exact += int(state["exact"])
        multilabel = bool(state["multilabel"])
        precision = _ratio(state["tp"], state["tp"] + state["fp"]) if multilabel else None
        recall = _ratio(state["tp"], state["tp"] + state["fn"]) if multilabel else None
        field_metrics.append({
            "field": field, "comparisons": comparisons, "exact_matches": state["exact"],
            "exact_accuracy": _ratio(state["exact"], comparisons),
            "label_f1": _f1(precision, recall) if precision is not None and recall is not None else None,
        })
    operational_metrics = _binary_metrics(operational["tp"], operational["fp"], operational["tn"], operational["fn"], binary_total)
    strict_metrics = _binary_metrics(strict["tp"], strict["fp"], strict["tn"], strict["fn"], binary_total)
    selective_metrics = _binary_metrics(selective["tp"], selective["fp"], selective["tn"], selective["fn"], definitive_total)
    rated = len(profile.entities)
    metrics: dict[str, Any] = {
        "source": profile.source, "ratings_path": str(profile.path), "classifications_path": str(classifications_path),
        "rated_entities": rated, "actual_entities": len(actual), "unrated_actual_entities": len(set(actual) - matched_actual_ids),
        "decision_accuracy": _ratio(decision_correct, rated),
        "classification_field_accuracy": _ratio(total_field_exact, total_field_comparisons),
        "classification_field_comparisons": total_field_comparisons,
        "retrieval_coverage": _ratio(rated - status_counts["not_retrieved"], rated) if candidates_supplied else None,
        "classification_coverage": _ratio(classified_count, rated),
        "definitive_coverage": _ratio(status_counts["classified"], rated),
        "abstention_rate": _ratio(sum(status_counts[name] for name in ("uncertain", "error", "not_retrieved", "not_classified", "unavailable")), rated),
        "status_counts": dict(sorted(status_counts.items())),
        "operational": operational_metrics, "strict": strict_metrics, "selective": selective_metrics,
    }
    confusion = [{"expected": expected, "actual": actual_value, "count": count} for (expected, actual_value), count in sorted(confusion_counts.items())]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    instance_columns = ["entity_id", "name", "match_field", "match_value", "status", "candidate_present", "classification_present", "expected_decision", "expected_disposition", "actual_decision", "actual_disposition", "decision_correct", "field_results_json", "expected_json", "actual_json"]
    instances_path = output_dir / "instances.csv"
    confusion_path = output_dir / "decision_confusion.csv"
    fields_path = output_dir / "field_metrics.csv"
    atomic_write_csv(instances_path, instances, instance_columns)
    atomic_write_csv(confusion_path, confusion, ["expected", "actual", "count"])
    atomic_write_csv(fields_path, field_metrics, ["field", "comparisons", "exact_matches", "exact_accuracy", "label_f1"])
    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "# Classification evaluation\n\n"
        f"- Source: `{profile.source}`\n- Rated entities: {rated}\n"
        f"- Operational DR: {operational_metrics['detection_rate']:.4f}\n"
        f"- Operational FPR: {operational_metrics['false_positive_rate']:.4f}\n"
        f"- Operational accuracy: {operational_metrics['accuracy']:.4f}\n"
        f"- Operational precision: {operational_metrics['precision']:.4f}\n"
        f"- Metadata field accuracy: {metrics['classification_field_accuracy']:.4f} "
        f"({metrics['classification_field_comparisons']} comparisons)\n"
        f"- Abstention rate: {metrics['abstention_rate']:.4f}\n\n"
        "## Interpreting the three views\n\n"
        "- **Operational:** asks what the complete pipeline delivers. Every non-inclusion predicts negative.\n"
        "- **Strict:** asks how performance looks when every unresolved boundary result is wrong.\n"
        "- **Selective:** measures only definite include/exclude decisions. Always read it with definitive coverage and abstention rate.\n\n"
        "Example: if exclusion was expected but the result is uncertain, operational counts a true negative, strict counts a false positive, and selective omits the entity. If inclusion was expected, the same unresolved result is a false negative operationally and strictly, and is omitted selectively.\n",
        encoding="utf-8",
    )
    outputs = [metrics_path, instances_path, confusion_path, fields_path, readme_path]
    return EvaluationResult(metrics, instances, field_metrics, confusion, outputs)
