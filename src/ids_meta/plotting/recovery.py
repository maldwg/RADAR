from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def recovered_taxonomy_table(input_dir: Path) -> pd.DataFrame | None:
    """Score fields only on reference positives matched to an accepted output.

    Reuse the persisted per-entity field checks so aliases, cited-paper fallback,
    unscored labels and exact multi-label matching follow the original evaluation.
    """
    audit_path = input_dir / "instances.csv"
    if not audit_path.is_file():
        raise ValueError("recovered taxonomy requires instances.csv")
    fields = pd.read_csv(input_dir / "field_metrics.csv", keep_default_na=False)
    audit = pd.read_csv(audit_path, keep_default_na=False)
    positive = audit["expected_disposition"].eq("include")
    recovered = audit.loc[positive & audit["actual_disposition"].eq("include")]
    checks = [json.loads(value) for value in recovered["field_results_json"]]
    if any(not isinstance(check, dict) or any(type(value) is not bool for value in check.values()) for check in checks):
        raise ValueError(f"invalid per-field checks in {audit_path}")
    expected_count = int(positive.sum())
    recovered_count = len(recovered)
    rows = []
    for field in fields.to_dict("records"):
        name = str(field["field"])
        supported = str(field["supported"]).casefold() == "true"
        results = [check[name] for check in checks if name in check] if supported else []
        comparisons = len(results)
        exact_matches = sum(results)
        value = exact_matches / comparisons if comparisons else None
        rows.append({
            "field": name, "supported": supported,
            "comparisons": comparisons, "exact_matches": exact_matches,
            "value": value, "percentage": value * 100 if value is not None else None,
            "expected_positive_entities": expected_count, "recovered": recovered_count,
            "missed": expected_count - recovered_count,
            "recovery_rate": recovered_count / expected_count if expected_count else None,
            "inventory_comparisons": field["comparisons"],
            "population": "recovered_reference_positives",
        })
    return pd.DataFrame(rows)
