from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def reference_uncertainty(input_dir: Path) -> dict[str, int | float | None] | None:
    """Count uncertain outcomes among attempted classifications of rated entities.

    Prefer the matched artifact/repository classification. If no artifact match
    exists, use its classified OpenAlex survey reference. Each reference entity
    contributes at most once; missing retrieval/classification is not an attempt.
    Errors remain in the attempted denominator but never count as uncertainty.
    """
    path = input_dir / "instances.csv"
    if not path.is_file():
        raise ValueError("reference uncertainty requires instances.csv")
    audit = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"classification_present", "actual_disposition", "status"}
    if not required <= set(audit.columns):
        raise ValueError("reference uncertainty requires current attempt metadata")
    counts: Counter[str] = Counter()
    for row in audit.to_dict("records"):
        status = row["status"].strip().casefold()
        if row["classification_present"].strip().casefold() == "true":
            disposition = row["actual_disposition"].strip().casefold()
            error = status == "error"
        elif row.get("survey_reference_classification_present", "").strip().casefold() == "true":
            disposition = row["survey_reference_actual_disposition"].strip().casefold()
            error = status == "survey_reference_classification_error"
        else:
            counts["not_attempted"] += 1
            continue
        outcome = "error" if error else {"include": "accept", "exclude": "exclude", "uncertain": "uncertain"}.get(disposition, "unavailable")
        counts[outcome] += 1
    attempted = len(audit) - counts["not_attempted"]
    return {
        "rated_entities": len(audit), "attempted": attempted,
        **{key: counts[key] for key in ("accept", "exclude", "uncertain", "error", "unavailable", "not_attempted")},
        "value": counts["uncertain"] / attempted if attempted else None,
    }
