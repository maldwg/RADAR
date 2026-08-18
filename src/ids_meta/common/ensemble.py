from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.run import utc_now
from ids_meta.common.text import stable_hash


def decision_disposition(decision: object) -> str:
    """Map a source-specific classification decision to a common vote group."""
    normalized = str(decision or "").strip().casefold()
    if normalized in {"accept", "accept_adjacent"}:
        return "include"
    if normalized == "exclude":
        return "exclude"
    if normalized == "uncertain":
        return "uncertain"
    return "unavailable"


def _included_decision(rows: list[dict[str, str]]) -> str:
    """Choose the modal exact include decision with a conservative tie break."""
    counts = Counter(str(row.get("decision", "")).strip().casefold() for row in rows)
    counts = Counter({key: value for key, value in counts.items() if key in {"accept", "accept_adjacent"}})
    if not counts:
        return "accept"
    maximum = max(counts.values())
    winners = {key for key, value in counts.items() if value == maximum}
    return "accept_adjacent" if "accept_adjacent" in winners else "accept"


def ensemble_classifications(
    source: str,
    model_inputs: dict[str, Path],
    output: Path,
    threshold: float,
    columns: Sequence[str],
) -> list[dict[str, object]]:
    """Combine model classification CSVs using an auditable threshold vote."""
    if source not in {"github", "openalex"}:
        raise ValueError(f"unsupported ensemble source: {source}")
    if not model_inputs:
        raise ValueError("at least one ensemble model is required")
    if not 0 < threshold <= 1:
        raise ValueError("ensemble threshold must be greater than 0 and at most 1")
    identity = "github_repo_id" if source == "github" else "openalex_id"
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    order: list[str] = []
    for model, path in model_inputs.items():
        rows = read_csv(path)
        require_columns(rows, (identity, "decision"), path)
        indexed[model] = {}
        for row in rows:
            key = str(row.get(identity, "")).strip()
            if not key:
                continue
            indexed[model][key] = row
            if key not in order:
                order.append(key)

    models = list(model_inputs)
    denominator = len(models)
    output_rows: list[dict[str, object]] = []
    for key in order:
        present = [(model, indexed[model][key]) for model in models if key in indexed[model]]
        dispositions = Counter(decision_disposition(row.get("decision")) for _model, row in present)
        missing = denominator - len(present)
        dispositions["unavailable"] += missing
        exact_counts = Counter(str(row.get("decision", "")).strip().casefold() or "missing" for _model, row in present)
        if missing:
            exact_counts["missing"] += missing
        scores = {name: dispositions[name] / denominator for name in ("include", "exclude", "uncertain", "unavailable")}
        valid_scores = {name: scores[name] for name in ("include", "exclude", "uncertain")}
        maximum = max(valid_scores.values())
        winners = [name for name, score in valid_scores.items() if score == maximum and score >= threshold]
        if len(winners) == 1:
            disposition = winners[0]
            vote_status = "threshold_met"
        elif len(winners) > 1:
            disposition = "uncertain"
            vote_status = "tie"
        else:
            disposition = "uncertain"
            vote_status = "below_threshold"

        agreeing = [(model, row) for model, row in present if decision_disposition(row.get("decision")) == disposition]
        if disposition == "include":
            decision = _included_decision([row for _model, row in agreeing])
            exact = [(model, row) for model, row in agreeing if str(row.get("decision", "")).strip().casefold() == decision]
        elif disposition == "exclude":
            decision = "exclude"
            exact = agreeing
        else:
            decision = "uncertain"
            exact = agreeing
        representative_model, representative = (exact or agreeing or present)[0]
        combined: dict[str, object] = dict(representative)
        combined.update({
            "pipeline_version": "ensemble-1",
            "profile_hash": stable_hash(source, *models, threshold),
            "llm_model": "ensemble",
            "decision": decision,
            "classified_at": utc_now(),
            "error": "",
            "ensemble_models": json.dumps(models, ensure_ascii=False),
            "ensemble_model_decisions": json.dumps({model: str(indexed[model].get(key, {}).get("decision", "")).strip() or "missing" for model in models}, ensure_ascii=False, sort_keys=True),
            "ensemble_representative_model": representative_model,
            "ensemble_threshold": f"{threshold:.3f}",
            "ensemble_disposition": disposition,
            "ensemble_agreement": f"{maximum:.3f}",
            "ensemble_include_score": "{:.3f}".format(scores.get("include", 0.0)),
            "ensemble_exclude_score": "{:.3f}".format(scores.get("exclude", 0.0)),
            "ensemble_uncertain_score": "{:.3f}".format(scores.get("uncertain", 0.0)),
            "ensemble_unavailable_score": "{:.3f}".format(scores.get("unavailable", 0.0)),
            "ensemble_decision_counts": json.dumps(dict(sorted(exact_counts.items())), ensure_ascii=False, sort_keys=True),
            "ensemble_vote_status": vote_status,
        })
        output_rows.append(combined)
    atomic_write_csv(output, output_rows, columns)
    return output_rows
