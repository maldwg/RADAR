from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ids_meta.common.text import matching_concepts


@dataclass(frozen=True)
class ScoreResult:
    score: float
    band: str
    signals: tuple[str, ...]


def score_evidence(text: str, profile: dict[str, Any], candidate_threshold: float, borderline_threshold: float) -> ScoreResult:
    """Score grouped concept matches and assign a candidate, borderline, or reject band."""
    score = 0.0
    signals: list[str] = []
    for section in ("positive", "negative"):
        groups = profile.get(section, {})
        if not isinstance(groups, dict):
            continue
        for name, group in groups.items():
            if not isinstance(group, dict):
                continue
            hits = matching_concepts(text, group.get("terms", []))
            if not hits:
                continue
            weight = float(group.get("weight", 0.0))
            score += weight
            signals.append(f"{name}:{'|'.join(hits)}:{weight:+g}")
    band = "candidate" if score >= candidate_threshold else "borderline" if score >= borderline_threshold else "reject"
    return ScoreResult(round(score, 3), band, tuple(signals))


def shortlist_candidates(
    rows: list[dict[str, str]], id_column: str, limit: int | None, secondary_score_column: str | None = None,
) -> list[dict[str, str]]:
    """Rank eligible candidates deterministically and return the requested prefix."""
    band_order = {"candidate": 0, "borderline": 1}

    def ranking_key(row: dict[str, str]) -> tuple[int, float, float, str]:
        """Order bands and scores first, leaving the stable ID as the final tie-break."""
        try:
            score = float(row.get("prefilter_score", ""))
        except (TypeError, ValueError):
            score = float("-inf")
        try:
            secondary = float(row.get(secondary_score_column, "")) if secondary_score_column else float("-inf")
        except (TypeError, ValueError):
            secondary = float("-inf")
        return band_order.get(row.get("prefilter_band", ""), 2), -score, -secondary, row.get(id_column, "")

    eligible = [row for row in rows if row.get("prefilter_band") in band_order]
    ranked = sorted(eligible, key=ranking_key)
    return ranked if limit is None else ranked[:max(0, limit)]
