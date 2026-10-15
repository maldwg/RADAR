from __future__ import annotations

from typing import Any

from radar.common.text import unicode_safe


MAX_SNOWBALL_DEPTH = 10
EVIDENCE_EXPANSION_COLUMNS = [
    "evidence_expansion_required", "evidence_expansion_attempted", "evidence_expansion_changed",
    "evidence_expansion_improved", "evidence_expansion_resolved",
    "evidence_initial_decision", "evidence_initial_vote_status", "evidence_initial_agreement",
    "evidence_initial_votes", "evidence_initial_uncertain_votes",
]


def snowball_depth(settings: dict[str, Any], default: int = MAX_SNOWBALL_DEPTH) -> int:
    """Return a safe configured backward-snowball depth bounded by the study protocol."""
    return max(0, min(MAX_SNOWBALL_DEPTH, int(settings.get("snowball_max_depth", default))))


def needs_more_evidence(result: dict[str, Any]) -> bool:
    """Return whether a normalized classification is unresolved and merits one evidence retry."""
    return str(result.get("decision", "")).casefold() == "uncertain"


def vote_round_needs_more_evidence(
    votes: dict[str, dict[str, Any]], result: dict[str, Any],
) -> bool:
    """Request enrichment for any uncertain vote or an unresolved aggregate vote."""
    any_uncertain = any(
        str(vote.get("decision", "")).strip().casefold() == "uncertain"
        for vote in votes.values()
    )
    unresolved_vote = str(result.get("ensemble_vote_status", "")).strip().casefold() in {
        "below_threshold", "tie",
    }
    return any_uncertain or unresolved_vote or needs_more_evidence(result)


def merge_new_evidence(target: dict[str, Any], discovered: dict[str, Any]) -> bool:
    """Merge non-empty evidence fields and report whether the usable evidence changed."""
    changed = False
    for key, value in discovered.items():
        value = unicode_safe(value)
        if value in (None, "", [], {}):
            continue
        if target.get(key) != value:
            target[key] = value
            changed = True
    return changed


def _round_summary(votes: dict[str, dict[str, Any]], result: dict[str, Any]) -> tuple[float, int]:
    """Return modal decision-group agreement and the number of uncertain votes."""
    groups = {"include": 0, "exclude": 0, "uncertain": 0, "unavailable": 0}
    for vote in votes.values():
        decision = str(vote.get("decision", "")).strip().casefold()
        group = "include" if decision in {"accept", "accept_adjacent"} else decision if decision in {"exclude", "uncertain"} else "unavailable"
        groups[group] += 1
    denominator = max(1, len(votes))
    agreement = max(groups["include"], groups["exclude"], groups["uncertain"]) / denominator
    try:
        agreement = float(result.get("ensemble_agreement", agreement))
    except (TypeError, ValueError):
        pass
    return agreement, groups["uncertain"]


def evidence_expansion_audit(
    initial_votes: dict[str, dict[str, Any]], initial_result: dict[str, Any],
    final_votes: dict[str, dict[str, Any]], final_result: dict[str, Any],
    *, required: bool, attempted: bool, changed: bool,
) -> dict[str, str]:
    """Build auditable escalation statistics from evidence-consistent vote rounds."""
    initial_agreement, initial_uncertain = _round_summary(initial_votes, initial_result)
    final_agreement, final_uncertain = _round_summary(final_votes, final_result)
    initial_decision = str(initial_result.get("decision", "")).strip().casefold()
    final_decision = str(final_result.get("decision", "")).strip().casefold()
    definitive = {"accept", "accept_adjacent", "exclude"}
    resolved = changed and initial_decision not in definitive and final_decision in definitive
    improved = changed and (
        resolved or final_agreement > initial_agreement or final_uncertain < initial_uncertain
    )
    return {
        "evidence_expansion_required": str(required).lower(),
        "evidence_expansion_attempted": str(attempted).lower(),
        "evidence_expansion_changed": str(changed).lower(),
        "evidence_expansion_improved": str(improved).lower(),
        "evidence_expansion_resolved": str(resolved).lower(),
        "evidence_initial_decision": initial_decision,
        "evidence_initial_vote_status": str(initial_result.get("ensemble_vote_status", "")),
        "evidence_initial_agreement": f"{initial_agreement:.3f}",
        "evidence_initial_votes": str(len(initial_votes)),
        "evidence_initial_uncertain_votes": str(initial_uncertain),
    }
