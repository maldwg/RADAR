from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.run import utc_now
from ids_meta.common.text import normalize_text, split_values, stable_hash


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



def _representative_quality(row: dict[str, str]) -> tuple[int, int, int]:
    """Rank coherent supporting runs by evidence, validation, and field completeness."""
    verified = int(_bool_text(row.get("evidence_quote_verified")) == "true")
    valid = int(not str(row.get("classification_validation_flags", "")).strip() and not str(row.get("error", "")).strip())
    populated = sum(bool(str(value).strip()) for key, value in row.items() if key not in {"primary_reason", "evidence_json", "model_evidence_json"})
    return verified, valid, populated


def _bool_text(value: object) -> str:
    """Normalize a boolean-like classification field to lowercase text."""
    return "true" if str(value or "").strip().casefold() in {"true", "1", "yes"} else "false"


def _split_labels(value: object) -> set[str]:
    """Split a semicolon-delimited normalized multi-label field."""
    return {item.strip() for item in str(value or "").split(";") if item.strip() and item.strip() != "unknown"}


def _artifact_identity_keys(row: dict[str, str]) -> set[str]:
    """Return punctuation-insensitive names and aliases for one classified resource."""
    values = [
        row.get("artifact_name", ""), row.get("artifact_display_name", ""),
        *split_values(row.get("artifact_aliases", "")),
    ]
    return {normalize_text(value).replace(" ", "") for value in values if len(normalize_text(value).replace(" ", "")) >= 5}


def _artifact_identity_audit(present: list[tuple[str, dict[str, str]]]) -> dict[str, Any]:
    """Report whether independently accepted model outputs identify a shared resource."""
    values = {model: sorted(_artifact_identity_keys(row)) for model, row in present}
    considered = [
        model for model, row in present
        if decision_disposition(row.get("decision")) == "include" and values[model]
    ]
    named = [set(values[model]) for model in considered]
    shared = set.intersection(*named) if named else set()
    return {
        "values": values, "considered_models": considered, "shared": sorted(shared),
        "status": "compatible" if len(named) < 2 or shared else "conflict",
    }


def _vote_scalar(
    votes: dict[str, dict[str, str]], field: str, threshold: float, fallback: str,
    normalizer: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """Threshold-vote one scalar field while retaining missing votes in the denominator."""
    normalize = normalizer or (lambda value: str(value or "").strip())
    values = {label: normalize(row.get(field)) if field in row else "" for label, row in votes.items()}
    counts = Counter(value if value else "missing" for value in values.values())
    eligible = {value: count for value, count in counts.items() if value != "missing"}
    maximum = max(eligible.values(), default=0)
    winners = sorted(value for value, count in eligible.items() if count == maximum and count / len(votes) >= threshold)
    selected = winners[0] if len(winners) == 1 else fallback
    status = "threshold_met" if len(winners) == 1 else "tie" if len(winners) > 1 else "below_threshold"
    return selected, {
        "selected": selected, "counts": dict(sorted(counts.items())),
        "agreement": maximum / len(votes), "status": status,
    }


def _vote_multilabel(
    votes: dict[str, dict[str, str]], field: str, threshold: float, fallback: str,
) -> tuple[str, dict[str, Any]]:
    """Vote each observed label independently and serialize the accepted label set."""
    values = {label: _split_labels(row.get(field)) if field in row else set() for label, row in votes.items()}
    labels = sorted(set().union(*values.values())) if values else []
    counts = {value: sum(value in row_values for row_values in values.values()) for value in labels}
    selected = [value for value in labels if counts[value] / len(votes) >= threshold]
    result = ";".join(selected) if selected else fallback
    return result, {
        "selected": selected if selected else [fallback] if fallback else [], "counts": counts,
        "agreement": {value: counts[value] / len(votes) for value in labels},
        "status": "threshold_met" if selected else "below_threshold",
    }


def _openalex_field_consensus(
    combined: dict[str, object], votes: dict[str, dict[str, str]], threshold: float,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Vote normalized OpenAlex taxonomy fields and return their audit details."""
    audit: dict[str, Any] = {}
    specs = {
        "paper_role": "uncertain", "relevance_tier": "uncertain", "artifact_type": "unknown",
        "domain": "unknown", "data_form": "unknown", "capture_environment": "unknown", "labeling": "unknown",
    }
    for field, fallback in specs.items():
        value, details = _vote_scalar(votes, field, threshold, fallback)
        combined[field], audit[field] = value, details
    for field in ("direct_release_evidence", "evidence_quote_verified"):
        value, details = _vote_scalar(votes, field, threshold, "false", _bool_text)
        combined[field], audit[field] = value, details
    audit["artifact_identity"] = _artifact_identity_audit(list(votes.items()))
    return combined, audit


def _github_field_consensus(
    combined: dict[str, object], votes: dict[str, dict[str, str]], threshold: float,
) -> tuple[dict[str, object], dict[str, Any]]:
    """Vote normalized GitHub taxonomy fields and return their audit details."""
    audit: dict[str, Any] = {}
    value, details = _vote_scalar(votes, "role", threshold, "uncertain")
    combined["role"], audit["role"] = value, details
    for field in ("independent_fork", "direct_self_description", "runtime_observation", "reproducible_execution", "offline_dataset_only", "implements_detection_engine", "requires_external_detector", "integrates_multiple_detectors", "cohesive_framework_architecture", "research_basis", "evidence_quote_verified"):
        value, details = _vote_scalar(votes, field, threshold, "false", _bool_text)
        combined[field], audit[field] = value, details
    for field, fallback in (("observation_scopes", "unknown"), ("capabilities", ""), ("detection_methods", "unknown")):
        value, details = _vote_multilabel(votes, field, threshold, fallback)
        combined[field], audit[field] = value, details
    value, details = _vote_scalar(votes, "upstream_full_name", threshold, "")
    combined["upstream_full_name"], audit["upstream_full_name"] = value, details
    return combined, audit


def _validate_openalex_consensus(row: dict[str, object], disposition: str, require_verified_quote: bool) -> tuple[str, list[str]]:
    """Check that voted OpenAlex fields form one internally consistent classification."""
    role, tier, artifact_type = str(row.get("paper_role", "")), str(row.get("relevance_tier", "")), str(row.get("artifact_type", ""))
    direct = _bool_text(row.get("direct_release_evidence")) == "true"
    quote = _bool_text(row.get("evidence_quote_verified")) == "true"
    name = str(row.get("artifact_name", "")).strip()
    accepted = {"original_release", "versioned_release", "generator_release"}
    excluded = {"survey", "downstream_application", "dataset_analysis", "unrelated"}
    artifact_ok = (
        role in {"original_release", "versioned_release"} and artifact_type in {"dataset", "dataset_and_generator"}
    ) or (role == "generator_release" and artifact_type in {"generator_framework", "dataset_and_generator"})
    flags: list[str] = []
    if role in accepted:
        if not artifact_ok:
            flags.append("role_artifact_type_mismatch")
        if tier not in {"core", "adjacent"} or (role == "generator_release" and tier == "adjacent"):
            flags.append("tier_not_accepted_for_role")
        if not direct:
            flags.append("direct_release_evidence_missing")
        if not name:
            flags.append("artifact_name_missing")
        if require_verified_quote and not quote:
            flags.append("verified_evidence_missing")
    if role in accepted and not flags:
        derived = "accept_adjacent" if tier == "adjacent" else "accept"
    elif role in excluded and (quote or not require_verified_quote):
        derived = "exclude"
    else:
        derived = "uncertain"
    if disposition != "uncertain" and decision_disposition(derived) != disposition:
        flags.append("decision_fields_conflict")
    return derived if not flags and disposition != "uncertain" else "uncertain", flags


def _validate_github_consensus(
    row: dict[str, object], disposition: str, system_type_profile: dict[str, Any] | None,
    require_verified_quote: bool,
) -> tuple[str, list[str]]:
    """Check GitHub field consistency and recompute all derived system-type fields."""
    from ids_meta.github.schema import ROLES, derive_system_types, system_type_names

    role = str(row.get("role", "uncertain"))
    self_description = _bool_text(row.get("direct_self_description")) == "true"
    runtime = _bool_text(row.get("runtime_observation")) == "true"
    runnable = _bool_text(row.get("reproducible_execution")) == "true"
    offline = _bool_text(row.get("offline_dataset_only")) == "true"
    implements_engine = _bool_text(row.get("implements_detection_engine")) == "true"
    requires_external = _bool_text(row.get("requires_external_detector")) == "true"
    integrates_multiple = _bool_text(row.get("integrates_multiple_detectors")) == "true"
    framework_architecture = _bool_text(row.get("cohesive_framework_architecture")) == "true"
    research_basis = _bool_text(row.get("research_basis")) == "true"
    independent = _bool_text(row.get("independent_fork")) == "true"
    quote = _bool_text(row.get("evidence_quote_verified")) == "true"
    is_fork = _bool_text(row.get("fork")) == "true"
    flags: list[str] = []
    if role == "native_detector":
        if not runtime:
            flags.append("runtime_observation_missing")
        if not runnable:
            flags.append("reproducible_execution_missing")
        if offline:
            flags.append("offline_dataset_only")
        if not implements_engine:
            flags.append("detection_engine_implementation_missing")
        if requires_external:
            flags.append("external_detector_required")
        if framework_architecture:
            flags.append("framework_architecture_conflict")
    elif role in {"classifier_engine", "detection_framework"}:
        if not runnable:
            flags.append("reproducible_execution_missing")
        if not implements_engine:
            flags.append("detection_engine_implementation_missing")
        if role == "detection_framework" and not framework_architecture:
            flags.append("framework_architecture_missing")
    elif role == "detection_platform":
        if not runnable:
            flags.append("reproducible_execution_missing")
        if not integrates_multiple:
            flags.append("multiple_detector_integration_missing")
    elif role == "research_project":
        if not runnable:
            flags.append("reproducible_execution_missing")
        if not research_basis:
            flags.append("research_basis_missing")
    if role in ROLES:
        if not self_description:
            flags.append("direct_self_description_missing")
        if is_fork and not independent:
            flags.append("independent_fork_evidence_missing")
        if require_verified_quote and not quote:
            flags.append("verified_evidence_missing")
    scopes = sorted(_split_labels(row.get("observation_scopes"))) or ["unknown"]
    capabilities = sorted(_split_labels(row.get("capabilities")))
    methods = sorted(_split_labels(row.get("detection_methods"))) or ["unknown"]
    codes = derive_system_types(scopes, capabilities, methods, role, system_type_profile)
    row.update({
        "primary_system_type": codes[0] if codes else "unknown",
        "system_types": ";".join(codes) if codes else "unknown",
        "system_type_names": ";".join(system_type_names(codes, system_type_profile)) if codes else "Unknown",
    })
    if role in ROLES and not flags:
        derived = "accept"
    elif role == "out_of_scope" and (quote or not require_verified_quote):
        derived = "exclude"
    else:
        derived = "uncertain"
    if disposition != "uncertain" and decision_disposition(derived) != disposition:
        flags.append("decision_fields_conflict")
    return derived if not flags and disposition != "uncertain" else "uncertain", flags


def consensus_classification(
    source: str, votes: dict[str, dict[str, str]], threshold: float,
    system_type_profile: dict[str, Any] | None = None, require_verified_quote: bool = True,
) -> dict[str, object]:
    """Combine consecutive decisions and normalized fields with conservative validation."""
    if source not in {"github", "openalex"} or not votes:
        raise ValueError("consensus requires a supported source and at least one vote")
    if not 0 < threshold <= 1:
        raise ValueError("consensus threshold must be greater than 0 and at most 1")
    labels = list(votes)
    denominator = len(labels)
    dispositions = Counter(decision_disposition(row.get("decision")) for row in votes.values())
    scores = {name: dispositions[name] / denominator for name in ("include", "exclude", "uncertain", "unavailable")}
    maximum = max(scores[name] for name in ("include", "exclude", "uncertain"))
    winners = [name for name in ("include", "exclude", "uncertain") if scores[name] == maximum and maximum >= threshold]
    disposition = winners[0] if len(winners) == 1 else "uncertain"
    status = "threshold_met" if len(winners) == 1 else "tie" if len(winners) > 1 else "below_threshold"
    agreeing = [(label, row) for label, row in votes.items() if decision_disposition(row.get("decision")) == disposition]
    if disposition == "include":
        exact_decision = _included_decision([row for _label, row in agreeing])
        exact = [(label, row) for label, row in agreeing if str(row.get("decision", "")).casefold() == exact_decision]
    elif disposition == "exclude":
        exact_decision, exact = "exclude", agreeing
    else:
        exact_decision, exact = "uncertain", agreeing
    representative_label, representative = max(exact or agreeing or list(votes.items()), key=lambda item: _representative_quality(item[1]))
    combined: dict[str, object] = dict(representative)
    structured = "paper_role" if source == "openalex" else "role"
    field_audit: dict[str, Any] = {}
    validation_flags: list[str] = []
    fallback_run = ""
    if any(structured in row for row in votes.values()):
        if source == "openalex":
            combined, field_audit = _openalex_field_consensus(combined, votes, threshold)
            final_decision, validation_flags = _validate_openalex_consensus(combined, disposition, require_verified_quote)
            artifact_identity_conflict = field_audit["artifact_identity"]["status"] == "conflict"
            if artifact_identity_conflict:
                final_decision = "uncertain"
                validation_flags.append("artifact_identity_consensus_conflict")
        else:
            combined, field_audit = _github_field_consensus(combined, votes, threshold)
            final_decision, validation_flags = _validate_github_consensus(combined, disposition, system_type_profile, require_verified_quote)
        if validation_flags and disposition in {"include", "exclude"} and not (source == "openalex" and artifact_identity_conflict):
            fallback = dict(representative)
            if source == "openalex":
                fallback_decision, fallback_flags = _validate_openalex_consensus(fallback, disposition, require_verified_quote)
            else:
                fallback_decision, fallback_flags = _validate_github_consensus(fallback, disposition, system_type_profile, require_verified_quote)
            if not fallback_flags and fallback_decision != "uncertain":
                combined, final_decision = fallback, fallback_decision
                fallback_run = representative_label
                validation_flags.append("field_consensus_fallback_to_representative")
        decision = final_decision
    else:
        decision = exact_decision
    combined.update({
        "pipeline_version": "consecutive-consensus-2", "llm_model": representative.get("llm_model", ""),
        "classifier_method": "llm", "classifier_id": representative.get("llm_model", ""),
        "classifier_profile_hash": representative.get("classifier_profile_hash", representative.get("profile_hash", "")),
        "classifier_supported_fields": "*",
        "decision": decision, "classified_at": utc_now(), "error": "",
        "ensemble_models": json.dumps(labels),
        "ensemble_model_decisions": json.dumps({label: row.get("decision", "") for label, row in votes.items()}, sort_keys=True),
        "ensemble_representative_model": representative_label, "ensemble_threshold": f"{threshold:.3f}",
        "ensemble_disposition": disposition, "ensemble_agreement": f"{maximum:.3f}",
        "ensemble_include_score": f"{scores['include']:.3f}", "ensemble_exclude_score": f"{scores['exclude']:.3f}",
        "ensemble_uncertain_score": f"{scores['uncertain']:.3f}", "ensemble_unavailable_score": f"{scores['unavailable']:.3f}",
        "ensemble_decision_counts": json.dumps(dict(Counter(row.get("decision", "") for row in votes.values())), sort_keys=True),
        "ensemble_vote_status": status,
        "consensus_field_audit": json.dumps(field_audit, sort_keys=True),
        "consensus_validation_flags": ";".join(dict.fromkeys(validation_flags)),
        "consensus_fallback_run": fallback_run,
    })
    return combined

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
        identity_audit: dict[str, Any] = {}
        identity_conflict = False
        if source == "openalex" and disposition == "include":
            identity_audit = _artifact_identity_audit(agreeing)
            identity_conflict = identity_audit["status"] == "conflict"
            if identity_conflict:
                decision = "uncertain"
                vote_status = "artifact_identity_conflict"
        representative_model, representative = (exact or agreeing or present)[0]
        combined: dict[str, object] = dict(representative)
        combined.update({
            "pipeline_version": "ensemble-2",
            "profile_hash": stable_hash(source, *models, threshold),
            "llm_model": "ensemble",
            "classifier_method": "ensemble",
            "classifier_id": "ensemble",
            "classifier_profile_hash": stable_hash(source, *models, threshold),
            "classifier_supported_fields": "*",
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
        if source == "openalex" and identity_audit:
            combined["consensus_field_audit"] = json.dumps({"artifact_identity": identity_audit}, ensure_ascii=False, sort_keys=True)
            if identity_conflict:
                flags = [value for value in str(combined.get("consensus_validation_flags", "")).split(";") if value]
                combined["consensus_validation_flags"] = ";".join(dict.fromkeys([*flags, "cross_model_artifact_identity_conflict"]))
        for field in (
            "evidence_expansion_required", "evidence_expansion_attempted", "evidence_expansion_changed",
            "evidence_expansion_improved", "evidence_expansion_resolved",
        ):
            combined[field] = str(any(_bool_text(row.get(field)) == "true" for _model, row in present)).lower()
        output_rows.append(combined)
    atomic_write_csv(output, output_rows, columns)
    return output_rows
