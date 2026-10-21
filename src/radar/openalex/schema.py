from __future__ import annotations

import json
from html import unescape
from typing import Any

from radar.common.baselines import CLASSIFIER_COLUMNS, KEYWORD_AUDIT_COLUMNS
from radar.common.nli import NLI_AUDIT_COLUMNS
from radar.common.evidence import EVIDENCE_EXPANSION_COLUMNS
from radar.common.paper_evidence import FULL_TEXT_AUDIT_FIELDS
from radar.common.text import normalize_text
PAPER_ROLES = (
    "original_release", "versioned_release", "generator_release", "survey",
    "downstream_application", "dataset_analysis", "unrelated", "uncertain",
)
ACCEPTED_ROLES = ("original_release", "versioned_release", "generator_release")
TIERS = ("core", "adjacent", "excluded", "uncertain")
ARTIFACT_TYPES = ("dataset", "generator_framework", "dataset_and_generator", "unknown")
DOMAINS = (
    "general_network", "host_endpoint", "iot_iiot", "cloud", "mobile_android",
    "electrical_smartgrid", "uav", "vpn", "unknown",
)
DATA_FORMS = ("packet", "flow", "network_traffic", "telemetry", "host_system_call", "log", "tabular_features", "can", "wireless", "mixed", "unknown")
ENVIRONMENTS = ("real_world", "synthetic_lab", "hybrid", "unknown")
LABELING = ("labeled", "partially_labeled", "unlabeled", "unknown")

METADATA_COLUMNS = [
    "schema_version", "run_id", "openalex_id", "doi", "title", "abstract", "authors", "venue", "publisher",
    "publication_year", "publication_date", "work_type", "primary_url", "pdf_url", "is_oa", "abstract_source", "full_text_url", *FULL_TEXT_AUDIT_FIELDS,
    "cited_by_count", "citation_as_of", "referenced_works_count", "metadata_fetched_at", "discovery_relations", "matched_queries", "evidence_hash",
    "retrieval_rank_score", "retrieval_score", "retrieval_title_score", "retrieval_signals", "survey_discovery_score", "snowball_depth",
]
CANDIDATE_COLUMNS = METADATA_COLUMNS + ["prefilter_score", "prefilter_band", "prefilter_signals"]
ENSEMBLE_COLUMNS = [
    "ensemble_models", "ensemble_model_decisions", "ensemble_representative_model", "ensemble_threshold", "ensemble_disposition",
    "ensemble_agreement", "ensemble_include_score", "ensemble_exclude_score",
    "ensemble_uncertain_score", "ensemble_unavailable_score", "ensemble_decision_counts",
    "ensemble_vote_status", "consensus_field_audit", "consensus_validation_flags", "consensus_fallback_run",
]
SURVEY_COLUMNS = METADATA_COLUMNS
CLASSIFICATION_COLUMNS = CANDIDATE_COLUMNS + CLASSIFIER_COLUMNS + [
    "pipeline_version", "profile_hash", "llm_model", "decision", "paper_role", "relevance_tier",
    "artifact_id", "artifact_name", "artifact_display_name", "resource_name", "artifact_aliases", "artifact_type", "domain", "data_form",
    "capture_environment", "labeling", "direct_release_evidence",
    "primary_reason", "classification_validation_flags", "model_evidence_json", "evidence_json",
    "evidence_quote_verified", "classified_at", "error",
] + ENSEMBLE_COLUMNS + EVIDENCE_EXPANSION_COLUMNS + KEYWORD_AUDIT_COLUMNS + NLI_AUDIT_COLUMNS
RECONCILIATION_COLUMNS = [
    "publication_family_id", "canonical_openalex_id", "canonical_doi", "canonical_title",
    "paper_relationship", "canonical_confidence", "canonical_reason", "canonical_resolution",
]
CATALOGUE_COLUMNS = CLASSIFICATION_COLUMNS + RECONCILIATION_COLUMNS

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "paper_role": {"type": "string", "enum": list(PAPER_ROLES)},
        "relevance_tier": {"type": "string", "enum": list(TIERS)},
        "artifact_name": {"type": "string"},
        "artifact_display_name": {"type": "string", "maxLength": 80},
        "artifact_aliases": {"type": "array", "items": {"type": "string"}},
        "artifact_type": {"type": "string", "enum": list(ARTIFACT_TYPES)},
        "domain": {"type": "string", "enum": list(DOMAINS)},
        "data_form": {"type": "string", "enum": list(DATA_FORMS)},
        "capture_environment": {"type": "string", "enum": list(ENVIRONMENTS)},
        "labeling": {"type": "string", "enum": list(LABELING)},
        "direct_release_evidence": {"type": "boolean"},
        "primary_reason": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "source": {"type": "string", "enum": ["OpenAlex metadata", "Abstract", "Full text"]}, "quote": {"type": "string"}
        }, "required": ["source", "quote"]}},
    },
    "required": ["paper_role", "relevance_tier", "artifact_name", "artifact_display_name", "artifact_aliases", "artifact_type", "domain", "data_form", "capture_environment", "labeling", "direct_release_evidence", "primary_reason", "evidence"],
}


def enum_value(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """Return a permitted taxonomy value or the supplied fallback."""
    text = str(value or "").strip()
    return text if text in allowed else default


def ensure_resource_name(row: dict[str, str]) -> str:
    """Expose the current schema's explicit display name without reconstructing it."""
    if "artifact_display_name" not in row:
        raise ValueError("paper row requires artifact_display_name")
    display = str(row["artifact_display_name"]).strip()
    if row.get("decision") in {"accept", "accept_adjacent"} and not display:
        raise ValueError("accepted paper requires an explicit artifact_display_name")
    row["resource_name"] = display
    return display


def normalize_response(raw: dict[str, Any], verified: list[dict[str, Any]], config: dict[str, Any], artifact_id: str) -> dict[str, str]:
    """Normalize and adjudicate a raw OpenAlex classifier response."""
    role = enum_value(raw.get("paper_role"), PAPER_ROLES, "uncertain")
    tier = enum_value(raw.get("relevance_tier"), TIERS, "uncertain")
    direct = bool(raw.get("direct_release_evidence"))
    artifact_name = str(raw.get("artifact_name", "")).strip()
    artifact_display_name = str(raw.get("artifact_display_name", "")).strip()[:80]
    aliases = raw.get("artifact_aliases") if isinstance(raw.get("artifact_aliases"), list) else []
    identity_values = [artifact_name, artifact_display_name, *(str(value).strip() for value in aliases)]
    identity_evidence = normalize_text(*(unescape(item.get("quote", "")) for item in verified))
    compact_evidence = identity_evidence.replace(" ", "")
    identity_ok = any(
        len(normalize_text(value).replace(" ", "")) >= 5
        and normalize_text(value).replace(" ", "") in compact_evidence
        for value in identity_values if value
    )
    artifact_type = enum_value(raw.get("artifact_type"), ARTIFACT_TYPES, "unknown")
    role_artifact_ok = (
        role in ("original_release", "versioned_release") and artifact_type in ("dataset", "dataset_and_generator")
    ) or (role == "generator_release" and artifact_type in ("generator_framework", "dataset_and_generator"))
    quote_ok = bool(verified) or not bool(config.get("require_verified_quote", True))
    validation_flags: list[str] = []
    if role in ACCEPTED_ROLES:
        if not role_artifact_ok:
            validation_flags.append("role_artifact_type_mismatch")
        if tier not in ("core", "adjacent") or (tier == "adjacent" and role == "generator_release"):
            validation_flags.append("tier_not_accepted_for_role")
        if not direct:
            validation_flags.append("direct_release_evidence_missing")
        if not artifact_name:
            validation_flags.append("artifact_name_missing")
        elif not identity_ok:
            validation_flags.append("artifact_identity_evidence_missing")
        if not quote_ok:
            validation_flags.append("verified_evidence_missing")
    elif role in ("survey", "downstream_application", "dataset_analysis", "unrelated") and not quote_ok:
        validation_flags.append("verified_evidence_missing")
    elif role == "uncertain":
        validation_flags.append("model_role_uncertain")
    if role in ACCEPTED_ROLES and role_artifact_ok and tier == "core" and direct and artifact_name and identity_ok and quote_ok:
        decision = "accept"
    elif role in ("original_release", "versioned_release") and role_artifact_ok and tier == "adjacent" and direct and artifact_name and identity_ok and quote_ok:
        decision = "accept_adjacent"
    elif role in ("survey", "downstream_application", "dataset_analysis", "unrelated") and quote_ok:
        decision = "exclude"
    else:
        decision = "uncertain"
    result = {
        "decision": decision, "paper_role": role, "relevance_tier": tier,
        "artifact_id": artifact_id, "artifact_name": artifact_name,
        "artifact_display_name": artifact_display_name,
        "artifact_aliases": ";".join(dict.fromkeys(str(item).strip() for item in aliases if str(item).strip())),
        "artifact_type": artifact_type,
        "domain": enum_value(raw.get("domain"), DOMAINS, "unknown"),
        "data_form": enum_value(raw.get("data_form"), DATA_FORMS, "unknown"),
        "capture_environment": enum_value(raw.get("capture_environment"), ENVIRONMENTS, "unknown"),
        "labeling": enum_value(raw.get("labeling"), LABELING, "unknown"),
        "direct_release_evidence": str(direct).lower(),
        "primary_reason": str(raw.get("primary_reason", "")).strip()[:1000],
        "classification_validation_flags": ";".join(validation_flags),
        "model_evidence_json": json.dumps(raw.get("evidence") if isinstance(raw.get("evidence"), list) else [], ensure_ascii=False),
        "evidence_json": json.dumps(verified, ensure_ascii=False), "evidence_quote_verified": str(bool(verified)).lower(),
    }
    ensure_resource_name(result)
    return result


def validate_processed_row(row: dict[str, str]) -> None:
    """Validate the required metadata and taxonomy of an accepted paper row."""
    required = ("artifact_id", "artifact_name", "artifact_display_name", "resource_name", "openalex_id", "title", "publication_year", "paper_role", "run_id", "evidence_json", "publication_family_id", "canonical_openalex_id", "paper_relationship")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"processed OpenAlex row is missing {', '.join(missing)}: {row.get('openalex_id', '<unknown>')}")
    if row.get("paper_role") not in ACCEPTED_ROLES:
        raise ValueError(f"invalid accepted paper role: {row.get('paper_role')}")
