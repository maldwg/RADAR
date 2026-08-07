from __future__ import annotations

import json
from typing import Any

PAPER_ROLES = (
    "original_release", "versioned_release", "generator_release", "survey",
    "downstream_application", "dataset_analysis", "unrelated", "uncertain",
)
ACCEPTED_ROLES = ("original_release", "versioned_release", "generator_release")
TIERS = ("core", "adjacent", "excluded", "uncertain")
ARTIFACT_TYPES = ("dataset", "generator_framework", "dataset_and_generator", "unknown")
DOMAINS = ("general_network", "iot_iiot", "ics_scada", "automotive_can", "host", "wireless", "dns", "encrypted_vpn_tor", "web_application", "malware_botnet", "multi_domain", "unknown")
DATA_FORMS = ("packet", "flow", "network_traffic", "telemetry", "host_system_call", "log", "tabular_features", "can", "wireless", "mixed", "unknown")
ENVIRONMENTS = ("real_world", "synthetic_lab", "hybrid", "unknown")
LABELING = ("labeled", "partially_labeled", "unlabeled", "unknown")

METADATA_COLUMNS = [
    "schema_version", "run_id", "openalex_id", "doi", "title", "abstract", "authors", "venue",
    "publication_year", "publication_date", "work_type", "primary_url", "pdf_url", "is_oa",
    "cited_by_count", "citation_as_of", "metadata_fetched_at", "discovery_relations", "matched_queries", "evidence_hash",
]
CANDIDATE_COLUMNS = METADATA_COLUMNS + ["prefilter_score", "prefilter_band", "prefilter_signals"]
CLASSIFICATION_COLUMNS = CANDIDATE_COLUMNS + [
    "pipeline_version", "profile_hash", "llm_model", "decision", "paper_role", "relevance_tier",
    "artifact_id", "artifact_name", "artifact_aliases", "artifact_type", "domain", "data_form",
    "capture_environment", "labeling", "classification_confidence", "direct_release_evidence",
    "primary_reason", "evidence_json", "evidence_quote_verified", "classified_at", "error",
]
RECONCILIATION_COLUMNS = [
    "publication_family_id", "canonical_openalex_id", "canonical_doi", "canonical_title",
    "paper_relationship", "canonical_confidence", "canonical_reason", "canonical_resolution",
]
CATALOGUE_COLUMNS = CLASSIFICATION_COLUMNS + RECONCILIATION_COLUMNS

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "paper_role": {"type": "string", "enum": list(PAPER_ROLES)},
        "relevance_tier": {"type": "string", "enum": list(TIERS)},
        "artifact_name": {"type": "string"},
        "artifact_aliases": {"type": "array", "items": {"type": "string"}},
        "artifact_type": {"type": "string", "enum": list(ARTIFACT_TYPES)},
        "domain": {"type": "string", "enum": list(DOMAINS)},
        "data_form": {"type": "string", "enum": list(DATA_FORMS)},
        "capture_environment": {"type": "string", "enum": list(ENVIRONMENTS)},
        "labeling": {"type": "string", "enum": list(LABELING)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "direct_release_evidence": {"type": "boolean"},
        "primary_reason": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}
        }, "required": ["source", "quote"]}},
    },
    "required": ["paper_role", "relevance_tier", "artifact_name", "artifact_aliases", "artifact_type", "domain", "data_form", "capture_environment", "labeling", "confidence", "direct_release_evidence", "primary_reason", "evidence"],
}


def enum_value(value: Any, allowed: tuple[str, ...], default: str) -> str:
    """Return a permitted taxonomy value or the supplied fallback."""
    text = str(value or "").strip()
    return text if text in allowed else default


def normalize_response(raw: dict[str, Any], verified: list[dict[str, str]], config: dict[str, Any], artifact_id: str) -> dict[str, str]:
    """Normalize and adjudicate a raw OpenAlex classifier response."""
    role = enum_value(raw.get("paper_role"), PAPER_ROLES, "uncertain")
    tier = enum_value(raw.get("relevance_tier"), TIERS, "uncertain")
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0) or 0)))
    direct = bool(raw.get("direct_release_evidence"))
    artifact_name = str(raw.get("artifact_name", "")).strip()
    artifact_type = enum_value(raw.get("artifact_type"), ARTIFACT_TYPES, "unknown")
    role_artifact_ok = (
        role in ("original_release", "versioned_release") and artifact_type in ("dataset", "dataset_and_generator")
    ) or (role == "generator_release" and artifact_type in ("generator_framework", "dataset_and_generator"))
    quote_ok = bool(verified) or not bool(config.get("require_verified_quote", True))
    if role in ACCEPTED_ROLES and role_artifact_ok and tier == "core" and confidence >= float(config.get("accept_confidence", .8)) and direct and artifact_name and quote_ok:
        decision = "accept"
    elif role in ("original_release", "versioned_release") and role_artifact_ok and tier == "adjacent" and confidence >= float(config.get("accept_confidence", .8)) and direct and artifact_name and quote_ok:
        decision = "accept_adjacent"
    elif role in ("survey", "downstream_application", "dataset_analysis", "unrelated") and confidence >= float(config.get("exclude_confidence", .75)):
        decision = "exclude"
    else:
        decision = "uncertain"
    aliases = raw.get("artifact_aliases") if isinstance(raw.get("artifact_aliases"), list) else []
    return {
        "decision": decision, "paper_role": role, "relevance_tier": tier,
        "artifact_id": artifact_id, "artifact_name": artifact_name,
        "artifact_aliases": ";".join(dict.fromkeys(str(item).strip() for item in aliases if str(item).strip())),
        "artifact_type": artifact_type,
        "domain": enum_value(raw.get("domain"), DOMAINS, "unknown"),
        "data_form": enum_value(raw.get("data_form"), DATA_FORMS, "unknown"),
        "capture_environment": enum_value(raw.get("capture_environment"), ENVIRONMENTS, "unknown"),
        "labeling": enum_value(raw.get("labeling"), LABELING, "unknown"),
        "classification_confidence": f"{confidence:.3f}", "direct_release_evidence": str(direct).lower(),
        "primary_reason": str(raw.get("primary_reason", "")).strip()[:1000],
        "evidence_json": json.dumps(verified, ensure_ascii=False), "evidence_quote_verified": str(bool(verified)).lower(),
    }


def validate_processed_row(row: dict[str, str]) -> None:
    """Validate the required metadata and taxonomy of an accepted paper row."""
    required = ("artifact_id", "artifact_name", "openalex_id", "title", "publication_year", "paper_role", "run_id", "evidence_json", "publication_family_id", "canonical_openalex_id", "paper_relationship")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"processed OpenAlex row is missing {', '.join(missing)}: {row.get('openalex_id', '<unknown>')}")
    if row.get("paper_role") not in ACCEPTED_ROLES:
        raise ValueError(f"invalid accepted paper role: {row.get('paper_role')}")
