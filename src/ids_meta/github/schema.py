from __future__ import annotations

import json
from typing import Any

from ids_meta.common.text import split_values

ROLES = (
    "standalone_detector", "platform_framework", "classifier_model",
    "rules_content", "support_tooling", "research_demo",
)
AUDIT_ROLES = ROLES + ("out_of_scope", "uncertain")
SCOPES = ("network", "host_endpoint", "wireless", "database", "provenance", "application", "multi_scope", "unknown")
CAPABILITIES = ("detection", "prevention", "response", "monitoring")
METHODS = (
    "signature_based", "rule_based", "anomaly_based", "integrity_based", "query_based",
    "ml_based", "ann_based", "dnn_based", "packet_analysis", "flow_analysis",
    "log_analysis", "system_call_analysis", "threat_intelligence_based", "hybrid", "unknown",
)

METADATA_COLUMNS = [
    "schema_version", "run_id", "github_repo_id", "node_id", "full_name", "owner", "name",
    "html_url", "description", "topics", "homepage", "license_spdx", "language", "fork",
    "parent_full_name", "archived", "stars", "forks", "open_issues", "created_at", "updated_at",
    "pushed_at", "default_branch", "last_commit_sha", "last_commit_at", "last_commit_url",
    "metadata_fetched_at", "evidence_hash",
]
CANDIDATE_COLUMNS = METADATA_COLUMNS + ["prefilter_score", "prefilter_band", "prefilter_signals", "matched_queries"]
CLASSIFICATION_COLUMNS = CANDIDATE_COLUMNS + [
    "pipeline_version", "profile_hash", "llm_model", "decision", "role", "role_confidence",
    "observation_scopes", "capabilities", "detection_methods", "derived_labels",
    "independent_fork", "direct_self_description", "primary_reason", "evidence_json",
    "evidence_quote_verified", "classified_at", "error",
]
RECONCILIATION_COLUMNS = [
    "project_id", "canonical_repo_id", "canonical_full_name", "repository_relationship",
    "canonical_confidence", "canonical_reason", "canonical_resolution",
]
CATALOGUE_COLUMNS = CLASSIFICATION_COLUMNS + RECONCILIATION_COLUMNS

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": list(AUDIT_ROLES)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "observation_scopes": {"type": "array", "items": {"type": "string", "enum": list(SCOPES)}},
        "capabilities": {"type": "array", "items": {"type": "string", "enum": list(CAPABILITIES)}},
        "detection_methods": {"type": "array", "items": {"type": "string", "enum": list(METHODS)}},
        "independent_fork": {"type": "boolean"},
        "direct_self_description": {"type": "boolean"},
        "primary_reason": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}
        }, "required": ["source", "quote"]}},
    },
    "required": ["role", "confidence", "observation_scopes", "capabilities", "detection_methods", "independent_fork", "direct_self_description", "primary_reason", "evidence"],
}


def _allowed(values: Any, allowed: tuple[str, ...], default: str | None = None) -> list[str]:
    """Filter model-provided list values to an allowed normalized vocabulary."""
    result = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip() in allowed)) if isinstance(values, list) else []
    return result or ([default] if default else [])


def derive_labels(scopes: list[str], capabilities: list[str]) -> list[str]:
    """Derive familiar IDS/IPS/DR labels from normalized scope and capability dimensions."""
    values: list[str] = []
    detected, prevents, responds = "detection" in capabilities, "prevention" in capabilities, "response" in capabilities
    for scope, detection_label, prevention_label, response_label in (
        ("network", "NIDS", "NIPS", "NDR"),
        ("host_endpoint", "HIDS", "HIPS", "EDR"),
        ("wireless", "WIDS", "WIPS", ""),
        ("database", "DBIDS", "DBIPS", ""),
        ("provenance", "PIDS", "", ""),
    ):
        if scope not in scopes:
            continue
        if detected and detection_label:
            values.append(detection_label)
        if prevents and prevention_label:
            values.append(prevention_label)
        if responds and response_label:
            values.append(response_label)
    if "multi_scope" in scopes and responds:
        values.append("XDR")
    if not values and detected:
        values.append("IDS")
    if not values and prevents:
        values.append("IPS")
    return list(dict.fromkeys(values))


def normalize_response(raw: dict[str, Any], verified: list[dict[str, str]], config: dict[str, Any], is_fork: bool) -> dict[str, str]:
    """Validate a repository model response and derive its auditable decision fields."""
    role = str(raw.get("role", "uncertain"))
    role = role if role in AUDIT_ROLES else "uncertain"
    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0) or 0)))
    scopes = _allowed(raw.get("observation_scopes", []), SCOPES, "unknown")
    capabilities = _allowed(raw.get("capabilities", []), CAPABILITIES)
    methods = _allowed(raw.get("detection_methods", []), METHODS, "unknown")
    independent = bool(raw.get("independent_fork"))
    self_description = bool(raw.get("direct_self_description"))
    require_quote = bool(config.get("require_verified_quote", True))
    evidence_ok = bool(verified) or not require_quote
    if role in ROLES and confidence >= float(config.get("accept_confidence", .8)) and self_description and evidence_ok and (not is_fork or independent):
        decision = "accept"
    elif role == "out_of_scope" and confidence >= float(config.get("exclude_confidence", .75)):
        decision = "exclude"
    else:
        decision = "uncertain"
    return {
        "decision": decision,
        "role": role,
        "role_confidence": f"{confidence:.3f}",
        "observation_scopes": ";".join(scopes),
        "capabilities": ";".join(capabilities),
        "detection_methods": ";".join(methods),
        "derived_labels": ";".join(derive_labels(scopes, capabilities)),
        "independent_fork": str(independent).lower(),
        "direct_self_description": str(self_description).lower(),
        "primary_reason": str(raw.get("primary_reason", "")).strip()[:1000],
        "evidence_json": json.dumps(verified, ensure_ascii=False),
        "evidence_quote_verified": str(bool(verified)).lower(),
    }


def validate_processed_row(row: dict[str, str]) -> None:
    """Validate required metadata and taxonomy fields in an accepted repository row."""
    required = ("github_repo_id", "html_url", "role", "run_id", "evidence_json", "project_id", "canonical_repo_id", "repository_relationship")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"processed GitHub row is missing {', '.join(missing)}: {row.get('full_name', '<unknown>')}")
    if row.get("role") not in ROLES:
        raise ValueError(f"processed GitHub row has invalid role: {row.get('role')}")
    for field, allowed in (("observation_scopes", SCOPES), ("capabilities", CAPABILITIES), ("detection_methods", METHODS)):
        invalid = set(split_values(row.get(field))) - set(allowed)
        if invalid:
            raise ValueError(f"invalid {field}: {sorted(invalid)}")
