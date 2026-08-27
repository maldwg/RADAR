from __future__ import annotations

import json
from typing import Any

from ids_meta.common.evidence import EVIDENCE_EXPANSION_COLUMNS
from ids_meta.common.text import normalize_text, split_values

ROLES = (
    "standalone_detector", "platform_framework", "classifier_model",
    "rules_content", "support_tooling", "research_demo",
)
AUDIT_ROLES = ROLES + ("out_of_scope", "uncertain")
SCOPES = ("network", "host_endpoint", "wireless", "database", "provenance", "application", "multi_scope", "unknown")
CAPABILITIES = ("detection", "prevention", "response", "monitoring")
METHODS = (
    "signature_based", "rule_based", "anomaly_based", "integrity_based", "query_based",
    "protocol_analysis", "ml_based", "ann_based", "dnn_based", "packet_analysis", "flow_analysis",
    "log_analysis", "system_call_analysis", "threat_intelligence_based", "hybrid", "unknown",
)

DEFAULT_SYSTEM_TYPE_PROFILE: dict[str, Any] = {"types": {
    "NIDS": {"name": "Network intrusion detection system", "priority": 100, "any_scopes": ["network"], "all_capabilities": ["detection"]},
    "NIPS": {"name": "Network intrusion prevention system", "priority": 99, "any_scopes": ["network"], "all_capabilities": ["prevention"]},
    "NDR": {"name": "Network detection and response", "priority": 98, "any_scopes": ["network"], "all_capabilities": ["detection", "response"]},
    "HIDS": {"name": "Host intrusion detection system", "priority": 100, "any_scopes": ["host_endpoint"], "all_capabilities": ["detection"]},
    "HIPS": {"name": "Host intrusion prevention system", "priority": 99, "any_scopes": ["host_endpoint"], "all_capabilities": ["prevention"]},
    "EDR": {"name": "Endpoint detection and response", "priority": 98, "any_scopes": ["host_endpoint"], "all_capabilities": ["detection", "response"]},
    "WIDS": {"name": "Wireless intrusion detection system", "priority": 100, "any_scopes": ["wireless"], "all_capabilities": ["detection"]},
    "WIPS": {"name": "Wireless intrusion prevention system", "priority": 99, "any_scopes": ["wireless"], "all_capabilities": ["prevention"]},
    "DBIDS": {"name": "Database intrusion detection system", "priority": 100, "any_scopes": ["database"], "all_capabilities": ["detection"]},
    "DBIPS": {"name": "Database intrusion prevention system", "priority": 99, "any_scopes": ["database"], "all_capabilities": ["prevention"]},
    "APIDS": {"name": "Application-protocol intrusion detection system", "priority": 100, "any_scopes": ["application"], "all_capabilities": ["detection"]},
    "PIDS": {"name": "Protocol-based intrusion detection system", "priority": 110, "any_methods": ["protocol_analysis"], "all_capabilities": ["detection"]},
    "ProvIDS": {"name": "Provenance-based intrusion detection system", "priority": 100, "any_scopes": ["provenance"], "all_capabilities": ["detection"]},
    "XDR": {"name": "Extended detection and response", "priority": 105, "any_scopes": ["multi_scope"], "all_capabilities": ["detection", "response"]},
    "IDS": {"name": "Intrusion detection system", "priority": 10, "all_capabilities": ["detection"], "fallback_only": True},
    "IPS": {"name": "Intrusion prevention system", "priority": 9, "all_capabilities": ["prevention"], "fallback_only": True},
}}

METADATA_COLUMNS = [
    "schema_version", "run_id", "github_repo_id", "node_id", "full_name", "owner", "name",
    "html_url", "description", "topics", "homepage", "license_spdx", "language", "fork",
    "parent_full_name", "archived", "stars", "forks", "open_issues", "created_at", "updated_at",
    "pushed_at", "default_branch", "last_commit_sha", "last_commit_at", "last_commit_url",
    "metadata_fetched_at", "evidence_hash",
]
CANDIDATE_COLUMNS = METADATA_COLUMNS + ["prefilter_score", "prefilter_band", "prefilter_signals", "matched_queries"]
ENSEMBLE_COLUMNS = [
    "ensemble_models", "ensemble_model_decisions", "ensemble_representative_model", "ensemble_threshold", "ensemble_disposition",
    "ensemble_agreement", "ensemble_include_score", "ensemble_exclude_score",
    "ensemble_uncertain_score", "ensemble_unavailable_score", "ensemble_decision_counts",
    "ensemble_vote_status", "consensus_field_audit", "consensus_validation_flags", "consensus_fallback_run",
]
CLASSIFICATION_COLUMNS = CANDIDATE_COLUMNS + [
    "pipeline_version", "profile_hash", "llm_model", "decision", "role",
    "observation_scopes", "capabilities", "detection_methods", "primary_system_type",
    "system_types", "system_type_names", "derived_labels",
    "independent_fork", "direct_self_description", "runtime_observation", "runnable_detector",
    "offline_dataset_only", "upstream_full_name", "classification_validation_flags", "primary_reason", "evidence_json",
    "evidence_quote_verified", "classified_at", "error",
] + ENSEMBLE_COLUMNS + EVIDENCE_EXPANSION_COLUMNS
RECONCILIATION_COLUMNS = [
    "project_id", "canonical_repo_id", "canonical_full_name", "repository_relationship",
    "canonical_confidence", "canonical_reason", "canonical_resolution",
    "canonical_score", "canonical_margin", "canonical_score_breakdown",
]
CATALOGUE_COLUMNS = CLASSIFICATION_COLUMNS + RECONCILIATION_COLUMNS

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "role": {"type": "string", "enum": list(AUDIT_ROLES)},
        "observation_scopes": {"type": "array", "items": {"type": "string", "enum": list(SCOPES)}},
        "capabilities": {"type": "array", "items": {"type": "string", "enum": list(CAPABILITIES)}},
        "detection_methods": {"type": "array", "items": {"type": "string", "enum": list(METHODS)}},
        "independent_fork": {"type": "boolean"},
        "direct_self_description": {"type": "boolean"},
        "runtime_observation": {"type": "boolean"},
        "runnable_detector": {"type": "boolean"},
        "offline_dataset_only": {"type": "boolean"},
        "upstream_full_name": {"type": "string"},
        "primary_reason": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}
        }, "required": ["source", "quote"]}},
    },
    "required": ["role", "observation_scopes", "capabilities", "detection_methods", "independent_fork", "direct_self_description", "runtime_observation", "runnable_detector", "offline_dataset_only", "upstream_full_name", "primary_reason", "evidence"],
}


def _allowed(values: Any, allowed: tuple[str, ...], default: str | None = None) -> list[str]:
    """Filter model-provided list values to an allowed normalized vocabulary."""
    result = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip() in allowed)) if isinstance(values, list) else []
    return result or ([default] if default else [])


def _matches_system_type(
    rule: dict[str, Any], scopes: set[str], capabilities: set[str], methods: set[str], role: str,
) -> bool:
    """Return whether normalized repository dimensions satisfy one configured type rule."""
    dimensions = {"scopes": scopes, "capabilities": capabilities, "methods": methods}
    for name, values in dimensions.items():
        any_values = {str(value) for value in rule.get(f"any_{name}", [])}
        all_values = {str(value) for value in rule.get(f"all_{name}", [])}
        if any_values and not any_values.intersection(values):
            return False
        if all_values and not all_values.issubset(values):
            return False
    roles = {str(value) for value in rule.get("any_roles", [])}
    return not roles or role in roles


def validate_system_type_profile(profile: dict[str, Any]) -> None:
    """Reject type rules that reference unsupported normalized dimension values."""
    types = profile.get("types", {})
    if not isinstance(types, dict) or not types:
        raise ValueError("system-type profile requires a non-empty types table")
    dimensions = {
        "scopes": set(SCOPES), "capabilities": set(CAPABILITIES),
        "methods": set(METHODS), "roles": set(AUDIT_ROLES),
    }
    for code, value in types.items():
        if not isinstance(value, dict):
            raise ValueError(f"system type {code!r} must be a table")
        for dimension, allowed in dimensions.items():
            selectors = (f"any_{dimension}",) if dimension == "roles" else (f"any_{dimension}", f"all_{dimension}")
            for selector in selectors:
                invalid = {str(item) for item in value.get(selector, [])} - allowed
                if invalid:
                    raise ValueError(f"system type {code!r} has invalid {selector}: {sorted(invalid)}")
        try:
            float(value.get("priority", 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"system type {code!r}.priority must be numeric") from exc


def derive_system_types(
    scopes: list[str], capabilities: list[str], methods: list[str] | None = None,
    role: str = "", profile: dict[str, Any] | None = None,
) -> list[str]:
    """Derive an ordered, extensible system-type list from configured dimension rules."""
    active_profile = profile or DEFAULT_SYSTEM_TYPE_PROFILE
    validate_system_type_profile(active_profile)
    types = active_profile.get("types", {})
    if not isinstance(types, dict):
        return []
    ordered = sorted(
        ((str(code), rule) for code, rule in types.items() if isinstance(rule, dict)),
        key=lambda item: (-float(item[1].get("priority", 0) or 0), item[0].casefold()),
    )
    scope_values, capability_values = set(scopes), set(capabilities)
    method_values = set(methods or [])
    specific = [
        code for code, rule in ordered
        if not bool(rule.get("fallback_only")) and _matches_system_type(rule, scope_values, capability_values, method_values, role)
    ]
    if specific:
        return specific
    return [
        code for code, rule in ordered
        if bool(rule.get("fallback_only")) and _matches_system_type(rule, scope_values, capability_values, method_values, role)
    ]


def system_type_names(codes: list[str], profile: dict[str, Any] | None = None) -> list[str]:
    """Resolve configured human-readable names for ordered system-type codes."""
    types = (profile or DEFAULT_SYSTEM_TYPE_PROFILE).get("types", {})
    return [str(types.get(code, {}).get("name", code)) for code in codes]


def derive_labels(scopes: list[str], capabilities: list[str]) -> list[str]:
    """Preserve the legacy derived-label interface using the default system taxonomy."""
    return derive_system_types(scopes, capabilities)


def normalize_response(
    raw: dict[str, Any], verified: list[dict[str, str]], config: dict[str, Any], is_fork: bool,
    system_type_profile: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Validate a repository model response and derive its auditable decision fields."""
    role = str(raw.get("role", "uncertain"))
    role = role if role in AUDIT_ROLES else "uncertain"
    scopes = _allowed(raw.get("observation_scopes", []), SCOPES, "unknown")
    capabilities = _allowed(raw.get("capabilities", []), CAPABILITIES)
    methods = _allowed(raw.get("detection_methods", []), METHODS, "unknown")
    type_codes = derive_system_types(scopes, capabilities, methods, role, system_type_profile)
    independent = bool(raw.get("independent_fork"))
    self_description = bool(raw.get("direct_self_description"))
    runtime_observation = bool(raw.get("runtime_observation"))
    runnable_detector = bool(raw.get("runnable_detector"))
    offline_dataset_only = bool(raw.get("offline_dataset_only"))
    upstream_full_name = str(raw.get("upstream_full_name", "")).strip().strip("/")
    verified_text = normalize_text(" ".join(item.get("quote", "") for item in verified))
    if upstream_full_name and (upstream_full_name.count("/") != 1 or normalize_text(upstream_full_name) not in verified_text):
        upstream_full_name = ""
    standalone_flags: list[str] = []
    if role == "standalone_detector":
        if not runtime_observation:
            standalone_flags.append("runtime_observation_missing")
        if not runnable_detector:
            standalone_flags.append("runnable_detector_missing")
        if offline_dataset_only:
            standalone_flags.append("offline_dataset_only")
    standalone_ok = not standalone_flags
    require_quote = bool(config.get("require_verified_quote", True))
    evidence_ok = bool(verified) or not require_quote
    if role in ROLES and self_description and evidence_ok and (not is_fork or independent) and standalone_ok:
        decision = "accept"
    elif role == "out_of_scope" and evidence_ok:
        decision = "exclude"
    else:
        decision = "uncertain"
    return {
        "decision": decision,
        "role": role,
        "observation_scopes": ";".join(scopes),
        "capabilities": ";".join(capabilities),
        "detection_methods": ";".join(methods),
        "primary_system_type": type_codes[0] if type_codes else "unknown",
        "system_types": ";".join(type_codes) if type_codes else "unknown",
        "system_type_names": ";".join(system_type_names(type_codes, system_type_profile)) if type_codes else "Unknown",
        "derived_labels": ";".join(type_codes),
        "independent_fork": str(independent).lower(),
        "direct_self_description": str(self_description).lower(),
        "runtime_observation": str(runtime_observation).lower(),
        "runnable_detector": str(runnable_detector).lower(),
        "offline_dataset_only": str(offline_dataset_only).lower(),
        "upstream_full_name": upstream_full_name,
        "classification_validation_flags": ";".join(standalone_flags),
        "primary_reason": str(raw.get("primary_reason", "")).strip()[:1000],
        "evidence_json": json.dumps(verified, ensure_ascii=False),
        "evidence_quote_verified": str(bool(verified)).lower(),
    }


def validate_processed_row(row: dict[str, str], system_type_profile: dict[str, Any] | None = None) -> None:
    """Validate required metadata and taxonomy fields in an accepted repository row."""
    required = ("github_repo_id", "html_url", "role", "run_id", "evidence_json", "project_id", "repository_relationship", "primary_system_type", "system_types")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"processed GitHub row is missing {', '.join(missing)}: {row.get('full_name', '<unknown>')}")
    if row.get("role") not in ROLES:
        raise ValueError(f"processed GitHub row has invalid role: {row.get('role')}")
    for field, allowed in (("observation_scopes", SCOPES), ("capabilities", CAPABILITIES), ("detection_methods", METHODS)):
        invalid = set(split_values(row.get(field))) - set(allowed)
        if invalid:
            raise ValueError(f"invalid {field}: {sorted(invalid)}")
    configured = set((system_type_profile or DEFAULT_SYSTEM_TYPE_PROFILE).get("types", {})) | {"unknown"}
    invalid_types = set(split_values(row.get("system_types"))) - configured
    if invalid_types:
        raise ValueError(f"invalid system_types: {sorted(invalid_types)}")
