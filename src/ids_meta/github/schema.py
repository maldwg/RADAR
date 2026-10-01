from __future__ import annotations

import json
from typing import Any

from ids_meta.common.baselines import CLASSIFIER_COLUMNS, KEYWORD_AUDIT_COLUMNS
from ids_meta.common.nli import NLI_AUDIT_COLUMNS
from ids_meta.common.evidence import EVIDENCE_EXPANSION_COLUMNS
from ids_meta.common.text import normalize_text, split_values

ROLES = (
    "native_detector", "classifier_engine", "detection_framework", "detection_platform",
    "research_project", "demo", "rules_content", "support_tooling",
)
AUDIT_ROLES = ROLES + ("out_of_scope", "uncertain")
SCOPES = ("network", "host_endpoint", "wireless", "database", "provenance", "application", "multi_scope", "unknown")
CAPABILITIES = ("detection", "prevention", "monitoring")
METHODS = (
    "signature_based", "rule_based", "anomaly_based", "protocol_analysis",
    "ml_based", "nn_based", "unknown",
)
OBSERVATION_OBJECTS = ("packet_analysis", "flow_analysis", "log_analysis", "system_calls", "unknown")

DEFAULT_SYSTEM_TYPE_PROFILE: dict[str, Any] = {"types": {
    "NIDS": {"name": "Network intrusion detection system", "description": "Detects intrusions in network activity."},
    "NIPS": {"name": "Network intrusion prevention system / network detection and response", "description": "Detects and responds to network threats."},
    "HIDS": {"name": "Host intrusion detection system", "description": "Detects intrusions on hosts or endpoints."},
    "HIPS": {"name": "Host intrusion prevention system / endpoint detection and response", "description": "Detects and responds to host or endpoint threats."},
    "WIDS": {"name": "Wireless intrusion detection system", "description": "Detects intrusions in wireless activity."},
    "WIPS": {"name": "Wireless intrusion prevention system", "description": "Detects and prevents wireless threats."},
    "DBIDS": {"name": "Database intrusion detection system", "description": "Detects intrusions in database activity."},
    "DBIPS": {"name": "Database intrusion prevention system", "description": "Detects and prevents database threats."},
    "APIDS": {"name": "Application-protocol intrusion detection system", "description": "Detects intrusions in application activity."},
    "PIDS": {"name": "Protocol-based intrusion detection system", "description": "Detects protocol violations or attacks."},
    "ProvIDS": {"name": "Provenance-based intrusion detection system", "description": "Detects intrusions using provenance."},
    "XDR": {"name": "Extended detection and response", "description": "Detects and responds across observation scopes."},
    "IDS": {"name": "Intrusion detection system", "description": "Generic intrusion detection system."},
    "IPS": {"name": "Intrusion prevention system", "description": "Generic intrusion prevention system."},
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
CLASSIFICATION_COLUMNS = CANDIDATE_COLUMNS + CLASSIFIER_COLUMNS + [
    "pipeline_version", "profile_hash", "llm_model", "decision", "role",
    "observation_scopes", "observation_objects", "capabilities", "detection_methods", "primary_system_type",
    "system_types", "system_type_names",
    "independent_fork", "direct_self_description", "runtime_observation", "reproducible_execution",
    "offline_dataset_only", "implements_detection_engine", "requires_external_detector",
    "integrates_multiple_detectors", "cohesive_framework_architecture", "research_basis", "upstream_full_name", "classification_validation_flags", "primary_reason", "evidence_json",
    "evidence_quote_verified", "classified_at", "error",
] + ENSEMBLE_COLUMNS + EVIDENCE_EXPANSION_COLUMNS + KEYWORD_AUDIT_COLUMNS + NLI_AUDIT_COLUMNS
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
        "observation_objects": {"type": "array", "items": {"type": "string", "enum": list(OBSERVATION_OBJECTS)}},
        "capabilities": {"type": "array", "items": {"type": "string", "enum": list(CAPABILITIES)}},
        "detection_methods": {"type": "array", "items": {"type": "string", "enum": list(METHODS)}},
        "primary_system_type": {"type": "string", "enum": [*DEFAULT_SYSTEM_TYPE_PROFILE["types"], "unknown"]},
        "system_types": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": [*DEFAULT_SYSTEM_TYPE_PROFILE["types"], "unknown"]}},
        "independent_fork": {"type": "boolean"},
        "direct_self_description": {"type": "boolean"},
        "runtime_observation": {"type": "boolean"},
        "reproducible_execution": {"type": "boolean"},
        "offline_dataset_only": {"type": "boolean"},
        "implements_detection_engine": {"type": "boolean"},
        "requires_external_detector": {"type": "boolean"},
        "integrates_multiple_detectors": {"type": "boolean"},
        "cohesive_framework_architecture": {"type": "boolean"},
        "research_basis": {"type": "boolean"},
        "upstream_full_name": {"type": "string"},
        "primary_reason": {"type": "string"},
        "evidence": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
            "source": {"type": "string"}, "quote": {"type": "string"}
        }, "required": ["source", "quote"]}},
    },
    "required": ["role", "observation_scopes", "observation_objects", "capabilities", "detection_methods", "primary_system_type", "system_types", "independent_fork", "direct_self_description", "runtime_observation", "reproducible_execution", "offline_dataset_only", "implements_detection_engine", "requires_external_detector", "integrates_multiple_detectors", "cohesive_framework_architecture", "research_basis", "upstream_full_name", "primary_reason", "evidence"],
}


def _allowed(values: Any, allowed: tuple[str, ...], default: str | None = None) -> list[str]:
    """Filter model-provided list values to an allowed normalized vocabulary."""
    result = list(dict.fromkeys(str(value).strip() for value in values if str(value).strip() in allowed)) if isinstance(values, list) else []
    return result or ([default] if default else [])


def validate_system_type_profile(profile: dict[str, Any]) -> None:
    """Validate the configured LLM-facing system-type vocabulary."""
    types = profile.get("types", {})
    if not isinstance(types, dict) or not types:
        raise ValueError("system-type profile requires a non-empty types table")
    for code, definition in types.items():
        if not isinstance(code, str) or not code.strip() or code.casefold() == "unknown":
            raise ValueError("system-type codes must be non-empty and must not redefine unknown")
        if not isinstance(definition, dict):
            raise ValueError(f"system type {code!r} must be a table")
        if not str(definition.get("name", "")).strip() or not str(definition.get("description", "")).strip():
            raise ValueError(f"system type {code!r} requires name and description")


def system_type_codes(profile: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return configured system-type codes in prompt order, plus the unknown fallback."""
    active = profile or DEFAULT_SYSTEM_TYPE_PROFILE
    validate_system_type_profile(active)
    return tuple(str(code) for code in active["types"]) + ("unknown",)


def response_schema(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the structured LLM schema using the configured system-type vocabulary."""
    schema = json.loads(json.dumps(RESPONSE_SCHEMA))
    codes = list(system_type_codes(profile))
    schema["properties"]["primary_system_type"]["enum"] = codes
    schema["properties"]["system_types"]["items"]["enum"] = codes
    return schema


def system_type_names(codes: list[str], profile: dict[str, Any] | None = None) -> list[str]:
    """Resolve configured human-readable names for ordered system-type codes."""
    types = (profile or DEFAULT_SYSTEM_TYPE_PROFILE).get("types", {})
    return [str(types.get(code, {}).get("name", code)) for code in codes]


def normalize_response(
    raw: dict[str, Any], verified: list[dict[str, str]], config: dict[str, Any], is_fork: bool,
    system_type_profile: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Validate and normalize a repository model response into auditable decision fields."""
    role = str(raw.get("role", "uncertain"))
    role = role if role in AUDIT_ROLES else "uncertain"
    scopes = _allowed(raw.get("observation_scopes", []), SCOPES, "unknown")
    objects = _allowed(raw.get("observation_objects", []), OBSERVATION_OBJECTS, "unknown")
    capabilities = _allowed(raw.get("capabilities", []), CAPABILITIES)
    methods = _allowed(raw.get("detection_methods", []), METHODS, "unknown")
    configured_types = system_type_codes(system_type_profile)
    type_codes = _allowed(raw.get("system_types", []), configured_types, "unknown")
    primary_type = str(raw.get("primary_system_type", "unknown")).strip()
    primary_type = primary_type if primary_type in configured_types else "unknown"
    if primary_type != "unknown":
        type_codes = [primary_type, *(code for code in type_codes if code != primary_type)]
    elif any(code != "unknown" for code in type_codes):
        primary_type = next(code for code in type_codes if code != "unknown")
    if len(type_codes) > 1 and "unknown" in type_codes:
        type_codes.remove("unknown")
    independent = bool(raw.get("independent_fork"))
    self_description = bool(raw.get("direct_self_description"))
    runtime_observation = bool(raw.get("runtime_observation"))
    reproducible_execution = bool(raw.get("reproducible_execution"))
    offline_dataset_only = bool(raw.get("offline_dataset_only"))
    implements_engine = bool(raw.get("implements_detection_engine"))
    requires_external = bool(raw.get("requires_external_detector"))
    integrates_multiple = bool(raw.get("integrates_multiple_detectors"))
    framework_architecture = bool(raw.get("cohesive_framework_architecture"))
    research_basis = bool(raw.get("research_basis"))
    upstream_full_name = str(raw.get("upstream_full_name", "")).strip().strip("/")
    verified_text = normalize_text(" ".join(item.get("quote", "") for item in verified))
    if upstream_full_name and (upstream_full_name.count("/") != 1 or normalize_text(upstream_full_name) not in verified_text):
        upstream_full_name = ""
    validation_flags: list[str] = []
    if role == "native_detector":
        if not runtime_observation:
            validation_flags.append("runtime_observation_missing")
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
        if offline_dataset_only:
            validation_flags.append("offline_dataset_only")
        if not implements_engine:
            validation_flags.append("detection_engine_implementation_missing")
        if requires_external:
            validation_flags.append("external_detector_required")
        if framework_architecture:
            validation_flags.append("framework_architecture_conflict")
    elif role == "classifier_engine":
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
        if not implements_engine:
            validation_flags.append("detection_engine_implementation_missing")
    elif role == "detection_framework":
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
        if not implements_engine:
            validation_flags.append("detection_engine_implementation_missing")
        if not framework_architecture:
            validation_flags.append("framework_architecture_missing")
    elif role == "detection_platform":
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
        if not integrates_multiple:
            validation_flags.append("multiple_detector_integration_missing")
    elif role == "research_project":
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
        if not research_basis:
            validation_flags.append("research_basis_missing")
    role_ok = not validation_flags
    require_quote = bool(config.get("require_verified_quote", True))
    evidence_ok = bool(verified) or not require_quote
    if role in ROLES and self_description and evidence_ok and (not is_fork or independent) and role_ok:
        decision = "accept"
    elif role == "out_of_scope" and evidence_ok:
        decision = "exclude"
    else:
        decision = "uncertain"
    return {
        "decision": decision,
        "role": role,
        "observation_scopes": ";".join(scopes),
        "observation_objects": ";".join(objects),
        "capabilities": ";".join(capabilities),
        "detection_methods": ";".join(methods),
        "primary_system_type": primary_type,
        "system_types": ";".join(type_codes) if type_codes else "unknown",
        "system_type_names": ";".join(system_type_names(type_codes, system_type_profile)) if type_codes else "Unknown",
        "independent_fork": str(independent).lower(),
        "direct_self_description": str(self_description).lower(),
        "runtime_observation": str(runtime_observation).lower(),
        "reproducible_execution": str(reproducible_execution).lower(),
        "offline_dataset_only": str(offline_dataset_only).lower(),
        "implements_detection_engine": str(implements_engine).lower(),
        "requires_external_detector": str(requires_external).lower(),
        "integrates_multiple_detectors": str(integrates_multiple).lower(),
        "cohesive_framework_architecture": str(framework_architecture).lower(),
        "research_basis": str(research_basis).lower(),
        "upstream_full_name": upstream_full_name,
        "classification_validation_flags": ";".join(validation_flags),
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
    for field, allowed in (("observation_scopes", SCOPES), ("observation_objects", OBSERVATION_OBJECTS), ("capabilities", CAPABILITIES), ("detection_methods", METHODS)):
        invalid = set(split_values(row.get(field))) - set(allowed)
        if invalid:
            raise ValueError(f"invalid {field}: {sorted(invalid)}")
    configured = set(system_type_codes(system_type_profile))
    invalid_types = set(split_values(row.get("system_types"))) - configured
    if invalid_types:
        raise ValueError(f"invalid system_types: {sorted(invalid_types)}")
    primary = str(row.get("primary_system_type", ""))
    if primary not in configured or primary not in set(split_values(row.get("system_types"))):
        raise ValueError(f"invalid primary_system_type: {primary!r}")
