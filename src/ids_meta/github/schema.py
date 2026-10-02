from __future__ import annotations

import json
from typing import Any

from ids_meta.common.baselines import CLASSIFIER_COLUMNS, KEYWORD_AUDIT_COLUMNS
from ids_meta.common.nli import NLI_AUDIT_COLUMNS
from ids_meta.common.evidence import EVIDENCE_EXPANSION_COLUMNS
from ids_meta.common.text import normalize_text, split_values

ROLES = (
    "native_detector", "classifier_engine", "detection_framework", "detection_platform",
    "traffic_analyzer", "antivirus", "research_project", "demo", "rules_content", "support_tooling",
)
AUDIT_ROLES = ROLES + ("out_of_scope", "uncertain")
SCOPES = ("network", "host_endpoint", "wireless", "database", "provenance", "application", "multi_scope", "unknown")
CAPABILITIES = ("detection", "prevention", "monitoring")
METHODS = (
    "signature_based", "rule_based", "anomaly_based", "protocol_analysis",
    "ml_based", "nn_based", "unknown",
)
OBSERVATION_OBJECTS = ("packet_analysis", "flow_analysis", "log_analysis", "system_calls", "file_analysis", "unknown")
CENTRALIZATION = ("centralized", "hierarchical", "fully_distributed", "unknown")

DEFAULT_SYSTEM_TYPE_PROFILE: dict[str, Any] = {"types": {
    "AV": {"name": "Antivirus / malware file scanner", "description": "Scans files, uploads, binaries, archives, or cloud-storage objects for malware or malicious content using signatures, hashes, YARA rules, heuristics, or learned models. On-demand and offline scanning and integrated external scanning engines are permitted. File scanning, upload rejection, or malware quarantine alone does not establish IDS/HIDS/HIPS; additional intrusion detection paths require separate evidence."},
    "NIDS": {"name": "Network intrusion detection system", "description": "Detects intrusions through inspection of network packets, frames, or protocol streams from network interfaces, taps, or saved packet captures, including a materially integrated packet detector. Requires an implemented traffic-inspection detection path; service logs, aggregated flow records, or endpoint connection permissions alone do not establish NIDS. Running a packet intrusion detector on one host does not disqualify it."},
    "NSM": {"name": "Network security monitoring / traffic analysis system", "description": "Principally analyzes network traffic for security telemetry and investigation; limited notices or detection hooks do not establish an IDS or response system."},
    "SM": {"name": "Security monitoring / telemetry analysis system", "description": "Principally collects, aggregates, searches, or analyzes security logs and non-network or mixed-source telemetry for visibility and investigation; intrusion decisions and alerts are not required. Generic observability and configuration or forwarding alone do not establish this type."},
    "NIPS": {"name": "Network intrusion prevention system", "description": "Combines network packet/interface intrusion inspection as defined for NIDS with implemented network enforcement, such as dropping or rejecting traffic or updating firewall rules. Firewall updates alone, log-derived IP bans, endpoint application connection policies, and an NDR marketing label do not establish NIPS."},
    "HIDS": {"name": "Host intrusion detection system", "description": "Detects intrusions from host or endpoint activity and state: files, processes, execution, system calls, registry, accounts, or operating-system audit events and logs describing these. Endpoint observations or local analysis may be forwarded to a central manager; centralized correlation does not remove HIDS. Local installation, malware-content scanning alone, or reading network-service authentication/traffic logs alone does not establish HIDS. Malware-content scanning belongs to AV; trusted-baseline file-integrity monitoring can support HIDS."},
    "HIPS": {"name": "Host intrusion prevention system", "description": "Combines host or endpoint intrusion detection as defined for HIDS with implemented enforcement on endpoint activity or state, such as blocking execution, terminating malicious processes, quarantining files, removing persistence, or disabling compromised accounts. Network-only firewall rules, source-IP bans, or connection isolation alone do not establish HIPS, even when executed on the protected host. Antivirus scanning and quarantine alone belong to AV and do not establish HIPS. An EDR label or an unimplemented response hook is insufficient."},
    "WIDS": {"name": "Wireless intrusion detection system", "description": "Detects intrusions in wireless activity."},
    "WIPS": {"name": "Wireless intrusion prevention system", "description": "Detects and prevents wireless threats."},
    "DBIDS": {"name": "Database intrusion detection system", "description": "Detects intrusions in database activity."},
    "DBIPS": {"name": "Database intrusion prevention system", "description": "Detects and prevents database threats."},
    "APIDS": {"name": "Application-protocol intrusion detection system", "description": "Detects intrusions in application activity."},
    "PIDS": {"name": "Protocol-based intrusion detection system", "description": "Detects protocol violations or attacks."},
    "ProvIDS": {"name": "Provenance-based intrusion detection system", "description": "Detects intrusions using provenance."},
    "XDR": {"name": "Extended detection and response", "description": "Detects and responds across observation scopes."},
    "IDS": {"name": "Intrusion detection system", "description": "General intrusion detection, including detectors analyzing network-service authentication/traffic logs, aggregated flow records, or local connection-policy events without a packet intrusion inspection path or endpoint activity detection as defined for NIDS/HIDS. Also covers detection when a more specific configured type is not supported. Single-host deployment does not imply HIDS; network-related evidence does not imply NIDS. Select alongside a specific type only for a separately evidenced general detection path."},
    "IPS": {"name": "Intrusion prevention system", "description": "General intrusion prevention with implemented enforcement, including network-service log detection followed by source-IP bans or firewall updates, and endpoint application connection-policy enforcement without a NIPS packet intrusion inspection path or HIPS endpoint enforcement. Fail2ban is an archetype. Also covers prevention when a more specific configured type is not supported. Alerts or response hooks alone are insufficient; select alongside a specific type only for a separately evidenced general prevention path."},
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
    "system_types", "system_type_names", "centralization",
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
        "centralization": {"type": "string", "enum": list(CENTRALIZATION)},
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
    "required": ["role", "centralization", "observation_scopes", "observation_objects", "capabilities", "detection_methods", "primary_system_type", "system_types", "independent_fork", "direct_self_description", "runtime_observation", "reproducible_execution", "offline_dataset_only", "implements_detection_engine", "requires_external_detector", "integrates_multiple_detectors", "cohesive_framework_architecture", "research_basis", "upstream_full_name", "primary_reason", "evidence"],
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
    centralization = str(raw.get("centralization", "unknown")).strip()
    centralization = centralization if centralization in CENTRALIZATION else "unknown"
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
    elif role == "traffic_analyzer":
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
    elif role == "antivirus":
        if not reproducible_execution:
            validation_flags.append("reproducible_execution_missing")
        if "file_analysis" not in objects:
            validation_flags.append("file_analysis_missing")
        if "detection" not in capabilities:
            validation_flags.append("detection_capability_missing")
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
        "centralization": centralization,
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
    if row.get("centralization") not in CENTRALIZATION:
        raise ValueError(f"processed GitHub row has invalid centralization: {row.get('centralization')}")
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
