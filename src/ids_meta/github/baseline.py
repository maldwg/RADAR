from __future__ import annotations

from typing import Any

from ids_meta.github.schema import (
    AUDIT_ROLES, CAPABILITIES, METHODS, ROLES, SCOPES,
    derive_system_types, system_type_names,
)


SUPPORTED_FIELDS = [
    "decision", "role", "observation_scopes", "capabilities", "detection_methods",
    "primary_system_type", "system_types",
]


def _allowed_list(value: object, allowed: tuple[str, ...], fallback: str = "") -> list[str]:
    """Normalize a predicted list against one schema vocabulary."""
    values = value if isinstance(value, list) else []
    selected = list(dict.fromkeys(str(item) for item in values if str(item) in allowed))
    return selected or ([fallback] if fallback else [])


def _normalize_result(
    predictions: dict[str, Any], config: dict[str, Any], classifier_name: str,
) -> dict[str, str]:
    """Normalize one non-LLM taxonomy result into the GitHub output contract."""
    role = str(predictions.get("role", "uncertain"))
    role = role if role in AUDIT_ROLES else "uncertain"
    scopes = _allowed_list(predictions.get("observation_scopes"), SCOPES, "unknown")
    capabilities = _allowed_list(predictions.get("capabilities"), CAPABILITIES)
    methods = _allowed_list(predictions.get("detection_methods"), METHODS, "unknown")
    profile = config.get("system_type_profile")
    types = derive_system_types(scopes, capabilities, methods, role, profile)
    derived_decision = "accept" if role in ROLES else "exclude" if role == "out_of_scope" else "uncertain"
    decision = str(predictions.get("decision", derived_decision))
    decision = decision if decision in {"accept", "exclude", "uncertain"} else derived_decision
    return {
        "decision": decision, "role": role,
        "observation_scopes": ";".join(scopes), "capabilities": ";".join(capabilities),
        "detection_methods": ";".join(methods),
        "primary_system_type": types[0] if types else "",
        "system_types": ";".join(types),
        "system_type_names": ";".join(system_type_names(types, profile)),
        "independent_fork": "unknown", "direct_self_description": "unknown",
        "runtime_observation": "unknown", "reproducible_execution": "unknown",
        "offline_dataset_only": "unknown", "implements_detection_engine": "unknown",
        "requires_external_detector": "unknown", "integrates_multiple_detectors": "unknown",
        "cohesive_framework_architecture": "unknown", "research_basis": "unknown",
        "upstream_full_name": "", "classification_validation_flags": "",
        "primary_reason": (
            f"{classifier_name} selected role={role}; configured role and relevance rules "
            f"map the result to decision={decision}."
        ),
        "evidence_json": "[]", "evidence_quote_verified": "false",
        "classifier_supported_fields": ";".join(SUPPORTED_FIELDS),
    }


def normalize_nli_result(
    predictions: dict[str, Any], _candidate: dict[str, str], config: dict[str, Any],
) -> dict[str, str]:
    """Map zero-shot semantic labels to decisions without LLM-specific evidence gates."""
    return _normalize_result(predictions, config, "Zero-shot NLI")


def normalize_keyword_result(
    predictions: dict[str, Any], _candidate: dict[str, str], config: dict[str, Any],
) -> dict[str, str]:
    """Map deterministic keyword labels to the complete GitHub taxonomy contract."""
    return _normalize_result(predictions, config, "Keyword taxonomy")
