from __future__ import annotations

from typing import Any

from ids_meta.openalex.schema import ACCEPTED_ROLES, PAPER_ROLES


SUPPORTED_FIELDS = [
    "decision", "paper_role", "relevance_tier", "artifact_type", "domain", "data_form",
    "capture_environment", "labeling",
]
EXCLUDED_ROLES = {"survey", "downstream_application", "dataset_analysis", "unrelated"}


def normalize_nli_result(
    predictions: dict[str, Any], _candidate: dict[str, str], _config: dict[str, Any],
) -> dict[str, str]:
    """Map the winning zero-shot paper role to a decision and retain taxonomy predictions."""
    role = str(predictions.get("paper_role", "uncertain"))
    role = role if role in PAPER_ROLES else "uncertain"
    decision = "accept" if role in ACCEPTED_ROLES else "exclude" if role in EXCLUDED_ROLES else "uncertain"
    return {
        "decision": decision, "paper_role": role,
        "relevance_tier": str(predictions.get("relevance_tier", "uncertain")),
        "artifact_id": "", "artifact_name": "", "artifact_display_name": "",
        "resource_name": "", "artifact_aliases": "",
        "artifact_type": str(predictions.get("artifact_type", "unknown")),
        "domain": str(predictions.get("domain", "unknown")),
        "data_form": str(predictions.get("data_form", "unknown")),
        "capture_environment": str(predictions.get("capture_environment", "unknown")),
        "labeling": str(predictions.get("labeling", "unknown")),
        "direct_release_evidence": "unknown", "classification_validation_flags": "",
        "primary_reason": (
            f"Zero-shot NLI selected paper_role={role}; release roles map to accept, "
            f"non-release roles to exclude, and low-margin results to uncertain."
        ),
        "model_evidence_json": "[]", "evidence_json": "[]",
        "evidence_quote_verified": "false",
        "classifier_supported_fields": ";".join(SUPPORTED_FIELDS),
    }
