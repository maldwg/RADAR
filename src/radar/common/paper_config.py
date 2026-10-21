"""One explicit operational configuration schema for scholarly paper studies."""
from __future__ import annotations

import math
from typing import Any

PAPER_PIPELINE_OPTIONS = {
    "collection": {"collection_limit", "collection_mode", "max_surveys", "snowball_start_papers", "snowball_max_depth",
        "survey_reference_fraction", "query_limit_factor", "survey_query_limit_factor", "reference_limit_factor",
        "survey_consensus_threshold", "retrieval_minimum_score", "retrieval_minimum_title_tokens", "retrieval_title_weight",
        "minimum_citations", "candidate_threshold", "borderline_threshold"},
    "evidence": {"fetch_full_text", "enrich_missing_abstracts", "enrich_missing_full_text", "full_text_lookup_providers", "enrich_uncertain_with_full_text",
        "evidence_lookup_timeout_seconds", "max_full_text_chars"},
    "classification": {"classification_limit", "classify_borderline", "consecutive_runs", "consensus_threshold", "ensemble"},
    "reconciliation": {"strategy", "reconcile_with_llm"},
}


def validate_paper_pipeline(pipeline: dict[str, Any], workflow: str) -> None:
    """Validate complete, identical paper settings and executable feature combinations."""
    for stage, keys in PAPER_PIPELINE_OPTIONS.items():
        options = pipeline.get(stage)
        if not isinstance(options, dict) or set(options) != keys:
            raise ValueError(f"paper pipeline.{stage} must declare exactly: {', '.join(sorted(keys))}")
    collection, evidence, classification, reconciliation = (pipeline[stage] for stage in PAPER_PIPELINE_OPTIONS)
    if collection["collection_mode"] not in {"ranked", "query_position"}:
        raise ValueError("collection_mode must be ranked or query_position")
    for key, minimum, maximum in (("max_surveys", 0, None), ("snowball_start_papers", 1, None),
        ("snowball_max_depth", 0, 10), ("survey_consensus_threshold", 2, None),
        ("retrieval_minimum_title_tokens", 1, None), ("minimum_citations", 0, None)):
        value = collection[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum or maximum is not None and value > maximum:
            raise ValueError(f"collection.{key} is outside its supported integer range")
    for key in ("survey_reference_fraction", "retrieval_minimum_score", "retrieval_title_weight", "candidate_threshold", "borderline_threshold"):
        value = collection[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"collection.{key} must be finite")
    if not 0 <= collection["survey_reference_fraction"] <= 0.9 or collection["retrieval_title_weight"] < 0:
        raise ValueError("invalid survey allocation or title weight")
    if collection["candidate_threshold"] < collection["borderline_threshold"]:
        raise ValueError("candidate_threshold must be at least borderline_threshold")
    for key in ("fetch_full_text", "enrich_missing_abstracts", "enrich_missing_full_text", "enrich_uncertain_with_full_text"):
        if not isinstance(evidence[key], bool):
            raise ValueError(f"evidence.{key} must be boolean")
    providers = evidence['full_text_lookup_providers']
    if not isinstance(providers, list) or any(not isinstance(name, str) or not name for name in providers) or len(set(providers)) != len(providers):
        raise ValueError('full_text_lookup_providers must be a list of distinct provider names')
    from radar.adapters.registry import provider_descriptor

    for name in providers:
        if 'evidence' not in provider_descriptor(name).capabilities:
            raise ValueError(f'full-text provider {name} does not implement evidence lookup')
    if isinstance(evidence["max_full_text_chars"], bool) or not isinstance(evidence["max_full_text_chars"], int) or evidence["max_full_text_chars"] < 1:
        raise ValueError("max_full_text_chars must be a positive integer")
    timeout = evidence["evidence_lookup_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("evidence_lookup_timeout_seconds must be finite and positive")
    if not isinstance(classification["classify_borderline"], bool) or not isinstance(reconciliation["reconcile_with_llm"], bool):
        raise ValueError("classify_borderline and reconcile_with_llm must be boolean")
    strategy = reconciliation["strategy"]
    expected = "dataset_release" if workflow == "dataset_catalogue" else "none"
    if strategy != expected:
        raise ValueError(f"{workflow} requires reconciliation strategy {expected}")
    if strategy == "none" and reconciliation["reconcile_with_llm"]:
        raise ValueError("disabled reconciliation cannot enable LLM reconciliation")
