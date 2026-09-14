from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError, Settings
from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.logging import logger
from ids_meta.common.ollama import format_evidence
from ids_meta.common.run import utc_now
from ids_meta.common.scoring import score_evidence
from ids_meta.common.text import stable_hash


LOG = logger("baselines")
BASELINE_VERSION = "keyword-baseline-2"
KNOWN_BASELINE_METHODS = ("keyword", "nli")
IMPLEMENTED_BASELINE_METHODS = ("keyword", "nli")
CLASSIFIER_COLUMNS = [
    "classifier_method", "classifier_id", "classifier_profile_hash",
    "classifier_supported_fields", "classifier_evidence_source",
]
KEYWORD_AUDIT_COLUMNS = [
    "keyword_score", "keyword_band", "keyword_signals",
    "keyword_accept_threshold", "keyword_uncertain_threshold",
]


@dataclass(frozen=True)
class KeywordBaselineConfig:
    """Validated source-specific decision thresholds for the keyword baseline."""

    accept_threshold: float
    uncertain_threshold: float


def configured_baseline_methods(settings: Settings, requested: str | None = None) -> list[str]:
    """Return configured or explicitly requested baseline methods in stable order."""
    if requested is None:
        raw = settings.section("baselines").get("methods", ["keyword"])
        if not isinstance(raw, list):
            raise ConfigError("baselines.methods must be an array of method names")
        methods = [str(value).strip().casefold() for value in raw]
    else:
        methods = [value.strip().casefold() for value in requested.split(",")]
    methods = list(dict.fromkeys(value for value in methods if value))
    if not methods:
        raise ConfigError("at least one baseline method must be configured")
    unknown = [value for value in methods if value not in KNOWN_BASELINE_METHODS]
    if unknown:
        raise ConfigError("unknown baseline methods: " + ",".join(unknown))
    unavailable = [value for value in methods if value not in IMPLEMENTED_BASELINE_METHODS]
    if unavailable:
        raise ConfigError("baseline methods are not implemented yet: " + ",".join(unavailable))
    return methods


def load_keyword_baseline_config(settings: Settings, source: str) -> KeywordBaselineConfig:
    """Load and validate the keyword decision thresholds for one source."""
    baselines = settings.section("baselines")
    enabled = baselines.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("baselines.enabled must be true or false")
    keyword = baselines.get("keyword", {})
    if not isinstance(keyword, dict):
        raise ConfigError("baselines.keyword must be a table")
    raw = keyword.get(source, {})
    if not isinstance(raw, dict):
        raise ConfigError(f"baselines.keyword.{source} must be a table")
    try:
        accept = float(raw.get("accept_threshold", 7.0))
        uncertain = float(raw.get("uncertain_threshold", 4.0))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"baselines.keyword.{source} thresholds must be numeric") from exc
    if accept <= uncertain:
        raise ConfigError(
            f"baselines.keyword.{source}.accept_threshold must be greater than uncertain_threshold"
        )
    return KeywordBaselineConfig(accept, uncertain)


def keyword_profile_hash(
    source: str, term_profile: dict[str, Any], config: KeywordBaselineConfig,
) -> str:
    """Fingerprint all semantics that can alter a keyword baseline decision."""
    payload = {
        "version": BASELINE_VERSION,
        "source": source,
        "term_profile": term_profile,
        "accept_threshold": config.accept_threshold,
        "uncertain_threshold": config.uncertain_threshold,
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _source_contract(source: str) -> tuple[str, str, list[str]]:
    """Resolve stable IDs, raw-cache namespaces, and CSV columns for one source."""
    if source == "github":
        from ids_meta.github.schema import CLASSIFICATION_COLUMNS

        return "github_repo_id", "github.repository", CLASSIFICATION_COLUMNS
    if source == "openalex":
        from ids_meta.openalex.schema import CLASSIFICATION_COLUMNS

        return "openalex_id", "openalex.work", CLASSIFICATION_COLUMNS
    raise ValueError(f"unsupported keyword baseline source: {source}")


def _evidence_text(source: str, payload: dict[str, Any]) -> str:
    """Format the same bounded source evidence used by prefiltering and LLMs."""
    if source == "github":
        from ids_meta.github.pipeline import evidence_sections
    elif source == "openalex":
        from ids_meta.openalex.pipeline import evidence_sections
    else:
        raise ValueError(f"unsupported keyword baseline source: {source}")
    return format_evidence(evidence_sections(payload), 30000)


def _candidate_payload(source: str, row: dict[str, str]) -> dict[str, Any]:
    """Reconstruct the evidence retained in a candidate audit when raw cache data is absent."""
    if source == "github":
        topics = [value.strip() for value in str(row.get("topics", "")).split(";") if value.strip()]
        return {"repository": {
            "full_name": row.get("full_name", ""), "description": row.get("description", ""),
            "topics": topics,
        }}
    if source == "openalex":
        return {"work": dict(row)}
    raise ValueError(f"unsupported keyword baseline source: {source}")


def classify_keyword_baseline(
    source: str,
    candidates_path: Path,
    output: Path,
    cache: PipelineCache,
    term_profile: dict[str, Any],
    config: KeywordBaselineConfig,
    *,
    resume: bool = True,
    progress_every: int = 25,
) -> list[dict[str, str]]:
    """Classify every candidate row with an independent, cached keyword decision rule."""
    identity, record_namespace, columns = _source_contract(source)
    candidates = read_csv(candidates_path)
    require_columns(candidates, (identity, "evidence_hash"), candidates_path)
    profile_hash = keyword_profile_hash(source, term_profile, config)
    decision_namespace = f"{source}.baseline.keyword"
    results: list[dict[str, str]] = []
    reused = 0
    candidate_fallbacks = 0
    progress_every = max(1, int(progress_every))
    LOG.info(
        "starting keyword baseline source=%s rows=%d accept_threshold=%g uncertain_threshold=%g resume=%s",
        source, len(candidates), config.accept_threshold, config.uncertain_threshold, resume,
    )
    for index, candidate in enumerate(candidates, 1):
        key = str(candidate.get(identity, "")).strip()
        if not key:
            raise ValueError(f"keyword baseline candidate is missing {identity}: row {index}")
        payload = cache.get_record(record_namespace, key)
        evidence_source = "raw_cache" if payload is not None else "candidate_csv"
        if payload is None:
            payload = _candidate_payload(source, candidate)
            candidate_fallbacks += 1
            if candidate_fallbacks == 1:
                LOG.warning(
                    "raw evidence missing; using candidate audit evidence source=%s first_id=%s",
                    source, key,
                )
        evidence = _evidence_text(source, payload)
        evidence_hash = stable_hash(evidence)
        cached = cache.get_decision(decision_namespace, key, profile_hash, evidence_hash) if resume else None
        if cached is not None:
            results.append({str(name): str(value) for name, value in cached.items()})
            reused += 1
            continue
        scored = score_evidence(
            evidence, term_profile, config.accept_threshold, config.uncertain_threshold,
        )
        decision = {
            "candidate": "accept", "borderline": "uncertain", "reject": "exclude",
        }[scored.band]
        classified_at = utc_now()
        result: dict[str, Any] = dict(candidate)
        result.update({
            "evidence_hash": evidence_hash,
            "pipeline_version": BASELINE_VERSION,
            "profile_hash": profile_hash,
            "llm_model": "",
            "classifier_method": "keyword",
            "classifier_id": "keyword-score",
            "classifier_profile_hash": profile_hash,
            "classifier_supported_fields": "decision",
            "classifier_evidence_source": evidence_source,
            "decision": decision,
            "primary_reason": (
                f"Keyword score {scored.score:g}; accept at >= {config.accept_threshold:g}, "
                f"uncertain at >= {config.uncertain_threshold:g}, otherwise exclude."
            ),
            "classification_validation_flags": "",
            "evidence_quote_verified": "false",
            "classified_at": classified_at,
            "error": "",
            "keyword_score": f"{scored.score:g}",
            "keyword_band": scored.band,
            "keyword_signals": ";".join(scored.signals),
            "keyword_accept_threshold": f"{config.accept_threshold:g}",
            "keyword_uncertain_threshold": f"{config.uncertain_threshold:g}",
        })
        serialized = {str(name): str(value) for name, value in result.items()}
        cache.put_decision(decision_namespace, key, profile_hash, evidence_hash, serialized, classified_at)
        results.append(serialized)
        if index % progress_every == 0:
            atomic_write_csv(output, results, columns)
            LOG.info("keyword baseline progress source=%s completed=%d/%d", source, index, len(candidates))
    atomic_write_csv(output, results, columns)
    counts: dict[str, int] = {}
    for row in results:
        decision = row.get("decision", "unavailable")
        counts[decision] = counts.get(decision, 0) + 1
    LOG.info(
        "keyword baseline complete source=%s rows=%d cached=%d candidate_evidence_fallbacks=%d decisions=%s output=%s",
        source, len(results), reused, candidate_fallbacks, counts, output,
    )
    return results
