from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radar.common.cache import COLLECTED_EVIDENCE_POLICY, PipelineCache, collected_record_hash
from radar.common.config import ConfigError, Settings, load_system_type_profile
from radar.common.csvio import atomic_write_csv, read_csv, require_columns
from radar.common.logging import logger
from radar.common.ollama import format_evidence
from radar.common.run import utc_now
from radar.common.scoring import ScoreResult
from radar.common.text import normalize_text, stable_hash


LOG = logger("baselines")
BASELINE_VERSION = "keyword-baseline-4"
KNOWN_BASELINE_METHODS = ("keyword", "nli", "llm_nli", "plain_llm")
IMPLEMENTED_BASELINE_METHODS = KNOWN_BASELINE_METHODS
CLASSIFIER_COLUMNS = [
    "classifier_method", "classifier_id", "classifier_profile_hash",
    "classifier_supported_fields", "classifier_evidence_source",
    "classifier_evidence_policy", "classifier_evidence_snapshot_hash",
]
KEYWORD_AUDIT_COLUMNS = [
    "keyword_score", "keyword_band", "keyword_signals",
    "keyword_accept_threshold", "keyword_uncertain_threshold",
    "keyword_taxonomy_scores_json", "keyword_taxonomy_signals_json",
]


@dataclass(frozen=True)
class KeywordBaselineConfig:
    """Validated source-specific decision thresholds for the keyword baseline."""

    accept_threshold: float
    uncertain_threshold: float


@dataclass(frozen=True)
class KeywordField:
    """One deterministic keyword classification field."""

    name: str
    mode: str
    fallback: str
    minimum_score: float
    margin: float
    labels: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class KeywordTaxonomyProfile:
    """Complete source taxonomy and decision mapping for the keyword baseline."""

    source: str
    fields: tuple[KeywordField, ...]
    decision_field: str
    include_labels: tuple[str, ...]
    exclude_labels: tuple[str, ...]
    raw: dict[str, Any]
    path: Path


def configured_baseline_methods(settings: Settings, requested: str | None = None) -> list[str]:
    """Return configured or explicitly requested baseline methods in stable order."""
    if requested is None:
        raw = settings.section("baselines").get("methods", ["keyword"])
        if not isinstance(raw, list):
            raise ConfigError("baselines.methods must be an array of method names")
        methods = [str(value).strip().casefold() for value in raw]
        llm_nli = settings.section("baselines").get("llm_nli", {})
        if not isinstance(llm_nli, dict) or not isinstance(llm_nli.get("enabled", False), bool):
            raise ConfigError("baselines.llm_nli.enabled must be true or false")
        if llm_nli.get("enabled", False) and "llm_nli" not in methods:
            methods.append("llm_nli")
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
    if settings.section("study").get("workflow") == "systematic_review" and any(method != "plain_llm" for method in methods):
        raise ConfigError("systematic_review currently supports only the plain_llm baseline")
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


def load_keyword_taxonomy_profile(settings: Settings, source: str) -> KeywordTaxonomyProfile:
    """Load and validate a source-complete, independently configured keyword taxonomy."""
    path = settings.path(f"{source}_keyword")
    if not path.is_file():
        raise ConfigError(f"keyword taxonomy profile does not exist: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if str(raw.get("schema_version", "")) != "1.0" or str(raw.get("source", "")) != source:
        raise ConfigError(
            f"keyword taxonomy profile must use schema_version 1.0 and source {source!r}: {path}"
        )
    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ConfigError(f"keyword taxonomy profile requires a non-empty [fields] table: {path}")
    fields: list[KeywordField] = []
    for name, value in raw_fields.items():
        if not isinstance(value, dict):
            raise ConfigError(f"keyword field {name!r} must be a table")
        mode = str(value.get("mode", "")).strip()
        fallback = str(value.get("fallback", "")).strip()
        labels = value.get("labels")
        if mode not in {"exclusive", "multilabel"}:
            raise ConfigError(f"keyword field {name!r} has unsupported mode {mode!r}")
        if not fallback:
            raise ConfigError(f"keyword field {name!r} requires a fallback")
        if not isinstance(labels, dict) or not labels:
            raise ConfigError(f"keyword field {name!r} requires labels")
        normalized: dict[str, dict[str, Any]] = {}
        for label, rule in labels.items():
            if not isinstance(rule, dict):
                raise ConfigError(f"keyword label {name}.{label} must be a table")
            terms = rule.get("terms")
            if not isinstance(terms, list) or not terms or any(
                not isinstance(term, str) or not term.strip() for term in terms
            ):
                raise ConfigError(f"keyword label {name}.{label} requires non-empty terms")
            try:
                weight = float(rule.get("weight", 1.0))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"keyword label {name}.{label}.weight must be numeric") from exc
            if weight <= 0:
                raise ConfigError(f"keyword label {name}.{label}.weight must be positive")
            normalized[str(label)] = {"terms": [term.strip() for term in terms], "weight": weight}
        try:
            minimum_score = float(value.get("minimum_score", 1.0))
            margin = float(value.get("margin", 1.0 if mode == "exclusive" else 0.0))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"keyword field {name!r} score settings must be numeric") from exc
        if minimum_score <= 0 or margin < 0:
            raise ConfigError(f"keyword field {name!r} requires minimum_score > 0 and margin >= 0")
        fields.append(KeywordField(str(name), mode, fallback, minimum_score, margin, normalized))
    if source == "github":
        from radar.github.schema import AUDIT_ROLES, CAPABILITIES, CENTRALIZATION, METHODS, OBSERVATION_OBJECTS, SCOPES, system_type_codes

        expected = {
            "role": set(AUDIT_ROLES) - {"uncertain"},
            "centralization": set(CENTRALIZATION) - {"unknown"},
            "observation_scopes": set(SCOPES) - {"unknown"},
            "observation_objects": set(OBSERVATION_OBJECTS) - {"unknown"},
            "capabilities": set(CAPABILITIES),
            "detection_methods": set(METHODS) - {"unknown"},
            "system_types": set(system_type_codes(load_system_type_profile(settings, "github"))) - {"unknown"},
        }
    elif source == "openalex":
        from radar.openalex.schema import (
            ARTIFACT_TYPES, DATA_FORMS, DOMAINS, ENVIRONMENTS, LABELING, PAPER_ROLES, TIERS,
        )

        expected = {
            "paper_role": set(PAPER_ROLES) - {"uncertain"},
            "relevance_tier": set(TIERS) - {"uncertain"},
            "artifact_type": set(ARTIFACT_TYPES) - {"unknown"},
            "domain": set(DOMAINS) - {"unknown"},
            "data_form": set(DATA_FORMS) - {"unknown"},
            "capture_environment": set(ENVIRONMENTS) - {"unknown"},
            "labeling": set(LABELING) - {"unknown"},
        }
    else:  # pragma: no cover - guarded by the CLI source choices
        raise ConfigError(f"unsupported keyword taxonomy source: {source}")
    configured = {field.name: set(field.labels) for field in fields}
    if set(configured) != set(expected):
        missing = sorted(set(expected) - set(configured))
        extra = sorted(set(configured) - set(expected))
        raise ConfigError(f"keyword taxonomy fields differ from {source} schema: missing={missing}, extra={extra}")
    for name, labels in expected.items():
        if configured[name] != labels:
            missing = sorted(labels - configured[name])
            extra = sorted(configured[name] - labels)
            raise ConfigError(
                f"keyword taxonomy labels differ for {name}: missing={missing}, extra={extra}"
            )
    decision = raw.get("decision")
    if not isinstance(decision, dict):
        raise ConfigError("keyword taxonomy profile requires a [decision] table")
    decision_field = str(decision.get("field", "")).strip()
    field = next((field for field in fields if field.name == decision_field), None)
    if field is None or field.mode != "exclusive":
        raise ConfigError("keyword decision.field must name an exclusive configured field")
    include = decision.get("include_labels")
    exclude = decision.get("exclude_labels")
    if not isinstance(include, list) or not isinstance(exclude, list):
        raise ConfigError("keyword decision label sets must be arrays")
    include_labels = tuple(str(label) for label in include)
    exclude_labels = tuple(str(label) for label in exclude)
    known_labels = set(field.labels)
    unknown = (set(include_labels) | set(exclude_labels)) - known_labels
    if unknown or not include_labels or not exclude_labels or set(include_labels) & set(exclude_labels):
        raise ConfigError("keyword decision labels must be known, non-empty, and disjoint")
    return KeywordTaxonomyProfile(
        source, tuple(fields), decision_field, include_labels, exclude_labels, raw, path,
    )


def keyword_profile_hash(
    source: str, term_profile: dict[str, Any], config: KeywordBaselineConfig,
    taxonomy_profile: KeywordTaxonomyProfile,
) -> str:
    """Fingerprint all semantics that can alter a keyword baseline decision."""
    payload = {
        "version": BASELINE_VERSION,
        "evidence_policy": COLLECTED_EVIDENCE_POLICY,
        "source": source,
        "term_profile": term_profile,
        "accept_threshold": config.accept_threshold,
        "uncertain_threshold": config.uncertain_threshold,
        "taxonomy_profile": taxonomy_profile.raw,
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def infer_keyword_fields(
    evidence: str, profile: KeywordTaxonomyProfile,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], dict[str, dict[str, list[str]]]]:
    """Classify every configured field from transparent weighted concept matches."""
    predictions: dict[str, Any] = {}
    scores: dict[str, dict[str, float]] = {}
    signals: dict[str, dict[str, list[str]]] = {}
    normalized_evidence = f" {normalize_text(evidence)} "
    for field in profile.fields:
        field_scores: dict[str, float] = {}
        field_signals: dict[str, list[str]] = {}
        for label, rule in field.labels.items():
            hits = [
                term for term in rule["terms"]
                if f" {normalize_text(term)} " in normalized_evidence
            ]
            field_signals[label] = hits
            field_scores[label] = round(len(hits) * float(rule["weight"]), 3)
        scores[field.name] = field_scores
        signals[field.name] = field_signals
        if field.mode == "exclusive":
            ranked = sorted(field_scores, key=lambda label: (-field_scores[label], label))
            best = ranked[0]
            second_score = field_scores[ranked[1]] if len(ranked) > 1 else 0.0
            predictions[field.name] = best if (
                field_scores[best] >= field.minimum_score
                and field_scores[best] - second_score >= field.margin
            ) else field.fallback
        else:
            selected = sorted(
                label for label, score in field_scores.items() if score >= field.minimum_score
            )
            predictions[field.name] = selected or [field.fallback]
    return predictions, scores, signals


def score_keyword_evidence(
    evidence: str, term_profile: dict[str, Any], accept_threshold: float,
    uncertain_threshold: float,
) -> ScoreResult:
    """Score broad relevance with normalized exact phrases for the classical baseline."""
    normalized_evidence = f" {normalize_text(evidence)} "
    score = 0.0
    signals: list[str] = []
    for section in ("positive", "negative"):
        groups = term_profile.get(section, {})
        if not isinstance(groups, dict):
            continue
        for name, group in groups.items():
            if not isinstance(group, dict):
                continue
            terms = group.get("terms", [])
            if not isinstance(terms, list):
                continue
            hits = [
                term for term in terms
                if isinstance(term, str) and f" {normalize_text(term)} " in normalized_evidence
            ]
            if not hits:
                continue
            weight = float(group.get("weight", 0.0))
            score += weight
            signals.append(f"{name}:{'|'.join(hits)}:{weight:+g}")
    band = (
        "candidate" if score >= accept_threshold
        else "borderline" if score >= uncertain_threshold else "reject"
    )
    return ScoreResult(round(score, 3), band, tuple(signals))


def _keyword_decision(
    predictions: dict[str, Any], profile: KeywordTaxonomyProfile, screening_band: str,
) -> str:
    """Map the configured class field to a decision, retaining the relevance-score gate."""
    label = str(predictions.get(profile.decision_field, ""))
    if label in profile.exclude_labels:
        return "exclude"
    if label in profile.include_labels:
        return "accept" if screening_band == "candidate" else "uncertain"
    return "exclude" if screening_band == "reject" else "uncertain"


def _source_contract(source: str) -> tuple[str, str, list[str]]:
    """Resolve stable IDs, raw-cache namespaces, and CSV columns for one source."""
    if source == "github":
        from radar.github.schema import CLASSIFICATION_COLUMNS

        return "github_repo_id", "github.repository", CLASSIFICATION_COLUMNS
    if source == "openalex":
        from radar.openalex.schema import CLASSIFICATION_COLUMNS

        return "openalex_id", "openalex.work", CLASSIFICATION_COLUMNS
    raise ValueError(f"unsupported keyword baseline source: {source}")


def baseline_inputs(source: str, candidates_path: Path, cache: PipelineCache,
                    limit: int | None = None) -> list[tuple[dict[str, str], dict[str, Any]]]:
    """Resolve every selected candidate to original evidence before any inference.

    The CSV defines the cohort only. Its text and the mutable working cache may
    already include RADAR enrichment, so neither is a baseline evidence fallback.
    """
    identity, namespace, _ = _source_contract(source)
    candidates = read_csv(candidates_path)
    require_columns(candidates, (identity, "evidence_hash"), candidates_path)
    if limit is not None:
        if limit < 1:
            raise ValueError("baseline limit must be positive")
        candidates = candidates[:limit]
    inputs = []
    for index, candidate in enumerate(candidates, 1):
        key = str(candidate.get(identity, "")).strip()
        if not key:
            raise ValueError(f"baseline candidate is missing {identity}: row {index}")
        inputs.append((candidate, cache.get_collected_record(namespace, key)))
    return inputs


def _evidence_text(source: str, payload: dict[str, Any]) -> str:
    """Format the same bounded source evidence used by prefiltering and LLMs."""
    if source == "github":
        from radar.github.pipeline import evidence_sections
    elif source == "openalex":
        from radar.openalex.pipeline import evidence_sections
    else:
        raise ValueError(f"unsupported keyword baseline source: {source}")
    return format_evidence(evidence_sections(payload), 30000)


def classify_keyword_baseline(
    source: str,
    candidates_path: Path,
    output: Path,
    cache: PipelineCache,
    term_profile: dict[str, Any],
    config: KeywordBaselineConfig,
    taxonomy_profile: KeywordTaxonomyProfile,
    *,
    resume: bool = True,
    progress_every: int = 25,
    adapter_config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Classify every candidate row with an independent, cached keyword decision rule."""
    identity, _, columns = _source_contract(source)
    inputs = baseline_inputs(source, candidates_path, cache)
    if taxonomy_profile.source != source:
        raise ValueError(f"keyword taxonomy source must be {source!r}")
    profile_hash = keyword_profile_hash(source, term_profile, config, taxonomy_profile)
    decision_namespace = f"{source}.baseline.keyword"
    results: list[dict[str, str]] = []
    reused = 0
    progress_every = max(1, int(progress_every))
    LOG.info(
        "starting keyword baseline source=%s rows=%d accept_threshold=%g uncertain_threshold=%g resume=%s evidence_source=collection_snapshot",
        source, len(inputs), config.accept_threshold, config.uncertain_threshold, resume,
    )
    for index, (candidate, payload) in enumerate(inputs, 1):
        key = str(candidate[identity]).strip()
        evidence_source = "collection_snapshot"
        evidence = _evidence_text(source, payload)
        evidence_hash = stable_hash(collected_record_hash(payload), evidence)
        cached = cache.get_decision(decision_namespace, key, profile_hash, evidence_hash) if resume else None
        if cached is not None:
            results.append({str(name): str(value) for name, value in cached.items()})
            reused += 1
            continue
        scored = score_keyword_evidence(
            evidence, term_profile, config.accept_threshold, config.uncertain_threshold,
        )
        predictions, taxonomy_scores, taxonomy_signals = infer_keyword_fields(
            evidence, taxonomy_profile,
        )
        predictions["decision"] = _keyword_decision(predictions, taxonomy_profile, scored.band)
        if source == "github":
            from radar.github.baseline import normalize_keyword_result
        else:
            from radar.openalex.baseline import normalize_keyword_result
        normalized = normalize_keyword_result(predictions, candidate, adapter_config or {})
        decision = normalized["decision"]
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
            "classifier_supported_fields": normalized["classifier_supported_fields"],
            "classifier_evidence_source": evidence_source,
            "classifier_evidence_policy": COLLECTED_EVIDENCE_POLICY,
            "classifier_evidence_snapshot_hash": collected_record_hash(payload),
            **normalized,
            "decision": decision,
            "primary_reason": (
                f"Keyword taxonomy selected {taxonomy_profile.decision_field}="
                f"{predictions.get(taxonomy_profile.decision_field)}; relevance score "
                f"{scored.score:g} produced band={scored.band}."
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
            "keyword_taxonomy_scores_json": json.dumps(taxonomy_scores, sort_keys=True),
            "keyword_taxonomy_signals_json": json.dumps(taxonomy_signals, ensure_ascii=False, sort_keys=True),
        })
        serialized = {str(name): str(value) for name, value in result.items()}
        cache.put_decision(decision_namespace, key, profile_hash, evidence_hash, serialized, classified_at)
        results.append(serialized)
        if index % progress_every == 0:
            atomic_write_csv(output, results, columns)
            LOG.info("keyword baseline progress source=%s completed=%d/%d", source, index, len(inputs))
    atomic_write_csv(output, results, columns)
    counts: dict[str, int] = {}
    for row in results:
        decision = row.get("decision", "unavailable")
        counts[decision] = counts.get(decision, 0) + 1
    LOG.info(
        "keyword baseline complete source=%s rows=%d cached=%d evidence_source=collection_snapshot decisions=%s output=%s",
        source, len(results), reused, counts, output,
    )
    return results
