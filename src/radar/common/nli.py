from __future__ import annotations

import json
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from radar.common.baselines import baseline_inputs
from radar.common.cache import COLLECTED_EVIDENCE_POLICY, PipelineCache, collected_record_hash
from radar.common.config import ConfigError, Settings
from radar.common.csvio import atomic_write_csv, read_csv, require_columns
from radar.common.logging import logger
from radar.common.ollama import EvidenceSection, format_evidence
from radar.common.run import utc_now
from radar.common.text import slug, stable_hash


LOG = logger("nli")
NLI_BASELINE_VERSION = "nli-baseline-3"
NLI_AUDIT_COLUMNS = [
    "nli_model", "nli_model_revision", "nli_scores_json", "nli_supporting_chunks_json",
    "nli_entailment_threshold", "nli_label_margin", "nli_top_k", "nli_max_chunks",
    "nli_inference_seconds",
]


@dataclass(frozen=True)
class NLIRuntimeConfig:
    """Validated local NLI model and inference parameters."""

    model: str
    revision: str
    device: str
    batch_size: int
    max_length: int
    premise_tokens: int
    chunk_overlap: int
    max_chunks: int
    top_k: int
    entailment_threshold: float
    label_margin: float


@dataclass(frozen=True)
class NLIField:
    """One configured semantic field and its natural-language hypotheses."""

    name: str
    mode: str
    fallback: str
    hypotheses: dict[str, str]


@dataclass(frozen=True)
class NLIProfile:
    """Validated source-specific zero-shot NLI classification profile."""

    source: str
    fields: tuple[NLIField, ...]
    raw: dict[str, Any]
    path: Path


@dataclass(frozen=True)
class EvidenceChunk:
    """One deterministic, source-labelled premise chunk."""

    index: int
    source: str
    text: str


@dataclass(frozen=True)
class NLIScore:
    """Normalized three-way NLI probabilities for one premise/hypothesis pair."""

    entailment: float
    neutral: float
    contradiction: float


class NLIBackend(Protocol):
    """Backend contract used by production Transformers and offline fake models."""

    model_id: str
    revision: str

    def chunk_sections(
        self, sections: Sequence[EvidenceSection], premise_tokens: int, overlap: int,
    ) -> list[EvidenceChunk]:
        """Split evidence into deterministic model-token chunks."""
        ...

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[NLIScore]:
        """Score premise/hypothesis pairs in their supplied order."""
        ...


def load_nli_runtime_config(settings: Settings) -> NLIRuntimeConfig:
    """Load and validate the shared local NLI runtime configuration."""
    baselines = settings.section("baselines")
    raw = baselines.get("nli", {})
    if not isinstance(raw, dict):
        raise ConfigError("baselines.nli must be a table")
    model = str(raw.get("model", "")).strip()
    revision = str(raw.get("revision", "")).strip()
    device = str(raw.get("device", "auto")).strip().casefold()
    if not model or not revision:
        raise ConfigError("baselines.nli.model and baselines.nli.revision are required")
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigError("baselines.nli.device must be auto, cpu, or cuda")
    try:
        config = NLIRuntimeConfig(
            model=model,
            revision=revision,
            device=device,
            batch_size=int(raw.get("batch_size", 8)),
            max_length=int(raw.get("max_length", 512)),
            premise_tokens=int(raw.get("premise_tokens", 384)),
            chunk_overlap=int(raw.get("chunk_overlap", 48)),
            max_chunks=int(raw.get("max_chunks", 6)),
            top_k=int(raw.get("top_k", 2)),
            entailment_threshold=float(raw.get("entailment_threshold", 0.5)),
            label_margin=float(raw.get("label_margin", 0.1)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("baselines.nli numeric settings are invalid") from exc
    if config.batch_size < 1 or config.max_length < 32 or config.premise_tokens < 16:
        raise ConfigError("baselines.nli batch and token limits must be positive")
    if config.premise_tokens >= config.max_length:
        raise ConfigError("baselines.nli.premise_tokens must be smaller than max_length")
    if config.chunk_overlap < 0 or config.chunk_overlap >= config.premise_tokens:
        raise ConfigError("baselines.nli.chunk_overlap must be in [0, premise_tokens)")
    if config.top_k < 1 or config.max_chunks < 1:
        raise ConfigError("baselines.nli.top_k and max_chunks must be at least 1")
    if not 0 < config.entailment_threshold < 1 or not 0 <= config.label_margin < 1:
        raise ConfigError("baselines.nli thresholds must lie between zero and one")
    return config


def load_nli_profile(settings: Settings, source: str) -> NLIProfile:
    """Load a complete source hypothesis profile selected by the main configuration."""
    path = settings.path(f"{source}_nli")
    if not path.is_file():
        raise ConfigError(f"NLI profile does not exist: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if str(raw.get("schema_version", "")) != "1.0" or str(raw.get("source", "")) != source:
        raise ConfigError(f"NLI profile must use schema_version 1.0 and source {source!r}: {path}")
    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ConfigError(f"NLI profile requires a non-empty [fields] table: {path}")
    fields: list[NLIField] = []
    for name, value in raw_fields.items():
        if not isinstance(value, dict):
            raise ConfigError(f"NLI field {name!r} must be a table")
        mode = str(value.get("mode", "")).strip()
        fallback = str(value.get("fallback", "")).strip()
        hypotheses = value.get("hypotheses")
        if mode not in {"exclusive", "multilabel", "boolean"}:
            raise ConfigError(f"NLI field {name!r} has unsupported mode {mode!r}")
        if not isinstance(hypotheses, dict) or not hypotheses:
            raise ConfigError(f"NLI field {name!r} requires hypotheses")
        normalized = {str(label): str(text).strip() for label, text in hypotheses.items()}
        if any(not label or not text for label, text in normalized.items()):
            raise ConfigError(f"NLI field {name!r} contains an empty label or hypothesis")
        if mode == "boolean" and set(normalized) != {"true"}:
            raise ConfigError(f"boolean NLI field {name!r} requires exactly one true hypothesis")
        if mode != "boolean" and not fallback:
            raise ConfigError(f"NLI field {name!r} requires a fallback")
        fields.append(NLIField(str(name), mode, fallback, normalized))
    return NLIProfile(source, tuple(fields), raw, path)


class TransformersNLIBackend:
    """Lazy local Hugging Face sequence-classification backend for three-way NLI."""

    def __init__(self, config: NLIRuntimeConfig):
        """Load a pinned model and map its semantic output labels explicitly."""
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "NLI dependencies are missing; install the project with the 'nli' extra"
            ) from exc
        self._torch = torch
        self.model_id = config.model
        self.revision = config.revision
        self.max_length = config.max_length
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model, revision=config.revision,
        )
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("baselines.nli.device=cuda but CUDA is unavailable")
        self.device = "cuda" if config.device == "cuda" or (
            config.device == "auto" and torch.cuda.is_available()
        ) else "cpu"
        self.model.to(self.device)
        self.model.eval()
        labels = {
            str(label).casefold(): int(index)
            for index, label in dict(self.model.config.id2label).items()
        }
        self.entailment_id = next((index for label, index in labels.items() if "entail" in label), -1)
        self.neutral_id = next((index for label, index in labels.items() if "neutral" in label), -1)
        self.contradiction_id = next((index for label, index in labels.items() if "contrad" in label), -1)
        if min(self.entailment_id, self.neutral_id, self.contradiction_id) < 0:
            raise RuntimeError(f"NLI model exposes unsupported labels: {self.model.config.id2label}")

    def chunk_sections(
        self, sections: Sequence[EvidenceSection], premise_tokens: int, overlap: int,
    ) -> list[EvidenceChunk]:
        """Split each evidence section by tokenizer IDs without crossing source boundaries."""
        return chunk_nli_sections(self.tokenizer, sections, premise_tokens, overlap)

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[NLIScore]:
        """Return normalized three-way probabilities for batched premise/hypothesis pairs."""
        results: list[NLIScore] = []
        torch = self._torch
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            encoded = self.tokenizer(
                [premise for premise, _hypothesis in batch],
                [hypothesis for _premise, hypothesis in batch],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            with torch.no_grad():
                probabilities = torch.softmax(self.model(**encoded).logits, dim=-1).detach().cpu()
            for values in probabilities:
                results.append(NLIScore(
                    entailment=float(values[self.entailment_id]),
                    neutral=float(values[self.neutral_id]),
                    contradiction=float(values[self.contradiction_id]),
                ))
        return results


def chunk_nli_sections(
    tokenizer: Any, sections: Sequence[EvidenceSection], premise_tokens: int, overlap: int,
) -> list[EvidenceChunk]:
    """Share identical reference-tokenizer chunk texts across NLI backends."""
    chunks: list[EvidenceChunk] = []
    step = premise_tokens - overlap
    for section in sections:
        token_ids = tokenizer.encode(str(section.text), add_special_tokens=False, verbose=False)
        for start in range(0, len(token_ids), step):
            selected = token_ids[start:start + premise_tokens]
            if not selected:
                continue
            text = tokenizer.decode(selected, skip_special_tokens=True).strip()
            if text:
                chunks.append(EvidenceChunk(len(chunks), section.source, text))
            if start + premise_tokens >= len(token_ids):
                break
    return chunks or [EvidenceChunk(0, "Evidence", "No substantive evidence was available.")]


def _aggregate(values: list[tuple[EvidenceChunk, NLIScore]], top_k: int) -> dict[str, Any]:
    """Aggregate strongest entailment and contradiction evidence while retaining support."""
    strongest_entailment = sorted(values, key=lambda item: item[1].entailment, reverse=True)[:top_k]
    strongest_contradiction = sorted(values, key=lambda item: item[1].contradiction, reverse=True)[:top_k]
    entailment = sum(item[1].entailment for item in strongest_entailment) / len(strongest_entailment)
    contradiction = sum(item[1].contradiction for item in strongest_contradiction) / len(strongest_contradiction)
    neutral = sum(item[1].neutral for item in strongest_entailment) / len(strongest_entailment)
    support = [{
        "chunk_index": chunk.index, "source": chunk.source,
        "entailment": round(score.entailment, 6),
        "excerpt": " ".join(chunk.text.split())[:500],
    } for chunk, score in strongest_entailment]
    return {
        "entailment": round(entailment, 6), "neutral": round(neutral, 6),
        "contradiction": round(contradiction, 6), "support": support,
    }


def _representative_chunks(chunks: list[EvidenceChunk], maximum: int) -> list[EvidenceChunk]:
    """Bound inference cost while retaining the start, end, and middle of each source."""
    if len(chunks) <= maximum:
        return chunks
    groups: dict[str, list[EvidenceChunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.source, []).append(chunk)
    candidates: list[EvidenceChunk] = []
    for round_index in range(max(len(values) for values in groups.values())):
        for values in groups.values():
            positions = [0, len(values) - 1]
            if len(values) > 2:
                middle = len(values) // 2
                positions.append(middle)
                positions.extend(
                    index for index in range(1, len(values) - 1) if index != middle
                )
            if round_index < len(positions):
                chunk = values[positions[round_index]]
                if chunk not in candidates:
                    candidates.append(chunk)
            if len(candidates) >= maximum:
                return sorted(candidates, key=lambda value: value.index)
    return sorted(candidates[:maximum], key=lambda value: value.index)


def infer_nli_fields(
    sections: Sequence[EvidenceSection], profile: NLIProfile, backend: NLIBackend,
    config: NLIRuntimeConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Score every configured hypothesis and convert scores into abstention-aware fields."""
    all_chunks = backend.chunk_sections(sections, config.premise_tokens, config.chunk_overlap)
    chunks = _representative_chunks(all_chunks, config.max_chunks)
    index: list[tuple[str, str, EvidenceChunk]] = []
    pairs: list[tuple[str, str]] = []
    for field in profile.fields:
        for label, hypothesis in field.hypotheses.items():
            for chunk in chunks:
                premise = f"Evidence source: {chunk.source}.\n{chunk.text}"
                index.append((field.name, label, chunk))
                pairs.append((premise, hypothesis))
    scores = backend.score(pairs, config.batch_size)
    if len(scores) != len(index):
        raise RuntimeError("NLI backend returned a different number of scores than requested")
    grouped: dict[str, dict[str, list[tuple[EvidenceChunk, NLIScore]]]] = {}
    for (field, label, chunk), score in zip(index, scores, strict=True):
        grouped.setdefault(field, {}).setdefault(label, []).append((chunk, score))
    audit: dict[str, Any] = {"_evidence": {
        "total_chunks": len(all_chunks), "used_chunks": len(chunks),
        "sources": [chunk.source for chunk in chunks],
    }}
    support: dict[str, Any] = {}
    predictions: dict[str, Any] = {}
    for field in profile.fields:
        field_scores = {
            label: _aggregate(values, config.top_k)
            for label, values in grouped[field.name].items()
        }
        audit[field.name] = {
            label: {key: value for key, value in values.items() if key != "support"}
            for label, values in field_scores.items()
        }
        support[field.name] = {label: values["support"] for label, values in field_scores.items()}
    predictions.update(decide_nli_fields(
        audit, profile, config.entailment_threshold, config.label_margin,
    ))
    return predictions, audit, support


def decide_nli_fields(
    audit: dict[str, Any], profile: NLIProfile, entailment_threshold: float,
    label_margin: float,
) -> dict[str, Any]:
    """Convert persisted aggregate scores into labels without running NLI inference."""
    if not 0 < entailment_threshold < 1 or not 0 <= label_margin < 1:
        raise ValueError("NLI thresholds must lie between zero and one")
    predictions: dict[str, Any] = {}
    for field in profile.fields:
        raw_scores = audit.get(field.name)
        if not isinstance(raw_scores, dict) or set(raw_scores) != set(field.hypotheses):
            raise ValueError(f"persisted NLI scores do not match field {field.name!r}")
        field_scores: dict[str, dict[str, float]] = {}
        for label, raw_values in raw_scores.items():
            if not isinstance(raw_values, dict):
                raise ValueError(f"persisted NLI score for {field.name}.{label} is invalid")
            try:
                field_scores[str(label)] = {
                    "entailment": float(raw_values["entailment"]),
                    "contradiction": float(raw_values["contradiction"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"persisted NLI score for {field.name}.{label} is incomplete"
                ) from exc
        if field.mode == "exclusive":
            ranked = sorted(
                field_scores, key=lambda label: field_scores[label]["entailment"], reverse=True,
            )
            best = ranked[0]
            second = field_scores[ranked[1]]["entailment"] if len(ranked) > 1 else 0.0
            best_score = field_scores[best]["entailment"]
            predictions[field.name] = best if (
                best_score >= entailment_threshold and best_score - second >= label_margin
            ) else field.fallback
        elif field.mode == "multilabel":
            selected = sorted(
                label for label, values in field_scores.items()
                if values["entailment"] >= entailment_threshold
                and values["entailment"] - values["contradiction"] >= label_margin
            )
            predictions[field.name] = selected or ([field.fallback] if field.fallback else [])
        else:
            values = field_scores["true"]
            if values["entailment"] >= entailment_threshold and (
                values["entailment"] - values["contradiction"] >= label_margin
            ):
                predictions[field.name] = "true"
            elif values["contradiction"] >= entailment_threshold and (
                values["contradiction"] - values["entailment"] >= label_margin
            ):
                predictions[field.name] = "false"
            else:
                predictions[field.name] = "unknown"
    return predictions


def rescore_nli_rows(
    source: str, rows: Sequence[dict[str, str]], profile: NLIProfile,
    entailment_threshold: float, label_margin: float,
    adapter_config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Replay saved NLI scores at new thresholds without model or network access."""
    _identity, _namespace, _columns, _evidence_builder, normalizer = _source_contract(source)
    rescored: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        try:
            raw_audit = json.loads(str(row.get("nli_scores_json", "")))
            if not isinstance(raw_audit, dict) or not raw_audit:
                raise ValueError("NLI score audit is empty")
            predictions = decide_nli_fields(
                raw_audit, profile, entailment_threshold, label_margin,
            )
            updated.update(normalizer(predictions, row, adapter_config or {}))
            if row.get("classifier_method") == "llm_nli":
                updated["classifier_supported_fields"] = row["classifier_supported_fields"]
            updated["nli_entailment_threshold"] = f"{entailment_threshold:g}"
            updated["nli_label_margin"] = f"{label_margin:g}"
            updated["error"] = ""
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            updated.update({
                "decision": "error", "classification_validation_flags": "nli_rescore_error",
                "primary_reason": "Saved NLI scores could not be rescored.",
                "error": str(exc)[:1000],
            })
        rescored.append({str(name): str(value) for name, value in updated.items()})
    return rescored


def nli_profile_hash(profile: NLIProfile, config: NLIRuntimeConfig) -> str:
    """Fingerprint model revision, hypotheses, thresholds, and chunking semantics."""
    payload = {
        "version": NLI_BASELINE_VERSION, "evidence_policy": COLLECTED_EVIDENCE_POLICY, "profile": profile.raw, "runtime": asdict(config),
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _source_contract(source: str) -> tuple[str, str, list[str], Any, Any]:
    """Load one source's evidence builder and NLI normalization adapter lazily."""
    if source == "github":
        from radar.github.baseline import normalize_nli_result
        from radar.github.pipeline import evidence_sections
        from radar.github.schema import CLASSIFICATION_COLUMNS

        return "github_repo_id", "github.repository", CLASSIFICATION_COLUMNS, evidence_sections, normalize_nli_result
    if source == "openalex":
        from radar.openalex.baseline import normalize_nli_result
        from radar.openalex.pipeline import evidence_sections
        from radar.openalex.schema import CLASSIFICATION_COLUMNS

        return "openalex_id", "openalex.work", CLASSIFICATION_COLUMNS, evidence_sections, normalize_nli_result
    raise ValueError(f"unsupported NLI baseline source: {source}")


def classify_nli_baseline(
    source: str, candidates_path: Path, output: Path, cache: PipelineCache,
    profile: NLIProfile, config: NLIRuntimeConfig, *, resume: bool = True,
    progress_every: int = 25, backend: NLIBackend | None = None,
    adapter_config: dict[str, Any] | None = None,
    method: str = "nli", limit: int | None = None,
) -> list[dict[str, str]]:
    """Run a resumable zero-shot NLI classification over the complete candidate audit."""
    identity, _, columns, evidence_builder, normalizer = _source_contract(source)
    inputs = baseline_inputs(source, candidates_path, cache, limit)
    if method not in {"nli", "llm_nli"}:
        raise ValueError(f"unsupported NLI method: {method}")
    if method == "llm_nli":
        if backend is None:
            raise ValueError("llm_nli requires an explicit scoring backend")
        from radar.common.llm_nli import LLM_NLI_AUDIT_COLUMNS

        columns = [*columns, *LLM_NLI_AUDIT_COLUMNS]
    active_backend = backend
    if active_backend is not None and (
        active_backend.model_id != config.model or active_backend.revision != config.revision
    ):
        raise ValueError("NLI backend identity does not match the configured model and revision")
    profile_hash = nli_profile_hash(profile, config)
    namespace = f"{source}.baseline.{method}.{slug(config.model)}"
    results: list[dict[str, str]] = []
    prior_rows = read_csv(output) if resume and output.is_file() else []
    prior_by_id = {
        str(row.get(identity, "")).strip(): row for row in prior_rows
        if str(row.get(identity, "")).strip()
        and row.get("classifier_profile_hash") == profile_hash
        and row.get("decision") != "error"
    }
    reused = 0
    progress_every = max(1, int(progress_every))
    LOG.info(
        "starting NLI baseline source=%s rows=%d model=%s revision=%s device=%s resume=%s evidence_source=collection_snapshot",
        source, len(inputs), config.model, config.revision, config.device, resume,
    )
    for index, (candidate, payload) in enumerate(inputs, 1):
        key = str(candidate[identity]).strip()
        evidence_source = "collection_snapshot"
        sections = evidence_builder(payload)
        evidence_hash = (
            stable_hash(json.dumps([(section.source, section.text) for section in sections], ensure_ascii=False))
            if method == "llm_nli" else stable_hash(format_evidence(sections, 30000))
        )
        evidence_hash = stable_hash(collected_record_hash(payload), evidence_hash)
        cached = cache.get_decision(namespace, key, profile_hash, evidence_hash) if resume else None
        if cached is None:
            prior = prior_by_id.get(key)
            cached = prior if prior is not None and prior.get("evidence_hash") == evidence_hash else None
        if cached is not None and cached.get("decision") != "error":
            results.append({str(name): str(value) for name, value in cached.items()})
            reused += 1
            continue
        started = time.monotonic()
        result: dict[str, Any] = dict(candidate)
        result.update({
            "evidence_hash": evidence_hash, "pipeline_version": NLI_BASELINE_VERSION,
            "profile_hash": profile_hash, "llm_model": "",
            "classifier_method": method, "classifier_id": config.model,
            "classifier_profile_hash": profile_hash,
            "classifier_evidence_source": evidence_source,
            "classifier_evidence_policy": COLLECTED_EVIDENCE_POLICY,
            "classifier_evidence_snapshot_hash": collected_record_hash(payload),
            "nli_model": config.model, "nli_model_revision": config.revision,
            "nli_entailment_threshold": f"{config.entailment_threshold:g}",
            "nli_label_margin": f"{config.label_margin:g}", "nli_top_k": str(config.top_k),
            "nli_max_chunks": str(config.max_chunks),
            "classified_at": utc_now(), "error": "",
        })
        try:
            if active_backend is None:
                active_backend = TransformersNLIBackend(config)
            if method == "llm_nli":
                active_backend.start_record()
            predictions, scores, support = infer_nli_fields(sections, profile, active_backend, config)
            normalized = normalizer(predictions, candidate, adapter_config or {})
            result.update(normalized)
            result["nli_scores_json"] = json.dumps(scores, ensure_ascii=False, sort_keys=True)
            result["nli_supporting_chunks_json"] = json.dumps(support, ensure_ascii=False, sort_keys=True)
        except Exception as exc:
            result.update({
                "decision": "error", "classification_validation_flags": "nli_inference_error",
                "primary_reason": "NLI inference failed.", "error": str(exc)[:1000],
                "classifier_supported_fields": "decision", "nli_scores_json": "{}",
                "nli_supporting_chunks_json": "{}",
            })
        result["nli_inference_seconds"] = f"{time.monotonic() - started:.6f}"
        if method == "llm_nli":
            result["classifier_supported_fields"] = ";".join(["decision", *[field.name for field in profile.fields]])
            result["nli_entailment_threshold"] = repr(config.entailment_threshold)
            result["nli_label_margin"] = repr(config.label_margin)
            result.update(active_backend.record_audit())
        serialized = {str(name): str(value) for name, value in result.items()}
        if serialized.get("decision") != "error":
            cache.put_decision(namespace, key, profile_hash, evidence_hash, serialized, serialized["classified_at"])
        results.append(serialized)
        if index % progress_every == 0:
            atomic_write_csv(output, results, columns)
            LOG.info("NLI baseline progress source=%s completed=%d/%d", source, index, len(inputs))
    atomic_write_csv(output, results, columns)
    LOG.info(
        "NLI baseline complete source=%s rows=%d cached=%d evidence_source=collection_snapshot output=%s",
        source, len(results), reused, output,
    )
    return results
