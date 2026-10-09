from __future__ import annotations

import json
import math
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from string import Template
from typing import Any, Sequence

import requests

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError, Settings
from ids_meta.common.nli import EvidenceChunk, NLIScore, NLIProfile, NLIRuntimeConfig, chunk_nli_sections
from ids_meta.common.ollama import EvidenceSection
from ids_meta.common.run import utc_now
from ids_meta.common.text import slug, stable_hash

VERSION = "ollama-nli-1"
LLM_NLI_AUDIT_COLUMNS = ["llm_nli_metadata_json", "llm_nli_pair_scores_json", "nli_evidence_chunks_json"]
CLASS_CODES = {"entailment": "A", "neutral": "B", "contradiction": "C"}


class NLIScoringError(RuntimeError):
    """The server did not return usable, complete class-token probabilities."""


@dataclass(frozen=True)
class LLMNLIConfig:
    models: tuple[str, ...]
    template: str
    top_logprobs: int = 20
    context_length: int = 4096
    seed: int = 42
    timeout_seconds: float = 300
    retries: int = 2
    entailment_threshold: float = .5
    label_margin: float = .1
    minimum_label_mass: float = 0

    def scoring_hash(self, reference: NLIRuntimeConfig) -> str:
        """Exclude decision thresholds from the reusable pair-score fingerprint."""
        return stable_hash(json.dumps({
            "version": VERSION, "template": self.template, "codes": CLASS_CODES,
            "top_logprobs": self.top_logprobs, "num_ctx": self.context_length,
            "seed": self.seed, "temperature": 0, "think": False,
            "reference_tokenizer": [reference.model, reference.revision],
            "minimum_label_mass": self.minimum_label_mass,
        }, sort_keys=True))


def load_llm_nli_config(settings: Settings) -> LLMNLIConfig:
    """Load the versioned prompt and validate the opt-in scoring configuration."""
    raw = settings.section("baselines").get("llm_nli", {})
    if not isinstance(raw, dict):
        raise ConfigError("baselines.llm_nli must be a table")
    models = raw.get("models", [])
    if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ConfigError("baselines.llm_nli.models must be a non-empty model array")
    models = list(dict.fromkeys(m.strip() for m in models))
    if len({slug(model) for model in models}) != len(models):
        raise ConfigError("LLM NLI models must have distinct output directory slugs")
    path = settings.section("paths").get("llm_nli_prompt", "config/prompts/llm_nli.toml")
    path = Path(path)
    path = path if path.is_absolute() else settings.root / path
    with path.open("rb") as handle:
        prompt = tomllib.load(handle)
    template = prompt.get("template", "")
    expected = {"premise", "hypothesis", "entailment_code", "neutral_code", "contradiction_code"}
    if prompt.get("schema_version") != "1.0" or not isinstance(template, str) or not Template(template).is_valid() or set(Template(template).get_identifiers()) != expected:
        raise ConfigError("LLM NLI prompt must contain exactly the premise, hypothesis and three class-code placeholders")
    try:
        config = LLMNLIConfig(
            models=tuple(models), template=template,
            **{name: int(raw.get(name, default)) for name, default in [("top_logprobs", 20), ("context_length", 4096), ("seed", 42), ("retries", 2)]},
            **{name: float(raw.get(name, default)) for name, default in [("timeout_seconds", 300), ("entailment_threshold", .5), ("label_margin", .1), ("minimum_label_mass", 0)]},
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError("LLM NLI numeric settings are invalid") from exc
    if not 3 <= config.top_logprobs <= 20 or config.context_length < 512 or config.retries < 0 or not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ConfigError("LLM NLI requires top_logprobs in [3,20], context_length >=512 and valid retry/timeout settings")
    if not 0 < config.entailment_threshold < 1 or not 0 <= config.label_margin < 1 or not 0 <= config.minimum_label_mass <= 1:
        raise ConfigError("LLM NLI thresholds must lie in their probability ranges")
    return config


def parse_class_scores(response: dict[str, Any], minimum_mass: float = 0) -> tuple[NLIScore, dict[str, Any]]:
    """Normalize all returned one-token class verbalizers at one output position."""
    message = response.get("message", {})
    if not isinstance(message, dict):
        raise NLIScoringError("Malformed Ollama message")
    answer = str(message.get("content", "")).strip()
    if message.get("thinking") or answer not in CLASS_CODES.values():
        raise NLIScoringError("Expected one A/B/C token with thinking disabled")
    entries = response.get("logprobs")
    if not isinstance(entries, list) or not entries:
        raise NLIScoringError("Ollama did not return token logprobs; check server/model support")
    position = None
    for index, entry in enumerate(entries):
        token = entry.get("token") if isinstance(entry, dict) else None
        if not isinstance(token, str):
            raise NLIScoringError("Malformed output token")
        if not token.strip():
            continue
        if token.strip() != answer:
            raise NLIScoringError("Class code is not a single token at the first non-whitespace position")
        position = index
        break
    if position is None:
        raise NLIScoringError("No decision token found")
    entry = entries[position]
    alternatives = entry.get("top_logprobs")
    if not isinstance(alternatives, list):
        raise NLIScoringError("Ollama did not return alternative token scores")
    tokens: dict[str, float] = {}
    seen_alternatives: set[str] = set()
    for index, item in enumerate([entry, *alternatives]):
        if not isinstance(item, dict) or not isinstance(item.get("token"), str):
            raise NLIScoringError("Malformed token alternative")
        token = item["token"]
        if index:
            if token in seen_alternatives:
                raise NLIScoringError("Duplicate alternative token scores")
            seen_alternatives.add(token)
        try:
            logprob = float(item["logprob"])
        except (KeyError, ValueError, TypeError) as exc:
            raise NLIScoringError("Missing numeric token logprob") from exc
        if not math.isfinite(logprob) or logprob > 1e-6:
            raise NLIScoringError("Invalid token log probability")
        if token in tokens:
            if token != entry["token"] or not math.isclose(tokens[token], logprob, abs_tol=1e-6):
                raise NLIScoringError("Duplicate or conflicting alternative token scores")
            # The selected token is normally also present in top_logprobs.
            continue
        tokens[token] = logprob
    grouped = {code: [value for token, value in tokens.items() if token.strip() == code] for code in CLASS_CODES.values()}
    if any(not values for values in grouped.values()):
        raise NLIScoringError("Incomplete A/B/C scores: a required class is outside top_logprobs; no zero-imputation is allowed")
    maximum = max(value for values in grouped.values() for value in values)
    weights = {code: sum(math.exp(value - maximum) for value in values) for code, values in grouped.items()}
    denominator = sum(weights.values())
    log_mass = maximum + math.log(denominator)
    mass = math.exp(log_mass)
    if mass > 1 + 1e-5 or mass < minimum_mass:
        raise NLIScoringError("Class probability mass is invalid or below the configured minimum")
    probabilities = {name: weights[code] / denominator for name, code in CLASS_CODES.items()}
    return NLIScore(**probabilities), {
        "answer": answer, "position": position, "label_mass": mass,
        "label_log_mass": log_mass, "probabilities": probabilities,
        "tokens": [{"token": token, "logprob": value} for token, value in tokens.items()],
    }


def augmented_profile(base: NLIProfile, config: LLMNLIConfig, reference: NLIRuntimeConfig, metadata: dict[str, Any], fields: Sequence[str] = ()) -> NLIProfile:
    """Keep model/scoring provenance and field scope in the row fingerprint."""
    if metadata.get("scoring_config_hash") != config.scoring_hash(reference):
        raise ConfigError("Saved LLM NLI scores do not match the active prompt/scoring configuration")
    selected = tuple(field for field in base.fields if not fields or field.name in fields)
    role = "role" if base.source == "github" else "paper_role"
    if role not in {field.name for field in selected} or set(fields) - {field.name for field in base.fields}:
        raise ConfigError(f"LLM NLI field selection must contain {role} and only known fields")
    raw = {**base.raw, "_llm_nli": metadata, "_field_scope": [field.name for field in selected]}
    return replace(base, fields=selected, raw=raw)


class OllamaNLIBackend:
    """Three-way generative NLI using audited Ollama output-token scores."""

    def __init__(self, model: str, config: LLMNLIConfig, reference: NLIRuntimeConfig, ollama: dict[str, Any], cache: PipelineCache | None, *, session: Any = None, tokenizer: Any = None):
        """Pin the installed model and server before permitting any pair scoring."""
        self.model_id, self.config, self.reference, self.cache = model, config, reference, cache
        self.url = str(ollama.get("url", "http://127.0.0.1:11434")).rstrip("/")
        self.session = session if session is not None else requests.Session()
        if ollama.get("api_key"):
            self.session.headers.update({"Authorization": f"Bearer {ollama['api_key']}"})
        self.keep_alive = str(ollama.get("keep_alive", "30m"))
        version = self._request("GET", "/api/version").get("version")
        tags = self._request("GET", "/api/tags").get("models", [])
        match = next((item for item in tags if model in {item.get("name"), item.get("model"), str(item.get("name", "")).removesuffix(":latest")}), None)
        if not version or not match or not match.get("digest"):
            raise NLIScoringError(f"Cannot pin server version and installed model digest for {model}")
        self.revision = str(match["digest"])
        self._request("POST", "/api/show", {"model": model})
        self.metadata = {"version": VERSION, "model": model, "model_digest": self.revision, "ollama_version": version, "scoring_config_hash": config.scoring_hash(reference), "class_codes": CLASS_CODES, "tokenizer_model": reference.model, "tokenizer_revision": reference.revision}
        self.fingerprint = stable_hash(json.dumps(self.metadata, sort_keys=True))
        self.tokenizer = tokenizer
        self.start_record()

    def _request(self, method: str, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Read native Ollama JSON with bounded retries and redacted transport errors."""
        for attempt in range(self.config.retries + 1):
            try:
                response = self.session.request(method, self.url + route, json=payload, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or data.get("error"):
                    raise NLIScoringError(str(data.get("error", "Malformed Ollama response")) if isinstance(data, dict) else "Malformed Ollama response")
                return data
            except requests.RequestException as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if attempt == self.config.retries or (status is not None and status != 429 and status < 500):
                    raise NLIScoringError(f"Ollama request failed ({type(exc).__name__}, status={status})") from exc
                time.sleep(min(2 ** attempt, 4))
        raise AssertionError("unreachable")

    def chunk_sections(self, sections: Sequence[EvidenceSection], premise_tokens: int, overlap: int) -> list[EvidenceChunk]:
        """Split evidence with the shared reference tokenizer and retain its text audit."""
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.reference.model, revision=self.reference.revision)
        chunks = chunk_nli_sections(self.tokenizer, sections, premise_tokens, overlap)
        self.chunks = [asdict(chunk) for chunk in chunks]
        return chunks

    def start_record(self) -> None:
        """Reset pair and evidence audit state for a new entity."""
        self.audit: list[dict[str, Any]] = []
        self.chunks: list[dict[str, Any]] = []

    def record_audit(self) -> dict[str, str]:
        """Serialize the current record's provenance, probabilities and evidence."""
        return {"llm_nli_metadata_json": json.dumps(self.metadata, sort_keys=True), "llm_nli_pair_scores_json": json.dumps(self.audit, ensure_ascii=False), "nli_evidence_chunks_json": json.dumps(self.chunks, ensure_ascii=False)}

    def score(self, pairs: Sequence[tuple[str, str]], batch_size: int) -> list[NLIScore]:
        """Score pairs sequentially, reusing only provenance-compatible successes."""
        # Ollama has no tensor-batch endpoint. Sequential requests bound GPU load.
        scores = []
        for premise, hypothesis in pairs:
            prompt = Template(self.config.template).substitute(premise=premise, hypothesis=hypothesis, **{name + "_code": code for name, code in CLASS_CODES.items()})
            key = stable_hash(prompt)
            cached = self.cache.get_decision("llm_nli.pairs", key, self.fingerprint, key) if self.cache is not None else None
            cache_hit = cached is not None
            if cached is None:
                started = time.monotonic()
                response = None
                try:
                    response = self._request("POST", "/api/chat", {
                        "model": self.model_id, "messages": [{"role": "user", "content": prompt}],
                        "stream": False, "think": False, "truncate": False, "logprobs": True,
                        "top_logprobs": self.config.top_logprobs, "keep_alive": self.keep_alive,
                        "options": {"temperature": 0, "seed": self.config.seed, "num_predict": 4, "num_ctx": self.config.context_length},
                    })
                    score, audit = parse_class_scores(response, self.config.minimum_label_mass)
                except NLIScoringError as exc:
                    self.audit.append({"premise": premise, "hypothesis": hypothesis, "error": str(exc), "response": response, "inference_seconds": time.monotonic() - started})
                    raise
                cached = {**audit, "inference_seconds": time.monotonic() - started, "prompt_eval_count": response.get("prompt_eval_count"), "eval_count": response.get("eval_count")}
                if self.cache is not None:
                    self.cache.put_decision("llm_nli.pairs", key, self.fingerprint, key, cached, utc_now())
            else:
                score = NLIScore(**cached["probabilities"])
            self.audit.append({"premise_hash": stable_hash(premise), "hypothesis": hypothesis, "cache_hit": cache_hit, **cached})
            scores.append(score)
        return scores

    def preflight(self) -> dict[str, Any]:
        """Check score availability on three synthetic pairs without loading a corpus."""
        self.start_record()
        pairs = [("The item is red.", "The item is red."), ("The item is red.", "The item is round."), ("The item is red and not blue.", "The item is blue.")]
        scores = self.score(pairs, 1)
        return {"metadata": self.metadata, "scores": [asdict(score) for score in scores], "audit": self.audit}

    def runtime_config(self) -> NLIRuntimeConfig:
        """Reuse the reference evidence policy with this model's decision cutoffs."""
        return replace(self.reference, model=self.model_id, revision=self.revision, entailment_threshold=self.config.entailment_threshold, label_margin=self.config.label_margin)
