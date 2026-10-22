from __future__ import annotations

import json
import time
from dataclasses import dataclass
from html import unescape
from typing import Any

import requests

from radar.common.evidence_quotes import QUOTE_GROUNDING, SourceQuoteIndex
from radar.common.text import normalize_text, stable_hash
from radar.common.logging import logger

LOG = logger("ollama")

EVIDENCE_MAX_CHARS = 14000


@dataclass(frozen=True)
class EvidenceSection:
    source: str
    text: str


class OllamaClient:
    def __init__(self, config: dict[str, Any]):
        """Initialize a deterministic structured-output Ollama client from configuration."""
        self.url = str(config.get("url", "http://127.0.0.1:11434")).rstrip("/")
        self.model = str(config.get("model", ""))
        self.api_key = str(config.get("api_key", ""))
        self.timeout = float(config.get("timeout_seconds", 300))
        self.retries = int(config.get("retries", 1))
        self.temperature = float(config.get("temperature", 0))
        self.seed = int(config.get("seed", 42))
        self.max_output_tokens = int(config.get("max_output_tokens", 900))
        self.context_length = int(config.get("context_length", 0))
        self.keep_alive = str(config.get("keep_alive", "")).strip()
        self.last_response: dict[str, Any] | None = None
        self.last_raw_response = ""
        self.last_elapsed_seconds = 0.0
        if self.context_length < 0:
            raise ValueError("ollama.context_length must be zero or a positive integer")
        if not self.model:
            raise ValueError("classification requires a selected model from ollama.models")

    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Request one schema-constrained JSON object with retry handling."""
        self.last_response = None
        self.last_raw_response = ""
        self.last_elapsed_seconds = 0.0
        started = time.monotonic()
        options = {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": self.max_output_tokens,
        }
        if self.context_length:
            options["num_ctx"] = self.context_length
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": schema,
            "options": options,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                LOG.debug("requesting structured response from model=%s attempt=%d/%d prompt_chars=%d", self.model, attempt + 1, self.retries + 1, len(prompt))
                response = requests.post(f"{self.url}/api/generate", json=payload, headers=headers, timeout=self.timeout)
                self.last_raw_response = response.text
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("Ollama response envelope must be a JSON object")
                self.last_response = body
                LOG.info(
                    "Ollama timing model=%s load=%.3fs prompt_eval=%.3fs eval=%.3fs "
                    "prompt_tokens=%s eval_tokens=%s done_reason=%s",
                    self.model, float(body.get("load_duration") or 0) / 1e9,
                    float(body.get("prompt_eval_duration") or 0) / 1e9,
                    float(body.get("eval_duration") or 0) / 1e9,
                    body.get("prompt_eval_count", ""), body.get("eval_count", ""),
                    body.get("done_reason", ""),
                )
                text = body.get("response") or (body.get("message") or {}).get("content")
                self.last_raw_response = str(text or "")
                parsed = json.loads(str(text))
                if not isinstance(parsed, dict):
                    raise ValueError("Ollama response must be a JSON object")
                return parsed
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < self.retries:
                    LOG.warning("Ollama request failed (%s); retrying", exc)
                    time.sleep(2**attempt)
            finally:
                self.last_elapsed_seconds = time.monotonic() - started
        raise RuntimeError(f"Ollama request failed: {error}") from error


def _evidence_windows(
    sections: list[EvidenceSection], max_chars: int,
) -> list[tuple[EvidenceSection, int, int]]:
    """Locate the original source windows included in the model's evidence budget."""
    windows = []
    used = 0
    for section in sections:
        remaining = max_chars - used
        if remaining <= 0:
            break
        start = len(section.text) - len(section.text.lstrip())
        end = min(len(section.text.rstrip()), start + remaining)
        if end > start:
            windows.append((section, start, end))
            used += end - start
    return windows


def format_evidence(sections: list[EvidenceSection], max_chars: int = EVIDENCE_MAX_CHARS) -> str:
    """Format named evidence sections within a shared character budget."""
    return "\n\n".join(
        f"--- {section.source} ---\n{section.text[start:end]}"
        for section, start, end in _evidence_windows(sections, max_chars)
    )


def verify_evidence(
    value: Any, sections: list[EvidenceSection], max_chars: int = EVIDENCE_MAX_CHARS,
) -> tuple[list[dict[str, Any]], bool]:
    """Ground catalogue quotes in visible source spans using the shared review checker."""
    supplied = {}
    ambiguous = set()
    for section, start, end in _evidence_windows(sections, max_chars):
        key = normalize_text(unescape(section.source))
        if key in supplied:
            ambiguous.add(key)
        supplied[key] = (section.source, SourceQuoteIndex(section.text[start:end]), start)
    verified: list[dict[str, Any]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("quote"), str):
            continue
        quote = item["quote"]
        source_key = normalize_text(unescape(str(item.get("source", ""))))
        if source_key not in supplied or source_key in ambiguous:
            continue
        source, index, offset = supplied[source_key]
        matches = index.match_fragments(quote)
        if matches:
            verified.append({
                "source": source,
                "quote": " … ".join(match.text for match in matches),
                "model_quote": quote,
                "source_spans": [
                    {"start": offset + match.start, "end": offset + match.end,
                     "quote": match.text, "method": match.method}
                    for match in matches
                ],
                "quote_grounding": QUOTE_GROUNDING,
            })
    return verified, bool(verified)


def decision_profile(
    model: str, prompt_version: str, config: dict[str, Any], prompt_config: dict[str, Any] | None = None,
) -> str:
    """Fingerprint model and decision settings that affect cached classifications."""
    relevant = {
        "model": model,
        "prompt_version": prompt_version,
        "quote_grounding": QUOTE_GROUNDING,
        "temperature": config.get("temperature"),
        "seed": config.get("seed"),
        "require_verified_quote": config.get("require_verified_quote"),
        "prompt_config": prompt_config or {},
    }
    return stable_hash(json.dumps(relevant, sort_keys=True))
