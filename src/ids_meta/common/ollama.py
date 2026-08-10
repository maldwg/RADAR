from __future__ import annotations

import json
import time
from dataclasses import dataclass
from html import unescape
from typing import Any

import requests

from ids_meta.common.text import normalize_text, stable_hash
from ids_meta.common.logging import logger

LOG = logger("ollama")


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
        if not self.model:
            raise ValueError("OLLAMA_MODEL or ollama.model is required for classification")

    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Request one schema-constrained JSON object with retry handling."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": self.temperature, "seed": self.seed, "num_predict": self.max_output_tokens},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                LOG.debug("requesting structured response from model=%s attempt=%d/%d prompt_chars=%d", self.model, attempt + 1, self.retries + 1, len(prompt))
                response = requests.post(f"{self.url}/api/generate", json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                text = body.get("response") or (body.get("message") or {}).get("content")
                parsed = json.loads(str(text))
                if not isinstance(parsed, dict):
                    raise ValueError("Ollama response must be a JSON object")
                return parsed
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < self.retries:
                    LOG.warning("Ollama request failed (%s); retrying", exc)
                    time.sleep(2**attempt)
        raise RuntimeError(f"Ollama request failed: {error}") from error


def format_evidence(sections: list[EvidenceSection], max_chars: int = 14000) -> str:
    """Format named evidence sections within a shared character budget."""
    blocks: list[str] = []
    used = 0
    for section in sections:
        remaining = max_chars - used
        if remaining <= 0:
            break
        text = section.text.strip()[:remaining]
        if text:
            blocks.append(f"--- {section.source} ---\n{text}")
            used += len(text)
    return "\n\n".join(blocks)


def verify_evidence(value: Any, sections: list[EvidenceSection]) -> tuple[list[dict[str, str]], bool]:
    """Keep quotes found in their claimed section, tolerating harmless source casing and HTML entities."""
    supplied = {
        normalize_text(unescape(section.source)): (section.source, normalize_text(unescape(section.text)))
        for section in sections
    }
    verified: list[dict[str, str]] = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        quote = str(item.get("quote", "")).strip()
        source_key = normalize_text(unescape(source))
        quote_text = normalize_text(unescape(quote))
        if source_key in supplied and quote_text and quote_text in supplied[source_key][1]:
            verified.append({"source": supplied[source_key][0], "quote": quote})
    return verified, bool(verified)


def decision_profile(
    model: str, prompt_version: str, config: dict[str, Any], prompt_config: dict[str, Any] | None = None,
) -> str:
    """Fingerprint model and decision settings that affect cached classifications."""
    relevant = {
        "model": model,
        "prompt_version": prompt_version,
        "temperature": config.get("temperature"),
        "seed": config.get("seed"),
        "require_verified_quote": config.get("require_verified_quote"),
        "prompt_config": prompt_config or {},
    }
    return stable_hash(json.dumps(relevant, sort_keys=True))
