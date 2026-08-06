#!/usr/bin/env python3
"""Classify OpenAlex IDS-dataset findings with an Ollama LLM.

This is intentionally separate from :mod:`openalex_dataset_discovery`.  It
uses the crawler's raw audit files (all discovered entities, their candidate
publications, and mention contexts) as evidence and never changes the
keyword-based crawl or any of its outputs.

The default input is ``ids_datasets_openalex_entity_audit.csv``.  Its sibling
``ids_datasets_openalex_candidates.csv`` and ``ids_datasets_openalex.csv``
are used automatically when present.  The classifier writes:

* ``ids_datasets_openalex.llm_classified.csv`` -- resumable audit decisions;
* ``ids_datasets_openalex.llm_accepted.csv`` -- accepted decision subset;
* ``ids_datasets_openalex.llm.csv`` -- compact accepted catalogue compatible
  with ``figure_creation/ids_dataset_plots_improved.py``.

Example:
    python3 openalex-llm-dataset-classifier.py --model qwen3:latest

Set OLLAMA_URL, OLLAMA_MODEL, and optionally OLLAMA_API_KEY to use a local or
remote Ollama endpoint.  ``--dry-run`` verifies input/output wiring without
calling a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENTITY_AUDIT = SCRIPT_DIR / "ids_datasets_openalex_entity_audit.csv"
DEFAULT_CLASSIFIED_OUT = SCRIPT_DIR / "ids_datasets_openalex.llm_classified.csv"
DEFAULT_ACCEPTED_OUT = SCRIPT_DIR / "ids_datasets_openalex.llm_accepted.csv"
DEFAULT_CATALOGUE_OUT = SCRIPT_DIR / "ids_datasets_openalex.llm.csv"

PIPELINE_VERSION = "1"
SPACE_RE = re.compile(r"\s+")

ARTIFACT_VALUES = ("Dataset", "Generator/framework", "Dataset + generator/framework", "Unknown")
DOMAIN_VALUES = (
    "General Network", "IoT/IIoT", "ICS/SCADA", "Automotive/CAN", "Host",
    "Wireless", "DNS", "Encrypted/VPN/Tor", "Web/App", "Malware/Botnet", "Unknown",
)
DATA_FORM_VALUES = (
    "Packet", "Flow", "Network traffic", "Telemetry", "Host/System-call",
    "Log", "Tabular features", "CAN", "Wireless", "Unknown",
)
TEMPORAL_VALUES = ("Dynamic", "Static", "Mixed", "Unknown")
ENVIRONMENT_VALUES = ("Real-world", "Synthetic/Lab", "Hybrid", "Unknown")
LABELING_VALUES = ("Labeled", "Partially labeled", "Unlabeled", "Unknown")
DECISIONS = ("accept", "exclude", "uncertain", "error", "dry_run")

AUDIT_COLUMNS = [
    "dataset_id", "name", "aliases", "canonical_openalex_id", "canonical_title",
    "candidate_titles", "mention_contexts", "survey_count", "trusted_survey_count",
    "candidate_count", "keyword_status", "keyword_confidence", "keyword_artifact_type",
    "keyword_domain", "keyword_data_form", "keyword_temporal_type", "keyword_environment",
    "keyword_labeling", "pipeline_version", "run_profile", "llm_model", "decision",
    "is_ids_dataset", "dataset_confidence", "artifact_type", "domain", "data_form",
    "temporal_type", "environment", "labeling", "primary_reason", "evidence_json",
    "evidence_quote_verified", "input_chars", "response_seconds", "error",
]

# This is the existing keyword-crawler output schema plus harmless LLM audit
# fields.  The old plotting script accesses fields by name and ignores extras.
CATALOGUE_COLUMNS = [
    "dataset_id", "name", "aliases", "artifact_type", "status", "canonical_openalex_id",
    "canonical_title", "description", "release_year", "canonical_publication_year", "year",
    "release_year_source", "doi", "url", "citations", "survey_count",
    "trusted_survey_count", "discovery_method", "discovery_methods", "survey_sources",
    "mention_sources", "domain", "data_form", "temporal_type", "environment", "labeling",
    "confidence", "canonical_score", "evidence", "type_evidence", "mention_contexts",
    "candidate_count", "llm_decision", "llm_model", "llm_dataset_confidence",
    "llm_primary_reason", "llm_evidence_quote_verified",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_ids_dataset": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "dataset_confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "artifact_type": {"type": "string", "enum": list(ARTIFACT_VALUES)},
        "domain": {"type": "string", "enum": list(DOMAIN_VALUES)},
        "data_form": {"type": "string", "enum": list(DATA_FORM_VALUES)},
        "temporal_type": {"type": "string", "enum": list(TEMPORAL_VALUES)},
        "environment": {"type": "string", "enum": list(ENVIRONMENT_VALUES)},
        "labeling": {"type": "string", "enum": list(LABELING_VALUES)},
        "primary_reason": {"type": "string"},
        "evidence": {
            "type": "array", "maxItems": 3,
            "items": {"type": "object", "properties": {
                "source": {"type": "string"}, "quote": {"type": "string"},
            }, "required": ["source", "quote"]},
        },
    },
    "required": [
        "is_ids_dataset", "dataset_confidence", "artifact_type", "domain", "data_form",
        "temporal_type", "environment", "labeling", "primary_reason", "evidence",
    ],
}


@dataclass(frozen=True)
class Config:
    input_path: Path
    candidates_path: Path | None
    keyword_catalogue_path: Path | None
    out_path: Path
    accepted_out_path: Path
    catalogue_out_path: Path
    model: str
    ollama_url: str
    api_key: str | None
    timeout: float
    request_retries: int
    retry_backoff_seconds: float
    keep_alive: str
    max_output_tokens: int
    temperature: float
    seed: int | None
    top_k: int
    top_p: float
    max_input_chars: int
    accept_confidence: float
    exclude_confidence: float
    require_evidence_quote: bool
    pause_seconds: float
    checkpoint_every: int
    resume: bool
    limit: int | None
    no_json_schema: bool
    think: bool
    debug_response: bool
    dry_run: bool
    export_catalogue_from: Path | None


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def clean(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def field(row: dict[str, str], name: str, default: str = "") -> str:
    return str(row.get(name) or default)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, str]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sibling_path(path: Path, suffix: str) -> Path:
    name = path.name.replace("_entity_audit", suffix)
    return path.with_name(name)


def candidate_groups(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if not path or not path.is_file():
        return {}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in load_csv(path):
        if field(row, "dataset_id"):
            groups[field(row, "dataset_id")].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (
            0 if field(row, "status") == "canonical_resolved" else 1,
            -as_float(row.get("score")), -as_float(row.get("candidate_citations")),
            field(row, "candidate_title").casefold(),
        ))
    return groups


def build_source_rows(config: Config) -> list[dict[str, str]]:
    entities = load_csv(config.input_path)
    candidates = candidate_groups(config.candidates_path)
    keyword: dict[str, dict[str, str]] = {}
    if config.keyword_catalogue_path and config.keyword_catalogue_path.is_file():
        keyword = {field(row, "dataset_id"): row for row in load_csv(config.keyword_catalogue_path) if field(row, "dataset_id")}
    source: list[dict[str, str]] = []
    for entity in entities:
        dataset_id = field(entity, "dataset_id")
        if not dataset_id:
            continue
        baseline = keyword.get(dataset_id, {})
        candidate_titles = []
        for candidate in candidates.get(dataset_id, [])[:8]:
            title = clean(candidate.get("candidate_title"))
            if title:
                candidate_titles.append(
                    f"[{field(candidate, 'status', 'unknown')}; score={field(candidate, 'score')}; "
                    f"year={field(candidate, 'candidate_year')}] {title}"
                )
        source.append({
            "dataset_id": dataset_id, "name": field(entity, "name") or field(baseline, "name"),
            "aliases": field(entity, "aliases") or field(baseline, "aliases"),
            "canonical_openalex_id": field(entity, "canonical_openalex_id") or field(baseline, "canonical_openalex_id"),
            "canonical_title": field(entity, "canonical_title") or field(baseline, "canonical_title"),
            "candidate_titles": "\n".join(candidate_titles),
            "mention_contexts": field(entity, "mention_contexts") or field(baseline, "mention_contexts"),
            "survey_count": field(entity, "survey_count") or field(baseline, "survey_count"),
            "trusted_survey_count": field(entity, "trusted_survey_count") or field(baseline, "trusted_survey_count"),
            "candidate_count": field(entity, "candidate_count") or field(baseline, "candidate_count"),
            "keyword_status": field(baseline, "status") or field(entity, "status"),
            "keyword_confidence": field(baseline, "confidence") or field(entity, "confidence"),
            "keyword_artifact_type": field(baseline, "artifact_type") or field(entity, "artifact_type"),
            "keyword_domain": field(baseline, "domain"), "keyword_data_form": field(baseline, "data_form"),
            "keyword_temporal_type": field(baseline, "temporal_type"),
            "keyword_environment": field(baseline, "environment"), "keyword_labeling": field(baseline, "labeling"),
            # Raw crawler metadata used for a plotting-compatible export; not
            # included in the LLM prompt as a classification signal.
            **{column: field(baseline, column) for column in CATALOGUE_COLUMNS if column not in {"dataset_id", "name", "aliases"}},
        })
    return source


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:max(1, limit - 1)] + "…"


def format_evidence(row: dict[str, str], max_chars: int) -> str:
    blocks = [
        f"Entity name: {field(row, 'name')}",
        f"Aliases: {field(row, 'aliases')}",
        f"Canonical candidate title: {field(row, 'canonical_title') or '(none resolved)'}",
        f"Candidate publication titles:\n{field(row, 'candidate_titles') or '(none)'}",
        f"Crawler mention contexts:\n{field(row, 'mention_contexts') or '(none)'}",
        f"Counts: survey mentions={field(row, 'survey_count') or '0'}; trusted-survey mentions={field(row, 'trusted_survey_count') or '0'}; candidate works={field(row, 'candidate_count') or '0'}.",
    ]
    return truncate("\n\n".join(blocks), max_chars)


def prompt_for(row: dict[str, str], max_chars: int) -> str:
    evidence = format_evidence(row, max_chars)
    return f"""You are auditing raw literature-crawler findings for a study of intrusion-detection-system (IDS) datasets.

Decide from the supplied evidence only whether this entity identifies a dataset, corpus, trace collection, benchmark, or dataset-generation framework that is intended for evaluating/training intrusion detection or attack/anomaly detection. A paper that merely *uses* a dataset, proposes a detector, or is a generic cybersecurity dataset is not enough. A generator/framework may be yes only when it creates IDS-relevant data.

Use `uncertain` when the evidence does not establish either outcome. Do not infer missing details. For yes, classify the best-supported subtype fields. Use `Unknown` for a subtype not supported by the evidence. `Dynamic` means sequential/raw traces, packets, flows, logs, telemetry, etc.; `Static` means fixed feature/tabular records; `Mixed` only when both are explicitly supported. Quote short exact fragments from EVIDENCE; do not quote the instructions.

Allowed values:
- artifact_type: {', '.join(ARTIFACT_VALUES)}
- domain: {', '.join(DOMAIN_VALUES)}
- data_form: {', '.join(DATA_FORM_VALUES)}
- temporal_type: {', '.join(TEMPORAL_VALUES)}
- environment: {', '.join(ENVIRONMENT_VALUES)}
- labeling: {', '.join(LABELING_VALUES)}

EVIDENCE
{evidence}
"""


def model_text(payload: dict[str, Any]) -> str:
    text = payload.get("response")
    if isinstance(text, str) and text.strip():
        return text
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    raise ValueError("Ollama response had no response text")


def parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model response must be a JSON object")
    return parsed


def ollama_generate(config: Config, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model, "prompt": prompt, "stream": False,
        "keep_alive": config.keep_alive, "think": config.think,
        "options": {"temperature": config.temperature, "top_k": config.top_k,
                    "top_p": config.top_p, "num_predict": config.max_output_tokens,
                    **({"seed": config.seed} if config.seed is not None else {})},
    }
    if not config.no_json_schema:
        payload["format"] = RESPONSE_SCHEMA
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(config.ollama_url.rstrip("/") + "/api/generate", data=json.dumps(payload).encode(), headers=headers, method="POST")
    last_error: Exception | None = None
    for attempt in range(config.request_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            if config.debug_response:
                print(json.dumps(response_payload, ensure_ascii=False)[:4000], file=sys.stderr)
            return parse_json(model_text(response_payload))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            if error.code < 500 and error.code != 429:
                raise RuntimeError(f"Ollama HTTP {error.code}: {detail}") from error
            last_error = RuntimeError(f"Ollama HTTP {error.code}: {detail}")
        except (TimeoutError, socket.timeout, urllib.error.URLError, http.client.HTTPException, OSError, ValueError, json.JSONDecodeError) as error:
            last_error = error
        if attempt < config.request_retries:
            delay = config.retry_backoff_seconds * (2 ** attempt)
            print(f"LLM request failed ({last_error}); retrying in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"Ollama request failed after {config.request_retries + 1} attempt(s): {last_error}") from last_error


def enum_value(value: Any, allowed: tuple[str, ...], default: str) -> str:
    text = clean(value)
    lookup = {item.casefold(): item for item in allowed}
    return lookup.get(text.casefold(), default)


def confidence(value: Any) -> float:
    return max(0.0, min(1.0, as_float(value)))


def quotes_verified(evidence: Any, prompt: str) -> tuple[list[dict[str, str]], bool]:
    if not isinstance(evidence, list):
        return [], False
    normalized_prompt = clean(prompt).casefold()
    items: list[dict[str, str]] = []
    for item in evidence[:3]:
        if not isinstance(item, dict):
            continue
        quote = clean(item.get("quote"))
        source = clean(item.get("source"))
        if quote:
            items.append({"source": source, "quote": quote})
    return items, bool(items) and all(clean(item["quote"]).casefold() in normalized_prompt for item in items)


def normalize_response(raw: dict[str, Any], prompt: str, config: Config) -> dict[str, str]:
    scope = enum_value(raw.get("is_ids_dataset"), ("yes", "no", "uncertain"), "uncertain")
    score = confidence(raw.get("dataset_confidence"))
    evidence, verified = quotes_verified(raw.get("evidence"), prompt)
    if scope == "yes" and score >= config.accept_confidence and (verified or not config.require_evidence_quote):
        decision = "accept"
    elif scope == "no" and score >= config.exclude_confidence:
        decision = "exclude"
    else:
        decision = "uncertain"
    return {
        "decision": decision, "is_ids_dataset": scope, "dataset_confidence": f"{score:.3f}",
        "artifact_type": enum_value(raw.get("artifact_type"), ARTIFACT_VALUES, "Unknown"),
        "domain": enum_value(raw.get("domain"), DOMAIN_VALUES, "Unknown"),
        "data_form": enum_value(raw.get("data_form"), DATA_FORM_VALUES, "Unknown"),
        "temporal_type": enum_value(raw.get("temporal_type"), TEMPORAL_VALUES, "Unknown"),
        "environment": enum_value(raw.get("environment"), ENVIRONMENT_VALUES, "Unknown"),
        "labeling": enum_value(raw.get("labeling"), LABELING_VALUES, "Unknown"),
        "primary_reason": clean(raw.get("primary_reason"))[:1200],
        "evidence_json": json.dumps(evidence, ensure_ascii=False),
        "evidence_quote_verified": "yes" if verified else "no",
    }


def run_profile(config: Config) -> str:
    value = {"version": PIPELINE_VERSION, "model": config.model, "max_input_chars": config.max_input_chars,
             "temperature": config.temperature, "seed": config.seed, "top_k": config.top_k, "top_p": config.top_p,
             "accept_confidence": config.accept_confidence, "exclude_confidence": config.exclude_confidence,
             "require_evidence_quote": config.require_evidence_quote, "think": config.think}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def completed_rows(path: Path, config: Config) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    profile = run_profile(config)
    return {field(row, "dataset_id"): row for row in load_csv(path) if field(row, "dataset_id") and field(row, "pipeline_version") == PIPELINE_VERSION and field(row, "run_profile") == profile and field(row, "llm_model") == config.model and not field(row, "error")}


def classify_row(row: dict[str, str], config: Config) -> dict[str, str]:
    prompt = prompt_for(row, config.max_input_chars)
    output = {column: field(row, column) for column in AUDIT_COLUMNS}
    output.update({"pipeline_version": PIPELINE_VERSION, "run_profile": run_profile(config), "llm_model": config.model,
                   "input_chars": str(len(format_evidence(row, config.max_input_chars))), "response_seconds": "", "error": ""})
    if config.dry_run:
        output.update({"decision": "dry_run", "is_ids_dataset": "", "dataset_confidence": "", "artifact_type": "", "domain": "", "data_form": "", "temporal_type": "", "environment": "", "labeling": "", "primary_reason": "", "evidence_json": "", "evidence_quote_verified": ""})
        return output
    started = time.monotonic()
    try:
        output.update(normalize_response(ollama_generate(config, prompt), prompt, config))
    except (RuntimeError, ValueError, json.JSONDecodeError, TimeoutError, socket.timeout, urllib.error.URLError, http.client.HTTPException, OSError) as error:
        output.update({"decision": "error", "is_ids_dataset": "", "dataset_confidence": "", "artifact_type": "", "domain": "", "data_form": "", "temporal_type": "", "environment": "", "labeling": "", "primary_reason": "", "evidence_json": "", "evidence_quote_verified": "", "error": str(error)[:1000]})
    output["response_seconds"] = f"{time.monotonic() - started:.2f}"
    return output


def confidence_label(score: str) -> str:
    value = as_float(score)
    return "High" if value >= .85 else "Medium" if value >= .65 else "Low"


def catalogue_row(audit: dict[str, str], source: dict[str, str]) -> dict[str, str]:
    row = {column: field(source, column) for column in CATALOGUE_COLUMNS}
    # Preserve crawler bibliographic data where available, but every taxonomy
    # field and the inclusion decision below comes from the LLM decision.
    for name in ("dataset_id", "name", "aliases", "canonical_openalex_id", "canonical_title", "survey_count", "trusted_survey_count", "candidate_count"):
        row[name] = field(audit, name) or row.get(name, "")
    row.update({
        "artifact_type": field(audit, "artifact_type"), "status": "llm_accepted",
        "description": field(source, "description") or field(audit, "canonical_title") or field(audit, "mention_contexts")[:420],
        "domain": field(audit, "domain"), "data_form": field(audit, "data_form"),
        "temporal_type": field(audit, "temporal_type"), "environment": field(audit, "environment"),
        "labeling": field(audit, "labeling"), "confidence": confidence_label(field(audit, "dataset_confidence")),
        "evidence": field(audit, "primary_reason"),
        "type_evidence": f"LLM classification; quote verified={field(audit, 'evidence_quote_verified')}; {field(audit, 'evidence_json')}",
        "llm_decision": field(audit, "decision"), "llm_model": field(audit, "llm_model"),
        "llm_dataset_confidence": field(audit, "dataset_confidence"),
        "llm_primary_reason": field(audit, "primary_reason"),
        "llm_evidence_quote_verified": field(audit, "evidence_quote_verified"),
    })
    return row


def write_all(rows: list[dict[str, str]], sources: dict[str, dict[str, str]], config: Config) -> None:
    write_csv(rows, config.out_path, AUDIT_COLUMNS)
    accepted = [row for row in rows if field(row, "decision") == "accept"]
    write_csv(accepted, config.accepted_out_path, AUDIT_COLUMNS)
    catalogue = [catalogue_row(row, sources.get(field(row, "dataset_id"), {})) for row in accepted]
    catalogue.sort(key=lambda row: (-as_float(row.get("llm_dataset_confidence")), field(row, "name").casefold()))
    write_csv(catalogue, config.catalogue_out_path, CATALOGUE_COLUMNS)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Zero-shot IDS dataset classifier for OpenAlex crawler audit files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_ENTITY_AUDIT, help="Raw entity-audit CSV from openalex_dataset_discovery.py.")
    parser.add_argument("--candidates", type=Path, default=None, help="Raw candidate audit CSV. Defaults to the sibling *_candidates.csv.")
    parser.add_argument("--keyword-catalogue", type=Path, default=None, help="Existing keyword accepted CSV for bibliographic metadata only; defaults to sibling ids_datasets_openalex.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CLASSIFIED_OUT, help="Wide resumable LLM audit CSV.")
    parser.add_argument("--accepted-out", type=Path, default=DEFAULT_ACCEPTED_OUT, help="Accepted LLM audit subset. Defaults beside --out.")
    parser.add_argument("--catalogue-out", type=Path, default=DEFAULT_CATALOGUE_OUT, help="Plot-compatible accepted LLM catalogue. Defaults beside --out.")
    parser.add_argument("--export-catalogue-from", type=Path, default=None, help="Regenerate accepted outputs from a prior audit CSV without model calls.")
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL"), help="Ollama model name (or OLLAMA_MODEL).")
    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"), help="Ollama base URL.")
    parser.add_argument("--api-key", default=os.getenv("OLLAMA_API_KEY"), help="Optional Bearer token for a remote Ollama proxy.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-entity request timeout in seconds.")
    parser.add_argument("--request-retries", type=int, default=1, help="Retries after transient request failures.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0, help="Initial retry delay; doubles per retry.")
    parser.add_argument("--keep-alive", default="600m", help="Ollama model keep-alive value.")
    parser.add_argument("--max-output-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42, help="Use -1 to omit a fixed seed.")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-input-chars", type=int, default=14000)
    parser.add_argument("--accept-confidence", type=float, default=.80)
    parser.add_argument("--exclude-confidence", type=float, default=.75)
    parser.add_argument("--require-evidence-quote", action=argparse.BooleanOptionalAction, default=True, help="Require every LLM evidence quote to occur in supplied evidence before acceptance.")
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-json-schema", action="store_true")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--debug-response", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.export_catalogue_from and not args.input.is_file():
        parser.error(f"--input does not exist: {args.input}")
    if not args.export_catalogue_from and not args.model and not args.dry_run:
        parser.error("--model or OLLAMA_MODEL is required unless --dry-run is used.")
    if args.timeout <= 0 or args.request_retries < 0 or args.retry_backoff_seconds < 0 or args.max_input_chars < 500 or args.max_output_tokens < 64 or args.checkpoint_every < 0 or args.pause_seconds < 0:
        parser.error("timeout must be positive; input/output/checkpoint/retry values must be valid non-negative values.")
    if args.temperature < 0 or args.top_k < 1 or not 0 < args.top_p <= 1 or not 0 <= args.accept_confidence <= 1 or not 0 <= args.exclude_confidence <= 1:
        parser.error("invalid sampling or confidence setting")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    out = args.out
    source_for_defaults = args.export_catalogue_from or out
    accepted_default = source_for_defaults.with_name(source_for_defaults.name.replace("_classified", "_accepted"))
    catalogue_default = source_for_defaults.with_name(source_for_defaults.name.replace("_classified", ""))
    candidates = args.candidates or sibling_path(args.input, "_candidates")
    keyword = args.keyword_catalogue or sibling_path(args.input, "")
    return Config(args.input, candidates if candidates.is_file() else None, keyword if keyword.is_file() else None,
                  out, args.accepted_out or accepted_default, args.catalogue_out or catalogue_default,
                  args.model or "dry-run", args.ollama_url, args.api_key, args.timeout, args.request_retries,
                  args.retry_backoff_seconds, args.keep_alive, args.max_output_tokens, args.temperature,
                  None if args.seed == -1 else args.seed, args.top_k, args.top_p, args.max_input_chars,
                  args.accept_confidence, args.exclude_confidence, args.require_evidence_quote, args.pause_seconds,
                  args.checkpoint_every, args.resume, args.limit, args.no_json_schema, args.think,
                  args.debug_response, args.dry_run, args.export_catalogue_from)


def main() -> int:
    configure_csv_limit()
    config = parse_args()
    if config.export_catalogue_from:
        if not config.export_catalogue_from.is_file():
            print(f"LLM audit CSV does not exist: {config.export_catalogue_from}", file=sys.stderr)
            return 2
        rows = load_csv(config.export_catalogue_from)
        # Existing audit files carry enough fields for a compatible catalogue.
        sources = {field(row, "dataset_id"): row for row in rows}
        write_all(rows, sources, config)
        print(f"Saved accepted audit: {config.accepted_out_path}")
        print(f"Saved plotting-compatible catalogue: {config.catalogue_out_path}")
        return 0
    sources_list = build_source_rows(config)
    sources = {field(row, "dataset_id"): row for row in sources_list}
    existing = completed_rows(config.out_path, config) if config.resume else {}
    results = list(existing.values())
    write_all(results, sources, config)
    processed = 0
    for source in sources_list:
        identifier = field(source, "dataset_id")
        if identifier in existing:
            continue
        if config.limit is not None and processed >= config.limit:
            break
        result = classify_row(source, config)
        results.append(result)
        processed += 1
        print(f"[{processed}] {identifier} ({field(result, 'name')}): {field(result, 'decision')}")
        if field(result, "decision") == "error" or (config.checkpoint_every and processed % config.checkpoint_every == 0):
            write_all(results, sources, config)
        if config.pause_seconds:
            time.sleep(config.pause_seconds)
    results.sort(key=lambda row: field(row, "dataset_id").casefold())
    write_all(results, sources, config)
    counts: dict[str, int] = defaultdict(int)
    for row in results:
        counts[field(row, "decision", "missing")] += 1
    print(f"Saved {len(results)} classified entities to {config.out_path}; decisions: {dict(counts)}")
    print(f"Saved accepted audit to {config.accepted_out_path}")
    print(f"Saved plotting-compatible catalogue to {config.catalogue_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
