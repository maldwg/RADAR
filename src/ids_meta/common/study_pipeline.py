"""Independent discovery and structured assessment for external-review replication."""
from __future__ import annotations

import json
import os
import tempfile
import tomllib
from datetime import date
from io import BytesIO
from pathlib import Path
from string import Template
from typing import Any

import requests

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.common.logging import logger
from ids_meta.common.limits import optional_limit, query_result_limit, limit_label
from ids_meta.common.ollama import OllamaClient
from ids_meta.common.provenance import file_hash
from ids_meta.common.run import utc_now
from ids_meta.common.sources import LocalSourceAdapter, content_hash
from ids_meta.common.screening import load_screening_profile
from ids_meta.common.study_screening import (
    EVIDENCE_NORMALIZATION, SCREENING_FIELDS,
    load_study_protocol, normalize_evidence_text,
)
from ids_meta.openalex.collect import OpenAlexClient, normalize_work

LOG = logger("study")
PROMPT_VERSION = "external-review-assessment-v4"


def _write_json(path: Path, value: Any) -> None:
    """Replace a JSON checkpoint without exposing partial file contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def collect_study(profile_path: Path, output: Path, *, limit: int | str | None = 100,
                  client: OpenAlexClient | None = None,
                  source_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Search independently of ratings, retain capped query audits and export a cohort."""
    if output.suffix.casefold() != ".json" or output.resolve() == profile_path.resolve():
        raise ValueError("study collection requires a separate JSON output")
    digest = file_hash(profile_path)
    with profile_path.open("rb") as handle:
        profile = tomllib.load(handle)
    if profile.get("schema_version") != "1.0" or profile.get("provider") != "openalex":
        raise ValueError("study search requires schema_version 1.0 and provider openalex")
    if profile.get("named_entity_seeding") is not False:
        raise ValueError("study search must explicitly disable named_entity_seeding")
    parameter = profile.get("search_parameter", "search")
    if parameter not in {"search", "search.exact", "search.title_and_abstract", "search.title_and_abstract.exact"}:
        raise ValueError("unsupported study search_parameter")
    queries = profile.get("queries")
    if not isinstance(queries, list) or not queries or any(
        not isinstance(q, dict) or not isinstance(q.get("name"), str) or not q["name"].strip()
        or not isinstance(q.get("search"), str) or not q["search"].strip() for q in queries
    ) or len({q["name"] for q in queries}) != len(queries):
        raise ValueError("study search requires distinct named queries")
    start, stop = profile.get("from_publication_date"), profile.get("to_publication_date")
    if not isinstance(start, str) or not isinstance(stop, str) or date.fromisoformat(start) > date.fromisoformat(stop):
        raise ValueError("study search requires an ordered ISO publication-date window")
    source_config = source_config or {}
    limit = optional_limit(limit, "collection_limit")
    cap = query_result_limit(source_config, limit)
    page_size = source_config.get("per_page", 200)
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 200:
        raise ValueError("OpenAlex per_page must be an integer between 1 and 200")
    client = client or OpenAlexClient(os.getenv("OPENALEX_API_KEY"), os.getenv("OPENALEX_MAILTO"),
        request_pause_seconds=float(source_config.get("request_pause_seconds", 1)))
    pool: dict[str, dict[str, Any]] = {}
    audits = []
    for query_index, query in enumerate(queries):
        cursor, scanned, total = "*", 0, None
        seen = set()
        pages = []
        exhausted = False
        LOG.info("searching study query=%s cap=%s", query["name"], limit_label(cap))
        while cap is None or scanned < cap:
            if cursor in seen:
                raise RuntimeError("OpenAlex repeated study-search cursor")
            seen.add(cursor)
            payload = client.get("/works", {parameter: query["search"], "cursor": cursor,
                "per-page": page_size if cap is None else min(page_size, cap - scanned), "select": client.select,
                "filter": f"from_publication_date:{start},to_publication_date:{stop}"})
            items, meta = payload.get("results"), payload.get("meta", {})
            if not isinstance(items, list) or not isinstance(meta, dict):
                raise ValueError("OpenAlex returned an invalid study-search page")
            total = meta.get("count", total)
            pages.append({"cursor": cursor, "response": payload})
            for item in items if cap is None else items[:cap - scanned]:
                work = normalize_work(item)
                key = work["openalex_id"]
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("OpenAlex study-search result lacks an identity")
                if key not in pool:
                    pool[key] = {**work, "paper_id": key, "discovery_queries": [], "query_ranks": {},
                        "selection_rank": [scanned + 1, query_index, key]}
                if query["name"] not in pool[key]["discovery_queries"]:
                    pool[key]["discovery_queries"].append(query["name"])
                pool[key]["query_ranks"].setdefault(query["name"], scanned + 1)
                pool[key]["selection_rank"] = min(pool[key]["selection_rank"], [scanned + 1, query_index, key])
                scanned += 1
            next_cursor = str(meta.get("next_cursor") or "")
            exhausted = not items or not next_cursor or (isinstance(total, int) and scanned >= total)
            if exhausted:
                break
            cursor = next_cursor
        audits.append({**query, "scanned": scanned, "reported_hits": total,
            "coverage": "complete" if exhausted else "bounded", "pages": pages})
    if file_hash(profile_path) != digest:
        raise ValueError("study search profile changed during collection")
    rows = sorted(pool.values(), key=lambda row: row["selection_rank"])[:limit]
    _write_json(output, rows)
    _write_json(output.with_suffix(".collection.json"), {"schema_version": "1.0", "provider": "openalex",
        "collected_at": utc_now(), "profile_sha256": digest, "profile": profile,
        "candidate_sha256": file_hash(output), "retained": len(rows), "pool_size": len(pool),
        "selection": "best per-query result rank, query declaration order, OpenAlex ID", "collection_limit": limit,
        "classification_limit": optional_limit(source_config.get("classification_limit"), "classification_limit"),
        "query_limit_factor": source_config.get("query_limit_factor", 3.0), "per_query_limit": cap,
        "selected_ids": [r["paper_id"] for r in rows], "queries": audits})
    LOG.info("study collection retained=%d pool=%d", len(rows), len(pool))
    return rows


def download_study_text(url: str, max_chars: int, timeout: float = 30) -> dict[str, Any]:
    """Extract an openly supplied PDF with bounded download and text size."""
    if not url.startswith(("https://", "http://")):
        return {"full_text_status": "no_pdf_url"}
    try:
        with requests.get(url, timeout=timeout, stream=True,
                          headers={"User-Agent": "ids-meta/0.1 review replication"}) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > 30 * 1024 * 1024:
                    raise ValueError("PDF exceeds download limit")
                chunks.append(chunk)
        data = b"".join(chunks)
        if not data.startswith(b"%PDF"):
            raise ValueError("PDF URL returned non-PDF content")
        from pypdf import PdfReader
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(data)).pages)
        return {"full_text": text[:max_chars], "full_text_url": url,
            "full_text_complete": len(text) <= max_chars, "full_text_status": "downloaded" if text.strip() else "empty_pdf",
            "full_text_original_sha256": content_hash(text)}
    except Exception as exc:
        LOG.info("study PDF unavailable (%s)", type(exc).__name__)
        return {"full_text_status": "download_failed", "full_text_error": type(exc).__name__}


def load_assessment_fields(protocol: Any) -> dict[str, Any]:
    """Load scientific extraction definitions from the study taxonomy."""
    profile = load_screening_profile(protocol.path)
    if file_hash(protocol.path) != protocol.sha256:
        raise ValueError("study protocol changed before assessment")
    profile_path = profile.taxonomy_path
    with profile_path.open("rb") as handle:
        taxonomy = tomllib.load(handle)
    if file_hash(profile_path) != protocol.source_profile_sha256:
        raise ValueError("study taxonomy changed before assessment")
    fields = taxonomy.get("assessment_fields", {})
    if not isinstance(fields, dict) or set(fields) != set(protocol.contract.fields):
        raise ValueError("study taxonomy must define each assessment field")
    for spec in fields.values():
        if not isinstance(spec, dict) or not isinstance(spec.get("description"), str) or not spec["description"].strip():
            raise ValueError("assessment fields require descriptions")
        values = spec.get("values")
        if not isinstance(values, list) or "unknown" not in values or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("assessment fields require labels including unknown")
    return fields


def _judgment_schema(values: list[str]) -> dict[str, Any]:
    """Build the structured schema for one evidence judgment."""
    return {"type": "object", "additionalProperties": False, "required": ["value", "quote", "source"],
        "properties": {"value": {"type": "string", "enum": values}, "quote": {"type": "string"},
            "source": {"type": "string", "enum": ["full_text", ""]}}}


def load_assessment_template(protocol: Any) -> tuple[str, Path, str]:
    """Load a study-owned prompt and require only blind assessment inputs."""
    path = load_screening_profile(protocol.path).prompt_path
    digest = file_hash(path)
    with path.open("rb") as handle:
        prompt = tomllib.load(handle)
    template = prompt.get("template")
    if prompt.get("schema_version") != "1.0" or not isinstance(template, str) or not Template(template).is_valid():
        raise ValueError("study assessment prompt requires schema_version 1.0 and a valid template")
    if set(Template(template).get_identifiers()) != {"eligibility", "quality", "fields", "evidence"}:
        raise ValueError("study assessment prompt requires exactly eligibility, quality, fields and evidence placeholders")
    return template, path, digest


def _assessment_prompt(protocol: Any, fields: dict[str, Any], row: dict[str, Any], template: str) -> tuple[str, dict[str, Any]]:
    """Build a blind assessment prompt and its response schema."""
    criteria = {key: _judgment_schema(["yes", "no", "unknown"]) for key in [*protocol.filters, *protocol.quality]}
    taxonomy = {key: _judgment_schema(spec["values"]) for key, spec in fields.items()}
    schema = {"type": "object", "additionalProperties": False, "required": ["criteria", "fields"], "properties": {
        "criteria": {"type": "object", "additionalProperties": False, "required": list(criteria), "properties": criteria},
        "fields": {"type": "object", "additionalProperties": False, "required": list(taxonomy), "properties": taxonomy}}}
    prompt = Template(template).substitute(eligibility=json.dumps(protocol.filters), quality=json.dumps(protocol.quality),
        fields=json.dumps(fields), evidence=json.dumps({k: row.get(k, '') for k in
            ('title', 'abstract', 'full_text', 'full_text_complete')}, ensure_ascii=False))
    return prompt, schema


def _validate_assessment(answer: Any, protocol: Any, fields: dict[str, Any], text: str) -> None:
    """Report every invalid judgment so one correction can address all errors."""
    if not isinstance(answer, dict) or set(answer) != {"criteria", "fields"}:
        raise ValueError("model output must contain criteria and fields")
    specifications = {
        "criteria": {key: ["yes", "no", "unknown"] for key in [*protocol.filters, *protocol.quality]},
        "fields": {key: spec["values"] for key, spec in fields.items()},
    }
    text = normalize_evidence_text(text)
    errors = []
    for section, specifications_for_section in specifications.items():
        judgments = answer[section]
        if not isinstance(judgments, dict) or set(judgments) != set(specifications_for_section):
            errors.append(f"{section}: assess exactly the required keys")
            continue
        for key, values in specifications_for_section.items():
            judgment = judgments[key]
            if (not isinstance(judgment, dict) or set(judgment) != {"value", "quote", "source"}
                or judgment.get("value") not in values
                or not isinstance(judgment.get("quote"), str)
                or not isinstance(judgment.get("source"), str)):
                errors.append(f"{section}.{key}: invalid judgment schema")
                continue
            quote, source = judgment["quote"], judgment["source"]
            if judgment["value"] == "unknown":
                if quote or source:
                    errors.append(f"{section}.{key}: unknown requires empty quote and source")
            elif source != "full_text" or not quote.strip() or normalize_evidence_text(quote) not in text:
                errors.append(f"{section}.{key}: quotation is absent from supplied full_text")
    if errors:
        raise ValueError("; ".join(errors))


def _assess_once(record: dict[str, Any], protocol: Any, fields: dict[str, Any], template: str,
                 client: Any, label: str) -> dict[str, Any]:
    """Obtain one validated assessment, allowing one literal-quote correction."""
    prompt, schema = _assessment_prompt(protocol, fields, record, template)
    vote: dict[str, Any] = {"label": label, "status": "error", "attempts": []}
    try:
        for attempt in range(2):
            answer = client.generate(prompt, schema)
            vote["model_response_json"] = json.dumps(answer, sort_keys=True)
            entry = {"response": answer, "validation_error": ""}
            vote["attempts"].append(entry)
            try:
                _validate_assessment(answer, protocol, fields, record["full_text"])
            except ValueError as exc:
                entry["validation_error"] = str(exc)
                if attempt:
                    raise
                LOG.info("correcting invalid study evidence paper_id=%s run=%s", record[protocol.contract.identity], label)
                prompt += ("\nCORRECTION REQUEST\nYour previous JSON failed validation. "
                    "Return a complete corrected JSON object. Copy the shortest sufficient literal "
                    "passage from full_text, preserving its characters; never paraphrase a quotation. "
                    "If evidence cannot be supported, use unknown with empty quote/source. "
                    "A passage's presence does not establish its relevance: recheck its meaning.\n"
                    + json.dumps(entry, ensure_ascii=False))
            else:
                vote.update(status="success", answer=answer)
                break
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        vote.update(error_type=type(exc).__name__, error_detail=str(exc) if isinstance(exc, ValueError) else "")
        LOG.warning("study model assessment failed paper_id=%s run=%s (%s)", record[protocol.contract.identity], label, type(exc).__name__)
    return vote


def classify_study(protocol_path: Path, input_path: Path, output: Path, ollama_config: dict[str, Any], *,
                   limit: int | None = None, resume: bool = False, dry_run: bool = False,
                   fetch_full_text: bool = False, max_chars: int = 120000, client: Any = None,
                   consecutive_runs: int = 1, consensus_threshold: float = 0.8,
                   evidence_lookup_timeout_seconds: float = 30) -> list[dict[str, Any]]:
    """Repeat validated assessments, vote, replay rules, and checkpoint each record."""
    from ids_meta.common.study_consensus import combine_assessments, normalize_consensus, validate_consensus

    validate_consensus(consecutive_runs, consensus_threshold)
    if output.suffix.casefold() != ".csv" or output.resolve() == input_path.resolve():
        raise ValueError("study classification requires a separate CSV output")
    limit = optional_limit(limit, "classification_limit")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_full_text_chars must be a positive integer")
    if isinstance(evidence_lookup_timeout_seconds, bool) or not isinstance(evidence_lookup_timeout_seconds, (int, float)) or not 0 < evidence_lookup_timeout_seconds < float('inf'):
        raise ValueError("evidence_lookup_timeout_seconds must be finite and positive")
    protocol = load_study_protocol(protocol_path)
    fields = load_assessment_fields(protocol)
    template, prompt_path, prompt_digest = load_assessment_template(protocol)
    input_digest = file_hash(input_path)
    candidates = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or any(not isinstance(r, dict) for r in candidates):
        raise ValueError("study candidates must be a JSON array")
    LocalSourceAdapter(protocol.contract).normalize(candidates)
    selected = candidates if limit is None else candidates[:limit]
    if any(not isinstance(row.get("full_text", ""), str) for row in selected):
        raise ValueError("candidate full_text must be a string")
    checkpoint = output.with_suffix(".assessments.json")
    config = {**ollama_config, "max_output_tokens": max(4000, int(ollama_config.get("max_output_tokens", 4000)))}
    model = str(config.get("model", ""))
    fingerprint = content_hash({"protocol": protocol.sha256, "source_profile": protocol.source_profile_sha256,
        "prompt": PROMPT_VERSION, "prompt_sha256": prompt_digest, "model": model,
        "evidence_normalization": EVIDENCE_NORMALIZATION,
        "inference": {k: config.get(k) for k in ('temperature', 'seed', 'context_length', 'max_output_tokens')},
        "max_chars": max_chars, "fetch_full_text": fetch_full_text, "dry_run": dry_run,
        "consecutive_runs": consecutive_runs, "consensus_threshold": consensus_threshold,
        "evidence_lookup_timeout_seconds": evidence_lookup_timeout_seconds})
    cached = {}
    if resume and checkpoint.is_file() and output.is_file():
        previous = json.loads(checkpoint.read_text())
        if previous.get("fingerprint") == fingerprint and previous.get("output_sha256") == file_hash(output):
            cached = {r["input_record_sha256"]: r for r in previous["records"]}
    assessed, rows = [], []
    payload = {"schema_version": "1.0", "fingerprint": fingerprint, "completed": False,
        "input_sha256": input_digest, "protocol_sha256": protocol.sha256,
        "source_profile_sha256": protocol.source_profile_sha256,
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_digest, "model": model,
        "evidence_normalization": EVIDENCE_NORMALIZATION,
        "inference": {k: config.get(k) for k in ('temperature', 'seed', 'context_length', 'max_output_tokens', 'timeout_seconds', 'retries')},
        "max_full_text_chars": max_chars, "fetch_full_text": fetch_full_text, "dry_run": dry_run,
        "evidence_lookup_timeout_seconds": evidence_lookup_timeout_seconds,
        "consecutive_runs": consecutive_runs, "consensus_threshold": consensus_threshold, "records": assessed}
    for position, candidate in enumerate(selected, 1):
        digest = content_hash(candidate)
        prior = cached.get(digest)
        if prior and prior["assessment_status"] == "success":
            record = prior
            LOG.info("restored study assessment %d/%d", position, len(selected))
        else:
            record = {k: v for k, v in candidate.items() if k not in {"criteria", *SCREENING_FIELDS, *fields}}
            text = record.get("full_text", "")
            if not text.strip() and fetch_full_text and not dry_run:
                LOG.info("fetching study PDF paper_id=%s", record[protocol.contract.identity])
                record.update(download_study_text(str(record.get("pdf_url", "")), max_chars, evidence_lookup_timeout_seconds))
                text = record.get("full_text", "")
            if len(text) > max_chars:
                record.update(full_text_original_sha256=content_hash(text), full_text_complete=False)
            record["full_text"] = text[:max_chars]
            record["input_record_sha256"] = digest
            votes = []
            for run in range(consecutive_runs):
                label = f"run-{run + 1}"
                if dry_run or not record["full_text"].strip():
                    votes.append({"label": label, "status": "dry_run" if dry_run else "missing_full_text", "attempts": []})
                    continue
                try:
                    client = client or OllamaClient(config)
                    LOG.info("assessing study paper_id=%s model=%s run=%s text_chars=%d",
                        record[protocol.contract.identity], model, label, len(record["full_text"]))
                    votes.append(_assess_once(record, protocol, fields, template, client, label))
                except (ValueError, RuntimeError, requests.RequestException) as exc:
                    votes.append({"label": label, "status": "error", "attempts": [], "error_type": type(exc).__name__})
            record = combine_assessments(record, votes, protocol, fields, consensus_threshold)
            successes = sum(vote["status"] == "success" for vote in votes)
            errors = [vote for vote in votes if vote["status"] == "error"]
            record["assessment_status"] = ("success" if successes == consecutive_runs else "partial_error" if successes
                else "error" if errors else votes[0]["status"])
            record["assessment_attempts_json"] = json.dumps([entry for vote in votes for entry in vote["attempts"]], sort_keys=True)
            responses = [vote["model_response_json"] for vote in votes if "model_response_json" in vote]
            if responses:
                record["model_response_json"] = responses[-1]
            if errors:
                record.update(assessment_error=errors[0]["error_type"], assessment_error_detail=errors[0].get("error_detail", ""))
        normalized = normalize_consensus(record, protocol)
        normalized.update(classifier_method="llm", classifier_id=f"study_llm:{model}", model=model,
            classifier_supported_fields="*", consecutive_runs=consecutive_runs,
            classifier_profile_hash=fingerprint, candidate_input_sha256=input_digest)
        assessed.append(record)
        rows.append(normalized)
        atomic_write_csv(output, rows, list(dict.fromkeys(k for row in rows for k in row)))
        payload["output_sha256"] = file_hash(output)
        _write_json(checkpoint, payload)
        LOG.info("study classification %d/%d status=%s decision=%s", position, len(selected), record["assessment_status"], normalized["decision"])
    if not rows:
        atomic_write_csv(output, [], [protocol.contract.identity, "decision", "classifier_supported_fields", *fields])
        payload["output_sha256"] = file_hash(output)
    if file_hash(input_path) != input_digest or file_hash(prompt_path) != prompt_digest:
        raise ValueError("study candidate or assessment prompt changed during classification")
    current_protocol = load_study_protocol(protocol_path)
    if (current_protocol.sha256, current_protocol.source_profile_sha256) != (protocol.sha256, protocol.source_profile_sha256):
        raise ValueError("study protocol inputs changed during classification")
    payload["completed"] = True
    _write_json(checkpoint, payload)
    return rows
