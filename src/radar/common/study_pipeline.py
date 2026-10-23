"""Independent discovery and structured assessment for external-review replication."""
from __future__ import annotations

import copy
import json
import tomllib
from pathlib import Path
from string import Template
from typing import Any

import requests

from radar.common.quality import QUALITY_COLUMNS
from radar.common.quality_config import render_quality_prompt
from radar.common.csvio import atomic_write_csv
from radar.common.logging import logger
from radar.common.limits import optional_limit
from radar.common.jsonio import write_json as _write_json
from radar.common.paper_evidence import DEFAULT_FULL_TEXT_PROVIDERS, FULL_TEXT_RETRIEVAL_POLICY, PaperEvidenceService
from radar.common.paper_collection import eligible_papers, prefilter_paper
from radar.common.evidence import merge_new_evidence
from radar.common.evidence_blocks import BLOCK_ID_PATTERN, EVIDENCE_BLOCK_VERSION, MAX_BLOCK_CHARS, EvidenceBlockIndex, clip_structure
from radar.common.ollama import OllamaClient
from radar.common.provenance import file_hash
from radar.common.run import utc_now
from radar.common.sources import LocalSourceAdapter, content_hash
from radar.common.screening import load_screening_profile
from radar.common.study_screening import (
    SCREENING_FIELDS, load_study_protocol,
)

LOG = logger("study")
PROMPT_VERSION = "external-review-independent-quality-v2"


def collect_study(profile_path: Path, output: Path, *, limit: int | str | None = 100,
                  client: Any = None,
                  source_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Collect one configured provider through the same union discovery runner."""
    from radar.adapters.registry import create_adapter
    from radar.common.discovery import DiscoverySource, collect_sources

    with profile_path.open("rb") as handle:
        provider = tomllib.load(handle).get("provider")
    config = source_config or {}
    adapter = create_adapter(provider, config, profile_path, client=client)
    return collect_sources([DiscoverySource(provider, profile_path, config, adapter)], output,
        limit=limit, entity_kind="paper")



def preserve_study_input(input_path: Path, output: Path) -> None:
    """Freeze an explicitly supplied run input before any evidence acquisition."""
    data = input_path.read_bytes()
    records = json.loads(data)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("study input must be a JSON array of candidate records")
    output.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() != output.resolve():
        output.write_bytes(data)
    digest = file_hash(output)
    audit = output.with_suffix(".collection.json")
    if audit.is_file() and json.loads(audit.read_text(encoding="utf-8")).get("candidate_sha256") == digest:
        return
    providers = sorted({member["provider"] for row in records for member in row.get("source_memberships", [])
        if isinstance(member, dict) and isinstance(member.get("provider"), str)})
    _write_json(audit, {"schema_version": "1.0", "origin": "explicit_run_input", "providers": providers,
        "candidate_sha256": digest, "input_path": str(input_path.resolve()), "retained": len(records),
        "collected_at": utc_now()})


def prefilter_study(input_path: Path, output: Path, profile: dict[str, Any], settings: dict[str, Any],
                    evidence_service: PaperEvidenceService, *, dry_run: bool = False,
                    freeze_uncertain_evidence: bool = False) -> list[dict[str, Any]]:
    """Freeze shared paper evidence and scores before models use different scientific strategies."""
    if input_path.resolve() == output.resolve() or output.suffix.lower() != ".json":
        raise ValueError("paper prefilter requires a separate JSON output")
    input_digest = file_hash(input_path)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("paper prefilter requires a JSON array")
    identities = []
    for row in rows:
        identity = row.get("paper_id")
        if not isinstance(identity, (str, int)) or isinstance(identity, bool) or not str(identity).strip():
            raise ValueError("paper prefilter requires paper_id")
        identities.append(str(identity).strip())
        if not isinstance(row.get("full_text", ""), str):
            raise ValueError("candidate full_text must be a string")
    if len(set(identities)) != len(identities):
        raise ValueError("paper prefilter requires unique paper_id values")
    audit = input_path.with_suffix(".collection.json")
    value = None
    if audit.is_file():
        value = json.loads(audit.read_text(encoding="utf-8"))
        if value.get("candidate_sha256") != input_digest:
            raise ValueError("collection audit does not match paper prefilter input")
    for row in rows:
        if not dry_run:
            row.update(evidence_service.prepare(row, uncertain=freeze_uncertain_evidence))
        surveys = {relation for relation in row.get("discovery_relations", []) if relation.startswith("survey_reference:")}
        row.update(prefilter_paper(row, profile, settings, len(surveys)))
    if file_hash(input_path) != input_digest:
        raise ValueError("paper prefilter input changed during acquisition")
    _write_json(output, rows)
    if value is not None:
        _write_json(output.with_suffix(".collection.json"), {**value, "candidate_sha256": file_hash(output),
            "prefilter_input_sha256": input_digest, "evidence_policy": evidence_service.config,
            "evidence_retrieval_policy": FULL_TEXT_RETRIEVAL_POLICY})
    return rows


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
    """Ask models to select any number of distinct source blocks, never to copy text."""
    return {"type": "object", "additionalProperties": False, "required": ["value", "evidence_ids"],
        "properties": {"value": {"type": "string", "enum": values},
            "evidence_ids": {"type": "array", "uniqueItems": True,
                "items": {"type": "string", "pattern": BLOCK_ID_PATTERN}}}}


def load_assessment_template(protocol: Any) -> tuple[str, Path, str]:
    """Load a study-owned prompt and require only blind assessment inputs."""
    path = load_screening_profile(protocol.path).prompt_path
    if file_hash(protocol.quality_config_path) != protocol.quality_config['_sha256']:
        raise ValueError('quality configuration changed before assessment')
    digest = file_hash(path)
    with path.open("rb") as handle:
        prompt = tomllib.load(handle)
    template = prompt.get("template")
    if prompt.get("schema_version") != "1.0" or not isinstance(template, str) or not Template(template).is_valid():
        raise ValueError("study assessment prompt requires schema_version 1.0 and a valid template")
    if set(Template(template).get_identifiers()) != {"eligibility", "quality", "fields", "evidence"}:
        raise ValueError("study assessment prompt requires exactly eligibility, quality, fields and evidence placeholders")
    return template, path, digest


def _assessment_prompt(protocol: Any, fields: dict[str, Any], row: dict[str, Any], template: str, index: EvidenceBlockIndex) -> tuple[str, dict[str, Any]]:
    """Build a blind assessment prompt and its response schema."""
    criteria = {key: _judgment_schema(["yes", "no", "unknown"]) for key in [*protocol.filters, *protocol.quality]}
    limit = protocol.quality_config["evidence"]["max_block_passages"]
    if limit:
        for key in protocol.quality:
            criteria[key]["properties"]["evidence_ids"]["maxItems"] = limit
    taxonomy = {key: _judgment_schema(spec["values"]) for key, spec in fields.items()}
    schema = {"type": "object", "additionalProperties": False, "required": ["criteria", "fields"], "properties": {
        "criteria": {"type": "object", "additionalProperties": False, "required": list(criteria), "properties": criteria},
        "fields": {"type": "object", "additionalProperties": False, "required": list(taxonomy), "properties": taxonomy}}}
    prompt = Template(template).substitute(eligibility=json.dumps(protocol.filters), quality=render_quality_prompt(protocol.quality, protocol.quality_config, "blocks"),
        fields=json.dumps(fields), evidence=json.dumps({'title': row.get('title', ''), 'abstract': row.get('abstract', ''),
            'full_text_complete': row.get('full_text_complete', False),
            'evidence_fingerprint': index.fingerprint, 'evidence_blocks': index.prompt_blocks()}, ensure_ascii=False))
    return prompt, schema


def _assessment_errors(answer: Any, protocol: Any, fields: dict[str, Any], index: EvidenceBlockIndex) -> dict[str, str]:
    """Distinguish malformed responses from unsupported individual block references."""
    if not isinstance(answer, dict) or set(answer) != {"criteria", "fields"}:
        return {"response": "model output must contain criteria and fields"}
    specifications = {
        "criteria": {key: ["yes", "no", "unknown"] for key in [*protocol.filters, *protocol.quality]},
        "fields": {key: spec["values"] for key, spec in fields.items()},
    }
    errors = {}
    for section, specifications_for_section in specifications.items():
        judgments = answer[section]
        if not isinstance(judgments, dict) or set(judgments) - set(specifications_for_section):
            errors[section] = "assess exactly the required keys"
            continue
        for key, values in specifications_for_section.items():
            error = "missing required criterion" if key not in judgments else index.judgment_error(judgments[key], values)
            limit = protocol.quality_config["evidence"]["max_block_passages"]
            if not error and section == "criteria" and key in protocol.quality and limit and len(judgments[key]["evidence_ids"]) > limit:
                error = "quality evidence_ids exceeds configured max_block_passages"
            if error:
                errors[f"{section}.{key}"] = error
    return errors


def _validate_assessment(answer: Any, protocol: Any, fields: dict[str, Any], index: EvidenceBlockIndex) -> None:
    """Require complete judgments grounded in this frozen extraction's block IDs."""
    errors = _assessment_errors(answer, protocol, fields, index)
    if errors:
        raise ValueError("; ".join(f"{key}: {reason}" for key, reason in errors.items()))


def _assess_once(record: dict[str, Any], protocol: Any, fields: dict[str, Any], template: str,
                 client: Any, label: str, index: EvidenceBlockIndex) -> dict[str, Any]:
    """Resolve one response locally and abstain on invalid references without a correction call."""
    prompt, schema = _assessment_prompt(protocol, fields, record, template, index)
    vote: dict[str, Any] = {"label": label, "status": "error", "attempts": []}
    try:
        response = client.generate(prompt, schema)
        vote["model_response_json"] = json.dumps(response, sort_keys=True)
        entry = {"response": copy.deepcopy(response), "validation_error": ""}
        vote["attempts"].append(entry)
        answer = copy.deepcopy(response)
        errors = _assessment_errors(answer, protocol, fields, index)
        quality_errors = {path: reason for path, reason in errors.items() if path.startswith("criteria.") and path.split(".")[1] in protocol.quality}
        if quality_errors:
            vote["quality_errors"] = {path: reason for path, reason in quality_errors.items()
                if reason != "evidence_ids must select distinct blocks from supplied full_text"}
            for path, reason in quality_errors.items():
                if reason == "evidence_ids must select distinct blocks from supplied full_text":
                    vote.setdefault("unsupported_evidence", []).append({"field": path, "judgment": copy.deepcopy(answer["criteria"].get(path.split(".")[1])), "reason": reason})
                answer["criteria"][path.split(".")[1]] = {"value": "unknown", "evidence_ids": []}
            errors = {path: reason for path, reason in errors.items() if path not in quality_errors}
        if errors:
            entry["validation_error"] = "; ".join(f"{key}: {reason}" for key, reason in errors.items())
            if any(reason != "evidence_ids must select distinct blocks from supplied full_text" for reason in errors.values()):
                raise ValueError(entry["validation_error"])
            unresolved = []
            for path, reason in errors.items():
                group, key = path.split(".")
                unresolved.append({"field": path, "judgment": copy.deepcopy(answer[group][key]), "reason": reason})
                answer[group][key] = {"value": "unknown", "evidence_ids": []}
            vote.setdefault("unsupported_evidence", []).extend(unresolved)
            LOG.warning("unsupported study evidence paper_id=%s run=%s fields=%s; using unknown",
                record[protocol.contract.identity], label, ",".join(errors))
        _validate_assessment(answer, protocol, fields, index)
        vote.update(status="success", answer=answer)
    except (RuntimeError, ValueError, requests.RequestException) as exc:
        vote.update(error_type=type(exc).__name__, error_detail=str(exc) if isinstance(exc, ValueError) else "")
        LOG.warning("study model assessment failed paper_id=%s run=%s (%s) detail=%s",
            record[protocol.contract.identity], label, type(exc).__name__, vote["error_detail"])
    return vote


def _assessment_votes(record: dict[str, Any], protocol: Any, fields: dict[str, Any], template: str,
                      client: Any, config: dict[str, Any], consecutive_runs: int, dry_run: bool) -> tuple[list[dict[str, Any]], Any]:
    """Assess one frozen evidence version, retaining failures in the vote denominator."""
    votes = []
    index = EvidenceBlockIndex.from_record(record)
    record.update(evidence_fingerprint=index.fingerprint, evidence_text_sha256=index.text_sha256)
    for run in range(consecutive_runs):
        label = f"run-{run + 1}"
        if dry_run or not record["full_text"].strip():
            votes.append({"label": label, "status": "dry_run" if dry_run else "missing_full_text", "attempts": []})
            continue
        try:
            client = client or OllamaClient(config)
            votes.append(_assess_once(record, protocol, fields, template, client, label, index))
        except (ValueError, RuntimeError, requests.RequestException) as exc:
            votes.append({"label": label, "status": "error", "attempts": [], "error_type": type(exc).__name__})
    return votes, client


def classify_study(protocol_path: Path, input_path: Path, output: Path, ollama_config: dict[str, Any], *,
                   limit: int | None = None, resume: bool = False, dry_run: bool = False,
                   fetch_full_text: bool = False, max_chars: int = 120000, client: Any = None,
                   consecutive_runs: int = 1, consensus_threshold: float = 0.8,
                   evidence_lookup_timeout_seconds: float = 30, enrich_missing_abstracts: bool = False,
                   enrich_missing_full_text: bool = False, full_text_lookup_providers: list[str] | None = None,
                   enrich_uncertain_with_full_text: bool = False, classify_borderline: bool = True,
                   evidence_service: PaperEvidenceService | None = None, quality_path: Path | None = None) -> list[dict[str, Any]]:
    """Repeat validated assessments, vote, replay rules, and checkpoint each record."""
    from radar.common.study_consensus import combine_assessments, normalize_consensus, validate_consensus

    validate_consensus(consecutive_runs, consensus_threshold)
    if output.suffix.casefold() != ".csv" or output.resolve() == input_path.resolve():
        raise ValueError("study classification requires a separate CSV output")
    limit = optional_limit(limit, "classification_limit")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_full_text_chars must be a positive integer")
    if isinstance(evidence_lookup_timeout_seconds, bool) or not isinstance(evidence_lookup_timeout_seconds, (int, float)) or not 0 < evidence_lookup_timeout_seconds < float('inf'):
        raise ValueError("evidence_lookup_timeout_seconds must be finite and positive")
    protocol = load_study_protocol(protocol_path, quality_path=quality_path)
    fields = load_assessment_fields(protocol)
    template, prompt_path, prompt_digest = load_assessment_template(protocol)
    input_digest = file_hash(input_path)
    candidates = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or any(not isinstance(r, dict) for r in candidates):
        raise ValueError("study candidates must be a JSON array")
    LocalSourceAdapter(protocol.contract).normalize(candidates)
    eligible = eligible_papers(candidates, classify_borderline)
    selected = eligible if limit is None else eligible[:limit]
    evidence_policy = {"fetch_full_text": fetch_full_text, "max_full_text_chars": max_chars,
        "evidence_lookup_timeout_seconds": evidence_lookup_timeout_seconds,
        "enrich_missing_abstracts": enrich_missing_abstracts, "enrich_missing_full_text": enrich_missing_full_text,
        "full_text_lookup_providers": list(full_text_lookup_providers) if full_text_lookup_providers is not None else list(DEFAULT_FULL_TEXT_PROVIDERS),
        "enrich_uncertain_with_full_text": enrich_uncertain_with_full_text}
    evidence_service = evidence_service or PaperEvidenceService(evidence_policy)
    if any(not isinstance(row.get("full_text", ""), str) for row in selected):
        raise ValueError("candidate full_text must be a string")
    checkpoint = output.with_suffix(".assessments.json")
    config = {**ollama_config, "max_output_tokens": max(4000, int(ollama_config.get("max_output_tokens", 4000)))}
    model = str(config.get("model", ""))
    fingerprint = content_hash({"protocol": protocol.sha256, "source_profile": protocol.source_profile_sha256,
        "prompt": PROMPT_VERSION, "prompt_sha256": prompt_digest, "quality_config_sha256": protocol.quality_config["_sha256"], "model": model,
        "evidence_block_version": EVIDENCE_BLOCK_VERSION, "evidence_block_max_chars": MAX_BLOCK_CHARS,
        "inference": {k: config.get(k) for k in ('temperature', 'seed', 'context_length', 'max_output_tokens')},
        "max_chars": max_chars, "evidence_policy": evidence_policy, "evidence_retrieval_policy": FULL_TEXT_RETRIEVAL_POLICY, "classify_borderline": classify_borderline, "dry_run": dry_run,
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
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_digest, "quality_config_sha256": protocol.quality_config["_sha256"],
        "quality_config_path": str(protocol.quality_config_path), "quality_config": protocol.quality_config, "model": model,
        "evidence_retrieval_policy": FULL_TEXT_RETRIEVAL_POLICY,
        "evidence_block_version": EVIDENCE_BLOCK_VERSION, "evidence_block_max_chars": MAX_BLOCK_CHARS,
        "inference": {k: config.get(k) for k in ('temperature', 'seed', 'context_length', 'max_output_tokens', 'timeout_seconds', 'retries')},
        "max_full_text_chars": max_chars, "fetch_full_text": fetch_full_text, "dry_run": dry_run,
        "enrich_missing_abstracts": enrich_missing_abstracts, "enrich_missing_full_text": enrich_missing_full_text,
        "full_text_lookup_providers": list(full_text_lookup_providers) if full_text_lookup_providers is not None else list(DEFAULT_FULL_TEXT_PROVIDERS),
        "enrich_uncertain_with_full_text": enrich_uncertain_with_full_text,
        "classify_borderline": classify_borderline,
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
            if not dry_run:
                record.update(evidence_service.prepare(record))
            text = record.get("full_text", "")
            if len(text) > max_chars:
                record.update(full_text_original_sha256=content_hash(text), full_text_complete=False)
            record["full_text"] = text[:max_chars]
            if record.get("full_text_structure"):
                record["full_text_structure"] = clip_structure(record["full_text_structure"], max_chars)
            record["input_record_sha256"] = digest
            votes, client = _assessment_votes(record, protocol, fields, template, client, config,
                consecutive_runs, dry_run)
            record = combine_assessments(record, votes, protocol, fields, consensus_threshold)
            initial_votes = votes
            attempted, changed = False, False
            if not dry_run and enrich_uncertain_with_full_text and normalize_consensus(record, protocol)["relevance_decision"] == "uncertain":
                attempted = True
                discovered = evidence_service.enrich(record)
                changed = merge_new_evidence(record, discovered)
                if changed and record.get("full_text", "").strip():
                    text = record["full_text"]
                    if len(text) > max_chars:
                        record.update(full_text_original_sha256=content_hash(text), full_text_complete=False)
                    record["full_text"] = text[:max_chars]
                    if record.get("full_text_structure"):
                        record["full_text_structure"] = clip_structure(record["full_text_structure"], max_chars)
                    votes, client = _assessment_votes(record, protocol, fields, template, client, config,
                        consecutive_runs, False)
                    record = combine_assessments(record, votes, protocol, fields, consensus_threshold)
            record.update(evidence_expansion_attempted=attempted, evidence_expansion_changed=changed,
                initial_assessment_votes_json=json.dumps(initial_votes, sort_keys=True))
            successes = sum(vote["status"] == "success" for vote in votes)
            errors = [vote for vote in votes if vote["status"] == "error"]
            record["assessment_status"] = ("success" if successes == consecutive_runs else "partial_error" if successes
                else "error" if errors else votes[0]["status"])
            record["assessment_attempts_json"] = json.dumps([entry for vote in votes for entry in vote["attempts"]], sort_keys=True)
            record["assessment_evidence_warnings_json"] = json.dumps([
                {"run": vote["label"], **warning} for vote in votes for warning in vote.get("unsupported_evidence", [])], sort_keys=True)
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
        atomic_write_csv(output, [], [protocol.contract.identity, "decision", "classifier_supported_fields", *fields, *QUALITY_COLUMNS])
        payload["output_sha256"] = file_hash(output)
    if file_hash(input_path) != input_digest or file_hash(prompt_path) != prompt_digest or file_hash(protocol.quality_config_path) != protocol.quality_config["_sha256"]:
        raise ValueError("study candidates, assessment prompt or shared quality configuration changed during classification")
    current_protocol = load_study_protocol(protocol_path, quality_path=quality_path)
    if (current_protocol.sha256, current_protocol.source_profile_sha256) != (protocol.sha256, protocol.source_profile_sha256):
        raise ValueError("study protocol inputs changed during classification")
    payload["completed"] = True
    _write_json(checkpoint, payload)
    return rows
