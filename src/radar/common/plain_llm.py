"""One model decision from collection-time evidence, without RADAR adjudication."""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

import requests

from radar.common.baselines import CLASSIFIER_COLUMNS, baseline_inputs
from radar.common.cache import COLLECTED_EVIDENCE_POLICY, PipelineCache, collected_record_hash
from radar.common.config import ConfigError, Settings, load_prompt_profile, load_system_type_profile
from radar.common.csvio import atomic_write_csv, read_csv
from radar.common.evidence_blocks import EvidenceBlockIndex, clip_structure
from radar.common.jsonio import write_json
from radar.common.limits import optional_limit
from radar.common.logging import logger
from radar.common.ollama import OllamaClient
from radar.common.paper_collection import eligible_papers
from radar.common.provenance import file_hash
from radar.common.run import utc_now
from radar.common.scoring import shortlist_candidates
from radar.common.sources import content_hash
from radar.common.text import slug

LOG = logger("plain_llm")
VERSION = "plain-llm-1"
REVIEW_EVIDENCE_POLICY = "frozen-review-collection-v1"
AUDIT_COLUMNS = ["plain_llm_model_digest", "plain_llm_prompt_hash", "plain_llm_schema_hash",
    "plain_llm_raw_response", "plain_llm_response_metadata_json", "plain_llm_elapsed_seconds",
    "plain_llm_request_count", "plain_llm_cache_reused", "plain_llm_status", "plain_llm_error_type"]
METRICS = ("model", "done", "done_reason", "created_at", "total_duration", "load_duration",
    "prompt_eval_count", "prompt_eval_cached_count", "prompt_eval_duration", "eval_count", "eval_duration")


def plain_llm_models(settings: Settings) -> list[str]:
    """Select independent baseline models, defaulting to the RADAR model list."""
    raw = settings.section("baselines").get("plain_llm", {})
    if not isinstance(raw, dict):
        raise ConfigError("baselines.plain_llm must be a table")
    allowed = {"models", "temperature", "seed", "max_output_tokens", "context_length", "timeout_seconds"}
    if set(raw) - allowed:
        raise ConfigError("unknown plain LLM settings: " + ", ".join(sorted(set(raw) - allowed)))
    models = raw.get("models", settings.section("ollama").get("models"))
    if not isinstance(models, list) or not models or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ConfigError("plain LLM models must be a non-empty array of model names")
    models = list(dict.fromkeys(m.strip() for m in models))
    if len({slug(m) for m in models}) != len(models):
        raise ConfigError("plain LLM models must have distinct output directory slugs")
    for model in models:
        plain_llm_config(settings, model)
    return models


def plain_llm_config(settings: Settings, model: str) -> dict[str, Any]:
    """Inherit generation settings while making the zero-retry policy immutable."""
    config = dict(settings.section("ollama"))
    config.pop("models", None)
    config.update({k: v for k, v in settings.section("baselines").get("plain_llm", {}).items() if k != "models"})
    config.update(model=model, retries=0)
    for name, default, minimum in (("seed", 42, None), ("context_length", 0, 0), ("max_output_tokens", 900, 1)):
        value = config.setdefault(name, default)
        if type(value) is not int or minimum is not None and value < minimum:
            raise ConfigError(f"plain LLM {name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    for name, default, minimum in (("temperature", 0, 0), ("timeout_seconds", 300, 0)):
        value = config.setdefault(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or name == "timeout_seconds" and value == 0:
            raise ConfigError(f"plain LLM {name} is invalid")
    if settings.section("study").get("workflow") == "systematic_review" and "max_output_tokens" not in settings.section("baselines").get("plain_llm", {}):
        config["max_output_tokens"] = max(4000, config["max_output_tokens"])
    return config


def installed_model_digests(config: dict[str, Any], models: list[str]) -> dict[str, str]:
    """Read installed model identities once; this metadata request never retries."""
    headers = {"Authorization": f"Bearer {config['api_key']}"} if config.get("api_key") else {}
    response = requests.get(str(config.get("url", "http://127.0.0.1:11434")).rstrip("/") + "/api/tags",
        headers=headers, timeout=config.get("timeout_seconds", 300))
    response.raise_for_status()
    body = response.json()
    tags = body.get("models", []) if isinstance(body, dict) else []
    if not isinstance(tags, list):
        raise ConfigError("Ollama model inventory must contain a model array")
    result = {}
    for model in models:
        names = {model, model + ":latest"} if ":" not in model.rsplit("/", 1)[-1] else {model}
        matches = [tag for tag in tags if isinstance(tag, dict) and (tag.get("name") in names or tag.get("model") in names)]
        if len(matches) != 1 or not isinstance(matches[0].get("digest"), str) or not matches[0]["digest"]:
            raise ConfigError(f"cannot pin installed plain LLM model digest: {model}")
        result[model] = matches[0]["digest"]
    return result



def _radar_shortlist_ids(settings: Settings, source: str, identity: str) -> list[str] | None:
    """Reuse only candidate identities from saved RADAR models, never their judgments."""
    cohorts = []
    for model in settings.section("ollama").get("models", []):
        path = settings.path("interim_dir") / source / "models" / slug(model) / "classifications.csv"
        if path.is_file():
            rows = read_csv(path)
            ids = [row.get(identity, "").strip() for row in rows]
            if any(not key for key in ids) or len(set(ids)) != len(ids):
                raise ValueError("saved RADAR shortlist requires unique non-empty identities")
            cohorts.append(ids)
    if not cohorts:
        return None
    if any(set(ids) != set(cohorts[0]) for ids in cohorts[1:]):
        raise ValueError("saved RADAR models have different candidate cohorts; finish a matched run before plain LLM benchmarking")
    return cohorts[0]


def plain_llm_inputs(settings: Settings, source: str, candidates_path: Path, cache: PipelineCache | None,
                     *, original_path: Path | None = None, limit: int | None = None) -> list[tuple[dict, dict]]:
    """Match RADAR's deterministic shortlist, then retrieve only original evidence."""
    classification = settings.section("pipeline").get("classification", {})
    cap = optional_limit(classification.get("classification_limit"), "classification_limit")
    workflow = settings.section("study").get("workflow")
    identity = "paper_id" if workflow == "systematic_review" else "github_repo_id" if source == "github" else "openalex_id"
    observed_ids = _radar_shortlist_ids(settings, source, identity)
    if workflow == "systematic_review":
        if original_path is None:
            raise ConfigError("plain review baseline requires the original collection path")
        audit = json.loads(original_path.with_suffix(".collection.json").read_text(encoding="utf-8"))
        if audit.get("candidate_sha256") != file_hash(original_path):
            raise ValueError("original review collection does not match its collection audit")
        originals = json.loads(original_path.read_text(encoding="utf-8"))
        candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
        for rows in (originals, candidates):
            if not isinstance(rows, list) or any(not isinstance(row, dict) or not isinstance(row.get("paper_id"), str) or not row["paper_id"].strip() for row in rows):
                raise ValueError("plain review baseline requires paper_id in every candidate")
            if len({row["paper_id"] for row in rows}) != len(rows):
                raise ValueError("plain review baseline requires unique paper_id values")
        by_id = {row["paper_id"]: row for row in originals}
        selected = eligible_papers(candidates, classification.get("classify_borderline", True))[:cap]
        if observed_ids is not None:
            selected = candidates
        if any(row["paper_id"] not in by_id for row in selected):
            raise ValueError("review shortlist contains IDs absent from the original collection")
        inputs = [(row, by_id[row["paper_id"]]) for row in selected]
    else:
        if cache is None:
            raise ConfigError("plain catalogue baseline requires the collection cache")
        inputs = baseline_inputs(source, candidates_path, cache)
        identity = "github_repo_id" if source == "github" else "openalex_id"
        candidates = [row for row, _ in inputs]
        if source == "openalex":
            candidates = eligible_papers(candidates, classification.get("classify_borderline", True))
        selected = shortlist_candidates(candidates, identity, cap, "retrieval_rank_score" if source == "openalex" else None)
        if observed_ids is not None:
            selected = [row for row, _ in inputs]
        by_id = {row[identity]: payload for row, payload in inputs}
        if len(by_id) != len(inputs):
            raise ValueError("plain LLM candidate identities must be unique")
        inputs = [(row, by_id[row[identity]]) for row in selected]
    if observed_ids is not None:
        by_identity = {row[identity]: (row, payload) for row, payload in inputs}
        if any(key not in by_identity for key in observed_ids):
            raise ValueError("saved RADAR shortlist contains IDs absent from the current candidate cohort")
        inputs = [by_identity[key] for key in observed_ids]
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ConfigError("plain LLM pilot limit must be positive")
        inputs = inputs[:limit]
    return inputs


def plain_llm_task(settings: Settings, source: str, payload: dict) -> tuple[str, dict]:
    """Reuse study instructions and vocabulary, requesting the model's own decision."""
    workflow = settings.section("study").get("workflow")
    if workflow == "systematic_review":
        from radar.common.study_pipeline import _assessment_prompt, load_assessment_fields, load_assessment_template
        from radar.common.study_screening import load_study_protocol

        protocol = load_study_protocol(settings.path("screening"))
        fields = load_assessment_fields(protocol)
        template, _, _ = load_assessment_template(protocol)
        record = copy.deepcopy(payload)
        max_chars = settings.section("pipeline").get("evidence", {}).get("max_full_text_chars", 120000)
        text = record.get("full_text", "")
        if not isinstance(text, str):
            raise ValueError("original review full_text must be a string")
        if len(text) > max_chars:
            record["full_text_complete"] = False
        record["full_text"] = text[:max_chars]
        if record.get("full_text_structure"):
            record["full_text_structure"] = clip_structure(record["full_text_structure"], max_chars)
        prompt, schema = _assessment_prompt(protocol, fields, record, template, EvidenceBlockIndex.from_record(record))
        instruction = (f"Make the final review decision yourself: accept requires all eligibility rules to be yes "
            f"and at least {protocol.quality_minimum} quality rules to be yes. Exclude if an eligibility rule is no "
            "or the quality minimum cannot be reached even if all unknown quality rules were yes. Otherwise return uncertain.")
        decisions = ["accept", "exclude", "uncertain"]
    elif source == "github":
        from radar.github.pipeline import evidence_sections, prompt as build_prompt
        from radar.github.schema import response_schema

        types = load_system_type_profile(settings, source)
        prompt = build_prompt(evidence_sections(payload), load_prompt_profile(settings, source), types)
        schema = response_schema(types)
        instruction = "Make the final inclusion decision yourself using the study eligibility rules: accept if satisfied, exclude if demonstrably failed, uncertain if evidence is insufficient."
        decisions = ["accept", "exclude", "uncertain"]
    else:
        from radar.openalex.pipeline import evidence_sections, prompt as build_prompt
        from radar.openalex.schema import RESPONSE_SCHEMA

        prompt = build_prompt(evidence_sections(payload), load_prompt_profile(settings, source))
        schema = copy.deepcopy(RESPONSE_SCHEMA)
        instruction = ("Make the final inclusion decision yourself using the study eligibility rules: accept for an eligible core release, "
            "accept_adjacent for an eligible adjacent release, exclude if demonstrably failed, uncertain if evidence is insufficient.")
        decisions = ["accept", "accept_adjacent", "exclude", "uncertain"]
    schema["properties"]["decision"] = {"type": "string", "enum": decisions}
    schema["required"].append("decision")
    if "primary_reason" not in schema["properties"]:
        schema["properties"]["primary_reason"] = {"type": "string"}
        schema["required"].append("primary_reason")
    prompt += "\nFINAL DECISION\n" + instruction + " Return all previously requested fields plus decision and primary_reason, matching the complete response schema."
    return prompt, schema


def validate_structure(value: Any, schema: dict, path: str = "response") -> None:
    """Check response shape and vocabulary without checking scientific consistency or evidence."""
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list), "string": isinstance(value, str),
        "boolean": type(value) is bool, "integer": type(value) is int, "number": type(value) in (int, float)}
    if kind in valid and not valid[kind]:
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value outside schema vocabulary")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value) or schema.get("additionalProperties") is False and set(value) - set(properties):
            raise ValueError(f"{path}: missing required or unexpected fields")
        for key, item in value.items():
            if key in properties:
                validate_structure(item, properties[key], path + "." + key)
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: invalid array length")
        if schema.get("uniqueItems") and len({content_hash(item) for item in value}) != len(value):
            raise ValueError(f"{path}: duplicate array items")
        for item in value:
            validate_structure(item, schema.get("items", {}), path + "[]")
    elif isinstance(value, str):
        if len(value) > schema.get("maxLength", math.inf) or schema.get("pattern") and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path}: invalid string format")


def _cell(value: Any) -> str:
    """Serialize model values without repairing labels, ordering, or decisions."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return ";".join(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def plain_llm_profile(settings: Settings, model: str, digest: str) -> dict:
    """Bind cached decisions to model weights, inference settings and scientific definitions."""
    config = plain_llm_config(settings, model)
    return {"version": VERSION, "model": model, "model_digest": digest,
        "workflow": settings.section("study").get("workflow"),
        "generation": {key: config[key] for key in ("temperature", "seed", "context_length", "max_output_tokens", "timeout_seconds")},
        "profiles": {name: file_hash(settings.path(name)) for name in ("screening", "taxonomy", "prompts")},
        "evidence_max_chars": settings.section("pipeline").get("evidence", {}).get("max_full_text_chars", 120000),
        "evidence_policy": REVIEW_EVIDENCE_POLICY if settings.section("study").get("workflow") == "systematic_review" else COLLECTED_EVIDENCE_POLICY,
        "requests_per_candidate": 1, "retries": 0, "enrichment": False, "reconciliation": False, "voting": False}


def classify_plain_llm(settings: Settings, source: str, candidates_path: Path, output: Path,
                       model: str, digest: str, inputs: list[tuple[dict, dict]], *, resume: bool = False) -> list[dict[str, str]]:
    """Make at most one generation per candidate and keep failures in the output cohort."""
    if output.resolve() == candidates_path.resolve() or output.suffix.lower() != ".csv":
        raise ConfigError("plain LLM requires a separate CSV output")
    review = settings.section("study").get("workflow") == "systematic_review"
    identity = "paper_id" if review else "github_repo_id" if source == "github" else "openalex_id"
    if review:
        from radar.common.study_screening import load_study_protocol

        metadata_columns = [identity, "title", "doi", "publication_year", "source_memberships", "identifiers"]
        fields = list(load_study_protocol(settings.path("screening")).contract.fields)
        columns = metadata_columns + fields + ["criteria_json", "field_assessments_json", "decision", "primary_reason", "error", "classified_at", "model"]
    else:
        if source == "github":
            from radar.github.schema import CANDIDATE_COLUMNS, CLASSIFICATION_COLUMNS
        else:
            from radar.openalex.schema import CANDIDATE_COLUMNS, CLASSIFICATION_COLUMNS
        metadata_columns = CANDIDATE_COLUMNS
        columns = CLASSIFICATION_COLUMNS
    columns = list(dict.fromkeys([*columns, *CLASSIFIER_COLUMNS, *AUDIT_COLUMNS, "profile_hash", "llm_model", "pipeline_version", "evidence_hash"]))
    profile = plain_llm_profile(settings, model, digest)
    profile_hash = content_hash(profile)
    input_digest = file_hash(candidates_path)
    checkpoint = output.with_suffix(".audit.json")
    cached = {}
    if resume and checkpoint.is_file() and output.is_file():
        previous = json.loads(checkpoint.read_text(encoding="utf-8"))
        if previous.get("profile_hash") == profile_hash and previous.get("output_sha256") == file_hash(output):
            cached = {entry["input_hash"]: entry for entry in previous.get("records", [])}
    records, rows = [], []
    audit = {"schema_version": "1.0", "profile": profile, "profile_hash": profile_hash, "candidate_input_sha256": input_digest,
        "selected_ids": [candidate[identity] for candidate, _ in inputs], "completed": False, "records": records}
    config = plain_llm_config(settings, model)
    client = None
    LOG.info("starting plain LLM source=%s model=%s rows=%d requests_per_candidate=1 retries=0 evidence_policy=%s", source, model, len(inputs), profile["evidence_policy"])
    for position, (candidate, payload) in enumerate(inputs, 1):
        prompt, schema = plain_llm_task(settings, source, payload)
        snapshot_hash = collected_record_hash(payload) if not review else content_hash(payload)
        input_hash = content_hash([candidate[identity], snapshot_hash, prompt, schema])
        entry = cached.get(input_hash)
        if entry is not None:
            entry = copy.deepcopy(entry)
            row = dict(entry["result"])
            row["plain_llm_cache_reused"] = "true"
            # Metadata follows the current common cohort, even when inference is reused.
            row.update({key: _cell(candidate[key]) for key in metadata_columns if key in candidate})
            entry["result"] = row
        else:
            client = client or OllamaClient(config)
            row = {key: _cell(candidate[key]) for key in metadata_columns if key in candidate}
            row.update(classifier_method="plain_llm", classifier_id="plain_llm:" + model,
                classifier_profile_hash=profile_hash, profile_hash=profile_hash, pipeline_version=VERSION,
                classifier_supported_fields="*", classifier_evidence_source="collection_snapshot",
                classifier_evidence_policy=profile["evidence_policy"], classifier_evidence_snapshot_hash=snapshot_hash,
                evidence_hash=input_hash, llm_model=model, model=model, classified_at=utc_now(), error="",
                plain_llm_model_digest=digest, plain_llm_prompt_hash=content_hash(prompt), plain_llm_schema_hash=content_hash(schema),
                plain_llm_request_count="1", plain_llm_cache_reused="false", plain_llm_status="success", plain_llm_error_type="")
            try:
                answer = client.generate(prompt, schema)
                metadata = client.last_response or {}
                if metadata.get("done") is not True or metadata.get("done_reason", "stop") != "stop":
                    raise ValueError("incomplete or truncated model response")
                expected_models = {model, model + ":latest"} if ":" not in model.rsplit("/", 1)[-1] else {model}
                if metadata.get("model") not in expected_models:
                    raise ValueError("response model differs from the selected plain LLM")
                validate_structure(answer, schema)
                if review:
                    row.update({name: _cell(judgment["value"]) for name, judgment in answer["fields"].items()})
                    row.update(criteria_json=_cell(answer["criteria"]), field_assessments_json=_cell(answer["fields"]),
                        decision=answer["decision"], primary_reason=answer["primary_reason"])
                else:
                    row.update({key: _cell(value) for key, value in answer.items() if key != "evidence"})
                    row.update(model_evidence_json=_cell(answer["evidence"]), evidence_json=_cell(answer["evidence"]), evidence_quote_verified="false")
                    if source == "openalex":
                        row["resource_name"] = answer["artifact_display_name"]
            except (ValueError, RuntimeError, requests.RequestException) as exc:
                row.update(decision="error", error=str(exc), plain_llm_status="error", plain_llm_error_type=type(exc).__name__)
                LOG.warning("plain LLM failed source=%s model=%s id=%s error=%s", source, model, candidate[identity], type(exc).__name__)
            metadata = {key: client.last_response[key] for key in METRICS if client.last_response and key in client.last_response}
            row.update(plain_llm_raw_response=client.last_raw_response, plain_llm_response_metadata_json=_cell(metadata),
                plain_llm_elapsed_seconds=str(client.last_elapsed_seconds))
            entry = {"id": candidate[identity], "input_hash": input_hash, "snapshot_hash": snapshot_hash,
                "prompt": prompt, "schema": schema, "result": row}
        rows.append(row)
        records.append(entry)
        atomic_write_csv(output, rows, columns)
        audit["output_sha256"] = file_hash(output)
        write_json(checkpoint, audit)
        LOG.info("plain LLM progress model=%s completed=%d/%d decision=%s", model, position, len(inputs), row["decision"])
    if not rows:
        atomic_write_csv(output, [], columns)
        audit["output_sha256"] = file_hash(output)
    if file_hash(candidates_path) != input_digest or plain_llm_profile(settings, model, digest) != profile:
        raise ValueError("plain LLM cohort or scientific settings changed during inference")
    audit["completed"] = True
    write_json(checkpoint, audit)
    return rows


def load_plain_llm_results(settings: Settings, output: Path, model: str) -> list[dict[str, str]]:
    """Read an intact completed baseline offline without making any model requests."""
    audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
    digest = audit.get("profile", {}).get("model_digest")
    if not digest or audit.get("profile") != plain_llm_profile(settings, model, digest) or audit.get("completed") is not True or audit.get("output_sha256") != file_hash(output):
        raise ConfigError("saved plain LLM results do not match the current configuration or audit")
    return read_csv(output)
