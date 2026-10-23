"""Independent, backend-specific content quality assessments and shared threshold rules."""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from typing import Any
from string import Template

from radar.common.ollama import EvidenceSection, format_evidence, verify_evidence
from radar.common.sources import content_hash
from radar.common.quality_config import assessment_config, render_quality_prompt

VERSION = "content-quality-v1"
QUALITY_COLUMNS = ["relevance_decision", "quality_status", "quality_qualified_decision", "quality_backend",
    "quality_profile_hash", "quality_config_hash", "quality_rules_json", "quality_assessments_json", "quality_supported_criteria",
    "quality_yes", "quality_no", "quality_unknown", "quality_not_assessed", "quality_error",
    "quality_evidence_hash", "quality_votes_json", "quality_inference_seconds"]


def quality_policy(rules: dict[str, Any]) -> dict[str, Any]:
    """Extract only the scientific quality definitions, independent of classifier taxonomy."""
    decision = rules.get("decision", {})
    return {"version": VERSION, "quality": rules.get("quality", {}),
        "minimum": decision.get("quality_minimum", 0), "required": decision.get("quality_required", [])}


def policy_rules(policy: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the same threshold inputs for offline reporting and consensus."""
    return {"quality": policy.get("quality", {}), "decision": {
        "quality_minimum": policy.get("minimum", 0), "quality_required": policy.get("required", [])}}


def quality_schema(schema: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    """Add a separate quality response object without changing the class schema definitions."""
    result = copy.deepcopy(schema)
    if not rules.get("quality"):
        return result
    judgment = {"type": "object", "additionalProperties": False, "required": ["value", "evidence"],
        "properties": {"value": {"type": "string", "enum": ["yes", "no", "unknown"]},
            "evidence": {"type": "array", "maxItems": assessment_config(rules)["evidence"]["max_quotation_passages"], "items": {"type": "object",
                "additionalProperties": False, "required": ["source", "quote"],
                "properties": {"source": {"type": "string"}, "quote": {"type": "string"}}}}}}
    result["properties"]["quality"] = {"type": "object", "additionalProperties": False,
        "required": list(rules["quality"]), "properties": {key: copy.deepcopy(judgment) for key in rules["quality"]}}
    result["required"].append("quality")
    return result


def quality_prompt(prompt: str, rules: dict[str, Any]) -> str:
    """Append framework-owned quality instructions, separate from study class templates."""
    if not rules.get("quality"):
        return prompt
    return prompt + "\n" + render_quality_prompt(rules["quality"], assessment_config(rules), "quotations")


def finish_quality(row: dict[str, Any], rules: dict[str, Any], judgments: dict[str, Any], backend: str,
                   *, supported: list[str] | None = None, error: str = "", evidence_hash: str = "") -> dict[str, Any]:
    """Keep relevance, quality and joint inclusion explicit; never overwrite the original decision."""
    policy = quality_policy(rules)
    criteria = policy["quality"]
    supported = list(criteria) if supported is None else supported
    assessments = {}
    for key in criteria:
        judgment = copy.deepcopy(judgments.get(key, {"value": "unknown" if key in supported else "not_assessed", "evidence": []}))
        if not isinstance(judgment, dict) or not isinstance(judgment.get("value"), str) or judgment["value"] not in {"yes", "no", "unknown", "not_assessed"}:
            judgment = {"value": "unknown", "evidence": [], "error": "invalid_quality_value"}
            error = error or "invalid_quality_value"
        assessments[key] = judgment
    counts = Counter(j["value"] for j in assessments.values())
    required = [assessments[key]["value"] for key in policy["required"]]
    if not criteria:
        status, error = "not_configured", ""
    elif error:
        status = "error"
    elif "no" in required or counts["yes"] + counts["unknown"] + counts["not_assessed"] < policy["minimum"]:
        status = "fail"
    elif all(value == "yes" for value in required) and counts["yes"] >= policy["minimum"]:
        status = "pass"
    elif not supported or "not_assessed" in required:
        status = "not_assessed"
    else:
        status = "unknown"
    relevance = str(row.get("relevance_decision") or row.get("decision") or "uncertain")
    if relevance in {"exclude", "error", "dry_run"}:
        qualified = relevance
    elif status == "fail":
        qualified = "exclude"
    elif status == "error":
        qualified = "error"
    elif relevance in {"accept", "accept_adjacent"} and status in {"pass", "not_configured"}:
        qualified = relevance
    else:
        qualified = "uncertain"
    row.update(relevance_decision=relevance, quality_status=status, quality_qualified_decision=qualified,
        quality_backend=backend, quality_profile_hash=content_hash(policy),
        quality_config_hash=rules.get("assessment", {}).get("_sha256", row.get("quality_config_hash", "")),
        quality_rules_json=json.dumps(policy, sort_keys=True), quality_assessments_json=json.dumps(assessments, sort_keys=True),
        quality_supported_criteria=";".join(supported), quality_yes=counts["yes"], quality_no=counts["no"],
        quality_unknown=counts["unknown"], quality_not_assessed=counts["not_assessed"], quality_error=error,
        quality_evidence_hash=evidence_hash)
    return row


def assess_llm_quality(row: dict[str, Any], raw: Any, sections: list[EvidenceSection], rules: dict[str, Any],
                       *, grounded: bool, backend: str = "radar_llm") -> dict[str, Any]:
    """Read the model's separate quality answer; RADAR grounds quotes, plain LLM retains claims."""
    judgments, errors = {}, []
    limit = assessment_config(rules)["evidence"]["max_quotation_passages"] if rules.get("quality") else 0
    if rules.get("quality") and isinstance(raw, dict) and set(raw) != set(rules["quality"]):
        errors.append("quality_keys")
    for key in rules.get("quality", {}):
        item = raw.get(key) if isinstance(raw, dict) else None
        valid = (isinstance(item, dict) and set(item) == {"value", "evidence"} and isinstance(item["value"], str) and item["value"] in {"yes", "no", "unknown"}
            and isinstance(item["evidence"], list) and len(item["evidence"]) <= limit and all(isinstance(e, dict) and set(e) == {"source", "quote"}
                and isinstance(e["source"], str) and isinstance(e["quote"], str) for e in item["evidence"]))
        if not valid:
            judgments[key] = {"value": "unknown", "evidence": [], "error": "invalid_or_missing_quality_response"}
            errors.append(key)
            continue
        item = copy.deepcopy(item)
        item["evidence_verified"] = False
        if grounded and item["value"] != "unknown":
            verified, found = verify_evidence(item["evidence"], sections)
            item.update(model_value=item["value"], evidence=verified, evidence_verified=found)
            if not found:
                item.update(value="unknown", warning="unsupported_quality_evidence")
        judgments[key] = item
    return finish_quality(row, rules, judgments, backend,
        error="invalid_quality_response:" + ",".join(errors) if errors else "",
        evidence_hash=content_hash(format_evidence(sections)))


def quality_consensus(row: dict[str, Any], votes: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Vote quality separately, including failed/missing assessors in every denominator."""
    present = [vote for vote in votes.values() if vote.get("quality_rules_json")]
    if not present:
        return finish_quality(row, {}, {}, "consensus", supported=[])
    policies = {str(vote["quality_rules_json"]) for vote in present}
    if len(policies) != 1:
        raise ValueError("quality consensus requires identical scientific quality policies")
    config_hashes = {vote.get("quality_config_hash", "") for vote in present}
    if len(config_hashes) != 1:
        raise ValueError("quality consensus requires identical shared assessment configurations")
    row["quality_config_hash"] = next(iter(config_hashes))
    policy = json.loads(next(iter(policies)))
    rules = policy_rules(policy)
    hashes = {vote.get("quality_evidence_hash") for vote in present if vote.get("quality_evidence_hash")}
    if len(hashes) > 1:
        return finish_quality(row, rules, {}, "consensus", error="quality_evidence_mismatch")
    judgments, audit = {}, {}
    for key in policy["quality"]:
        items = []
        for vote in votes.values():
            value = json.loads(str(vote.get("quality_assessments_json") or "{}"))
            items.append(value.get(key, {"value": "unknown", "evidence": []}) if not vote.get("quality_error") else {"value": "unknown", "evidence": []})
        counts = Counter(item["value"] for item in items)
        maximum = max(counts.values(), default=0)
        winners = [value for value, count in counts.items() if count == maximum and count / len(votes) >= threshold]
        winner = winners[0] if len(winners) == 1 else "unknown"
        evidence = {content_hash(e): e for item in items if item["value"] == winner for e in item.get("evidence", [])}
        judgments[key] = {"value": winner, "evidence": list(evidence.values())}
        audit[key] = {"counts": dict(counts), "denominator": len(votes), "selected": winner}
    all_error = len(present) == len(votes) and all(vote.get("quality_error") for vote in present)
    finish_quality(row, rules, judgments, "consensus", error="all_quality_assessments_failed" if all_error else "",
        evidence_hash=next(iter(hashes), ""))
    row["quality_votes_json"] = json.dumps(audit, sort_keys=True)
    return row


def assess_nli_quality(row: dict[str, Any], sections: list[EvidenceSection], rules: dict[str, Any],
                       backend: Any, config: Any, method: str) -> dict[str, Any]:
    """Use the selected NLI backend itself; neutrality and lack of entailment never mean failure."""
    from radar.common.nli import _representative_chunks

    criteria = rules.get("quality", {})
    if not criteria:
        return finish_quality(row, rules, {}, method)
    quality_nli = assessment_config(rules)["nli"]
    sections = [section for section in sections if section.text.strip() and section.source not in {"GitHub metadata", "OpenAlex metadata"}]
    if not sections:
        return finish_quality(row, rules, {}, method, evidence_hash=content_hash([]))
    try:
        chunks = _representative_chunks(backend.chunk_sections(sections, config.premise_tokens, config.chunk_overlap), config.max_chunks)
        pairs = [(Template(quality_nli["premise_template"]).substitute(source=chunk.source, text=chunk.text),
            Template(quality_nli["hypothesis_template"]).substitute(criterion=description))
            for description in criteria.values() for chunk in chunks]
        scores = backend.score(pairs, config.batch_size)
        if not chunks or len(scores) != len(pairs):
            raise ValueError("quality NLI returned incomplete scores")
        judgments = {}
        for position, key in enumerate(criteria):
            selected = scores[position * len(chunks):(position + 1) * len(chunks)]
            for score in selected:
                values = [score.entailment, score.neutral, score.contradiction]
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values) or not math.isclose(sum(values), 1, abs_tol=1e-4):
                    raise ValueError("invalid quality NLI probabilities")
            entailment = max(score.entailment for score in selected)
            contradiction = max(score.contradiction for score in selected)
            positive = max(selected, key=lambda score: score.entailment)
            negative = max(selected, key=lambda score: score.contradiction)
            yes = entailment >= config.quality_entailment_threshold and entailment - max(contradiction, positive.neutral) >= config.quality_label_margin
            no = contradiction >= config.quality_entailment_threshold and contradiction - max(entailment, negative.neutral) >= config.quality_label_margin
            value = "yes" if yes and not no else "no" if no and not yes else "unknown"
            judgments[key] = {"value": value, "hypothesis": Template(quality_nli["hypothesis_template"]).substitute(criterion=criteria[key]), "evidence": [
                {"source": chunk.source, "quote": chunk.text, "entailment": score.entailment,
                 "neutral": score.neutral, "contradiction": score.contradiction}
                for chunk, score in zip(chunks, selected, strict=True)],
                "entailment_threshold": config.quality_entailment_threshold, "label_margin": config.quality_label_margin}
        return finish_quality(row, rules, judgments, method, evidence_hash=content_hash([(chunk.source, chunk.text) for chunk in chunks]))
    except Exception as exc:
        return finish_quality(row, rules, {}, method, error=f"quality_nli_error:{type(exc).__name__}")
