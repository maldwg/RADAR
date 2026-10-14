"""Evidence-preserving votes for repeated and multi-model review assessments."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
from typing import Any

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.common.provenance import file_hash
from ids_meta.common.sources import content_hash
from ids_meta.common.study_screening import StudyProtocol, StudyScreeningAdapter


def validate_consensus(consecutive_runs: int, threshold: float) -> None:
    """Reject invalid voting settings before model or evidence acquisition."""
    if isinstance(consecutive_runs, bool) or not isinstance(consecutive_runs, int) or consecutive_runs < 1:
        raise ValueError("consecutive_runs must be a positive integer")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
        raise ValueError("consensus_threshold must be greater than zero and at most one")


def _winner(counts: Counter, denominator: int, threshold: float, fallback: str) -> str:
    """Require a unique winner measured against every configured voter."""
    maximum = max(counts.values(), default=0)
    winners = [value for value, count in counts.items() if count == maximum and count / denominator >= threshold]
    return winners[0] if len(winners) == 1 else fallback


def combine_assessments(record: dict[str, Any], votes: list[dict[str, Any]], protocol: StudyProtocol,
                        fields: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Vote criterion judgments and fields while keeping actual supporting quotes."""
    validate_consensus(len(votes), threshold)
    result = copy.deepcopy(record)
    result["criteria"] = {}
    selected_fields, audit = {}, {}
    denominator = len(votes)
    for group, specs in (("criteria", {key: ["yes", "no", "unknown"] for key in [*protocol.filters, *protocol.quality]}),
                         ("fields", {key: spec["values"] for key, spec in fields.items()})):
        selected = result["criteria"] if group == "criteria" else selected_fields
        for key, values in specs.items():
            judgments = []
            for vote in votes:
                answer = vote.get("answer", {}) if vote["status"] == "success" else {}
                judgment = answer.get(group, {}).get(key, {"value": "unknown", "quote": "", "source": ""})
                if judgment.get("value") not in values:
                    raise ValueError(f"invalid consensus judgment: {group}.{key}")
                judgments.append(judgment)
            counts = Counter(judgment["value"] for judgment in judgments)
            winner = _winner(counts, denominator, threshold, "unknown")
            selected[key] = (copy.deepcopy(next(j for j in judgments if j["value"] == winner))
                if winner != "unknown" else {"value": "unknown", "quote": "", "source": ""})
            audit[f"{group}.{key}"] = {"selected": winner, "counts": dict(counts),
                "agreement": counts[winner] / denominator, "voters": denominator}
    adapter = StudyScreeningAdapter(protocol)
    decisions = Counter()
    for vote in votes:
        if vote["status"] == "success":
            answer = vote["answer"]
            trial = {**record, "criteria": answer["criteria"],
                **{key: answer["fields"][key]["value"] for key in fields}}
            decisions[adapter.normalize([trial])[0]["decision"]] += 1
        else:
            decisions["error" if vote["status"] == "error" else "uncertain"] += 1
    disposition = _winner(Counter({key: decisions[key] for key in ("accept", "exclude", "uncertain")}),
        denominator, threshold, "uncertain")
    if decisions["error"] == denominator:
        disposition = "error"
    result.update(**{key: judgment["value"] for key, judgment in selected_fields.items()},
        field_assessments_json=json.dumps(selected_fields, sort_keys=True),
        consensus_threshold=threshold, consensus_disposition=disposition,
        consensus_decision_counts=json.dumps(dict(decisions), sort_keys=True),
        consensus_field_audit=json.dumps(audit, sort_keys=True),
        assessment_votes_json=json.dumps(votes, sort_keys=True))
    return result


def normalize_consensus(record: dict[str, Any], protocol: StudyProtocol) -> dict[str, Any]:
    """Replay scientific rules and abstain when disposition and voted rules conflict."""
    row = StudyScreeningAdapter(protocol).normalize([record])[0]
    disposition = record.get("consensus_disposition", row["decision"])
    if disposition == "error":
        row.update(decision="error", error=record.get("assessment_error", "assessment_votes_failed"))
    elif disposition != row["decision"]:
        row.update(decision="uncertain", decision_basis="consensus_disposition_conflict",
            consensus_validation_flags="decision_criteria_conflict")
    return row


def ensemble_study(protocol_path: Path, model_inputs: dict[str, Path], output: Path, threshold: float) -> list[dict[str, Any]]:
    """Combine completed model assessments from one frozen cohort and protocol."""
    from ids_meta.common.study_pipeline import load_assessment_fields, _write_json, _validate_assessment, load_assessment_template
    from ids_meta.common.study_screening import load_study_protocol

    validate_consensus(len(model_inputs), threshold)
    protocol = load_study_protocol(protocol_path)
    fields = load_assessment_fields(protocol)
    _, _, prompt_digest = load_assessment_template(protocol)
    checkpoints = {}
    context = None
    keys = ("input_sha256", "protocol_sha256", "source_profile_sha256", "prompt_sha256",
        "evidence_normalization", "max_full_text_chars", "fetch_full_text", "evidence_lookup_timeout_seconds",
        "dry_run", "consecutive_runs", "consensus_threshold",
        "enrich_missing_abstracts", "enrich_uncertain_with_full_text", "classify_borderline")
    for model, path in model_inputs.items():
        checkpoint = json.loads(path.with_suffix(".assessments.json").read_text())
        if checkpoint.get("completed") is not True or checkpoint.get("output_sha256") != file_hash(path):
            raise ValueError("review ensemble requires completed, unchanged classification outputs")
        if checkpoint.get("model") != model:
            raise ValueError("review ensemble model label does not match its checkpoint")
        if any(key not in checkpoint for key in keys):
            raise ValueError("review ensemble requires complete assessment provenance")
        current = {key: checkpoint[key] for key in keys}
        current["selected_ids"] = [row[protocol.contract.identity] for row in checkpoint["records"]]
        if context is not None and context != current:
            raise ValueError("review ensemble mixes incompatible cohorts or assessment settings")
        if checkpoint["protocol_sha256"] != protocol.sha256 or checkpoint["source_profile_sha256"] != protocol.source_profile_sha256 or checkpoint["prompt_sha256"] != prompt_digest:
            raise ValueError("review ensemble protocol does not match its assessments")
        context = current
        checkpoints[model] = checkpoint
    rows = []
    for index, original in enumerate(next(iter(checkpoints.values()))["records"]):
        record = {key: value for key, value in original.items() if key not in {
            "consensus_disposition", "assessment_error", "assessment_error_detail", "model_response_json", "assessment_attempts_json"}}
        votes = []
        for model, checkpoint in checkpoints.items():
            member = checkpoint["records"][index]
            evidence_keys = ("full_text", "full_text_complete", "full_text_url", "full_text_original_sha256")
            if content_hash({key: member.get(key) for key in evidence_keys}) != content_hash({key: record.get(key) for key in evidence_keys}):
                raise ValueError("review ensemble members assessed different full-text evidence")
            answer = {"criteria": member["criteria"], "fields": json.loads(member["field_assessments_json"])}
            _validate_assessment(answer, protocol, fields, record.get("full_text", ""))
            votes.append({"label": model, "status": "success" if member["assessment_status"] in {"success", "partial_error"}
                and member.get("consensus_disposition") != "error" else member["assessment_status"],
                "answer": answer})
        combined = combine_assessments(record, votes, protocol, fields, threshold)
        # A model's row-level abstention must also remain an abstention in the ensemble.
        model_decisions = Counter(checkpoint["records"][index]["consensus_disposition"] for checkpoint in checkpoints.values())
        combined["consensus_disposition"] = _winner(Counter({key: model_decisions[key] for key in ("accept", "exclude", "uncertain")}),
            len(checkpoints), threshold, "uncertain")
        if model_decisions["error"] == len(checkpoints):
            combined["consensus_disposition"] = "error"
        combined["consensus_decision_counts"] = json.dumps(dict(model_decisions), sort_keys=True)
        combined["assessment_status"] = "ensemble"
        row = normalize_consensus(combined, protocol)
        row.update(classifier_method="ensemble", classifier_id="study_ensemble:" + ",".join(checkpoints),
            classifier_supported_fields="*", ensemble_models=";".join(checkpoints), ensemble_threshold=threshold,
            classifier_profile_hash=content_hash({"context": context, "models": list(checkpoints), "threshold": threshold}),
            candidate_input_sha256=context["input_sha256"])
        rows.append(row)
    columns = list(dict.fromkeys(key for row in rows for key in row)) or [protocol.contract.identity, "decision", "classifier_supported_fields"]
    atomic_write_csv(output, rows, columns)
    _write_json(output.with_suffix(".ensemble.json"), {"context": context, "models": list(checkpoints), "threshold": threshold,
        "inputs": {model: file_hash(path) for model, path in model_inputs.items()}, "output_sha256": file_hash(output)})
    return rows
