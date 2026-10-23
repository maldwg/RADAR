"""Replay evidence-backed systematic-review eligibility and quality assessments."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from radar.common.quality import QUALITY_COLUMNS, finish_quality
from radar.common.csvio import atomic_write_csv
from radar.common.evidence_blocks import EVIDENCE_BLOCK_VERSION, EvidenceBlockIndex
from radar.common.provenance import file_hash
from radar.common.sources import LocalSourceAdapter, SourceContract, load_source_contract
from radar.common.screening import load_screening_profile
from radar.common.quality_config import load_quality_config, quality_for_study


SCREENING_FIELDS = {"decision", "decision_basis", "criteria_json", "protocol_sha256",
    "source_profile_sha256", "quality_yes", "quality_no", "quality_unknown",
    "failed_filters", "unknown_filters", "assessment_input_sha256", "error", "evidence_block_version", "evidence_passages_json", *QUALITY_COLUMNS}

@dataclass(frozen=True)
class StudyProtocol:
    """Pin an external review's rules separately from its reference labels."""

    path: Path
    contract: SourceContract
    filters: dict[str, str]
    quality: dict[str, str]
    quality_minimum: int
    sha256: str
    source_profile_sha256: str
    quality_config_path: Path
    quality_config: dict[str, Any]
    quality_required: tuple[str, ...] = ()


def load_study_protocol(path: Path, *, quality_path: Path | None = None) -> StudyProtocol:
    """Validate a local, versioned assessment protocol before reading records."""
    profile = load_screening_profile(path)
    if quality_path is None:
        quality_path, quality_config = quality_for_study(path.parent / "study.toml")
    else:
        quality_path, quality_config = quality_path.resolve(), load_quality_config(quality_path)
    if profile.strategy != "criterion_assessment":
        raise ValueError("assessment replay requires criterion_assessment screening")
    digest = profile.sha256
    profile_path = profile.taxonomy_path
    profile_digest = file_hash(profile_path)
    contract = load_source_contract(profile_path)
    if SCREENING_FIELDS & set(contract.mapping) or contract.identity in SCREENING_FIELDS:
        raise ValueError("study source profile cannot override screening output fields")
    filters, quality, minimum = profile.eligibility, profile.quality, profile.quality_minimum
    if file_hash(path) != digest or file_hash(profile_path) != profile_digest:
        raise ValueError("study protocol inputs changed while loading")
    return StudyProtocol(path, contract, filters, quality, minimum, digest, profile_digest, quality_path, quality_config, profile.quality_required)


class StudyScreeningAdapter(LocalSourceAdapter):
    """Turn supplied criterion judgments into auditable inclusion decisions.

    This adapter resolves source block IDs and applies rules; it does not infer criteria or
    validate the semantic correctness of a human/model's assessment.
    """

    def __init__(self, protocol: StudyProtocol):
        """Initialize the adapter or fixture state."""
        super().__init__(protocol.contract)
        self.protocol = protocol

    def normalize(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate evidence and replay the screening rules."""
        screened = []
        expected = set(self.protocol.filters) | set(self.protocol.quality)
        for index, row in enumerate(rows, 1):
            if SCREENING_FIELDS & set(row):
                raise ValueError(f"row {index} supplies reserved screening output fields")
            criteria = row.get("criteria", {})
            if not isinstance(criteria, dict) or set(criteria) - expected:
                raise ValueError(f"row {index} has invalid or unknown criterion IDs")
            text = row.get("full_text", "")
            if not isinstance(text, str):
                raise ValueError(f"row {index} full_text must be a string")
            block_index = EvidenceBlockIndex.from_record(row)
            assessments: dict[str, dict[str, Any]] = {}
            for key in [*self.protocol.filters, *self.protocol.quality]:
                assessment = criteria.get(key, {"value": "unknown", "evidence_ids": []})
                error = block_index.judgment_error(assessment, ["yes", "no", "unknown"])
                limit = self.protocol.quality_config["evidence"]["max_block_passages"]
                if not error and key in self.protocol.quality and limit and len(assessment["evidence_ids"]) > limit:
                    error = "quality evidence_ids exceeds configured max_block_passages"
                if error:
                    raise ValueError(f"row {index} {key}: {error}")
                assessments[key] = {"value": assessment["value"], "evidence_ids": list(assessment["evidence_ids"])}
            failed = [key for key in self.protocol.filters if assessments[key]["value"] == "no"]
            unknown = [key for key in self.protocol.filters if assessments[key]["value"] == "unknown"]
            quality = Counter(assessments[key]["value"] for key in self.protocol.quality)
            mandatory = [assessments[key]["value"] for key in self.protocol.quality_required]
            if failed:
                decision, basis = "exclude", "eligibility_failed"
            elif "no" in mandatory or quality["yes"] + quality["unknown"] < self.protocol.quality_minimum:
                decision, basis = "exclude", "quality_threshold_unreachable"
            elif unknown:
                decision, basis = "uncertain", "eligibility_incomplete"
            elif quality["yes"] < self.protocol.quality_minimum or any(value != "yes" for value in mandatory):
                decision, basis = "uncertain", "quality_incomplete"
            else:
                decision, basis = "accept", "eligibility_and_quality_passed" if self.protocol.quality else "eligibility_passed"
            item = {key: value for key, value in row.items() if key != "criteria"}
            item.update(decision=decision, decision_basis=basis, failed_filters=failed,
                unknown_filters=unknown, quality_yes=quality["yes"], quality_no=quality["no"],
                quality_unknown=quality["unknown"], criteria_json=json.dumps(assessments, sort_keys=True),
                protocol_sha256=self.protocol.sha256,
                source_profile_sha256=self.protocol.source_profile_sha256,
                evidence_block_version=EVIDENCE_BLOCK_VERSION,
                evidence_fingerprint=block_index.fingerprint, evidence_text_sha256=block_index.text_sha256,
                evidence_passages_json=json.dumps(block_index.manifest(list(assessments.values()) +
                    list(json.loads(row.get('field_assessments_json', '{}')).values())), sort_keys=True))
            item["relevance_decision"] = "exclude" if failed else "uncertain" if unknown else "accept"
            rules = {"quality": self.protocol.quality, "assessment": self.protocol.quality_config, "decision": {"quality_minimum": self.protocol.quality_minimum,
                "quality_required": list(self.protocol.quality_required)}}
            judgments = {key: {"value": assessments[key]["value"], "evidence": [
                {"source": "Full text", "quote": block_index.by_id[identifier].text, "block_id": identifier}
                for identifier in assessments[key]["evidence_ids"]], "evidence_verified": bool(assessments[key]["evidence_ids"])}
                for key in self.protocol.quality}
            finish_quality(item, rules, judgments, "radar_llm", evidence_hash=block_index.fingerprint)
            screened.append(item)
        return super().normalize(screened)


def screen_study(protocol_path: Path, input_path: Path, output_path: Path, *, quality_path: Path | None = None) -> list[dict[str, Any]]:
    """Screen local JSON assessments and atomically export portable classifications."""
    if output_path.suffix.casefold() != ".csv":
        raise ValueError("study screening output must be a CSV file")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("study screening input and output must differ")
    protocol = load_study_protocol(protocol_path, quality_path=quality_path)
    digest = file_hash(input_path)
    raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw or any(not isinstance(row, dict) for row in raw):
        raise ValueError("study assessments must be a non-empty JSON array of record objects")
    rows = StudyScreeningAdapter(protocol).normalize(raw)
    if file_hash(input_path) != digest:
        raise ValueError("study assessments changed while screening")
    for row in rows:
        row["assessment_input_sha256"] = digest
        row["classifier_supported_fields"] = "*"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    atomic_write_csv(output_path, rows, columns)
    return rows
