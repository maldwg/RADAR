"""Replay evidence-backed systematic-review eligibility and quality assessments."""
from __future__ import annotations

import json
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ids_meta.common.csvio import atomic_write_csv
from ids_meta.common.provenance import file_hash
from ids_meta.common.sources import LocalSourceAdapter, SourceContract, load_source_contract


SCREENING_FIELDS = {"decision", "decision_basis", "criteria_json", "protocol_sha256",
    "source_profile_sha256", "quality_yes", "quality_no", "quality_unknown",
    "failed_filters", "unknown_filters", "assessment_input_sha256", "error", "evidence_normalization"}

EVIDENCE_NORMALIZATION = "whitespace-and-latin-ligatures-v1"


def normalize_evidence_text(text: str) -> str:
    """Compare PDF typography without changing words, punctuation or mathematics."""
    ligatures = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
        "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
    return " ".join(text.translate(ligatures).split())


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


def load_study_protocol(path: Path) -> StudyProtocol:
    """Validate a local, versioned assessment protocol before reading records."""
    digest = file_hash(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != "1.0":
        raise ValueError("study protocol requires schema_version 1.0")
    profile = raw.get("source_profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("study protocol requires source_profile")
    profile_path = path.parent / profile
    profile_digest = file_hash(profile_path)
    contract = load_source_contract(profile_path)
    if SCREENING_FIELDS & set(contract.mapping) or contract.identity in SCREENING_FIELDS:
        raise ValueError("study source profile cannot override screening output fields")
    groups = []
    for name in ("filters", "quality"):
        group = raw.get(name)
        if not isinstance(group, dict) or not group or any(
            not isinstance(key, str) or not key.strip()
            or not isinstance(value, str) or not value.strip()
            for key, value in group.items()
        ):
            raise ValueError(f"study protocol {name} must define non-empty criterion descriptions")
        groups.append(group)
    filters, quality = groups
    if set(filters) & set(quality):
        raise ValueError("filter and quality criterion IDs must be disjoint")
    minimum = raw.get("quality_minimum")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= len(quality):
        raise ValueError("quality_minimum must be an integer between 1 and the number of quality criteria")
    if file_hash(path) != digest or file_hash(profile_path) != profile_digest:
        raise ValueError("study protocol inputs changed while loading")
    return StudyProtocol(path, contract, filters, quality, minimum, digest, profile_digest)


class StudyScreeningAdapter(LocalSourceAdapter):
    """Turn supplied criterion judgments into auditable inclusion decisions.

    This adapter checks quotes and applies rules; it does not infer criteria or
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
            text = normalize_evidence_text(text)
            assessments: dict[str, dict[str, str]] = {}
            for key in [*self.protocol.filters, *self.protocol.quality]:
                assessment = criteria.get(key, {"value": "unknown"})
                if not isinstance(assessment, dict):
                    raise ValueError(f"row {index} {key} must be an assessment object")
                value = assessment.get("value")
                if value not in ("yes", "no", "unknown"):
                    raise ValueError(f"row {index} {key} requires yes, no or unknown")
                quote, source = assessment.get("quote", ""), assessment.get("source", "")
                if not isinstance(quote, str) or not isinstance(source, str):
                    raise ValueError(f"row {index} {key} quote and source must be strings")
                if value != "unknown" and (
                    source != "full_text" or not quote.strip()
                    or normalize_evidence_text(quote) not in text
                ):
                    raise ValueError(f"row {index} {key} requires a quote present in supplied full_text")
                assessments[key] = {"value": value, "quote": quote, "source": source}
            failed = [key for key in self.protocol.filters if assessments[key]["value"] == "no"]
            unknown = [key for key in self.protocol.filters if assessments[key]["value"] == "unknown"]
            quality = Counter(assessments[key]["value"] for key in self.protocol.quality)
            if failed:
                decision, basis = "exclude", "eligibility_failed"
            elif quality["yes"] + quality["unknown"] < self.protocol.quality_minimum:
                decision, basis = "exclude", "quality_threshold_unreachable"
            elif unknown:
                decision, basis = "uncertain", "eligibility_incomplete"
            elif quality["yes"] < self.protocol.quality_minimum:
                decision, basis = "uncertain", "quality_incomplete"
            else:
                decision, basis = "accept", "eligibility_and_quality_passed"
            item = {key: value for key, value in row.items() if key != "criteria"}
            item.update(decision=decision, decision_basis=basis, failed_filters=failed,
                unknown_filters=unknown, quality_yes=quality["yes"], quality_no=quality["no"],
                quality_unknown=quality["unknown"], criteria_json=json.dumps(assessments, sort_keys=True),
                protocol_sha256=self.protocol.sha256,
                source_profile_sha256=self.protocol.source_profile_sha256,
                evidence_normalization=EVIDENCE_NORMALIZATION)
            screened.append(item)
        return super().normalize(screened)


def screen_study(protocol_path: Path, input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Screen local JSON assessments and atomically export portable classifications."""
    if output_path.suffix.casefold() != ".csv":
        raise ValueError("study screening output must be a CSV file")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("study screening input and output must differ")
    protocol = load_study_protocol(protocol_path)
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
