"""Scientific screening profiles shared by catalogue and review workflows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

from radar.common.provenance import file_hash


STRATEGIES = {"repository_role_validation", "dataset_release_validation", "criterion_assessment"}


@dataclass(frozen=True)
class ScreeningProfile:
    """Declare scientific rules separately from execution settings and taxonomy."""

    path: Path
    protocol_id: str
    strategy: str
    taxonomy_path: Path
    prompt_path: Path
    eligibility: dict[str, str]
    quality: dict[str, str]
    quality_minimum: int
    provenance: dict[str, Any]
    sha256: str
    quality_required: tuple[str, ...] = ()

    def prompt_rules(self) -> dict[str, Any]:
        """Return scientific prompt inputs without operational or reference data."""
        return {"eligibility": self.eligibility, "quality": self.quality,
            "decision": {"strategy": self.strategy, "quality_minimum": self.quality_minimum, "quality_required": list(self.quality_required)}}


def _criteria(value: Any, name: str, required: bool = False) -> dict[str, str]:
    """Validate criterion IDs and scientific descriptions without runtime keys."""
    if not isinstance(value, dict) or (required and not value) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(text, str) or not text.strip()
        for key, text in value.items()
    ):
        raise ValueError(f"screening {name} must contain non-empty criterion descriptions")
    return value


def load_screening_profile(path: Path) -> ScreeningProfile:
    """Read one strict scientific profile with optional quality assessment."""
    path = path.resolve()
    digest = file_hash(path)
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != "1.0":
        raise ValueError("screening profile requires schema_version 1.0")
    allowed = {"schema_version", "protocol_id", "taxonomy", "prompt", "eligibility", "quality", "decision", "provenance"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown scientific screening settings: {', '.join(sorted(unknown))}")
    protocol_id = raw.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("screening requires a non-empty protocol_id")
    references = []
    for key in ("taxonomy", "prompt"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"screening requires {key}")
        reference = (path.parent / value).resolve()
        if not reference.is_file():
            raise ValueError(f"screening {key} profile does not exist: {reference}")
        references.append(reference)
    eligibility = _criteria(raw.get("eligibility"), "eligibility", required=True)
    quality = _criteria(raw.get("quality", {}), "quality")
    if set(eligibility) & set(quality):
        raise ValueError("eligibility and quality criterion IDs must be disjoint")
    decision = raw.get("decision")
    if not isinstance(decision, dict) or set(decision) - {"strategy", "quality_minimum", "quality_required"}:
        raise ValueError("screening decision requires a strategy and optional quality_minimum")
    strategy = decision.get("strategy")
    if not isinstance(strategy, str) or strategy not in STRATEGIES:
        raise ValueError(f"unsupported screening strategy: {strategy!r}")
    minimum = decision.get("quality_minimum", 0)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or (
        quality and not 1 <= minimum <= len(quality)
    ) or (not quality and minimum != 0):
        raise ValueError("quality_minimum must be 0 without quality criteria, or between 1 and their count otherwise")
    required = decision.get("quality_required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) or key not in quality for key in required) or len(set(required)) != len(required):
        raise ValueError("quality_required must contain distinct configured quality criterion IDs")
    provenance = raw.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("screening provenance must be a table")
    if file_hash(path) != digest:
        raise ValueError("screening profile changed while loading")
    return ScreeningProfile(path, protocol_id, strategy, *references, eligibility, quality, minimum, provenance, digest, tuple(required))
