from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

from ids_meta.common.cache import PipelineCache
from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.logging import logger
from ids_meta.common.ollama import OllamaClient, decision_profile
from ids_meta.common.prompts import render_prompt
from ids_meta.common.reconcile import group_by_links, margin_confidence
from ids_meta.common.run import utc_now
from ids_meta.common.text import normalize_text, stable_hash
from ids_meta.openalex.entities import canonicalize_artifact_ids
from ids_meta.openalex.schema import CATALOGUE_COLUMNS, ensure_resource_name

LOG = logger("openalex.reconcile")
PROMPT_VERSION = "openalex-reconcile-4"

ROLE_WEIGHT = {"original_release": 6, "generator_release": 6, "versioned_release": 2}
TITLE_RELEASE_PHRASES = (
    "introducing", "introduces", "generation of a new", "generating a new",
    "new intrusion detection dataset", "new network dataset", "dataset release", "corpus release",
)
RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "canonical_openalex_id": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["canonical_openalex_id", "reason"],
}
class Generator(Protocol):
    model: str

    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one structured model response for the supplied prompt."""
        ...


def _year(row: dict[str, str]) -> int:
    """Parse a publication year, sorting missing or invalid years last."""
    try:
        return int(row.get("publication_year") or 9999)
    except ValueError:
        return 9999


def _title_similarity(left: dict[str, str], right: dict[str, str]) -> float:
    """Measure normalized title similarity between two work records."""
    return SequenceMatcher(None, normalize_text(left.get("title", "")), normalize_text(right.get("title", ""))).ratio()


def _same_manifestation(left: dict[str, str], right: dict[str, str]) -> bool:
    """Return whether two records represent the same published manifestation."""
    def doi(row: dict[str, str]) -> str:
        """Normalize a DOI for exact comparison."""
        return row.get("doi", "").casefold().strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")

    left_doi, right_doi = doi(left), doi(right)
    left_title, right_title = normalize_text(left.get("title", "")), normalize_text(right.get("title", ""))
    return bool(left_doi and left_doi == right_doi) or bool(left_title and right_title and _title_similarity(left, right) >= .92)


def _manifestation_quality(row: dict[str, str]) -> int:
    """Prefer a stable published record over a metadata-thin preprint manifestation."""
    score = 0
    if row.get("doi"):
        score += 3
    if row.get("venue"):
        score += 2
    if row.get("publication_date"):
        score += 1
    work_type = normalize_text(row.get("work_type", ""))
    if work_type in {"article", "proceedings article", "book chapter"}:
        score += 1
    if "preprint" in work_type:
        score -= 1
    return score


def _title_release_score(row: dict[str, str]) -> int:
    """Reward titles that explicitly identify the paper as a resource release."""
    title = normalize_text(row.get("title", ""))
    return 3 if any(phrase in title for phrase in TITLE_RELEASE_PHRASES) else 0


def _release_score(row: dict[str, str], earliest_year: int) -> float:
    """Score structural evidence that a paper is an artifact's base release."""
    score = ROLE_WEIGHT.get(row.get("paper_role", ""), 0) * 10.0
    if row.get("direct_release_evidence") == "true":
        score += 4
    if row.get("relevance_tier") == "core":
        score += 1
    score += _title_release_score(row)
    # Earlier distinct releases are more likely to be the base. Citation counts are deliberately absent.
    year = _year(row)
    if year != 9999:
        score += max(0, min(8, year - earliest_year) * -1 + 8)
    return score


def _relationship(row: dict[str, str], canonical: dict[str, str]) -> str:
    """Label a paper's relationship to its selected canonical base paper."""
    if row.get("openalex_id") == canonical.get("openalex_id"):
        return "canonical_base"
    if _same_manifestation(row, canonical):
        return "alternate_manifestation"
    if row.get("paper_role") == "versioned_release":
        return "versioned_release"
    return "related_release"


def _prompt(group: list[dict[str, str]], scores: dict[str, float], prompt_config: dict[str, Any] | None = None) -> str:
    """Construct a group-level prompt for ambiguous base-paper selection."""
    items = []
    for row in group:
        items.append("\n".join((
            f"openalex_id: {row.get('openalex_id', '')}",
            f"title: {row.get('title', '')}",
            f"year: {row.get('publication_year', '')}",
            f"doi: {row.get('doi', '')}",
            f"paper_role: {row.get('paper_role', '')}",
            f"artifact_name: {row.get('artifact_name', '')}",
            f"artifact_display_name: {row.get('artifact_display_name', '')}",
            f"reason: {row.get('primary_reason', '')[:500]}",
            f"verified_release_evidence: {row.get('evidence_json', '')[:1000]}",
            f"deterministic_score: {scores.get(row.get('openalex_id', ''), 0):.1f}",
        )))
    return render_prompt(prompt_config, {}, "\n\n---\n\n".join(items), "reconciliation_template")


def reconcile(
    classifications_path: Path,
    output: Path,
    cache: PipelineCache | None,
    ollama_config: dict[str, Any],
    reconciliation_config: dict[str, Any],
    dry_run: bool = False,
    generator: Generator | None = None,
    prompt_config: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Group related releases and select one canonical base paper per family."""
    rows = read_csv(classifications_path)
    require_columns(rows, ("openalex_id", "decision", "paper_role", "artifact_id", "artifact_name"), classifications_path)
    accepted = [dict(row) for row in rows if row.get("decision") in {"accept", "accept_adjacent"}]
    canonicalize_artifact_ids(accepted)
    groups: dict[str, list[dict[str, str]]] = {}
    for row in accepted:
        groups.setdefault(row.get("artifact_id", ""), []).append(row)
    LOG.info("reconciling OpenAlex catalogue accepted=%d artifact_families=%d", len(accepted), len(groups))

    model_client = generator
    profile_hash = decision_profile(generator.model, PROMPT_VERSION, ollama_config, {"reconciliation_template": (prompt_config or {}).get("reconciliation_template", "")}) if generator else ""
    ambiguity_margin = float(reconciliation_config.get("canonical_ambiguity_margin", 2.0))
    use_llm = bool(reconciliation_config.get("reconcile_with_llm", True)) and not dry_run
    reconciled_fields: dict[str, dict[str, str]] = {}

    for group_index, (artifact_id, group) in enumerate(sorted(groups.items()), 1):
        group = sorted(group, key=lambda row: row.get("openalex_id", ""))
        earliest_year = min((_year(row) for row in group), default=9999)
        manifestation_groups = group_by_links(group, _same_manifestation)
        representatives = [
            sorted(cluster, key=lambda row: (-_manifestation_quality(row), row.get("openalex_id", "")))[0]
            for cluster in manifestation_groups
        ]
        scores = {row["openalex_id"]: _release_score(row, earliest_year) for row in group}
        ordered = sorted(
            representatives,
            key=lambda row: (-scores[row["openalex_id"]], -_manifestation_quality(row), row.get("openalex_id", "")),
        )
        canonical = ordered[0]
        margin = scores[canonical["openalex_id"]] - (scores[ordered[1]["openalex_id"]] if len(ordered) > 1 else 999)
        resolution = "deterministic"
        reason = f"highest structural release score {scores[canonical['openalex_id']]:.1f}; margin {margin:.1f}"
        adjudicated = False
        if len(manifestation_groups) > 1 and margin < ambiguity_margin and use_llm:
            group_key = stable_hash(*sorted(row["openalex_id"] for row in group))
            evidence_hash = stable_hash(json.dumps(group, sort_keys=True))
            try:
                if model_client is None:
                    model_client = OllamaClient(ollama_config)
                    profile_hash = decision_profile(model_client.model, PROMPT_VERSION, ollama_config, {"reconciliation_template": (prompt_config or {}).get("reconciliation_template", "")})
                cached = cache.get_decision("openalex.reconciliation", group_key, profile_hash, evidence_hash) if cache else None
                raw = cached or model_client.generate(_prompt(group, scores, prompt_config), RESPONSE_SCHEMA)
                selected = str(raw.get("canonical_openalex_id", ""))
                choices = {row["openalex_id"]: row for row in group}
                if selected in choices:
                    canonical = choices[selected]
                    reason = str(raw.get("reason", "group LLM adjudication"))[:1000]
                    resolution = "llm_group_adjudication"
                    adjudicated = True
                    if cache and not cached:
                        cache.put_decision("openalex.reconciliation", group_key, profile_hash, evidence_hash, raw, utc_now())
            except Exception as exc:
                resolution = "ambiguous_deterministic_fallback"
                reason += f"; group adjudication failed: {str(exc)[:240]}"
                LOG.warning("OpenAlex group adjudication failed group=%s: %s", group_key, exc)
        elif len(manifestation_groups) > 1 and margin < ambiguity_margin:
            resolution = "ambiguous_deterministic_fallback"

        confidence = margin_confidence(margin, len(group), adjudicated)
        family_id = f"openalex-family-{stable_hash(artifact_id, canonical['openalex_id'], length=12)}"
        LOG.info(
            "reconciled artifact family %d/%d members=%d canonical=%s resolution=%s",
            group_index, len(groups), len(group), canonical.get("title", canonical["openalex_id"]), resolution,
        )
        for row in group:
            reconciled_fields[row["openalex_id"]] = {
                "artifact_id": artifact_id,
                "publication_family_id": family_id,
                "canonical_openalex_id": canonical["openalex_id"],
                "canonical_doi": canonical.get("doi", ""),
                "canonical_title": canonical.get("title", ""),
                "paper_relationship": _relationship(row, canonical),
                "canonical_confidence": f"{confidence:.3f}",
                "canonical_reason": reason,
                "canonical_resolution": resolution,
            }

    output_rows: list[dict[str, str]] = []
    for row in rows:
        value = dict(row)
        value.update(reconciled_fields.get(row.get("openalex_id", ""), {}))
        if value.get("artifact_name"):
            ensure_resource_name(value)
        output_rows.append(value)
    atomic_write_csv(output, output_rows, CATALOGUE_COLUMNS)
    LOG.info("OpenAlex reconciliation complete rows=%d output=%s", len(output_rows), output)
    return output_rows
