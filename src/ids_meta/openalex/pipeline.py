from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ids_meta.common.cache import PipelineCache
from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.ollama import EvidenceSection, OllamaClient, decision_profile, format_evidence, verify_evidence
from ids_meta.common.prompts import description_block, render_prompt
from ids_meta.common.run import utc_now
from ids_meta.common.scoring import score_evidence, shortlist_candidates
from ids_meta.common.text import normalize_text, slug, stable_hash
from ids_meta.common.logging import logger
from ids_meta.openalex.schema import ARTIFACT_TYPES, CANDIDATE_COLUMNS, CATALOGUE_COLUMNS, CLASSIFICATION_COLUMNS, DATA_FORMS, DOMAINS, ENVIRONMENTS, LABELING, PAPER_ROLES, RESPONSE_SCHEMA, SURVEY_COLUMNS, TIERS, ensure_resource_name, normalize_response, validate_processed_row

PROMPT_VERSION = "openalex-5"
LOG = logger("openalex.pipeline")

class Generator(Protocol):
    model: str
    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one structured model response for the supplied prompt."""
        ...


def evidence_sections(payload: dict[str, Any]) -> list[EvidenceSection]:
    """Build bounded, labelled evidence sections from a cached OpenAlex work."""
    work = payload.get("work") or {}
    metadata = "\n".join(filter(None, (
        f"Title: {work.get('title', '')}", f"Authors: {work.get('authors', '')}" if work.get("authors") else "",
        f"Year: {work.get('publication_year', '')}" if work.get("publication_year") else "",
        f"Venue: {work.get('venue', '')}" if work.get("venue") else "",
    )))
    sections = [EvidenceSection("OpenAlex metadata", metadata)]
    if work.get("abstract"):
        sections.append(EvidenceSection("Abstract", str(work["abstract"])))
    return sections


def metadata_row(key: str, payload: dict[str, Any], fetched_at: str, run_id: str) -> dict[str, object]:
    """Create a candidate CSV row from cached work metadata and provenance."""
    work = payload.get("work") or {}
    sections = evidence_sections(payload)
    return {
        "schema_version": "1.0", "run_id": run_id, "openalex_id": key,
        "doi": work.get("doi", ""), "title": work.get("title", ""), "abstract": work.get("abstract", ""),
        "authors": work.get("authors", ""), "venue": work.get("venue", ""), "publication_year": work.get("publication_year", ""),
        "publication_date": work.get("publication_date", ""), "work_type": work.get("work_type", ""),
        "primary_url": work.get("primary_url", ""), "pdf_url": work.get("pdf_url", ""), "is_oa": work.get("is_oa", "false"),
        "cited_by_count": work.get("cited_by_count", 0), "citation_as_of": fetched_at[:10],
        "referenced_works_count": len(work.get("referenced_works") or []), "metadata_fetched_at": fetched_at,
        "discovery_relations": ";".join(payload.get("discovery_relations", [])), "matched_queries": ";".join(payload.get("matched_queries", [])),
        "evidence_hash": stable_hash(format_evidence(sections, 30000)),
        "retrieval_rank_score": payload.get("retrieval_rank_score", ""), "retrieval_score": payload.get("retrieval_score", ""),
        "retrieval_title_score": payload.get("retrieval_title_score", ""),
        "retrieval_signals": ";".join(payload.get("retrieval_signals", [])), "survey_discovery_score": payload.get("survey_discovery_score", ""),
    }


def prefilter(config: dict[str, Any], profile: dict[str, Any], cache: PipelineCache, output: Path, run_id: str) -> list[dict[str, object]]:
    """Score cached works and export every candidate with its prefilter band."""
    rows: list[dict[str, object]] = []
    counts = {"candidate": 0, "borderline": 0, "reject": 0}
    latest_collection = cache.get_record("openalex.collection", "latest") or {}
    active_ids = {str(value) for value in latest_collection.get("work_ids", []) if value}
    auxiliary_survey_ids = {str(value) for value in latest_collection.get("survey_ids", []) if value}
    active_ids.difference_update(auxiliary_survey_ids)
    LOG.info(
        "starting OpenAlex paper prefilter active_collection=%s active_pool=%s",
        latest_collection.get("selected_at", "legacy_cache_fallback"), len(active_ids) if active_ids else "all_cached_works",
    )
    for key, payload, fetched_at in cache.records("openalex.work"):
        if active_ids and key not in active_ids:
            continue
        result = score_evidence(format_evidence(evidence_sections(payload), 30000), profile, float(config.get("candidate_threshold", 7)), float(config.get("borderline_threshold", 4)))
        row = metadata_row(key, payload, fetched_at, run_id)
        row.update({"prefilter_score": result.score, "prefilter_band": result.band, "prefilter_signals": ";".join(result.signals)})
        rows.append(row)
        counts[result.band] += 1
        LOG.info(
            "prefiltered paper score=%.1f band=%s retrieval_rank_score=%s citations=%s provenance=%s title=%s",
            result.score, result.band, row.get("retrieval_rank_score", ""), row.get("cited_by_count", 0), row.get("discovery_relations", "") or "unknown",
            row.get("title", key),
        )
    atomic_write_csv(output, rows, CANDIDATE_COLUMNS)
    LOG.info("OpenAlex prefilter complete rows=%d candidate=%d borderline=%d reject=%d output=%s", len(rows), counts["candidate"], counts["borderline"], counts["reject"], output)
    return rows


def export_survey_sources(cache: PipelineCache, output: Path, run_id: str) -> list[dict[str, object]]:
    """Export auxiliary survey discovery records without adding them to candidates."""
    latest_collection = cache.get_record("openalex.collection", "latest") or {}
    selected_at = str(latest_collection.get("selected_at", utc_now()))
    rows: list[dict[str, object]] = []
    for survey_id in latest_collection.get("survey_ids", []):
        payload = cache.get_record("openalex.work", str(survey_id))
        if not payload:
            continue
        row = metadata_row(str(survey_id), payload, selected_at, run_id)
        rows.append(row)
        LOG.info(
            "survey audit score=%s citations=%s references=%s title=%s",
            row.get("survey_discovery_score", ""), row.get("cited_by_count", 0),
            row.get("referenced_works_count", 0), row.get("title", survey_id),
        )
    atomic_write_csv(output, rows, SURVEY_COLUMNS)
    LOG.info("exported auxiliary survey sources rows=%d output=%s", len(rows), output)
    return rows


def prompt(sections: list[EvidenceSection], prompt_config: dict[str, Any] | None = None) -> str:
    """Construct the evidence-grounded OpenAlex adjudication prompt."""
    settings = prompt_config or {}
    blocks = {
        "roles": description_block(settings, "roles", PAPER_ROLES),
        "tiers": description_block(settings, "tiers", TIERS),
        "artifact_types": description_block(settings, "artifact_types", ARTIFACT_TYPES),
        "domains": description_block(settings, "domains", DOMAINS),
        "data_forms": description_block(settings, "data_forms", DATA_FORMS),
        "environments": description_block(settings, "environments", ENVIRONMENTS),
        "labeling": description_block(settings, "labeling", LABELING),
    }
    return render_prompt(settings, blocks, format_evidence(sections))


def classify(candidates_path: Path, output: Path, cache: PipelineCache, ollama_config: dict[str, Any], limit: int | None = None, resume: bool = True, dry_run: bool = False, generator: Generator | None = None, prompt_config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Classify eligible papers while checkpointing auditable LLM decisions."""
    rows = read_csv(candidates_path)
    require_columns(rows, ("openalex_id", "prefilter_band", "evidence_hash"), candidates_path)
    eligible = shortlist_candidates(rows, "openalex_id", None, "retrieval_rank_score")
    selected = shortlist_candidates(rows, "openalex_id", limit, "retrieval_rank_score")
    client = generator if generator is not None else (None if dry_run else OllamaClient(ollama_config))
    model = getattr(client, "model", "dry-run")
    profile_hash = decision_profile(model, PROMPT_VERSION, ollama_config, prompt_config)
    results: list[dict[str, str]] = []
    new_decisions = 0
    LOG.info(
        "starting OpenAlex LLM classification eligible=%d shortlist=%d limit=%s ranking=band_then_prefilter_score_then_retrieval_rank resume=%s dry_run=%s model=%s",
        len(eligible), len(selected), limit if limit is not None else "all_eligible", resume, dry_run, model,
    )
    for rank, row in enumerate(selected, 1):
        key, evidence_hash = row["openalex_id"], row["evidence_hash"]
        cached = cache.get_decision("openalex.classification", key, profile_hash, evidence_hash) if resume else None
        if cached:
            results.append({str(k): str(v) for k, v in cached.items()})
            LOG.info(
                "reused cached paper decision rank=%d/%d score=%s retrieval_rank_score=%s citations=%s provenance=%s title=%s decision=%s role=%s",
                rank, len(selected), row.get("prefilter_score", ""), row.get("retrieval_rank_score", ""), row.get("cited_by_count", "0"),
                row.get("discovery_relations", "") or "unknown", row.get("title", key),
                cached.get("decision", ""), cached.get("paper_role", ""),
            )
            continue
        payload = cache.get_record("openalex.work", key)
        if not payload:
            raise ValueError(f"raw OpenAlex work is missing: {key}")
        sections = evidence_sections(payload)
        result = dict(row)
        result.update({"pipeline_version": PROMPT_VERSION, "profile_hash": profile_hash, "llm_model": model, "classified_at": utc_now(), "error": ""})
        LOG.info(
            "classifying paper rank=%d/%d score=%s band=%s retrieval_rank_score=%s citations=%s provenance=%s title=%s",
            rank, len(selected), row.get("prefilter_score", ""), row.get("prefilter_band", ""),
            row.get("retrieval_rank_score", ""), row.get("cited_by_count", "0"), row.get("discovery_relations", "") or "unknown", row.get("title", key),
        )
        if dry_run:
            result.update({"decision": "dry_run", "paper_role": "", "relevance_tier": "", "artifact_id": "", "artifact_name": "", "artifact_display_name": "", "resource_name": "", "artifact_aliases": "", "artifact_type": "", "domain": "", "data_form": "", "capture_environment": "", "labeling": "", "direct_release_evidence": "", "primary_reason": "", "classification_validation_flags": "", "model_evidence_json": "", "evidence_json": "", "evidence_quote_verified": "false"})
        else:
            try:
                raw = client.generate(prompt(sections, prompt_config), RESPONSE_SCHEMA)  # type: ignore[union-attr]
                verified, _ = verify_evidence(raw.get("evidence"), sections)
                artifact_name = str(raw.get("artifact_name", "")).strip()
                artifact_id = f"{slug(artifact_name)}-{stable_hash(normalize_text(artifact_name), length=8)}" if artifact_name else ""
                result.update(normalize_response(raw, verified, ollama_config, artifact_id))
            except Exception as exc:
                result.update({"decision": "error", "paper_role": "", "relevance_tier": "", "artifact_id": "", "artifact_name": "", "artifact_display_name": "", "resource_name": "", "artifact_aliases": "", "artifact_type": "", "domain": "", "data_form": "", "capture_environment": "", "labeling": "", "direct_release_evidence": "", "primary_reason": "", "classification_validation_flags": "classification_error", "model_evidence_json": "", "evidence_json": "", "evidence_quote_verified": "false", "error": str(exc)[:1000]})
        cache.put_decision("openalex.classification", key, profile_hash, evidence_hash, result, result["classified_at"])
        results.append(result)
        new_decisions += 1
        atomic_write_csv(output, results, CLASSIFICATION_COLUMNS)
        LOG.info("classified paper rank=%d/%d %s decision=%s role=%s tier=%s evidence_verified=%s validation_flags=%s", rank, len(selected), row.get("title", key), result.get("decision", ""), result.get("paper_role", ""), result.get("relevance_tier", ""), result.get("evidence_quote_verified", "false"), result.get("classification_validation_flags", "") or "none")
    atomic_write_csv(output, results, CLASSIFICATION_COLUMNS)
    decision_counts: dict[str, int] = {}
    for row in results:
        decision_counts[row.get("decision", "unknown")] = decision_counts.get(row.get("decision", "unknown"), 0) + 1
    LOG.info("OpenAlex classification complete rows=%d new_decisions=%d cached=%d decisions=%s output=%s", len(results), new_decisions, len(results) - new_decisions, decision_counts, output)
    return results


def export(reconciled_path: Path, main_output: Path, adjacent_output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Validate and export core and adjacent accepted-paper catalogues."""
    rows = read_csv(reconciled_path)
    require_columns(rows, ("decision", "paper_role", "openalex_id", "publication_family_id", "canonical_openalex_id", "paper_relationship"), reconciled_path)
    main = [row for row in rows if row.get("decision") == "accept"]
    adjacent = [row for row in rows if row.get("decision") == "accept_adjacent"]
    for row in main + adjacent:
        ensure_resource_name(row)
        validate_processed_row(row)
    key = lambda row: (int(row.get("publication_year") or 0), row.get("artifact_name", "").casefold())
    main.sort(key=key)
    adjacent.sort(key=key)
    atomic_write_csv(main_output, main, CATALOGUE_COLUMNS)
    atomic_write_csv(adjacent_output, adjacent, CATALOGUE_COLUMNS)
    LOG.info("exported OpenAlex catalogues core=%d adjacent=%d source_rows=%d core_output=%s adjacent_output=%s", len(main), len(adjacent), len(rows), main_output, adjacent_output)
    return main, adjacent
