from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from ids_meta.common.cache import PipelineCache
from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.ensemble import consensus_classification
from ids_meta.common.evidence import evidence_expansion_audit, merge_new_evidence, vote_round_needs_more_evidence
from ids_meta.common.ollama import EvidenceSection, OllamaClient, decision_profile, format_evidence, verify_evidence
from ids_meta.common.prompts import description_block, render_prompt
from ids_meta.common.run import utc_now
from ids_meta.common.scoring import score_evidence, shortlist_candidates
from ids_meta.common.text import normalize_text, slug, stable_hash
from ids_meta.common.logging import logger
from ids_meta.openalex.enrich import ScholarlyEvidenceClient
from ids_meta.openalex.schema import ARTIFACT_TYPES, CANDIDATE_COLUMNS, CATALOGUE_COLUMNS, CLASSIFICATION_COLUMNS, DATA_FORMS, DOMAINS, ENVIRONMENTS, LABELING, PAPER_ROLES, RESPONSE_SCHEMA, SURVEY_COLUMNS, TIERS, ensure_resource_name, normalize_response, validate_processed_row

PROMPT_VERSION = "openalex-12"
LOG = logger("openalex.pipeline")

class Generator(Protocol):
    model: str
    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one structured model response for the supplied prompt."""
        ...


class EvidenceEnricher(Protocol):
    def enrich(self, work: dict[str, Any]) -> dict[str, str]:
        """Return discovered abstract and optional full-text evidence."""
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
    if work.get("full_text"):
        sections.append(EvidenceSection("Full text", str(work["full_text"])))
    return sections


def metadata_row(key: str, payload: dict[str, Any], fetched_at: str, run_id: str) -> dict[str, object]:
    """Create a candidate CSV row from cached work metadata and provenance."""
    work = payload.get("work") or {}
    sections = evidence_sections(payload)
    return {
        "schema_version": "1.0", "run_id": run_id, "openalex_id": key,
        "doi": work.get("doi", ""), "title": work.get("title", ""), "abstract": work.get("abstract", ""),
        "authors": work.get("authors", ""), "venue": work.get("venue", ""), "publisher": work.get("publisher", ""),
        "publication_year": work.get("publication_year", ""),
        "publication_date": work.get("publication_date", ""), "work_type": work.get("work_type", ""),
        "primary_url": work.get("primary_url", ""), "pdf_url": work.get("pdf_url", ""), "is_oa": work.get("is_oa", "false"),
        "abstract_source": work.get("abstract_source", "OpenAlex" if work.get("abstract") else ""), "full_text_url": work.get("full_text_url", ""),
        "cited_by_count": work.get("cited_by_count", 0), "citation_as_of": fetched_at[:10],
        "referenced_works_count": len(work.get("referenced_works") or []), "metadata_fetched_at": fetched_at,
        "discovery_relations": ";".join(payload.get("discovery_relations", [])), "matched_queries": ";".join(payload.get("matched_queries", [])),
        "evidence_hash": stable_hash(format_evidence(sections, 30000)),
        "retrieval_rank_score": payload.get("retrieval_rank_score", ""), "retrieval_score": payload.get("retrieval_score", ""),
        "retrieval_title_score": payload.get("retrieval_title_score", ""),
        "retrieval_signals": ";".join(payload.get("retrieval_signals", [])), "survey_discovery_score": payload.get("survey_discovery_score", ""),
        "snowball_depth": payload.get("snowball_depth", ""),
    }


def prefilter(
    config: dict[str, Any], profile: dict[str, Any], cache: PipelineCache, output: Path, run_id: str,
    enricher: EvidenceEnricher | None = None,
) -> list[dict[str, object]]:
    """Score cached works and export every candidate with its prefilter band."""
    rows: list[dict[str, object]] = []
    counts = {"candidate": 0, "borderline": 0, "reject": 0}
    minimum_citations = max(0, int(config.get("minimum_citations", 0)))
    latest_collection = cache.get_record("openalex.collection", "latest")
    if latest_collection is None or "work_ids" not in latest_collection or not latest_collection.get("selected_at"):
        raise ValueError("openalex prefilter requires a completed collection with selected IDs and a timestamp")
    active_ids = {str(value) for value in latest_collection["work_ids"]}
    evidence_client = enricher
    if evidence_client is None and bool(config.get("enrich_missing_abstracts", False)):
        evidence_client = ScholarlyEvidenceClient(
            timeout=float(config.get("evidence_lookup_timeout_seconds", 30)),
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
            max_full_text_chars=int(config.get("max_full_text_chars", 30000)),
        )
    LOG.info(
        "starting OpenAlex paper prefilter active_collection=%s active_pool=%s",
        latest_collection["selected_at"], len(active_ids),
    )
    for key, payload, fetched_at in cache.records("openalex.work"):
        if key not in active_ids:
            continue
        citations = int((payload.get("work") or {}).get("cited_by_count") or 0)
        if citations < minimum_citations:
            LOG.debug("skipping paper below citation floor citations=%d minimum=%d title=%s", citations, minimum_citations, (payload.get("work") or {}).get("title", key))
            continue
        work = payload.get("work") or {}
        if evidence_client is not None and not work.get("abstract") and not work.get("full_text"):
            try:
                enriched = evidence_client.enrich(work)
            except Exception as exc:
                LOG.warning("prefilter evidence enrichment failed title=%s error=%s", work.get("title", key), exc)
                enriched = {}
            if enriched:
                work.update(enriched)
                payload["work"] = work
                cache.put_record("openalex.work", key, payload, utc_now())
                LOG.info(
                    "enriched paper before prefilter title=%s abstract=%s full_text=%s",
                    work.get("title", key), bool(work.get("abstract")), bool(work.get("full_text")),
                )
        result = score_evidence(format_evidence(evidence_sections(payload), 30000), profile, float(config.get("candidate_threshold", 7)), float(config.get("borderline_threshold", 4)))
        relations = set(payload.get("discovery_relations", []))
        survey_sources = {value for value in relations if str(value).startswith("survey_reference:")}
        consensus_threshold = max(2, int(config.get("survey_consensus_threshold", 2)))
        band = "borderline" if result.band == "reject" and len(survey_sources) >= consensus_threshold else result.band
        signals = list(result.signals)
        if band != result.band:
            signals.append(f"survey_consensus:{len(survey_sources)}")
        row = metadata_row(key, payload, fetched_at, run_id)
        row.update({"prefilter_score": result.score, "prefilter_band": band, "prefilter_signals": ";".join(signals)})
        rows.append(row)
        counts[band] += 1
        LOG.info(
            "prefiltered paper score=%.1f band=%s retrieval_rank_score=%s citations=%s provenance=%s title=%s",
            result.score, band, row.get("retrieval_rank_score", ""), row.get("cited_by_count", 0), row.get("discovery_relations", "") or "unknown",
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


def classify(candidates_path: Path, output: Path, cache: PipelineCache, ollama_config: dict[str, Any], limit: int | None = None, resume: bool = True, dry_run: bool = False, generator: Generator | None = None, prompt_config: dict[str, Any] | None = None, openalex_config: dict[str, Any] | None = None, enricher: EvidenceEnricher | None = None, consecutive_runs: int = 1, consensus_threshold: float = 0.5) -> list[dict[str, str]]:
    """Classify eligible papers while checkpointing auditable LLM decisions."""
    rows = read_csv(candidates_path)
    require_columns(rows, ("openalex_id", "prefilter_band", "evidence_hash"), candidates_path)
    eligible = shortlist_candidates(rows, "openalex_id", None, "retrieval_rank_score")
    selected = shortlist_candidates(rows, "openalex_id", limit, "retrieval_rank_score")
    client = generator if generator is not None else (None if dry_run else OllamaClient(ollama_config))
    enrichment_config = openalex_config or {}
    evidence_client = enricher
    if evidence_client is None and not dry_run and (
        bool(enrichment_config.get("enrich_missing_abstracts", False))
        or bool(enrichment_config.get("enrich_uncertain_with_full_text", False))
    ):
        evidence_client = ScholarlyEvidenceClient(
            timeout=float(enrichment_config.get("evidence_lookup_timeout_seconds", 30)),
            api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
            max_full_text_chars=int(enrichment_config.get("max_full_text_chars", 30000)),
        )
    model = getattr(client, "model", str(ollama_config.get("model") or "dry-run"))
    consecutive_runs = max(1, int(consecutive_runs))
    decision_config = {
        "prompt": prompt_config or {}, "consecutive_runs": consecutive_runs,
        "consensus_threshold": consensus_threshold,
        "evidence_fallback": {
            "missing_abstract": bool(enrichment_config.get("enrich_missing_abstracts", False)),
            "uncertain_full_text": bool(enrichment_config.get("enrich_uncertain_with_full_text", False)),
            "max_full_text_chars": int(enrichment_config.get("max_full_text_chars", 30000)),
        },
    }
    profile_hash = decision_profile(model, PROMPT_VERSION, ollama_config, decision_config)
    results: list[dict[str, str]] = []
    new_decisions = 0
    LOG.info(
        "starting OpenAlex LLM classification eligible=%d shortlist=%d limit=%s ranking=band_then_prefilter_score_then_retrieval_rank resume=%s dry_run=%s model=%s",
        len(eligible), len(selected), limit if limit is not None else "all_eligible", resume, dry_run, model,
    )
    for rank, row in enumerate(selected, 1):
        key = row["openalex_id"]
        payload = cache.get_record("openalex.work", key)
        if not payload:
            raise ValueError(f"raw OpenAlex work is missing: {key}")
        work = payload.get("work") or {}
        if evidence_client is not None and not work.get("abstract") and not work.get("full_text"):
            try:
                enriched = evidence_client.enrich(work)
            except Exception as exc:  # Enrichment is best-effort and must not block classification.
                LOG.warning("paper evidence enrichment failed title=%s error=%s", row.get("title", key), exc)
                enriched = {}
            if enriched:
                work.update(enriched)
                payload["work"] = work
                cache.put_record("openalex.work", key, payload, utc_now())
                sections = evidence_sections(payload)
                row.update({
                    "abstract": work.get("abstract", ""),
                    "abstract_source": work.get("abstract_source", ""),
                    "pdf_url": work.get("pdf_url", "") or work.get("full_text_url", ""),
                    "full_text_url": work.get("full_text_url", ""),
                    "evidence_hash": stable_hash(format_evidence(sections, 30000)),
                })
                atomic_write_csv(candidates_path, rows, CANDIDATE_COLUMNS)
                LOG.info("enriched missing paper evidence title=%s abstract=%s full_text=%s", row.get("title", key), bool(work.get("abstract")), bool(work.get("full_text")))
        evidence_hash = row["evidence_hash"]
        cached = cache.get_decision(
            "openalex.classification", key, profile_hash, evidence_hash,
        ) if resume else None
        if cached and cached.get("decision") != "error":
            results.append({str(k): str(v) for k, v in cached.items()})
            LOG.info(
                "reused cached paper decision rank=%d/%d score=%s retrieval_rank_score=%s citations=%s provenance=%s title=%s decision=%s role=%s",
                rank, len(selected), row.get("prefilter_score", ""), row.get("retrieval_rank_score", ""), row.get("cited_by_count", "0"),
                row.get("discovery_relations", "") or "unknown", row.get("title", key),
                cached.get("decision", ""), cached.get("paper_role", ""),
            )
            continue
        if cached:
            LOG.info("retrying cached paper classification error rank=%d/%d title=%s", rank, len(selected), row.get("title", key))
        sections = evidence_sections(payload)
        result = dict(row)
        result.update({
            "pipeline_version": PROMPT_VERSION, "profile_hash": profile_hash, "llm_model": model,
            "classifier_method": "llm", "classifier_id": model,
            "classifier_profile_hash": profile_hash, "classifier_supported_fields": "*",
            "classifier_evidence_source": "raw_cache",
            "classified_at": utc_now(), "error": "",
        })
        LOG.info(
            "classifying paper rank=%d/%d score=%s band=%s retrieval_rank_score=%s citations=%s provenance=%s title=%s",
            rank, len(selected), row.get("prefilter_score", ""), row.get("prefilter_band", ""),
            row.get("retrieval_rank_score", ""), row.get("cited_by_count", "0"), row.get("discovery_relations", "") or "unknown", row.get("title", key),
        )
        if dry_run:
            result.update({"decision": "dry_run", "paper_role": "", "relevance_tier": "", "artifact_id": "", "artifact_name": "", "artifact_display_name": "", "resource_name": "", "artifact_aliases": "", "artifact_type": "", "domain": "", "data_form": "", "capture_environment": "", "labeling": "", "direct_release_evidence": "", "primary_reason": "", "classification_validation_flags": "", "model_evidence_json": "", "evidence_json": "", "evidence_quote_verified": "false"})
        else:
            def decide(
                active_sections: list[EvidenceSection], stop_on_uncertain: bool = False,
                initial_votes: dict[str, dict[str, str]] | None = None,
            ) -> tuple[dict[str, str], dict[str, dict[str, str]], bool]:
                """Run and normalize the configured paper-classification votes."""
                votes = dict(initial_votes or {})
                interrupted = False
                for run_index in range(len(votes) + 1, consecutive_runs + 1):
                    vote = dict(result)
                    try:
                        raw = client.generate(prompt(active_sections, prompt_config), RESPONSE_SCHEMA)  # type: ignore[union-attr]
                        verified, _ = verify_evidence(raw.get("evidence"), active_sections)
                        artifact_name = str(raw.get("artifact_name", "")).strip()
                        artifact_id = f"{slug(artifact_name)}-{stable_hash(normalize_text(artifact_name), length=8)}" if artifact_name else ""
                        vote.update(normalize_response(raw, verified, ollama_config, artifact_id))
                    except Exception as exc:
                        vote.update({"decision": "error", "paper_role": "", "relevance_tier": "", "artifact_id": "", "artifact_name": "", "artifact_display_name": "", "resource_name": "", "artifact_aliases": "", "artifact_type": "", "domain": "", "data_form": "", "capture_environment": "", "labeling": "", "direct_release_evidence": "", "primary_reason": "", "classification_validation_flags": "classification_error", "model_evidence_json": "", "evidence_json": "", "evidence_quote_verified": "false", "error": str(exc)[:1000]})
                    votes[f"run-{run_index}"] = {str(k): str(v) for k, v in vote.items()}
                    if stop_on_uncertain and vote.get("decision") == "uncertain":
                        interrupted = True
                        break
                decision = dict(consensus_classification("openalex", votes, consensus_threshold, require_verified_quote=bool(ollama_config.get("require_verified_quote", True)))) if consecutive_runs > 1 else votes["run-1"]
                return decision, votes, interrupted

            result, votes, interrupted = decide(sections, stop_on_uncertain=True)
            initial_result = dict(result)
            initial_votes = {label: dict(vote) for label, vote in votes.items()}
            expansion_required = vote_round_needs_more_evidence(votes, result)
            expansion_attempted = False
            expansion_changed = False
            final_votes = votes
            retry_uncertain = bool(enrichment_config.get("enrich_uncertain_with_full_text", True))
            round_restarted = False
            if retry_uncertain and expansion_required and evidence_client is not None and not work.get("full_text"):
                expansion_attempted = True
                try:
                    discovered = evidence_client.enrich(work)
                except Exception as exc:
                    LOG.warning("uncertain-paper full-text enrichment failed title=%s error=%s", row.get("title", key), exc)
                    discovered = {}
                expansion_changed = merge_new_evidence(work, discovered)
                if expansion_changed:
                    payload["work"] = work
                    cache.put_record("openalex.work", key, payload, utc_now())
                    sections = evidence_sections(payload)
                    evidence_hash = stable_hash(format_evidence(sections, 30000))
                    row.update({
                        "abstract": work.get("abstract", ""), "abstract_source": work.get("abstract_source", ""),
                        "pdf_url": work.get("pdf_url", "") or work.get("full_text_url", ""),
                        "full_text_url": work.get("full_text_url", ""), "evidence_hash": evidence_hash,
                    })
                    atomic_write_csv(candidates_path, rows, CANDIDATE_COLUMNS)
                    LOG.info("discarding initial vote round and restarting with expanded abstract/full-text evidence title=%s", row.get("title", key))
                    result, final_votes, _interrupted = decide(sections)
                    round_restarted = True
            if interrupted and not round_restarted:
                result, votes, _interrupted = decide(sections, initial_votes=votes)
                final_votes = votes
            result.update(evidence_expansion_audit(
                initial_votes, initial_result, final_votes, result,
                required=expansion_required, attempted=expansion_attempted, changed=expansion_changed,
            ))
        if result.get("decision") != "error":
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
