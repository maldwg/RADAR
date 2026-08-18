from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from ids_meta.common.cache import PipelineCache
from ids_meta.common.csvio import atomic_write_csv, read_csv, require_columns
from ids_meta.common.ollama import EvidenceSection, OllamaClient, decision_profile, format_evidence, verify_evidence
from ids_meta.common.prompts import description_block, render_prompt
from ids_meta.common.run import utc_now
from ids_meta.common.scoring import score_evidence, shortlist_candidates
from ids_meta.common.text import split_values, stable_hash
from ids_meta.common.logging import logger
from ids_meta.github.schema import AUDIT_ROLES, CAPABILITIES, CANDIDATE_COLUMNS, CATALOGUE_COLUMNS, CLASSIFICATION_COLUMNS, METHODS, RESPONSE_SCHEMA, SCOPES, derive_system_types, normalize_response, system_type_names, validate_processed_row

PROMPT_VERSION = "github-5"
LOG = logger("github.pipeline")

class Generator(Protocol):
    model: str
    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one structured model response for the supplied prompt."""
        ...


def evidence_sections(payload: dict[str, Any]) -> list[EvidenceSection]:
    """Build named repository evidence sections for scoring and classification."""
    repo = payload.get("repository") or {}
    metadata = "\n".join(filter(None, (
        f"Repository: {repo.get('full_name', '')}",
        f"Description: {repo.get('description', '')}" if repo.get("description") else "",
        f"Topics: {'; '.join(repo.get('topics') or [])}" if repo.get("topics") else "",
    )))
    sections = [EvidenceSection("GitHub metadata", metadata)]
    if payload.get("readme_text"):
        sections.append(EvidenceSection("README", str(payload["readme_text"])))
    if payload.get("docs_text"):
        sections.append(EvidenceSection("Documentation", str(payload["docs_text"])))
    return sections


def metadata_row(key: str, payload: dict[str, Any], fetched_at: str, run_id: str) -> dict[str, object]:
    """Flatten cached GitHub evidence into the candidate CSV metadata schema."""
    repo = payload.get("repository") or {}
    owner = repo.get("owner") or {}
    license_info = repo.get("license") or {}
    sections = evidence_sections(payload)
    evidence_hash = stable_hash(format_evidence(sections, 30000))
    return {
        "schema_version": "1.0", "run_id": run_id, "github_repo_id": key,
        "node_id": repo.get("node_id", ""), "full_name": repo.get("full_name", ""),
        "owner": owner.get("login", ""), "name": repo.get("name", ""), "html_url": repo.get("html_url", ""),
        "description": repo.get("description", ""), "topics": ";".join(repo.get("topics") or []),
        "homepage": repo.get("homepage", ""), "license_spdx": license_info.get("spdx_id", ""),
        "language": repo.get("language", ""), "fork": str(bool(repo.get("fork"))).lower(),
        "parent_full_name": payload.get("parent_full_name", ""), "archived": str(bool(repo.get("archived"))).lower(),
        "stars": int(repo.get("stargazers_count") or 0), "forks": int(repo.get("forks_count") or 0),
        "open_issues": int(repo.get("open_issues_count") or 0), "created_at": repo.get("created_at", ""),
        "updated_at": repo.get("updated_at", ""), "pushed_at": repo.get("pushed_at", ""),
        "default_branch": repo.get("default_branch", ""), "last_commit_sha": payload.get("last_commit_sha", ""),
        "last_commit_at": payload.get("last_commit_at", ""), "last_commit_url": payload.get("last_commit_url", ""),
        "metadata_fetched_at": fetched_at, "evidence_hash": evidence_hash,
    }


def prefilter(config: dict[str, Any], profile: dict[str, Any], cache: PipelineCache, output: Path, run_id: str) -> list[dict[str, object]]:
    """Score every cached repository and export all prefilter bands."""
    rows: list[dict[str, object]] = []
    counts = {"candidate": 0, "borderline": 0, "reject": 0}
    latest_collection = cache.get_record("github.collection", "latest") or {}
    active_ids = {str(value) for value in latest_collection.get("repository_ids", []) if value}
    LOG.info("starting GitHub keyword prefilter active_collection=%s active_pool=%s", latest_collection.get("selected_at", "legacy_cache_fallback"), len(active_ids) if active_ids else "all_cached_repositories")
    for key, payload, fetched_at in cache.records("github.repository"):
        if active_ids and key not in active_ids:
            continue
        sections = evidence_sections(payload)
        result = score_evidence(
            format_evidence(sections, 30000), profile,
            float(config.get("candidate_threshold", 7)), float(config.get("borderline_threshold", 4)),
        )
        row = metadata_row(key, payload, fetched_at, run_id)
        row.update({"prefilter_score": result.score, "prefilter_band": result.band, "prefilter_signals": ";".join(result.signals), "matched_queries": ";".join(payload.get("matched_queries", []))})
        rows.append(row)
        counts[result.band] += 1
        LOG.debug("prefiltered %s band=%s score=%.3f", row.get("full_name", key), result.band, result.score)
    atomic_write_csv(output, rows, CANDIDATE_COLUMNS)
    LOG.info("GitHub prefilter complete rows=%d candidate=%d borderline=%d reject=%d output=%s", len(rows), counts["candidate"], counts["borderline"], counts["reject"], output)
    return rows


def prompt(sections: list[EvidenceSection], prompt_config: dict[str, Any] | None = None) -> str:
    """Construct the repository-classification prompt from untrusted evidence."""
    settings = prompt_config or {}
    blocks = {
        "roles": description_block(settings, "roles", AUDIT_ROLES),
        "scopes": description_block(settings, "scopes", SCOPES),
        "capabilities": description_block(settings, "capabilities", CAPABILITIES),
        "methods": description_block(settings, "methods", METHODS),
    }
    return render_prompt(settings, blocks, format_evidence(sections))


def classify(
    candidates_path: Path, output: Path, cache: PipelineCache, ollama_config: dict[str, Any],
    limit: int | None = None, resume: bool = True, dry_run: bool = False, generator: Generator | None = None,
    prompt_config: dict[str, Any] | None = None, system_type_profile: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Classify eligible repository candidates with resumable structured LLM calls."""
    rows = read_csv(candidates_path)
    require_columns(rows, ("github_repo_id", "prefilter_band", "evidence_hash"), candidates_path)
    eligible = shortlist_candidates(rows, "github_repo_id", None)
    selected = shortlist_candidates(rows, "github_repo_id", limit)
    client = generator if generator is not None else (None if dry_run else OllamaClient(ollama_config))
    model = getattr(client, "model", str(ollama_config.get("model") or "dry-run"))
    decision_config = {"prompt": prompt_config or {}, "system_types": system_type_profile or {}}
    profile_hash = decision_profile(model, PROMPT_VERSION, ollama_config, decision_config)
    results: list[dict[str, str]] = []
    new_decisions = 0
    LOG.info(
        "starting GitHub LLM classification eligible=%d shortlist=%d limit=%s ranking=band_then_score resume=%s dry_run=%s model=%s",
        len(eligible), len(selected), limit if limit is not None else "all_eligible", resume, dry_run, model,
    )
    for rank, row in enumerate(selected, 1):
        key, evidence_hash = row["github_repo_id"], row["evidence_hash"]
        cached = cache.get_decision("github.classification", key, profile_hash, evidence_hash) if resume else None
        if cached:
            results.append({str(k): str(v) for k, v in cached.items()})
            LOG.info("reused cached decision rank=%d/%d %s decision=%s role=%s", rank, len(selected), row.get("full_name", key), cached.get("decision", ""), cached.get("role", ""))
            continue
        payload = cache.get_record("github.repository", key)
        if not payload:
            raise ValueError(f"raw GitHub evidence is missing for repository id {key}")
        sections = evidence_sections(payload)
        result = dict(row)
        result.update({"pipeline_version": PROMPT_VERSION, "profile_hash": profile_hash, "llm_model": model, "classified_at": utc_now(), "error": ""})
        LOG.info("classifying rank=%d/%d score=%s band=%s %s", rank, len(selected), row.get("prefilter_score", ""), row.get("prefilter_band", ""), row.get("full_name", key))
        if dry_run:
            result.update({"decision": "dry_run", "role": "", "observation_scopes": "", "capabilities": "", "detection_methods": "", "primary_system_type": "", "system_types": "", "system_type_names": "", "derived_labels": "", "independent_fork": "", "direct_self_description": "", "runtime_observation": "", "runnable_detector": "", "offline_dataset_only": "", "upstream_full_name": "", "classification_validation_flags": "", "primary_reason": "", "evidence_json": "", "evidence_quote_verified": "false"})
        else:
            try:
                raw = client.generate(prompt(sections, prompt_config), RESPONSE_SCHEMA)  # type: ignore[union-attr]
                verified, _ = verify_evidence(raw.get("evidence"), sections)
                result.update(normalize_response(raw, verified, ollama_config, row.get("fork") == "true", system_type_profile))
            except Exception as exc:
                result.update({"decision": "error", "role": "", "observation_scopes": "", "capabilities": "", "detection_methods": "", "primary_system_type": "", "system_types": "", "system_type_names": "", "derived_labels": "", "independent_fork": "", "direct_self_description": "", "runtime_observation": "", "runnable_detector": "", "offline_dataset_only": "", "upstream_full_name": "", "classification_validation_flags": "", "primary_reason": "", "evidence_json": "", "evidence_quote_verified": "false", "error": str(exc)[:1000]})
        cache.put_decision("github.classification", key, profile_hash, evidence_hash, result, result["classified_at"])
        results.append(result)
        new_decisions += 1
        atomic_write_csv(output, results, CLASSIFICATION_COLUMNS)
        LOG.info("classified rank=%d/%d %s decision=%s role=%s system_type=%s evidence_verified=%s", rank, len(selected), row.get("full_name", key), result.get("decision", ""), result.get("role", ""), result.get("primary_system_type", ""), result.get("evidence_quote_verified", "false"))
    atomic_write_csv(output, results, CLASSIFICATION_COLUMNS)
    decision_counts: dict[str, int] = {}
    for row in results:
        decision_counts[row.get("decision", "unknown")] = decision_counts.get(row.get("decision", "unknown"), 0) + 1
    LOG.info("GitHub classification complete rows=%d new_decisions=%d cached=%d decisions=%s output=%s", len(results), new_decisions, len(results) - new_decisions, decision_counts, output)
    return results


def export(
    reconciled_path: Path, output: Path, system_type_profile: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Validate and export accepted reconciled repositories as the final catalogue."""
    rows = read_csv(reconciled_path)
    require_columns(rows, ("decision", "role", "github_repo_id", "project_id", "canonical_repo_id", "repository_relationship"), reconciled_path)
    accepted = [row for row in rows if row.get("decision") == "accept"]
    for row in accepted:
        codes = derive_system_types(
            split_values(row.get("observation_scopes")), split_values(row.get("capabilities")),
            split_values(row.get("detection_methods")), row.get("role", ""), system_type_profile,
        )
        row.update({
            "primary_system_type": codes[0] if codes else "unknown",
            "system_types": ";".join(codes) if codes else "unknown",
            "system_type_names": ";".join(system_type_names(codes, system_type_profile)) if codes else "Unknown",
            "derived_labels": ";".join(codes),
        })
        validate_processed_row(row, system_type_profile)
    accepted.sort(key=lambda row: (row.get("role", ""), row.get("full_name", "").casefold()))
    atomic_write_csv(output, accepted, CATALOGUE_COLUMNS)
    LOG.info("exported GitHub catalogue accepted=%d source_rows=%d output=%s", len(accepted), len(rows), output)
    return accepted
