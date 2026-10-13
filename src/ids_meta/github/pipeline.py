from __future__ import annotations

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
from ids_meta.common.text import split_values, stable_hash
from ids_meta.common.logging import logger
from ids_meta.github.schema import AUDIT_ROLES, CAPABILITIES, CENTRALIZATION, CANDIDATE_COLUMNS, CATALOGUE_COLUMNS, CLASSIFICATION_COLUMNS, METHODS, OBSERVATION_OBJECTS, SCOPES, normalize_response, response_schema, system_type_names, validate_processed_row, validate_system_type_profile
from ids_meta.github.enrich import GitHubEvidenceClient

PROMPT_VERSION = "github-15"
LOG = logger("github.pipeline")

class Generator(Protocol):
    model: str
    def generate(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return one structured model response for the supplied prompt."""
        ...


class EvidenceEnricher(Protocol):
    def enrich(self, payload: dict[str, Any]) -> dict[str, str]:
        """Return additional repository evidence."""
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
    latest_collection = cache.get_record("github.collection", "latest")
    if latest_collection is None or "repository_ids" not in latest_collection or not latest_collection.get("selected_at"):
        raise ValueError("github prefilter requires a completed collection with selected IDs and a timestamp")
    active_ids = {str(value) for value in latest_collection["repository_ids"]}
    LOG.info("starting GitHub keyword prefilter active_collection=%s active_pool=%s", latest_collection["selected_at"], len(active_ids))
    for key, payload, fetched_at in cache.records("github.repository"):
        if key not in active_ids:
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


def prompt(
    sections: list[EvidenceSection], prompt_config: dict[str, Any] | None = None,
    system_type_profile: dict[str, Any] | None = None,
) -> str:
    """Construct the repository-classification prompt from untrusted evidence."""
    settings = prompt_config or {}
    if system_type_profile is None:
        raise ValueError("GitHub classification requires a configured system-type profile")
    types = system_type_profile
    validate_system_type_profile(types)
    type_lines = [
        f"- {code} ({definition['name']}): {definition['description']}"
        for code, definition in types["types"].items()
    ]
    type_lines.append("- unknown: The evidence supports no configured security system type.")
    blocks = {
        "roles": description_block(settings, "roles", AUDIT_ROLES),
        "scopes": description_block(settings, "scopes", SCOPES),
        "observation_objects": description_block(settings, "observation_objects", OBSERVATION_OBJECTS),
        "capabilities": description_block(settings, "capabilities", CAPABILITIES),
        "methods": description_block(settings, "methods", METHODS),
        "centralization": description_block(settings, "centralization", CENTRALIZATION),
        "system_types": "\n".join(type_lines),
    }
    return render_prompt(settings, blocks, format_evidence(sections))


def classify(
    candidates_path: Path, output: Path, cache: PipelineCache, ollama_config: dict[str, Any],
    limit: int | None = None, resume: bool = True, dry_run: bool = False, generator: Generator | None = None,
    prompt_config: dict[str, Any] | None = None, system_type_profile: dict[str, Any] | None = None,
    consecutive_runs: int = 1, consensus_threshold: float = 0.5,
    github_config: dict[str, Any] | None = None, enricher: EvidenceEnricher | None = None,
) -> list[dict[str, str]]:
    """Classify eligible repository candidates with resumable structured LLM calls."""
    rows = read_csv(candidates_path)
    require_columns(rows, ("github_repo_id", "prefilter_band", "evidence_hash"), candidates_path)
    eligible = shortlist_candidates(rows, "github_repo_id", None)
    selected = shortlist_candidates(rows, "github_repo_id", limit)
    client = generator if generator is not None else (None if dry_run else OllamaClient(ollama_config))
    enrichment_config = github_config or {}
    evidence_client = enricher
    if evidence_client is None and not dry_run and bool(enrichment_config.get("enrich_uncertain_evidence", False)):
        evidence_client = GitHubEvidenceClient(enrichment_config)
    model = getattr(client, "model", str(ollama_config.get("model") or "dry-run"))
    consecutive_runs = max(1, int(consecutive_runs))
    decision_config = {"prompt": prompt_config or {}, "system_types": system_type_profile or {},
                       "consecutive_runs": consecutive_runs, "consensus_threshold": consensus_threshold,
                       "evidence_fallback": {
                           "enabled": bool(enrichment_config.get("enrich_uncertain_evidence", False)),
                           "max_docs_files": int(enrichment_config.get("uncertain_max_docs_files", 10)),
                       }}
    profile_hash = decision_profile(model, PROMPT_VERSION, ollama_config, decision_config)
    results: list[dict[str, str]] = []
    new_decisions = 0
    LOG.info(
        "starting GitHub LLM classification eligible=%d shortlist=%d limit=%s ranking=band_then_score resume=%s dry_run=%s model=%s",
        len(eligible), len(selected), limit if limit is not None else "all_eligible", resume, dry_run, model,
    )
    for rank, row in enumerate(selected, 1):
        key, evidence_hash = row["github_repo_id"], row["evidence_hash"]
        cached = cache.get_decision(
            "github.classification", key, profile_hash, evidence_hash,
        ) if resume else None
        if cached:
            results.append({str(k): str(v) for k, v in cached.items()})
            LOG.info("reused cached decision rank=%d/%d %s decision=%s role=%s", rank, len(selected), row.get("full_name", key), cached.get("decision", ""), cached.get("role", ""))
            continue
        payload = cache.get_record("github.repository", key)
        if not payload:
            raise ValueError(f"raw GitHub evidence is missing for repository id {key}")
        sections = evidence_sections(payload)
        result = dict(row)
        result.update({
            "pipeline_version": PROMPT_VERSION, "profile_hash": profile_hash, "llm_model": model,
            "classifier_method": "llm", "classifier_id": model,
            "classifier_profile_hash": profile_hash, "classifier_supported_fields": "*",
            "classifier_evidence_source": "raw_cache",
            "classified_at": utc_now(), "error": "",
            "centralization": "unknown",
        })
        LOG.info("classifying rank=%d/%d score=%s band=%s %s", rank, len(selected), row.get("prefilter_score", ""), row.get("prefilter_band", ""), row.get("full_name", key))
        if dry_run:
            result.update({"decision": "dry_run", "role": "", "observation_scopes": "", "observation_objects": "", "capabilities": "", "detection_methods": "", "primary_system_type": "", "system_types": "", "system_type_names": "", "independent_fork": "", "direct_self_description": "", "runtime_observation": "", "reproducible_execution": "", "offline_dataset_only": "", "implements_detection_engine": "", "requires_external_detector": "", "integrates_multiple_detectors": "", "cohesive_framework_architecture": "", "research_basis": "", "upstream_full_name": "", "classification_validation_flags": "", "primary_reason": "", "evidence_json": "", "evidence_quote_verified": "false"})
        else:
            def decide(
                active_sections: list[EvidenceSection], stop_on_uncertain: bool = False,
                initial_votes: dict[str, dict[str, str]] | None = None,
            ) -> tuple[dict[str, str], dict[str, dict[str, str]], bool]:
                """Run and normalize the configured repository-classification votes."""
                votes = dict(initial_votes or {})
                interrupted = False
                for run_index in range(len(votes) + 1, consecutive_runs + 1):
                    vote = dict(result)
                    try:
                        raw = client.generate(prompt(active_sections, prompt_config, system_type_profile), response_schema(system_type_profile))  # type: ignore[union-attr]
                        verified, _ = verify_evidence(raw.get("evidence"), active_sections)
                        vote.update(normalize_response(raw, verified, ollama_config, row.get("fork") == "true", system_type_profile))
                    except Exception as exc:
                        vote.update({"decision": "error", "role": "", "observation_scopes": "", "observation_objects": "", "capabilities": "", "detection_methods": "", "primary_system_type": "", "system_types": "", "system_type_names": "", "independent_fork": "", "direct_self_description": "", "runtime_observation": "", "reproducible_execution": "", "offline_dataset_only": "", "implements_detection_engine": "", "requires_external_detector": "", "integrates_multiple_detectors": "", "cohesive_framework_architecture": "", "research_basis": "", "upstream_full_name": "", "classification_validation_flags": "", "primary_reason": "", "evidence_json": "", "evidence_quote_verified": "false", "error": str(exc)[:1000]})
                    votes[f"run-{run_index}"] = {str(k): str(v) for k, v in vote.items()}
                    if stop_on_uncertain and vote.get("decision") == "uncertain":
                        interrupted = True
                        break
                decision = dict(consensus_classification("github", votes, consensus_threshold, system_type_profile, bool(ollama_config.get("require_verified_quote", True)))) if consecutive_runs > 1 else votes["run-1"]
                return decision, votes, interrupted

            result, votes, interrupted = decide(sections, stop_on_uncertain=True)
            initial_result = dict(result)
            initial_votes = {label: dict(vote) for label, vote in votes.items()}
            expansion_required = vote_round_needs_more_evidence(votes, result)
            expansion_attempted = False
            expansion_changed = False
            final_votes = votes
            round_restarted = False
            if expansion_required and evidence_client is not None:
                expansion_attempted = True
                try:
                    discovered = evidence_client.enrich(payload)
                except Exception as exc:
                    LOG.warning("uncertain-repository evidence expansion failed repository=%s error=%s", row.get("full_name", key), exc)
                    discovered = {}
                expansion_changed = merge_new_evidence(payload, discovered)
                if expansion_changed:
                    cache.put_record("github.repository", key, payload, utc_now())
                    sections = evidence_sections(payload)
                    evidence_hash = stable_hash(format_evidence(sections, 30000))
                    row["evidence_hash"] = evidence_hash
                    atomic_write_csv(candidates_path, rows, CANDIDATE_COLUMNS)
                    LOG.info("discarding initial vote round and restarting with expanded README/docs evidence repository=%s", row.get("full_name", key))
                    result, final_votes, _interrupted = decide(sections)
                    round_restarted = True
            if interrupted and not round_restarted:
                result, votes, _interrupted = decide(sections, initial_votes=votes)
                final_votes = votes
            result.update(evidence_expansion_audit(
                initial_votes, initial_result, final_votes, result,
                required=expansion_required, attempted=expansion_attempted, changed=expansion_changed,
            ))
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
        codes = split_values(row.get("system_types"))
        row["system_type_names"] = ";".join(system_type_names(codes, system_type_profile)) if codes else "Unknown"
        validate_processed_row(row, system_type_profile)
    accepted.sort(key=lambda row: (row.get("role", ""), row.get("full_name", "").casefold()))
    atomic_write_csv(output, accepted, CATALOGUE_COLUMNS)
    LOG.info("exported GitHub catalogue accepted=%d source_rows=%d output=%s", len(accepted), len(rows), output)
    return accepted
