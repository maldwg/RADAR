from __future__ import annotations

import os
import math
import time
from typing import Any, Iterator

import requests

from ids_meta.common.cache import PipelineCache
from ids_meta.common.run import utc_now
from ids_meta.common.scoring import ScoreResult, score_evidence
from ids_meta.common.text import contains_concept, normalize_text
from ids_meta.common.logging import logger

LOG = logger("openalex.collect")


def reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    """Rebuild an abstract from OpenAlex's inverted word-position index."""
    if not index:
        return ""
    words = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _position, word in sorted(words))


def normalize_work(item: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAlex API work into the pipeline's raw-work schema."""
    primary = item.get("primary_location") or {}
    source = primary.get("source") or {}
    best_oa = item.get("best_oa_location") or {}
    authors = [
        str((authorship.get("author") or {}).get("display_name", ""))
        for authorship in item.get("authorships") or []
        if (authorship.get("author") or {}).get("display_name")
    ]
    open_access = item.get("open_access") or {}
    return {
        "openalex_id": item.get("id", ""), "doi": str(item.get("doi") or "").removeprefix("https://doi.org/"),
        "title": item.get("title") or item.get("display_name") or "", "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
        "authors": ";".join(authors), "venue": source.get("display_name", ""),
        "publication_year": item.get("publication_year") or "", "publication_date": item.get("publication_date") or "",
        "work_type": item.get("type") or "", "primary_url": primary.get("landing_page_url") or item.get("doi") or item.get("id") or "",
        "pdf_url": best_oa.get("pdf_url") or primary.get("pdf_url") or "", "is_oa": str(bool(open_access.get("is_oa"))).lower(),
        "cited_by_count": int(item.get("cited_by_count") or 0), "referenced_works": item.get("referenced_works") or [],
    }


def is_survey_work(work: dict[str, Any]) -> bool:
    """Return whether a work is a survey without an explicit resource release claim."""
    text = f"{work.get('title', '')} {work.get('abstract', '')}"
    survey = str(work.get("work_type", "")).casefold() == "review" or any(
        contains_concept(text, term) for term in ("survey", "systematic review", "literature review", "dataset review", "mapping study")
    )
    release = any(contains_concept(text, term) for term in (
        "new dataset", "present a dataset", "presents a dataset", "introduce a dataset", "introduces a dataset",
        "release a dataset", "release the dataset", "released dataset", "public dataset", "publicly available",
        "made available", "open source", "open-source", "dataset toolkit", "data generation framework",
    ))
    return survey and not release


def work_text(work: dict[str, Any], title_only: bool = False) -> str:
    """Build the generic textual evidence used for retrieval-stage ranking."""
    if title_only:
        return str(work.get("title", ""))
    return "\n".join(str(work.get(field, "")) for field in ("title", "abstract", "venue", "work_type"))


def work_scores(work: dict[str, Any], profile: dict[str, Any]) -> tuple[ScoreResult, ScoreResult]:
    """Compute full-evidence and title-only retrieval scores for one work."""
    candidate = float("-inf")
    borderline = float("-inf")
    return (
        score_evidence(work_text(work), profile, candidate, borderline),
        score_evidence(work_text(work, title_only=True), profile, candidate, borderline),
    )


def usable_title(work: dict[str, Any], minimum_tokens: int) -> bool:
    """Reject blank or structurally uninformative scholarly-work titles."""
    tokens = normalize_text(str(work.get("title", ""))).split()
    return len(tokens) >= minimum_tokens and any(any(character.isalpha() for character in token) for token in tokens)


def metadata_quality(work: dict[str, Any]) -> int:
    """Count useful bibliographic fields without using popularity for selection."""
    return sum(bool(work.get(field)) for field in ("abstract", "doi", "authors", "venue", "publication_year", "publication_date"))


def balanced_search(
    client: "OpenAlexClient", queries: list[str], per_page: int, max_pages: int, scan_budget: int,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int], set[str]]:
    """Scan query iterators round-robin so no early query monopolizes retrieval."""
    iterators = {query: iter(client.search(query, per_page, max_pages)) for query in queries}
    active = list(queries)
    scanned = {query: 0 for query in queries}
    exhausted: set[str] = set()
    results: list[tuple[str, dict[str, Any]]] = []
    while active and len(results) < scan_budget:
        for query in list(active):
            if len(results) >= scan_budget:
                break
            try:
                item = next(iterators[query])
            except StopIteration:
                active.remove(query)
                exhausted.add(query)
                continue
            scanned[query] += 1
            results.append((query, item))
    return results, scanned, exhausted


def rank_work_pool(
    pool: dict[str, dict[str, Any]], profile: dict[str, Any], minimum_score: float, minimum_title_tokens: int,
    title_weight: float = 0.5, retain_below_minimum: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, tuple[ScoreResult, ScoreResult]]]:
    """Filter and rank a deduplicated work pool using generic textual evidence."""
    retain_below_minimum = retain_below_minimum or set()
    scores: dict[str, tuple[ScoreResult, ScoreResult]] = {}
    retained: list[dict[str, Any]] = []
    for key, work in pool.items():
        full, title = work_scores(work, profile)
        scores[key] = full, title
        if usable_title(work, minimum_title_tokens) and (full.score >= minimum_score or key in retain_below_minimum):
            retained.append(work)
    retained.sort(key=lambda work: (
        -(scores[str(work.get("openalex_id", ""))][0].score + title_weight * scores[str(work.get("openalex_id", ""))][1].score),
        -scores[str(work.get("openalex_id", ""))][0].score,
        -scores[str(work.get("openalex_id", ""))][1].score,
        -metadata_quality(work),
        str(work.get("openalex_id", "")),
    ))
    return retained, scores


class OpenAlexClient:
    api_root = "https://api.openalex.org"
    select = "id,doi,title,display_name,publication_year,publication_date,cited_by_count,abstract_inverted_index,primary_location,best_oa_location,open_access,authorships,referenced_works,type"

    def __init__(self, api_key: str | None = None, mailto: str | None = None, timeout: float = 30, request_pause_seconds: float = 0.5):
        """Initialize an OpenAlex client with credentials, timeout, and pacing."""
        self.api_key, self.mailto, self.timeout = api_key, mailto, timeout
        self.request_pause_seconds = max(0.0, request_pause_seconds)
        self._last_request_at: float | None = None
        self.session = requests.Session()

    def _wait_before_request(self) -> None:
        """Pause as needed to maintain the configured minimum request interval."""
        now = time.monotonic()
        if self._last_request_at is not None:
            wait = max(0.0, self.request_pause_seconds - (now - self._last_request_at))
            if wait:
                LOG.debug("pacing OpenAlex API call; sleeping %.2f seconds", wait)
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch and decode one OpenAlex response with bounded retries."""
        values = dict(params or {})
        if self.api_key:
            values["api_key"] = self.api_key
        if self.mailto:
            values["mailto"] = self.mailto
        last: Exception | None = None
        for attempt in range(5):
            try:
                self._wait_before_request()
                response = self.session.get(path if path.startswith("http") else self.api_root + path, params=values, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    wait = min(60, int(response.headers.get("Retry-After", 2**attempt) or 2**attempt))
                    LOG.warning("OpenAlex HTTP %d for %s; waiting %d seconds", response.status_code, path, wait)
                    time.sleep(wait)
                    continue
                if 400 <= response.status_code < 500:
                    raise RuntimeError(f"OpenAlex HTTP {response.status_code} for {path}: {response.text[:300]}")
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last = exc
                if attempt < 4:
                    LOG.warning("OpenAlex request error for %s (%s); retrying", path, exc)
                    time.sleep(2**attempt)
        raise RuntimeError(f"OpenAlex request failed for {path}: {last}") from last

    def search(self, query: str, per_page: int, max_pages: int) -> Iterator[dict[str, Any]]:
        """Yield works from a cursor-paginated OpenAlex full-text search."""
        cursor = "*"
        for page in range(1, max_pages + 1):
            payload = self.get("/works", {"search": query, "per-page": per_page, "cursor": cursor, "select": self.select})
            LOG.debug("OpenAlex query=%r page=%d results=%d", query, page, len(payload.get("results", [])))
            for item in payload.get("results", []):
                yield item
            cursor = str((payload.get("meta") or {}).get("next_cursor") or "")
            if not cursor:
                break

    def work(self, openalex_id: str) -> dict[str, Any]:
        """Fetch a single work by its stable OpenAlex identifier."""
        key = openalex_id.rsplit("/", 1)[-1]
        return self.get(f"/works/{key}", {"select": self.select})

    def works_by_ids(self, openalex_ids: list[str], batch_size: int = 50) -> Iterator[dict[str, Any]]:
        """Resolve references in OR-filter batches instead of one API request per cited work."""
        size = max(1, min(100, batch_size))
        for offset in range(0, len(openalex_ids), size):
            keys = [value.rsplit("/", 1)[-1] for value in openalex_ids[offset:offset + size]]
            payload = self.get("/works", {
                "filter": f"openalex_id:{'|'.join(keys)}", "per_page": len(keys), "select": self.select,
            })
            yield from payload.get("results", [])


def collect(settings: dict[str, Any], profile: dict[str, Any], cache: PipelineCache, limit: int | None = None, client: OpenAlexClient | None = None) -> dict[str, int]:
    """Collect direct and survey-snowballed OpenAlex works into the raw cache."""
    client = client or OpenAlexClient(
        os.getenv("OPENALEX_API_KEY"), os.getenv("OPENALEX_MAILTO"),
        request_pause_seconds=float(settings.get("request_pause_seconds", 0.5)),
    )
    per_page, pages = int(settings.get("per_page", 100)), int(settings.get("max_pages_per_query", 10))
    max_surveys = max(0, int(settings.get("max_surveys", 25)))
    reference_limit = max(0, int(settings.get("max_reference_works_per_survey", 300)))
    reference_total_limit = max(0, int(settings.get("max_survey_reference_candidates_total", 1000)))
    reference_fraction = max(0.0, min(0.9, float(settings.get("survey_reference_fraction", .35))))
    direct_scan_multiplier = max(1.0, float(settings.get("direct_scan_multiplier", 3.0)))
    survey_scan_multiplier = max(1.0, float(settings.get("survey_scan_multiplier", 4.0)))
    reference_scan_multiplier = max(1.0, float(settings.get("reference_scan_multiplier", 3.0)))
    survey_consensus_threshold = max(2, int(settings.get("survey_consensus_threshold", 2)))
    minimum_score = float(settings.get("retrieval_minimum_score", 1.0))
    minimum_citations = max(0, int(settings.get("minimum_citations", 0)))
    minimum_title_tokens = max(1, int(settings.get("retrieval_minimum_title_tokens", 3)))
    title_weight = max(0.0, float(settings.get("retrieval_title_weight", 0.5)))
    reference_budget = reference_total_limit if limit is None else min(reference_total_limit, int(limit * reference_fraction))
    survey_cap = max_surveys if limit is None else min(max_surveys, max(1, limit // 10))
    works: dict[str, dict[str, Any]] = {}
    relations: dict[str, set[str]] = {}
    queries: dict[str, set[str]] = {}
    survey_ids: list[str] = []
    fetched_at = utc_now()

    def store(normalized: dict[str, Any], relation: str, query: str = "", attributes: dict[str, Any] | None = None) -> str:
        """Store one normalized work and merge its discovery provenance."""
        key = str(normalized.get("openalex_id", ""))
        if not key:
            return ""
        works[key] = normalized
        relations.setdefault(key, set()).add(relation)
        if query:
            queries.setdefault(key, set()).add(query)
        existing = cache.get_record("openalex.work", key) or {}
        payload = {
            "work": normalized,
            "discovery_relations": sorted(set(existing.get("discovery_relations", [])) | relations[key]),
            "matched_queries": sorted(set(existing.get("matched_queries", [])) | queries.get(key, set())),
        }
        for field in ("retrieval_rank_score", "retrieval_score", "retrieval_title_score", "retrieval_signals", "survey_discovery_score"):
            value = (attributes or {}).get(field, existing.get(field))
            if value is not None:
                payload[field] = value
        cache.put_record("openalex.work", key, payload, fetched_at)
        return key

    LOG.info(
        "starting OpenAlex collection direct_queries=%d survey_queries=%d limit=%s pacing=%.2fs survey_cap=%d reference_budget=%d direct_scan_multiplier=%.1f reference_scan_multiplier=%.1f",
        len(profile.get("direct_queries", [])), len(profile.get("survey_queries", [])), limit or "none",
        float(settings.get("request_pause_seconds", 0.5)), survey_cap, reference_budget,
        direct_scan_multiplier, reference_scan_multiplier,
    )

    # Oversample all survey queries evenly, then choose useful reference sources without citation-based ranking.
    survey_queries = [str(query) for query in profile.get("survey_queries", [])]
    survey_scan_budget = max(len(survey_queries), math.ceil(max(1, survey_cap) * survey_scan_multiplier)) if survey_cap else 0
    survey_results, survey_scanned, survey_exhausted = balanced_search(client, survey_queries, per_page, pages, survey_scan_budget) if survey_scan_budget else ([], {}, set())
    survey_pool: dict[str, dict[str, Any]] = {}
    survey_matches: dict[str, set[str]] = {}
    for query, item in survey_results:
        normalized = normalize_work(item)
        key = str(normalized.get("openalex_id", ""))
        if key and is_survey_work(normalized):
            survey_pool[key] = normalized
            survey_matches.setdefault(key, set()).add(query)
    positive_profile = {"positive": profile.get("positive", {})}
    ranked_surveys, survey_scores = rank_work_pool(survey_pool, positive_profile, float("-inf"), minimum_title_tokens, title_weight)
    for rank, normalized in enumerate(ranked_surveys[:survey_cap], 1):
        key = str(normalized["openalex_id"])
        full_score, title_score = survey_scores[key]
        survey_rank_score = full_score.score + title_weight * title_score.score
        attributes = {"survey_discovery_score": survey_rank_score}
        for query in sorted(survey_matches.get(key, set())) or [""]:
            store(normalized, "survey_search", query, attributes)
        survey_ids.append(key)
        LOG.info(
            "using survey rank=%d/%d retrieval_rank_score=%.1f score=%.1f title_score=%.1f citations=%d references=%d provenance=survey_search title=%s",
            rank, min(survey_cap, len(ranked_surveys)), survey_rank_score, full_score.score, title_score.score,
            int(normalized.get("cited_by_count") or 0), len(normalized.get("referenced_works") or []), normalized.get("title", key),
        )
    selected_survey_ids = set(survey_ids)
    for query in survey_queries:
        selected_count = sum(query in survey_matches.get(key, set()) and key in selected_survey_ids for key in survey_pool)
        status = "complete" if query in survey_exhausted else "bounded"
        cache.put_coverage("openalex", query, "survey_cursor", status, survey_scanned.get(query, 0), {
            "configured_page_cap": pages, "survey_pool_matches": sum(query in values for values in survey_matches.values()),
            "surveys_selected": selected_count, "survey_cap": survey_cap, "balanced_scan_budget": survey_scan_budget,
        }, utc_now())
        LOG.info("finished survey query=%r scanned=%d selected=%d coverage=%s", query, survey_scanned.get(query, 0), selected_count, status)

    reference_sources: dict[str, set[str]] = {}
    for survey_id in survey_ids:
        for reference_id in (works.get(survey_id, {}).get("referenced_works") or [])[:reference_limit]:
            if reference_id:
                reference_sources.setdefault(str(reference_id), set()).add(survey_id)
    reference_budget = min(reference_budget, len(reference_sources))
    LOG.info("survey discovery complete surveys=%d distinct_reference_candidates=%d reserved_reference_budget=%d", len(survey_ids), len(reference_sources), reference_budget)

    # Oversample every direct query round-robin and retain the strongest generic evidence across the combined pool.
    direct_budget = None if limit is None else max(0, limit - reference_budget)
    direct_queries = [str(query) for query in profile.get("direct_queries", [])]
    direct_capacity = per_page * pages * max(1, len(direct_queries))
    desired_direct = direct_capacity if direct_budget is None else direct_budget
    direct_scan_budget = min(direct_capacity, max(len(direct_queries), math.ceil(max(1, desired_direct) * direct_scan_multiplier))) if direct_queries else 0
    direct_results, direct_scanned, direct_exhausted = balanced_search(client, direct_queries, per_page, pages, direct_scan_budget) if direct_scan_budget else ([], {}, set())
    direct_pool: dict[str, dict[str, Any]] = {}
    direct_matches: dict[str, set[str]] = {}
    for query, item in direct_results:
        normalized = normalize_work(item)
        key = str(normalized.get("openalex_id", ""))
        if key and int(normalized.get("cited_by_count") or 0) >= minimum_citations:
            direct_pool[key] = normalized
            direct_matches.setdefault(key, set()).add(query)
    ranked_direct, direct_scores = rank_work_pool(direct_pool, profile, minimum_score, minimum_title_tokens, title_weight)
    selected_direct_ids: set[str] = set()
    for normalized in ranked_direct:
        if direct_budget is not None and len(selected_direct_ids) >= direct_budget:
            break
        key = str(normalized["openalex_id"])
        full_score, title_score = direct_scores[key]
        rank_score = full_score.score + title_weight * title_score.score
        attributes = {
            "retrieval_rank_score": rank_score,
            "retrieval_score": full_score.score, "retrieval_title_score": title_score.score,
            "retrieval_signals": list(full_score.signals),
        }
        for query in sorted(direct_matches.get(key, set())) or [""]:
            store(normalized, "direct_search", query, attributes)
        selected_direct_ids.add(key)
        LOG.info(
            "selected direct paper rank=%d retrieval_rank_score=%.1f score=%.1f title_score=%.1f citations=%d provenance=direct_search queries=%s title=%s",
            len(selected_direct_ids), rank_score, full_score.score, title_score.score, int(normalized.get("cited_by_count") or 0),
            ";".join(sorted(direct_matches.get(key, set()))), normalized.get("title", key),
        )
    for query in direct_queries:
        selected_count = sum(query in direct_matches.get(key, set()) for key in selected_direct_ids)
        status = "complete" if query in direct_exhausted else "bounded"
        cache.put_coverage("openalex", query, "direct_cursor", status, direct_scanned.get(query, 0), {
            "configured_page_cap": pages, "selected_results": selected_count,
            "balanced_scan_budget": direct_scan_budget, "minimum_retrieval_score": minimum_score,
        }, utc_now())
        LOG.info("finished direct query=%r scanned=%d selected=%d coverage=%s", query, direct_scanned.get(query, 0), selected_count, status)
    if direct_budget is not None and len(selected_direct_ids) < direct_budget:
        LOG.warning("direct retrieval produced %d/%d retained candidate papers after generic quality ranking", len(selected_direct_ids), direct_budget)

    # Prefer references cited by several selected surveys, resolve an oversampled pool, then rank their actual metadata.
    survey_reference_ids: set[str] = set()
    reference_edges = 0
    reference_scan_budget = min(len(reference_sources), math.ceil(reference_budget * reference_scan_multiplier)) if reference_budget else 0
    reference_ids = sorted(reference_sources, key=lambda key: (-len(reference_sources[key]), key))[:reference_scan_budget]
    reference_pool: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for reference_id in reference_ids:
        if reference_id in works:
            reference_pool[reference_id] = works[reference_id]
            continue
        cached = cache.get_record("openalex.work", reference_id)
        if cached and cached.get("work"):
            reference_pool[reference_id] = cached["work"]
        else:
            unresolved.append(reference_id)
    if unresolved:
        LOG.info("batch-resolving snowball pool unresolved=%d scan_budget=%d", len(unresolved), reference_scan_budget)
        try:
            for item in client.works_by_ids(unresolved):
                normalized = normalize_work(item)
                if normalized.get("openalex_id"):
                    reference_pool[str(normalized["openalex_id"])] = normalized
        except RuntimeError as exc:
            LOG.warning("reference batch failed: %s", exc)
    reference_pool = {
        key: work for key, work in reference_pool.items()
        if int(work.get("cited_by_count") or 0) >= minimum_citations
    }
    consensus_reference_ids = {
        key for key in reference_pool
        if len(reference_sources.get(key, set())) >= survey_consensus_threshold
    }
    ranked_references, reference_scores = rank_work_pool(
        reference_pool, profile, minimum_score, minimum_title_tokens, title_weight,
        retain_below_minimum=consensus_reference_ids,
    )
    ranked_references.sort(key=lambda work: (
        -len(reference_sources.get(str(work.get("openalex_id", "")), set())),
        -(reference_scores[str(work.get("openalex_id", ""))][0].score + title_weight * reference_scores[str(work.get("openalex_id", ""))][1].score),
        str(work.get("openalex_id", "")),
    ))
    new_reference_works = 0
    new_reference_ids: set[str] = set()
    for normalized in ranked_references:
        key = str(normalized["openalex_id"])
        if is_survey_work(normalized):
            continue
        is_new_candidate = key not in selected_direct_ids and key not in new_reference_ids
        if is_new_candidate and new_reference_works >= reference_budget:
            continue
        full_score, title_score = reference_scores[key]
        rank_score = full_score.score + title_weight * title_score.score
        attributes = {
            "retrieval_rank_score": rank_score,
            "retrieval_score": full_score.score, "retrieval_title_score": title_score.score,
            "retrieval_signals": list(full_score.signals),
        }
        store(normalized, "survey_reference", attributes=attributes)
        for survey_id in sorted(reference_sources.get(key, set())):
            store(normalized, f"survey_reference:{survey_id.rsplit('/', 1)[-1]}", attributes=attributes)
            reference_edges += 1
        survey_reference_ids.add(key)
        if is_new_candidate:
            new_reference_works += 1
            new_reference_ids.add(key)
        LOG.info(
            "selected snowball paper rank=%d retrieval_rank_score=%.1f score=%.1f title_score=%.1f citations=%d survey_sources=%d provenance=survey_reference title=%s",
            len(survey_reference_ids), rank_score, full_score.score, title_score.score, int(normalized.get("cited_by_count") or 0),
            len(reference_sources.get(key, set())), normalized.get("title", key),
        )

    # Backfill unused snowball capacity with the next strongest direct candidates.
    candidate_ids = selected_direct_ids | new_reference_ids
    if limit is not None and len(candidate_ids) < limit:
        for normalized in ranked_direct:
            if len(candidate_ids) >= limit:
                break
            key = str(normalized["openalex_id"])
            if key in candidate_ids:
                continue
            full_score, title_score = direct_scores[key]
            rank_score = full_score.score + title_weight * title_score.score
            attributes = {
                "retrieval_rank_score": rank_score, "retrieval_score": full_score.score,
                "retrieval_title_score": title_score.score, "retrieval_signals": list(full_score.signals),
            }
            for query in sorted(direct_matches.get(key, set())) or [""]:
                store(normalized, "direct_search", query, attributes)
            selected_direct_ids.add(key)
            candidate_ids.add(key)
            LOG.info(
                "backfilled direct paper rank=%d retrieval_rank_score=%.1f score=%.1f title_score=%.1f citations=%d title=%s",
                len(selected_direct_ids), rank_score, full_score.score, title_score.score,
                int(normalized.get("cited_by_count") or 0), normalized.get("title", key),
            )
    for survey_id in survey_ids:
        available = {key for key, sources in reference_sources.items() if survey_id in sources}
        scanned_ids = available & set(reference_ids)
        selected_ids = available & survey_reference_ids
        status = "complete" if len(scanned_ids) == len(available) else "bounded"
        cache.put_coverage("openalex", f"references:{survey_id}", "reference_list", status, len(scanned_ids), {
            "available_references": len(available), "scanned_references": len(scanned_ids),
            "selected_references": len(selected_ids), "per_survey_limit": reference_limit,
            "total_reference_budget": reference_budget, "reference_scan_budget": reference_scan_budget,
        }, utc_now())
        LOG.info("finished survey snowball survey=%s scanned=%d selected=%d coverage=%s", survey_id, len(scanned_ids), len(selected_ids), status)
    LOG.info(
        "OpenAlex collection complete candidate_papers=%d auxiliary_surveys=%d direct_candidates=%d new_snowball_candidates=%d reference_edges=%d",
        len(candidate_ids), len(survey_ids), len(selected_direct_ids), len(new_reference_ids), reference_edges,
    )
    cache.put_record("openalex.collection", "latest", {
        "work_ids": sorted(candidate_ids), "selected_at": fetched_at, "collection_limit": limit,
        "survey_ids": sorted(survey_ids), "direct_ids": sorted(selected_direct_ids),
        "survey_reference_ids": sorted(new_reference_ids),
    }, fetched_at)
    return {
        "discovered": len(candidate_ids), "survey_sources": len(survey_ids), "auxiliary_cached_works": len(works) - len(candidate_ids),
        "direct_selected": len(selected_direct_ids), "survey_reference_works": len(new_reference_ids), "survey_reference_edges": reference_edges,
        "survey_results_scanned": len(survey_results), "direct_results_scanned": len(direct_results),
        "reference_results_scanned": len(reference_ids),
    }
