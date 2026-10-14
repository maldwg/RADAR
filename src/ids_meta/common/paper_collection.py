"""Shared scholarly collection: direct retrieval, survey seeds and backward traversal."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Iterator

from ids_meta.adapters.contracts import DiscoveryAdapter, LookupAdapter, ProviderQuery, ReferenceAdapter, SourceRecord
from ids_meta.common.cache import PipelineCache
from ids_meta.common.evidence import snowball_depth
from ids_meta.common.limits import query_result_limit, limit_label
from ids_meta.common.logging import logger
from ids_meta.common.run import utc_now
from ids_meta.common.scoring import ScoreResult, score_evidence
from ids_meta.common.text import contains_concept, normalize_text

LOG = logger("paper_collection")


class PaperSearchClient:
    """Provide normalized paper iteration to the common collection algorithm."""
    def __init__(self, adapter: DiscoveryAdapter, profile: dict[str, Any]):
        """Validate provider queries and retain raw responses and memberships."""
        self.adapter = adapter
        self.queries = adapter.queries(profile)
        for query in self.queries:
            adapter.validate_query(query)
        self.records: dict[str, SourceRecord] = {}
        self.memberships: dict[str, list[dict[str, Any]]] = {}
        self.audits: list[dict[str, Any]] = []
        self.lookups: list[dict[str, Any]] = []

    def _work(self, record: SourceRecord) -> dict[str, Any]:
        """Keep normalized metadata/evidence and a provider-native traversal identity."""
        self.records[record.native_id] = record
        return {**record.metadata, **record.evidence, "native_id": record.native_id}

    def search(self, text: str, per_page: int, max_pages: int | None) -> Iterator[dict[str, Any]]:
        """Page explicit provider queries, including study-owned survey variants."""
        query = next((query for query in self.queries if query.text == text), None)
        if query is None:
            query = ProviderQuery(text, text, dict(self.queries[0].parameters))
        self.adapter.validate_query(query)
        audit = {"provider": self.adapter.descriptor.name, "name": query.name, "search": text,
            "parameters": query.parameters, "scanned": 0, "reported_hits": None, "coverage": "bounded", "pages": []}
        self.audits.append(audit)
        query_index = len(self.audits) - 1
        cursor, seen, complete, page_number = None, set(), True, 0
        while max_pages is None or page_number < max_pages:
            if cursor in seen:
                raise RuntimeError("provider repeated study-search cursor")
            seen.add(cursor)
            page = self.adapter.search(query, cursor, per_page)
            page_number += 1
            complete = complete and page.complete
            audit["reported_hits"] = page.total
            audit["pages"].append({"cursor": cursor, "response": page.raw})
            for raw in page.records:
                record = self.adapter.normalize(raw)
                audit["scanned"] += 1
                membership = {"provider": record.provider, "native_id": record.native_id,
                    "query": query.name, "rank": audit["scanned"],
                    "selection_rank": [audit["scanned"], query_index, record.native_id]}
                self.memberships.setdefault(record.native_id, []).append(membership)
                work = self._work(record)
                work["query_rank"] = membership["selection_rank"]
                yield work
            if not page.records or page.next_cursor is None:
                audit["coverage"] = "complete" if complete else "bounded"
                return
            cursor = page.next_cursor

    def works_by_ids(self, identifiers: list[str]) -> Iterator[dict[str, Any]]:
        """Resolve references through an optional batch lookup or individual lookups."""
        if not isinstance(self.adapter, LookupAdapter):
            raise ValueError("reference traversal requires provider lookup capability")
        batch = getattr(self.adapter, "lookup_many", None)
        records = batch(identifiers) if batch else (self.adapter.lookup(identifier) for identifier in identifiers)
        for record in records:
            self.lookups.append({"native_id": record.native_id, "response": record.native})
            yield self._work(record)


DEFAULT_SURVEY_TERMS = ("survey", "systematic review", "literature review", "dataset review", "mapping study")
DEFAULT_RELEASE_TERMS = (
    "new dataset", "present a dataset", "presents a dataset", "introduce a dataset", "introduces a dataset",
    "release a dataset", "release the dataset", "released dataset", "public dataset", "publicly available",
    "made available", "open source", "open-source", "dataset toolkit", "data generation framework",
)


def is_survey_work(
    work: dict[str, Any], survey_terms: list[str] | tuple[str, ...] = DEFAULT_SURVEY_TERMS,
    release_terms: list[str] | tuple[str, ...] = DEFAULT_RELEASE_TERMS,
) -> bool:
    """Return whether a work matches configured survey terms without a release claim."""
    text = f"{work.get('title', '')} {work.get('abstract', '')}"
    survey = str(work.get("work_type", "")).casefold() == "review" or any(
        contains_concept(text, term) for term in survey_terms
    )
    release = any(contains_concept(text, term) for term in release_terms)
    return survey and not release


def work_text(work: dict[str, Any], title_only: bool = False) -> str:
    """Build the generic textual evidence used for retrieval-stage ranking."""
    if title_only:
        return str(work.get("title", ""))
    return "\n".join(str(work.get(field, "")) for field in ("title", "abstract", "venue", "work_type"))


def prefilter_paper(work: dict[str, Any], profile: dict[str, Any], settings: dict[str, Any],
                    survey_sources: int = 0) -> dict[str, Any]:
    """Apply the same evidence scores and survey consensus rules to any paper study."""
    result = score_evidence(work_text(work) + "\n" + str(work.get("full_text", "")), profile,
        float(settings.get("candidate_threshold", 7)), float(settings.get("borderline_threshold", 4)))
    band, signals = result.band, list(result.signals)
    threshold = settings.get("survey_consensus_threshold", 2)
    if band == "reject" and survey_sources >= threshold:
        band = "borderline"
        signals.append(f"survey_consensus:{survey_sources}")
    if int(work.get("cited_by_count") or 0) < settings.get("minimum_citations", 0):
        band = "reject"
    return {"prefilter_score": result.score, "prefilter_band": band, "prefilter_signals": ";".join(signals)}


def eligible_papers(rows: list[dict[str, Any]], classify_borderline: bool) -> list[dict[str, Any]]:
    """Choose the same candidate bands before either scientific screening strategy."""
    return [row for row in rows if row.get("prefilter_band", "candidate") == "candidate"
        or classify_borderline and row.get("prefilter_band", "candidate") == "borderline"]


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


def sample_query_results(
    client: "PaperSearchClient", queries: list[str], sample_size: int | None, per_page: int, max_pages: int | None,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, int], set[str]]:
    """Scan each query up to its optional result cap, or until exhausted."""
    scanned = {query: 0 for query in queries}
    exhausted: set[str] = set()
    results: list[tuple[str, dict[str, Any]]] = []
    for query in queries:
        iterator = iter(client.search(query, per_page if sample_size is None else min(per_page, sample_size), max_pages))
        while sample_size is None or scanned[query] < sample_size:
            try:
                item = next(iterator)
            except StopIteration:
                exhausted.add(query)
                break
            scanned[query] += 1
            results.append((query, item))
    return results, scanned, exhausted



def survey_query_candidates(
    client: "PaperSearchClient", queries: list[str], sample_size: int | None, per_page: int, pages: int | None,
    survey_terms: list[str], release_terms: list[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, int], set[str]]:
    """Sample survey queries and return only configured survey matches."""
    results, scanned, exhausted = sample_query_results(client, queries, sample_size, per_page, pages)
    pool: dict[str, dict[str, Any]] = {}
    matches: dict[str, set[str]] = {}
    for query, item in results:
        work = item
        key = str(work.get("native_id", ""))
        if key and is_survey_work(work, survey_terms, release_terms):
            pool[key] = work
            matches.setdefault(key, set()).add(query)
    return pool, matches, scanned, exhausted


def direct_query_candidates(
    client: "PaperSearchClient", queries: list[str], sample_size: int | None, per_page: int, pages: int | None,
    minimum_citations: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]], dict[str, int], set[str]]:
    """Sample direct queries and deduplicate eligible papers across them."""
    results, scanned, exhausted = sample_query_results(client, queries, sample_size, per_page, pages)
    pool: dict[str, dict[str, Any]] = {}
    matches: dict[str, set[str]] = {}
    for query, item in results:
        work = item
        key = str(work.get("native_id", ""))
        if key and int(work.get("cited_by_count") or 0) >= minimum_citations:
            if key in pool:
                work["query_rank"] = min(pool[key]["query_rank"], work["query_rank"])
            pool[key] = work
            matches.setdefault(key, set()).add(query)
    return pool, matches, scanned, exhausted


def survey_reference_candidates(
    client: "PaperSearchClient", reference_sources: dict[str, set[str]], sample_size: int | None,
    works: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Resolve at most x survey references, preferring multi-survey consensus."""
    reference_ids = sorted(reference_sources, key=lambda key: (-len(reference_sources[key]), key))[:sample_size]
    pool: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for reference_id in reference_ids:
        if reference_id in works:
            pool[reference_id] = works[reference_id]
        else:
            unresolved.append(reference_id)
    if unresolved:
        for item in client.works_by_ids(unresolved):
            work = item
            if work.get("native_id"):
                pool[str(work["native_id"])] = work
    return pool, reference_ids


def publisher_for_work(work: dict[str, Any], publishers: dict[str, list[str]]) -> str:
    """Resolve a work to the first configured publisher family using venue metadata."""
    haystack = normalize_text(f"{work.get('publisher', '')} {work.get('venue', '')}")
    for name, aliases in publishers.items():
        if any(normalize_text(alias) in haystack for alias in aliases if normalize_text(alias)):
            return str(name)
    return ""


def publisher_diverse_surveys(
    ranked: list[dict[str, Any]], cap: int, publishers: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Select one survey per requested publisher when available, then fill by rank."""
    if cap <= 0:
        return [], set()
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    covered: set[str] = set()
    for publisher in publishers:
        match = next((work for work in ranked if publisher_for_work(work, publishers) == publisher), None)
        if match is None:
            continue
        key = str(match.get("native_id", ""))
        if key and key not in chosen_ids and len(chosen) < cap:
            chosen.append(match); chosen_ids.add(key); covered.add(publisher)
    for work in ranked:
        key = str(work.get("native_id", ""))
        if key and key not in chosen_ids and len(chosen) < cap:
            chosen.append(work); chosen_ids.add(key)
            publisher = publisher_for_work(work, publishers)
            if publisher:
                covered.add(publisher)
    return chosen, covered


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
        -(scores[str(work.get("native_id", ""))][0].score + title_weight * scores[str(work.get("native_id", ""))][1].score),
        -scores[str(work.get("native_id", ""))][0].score,
        -scores[str(work.get("native_id", ""))][1].score,
        -metadata_quality(work),
        str(work.get("native_id", "")),
    ))
    return retained, scores



def collect_papers(settings: dict[str, Any], profile: dict[str, Any], cache: PipelineCache,
                   limit: int | None, adapter: DiscoveryAdapter) -> tuple[dict[str, int], dict[str, SourceRecord], dict[str, list[dict[str, Any]]]]:
    """Collect ranked direct and survey-snowballed papers through one provider adapter."""
    client = PaperSearchClient(adapter, profile)
    provider = adapter.descriptor.name
    mode = settings.get("collection_mode", "ranked")
    if mode not in {"ranked", "query_position"}:
        raise ValueError("collection_mode must be ranked or query_position")
    if (settings.get("max_surveys", 0) or settings.get("snowball_max_depth", 0)) and not isinstance(adapter, ReferenceAdapter):
        raise ValueError("survey/snowball collection requires provider reference capability")
    per_page = int(settings.get("per_page", 100))
    max_surveys = max(0, int(settings.get("max_surveys", 25)))
    reference_fraction = max(0.0, min(0.9, float(settings.get("survey_reference_fraction", .35))))
    query_sample_size = query_result_limit(settings, limit)
    survey_query_sample_size = query_result_limit(settings, limit, "survey_query_limit_factor")
    reference_sample_size = query_result_limit(settings, limit, "reference_limit_factor")
    reference_limit = reference_sample_size
    pages = None if query_sample_size is None else math.ceil(query_sample_size / per_page)
    survey_pages = None if survey_query_sample_size is None else math.ceil(survey_query_sample_size / per_page)
    survey_consensus_threshold = max(2, int(settings.get("survey_consensus_threshold", 2)))
    max_depth = snowball_depth(settings)
    start_papers = max(1, int(settings.get("snowball_start_papers", 20)))
    minimum_score = float(settings.get("retrieval_minimum_score", 1.0))
    minimum_citations = max(0, int(settings.get("minimum_citations", 0)))
    minimum_title_tokens = max(1, int(settings.get("retrieval_minimum_title_tokens", 3)))
    title_weight = max(0.0, float(settings.get("retrieval_title_weight", 0.5)))
    reference_budget = None if limit is None else int(limit * reference_fraction)
    # Surveys are auxiliary traversal sources and do not consume the final candidate limit.
    survey_cap = max_surveys
    works: dict[str, dict[str, Any]] = {}
    relations: dict[str, set[str]] = {}
    queries: dict[str, set[str]] = {}
    attributes_by_work: dict[str, dict[str, Any]] = {}
    survey_ids: list[str] = []
    fetched_at = utc_now()

    def store(normalized: dict[str, Any], relation: str, query: str = "", attributes: dict[str, Any] | None = None) -> str:
        """Store one normalized work and merge its discovery provenance."""
        key = str(normalized.get("native_id", ""))
        if not key:
            return ""
        works[key] = normalized
        relations.setdefault(key, set()).add(relation)
        if query:
            queries.setdefault(key, set()).add(query)
        attributes_by_work.setdefault(key, {}).update(attributes or {})
        payload = {
            "work": normalized,
            "discovery_relations": sorted(relations[key]),
            "matched_queries": sorted(queries.get(key, set())),
        }
        for field in ("retrieval_rank_score", "retrieval_score", "retrieval_title_score", "retrieval_signals", "survey_discovery_score", "snowball_depth"):
            value = attributes_by_work[key].get(field)
            if value is not None:
                payload[field] = value
        cache.put_record(f"{provider}.work", key, payload, fetched_at)
        return key

    LOG.info(
        "starting scholarly collection direct_queries=%d survey_queries=%d query_cap=%s survey_query_cap=%s reference_scan_cap=%s select_best=%s survey_cap=%d reference_budget=%s",
        len(client.queries), len(profile.get("survey_queries", [])), limit_label(query_sample_size),
        limit_label(survey_query_sample_size), limit_label(reference_sample_size),
        limit_label(limit), survey_cap, limit_label(reference_budget),
    )

    # Sample x results per survey query, then select the best configured survey sources.
    survey_queries = [str(query) for query in profile.get("survey_queries", [])]
    configured_publishers = profile.get("survey_publishers", {})
    publishers = {
        str(name): [str(alias) for alias in aliases]
        for name, aliases in configured_publishers.items()
        if isinstance(aliases, list)
    } if isinstance(configured_publishers, dict) else {}
    publisher_queries = [
        f"{query} {aliases[0]}" for query in survey_queries
        for aliases in publishers.values() if aliases
    ]
    survey_search_queries = list(dict.fromkeys([*survey_queries, *publisher_queries]))
    identification = profile.get("survey_identification", {})
    survey_terms = [str(value) for value in identification.get("survey_terms", DEFAULT_SURVEY_TERMS)]
    release_terms = [str(value) for value in identification.get("release_terms", DEFAULT_RELEASE_TERMS)]
    survey_pool, survey_matches, survey_scanned, survey_exhausted = survey_query_candidates(
        client, survey_search_queries, survey_query_sample_size, per_page, survey_pages, survey_terms, release_terms,
    ) if survey_cap else ({}, {}, {}, set())
    positive_profile = {"positive": profile.get("positive", {})}
    ranked_surveys, survey_scores = rank_work_pool(survey_pool, positive_profile, float("-inf"), minimum_title_tokens, title_weight)
    selected_surveys, covered_publishers = publisher_diverse_surveys(ranked_surveys, survey_cap, publishers)
    for rank, normalized in enumerate(selected_surveys, 1):
        key = str(normalized["native_id"])
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
    for query in survey_search_queries:
        selected_count = sum(query in survey_matches.get(key, set()) and key in selected_survey_ids for key in survey_pool)
        status = "complete" if query in survey_exhausted else "bounded"
        cache.put_coverage(provider, query, "survey_cursor", status, survey_scanned.get(query, 0), {
            "configured_page_cap": survey_pages, "survey_pool_matches": sum(query in values for values in survey_matches.values()),
            "surveys_selected": selected_count, "survey_cap": survey_cap, "sample_size_per_query": survey_query_sample_size,
        }, utc_now())
        LOG.info("finished survey query=%r scanned=%d selected=%d coverage=%s", query, survey_scanned.get(query, 0), selected_count, status)
    missing_publishers = set(publishers) - covered_publishers
    cache.put_coverage(provider, "survey_publishers", "publisher_targets",
                       "complete" if not missing_publishers else "bounded", len(covered_publishers), {
                           "requested_publishers": sorted(publishers),
                           "covered_publishers": sorted(covered_publishers),
                           "missing_publishers": sorted(missing_publishers),
                       }, utc_now())
    if missing_publishers:
        LOG.warning("no matching survey found for configured publishers=%s", ",".join(sorted(missing_publishers)))

    reference_sources: dict[str, set[str]] = {}
    for survey_id in survey_ids:
        for reference_id in (works.get(survey_id, {}).get("referenced_works") or [])[:reference_limit]:
            if reference_id:
                reference_sources.setdefault(str(reference_id), set()).add(survey_id)
    LOG.info("survey discovery complete surveys=%d distinct_reference_candidates=%d reserved_reference_budget=%s", len(survey_ids), len(reference_sources), limit_label(reference_budget))

    # Sample x per direct query, then select the best y from the deduplicated pool.
    direct_budget = None if limit is None else max(0, limit - reference_budget)
    direct_queries = [query.text for query in client.queries]
    direct_pool, direct_matches, direct_scanned, direct_exhausted = direct_query_candidates(
        client, direct_queries, query_sample_size, per_page, pages, minimum_citations,
    )
    ranked_direct, direct_scores = rank_work_pool(direct_pool, profile, minimum_score, minimum_title_tokens, title_weight)
    if mode == "query_position":
        ranked_direct.sort(key=lambda work: work["query_rank"])
    selected_direct_ids: set[str] = set()
    for normalized in ranked_direct:
        if direct_budget is not None and len(selected_direct_ids) >= direct_budget:
            break
        key = str(normalized["native_id"])
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
        cache.put_coverage(provider, query, "direct_cursor", status, direct_scanned.get(query, 0), {
            "configured_page_cap": pages, "selected_results": selected_count,
            "sample_size_per_query": query_sample_size, "minimum_retrieval_score": minimum_score,
        }, utc_now())
        LOG.info("finished direct query=%r scanned=%d selected=%d coverage=%s", query, direct_scanned.get(query, 0), selected_count, status)
    if direct_budget is not None and len(selected_direct_ids) < direct_budget:
        LOG.warning("direct retrieval produced %d/%d retained candidate papers after generic quality ranking", len(selected_direct_ids), direct_budget)

    # Traverse references from selected surveys and the strongest direct papers until
    # convergence or the configured protocol depth. Relevant surveys remain traversal
    # nodes only; non-survey papers may enter the candidate set.
    survey_reference_ids: set[str] = set()
    new_reference_ids: set[str] = set()
    reference_edges = 0
    reference_ids: list[str] = []
    examined_ids: set[str] = set(survey_ids) | set(selected_direct_ids)
    direct_seed_ids = [
        str(work.get("native_id", "")) for work in ranked_direct
        if str(work.get("native_id", "")) in selected_direct_ids
    ][:start_papers]
    frontier = list(dict.fromkeys([*survey_ids, *direct_seed_ids]))
    reached_depth = 0
    for depth in range(1, max_depth + 1):
        depth_sources: dict[str, set[str]] = {}
        for parent_id in frontier:
            parent = works[parent_id]
            for reference_id in (parent.get("referenced_works") or [])[:reference_limit]:
                child_id = str(reference_id)
                if child_id and child_id not in examined_ids:
                    depth_sources.setdefault(child_id, set()).add(parent_id)
                    reference_sources.setdefault(child_id, set()).add(parent_id)
                    reference_edges += 1
        remaining_scan_budget = None if reference_sample_size is None else max(0, reference_sample_size - len(reference_ids))
        if not depth_sources or remaining_scan_budget == 0:
            break
        try:
            depth_pool, sampled_ids = survey_reference_candidates(
                client, depth_sources, remaining_scan_budget, works,
            )
        except RuntimeError as exc:
            LOG.warning("snowball reference batch failed depth=%d error=%s", depth, exc)
            break
        reference_ids.extend(sampled_ids)
        examined_ids.update(sampled_ids)
        consensus_ids = {
            key for key in depth_pool
            if len(reference_sources.get(key, set())) >= survey_consensus_threshold
        }
        traversal_survey_ids = {
            key for key, work in depth_pool.items()
            if is_survey_work(work, survey_terms, release_terms)
        }
        ranked_references, reference_scores = rank_work_pool(
            depth_pool, profile, minimum_score, minimum_title_tokens, title_weight,
            retain_below_minimum=consensus_ids | traversal_survey_ids,
        )
        next_frontier: list[str] = []
        for normalized in ranked_references:
            key = str(normalized["native_id"])
            survey = is_survey_work(normalized, survey_terms, release_terms)
            if not survey and int(normalized.get("cited_by_count") or 0) < minimum_citations:
                continue
            full_score, title_score = reference_scores[key]
            rank_score = full_score.score + title_weight * title_score.score
            attributes = {
                "retrieval_rank_score": rank_score, "retrieval_score": full_score.score,
                "retrieval_title_score": title_score.score, "retrieval_signals": list(full_score.signals),
                "snowball_depth": depth,
            }
            parents = sorted(reference_sources.get(key, set()))
            store(normalized, "backward_reference", attributes=attributes)
            for parent_id in parents:
                store(normalized, f"backward_reference:{parent_id.rsplit('/', 1)[-1]}", attributes=attributes)
                if parent_id in set(survey_ids):
                    store(normalized, "survey_reference", attributes=attributes)
                    store(normalized, f"survey_reference:{parent_id.rsplit('/', 1)[-1]}", attributes=attributes)
                    survey_reference_ids.add(key)
            if survey:
                if key not in survey_ids:
                    survey_ids.append(key)
                store(normalized, "snowball_survey", attributes=attributes)
                next_frontier.append(key)
                continue
            next_frontier.append(key)
            if key not in selected_direct_ids and (reference_budget is None or len(new_reference_ids) < reference_budget):
                new_reference_ids.add(key)
            LOG.info(
                "examined relevant snowball paper depth=%d score=%.1f citations=%d parents=%d candidate=%s title=%s",
                depth, rank_score, int(normalized.get("cited_by_count") or 0), len(parents),
                key in new_reference_ids, normalized.get("title", key),
            )
        reached_depth = depth
        frontier = list(dict.fromkeys(next_frontier))
        cache.put_coverage(provider, f"snowball_depth:{depth}", "backward_references",
                           "complete" if not frontier else "bounded", len(sampled_ids), {
                               "frontier_in": len(depth_sources), "relevant_frontier_out": len(frontier),
                               "configured_max_depth": max_depth,
                           }, utc_now())
        if not frontier:
            break

    # Backfill unused snowball capacity with the next strongest direct candidates.
    candidate_ids = selected_direct_ids | new_reference_ids
    if limit is not None and len(candidate_ids) < limit:
        for normalized in ranked_direct:
            if len(candidate_ids) >= limit:
                break
            key = str(normalized["native_id"])
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
        cache.put_coverage(provider, f"references:{survey_id}", "reference_list", status, len(scanned_ids), {
            "available_references": len(available), "scanned_references": len(scanned_ids),
            "selected_references": len(selected_ids), "per_survey_limit": reference_limit,
            "total_reference_budget": reference_budget, "reference_sample_size": reference_sample_size,
        }, utc_now())
        LOG.info("finished survey snowball survey=%s scanned=%d selected=%d coverage=%s", survey_id, len(scanned_ids), len(selected_ids), status)
    LOG.info(
        "scholarly collection complete candidate_papers=%d auxiliary_surveys=%d direct_candidates=%d new_snowball_candidates=%d reference_edges=%d",
        len(candidate_ids), len(survey_ids), len(selected_direct_ids), len(new_reference_ids), reference_edges,
    )
    selection_order = [work["native_id"] for work in ranked_direct if work["native_id"] in candidate_ids]
    selection_order.extend(sorted(candidate_ids - set(selection_order)))
    source_records = {}
    for key in candidate_ids:
        if key in client.records:
            payload = cache.get_record(f"{provider}.work", key)
            source_records[key] = replace(client.records[key], metadata={**client.records[key].metadata,
                **prefilter_paper(payload["work"], profile, settings,
                    len({relation for relation in payload.get("discovery_relations", []) if relation.startswith("survey_reference:")})),
                **{field: payload[field] for field in ("retrieval_rank_score", "retrieval_score", "retrieval_title_score", "retrieval_signals", "snowball_depth", "discovery_relations", "matched_queries") if field in payload}})
    cache.put_record(f"{provider}.collection", "latest", {
        "collection_mode": mode,
        "work_ids": sorted(candidate_ids), "selected_at": fetched_at, "collection_limit": limit,
        "classification_limit": settings.get("classification_limit"),
        "sample_size_per_query": query_sample_size, "survey_sample_size_per_query": survey_query_sample_size,
        "reference_sample_size": reference_sample_size,
        "query_limit_factor": settings.get("query_limit_factor", 3.0),
        "survey_query_limit_factor": settings.get("survey_query_limit_factor", settings.get("query_limit_factor", 3.0)),
        "reference_limit_factor": settings.get("reference_limit_factor", settings.get("query_limit_factor", 3.0)),
        "survey_ids": sorted(survey_ids), "direct_ids": sorted(selected_direct_ids),
        "survey_reference_ids": sorted(new_reference_ids),
        "snowball_seed_paper_ids": direct_seed_ids, "snowball_max_depth": max_depth,
        "snowball_reached_depth": reached_depth, "survey_publishers_covered": sorted(covered_publishers),
        "survey_publishers_missing": sorted(missing_publishers),
        "selection_order": selection_order, "queries": client.audits, "lookup_responses": client.lookups,
    }, fetched_at)
    counts = {
        "discovered": len(candidate_ids), "survey_sources": len(survey_ids), "auxiliary_cached_works": len(works) - len(candidate_ids),
        "direct_selected": len(selected_direct_ids), "survey_reference_works": len(new_reference_ids), "survey_reference_edges": reference_edges,
        "survey_results_scanned": sum(survey_scanned.values()), "direct_results_scanned": sum(direct_scanned.values()),
        "reference_results_scanned": len(reference_ids),
        "snowball_seed_papers": len(direct_seed_ids), "snowball_reached_depth": reached_depth,
        "survey_publishers_covered": len(covered_publishers), "survey_publishers_missing": len(missing_publishers),
    }
    return counts, source_records, client.memberships
