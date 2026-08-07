from __future__ import annotations

import os
import time
from typing import Any, Iterator

import requests

from ids_meta.common.cache import PipelineCache
from ids_meta.common.run import utc_now
from ids_meta.common.text import contains_concept
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
    """Return whether a work is explicitly identifiable as a survey or review."""
    text = f"{work.get('title', '')} {work.get('abstract', '')}"
    return str(work.get("work_type", "")).casefold() == "review" or any(
        contains_concept(text, term) for term in ("survey", "systematic review", "literature review", "dataset review", "mapping study")
    )


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
    progress_every = max(1, int(settings.get("progress_every", 25)))
    max_surveys = max(0, int(settings.get("max_surveys", 25)))
    reference_limit = max(0, int(settings.get("max_reference_works_per_survey", 300)))
    reference_total_limit = max(0, int(settings.get("max_survey_reference_candidates_total", 1000)))
    reference_fraction = max(0.0, min(0.9, float(settings.get("survey_reference_fraction", .35))))
    reference_budget = reference_total_limit if limit is None else min(reference_total_limit, int(limit * reference_fraction))
    survey_cap = max_surveys if limit is None else min(max_surveys, max(1, limit // 10))
    works: dict[str, dict[str, Any]] = {}
    relations: dict[str, set[str]] = {}
    queries: dict[str, set[str]] = {}
    survey_ids: list[str] = []
    fetched_at = utc_now()

    def store(normalized: dict[str, Any], relation: str, query: str = "") -> str:
        """Store one normalized work and merge its discovery provenance."""
        key = str(normalized.get("openalex_id", ""))
        if not key:
            return ""
        works[key] = normalized
        relations.setdefault(key, set()).add(relation)
        if query:
            queries.setdefault(key, set()).add(query)
        existing = cache.get_record("openalex.work", key) or {}
        cache.put_record("openalex.work", key, {
            "work": normalized,
            "discovery_relations": sorted(set(existing.get("discovery_relations", [])) | relations[key]),
            "matched_queries": sorted(set(existing.get("matched_queries", [])) | queries.get(key, set())),
        }, fetched_at)
        return key

    LOG.info(
        "starting OpenAlex collection direct_queries=%d survey_queries=%d limit=%s pacing=%.2fs survey_cap=%d reference_budget=%d",
        len(profile.get("direct_queries", [])), len(profile.get("survey_queries", [])), limit or "none",
        float(settings.get("request_pause_seconds", 0.5)), survey_cap, reference_budget,
    )

    # Find survey sources first so a small --limit cannot be consumed entirely by direct-search results.
    for query in profile.get("survey_queries", []):
        if survey_cap <= 0:
            break
        scanned = found = 0
        LOG.info("searching OpenAlex surveys query=%r", query)
        for item in client.search(str(query), per_page, pages):
            scanned += 1
            normalized = normalize_work(item)
            if not is_survey_work(normalized):
                continue
            key = store(normalized, "survey_search", str(query))
            if key and key not in survey_ids:
                survey_ids.append(key)
                found += 1
                LOG.info("found survey %d/%d: %s", len(survey_ids), survey_cap, normalized.get("title", key))
            if len(survey_ids) >= survey_cap:
                break
        bounded = scanned >= per_page * pages or len(survey_ids) >= survey_cap
        cache.put_coverage("openalex", str(query), "survey_cursor", "bounded" if bounded else "complete", scanned, {
            "configured_page_cap": pages, "surveys_found": found, "survey_cap": survey_cap,
        }, utc_now())
        LOG.info("finished survey query=%r scanned=%d surveys_found=%d total_surveys=%d", query, scanned, found, len(survey_ids))
        if len(survey_ids) >= survey_cap:
            break

    available_reference_ids = {
        str(reference_id)
        for survey_id in survey_ids
        for reference_id in (works.get(survey_id, {}).get("referenced_works") or [])[:reference_limit]
        if reference_id and str(reference_id) not in works
    }
    reference_budget = min(reference_budget, len(available_reference_ids))
    LOG.info("survey discovery complete surveys=%d distinct_reference_candidates=%d reserved_reference_budget=%d", len(survey_ids), len(available_reference_ids), reference_budget)
    direct_target = None if limit is None else max(len(works), limit - reference_budget)
    for query in profile.get("direct_queries", []):
        count = 0
        LOG.info("searching OpenAlex relation=direct_search query=%r", query)
        for item in client.search(str(query), per_page, pages):
            if direct_target is not None and len(works) >= direct_target:
                break
            store(normalize_work(item), "direct_search", str(query))
            count += 1
            if count % progress_every == 0:
                LOG.info("OpenAlex direct-query progress query=%r results=%d unique_works=%d", query, count, len(works))
            if direct_target is not None and len(works) >= direct_target:
                break
        bounded = count >= per_page * pages or (direct_target is not None and len(works) >= direct_target)
        cache.put_coverage("openalex", str(query), "direct_cursor", "bounded" if bounded else "complete", count, {
            "configured_page_cap": pages, "may_have_more_results": bounded,
        }, utc_now())
        LOG.info("finished direct query=%r results=%d unique_works=%d coverage=%s", query, count, len(works), "bounded" if bounded else "complete")
        if direct_target is not None and len(works) >= direct_target:
            break

    new_reference_ids: set[str] = set()
    survey_reference_ids: set[str] = set()
    reference_edges = 0
    LOG.info("starting reference snowball surveys=%d per_survey_limit=%d total_new_work_budget=%d", len(survey_ids), reference_limit, reference_budget)
    for survey_id in survey_ids:
        survey = works.get(survey_id, {})
        all_references = list(dict.fromkeys(str(value) for value in survey.get("referenced_works") or [] if value))
        references = all_references[:reference_limit]
        selected: list[str] = []
        for reference_id in references:
            if reference_id in works:
                selected.append(reference_id)
                continue
            if len(new_reference_ids) >= reference_budget or (limit is not None and len(works) + len([value for value in selected if value not in works]) >= limit):
                break
            selected.append(reference_id)
            new_reference_ids.add(reference_id)
        source_tag = f"survey_reference:{survey_id.rsplit('/', 1)[-1]}"
        unresolved: list[str] = []
        for reference_id in selected:
            cached = cache.get_record("openalex.work", reference_id)
            if reference_id in works:
                normalized = works[reference_id]
            elif cached and cached.get("work"):
                normalized = cached["work"]
                relations.setdefault(reference_id, set()).update(cached.get("discovery_relations", []))
                queries.setdefault(reference_id, set()).update(cached.get("matched_queries", []))
            else:
                unresolved.append(reference_id)
                continue
            key = store(normalized, "survey_reference")
            relations[key].add(source_tag)
            store(normalized, source_tag)
            survey_reference_ids.add(key)
            reference_edges += 1
        resolved_batch: set[str] = set()
        if unresolved:
            LOG.info("batch-resolving survey=%s unresolved_references=%d", survey_id, len(unresolved))
            try:
                for item in client.works_by_ids(unresolved):
                    normalized = normalize_work(item)
                    key = store(normalized, "survey_reference")
                    if key:
                        relations[key].add(source_tag)
                        store(normalized, source_tag)
                        resolved_batch.add(key)
                        survey_reference_ids.add(key)
                        reference_edges += 1
            except RuntimeError as exc:
                LOG.warning("reference batch failed survey=%s: %s", survey_id, exc)
        truncated = len(selected) < len(all_references)
        cache.put_coverage("openalex", f"references:{survey_id}", "reference_list", "bounded" if truncated else "complete", len(selected), {
            "available_references": len(all_references), "selected_references": len(selected),
            "resolved_references": len(selected) - len(unresolved) + len(resolved_batch),
            "unresolved_references": max(0, len(unresolved) - len(resolved_batch)),
            "per_survey_limit": reference_limit, "total_reference_budget": reference_budget,
        }, utc_now())
        LOG.info("finished survey snowball survey=%s selected=%d resolved=%d unique_works=%d", survey_id, len(selected), len(selected) - len(unresolved) + len(resolved_batch), len(works))
        if len(new_reference_ids) >= reference_budget or (limit is not None and len(works) >= limit):
            break
    LOG.info("OpenAlex collection complete discovered=%d survey_sources=%d survey_reference_works=%d reference_edges=%d", len(works), len(survey_ids), len(survey_reference_ids), reference_edges)
    return {"discovered": len(works), "survey_sources": len(survey_ids), "survey_reference_works": len(survey_reference_ids), "survey_reference_edges": reference_edges}
