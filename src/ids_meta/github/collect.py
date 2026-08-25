from __future__ import annotations

import base64
import math
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterator

import requests

from ids_meta.common.cache import PipelineCache
from ids_meta.common.run import utc_now
from ids_meta.common.logging import logger

LOG = logger("github.collect")


class GitHubClient:
    api_root = "https://api.github.com"

    def __init__(
        self, token: str | None = None, timeout: float = 30, request_pause_seconds: float = 1.0,
        search_pause_seconds: float = 2.2, rate_limit_max_wait_seconds: float = 3700,
        rate_limit_fallback_wait_seconds: float = 60, adaptive_rate_limit_pacing: bool = True,
    ):
        """Configure an authenticated, paced GitHub API session."""
        self.timeout = timeout
        self.authenticated = bool(token)
        self.request_pause_seconds = max(0.0, request_pause_seconds)
        self.search_pause_seconds = max(self.request_pause_seconds, search_pause_seconds)
        self.rate_limit_max_wait_seconds = max(0.0, rate_limit_max_wait_seconds)
        self.rate_limit_fallback_wait_seconds = max(1.0, rate_limit_fallback_wait_seconds)
        self.adaptive_rate_limit_pacing = adaptive_rate_limit_pacing
        self._adaptive_core_pause = 0.0
        self._adaptive_search_pause = 0.0
        self._last_request_at: float | None = None
        self._last_search_at: float | None = None
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    @staticmethod
    def _header_int(response: requests.Response, name: str) -> int | None:
        """Parse an integer response header, returning ``None`` when invalid."""
        try:
            value = response.headers.get(name)
            return int(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    def _rate_limit_wait(self, response: requests.Response, path: str) -> None:
        """Wait for a reported GitHub quota reset or raise when the wait is excessive."""
        now = time.time()
        reset = self._header_int(response, "X-RateLimit-Reset")
        retry_after = self._header_int(response, "Retry-After")
        if retry_after is not None:
            wait = max(1.0, float(retry_after))
        elif reset is not None and reset > now:
            wait = max(1.0, reset - now + 2)
        else:
            wait = self.rate_limit_fallback_wait_seconds
        reset_at = datetime.fromtimestamp(reset, timezone.utc).isoformat().replace("+00:00", "Z") if reset else "unknown"
        resource = response.headers.get("X-RateLimit-Resource", "unknown")
        limit = response.headers.get("X-RateLimit-Limit", "unknown")
        remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
        used = response.headers.get("X-RateLimit-Used", "unknown")
        auth_mode = "token" if self.authenticated else "anonymous"
        if wait > self.rate_limit_max_wait_seconds:
            raise RuntimeError(
                f"GitHub {resource} rate limit exceeded for {path}; reset={reset_at}, wait={wait:.0f}s, "
                f"limit={limit}, remaining={remaining}, used={used}, authentication={auth_mode}. "
                "Increase github.rate_limit_max_wait_seconds or resume after reset."
            )
        LOG.warning(
            "GitHub %s rate limit reached for %s; waiting %.0f seconds until reset=%s "
            "(limit=%s remaining=%s used=%s authentication=%s)",
            resource, path, wait, reset_at, limit, remaining, used, auth_mode,
        )
        if not self.authenticated:
            LOG.warning("GitHub requests are anonymous; set GITHUB_TOKEN or GH_TOKEN for the authenticated API allowance")
        time.sleep(wait)

    def _observe_rate_limit(self, response: requests.Response) -> None:
        """Adapt request spacing to the remaining GitHub quota window."""
        if not self.adaptive_rate_limit_pacing:
            return
        remaining = self._header_int(response, "X-RateLimit-Remaining")
        reset = self._header_int(response, "X-RateLimit-Reset")
        if remaining is None or reset is None or remaining <= 0:
            return
        # Spread the remaining requests across the remaining quota window instead of bursting into the limit.
        suggested = max(0.0, (reset - time.time() + 2) / remaining)
        resource = response.headers.get("X-RateLimit-Resource", "core")
        if resource == "search":
            self._adaptive_search_pause = suggested
        else:
            self._adaptive_core_pause = suggested
        LOG.debug(
            "GitHub quota pacing resource=%s remaining=%d reset_in=%.0fs suggested_interval=%.2fs",
            resource, remaining, max(0.0, reset - time.time()), suggested,
        )

    def _wait_before_request(self, search_request: bool) -> None:
        """Enforce configured and quota-derived spacing before an API call."""
        now = time.monotonic()
        waits: list[float] = []
        if self._last_request_at is not None:
            adaptive = self._adaptive_search_pause if search_request else self._adaptive_core_pause
            waits.append(max(self.request_pause_seconds, adaptive) - (now - self._last_request_at))
        if search_request and self._last_search_at is not None:
            waits.append(max(self.search_pause_seconds, self._adaptive_search_pause) - (now - self._last_search_at))
        wait = max([0.0, *waits])
        if wait:
            LOG.debug("pacing GitHub API call; sleeping %.2f seconds", wait)
            time.sleep(wait)
        requested_at = time.monotonic()
        self._last_request_at = requested_at
        if search_request:
            self._last_search_at = requested_at

    def request(self, path: str, params: dict[str, Any] | None = None, accept: str | None = None) -> requests.Response:
        """Perform a GitHub request with pacing, rate-limit handling, and retries."""
        headers = {"Accept": accept} if accept else None
        last: Exception | None = None
        attempt = 0
        rate_limit_events = 0
        while attempt < 5:
            try:
                self._wait_before_request(path.startswith("/search/") or "/search/" in path)
                response = self.session.get(path if path.startswith("http") else self.api_root + path, params=params, headers=headers, timeout=self.timeout)
                rate_limited = response.status_code in {403, 429} and (
                    response.headers.get("X-RateLimit-Remaining") == "0"
                    or response.headers.get("Retry-After") is not None
                    or "rate limit" in response.text.casefold()
                    or "abuse detection" in response.text.casefold()
                )
                if rate_limited:
                    rate_limit_events += 1
                    if rate_limit_events > 3:
                        raise RuntimeError(f"GitHub continued returning rate-limit responses for {path} after waiting for reset")
                    self._rate_limit_wait(response, path)
                    # A primary/secondary limit wait is not a transient-request retry.
                    continue
                if response.status_code >= 500:
                    attempt += 1
                    last = RuntimeError(f"GitHub HTTP {response.status_code} for {path}")
                    LOG.warning("GitHub HTTP %d for %s; retrying attempt=%d/5", response.status_code, path, attempt)
                    time.sleep(2**attempt)
                    continue
                if 400 <= response.status_code < 500:
                    raise RuntimeError(f"GitHub HTTP {response.status_code} for {path}: {response.text[:300]}")
                response.raise_for_status()
                self._observe_rate_limit(response)
                return response
            except requests.RequestException as exc:
                last = exc
                attempt += 1
                if attempt < 5:
                    LOG.warning("GitHub request error for %s (%s); retrying attempt=%d/5", path, exc, attempt)
                    time.sleep(2**attempt)
        raise RuntimeError(f"GitHub request failed for {path}: {last}") from last

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Request a GitHub endpoint and decode its JSON response."""
        return self.request(path, params).json()

    def search(self, query: str, page: int = 1, per_page: int = 100) -> dict[str, Any]:
        """Search GitHub repositories using the configured result ordering."""
        return self.get_json("/search/repositories", {"q": query, "page": page, "per_page": per_page})

    def raw_readme(self, full_name: str) -> str:
        """Fetch a repository README as raw text, returning blank on failure."""
        try:
            return self.request(f"/repos/{full_name}/readme", accept="application/vnd.github.raw+json").text
        except RuntimeError:
            return ""

    def docs(self, full_name: str, max_files: int) -> str:
        """Fetch a bounded set of textual files from a repository's docs directory."""
        if max_files <= 0:
            return ""
        try:
            entries = self.get_json(f"/repos/{full_name}/contents/docs")
        except RuntimeError:
            return ""
        texts: list[str] = []
        for entry in entries if isinstance(entries, list) else []:
            if len(texts) >= max_files:
                break
            if entry.get("type") != "file" or not str(entry.get("name", "")).casefold().endswith((".md", ".rst", ".txt")):
                continue
            try:
                item = self.get_json(str(entry.get("url")))
                content = base64.b64decode(item.get("content", "")).decode("utf-8", errors="replace")
                texts.append(f"# {entry.get('path')}\n{content}")
            except (RuntimeError, ValueError):
                continue
        return "\n\n".join(texts)


def _split_range(start: date, end: date) -> tuple[tuple[date, date], tuple[date, date]]:
    """Split an inclusive date range into two non-overlapping halves."""
    middle = start + (end - start) // 2
    return (start, middle), (middle + timedelta(days=1), end)


def search_shards(client: GitHubClient, base_query: str, start: date, end: date, per_page: int) -> Iterator[tuple[str, dict[str, Any], bool]]:
    """Recursively shard and page a search below GitHub's 1,000-result cap."""
    shard = f"{start.isoformat()}..{end.isoformat()}"
    query = f"{base_query} created:{shard}"
    first = client.search(query, 1, per_page)
    total = int(first.get("total_count") or 0)
    LOG.debug("query=%r shard=%s total=%d", base_query, shard, total)
    if total > 1000 and start < end:
        LOG.info("query=%r shard=%s exceeds GitHub's 1,000-result cap; splitting date range", base_query, shard)
        left, right = _split_range(start, end)
        yield from search_shards(client, base_query, *right, per_page)
        yield from search_shards(client, base_query, *left, per_page)
        return
    incomplete = bool(first.get("incomplete_results")) or total > 1000
    yield shard, first, incomplete
    pages = min(10, math.ceil(min(total, 1000) / per_page))
    for page in range(2, pages + 1):
        yield shard, client.search(query, page, per_page), incomplete


def search_best_matches(client: GitHubClient, query: str, per_page: int) -> Iterator[tuple[str, dict[str, Any], bool]]:
    """Page GitHub's native best-match results without introducing a date bias."""
    first = client.search(query, 1, per_page)
    total = int(first.get("total_count") or 0)
    incomplete = bool(first.get("incomplete_results"))
    yield "all", first, incomplete
    pages = min(10, math.ceil(min(total, 1000) / per_page))
    for page in range(2, pages + 1):
        yield "all", client.search(query, page, per_page), incomplete


def _enrich(client: GitHubClient, item: dict[str, Any], max_docs_files: int) -> dict[str, Any]:
    """Fetch repository metadata, README, documentation, fork, and commit evidence."""
    full_name = item.get("full_name", "")
    LOG.debug("fetching repository evidence for %s", full_name)
    repo = client.get_json(f"/repos/{full_name}")
    readme = client.raw_readme(full_name)
    docs = client.docs(full_name, max_docs_files)
    commit_sha = ""
    last_commit_at = ""
    last_commit_url = ""
    try:
        commits = client.get_json(f"/repos/{full_name}/commits", {"per_page": 1})
        if isinstance(commits, list) and commits:
            latest = commits[0]
            commit_sha = str(latest.get("sha", ""))
            commit = latest.get("commit") or {}
            last_commit_at = str((commit.get("committer") or {}).get("date") or (commit.get("author") or {}).get("date") or "")
            last_commit_url = str(latest.get("html_url") or "")
    except RuntimeError:
        pass
    parent = repo.get("parent") or {}
    return {
        "repository": repo, "readme_text": readme, "docs_text": docs,
        "last_commit_sha": commit_sha, "last_commit_at": last_commit_at, "last_commit_url": last_commit_url,
        "parent_full_name": parent.get("full_name", ""),
    }


def collect(settings: dict[str, Any], profile: dict[str, Any], cache: PipelineCache, limit: int | None = None, client: GitHubClient | None = None, resume: bool = True) -> dict[str, int]:
    """Discover, deduplicate, enrich, cache, and report GitHub candidates."""
    client = client or GitHubClient(
        os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
        request_pause_seconds=float(settings.get("request_pause_seconds", 1.0)),
        search_pause_seconds=float(settings.get("search_pause_seconds", 2.2)),
        rate_limit_max_wait_seconds=float(settings.get("rate_limit_max_wait_seconds", 3700)),
        rate_limit_fallback_wait_seconds=float(settings.get("rate_limit_fallback_wait_seconds", 60)),
        adaptive_rate_limit_pacing=bool(settings.get("adaptive_rate_limit_pacing", True)),
    )
    if not client.authenticated:
        LOG.warning("GitHub collection is running anonymously; enrichment will be very slow. Set GITHUB_TOKEN or GH_TOKEN before running the CLI")
    per_page = int(settings.get("per_page", 100))
    start = date.fromisoformat(str(settings.get("created_from", "2008-01-01")))
    end = datetime.now(timezone.utc).date()
    minimum_stars = max(0, int(settings.get("minimum_stars", 100)))
    query_sample_size = max(1, int(settings.get("query_sample_size", 100)))
    native_search = query_sample_size <= 1000
    strategy_kind = "native_best_match" if native_search else "date_shards"
    search_strategy = f"github_{strategy_kind}_v3_stars_{minimum_stars}_cap_{query_sample_size}"
    seen: dict[str, dict[str, Any]] = {}
    matched: dict[str, set[str]] = {}
    best_match_rank: dict[str, int] = {}
    if resume:
        for key, payload, _cached_at in cache.records("github.search_candidate"):
            if payload.get("item") and payload.get("search_strategy") == search_strategy:
                seen[key] = payload["item"]
                matched[key] = set(payload.get("matched_queries", []))
                best_match_rank[key] = int(payload.get("best_match_rank") or 10**9)
    fetched_at = utc_now()
    incomplete_shards: set[tuple[str, str]] = set()
    queries = [str(query) for query in profile.get("search_queries", [])]
    scan_target = len(queries) * query_sample_size
    qualified_queries = {query: f"{query} stars:>={minimum_stars}" for query in queries}
    LOG.info(
        "starting GitHub collection queries=%d resume=%s sample_per_query=%d select_best=%s minimum_stars=%d cached_search_candidates=%d ordering=github_best_match pacing=%.2fs/search=%.2fs authentication=%s",
        len(queries), resume, query_sample_size, limit or "all", minimum_stars, len(seen),
        float(settings.get("request_pause_seconds", 1.0)), float(settings.get("search_pause_seconds", 2.2)),
        "token" if client.authenticated else "anonymous",
    )
    if not (resume and limit is not None and len(seen) >= limit):
        if native_search:
            iterators = {query: iter(search_best_matches(client, qualified_queries[query], per_page)) for query in queries}
        else:
            iterators = {query: iter(search_shards(client, qualified_queries[query], start, end, per_page)) for query in queries}
        active = list(queries)
        query_positions = {query: 0 for query in queries}
        per_query_target = query_sample_size
        while active:
            for query in list(active):
                try:
                    shard, page, incomplete = next(iterators[query])
                except StopIteration:
                    active.remove(query)
                    continue
                if incomplete:
                    incomplete_shards.add((query, shard))
                cache.put_coverage("github", query, shard, "incomplete" if incomplete else "complete", int(page.get("total_count") or 0), {"incomplete_results": bool(page.get("incomplete_results")), "ordering": "best_match"}, utc_now())
                for item in page.get("items", []):
                    key = str(item.get("id") or "")
                    if not key:
                        continue
                    query_positions[query] += 1
                    seen[key] = item
                    matched.setdefault(key, set()).add(query)
                    best_match_rank[key] = min(best_match_rank.get(key, 10**9), query_positions[query])
                    cache.put_record("github.search_candidate", key, {
                        "item": item, "matched_queries": sorted(matched[key]),
                        "best_match_rank": best_match_rank[key],
                        "search_strategy": search_strategy,
                    }, fetched_at)
                    if per_query_target is not None and query_positions[query] >= per_query_target:
                        if query in active:
                            active.remove(query)
                        break
    ranked_ids = sorted(seen, key=lambda key: (
        best_match_rank.get(key, 10**9), -len(matched.get(key, set())), key,
    ))
    selected_ids = ranked_ids if limit is None else ranked_ids[:max(0, limit)]
    seen = {key: seen[key] for key in selected_ids}
    matched = {key: matched.get(key, set()) for key in selected_ids}
    cache.put_record("github.collection", "latest", {
        "repository_ids": selected_ids, "selected_at": fetched_at, "collection_limit": limit,
        "ordering": "github_best_match", "sample_size_per_query": query_sample_size,
        "selection_size": limit, "search_strategy": search_strategy,
    }, fetched_at)
    progress_every = max(1, int(settings.get("progress_every", 25)))
    reused = 0
    for index, (key, item) in enumerate(seen.items()):
        existing = cache.get_record("github.repository", key)
        if existing and resume:
            existing["matched_queries"] = sorted(set(existing.get("matched_queries", [])) | matched[key])
            cache.put_record("github.repository", key, existing, fetched_at)
            reused += 1
            if (index + 1) % progress_every == 0 or index + 1 == len(seen):
                LOG.info("GitHub enrichment progress %d/%d (cache_reused=%d)", index + 1, len(seen), reused)
            continue
        LOG.info("enriching repository %d/%d: %s", index + 1, len(seen), item.get("full_name", key))
        payload = _enrich(client, item, int(settings.get("max_docs_files", 3)))
        payload["matched_queries"] = sorted(matched[key])
        cache.put_record("github.repository", key, payload, fetched_at)
        LOG.info("finished repository %d/%d: %s", index + 1, len(seen), item.get("full_name", key))
    LOG.info("GitHub collection complete discovered=%d reused=%d incomplete_shards=%d", len(seen), reused, len(incomplete_shards))
    return {"discovered": len(seen), "cache_reused": reused, "incomplete_shards": len(incomplete_shards)}
