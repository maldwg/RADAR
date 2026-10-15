"""GitHub transport and repository normalization with explicit search bounds."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from radar.adapters.contracts import ProviderDescriptor, ProviderQuery, SearchPage, SourceRecord
from radar.adapters.profiles import named_queries, validated_page_size, validate_query_fields
from radar.github.collect import GitHubClient, collect

DESCRIPTOR = ProviderDescriptor("github", ("repository",), ("discovery",), ("repository_catalogue",))


class GitHubAdapter:
    """Expose repository discovery while retaining date sharding in the catalogue strategy."""
    descriptor = DESCRIPTOR

    def __init__(self, settings: dict[str, Any], profile_path: Path, client: Any = None):
        """Bind settings and defer client construction until a request."""
        self.settings, self.profile_path, self.client = settings, profile_path, client
        validated_page_size(settings.get("per_page", 100), 100)

    def _client(self) -> GitHubClient:
        """Construct the existing adaptive rate-limit-aware GitHub client."""
        if self.client is None:
            self.client = GitHubClient(os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN"),
                request_pause_seconds=float(self.settings.get("request_pause_seconds", 1)),
                search_pause_seconds=float(self.settings.get("search_pause_seconds", 2.2)),
                rate_limit_max_wait_seconds=float(self.settings.get("rate_limit_max_wait_seconds", 3700)),
                rate_limit_fallback_wait_seconds=float(self.settings.get("rate_limit_fallback_wait_seconds", 60)),
                adaptive_rate_limit_pacing=bool(self.settings.get("adaptive_rate_limit_pacing", True)))
        return self.client

    def queries(self, profile: dict[str, Any]) -> list[ProviderQuery]:
        """Accept native GitHub qualifiers without scholarly query translation."""
        return named_queries(profile, self.descriptor.name)

    def search(self, query: ProviderQuery, cursor: str | None, page_size: int) -> SearchPage:
        """Expose API pagination and mark the 1,000-result ceiling as incomplete."""
        self.validate_query(query)
        page_number = int(cursor or "1")
        size = validated_page_size(page_size, 100)
        payload = self._client().search(query.text, page=page_number, per_page=size)
        items, total = payload.get("items"), payload.get("total_count")
        if not isinstance(items, list) or not isinstance(total, int):
            raise ValueError("GitHub returned an invalid search page")
        consumed = page_number * size
        more = bool(items) and consumed < min(total, 1000)
        return SearchPage(items, str(page_number + 1) if more else None, payload, total,
            not payload.get("incomplete_results", False) and total <= 1000)

    def validate_query(self, query: ProviderQuery) -> None:
        """Require a native GitHub search without foreign query parameters."""
        validate_query_fields(query, set())

    def normalize(self, raw: dict[str, Any]) -> SourceRecord:
        """Normalize stable repository identity and discovery metadata."""
        native = raw.get("id")
        if not isinstance(native, int) or isinstance(native, bool):
            raise ValueError("GitHub result lacks a numeric repository identity")
        return SourceRecord("github", str(native), "repository", {},
            {"github_repo_id": str(native), "full_name": raw.get("full_name", ""),
             "title": raw.get("full_name", ""), "primary_url": raw.get("html_url", "")},
            {"abstract": raw.get("description") or ""}, raw)

    def collect_catalogue(self, settings: dict[str, Any], profile: dict[str, Any], cache: Any,
                          limit: int | None, *, resume: bool = False) -> dict[str, int]:
        """Keep date sharding, native evidence and cache semantics in one strategy."""
        return collect(settings, profile, cache, limit, client=self.client, resume=resume)
