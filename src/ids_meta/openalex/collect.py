from __future__ import annotations

from typing import Any, Iterator

import requests

from ids_meta.adapters.http import JsonTransport
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
        "publisher": source.get("host_organization_name") or source.get("publisher") or "",
        "publication_year": item.get("publication_year") or "", "publication_date": item.get("publication_date") or "",
        "work_type": item.get("type") or "", "primary_url": primary.get("landing_page_url") or item.get("doi") or item.get("id") or "",
        "pdf_url": best_oa.get("pdf_url") or primary.get("pdf_url") or "", "is_oa": str(bool(open_access.get("is_oa"))).lower(),
        "cited_by_count": int(item.get("cited_by_count") or 0), "referenced_works": item.get("referenced_works") or [],
    }


class OpenAlexClient:
    api_root = "https://api.openalex.org"
    select = "id,doi,title,display_name,publication_year,publication_date,cited_by_count,abstract_inverted_index,primary_location,best_oa_location,open_access,authorships,referenced_works,type"

    def __init__(self, api_key: str | None = None, mailto: str | None = None, timeout: float = 30, request_pause_seconds: float = 0.5):
        """Initialize an OpenAlex client with credentials, timeout, and pacing."""
        self.api_key, self.mailto, self.timeout = api_key, mailto, timeout
        self.transport = JsonTransport(timeout=timeout, pause=max(0.0, request_pause_seconds), retries=4)
        self.transport.session = requests.Session()

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Add provider credentials and use shared bounded JSON retries and pacing."""
        values = dict(params or {})
        if self.api_key:
            values["api_key"] = self.api_key
        if self.mailto:
            values["mailto"] = self.mailto
        try:
            return self.transport.get(path if path.startswith("http") else self.api_root + path, values)
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenAlex request failed for {path}") from exc

    def search(self, query: str, per_page: int, max_pages: int | None) -> Iterator[dict[str, Any]]:
        """Yield works from a cursor-paginated OpenAlex full-text search."""
        cursor = "*"
        page = 0
        seen_cursors: set[str] = set()
        while max_pages is None or page < max_pages:
            if cursor in seen_cursors:
                raise RuntimeError(f"OpenAlex repeated search cursor for {query!r}")
            seen_cursors.add(cursor)
            page += 1
            payload = self.get("/works", {"search": query, "per-page": per_page, "cursor": cursor, "select": self.select})
            LOG.debug("OpenAlex query=%r page=%d results=%d", query, page, len(payload.get("results", [])))
            for item in payload.get("results", []):
                yield item
            cursor = str((payload.get("meta") or {}).get("next_cursor") or "")
            if not cursor or not payload.get("results"):
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
