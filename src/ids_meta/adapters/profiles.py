"""Shared structural validation; query syntax remains provider-owned."""
from __future__ import annotations

from datetime import date
from typing import Any
from urllib.parse import unquote

from ids_meta.adapters.contracts import ProviderQuery


def named_queries(profile: dict[str, Any], provider: str) -> list[ProviderQuery]:
    """Validate distinct names and text without assuming a search language."""
    if profile.get("schema_version") != "1.0" or profile.get("provider") != provider:
        raise ValueError(f"search requires schema_version 1.0 and provider {provider}")
    if profile.get("named_entity_seeding") is not False:
        raise ValueError("study search must explicitly disable named_entity_seeding")
    rows = profile.get("queries")
    if not isinstance(rows, list) or not rows or any(
        not isinstance(row, dict) or set(row) != {"name", "search"}
        or any(not isinstance(row.get(key), str) or not row[key].strip() for key in ("name", "search"))
        for row in rows
    ) or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("study search requires distinct named queries with name and search")
    return [ProviderQuery(row["name"], row["search"]) for row in rows]


def publication_window(profile: dict[str, Any]) -> tuple[str, str]:
    """Validate an explicit ISO date range shared by scholarly providers."""
    start, stop = profile.get("from_publication_date"), profile.get("to_publication_date")
    if not isinstance(start, str) or not isinstance(stop, str) or date.fromisoformat(start) > date.fromisoformat(stop):
        raise ValueError("study search requires an ordered ISO publication-date window")
    return start, stop


def strong_identifiers(doi: Any = "", arxiv: Any = "") -> dict[str, str]:
    """Normalize DOI and arXiv identifiers for conservative same-kind matching."""
    result = {}
    if doi:
        value = unquote(str(doi).strip()).lower()
        for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
            value = value.removeprefix(prefix)
        if not value.startswith("10.") or "/" not in value:
            raise ValueError(f"invalid DOI: {doi!r}")
        result["doi"] = value
    if arxiv:
        import re
        value = str(arxiv).strip().removeprefix("https://arxiv.org/abs/").removeprefix("arXiv:")
        result["arxiv"] = re.sub(r"v\d+$", "", value)
    return result


def validated_page_size(value: Any, maximum: int) -> int:
    """Reject invalid configured page sizes before any request."""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"per_page must be an integer between 1 and {maximum}")
    return value


def validate_query_fields(query: ProviderQuery, required: set[str]) -> None:
    """Validate the structural query envelope without interpreting its text."""
    if any(not isinstance(value, str) or not value.strip() for value in (query.name, query.text)):
        raise ValueError("provider query requires a nonempty name and text")
    if not isinstance(query.parameters, dict) or set(query.parameters) != required:
        raise ValueError("unsupported provider query parameters")
