"""Semantic Scholar discovery and lookup through its Academic Graph API."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from radar.adapters.contracts import ProviderDescriptor, ProviderQuery, SearchPage, SourceRecord
from radar.adapters.http import JsonTransport
from radar.adapters.profiles import identifiers_conflict, named_queries, paper_identifiers, publication_window, strong_identifiers, validated_page_size, validate_query_fields
from radar.common.retrieval_diagnostics import lookup_diagnostics, lookup_failure
from radar.common.pdf import download_pdf_text
from radar.common.text import normalize_text

DESCRIPTOR = ProviderDescriptor("semantic_scholar", ("paper",), ("discovery", "lookup", "evidence"), default_page_size=1000, evidence_formats=("pdf",))


class SemanticScholarAdapter:
    """Use explicit bulk-search syntax and token pagination, without named seeds."""
    descriptor = DESCRIPTOR
    api_root = "https://api.semanticscholar.org/graph/v1"
    fields = "title,abstract,externalIds,url,openAccessPdf,year,publicationDate,authors,venue,citationCount"

    def __init__(self, settings: dict[str, Any], profile_path: Path, client: Any = None):
        """Configure API-key authentication independently of any study."""
        self.settings, self.profile_path = settings, profile_path
        validated_page_size(settings.get("per_page", 1000), 1000)
        headers = {"User-Agent": "radar/0.1"}
        if os.getenv("SEMANTIC_SCHOLAR_API_KEY"):
            headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
        self.client = client or JsonTransport(headers=headers,
            timeout=float(settings.get("timeout_seconds", 30)),
            pause=float(settings.get("request_pause_seconds", 1)))

    def queries(self, profile: dict[str, Any]) -> list[ProviderQuery]:
        """Require the native bulk syntax instead of rewriting another provider's query."""
        queries = named_queries(profile, self.descriptor.name)
        if profile.get("search_parameter") != "bulk":
            raise ValueError("Semantic Scholar search_parameter must be bulk")
        start, stop = publication_window(profile)
        return [ProviderQuery(q.name, q.text, {"publicationDateOrYear": f"{start}:{stop}"}) for q in queries]

    def search(self, query: ProviderQuery, cursor: str | None, page_size: int) -> SearchPage:
        """Retrieve up to 1,000 records; the runner applies any smaller study cap."""
        self.validate_query(query)
        validated_page_size(page_size, 1000)
        params = {"query": query.text, "fields": self.fields, **query.parameters}
        if cursor is not None:
            params["token"] = cursor
        payload = self.client.get(f"{self.api_root}/paper/search/bulk", params)
        if not isinstance(payload.get("data"), list):
            raise ValueError("Semantic Scholar returned an invalid search page")
        return SearchPage(payload["data"], str(payload["token"]) if payload.get("token") else None,
            payload, payload.get("total"))

    def validate_query(self, query: ProviderQuery) -> None:
        """Require a native bulk query and supported scholarly date window."""
        validate_query_fields(query, {"publicationDateOrYear"})
        window = query.parameters["publicationDateOrYear"]
        if not isinstance(window, str) or len(window.split(":")) != 2:
            raise ValueError("Semantic Scholar requires a publication date window")
        start, stop = window.split(":")
        publication_window({"from_publication_date": start, "to_publication_date": stop})

    def normalize(self, raw: dict[str, Any]) -> SourceRecord:
        """Retain native identity, DOI/arXiv, metadata and available abstract."""
        native = raw.get("paperId")
        if not isinstance(native, str) or not native.strip():
            raise ValueError("Semantic Scholar result lacks paperId")
        external = raw.get("externalIds") or {}
        pdf = raw.get("openAccessPdf") or {}
        identifiers = strong_identifiers(external.get("DOI"), external.get("ArXiv"))
        return SourceRecord("semantic_scholar", native, "paper", identifiers,
            {"semantic_scholar_id": native, "title": raw.get("title") or "",
             "doi": identifiers.get("doi", ""), "publication_year": raw.get("year") or "",
             "publication_date": raw.get("publicationDate") or "", "venue": raw.get("venue") or "",
             "authors": ";".join(a.get("name", "") for a in raw.get("authors") or []),
             "primary_url": raw.get("url") or "", "pdf_url": pdf.get("url") or "",
             "cited_by_count": raw.get("citationCount") or 0},
            {"abstract": raw.get("abstract") or ""}, raw)

    def lookup(self, identifier: str) -> SourceRecord:
        """Resolve a paper ID, DOI-prefixed ID or another supported external ID."""
        return self.normalize(self.client.get(f"{self.api_root}/paper/{quote(identifier, safe=':')}",
            {"fields": self.fields}))

    def enrich(self, work: dict[str, Any]) -> dict[str, Any]:
        """Find missing evidence by DOI or exact title, using shared bounded PDF extraction."""
        title = str(work.get('title') or '').strip()
        identifiers = paper_identifiers(work)
        record = None
        attempts = []
        for key, prefix in (('doi', 'DOI:'), ('arxiv', 'ARXIV:')):
            if not identifiers.get(key):
                continue
            try:
                candidate = self.lookup(prefix + identifiers[key])
                conflict = identifiers_conflict(identifiers, candidate.identifiers)
                attempts.append({'method': 'identifier', 'identifier': prefix + identifiers[key], 'status': 'identity_conflict' if conflict else 'matched'})
                if not conflict:
                    record = candidate
                    break
            except (requests.RequestException, ValueError) as exc:
                attempts.append({'method': 'identifier', 'identifier': prefix + identifiers[key], **lookup_failure(exc)})
        if record is None and title:
            try:
                payload = self.client.get(f"{self.api_root}/paper/search",
                    {"query": title, "limit": 10, "fields": self.fields})
                matches = [raw for raw in payload.get("data", [])
                    if normalize_text(raw.get("title")) == normalize_text(title)]
                if len(matches) == 1:
                    candidate = self.normalize(matches[0])
                    conflict = identifiers_conflict(identifiers, candidate.identifiers)
                    attempts.append({'method': 'title', 'status': 'identity_conflict' if conflict else 'matched'})
                    if not conflict:
                        record = candidate
                else:
                    attempts.append({'method': 'title', 'status': 'no_match' if not matches else 'ambiguous'})
            except (requests.RequestException, ValueError) as exc:
                attempts.append({'method': 'title', **lookup_failure(exc)})
        if record is None:
            return lookup_diagnostics(attempts, resolved=False)
        url = record.metadata.get("pdf_url", "")
        maximum = int(self.settings.get("max_full_text_chars", 30000))
        result = {"abstract": record.evidence.get("abstract", ""), "abstract_source": "Semantic Scholar" if record.evidence.get("abstract") else "",
            "full_text_url": url, "evidence_provider": "semantic_scholar", "evidence_native_id": record.native_id,
            "evidence_identifiers": record.identifiers, **lookup_diagnostics(attempts, resolved=True)}
        if url and maximum:
            result.update(download_pdf_text(url, maximum, float(self.settings.get("timeout_seconds", 30))))
        return {key: value for key, value in result.items() if value != ""}
