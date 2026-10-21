"""OpenAlex operations, independent of dataset and treatment-effect protocols."""
from __future__ import annotations

import os
import re

import requests
from pathlib import Path
from typing import Any

from radar.adapters.contracts import ProviderDescriptor, ProviderQuery, SearchPage, SourceRecord
from radar.adapters.profiles import identifiers_conflict, named_queries, paper_identifiers, publication_window, strong_identifiers, validated_page_size, validate_query_fields
from radar.openalex.collect import OpenAlexClient, normalize_work
from radar.common.retrieval_diagnostics import lookup_diagnostics, lookup_failure
from radar.common.paper_collection import collect_papers

DESCRIPTOR = ProviderDescriptor("openalex", ("paper",), ("discovery", "lookup", "references", "evidence"), ("dataset_catalogue",), 200, evidence_formats=("tei_xml", "pdf"))


class OpenAlexAdapter:
    """Use the existing rate-aware client behind the shared discovery interface."""
    descriptor = DESCRIPTOR

    def __init__(self, settings: dict[str, Any], profile_path: Path, client: Any = None):
        """Bind provider settings and allow an injected offline client."""
        self.settings, self.profile_path, self.client = settings, profile_path, client
        validated_page_size(settings.get("per_page", 200), 200)

    def _client(self) -> OpenAlexClient:
        """Construct transport only when an operation actually requires it."""
        if self.client is None:
            self.client = OpenAlexClient(os.getenv("OPENALEX_API_KEY"), os.getenv("OPENALEX_MAILTO"),
                timeout=float(self.settings.get("timeout_seconds", 30)),
                request_pause_seconds=float(self.settings.get("request_pause_seconds", 1)))
        return self.client

    def queries(self, profile: dict[str, Any]) -> list[ProviderQuery]:
        """Keep OpenAlex search parameters and filters explicit."""
        queries = named_queries(profile, self.descriptor.name)
        parameter = profile.get("search_parameter", "search")
        if parameter not in {"search", "search.exact", "search.title_and_abstract", "search.title_and_abstract.exact"}:
            raise ValueError("unsupported OpenAlex search_parameter")
        start, stop = publication_window(profile)
        return [ProviderQuery(q.name, q.text, {"search_parameter": parameter,
            "filter": f"from_publication_date:{start},to_publication_date:{stop}"}) for q in queries]

    def search(self, query: ProviderQuery, cursor: str | None, page_size: int) -> SearchPage:
        """Retrieve a cursor page and preserve the complete native response."""
        self.validate_query(query)
        client = self._client()
        payload = client.get("/works", {query.parameters["search_parameter"]: query.text,
            "cursor": cursor or "*", "per-page": validated_page_size(page_size, 200), "select": client.select,
            "filter": query.parameters["filter"]})
        items, meta = payload.get("results"), payload.get("meta", {})
        if not isinstance(items, list) or not isinstance(meta, dict):
            raise ValueError("OpenAlex returned an invalid search page")
        return SearchPage(items, str(meta["next_cursor"]) if meta.get("next_cursor") else None,
            payload, meta.get("count"))

    def validate_query(self, query: ProviderQuery) -> None:
        """Require the declared OpenAlex parameter and publication filter."""
        validate_query_fields(query, {"search_parameter", "filter"})
        if query.parameters["search_parameter"] not in {"search", "search.exact", "search.title_and_abstract", "search.title_and_abstract.exact"}:
            raise ValueError("unsupported OpenAlex search_parameter")
        if not isinstance(query.parameters["filter"], str) or not query.parameters["filter"]:
            raise ValueError("OpenAlex query requires a filter")

    def normalize(self, raw: dict[str, Any]) -> SourceRecord:
        """Normalize metadata and abstract without assigning scientific labels."""
        work = normalize_work(raw)
        native = work["openalex_id"]
        if not isinstance(native, str) or not native.strip():
            raise ValueError("OpenAlex search result lacks an identity")
        abstract = work.pop("abstract")
        return SourceRecord(self.descriptor.name, native, "paper", strong_identifiers(work.get("doi")),
            work, {"abstract": abstract}, raw)

    def lookup(self, identifier: str) -> SourceRecord:
        """Resolve one OpenAlex work or supported external identifier."""
        return self.normalize(self._client().work(identifier))

    def enrich(self, work: dict[str, Any]) -> dict[str, Any]:
        """Resolve this paper's cached full text and all alternate PDF locations."""
        identifiers = paper_identifiers(work)
        candidates = []
        native = str(work.get('openalex_id') or '')
        if re.fullmatch(r'(?:https?://openalex.org/)?W[0-9]+', native, re.I):
            candidates.append(native)
        if identifiers.get('doi'):
            candidates.append('doi:' + identifiers['doi'])
        record = None
        attempts = []
        for identifier in candidates:
            try:
                candidate = self.lookup(identifier)
                conflict = identifiers_conflict(identifiers, paper_identifiers(candidate.metadata))
                attempts.append({'method': 'identifier', 'identifier': identifier, 'status': 'identity_conflict' if conflict else 'matched'})
                if not conflict:
                    record = candidate
                    break
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                attempts.append({'method': 'identifier', 'identifier': identifier, **lookup_failure(exc)})
        title = str(work.get('title') or '').strip()
        if record is None and title:
            from radar.common.text import normalize_text

            try:
                client = self._client()
                payload = client.get('/works', {'search': title, 'per-page': 10, 'select': client.select})
                matches = [raw for raw in payload.get('results', [])
                    if normalize_text(raw.get('title') or raw.get('display_name')) == normalize_text(title)]
                if len(matches) == 1:
                    candidate = self.normalize(matches[0])
                    conflict = identifiers_conflict(identifiers, paper_identifiers(candidate.metadata))
                    attempts.append({'method': 'title', 'status': 'identity_conflict' if conflict else 'matched'})
                    if not conflict:
                        record = candidate
                else:
                    attempts.append({'method': 'title', 'status': 'no_match' if not matches else 'ambiguous'})
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                attempts.append({'method': 'title', **lookup_failure(exc)})
        if record is None:
            return lookup_diagnostics(attempts, resolved=False)
        return {'abstract': record.evidence.get('abstract', ''), 'abstract_source': 'OpenAlex' if record.evidence.get('abstract') else '',
            'pdf_urls': record.metadata.get('pdf_urls', []), 'content_urls': record.metadata.get('content_urls', {}),
            'evidence_provider': 'openalex', 'evidence_native_id': record.native_id,
            'evidence_identifiers': paper_identifiers(record.metadata), **lookup_diagnostics(attempts, resolved=True)}

    def references(self, identifier: str) -> list[str]:
        """Expose references independently of a study snowballing policy."""
        return list(self.lookup(identifier).metadata.get("referenced_works", []))

    def lookup_many(self, identifiers: list[str]) -> list[SourceRecord]:
        """Resolve reference batches using OpenAlex's OR-filter endpoint."""
        return [self.normalize(raw) for raw in self._client().works_by_ids(identifiers)]

    def collect_catalogue(self, settings: dict[str, Any], profile: dict[str, Any], cache: Any,
                          limit: int | None, *, resume: bool = False) -> dict[str, int]:
        """Preserve survey discovery and release selection as a catalogue strategy."""
        counts, _, _ = collect_papers(settings, profile, cache, limit, self)
        return counts
