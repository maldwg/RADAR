"""Frozen CSV/JSON input as a discovery provider, with file pagination."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from radar.adapters.contracts import ProviderDescriptor, ProviderQuery, SearchPage, SourceRecord
from radar.adapters.profiles import strong_identifiers, validated_page_size, validate_query_fields
from radar.common.csvio import configure_csv_limit
from radar.common.provenance import file_hash

DESCRIPTOR = ProviderDescriptor("local", ("paper", "repository", "artifact"))
METADATA_FIELDS = {"title", "full_name", "doi", "arxiv", "publication_year", "publication_date", "authors",
                   "venue", "primary_url", "pdf_url", "cited_by_count"}
EVIDENCE_FIELDS = {"abstract", "full_text", "full_text_complete", "full_text_url"}


class LocalDiscoveryAdapter:
    """Import metadata and evidence while excluding prior scientific judgments."""
    descriptor = DESCRIPTOR

    def __init__(self, settings: dict[str, Any], profile_path: Path, client: Any = None):
        """Bind file location semantics; local providers never require HTTP clients."""
        self.settings, self.profile_path = settings, profile_path
        self.rows: list[dict[str, Any]] | None = None
        self.path: Path | None = None
        self.digest = ""
        self.mapping: dict[str, str] = {}
        self.identity = ""
        self.kind = ""
        validated_page_size(settings.get("per_page", 100), 10000)

    def queries(self, profile: dict[str, Any]) -> list[ProviderQuery]:
        """Validate one explicitly supplied file and its metadata field mapping."""
        allowed = {"schema_version", "provider", "path", "identity", "entity_kind", "mapping", "named_entity_seeding"}
        if set(profile) - allowed or profile.get("schema_version") != "1.0" or profile.get("provider") != "local":
            raise ValueError("local search requires a versioned local file profile")
        if profile.get("named_entity_seeding") is not False:
            raise ValueError("local input must explicitly disable named_entity_seeding")
        if not isinstance(profile.get("path"), str) or not profile["path"].strip():
            raise ValueError("local search requires path")
        path = (self.profile_path.parent / profile["path"]).resolve()
        if not path.is_file() or path.suffix.lower() not in {".csv", ".json"}:
            raise ValueError("local input must be an existing CSV or JSON file")
        identity, kind = profile.get("identity"), profile.get("entity_kind")
        if not isinstance(identity, str) or not identity.strip() or kind not in self.descriptor.entity_kinds:
            raise ValueError("local input requires an identity column and supported entity_kind")
        mapping = profile.get("mapping", {})
        if not isinstance(mapping, dict) or set(mapping) - (METADATA_FIELDS | EVIDENCE_FIELDS) or any(
            not isinstance(value, str) or not value.strip() for value in mapping.values()
        ):
            raise ValueError("local mapping accepts metadata and evidence fields only")
        self.path, self.identity, self.kind, self.mapping = path, identity, kind, mapping
        self.digest = file_hash(path)
        return [ProviderQuery("file", str(path), {"sha256": self.digest})]

    def search(self, query: ProviderQuery, cursor: str | None, page_size: int) -> SearchPage:
        """Page frozen rows and detect file changes rather than silently refreshing."""
        self.validate_query(query)
        if self.path is None or file_hash(self.path) != self.digest:
            raise ValueError("local input changed during collection")
        if self.rows is None:
            if self.path.suffix.lower() == ".json":
                rows = json.loads(self.path.read_text(encoding="utf-8"))
            else:
                configure_csv_limit()
                with self.path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError("local input requires an array of objects")
            seen = set()
            for row in rows:
                native = row.get(self.identity)
                if not isinstance(native, (str, int)) or isinstance(native, bool) or not str(native).strip():
                    raise ValueError("local record lacks its declared identity")
                identity = str(native).strip()
                if identity in seen:
                    raise ValueError(f"duplicate local identity: {identity}")
                seen.add(identity)
            self.rows = rows
        offset = int(cursor or "0")
        size = validated_page_size(page_size, 10000)
        items = self.rows[offset:offset + size]
        next_offset = offset + len(items)
        return SearchPage(items, str(next_offset) if next_offset < len(self.rows) else None,
            {"path": str(self.path), "sha256": self.digest, "offset": offset, "records": items}, len(self.rows))

    def validate_query(self, query: ProviderQuery) -> None:
        """Require the configured frozen file rather than accepting arbitrary paths."""
        validate_query_fields(query, {"sha256"})
        if query.text != str(self.path) or query.parameters["sha256"] != self.digest:
            raise ValueError("local query does not identify the frozen input")

    def normalize(self, raw: dict[str, Any]) -> SourceRecord:
        """Import only declared evidence and metadata, retaining original rows for audit."""
        native = raw.get(self.identity)
        if not isinstance(native, (str, int)) or isinstance(native, bool) or not str(native).strip():
            raise ValueError("local record lacks its declared identity")
        values = {key: raw.get(self.mapping.get(key, key), "") for key in METADATA_FIELDS | EVIDENCE_FIELDS}
        complete = values["full_text_complete"]
        if isinstance(complete, str):
            if complete.lower() not in {"", "true", "false"}:
                raise ValueError("local full_text_complete must be boolean")
            values["full_text_complete"] = complete.lower() == "true"
        elif not isinstance(complete, bool):
            raise ValueError("local full_text_complete must be boolean")
        return SourceRecord("local", str(native).strip(), self.kind,
            strong_identifiers(values.get("doi"), values.get("arxiv")),
            {key: values[key] for key in METADATA_FIELDS}, {key: values[key] for key in EVIDENCE_FIELDS}, raw)
