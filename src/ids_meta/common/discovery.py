"""Provider-independent discovery budgets, audits and frozen union cohorts."""
from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ids_meta.adapters.contracts import DiscoveryAdapter, SourceRecord
from ids_meta.adapters.registry import create_adapter
from ids_meta.common.jsonio import write_json
from ids_meta.common.limits import optional_limit, query_result_limit
from ids_meta.common.provenance import file_hash
from ids_meta.common.run import utc_now
from ids_meta.common.sources import content_hash
from ids_meta.common.text import normalize_text


@dataclass(frozen=True)
class DiscoverySource:
    """Bind one study search to provider settings and an optional injected adapter."""
    name: str
    profile_path: Path
    settings: dict[str, Any] = field(default_factory=dict)
    adapter: DiscoveryAdapter | None = None
    cache_path: Path | None = None


def _merge_records(entries: list[tuple[SourceRecord, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Deduplicate same-kind strong identities; flag conflicting or title-only matches."""
    groups: list[list[int]] = []
    index: dict[tuple[str, str, str], int] = {}
    conflicts: set[int] = set()
    for position, (record, _) in enumerate(entries):
        keys = [(record.entity_kind, f"native:{record.provider}", record.native_id)] + [
            (record.entity_kind, key, value) for key, value in record.identifiers.items()]
        matches = {index[key] for key in keys if key in index}
        members = [position] + [member for group in matches for member in groups[group]]
        identities: dict[str, set[str]] = {}
        for member in members:
            for key, value in entries[member][0].identifiers.items():
                identities.setdefault(key, set()).add(value)
        if any(len(values) > 1 for values in identities.values()):
            conflicts.update(members)
            # A conflicting record must not overwrite a strong identifier's owner.
            group = len(groups)
            groups.append([position])
            index.setdefault(keys[0], group)
            for key in keys[1:]:
                index.setdefault(key, group)
            continue
        group = min(matches) if matches else len(groups)
        if not matches:
            groups.append([])
        groups[group] = members
        for old_group in matches - {group}:
            groups[old_group] = []
        for member in members:
            item = entries[member][0]
            index[(item.entity_kind, f"native:{item.provider}", item.native_id)] = group
            for key, value in item.identifiers.items():
                index[(item.entity_kind, key, value)] = group
    rows = []
    for members in groups:
        if not members:
            continue
        ordered = sorted(members, key=lambda member: entries[member][1]["selection_rank"])
        records = [entries[member][0] for member in ordered]
        first = records[0]
        identifiers = {key: value for record in records for key, value in record.identifiers.items()}
        if "doi" in identifiers:
            identity = "doi:" + identifiers["doi"]
        elif "arxiv" in identifiers:
            identity = "arxiv:" + identifiers["arxiv"]
        else:
            identity = min(f"{record.provider}:{record.native_id}" for record in records)
        metadata: dict[str, Any] = {}
        for record in records:
            for key, value in record.metadata.items():
                if value not in (None, "", []) and key not in metadata:
                    metadata[key] = value
        evidence = max(records, key=lambda record: (
            bool(record.evidence.get("full_text_complete")), len(record.evidence.get("full_text") or ""))).evidence
        abstract = max((record.evidence.get("abstract") or "" for record in records), key=len)
        memberships = [entries[member][1] for member in sorted(members)]
        row = {**metadata, **evidence, "abstract": abstract,
            "entity_id": content_hash([first.entity_kind, identity]), "entity_kind": first.entity_kind,
            "identifiers": identifiers, "source_memberships": memberships,
            "source_records": [asdict(record) for record in records],
            "discovery_queries": list(dict.fromkeys(member["query"] for member in memberships)),
            "query_ranks": {f'{member["provider"]}:{member["query"]}': member["rank"] for member in reversed(memberships)},
            "selection_rank": min(member["selection_rank"] for member in memberships),
            "identity_conflict": bool(set(members) & conflicts)}
        row[{"paper": "paper_id", "repository": "repository_id", "artifact": "artifact_id"}[first.entity_kind]] = identity
        rows.append(row)
    titles: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        title = normalize_text(row.get("title", ""))
        if title:
            titles.setdefault((row["entity_kind"], title), []).append(row)
    for matches in titles.values():
        if len(matches) > 1:
            for row in matches:
                row["possible_title_duplicates"] = [other["entity_id"] for other in matches if other is not row]
    return sorted(rows, key=lambda row: row["selection_rank"])


def collect_sources(sources: list[DiscoverySource], output: Path, *, limit: int | str | None = 100,
                    entity_kind: str | None = None) -> list[dict[str, Any]]:
    """Collect and freeze a union once, retaining query memberships and raw pages."""
    if not sources or len({source.name for source in sources}) != len(sources):
        raise ValueError("discovery requires distinct configured providers")
    if output.suffix.lower() != ".json" or any(output.resolve() == source.profile_path.resolve() for source in sources):
        raise ValueError("study collection requires a separate JSON output")
    limit = optional_limit(limit, "collection_limit")
    prepared = []
    for source in sources:
        digest = file_hash(source.profile_path)
        with source.profile_path.open("rb") as handle:
            profile = tomllib.load(handle)
        adapter = source.adapter or create_adapter(source.name, source.settings, source.profile_path)
        if adapter.descriptor.name != source.name or "discovery" not in adapter.descriptor.capabilities:
            raise ValueError("provider does not support discovery")
        queries = adapter.queries(profile)
        for query in queries:
            adapter.validate_query(query)
        if entity_kind is not None and entity_kind not in adapter.descriptor.entity_kinds:
            raise ValueError(f"provider {source.name} cannot discover {entity_kind}")
        cap = query_result_limit(source.settings, limit)
        prepared.append((source, digest, profile, adapter, queries, cap))
    entries, source_audits, audits = [], [], []
    query_order = 0
    for source, digest, profile, adapter, queries, cap in prepared:
        provider_audits = []
        if source.settings.get("collection_mode") is not None:
            from ids_meta.common.cache import PipelineCache
            from ids_meta.common.paper_collection import collect_papers

            with PipelineCache(source.cache_path or output.parent / f"{source.name}.collection.sqlite3") as cache:
                counts, records, memberships = collect_papers(source.settings, profile, cache, limit, adapter)
                selection = cache.get_record(f"{source.name}.collection", "latest")
            for position, key in enumerate(selection["selection_order"], 1):
                record = records[key]
                for member in memberships.get(key) or [{"provider": source.name, "native_id": key,
                    "query": "backward_reference", "rank": position, "selection_rank": [position, 0, key]}]:
                    membership = {**member, "selection_rank": [position, query_order, key]}
                    entries.append((record, membership))
            provider_audits = selection["queries"]
            audits.extend(provider_audits)
            source_audits.append({"provider": source.name, "profile_sha256": digest, "profile": profile,
                "collection_limit": limit, "classification_limit": optional_limit(source.settings.get("classification_limit"), "classification_limit"),
                "query_limit_factor": source.settings.get("query_limit_factor", 3.0), "per_query_limit": cap,
                "queries": provider_audits, "selection": selection, "counts": counts})
            query_order += max(1, len(queries))
            continue
        for query in queries:
            cursor, scanned, total, exhausted, complete = None, 0, None, False, True
            seen, pages = set(), []
            while cap is None or scanned < cap:
                if cursor in seen:
                    raise RuntimeError(f"{source.name} repeated study-search cursor")
                seen.add(cursor)
                size = source.settings.get("per_page", adapter.descriptor.default_page_size)
                if cap is not None:
                    size = min(size, cap)
                page = adapter.search(query, cursor, size)
                total, complete = page.total, complete and page.complete
                pages.append({"cursor": cursor, "response": page.raw})
                items = page.records if cap is None else page.records[:cap - scanned]
                for raw in items:
                    if not isinstance(raw, dict):
                        raise ValueError("search records must be objects")
                    record = adapter.normalize(raw)
                    if not record.native_id or record.provider != source.name or record.entity_kind not in adapter.descriptor.entity_kinds:
                        raise ValueError("normalized record violates its provider descriptor")
                    if any(not isinstance(record.evidence.get(key, ""), str) for key in ("abstract", "full_text")):
                        raise ValueError("normalized evidence text must be strings")
                    if entity_kind is not None and record.entity_kind != entity_kind:
                        raise ValueError("discovery cannot combine incompatible entity kinds for this study")
                    scanned += 1
                    entries.append((record, {"provider": source.name, "native_id": record.native_id,
                        "query": query.name, "rank": scanned,
                        "selection_rank": [scanned, query_order, record.native_id]}))
                exhausted = not page.records or (page.next_cursor is None and len(items) == len(page.records))
                if exhausted:
                    break
                cursor = page.next_cursor
                if cursor is None:
                    break
            audit = {"provider": source.name, "name": query.name, "search": query.text,
                "parameters": query.parameters, "scanned": scanned, "reported_hits": total,
                "coverage": "complete" if exhausted and complete else "bounded", "pages": pages}
            audits.append(audit)
            provider_audits.append(audit)
            query_order += 1
        source_audits.append({"provider": source.name, "profile_sha256": digest, "profile": profile,
            "collection_limit": limit, "classification_limit": optional_limit(source.settings.get("classification_limit"), "classification_limit"),
            "query_limit_factor": source.settings.get("query_limit_factor", 3.0), "per_query_limit": cap,
            "queries": provider_audits})
    for source, digest, profile, adapter, queries, _ in prepared:
        if file_hash(source.profile_path) != digest:
            raise ValueError("study search profile changed during collection")
        if adapter.queries(profile) != queries:
            raise ValueError("discovery inputs changed during collection")
    pool = _merge_records(entries)
    rows = pool[:limit]
    # Single-source query names are unambiguous and remain convenient for audits.
    if len(sources) == 1:
        for row in rows:
            row["query_ranks"] = {key.split(":", 1)[1]: value for key, value in row["query_ranks"].items()}
    write_json(output, rows)
    audit = {"schema_version": "1.0", "providers": [source.name for source in sources],
        "collected_at": utc_now(), "candidate_sha256": file_hash(output), "retained": len(rows),
        "pool_size": len(pool), "selection": "best per-query rank, provider/query declaration order, native identity",
        "collection_limit": limit, "sources": source_audits, "queries": audits,
        "selected_ids": [row.get("paper_id", row["entity_id"]) for row in rows],
        "source_recovery": {source.name: {"discovered": sum(any(member["provider"] == source.name for member in row["source_memberships"]) for row in pool),
            "retained": sum(any(member["provider"] == source.name for member in row["source_memberships"]) for row in rows)} for source in sources}}
    if any(source.settings.get("collection_mode") for source in sources):
        audit["selection"] = "configured per-provider paper collection order, provider declaration order, native identity"
    if len(sources) == 1:
        audit.update({key: source_audits[0][key] for key in ("provider", "profile_sha256", "profile",
            "classification_limit", "query_limit_factor", "per_query_limit")})
    write_json(output.with_suffix(".collection.json"), audit)
    return rows
