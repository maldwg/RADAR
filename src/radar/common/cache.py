from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from radar.common.logging import logger
from radar.common.text import stable_hash, unicode_safe

LOG = logger("cache")

COLLECTED_EVIDENCE_POLICY = "collection-snapshot-v1"

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS records (
    namespace TEXT NOT NULL, record_key TEXT NOT NULL,
    payload_json TEXT NOT NULL, fetched_at TEXT NOT NULL,
    PRIMARY KEY(namespace, record_key)
);
CREATE TABLE IF NOT EXISTS decisions (
    namespace TEXT NOT NULL, record_key TEXT NOT NULL,
    profile_hash TEXT NOT NULL, evidence_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL, classified_at TEXT NOT NULL,
    PRIMARY KEY(namespace, record_key, profile_hash, evidence_hash)
);
CREATE TABLE IF NOT EXISTS coverage (
    source TEXT NOT NULL, query TEXT NOT NULL, shard TEXT NOT NULL,
    status TEXT NOT NULL, total_count INTEGER, details_json TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    PRIMARY KEY(source, query, shard)
);
"""


def _json_payload(value: Any, context: str) -> str:
    """Serialize JSON after replacing invalid surrogate code points at the boundary."""
    cleaned = unicode_safe(value)
    if cleaned != value:
        LOG.warning("replaced invalid Unicode surrogate code points while caching %s", context)
    return json.dumps(cleaned, ensure_ascii=False)


def collected_record_hash(payload: dict[str, Any]) -> str:
    """Identify original collection evidence independently of later enrichment."""
    return stable_hash(COLLECTED_EVIDENCE_POLICY, json.dumps(unicode_safe(payload), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")))


class PipelineCache:
    def __init__(self, path: Path, *, read_only: bool = False):
        """Open a current-format cache using Python's SQLite module."""
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.read_only = read_only
        self.connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) if read_only else sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        if not read_only:
            self.connection.executescript(SCHEMA)

    def close(self) -> None:
        """Close the database connection."""
        self.connection.close()

    def __enter__(self) -> PipelineCache:
        """Return the cache for a managed operation."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the cache when leaving its context."""
        self.close()

    def put_record(self, namespace: str, key: str, payload: dict[str, Any], fetched_at: str) -> None:
        """Operate on current-format cached records."""
        self.connection.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)",
            (namespace, key, _json_payload(payload, f"record {namespace}/{key}"), fetched_at))
        self.connection.commit()

    def put_collected_record(self, namespace: str, key: str, payload: dict[str, Any], fetched_at: str) -> None:
        """Atomically save an acquisition snapshot and the mutable working record.

        Only collection calls this method. Prefiltering/classification enrichment
        uses put_record, so it cannot replace the original baseline evidence.
        An explicit new collection replaces both versions with its fresh input.
        """
        payload = unicode_safe(payload)
        snapshot = {"policy": COLLECTED_EVIDENCE_POLICY, "payload": payload,
            "sha256": collected_record_hash(payload)}
        with self.connection:
            self.connection.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", [
                (namespace + ".collected", key, _json_payload(snapshot, f"collection {namespace}/{key}"), fetched_at),
                (namespace, key, _json_payload(payload, f"record {namespace}/{key}"), fetched_at),
            ])

    def get_collected_record(self, namespace: str, key: str) -> dict[str, Any]:
        """Require an intact collection snapshot instead of guessing from enriched data."""
        snapshot = self.get_record(namespace + ".collected", key)
        if snapshot is None:
            raise ValueError(f"missing collected evidence snapshot for {namespace}/{key}; "
                "rerun collection with --no-resume before running baselines; "
                "enriched records and candidate CSVs cannot recover the original text")
        payload = snapshot.get("payload")
        if (snapshot.get("policy") != COLLECTED_EVIDENCE_POLICY or not isinstance(payload, dict)
                or snapshot.get("sha256") != collected_record_hash(payload)):
            raise ValueError(f"invalid collected evidence snapshot for {namespace}/{key}; rerun collection with --no-resume")
        return payload

    def get_record(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Operate on current-format cached records."""
        row = self.connection.execute("SELECT payload_json FROM records WHERE namespace=? AND record_key=?", (namespace, key)).fetchone()
        return json.loads(row[0]) if row else None

    def records(self, namespace: str) -> Iterable[tuple[str, dict[str, Any], str]]:
        """Operate on current-format cached records."""
        rows = self.connection.execute("SELECT record_key, payload_json, fetched_at FROM records WHERE namespace=? ORDER BY record_key", (namespace,))
        for row in rows:
            yield row[0], json.loads(row[1]), row[2]

    def put_decision(self, namespace: str, key: str, profile_hash: str, evidence_hash: str, payload: dict[str, Any], classified_at: str) -> None:
        """Operate on current-format cached records."""
        self.connection.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
            (namespace, key, profile_hash, evidence_hash, _json_payload(payload, f"decision {namespace}/{key}"), classified_at))
        self.connection.commit()

    def get_decision(self, namespace: str, key: str, profile_hash: str, evidence_hash: str) -> dict[str, Any] | None:
        """Operate on current-format cached records."""
        row = self.connection.execute("SELECT payload_json FROM decisions WHERE namespace=? AND record_key=? AND profile_hash=? AND evidence_hash=?",
            (namespace, key, profile_hash, evidence_hash)).fetchone()
        return json.loads(row[0]) if row else None

    def put_coverage(self, source: str, query: str, shard: str, status: str, total_count: int | None, details: dict[str, Any], completed_at: str) -> None:
        """Operate on current-format cached records."""
        self.connection.execute("INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source, query, shard, status, total_count, _json_payload(details, f"coverage {source}/{query}/{shard}"), completed_at))
        self.connection.commit()

    def coverage_rows(self, source: str) -> list[dict[str, Any]]:
        """Operate on current-format cached records."""
        return [dict(row) for row in self.connection.execute("SELECT * FROM coverage WHERE source=? ORDER BY query, shard", (source,))]
