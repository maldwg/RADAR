from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ids_meta.common.logging import logger
from ids_meta.common.text import unicode_safe

LOG = logger("cache")

try:
    import sqlite3
except ModuleNotFoundError:  # Some research venv builds omit the optional _sqlite3 extension.
    sqlite3 = None  # type: ignore[assignment]


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


def _sql(value: object) -> str:
    """Encode a primitive value as a safe SQLite CLI literal."""
    if value is None:
        return "NULL"
    if isinstance(value, int):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _json_payload(value: Any, context: str) -> str:
    """Serialize JSON after replacing invalid surrogate code points at the boundary."""
    cleaned = unicode_safe(value)
    if cleaned != value:
        LOG.warning("replaced invalid Unicode surrogate code points while caching %s", context)
    return json.dumps(cleaned, ensure_ascii=False)


class PipelineCache:
    def __init__(self, path: Path):
        """Open or create the SQLite cache using Python or the SQLite CLI fallback."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path) if sqlite3 is not None else None
        if self.connection is not None:
            self.connection.row_factory = sqlite3.Row
            self.connection.executescript(SCHEMA)
        else:
            LOG.info("Python sqlite extension unavailable; using sqlite3 executable for %s", path)
            self._cli(SCHEMA)

    def _cli(self, sql: str, *, json_output: bool = False) -> list[dict[str, Any]]:
        """Execute SQL through the SQLite command-line fallback."""
        command = ["sqlite3"]
        if json_output:
            command.append("-json")
        command.append(str(self.path))
        result = subprocess.run(command, input=sql, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"sqlite3 CLI failed: {result.stderr.strip()}")
        return json.loads(result.stdout or "[]") if json_output else []

    def close(self) -> None:
        """Close the Python SQLite connection when one is active."""
        if self.connection is not None:
            self.connection.close()

    def __enter__(self) -> "PipelineCache":
        """Return this cache for use as a context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the cache when leaving its context manager."""
        self.close()

    def put_record(self, namespace: str, key: str, payload: dict[str, Any], fetched_at: str) -> None:
        """Insert or replace a timestamped raw record in a namespace."""
        values = (namespace, key, _json_payload(payload, f"record {namespace}/{key}"), fetched_at)
        if self.connection is not None:
            self.connection.execute("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", values)
            self.connection.commit()
        else:
            self._cli(f"INSERT OR REPLACE INTO records VALUES ({','.join(_sql(value) for value in values)});")

    def get_record(self, namespace: str, key: str) -> dict[str, Any] | None:
        """Return one cached raw record or ``None`` when it is absent."""
        if self.connection is not None:
            row = self.connection.execute("SELECT payload_json FROM records WHERE namespace=? AND record_key=?", (namespace, key)).fetchone()
            return json.loads(row[0]) if row else None
        rows = self._cli(f"SELECT payload_json FROM records WHERE namespace={_sql(namespace)} AND record_key={_sql(key)};", json_output=True)
        return json.loads(rows[0]["payload_json"]) if rows else None

    def records(self, namespace: str) -> Iterable[tuple[str, dict[str, Any], str]]:
        """Iterate deterministically over all raw records in a namespace."""
        if self.connection is not None:
            rows: Iterable[Any] = self.connection.execute("SELECT record_key, payload_json, fetched_at FROM records WHERE namespace=? ORDER BY record_key", (namespace,))
        else:
            rows = self._cli(f"SELECT record_key, payload_json, fetched_at FROM records WHERE namespace={_sql(namespace)} ORDER BY record_key;", json_output=True)
        for row in rows:
            yield row[0] if self.connection is not None else row["record_key"], json.loads(row[1] if self.connection is not None else row["payload_json"]), row[2] if self.connection is not None else row["fetched_at"]

    def put_decision(self, namespace: str, key: str, profile_hash: str, evidence_hash: str, payload: dict[str, Any], classified_at: str) -> None:
        """Cache a decision under its model/profile and evidence fingerprints."""
        values = (namespace, key, profile_hash, evidence_hash, _json_payload(payload, f"decision {namespace}/{key}"), classified_at)
        if self.connection is not None:
            self.connection.execute("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?)", values)
            self.connection.commit()
        else:
            self._cli(f"INSERT OR REPLACE INTO decisions VALUES ({','.join(_sql(value) for value in values)});")

    def get_decision(self, namespace: str, key: str, profile_hash: str, evidence_hash: str) -> dict[str, Any] | None:
        """Return a decision only when all resume fingerprints match."""
        if self.connection is not None:
            row = self.connection.execute("SELECT payload_json FROM decisions WHERE namespace=? AND record_key=? AND profile_hash=? AND evidence_hash=?", (namespace, key, profile_hash, evidence_hash)).fetchone()
            return json.loads(row[0]) if row else None
        rows = self._cli(
            f"SELECT payload_json FROM decisions WHERE namespace={_sql(namespace)} AND record_key={_sql(key)} AND profile_hash={_sql(profile_hash)} AND evidence_hash={_sql(evidence_hash)};",
            json_output=True,
        )
        return json.loads(rows[0]["payload_json"]) if rows else None

    def get_compatible_decision(
        self, namespace: str, key: str, profile_hash: str,
        evidence_hash: str, compatible_profile_hashes: Iterable[str] = (),
    ) -> dict[str, Any] | None:
        """Reuse and migrate a decision from a known-compatible historical cache key."""
        for candidate_hash in (profile_hash, *compatible_profile_hashes):
            cached = self.get_decision(namespace, key, candidate_hash, evidence_hash)
            if cached is None:
                continue
            if candidate_hash != profile_hash:
                cached = dict(cached)
                for field in ("profile_hash", "classifier_profile_hash"):
                    if field in cached:
                        cached[field] = profile_hash
                classified_at = str(cached.get("classified_at", ""))
                self.put_decision(
                    namespace, key, profile_hash, evidence_hash, cached, classified_at,
                )
                LOG.info(
                    "migrated compatible cached decision namespace=%s key=%s old_profile=%s new_profile=%s",
                    namespace, key, candidate_hash, profile_hash,
                )
            return cached
        return None

    def put_coverage(self, source: str, query: str, shard: str, status: str, total_count: int | None, details: dict[str, Any], completed_at: str) -> None:
        """Record the completeness status and details of a discovery query shard."""
        values = (source, query, shard, status, total_count, _json_payload(details, f"coverage {source}/{query}/{shard}"), completed_at)
        if self.connection is not None:
            self.connection.execute("INSERT OR REPLACE INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?)", values)
            self.connection.commit()
        else:
            self._cli(f"INSERT OR REPLACE INTO coverage VALUES ({','.join(_sql(value) for value in values)});")

    def coverage_rows(self, source: str) -> list[dict[str, Any]]:
        """Return all recorded coverage rows for a source in stable order."""
        if self.connection is not None:
            return [dict(row) for row in self.connection.execute("SELECT * FROM coverage WHERE source=? ORDER BY query, shard", (source,))]
        return self._cli(f"SELECT * FROM coverage WHERE source={_sql(source)} ORDER BY query, shard;", json_output=True)
