from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ids_meta.common.text import stable_hash
from ids_meta.common.logging import logger

LOG = logger("run")


def utc_now() -> str:
    """Return the current time as a timezone-aware UTC ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def git_revision(root: Path) -> str:
    """Return the current Git revision or ``unknown`` when unavailable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class RunManifest:
    root: Path
    runs_dir: Path
    source: str
    stage: str
    config: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(init=False)
    started_at: str = field(default_factory=utc_now)
    counts: dict[str, int] = field(default_factory=dict)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"

    def __post_init__(self) -> None:
        """Derive a unique, date-prefixed identifier for this pipeline run."""
        self.run_id = f"{self.started_at[:10]}-{stable_hash(self.source, self.stage, self.started_at, length=10)}"

    def write(self) -> Path:
        """Persist the run metadata, counts, configuration hash, and coverage."""
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        path = self.runs_dir / f"{self.run_id}.json"
        payload = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "source": self.source,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": utc_now() if self.status != "running" else "",
            "git_revision": git_revision(self.root),
            "config_hash": stable_hash(json.dumps(self.config, sort_keys=True)),
            "parameters": self.parameters,
            "counts": self.counts,
            "coverage": self.coverage,
            "coverage_note": "Coverage is limited to recorded queries and source API indexing; completeness beyond that is not implied.",
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        LOG.info("wrote run manifest %s", path)
        return path
