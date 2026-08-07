from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def configure_csv_limit() -> None:
    """Raise Python's CSV field-size limit to the largest supported value."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV file into string-keyed row dictionaries."""
    configure_csv_limit()
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    """Write a schema-ordered CSV through a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in columns})
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def require_columns(rows: list[dict[str, str]], required: Iterable[str], source: Path) -> None:
    """Raise an error when a CSV source lacks any required columns."""
    if rows:
        columns = set(rows[0])
    else:
        configure_csv_limit()
        with source.open(newline="", encoding="utf-8") as handle:
            columns = set(csv.DictReader(handle).fieldnames or [])
    missing = set(required) - columns
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")
