"""Resolve optional pipeline limits and proportional discovery budgets."""

from __future__ import annotations

import math
from typing import Any

from ids_meta.common.config import ConfigError


def optional_limit(value: Any, name: str) -> int | None:
    """Parse a positive record limit, treating missing or all values as unbounded."""
    if value is None or isinstance(value, str) and value.strip().casefold() in {"all", "all_eligible", "unlimited"}:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"{name} must be a positive integer or 'all'") from exc
    if isinstance(value, bool) or result < 1 or isinstance(value, float) and value != result:
        raise ConfigError(f"{name} must be a positive integer or 'all'")
    return result


def query_result_limit(settings: dict[str, Any], collection_limit: int | None, factor_key: str = "query_limit_factor") -> int | None:
    """Multiply the largest finite pipeline limit; no finite limits means no cap."""
    collection = optional_limit(collection_limit, "collection_limit")
    classification = optional_limit(settings.get("classification_limit"), "classification_limit")
    value = settings.get(factor_key, settings.get("query_limit_factor", 3.0))
    try:
        factor = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigError(f"{factor_key} must be a finite positive number") from exc
    if isinstance(value, bool) or not math.isfinite(factor) or factor <= 0:
        raise ConfigError(f"{factor_key} must be a finite positive number")
    limits = [limit for limit in (collection, classification) if limit is not None]
    if not limits:
        return None
    capacity = max(limits) * factor
    if not math.isfinite(capacity):
        raise ConfigError(f"{factor_key} produces a nonfinite query cap")
    return math.ceil(capacity)


def limit_label(limit: int | None) -> str:
    """Format a discovery or collection limit without implying a zero cap."""
    return f"{limit:,}" if limit is not None else "unlimited"


def capacity_min(first: int | None, second: int | None) -> int | None:
    """Return the smaller finite capacity, or unbounded when neither is finite."""
    limits = [value for value in (first, second) if value is not None]
    return min(limits) if limits else None
