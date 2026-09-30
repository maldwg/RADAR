from __future__ import annotations

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")
SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def unicode_safe(value: Any) -> Any:
    """Recursively replace invalid UTF-16 surrogate code points with U+FFFD."""
    if isinstance(value, str):
        return SURROGATE_RE.sub("\ufffd", value)
    if isinstance(value, dict):
        return {unicode_safe(key): unicode_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unicode_safe(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unicode_safe(item) for item in value)
    return value


def unicode_safe_text(value: object) -> str:
    """Convert a value to text and replace invalid UTF-16 surrogate code points."""
    return SURROGATE_RE.sub("\ufffd", str(value or ""))



def normalize_text(*values: object) -> str:
    """Normalize values to case-folded Unicode alphanumeric tokens."""
    text = " ".join(str(value or "") for value in values)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_RE.findall(text))


def stable_hash(*values: object, length: int = 20) -> str:
    """Create a deterministic truncated SHA-256 fingerprint from ordered values."""
    payload = "\x1f".join(unicode_safe_text(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def slug(value: object, limit: int = 64) -> str:
    """Convert a value into a bounded lowercase hyphenated identifier."""
    normalized = normalize_text(value).replace(" ", "-")
    return normalized[:limit].strip("-") or "unknown"


def split_values(value: object) -> list[str]:
    """Split semicolon- or comma-delimited values while preserving first occurrence."""
    return list(dict.fromkeys(part.strip() for part in re.split(r"[;,]", str(value or "")) if part.strip()))


def _similar(left: str, right: str, threshold: float) -> bool:
    """Test exact or typo-tolerant similarity for sufficiently long strings."""
    if left == right:
        return True
    if min(len(left), len(right)) < 5:
        return False
    return SequenceMatcher(None, left, right).ratio() >= threshold


def contains_concept(text: str, concept: str, threshold: float = 0.86) -> bool:
    """Find a concept in text using normalized exact or typo-tolerant token matching."""
    haystack = normalize_text(text)
    needle = normalize_text(concept)
    if not needle:
        return False
    if f" {needle} " in f" {haystack} ":
        return True
    words = haystack.split()
    required = needle.split()
    if len(required) == 1:
        return any(_similar(word, required[0], threshold) for word in words)
    for start in range(len(words)):
        window = words[start : start + len(required) + 2]
        if all(any(_similar(word, target, threshold) for word in window) for target in required):
            return True
    return False


def matching_concepts(text: str, concepts: Iterable[str]) -> list[str]:
    """Return every configured concept that matches the supplied text."""
    return [concept for concept in concepts if contains_concept(text, concept)]
