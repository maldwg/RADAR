from __future__ import annotations

from difflib import SequenceMatcher

from radar.common.text import normalize_text, split_values, stable_hash


def _names(row: dict[str, str]) -> set[str]:
    """Return normalized artifact names and aliases suitable for comparison."""
    values = [row.get("artifact_name", ""), row.get("artifact_display_name", ""), *split_values(row.get("artifact_aliases", ""))]
    return {normalize_text(value) for value in values if len(normalize_text(value)) >= 5}


def _related(left: set[str], right: set[str]) -> bool:
    """Return whether two artifact-name sets are high-confidence variants."""
    if left & right:
        return True
    for first in left:
        first_tokens = set(first.split())
        for second in right:
            second_tokens = set(second.split())
            token_overlap = len(first_tokens & second_tokens) / max(1, min(len(first_tokens), len(second_tokens)))
            if len(first_tokens) == len(second_tokens) and token_overlap >= .75 and SequenceMatcher(None, first, second).ratio() >= .88:
                return True
    return False


def canonicalize_artifact_ids(rows: list[dict[str, str]]) -> None:
    """Merge high-confidence spelling/alias variants without a named artifact registry."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        """Find a row's canonical union-find representative."""
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        """Merge two artifact groups under a deterministic representative."""
        a, b = find(left), find(right)
        if a != b:
            parents[max(a, b)] = min(a, b)

    name_sets = [_names(row) for row in rows]
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if _related(name_sets[left], name_sets[right]):
                union(left, right)
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    for indices in groups.values():
        canonical_name = min(
            (rows[index].get("artifact_name", "") for index in indices if rows[index].get("artifact_name")),
            key=lambda value: (len(normalize_text(value)), normalize_text(value)),
        )
        artifact_id = f"{'-'.join(normalize_text(canonical_name).split())[:64]}-{stable_hash(normalize_text(canonical_name), length=8)}"
        for index in indices:
            rows[index]["artifact_id"] = artifact_id
