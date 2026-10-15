from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


class UnionFind:
    def __init__(self, size: int):
        """Create one disjoint set for each indexed value."""
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        """Return a set representative while applying path compression."""
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def union(self, left: int, right: int) -> None:
        """Merge the two sets containing the supplied indices."""
        first, second = self.find(left), self.find(right)
        if first != second:
            self.parents[max(first, second)] = min(first, second)

    def groups(self, values: list[T]) -> list[list[T]]:
        """Materialize current disjoint sets as grouped input values."""
        grouped: dict[int, list[T]] = defaultdict(list)
        for index, value in enumerate(values):
            grouped[self.find(index)].append(value)
        return list(grouped.values())


def group_by_links(values: list[T], should_link: Callable[[T, T], bool]) -> list[list[T]]:
    """Build transitive groups from a pairwise relationship predicate."""
    unions = UnionFind(len(values))
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if should_link(values[left], values[right]):
                unions.union(left, right)
    return unions.groups(values)


def margin_confidence(margin: float, group_size: int, adjudicated: bool = False) -> float:
    """Convert a canonical-ranking margin into a coarse confidence value."""
    if group_size == 1:
        return 1.0
    if adjudicated:
        return .95
    if margin >= 4:
        return .96
    if margin >= 2:
        return .90
    if margin >= 1:
        return .80
    return .60
