"""Small provider contracts with separate optional operations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ProviderDescriptor:
    """Advertise entity kinds and operations before clients are constructed."""
    name: str
    entity_kinds: tuple[str, ...]
    capabilities: tuple[str, ...] = ("discovery",)
    catalogue_workflows: tuple[str, ...] = ()
    default_page_size: int = 100


@dataclass(frozen=True)
class ProviderQuery:
    """Carry an explicit provider-native query without translating its syntax."""
    name: str
    text: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchPage:
    """Return raw records, opaque continuation and the complete response audit."""
    records: list[dict[str, Any]]
    next_cursor: str | None
    raw: dict[str, Any]
    total: int | None = None
    complete: bool = True


@dataclass(frozen=True)
class SourceRecord:
    """Separate provider identity, strong identifiers and evidence from taxonomy."""
    provider: str
    native_id: str
    entity_kind: str
    identifiers: dict[str, str]
    metadata: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)
    native: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DiscoveryAdapter(Protocol):
    """Required operations for independently discovering normalized entities."""
    descriptor: ProviderDescriptor

    def queries(self, profile: dict[str, Any]) -> list[ProviderQuery]:
        """Validate a provider search profile without performing requests."""
        ...

    def validate_query(self, query: ProviderQuery) -> None:
        """Reject unsupported query parameters before any request is made."""
        ...

    def search(self, query: ProviderQuery, cursor: str | None, page_size: int) -> SearchPage:
        """Retrieve one page using provider-native pagination."""
        ...

    def normalize(self, raw: dict[str, Any]) -> SourceRecord:
        """Normalize one record without applying scientific eligibility rules."""
        ...


@runtime_checkable
class LookupAdapter(Protocol):
    """Optional lookup; discovery-only adapters need not implement it."""
    def lookup(self, identifier: str) -> SourceRecord:
        """Resolve a provider-native or supported external identifier."""
        ...


@runtime_checkable
class EvidenceAdapter(Protocol):
    """Optional enrichment kept separate from discovery and study decisions."""
    def enrich(self, work: dict[str, Any]) -> dict[str, Any]:
        """Retrieve evidence without emitting taxonomy or eligibility labels."""
        ...


@runtime_checkable
class ReferenceAdapter(Protocol):
    """Optional scholarly reference traversal for explicit collection strategies."""
    def references(self, identifier: str) -> list[str]:
        """Return referenced provider identifiers."""
        ...


@runtime_checkable
class CatalogueAdapter(Protocol):
    """Optional collection strategy preserving domain-specific cohort selection."""
    def collect_catalogue(self, settings: dict[str, Any], profile: dict[str, Any], cache: Any,
                          limit: int | None, *, resume: bool = False) -> dict[str, int]:
        """Collect through a strategy supported by the provider descriptor."""
        ...
