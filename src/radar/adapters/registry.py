"""The single registration point for supported provider operations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from radar.adapters.contracts import (
    CatalogueAdapter, DiscoveryAdapter, EvidenceAdapter, LookupAdapter,
    ProviderDescriptor, ReferenceAdapter,
)
from radar.adapters.github import GitHubAdapter
from radar.adapters.local import LocalDiscoveryAdapter
from radar.adapters.openalex import OpenAlexAdapter
from radar.adapters.semantic_scholar import SemanticScholarAdapter


@dataclass(frozen=True)
class ProviderRegistration:
    """Associate a descriptor with its lazy, injectable adapter factory."""
    descriptor: ProviderDescriptor
    factory: Callable[..., DiscoveryAdapter]


PROVIDERS: dict[str, ProviderRegistration] = {}


def register_provider(descriptor: ProviderDescriptor, factory: Callable[..., DiscoveryAdapter]) -> None:
    """Register an extension once and reject accidental name replacement."""
    if descriptor.name in PROVIDERS:
        raise ValueError(f"duplicate provider registration: {descriptor.name}")
    PROVIDERS[descriptor.name] = ProviderRegistration(descriptor, factory)


def provider_descriptor(name: str) -> ProviderDescriptor:
    """Resolve capabilities without constructing clients or issuing requests."""
    if name not in PROVIDERS:
        raise ValueError(f"unregistered provider: {name!r}")
    return PROVIDERS[name].descriptor


def create_adapter(name: str, settings: dict[str, Any], profile_path: Path, *, client: Any = None) -> DiscoveryAdapter:
    """Create the registered adapter without provider switches in callers."""
    provider_descriptor(name)
    adapter = PROVIDERS[name].factory(settings, profile_path, client=client)
    if not isinstance(adapter, DiscoveryAdapter):
        raise ValueError("registered adapter does not implement the discovery interface")
    if adapter.descriptor != PROVIDERS[name].descriptor:
        raise ValueError("adapter descriptor differs from its registration")
    for capability, protocol in (("lookup", LookupAdapter), ("evidence", EvidenceAdapter), ("references", ReferenceAdapter)):
        if capability in adapter.descriptor.capabilities and not isinstance(adapter, protocol):
            raise ValueError(f"adapter advertises unimplemented {capability}")
    if adapter.descriptor.catalogue_workflows and not isinstance(adapter, CatalogueAdapter):
        raise ValueError("adapter advertises unimplemented catalogue workflows")
    return adapter


for _adapter in (GitHubAdapter, OpenAlexAdapter, SemanticScholarAdapter, LocalDiscoveryAdapter):
    register_provider(_adapter.descriptor, _adapter)
