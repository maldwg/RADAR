"""Reporting and record-import contracts, separate from discovery adapters."""
from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ids_meta.common.csvio import configure_csv_limit


def content_hash(value: Any) -> str:
    """Fingerprint a JSON-compatible definition deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class SourceContract:
    """Declare provider identity separately from taxonomy and entity semantics."""
    name: str
    entity_kind: str
    identity: str
    label: str
    selectors: dict[str, str]
    fields: tuple[str, ...] = ()
    multilabel: tuple[str, ...] = ()
    display_name: str = ""
    capabilities: tuple[str, ...] = ()
    field_labels: dict[str, str] = field(default_factory=dict)
    mapping: dict[str, str] = field(default_factory=dict)
    taxonomy_version: str = "1.0"
    decisions: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Identify the complete interpretation contract, including field mappings."""
        return content_hash(asdict(self))


BUILTIN_CONTRACTS = {
    "github": SourceContract("github", "repository", "github_repo_id", "full_name", {"repository": "full_name", "id": "github_repo_id"},
        ("role", "observation_scopes", "observation_objects", "capabilities", "detection_methods", "primary_system_type", "system_types", "centralization"),
        ("observation_scopes", "observation_objects", "capabilities", "detection_methods", "system_types"), "GitHub", ("repository_metadata", "evidence")),
    "openalex": SourceContract("openalex", "paper", "openalex_id", "title", {"openalex_url": "openalex_id", "title": "title", "id": "openalex_id", "artifact": "artifact_name"},
        ("paper_role", "relevance_tier", "artifact_type", "domain", "data_form", "capture_environment", "labeling"),
        ("artifact_aliases",), "OpenAlex", ("publication_dates", "citations", "evidence", "artifact_reference")),
}


def source_contract(name: str, supplied: SourceContract | None = None) -> SourceContract:
    """Resolve a provider without treating unknown providers as GitHub or OpenAlex."""
    result = supplied or BUILTIN_CONTRACTS.get(name)
    if result is None or result.name != name:
        raise ValueError(f"provider {name!r} requires an explicit source contract")
    return result


def load_source_contract(path: Path) -> SourceContract:
    """Load a versioned local provider, entity and research profile."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != "1.0":
        raise ValueError("source profile requires schema_version 1.0")
    provider, taxonomy = raw.get("provider", {}), raw.get("taxonomy", {})
    if not isinstance(provider, dict) or not isinstance(taxonomy, dict):
        raise ValueError("provider and taxonomy must be tables")
    for key in ("name", "entity_kind", "identity", "label"):
        if not isinstance(provider.get(key), str) or not provider[key].strip():
            raise ValueError(f"provider.{key} must be a non-empty string")
    if provider["entity_kind"] not in {"repository", "paper", "artifact"}:
        raise ValueError("entity_kind must be repository, paper or artifact")
    selectors = raw.get("selectors", {"id": provider["identity"]})
    mapping, labels, decisions = raw.get("mapping", {}), taxonomy.get("labels", {}), raw.get("decisions", {})
    for name, values in (("selectors", selectors), ("mapping", mapping), ("taxonomy.labels", labels), ("decisions", decisions)):
        if not isinstance(values, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip() for k, v in values.items()):
            raise ValueError(f"{name} must map names to non-empty strings")
    if selectors.get("id") != provider["identity"]:
        raise ValueError("selectors.id must identify provider.identity")
    if not set(decisions.values()) <= {'accept', 'accept_adjacent', 'exclude', 'uncertain', 'error'}:
        raise ValueError('decision mappings must target accept, accept_adjacent, exclude, uncertain or error')
    arrays = {}
    for name, value in (("fields", taxonomy.get("fields", [])), ("multilabel", taxonomy.get("multilabel", [])), ("capabilities", provider.get("capabilities", []))):
        if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
            raise ValueError(f"{name} must be an array of non-empty strings")
        arrays[name] = tuple(dict.fromkeys(value))
    if not set(arrays["multilabel"]) <= set(arrays["fields"]):
        raise ValueError("multilabel fields must be declared taxonomy fields")
    return SourceContract(provider["name"], provider["entity_kind"], provider["identity"], provider["label"], selectors,
        arrays["fields"], arrays["multilabel"], provider.get("display_name", provider["name"]), arrays["capabilities"], labels, mapping, str(taxonomy.get("version", "1.0")), decisions)


def entity_identifier(contract: SourceContract, external_id: str) -> str:
    """Keep providers and entity kinds distinct, independent of mutable titles."""
    return content_hash([contract.name, contract.entity_kind, external_id])


class LocalSourceAdapter:
    """Adapt CSV/JSON records without provider-specific plotting logic."""
    def __init__(self, contract: SourceContract):
        """Bind a validated provider contract."""
        self.contract = contract

    def normalize(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Preserve native fields and add deterministic, collision-checked identities."""
        result, seen = [], set()
        for index, row in enumerate(rows, 1):
            item = dict(row)
            for target, origin in self.contract.mapping.items():
                if origin in row:
                    item[target] = row[origin]
            native = item.get(self.contract.identity)
            external = str(native).strip() if isinstance(native, (str, int)) and not isinstance(native, bool) else ''
            if not external or external in seen:
                raise ValueError(f"missing or duplicate {self.contract.identity} at row {index}: {external!r}")
            seen.add(external)
            item[self.contract.identity] = external
            item["entity_id"] = entity_identifier(self.contract, external)
            item["provider"] = self.contract.name
            item["entity_kind"] = self.contract.entity_kind
            if isinstance(item.get('error'), bool):
                item['error'] = 'true' if item['error'] else ''
            if item.get('decision'):
                decision = str(item['decision']).strip().casefold()
                item['decision'] = self.contract.decisions.get(decision, decision)
                if item['decision'] not in {'accept', 'accept_adjacent', 'exclude', 'uncertain', 'error'}:
                    raise ValueError(f'unsupported decision at row {index}: {decision}')
            # Keep collection dates and citation counts as optional capabilities.
            for key, value in list(item.items()):
                if isinstance(value, list):
                    item[key] = (json.dumps(value, sort_keys=True) if any(isinstance(v, (dict, list)) for v in value)
                        else ";".join(str(v) for v in value))
                elif isinstance(value, dict):
                    item[key] = json.dumps(value, sort_keys=True)
            result.append(item)
        return result

    def read(self, path: Path) -> list[dict[str, Any]]:
        """Read local CSV or a JSON array using the same normalization contract."""
        if path.suffix.casefold() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            configure_csv_limit()
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("local input must contain an array of record objects")
        return self.normalize(rows)
