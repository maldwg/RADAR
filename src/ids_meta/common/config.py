from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    root: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        """Return a named configuration table after validating its mapping type."""
        value = self.values.get(name, {})
        if not isinstance(value, dict):
            raise ConfigError(f"configuration section {name!r} must be a table")
        return value

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the project root when necessary."""
        value = self.section("paths").get(key)
        if not isinstance(value, str) or not value:
            raise ConfigError(f"paths.{key} must be a non-empty string")
        path = Path(value)
        return path if path.is_absolute() else self.root / path


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file or raise a configuration-specific error when absent."""
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_settings(path: Path | str = Path("config/default.toml")) -> Settings:
    """Load, validate, and environment-override the main pipeline settings."""
    config_path = Path(path).resolve()
    values = _read_toml(config_path)
    if str(values.get("schema_version", "")) != "1.0":
        raise ConfigError("only configuration schema_version 1.0 is supported")
    # default.toml lives in <root>/config; custom configs are interpreted from cwd.
    root = config_path.parent.parent if config_path.parent.name == "config" else Path.cwd().resolve()
    ollama = values.setdefault("ollama", {})
    ollama["url"] = os.getenv("OLLAMA_URL", str(ollama.get("url", "http://127.0.0.1:11434")))
    ollama["model"] = os.getenv("OLLAMA_MODEL", str(ollama.get("model", "")))
    if os.getenv("OLLAMA_API_KEY"):
        ollama["api_key"] = os.environ["OLLAMA_API_KEY"]
    return Settings(root=root, values=values)


def load_term_profile(settings: Settings, source: str) -> dict[str, Any]:
    """Load the generic discovery and scoring profile for one source."""
    return _read_toml(settings.path(f"{source}_terms"))


def load_prompt_profile(settings: Settings, source: str) -> dict[str, Any]:
    """Load and validate one source-specific classifier-prompt profile."""
    profile = _read_toml(settings.path(f"{source}_prompts"))
    if str(profile.get("schema_version", "")) != "1.0":
        raise ConfigError(f"only {source} prompt schema_version 1.0 is supported")
    for key in ("template", "reconciliation_template"):
        if not isinstance(profile.get(key), str) or not str(profile[key]).strip():
            raise ConfigError(f"{source} prompt profile requires a non-empty {key}")
        if "${evidence}" not in str(profile[key]):
            raise ConfigError(f"{source} prompt profile {key} must contain ${{evidence}}")
    if source == "github":
        from ids_meta.github.schema import AUDIT_ROLES, CAPABILITIES, CENTRALIZATION, METHODS, OBSERVATION_OBJECTS, SCOPES

        taxonomies = {"roles": AUDIT_ROLES, "scopes": SCOPES, "observation_objects": OBSERVATION_OBJECTS, "capabilities": CAPABILITIES, "methods": METHODS}
        taxonomies["centralization"] = CENTRALIZATION
    elif source == "openalex":
        from ids_meta.openalex.schema import ARTIFACT_TYPES, DATA_FORMS, DOMAINS, ENVIRONMENTS, LABELING, PAPER_ROLES, TIERS

        taxonomies = {
            "roles": PAPER_ROLES, "tiers": TIERS, "artifact_types": ARTIFACT_TYPES,
            "domains": DOMAINS, "data_forms": DATA_FORMS, "environments": ENVIRONMENTS,
            "labeling": LABELING,
        }
    else:
        raise ConfigError(f"unsupported prompt source: {source}")
    for table, values in taxonomies.items():
        descriptions = profile.get(table)
        if not isinstance(descriptions, dict):
            raise ConfigError(f"{source} prompt profile requires a [{table}] table")
        missing = [value for value in values if not str(descriptions.get(value, "")).strip()]
        if missing:
            raise ConfigError(f"{source} prompt profile [{table}] is missing: {', '.join(missing)}")
    return profile


def load_system_type_profile(settings: Settings, source: str) -> dict[str, Any]:
    """Load and validate one source system-type vocabulary used by classifiers."""
    profile = _read_toml(settings.path(f"{source}_system_types"))
    if str(profile.get("schema_version", "")) != "1.0":
        raise ConfigError(f"only {source} system-type schema_version 1.0 is supported")
    types = profile.get("types")
    if not isinstance(types, dict) or not types:
        raise ConfigError(f"{source} system-type profile requires a non-empty [types] table")
    for code, definition in types.items():
        if not isinstance(code, str) or not code.strip() or code.casefold() == "unknown":
            raise ConfigError("system-type codes must be non-empty and must not redefine unknown")
        if not isinstance(definition, dict):
            raise ConfigError(f"system type {code!r} must be a table")
        if not str(definition.get("name", "")).strip() or not str(definition.get("description", "")).strip():
            raise ConfigError(f"system type {code!r} requires name and description")
    return profile
