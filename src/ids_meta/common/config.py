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
