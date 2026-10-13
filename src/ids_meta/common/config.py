from __future__ import annotations

import os
import copy
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


DEFAULT_STUDY = Path("config/studies/ids_repositories/study.toml")


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

    def require_source(self, source: str) -> None:
        """Prevent a study from executing an unselected provider."""
        study = self.section("study")
        if study and source not in study["sources"]:
            raise ConfigError(f"study {study['id']!r} does not select source {source!r}")


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file or raise a configuration-specific error when absent."""
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _reference(parent: Path, value: Any, label: str) -> Path:
    """Resolve explicit file references against the declaring configuration."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path")
    return (parent / value).resolve()


def _compose(target: dict[str, Any], incoming: dict[str, Any], label: str = "") -> None:
    """Combine disjoint settings, rejecting ambiguous implicit overrides."""
    for key, value in incoming.items():
        name = f"{label}.{key}" if label else key
        if key not in target:
            target[key] = copy.deepcopy(value)
        elif isinstance(target[key], dict) and isinstance(value, dict):
            _compose(target[key], value, name)
        else:
            raise ConfigError(f"setting {name} is declared in more than one profile")


def load_settings(path: Path | str = DEFAULT_STUDY) -> Settings:
    """Resolve a study manifest into shared execution settings and local profiles.

    All file references are relative to the file declaring them. The flat paths
    in Settings are the execution representation, not another config format.
    """
    config_path = Path(path).resolve()
    manifest = _read_toml(config_path)
    if manifest.get("schema_version") != "2.0":
        raise ConfigError("study manifest requires schema_version 2.0")
    allowed = {"schema_version", "id", "workflow", "project_root", "runtime", "models", "methods", "reporting",
        "output_root", "sources", "evaluation", "protocol", "taxonomy", "collection", "assessment"}
    if set(manifest) - allowed:
        raise ConfigError(f"unknown study settings: {', '.join(sorted(set(manifest) - allowed))}")
    study_id = manifest.get("id")
    if not isinstance(study_id, str) or not re.fullmatch(r"[a-z][a-z0-9_\-]*", study_id):
        raise ConfigError("study id must be a filesystem-safe lowercase identifier")
    workflow = manifest.get("workflow")
    supported = {"repository_catalogue": {"github"}, "dataset_catalogue": {"openalex"}, "systematic_review": {"openalex"}}
    if not isinstance(workflow, str) or workflow not in supported:
        raise ConfigError(f"unsupported study workflow: {workflow!r}")
    root = _reference(config_path.parent, manifest.get("project_root"), "project_root")
    if not root.is_dir():
        raise ConfigError(f"project_root is not a directory: {root}")
    values: dict[str, Any] = {"schema_version": "2.0"}
    fingerprints = {str(config_path): hashlib.sha256(config_path.read_bytes()).hexdigest()}

    def read(value: Any, label: str) -> tuple[Path, dict[str, Any]]:
        """Read and fingerprint a profile referenced by the study manifest."""
        profile_path = _reference(config_path.parent, value, label)
        profile = _read_toml(profile_path)
        if profile.get("schema_version") != "1.0":
            raise ConfigError(f"{label} profile requires schema_version 1.0: {profile_path}")
        fingerprints[str(profile_path)] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        return profile_path, profile

    for key in ("runtime", "models", "methods", "reporting"):
        if key in manifest:
            _, profile = read(manifest[key], key)
            _compose(values, {k: v for k, v in profile.items() if k != "schema_version"})
        elif key in {"runtime", "models", "reporting"}:
            raise ConfigError(f"study requires {key}")
    output = _reference(config_path.parent, manifest.get("output_root"), "output_root")
    paths = {key: str(output / directory) for key, directory in {
        "raw_dir": "raw", "interim_dir": "interim", "processed_dir": "processed",
        "plots_dir": "plots", "runs_dir": "runs", "evaluation_dir": "evaluation",
    }.items()}
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources or any(not isinstance(item, dict) for item in sources):
        raise ConfigError("study sources must be a non-empty array of tables")
    names = []
    for item in sources:
        source = item.get("name")
        if not isinstance(source, str) or source not in supported[workflow]:
            raise ConfigError(f"source {source!r} is unsupported for workflow {workflow!r}")
        if source in names:
            raise ConfigError(f"duplicate study source: {source}")
        names.append(source)
        keys = {"name", "profile", "settings", "search", "prompts", "taxonomy", "keyword", "nli", "llm_nli_prompt"}
        if set(item) - keys:
            raise ConfigError(f"unknown {source} source settings: {', '.join(sorted(set(item) - keys))}")
        _, provider = read(item.get("profile"), f"sources.{source}.profile")
        if provider.get("name") != source:
            raise ConfigError(f"source profile name must be {source!r}")
        expected_kind = "repository" if workflow == "repository_catalogue" else "paper"
        if provider.get("entity_kind") != expected_kind:
            raise ConfigError(f"source {source!r} must provide {expected_kind} entities")
        source_settings = copy.deepcopy(provider.get("settings", {}))
        if not isinstance(source_settings, dict):
            raise ConfigError(f"sources.{source}.settings must be a table")
        if "settings" in item:
            _, selection = read(item["settings"], f"sources.{source}.settings")
            _compose(source_settings, {k: v for k, v in selection.items() if k != "schema_version"})
        values[source] = source_settings
        paths[f"{source}_cache"] = str(output / "raw" / source / "cache.sqlite3")
        required = {"search"} if workflow == "systematic_review" else {"search", "prompts", "taxonomy", "keyword", "nli", "llm_nli_prompt", "settings"}
        for key in required:
            if key not in item:
                raise ConfigError(f"sources.{source} requires {key}")
        for key, target in {"search": "terms", "prompts": "prompts", "taxonomy": "taxonomy", "keyword": "keyword", "nli": "nli", "llm_nli_prompt": "llm_nli_prompt"}.items():
            if key not in item:
                continue
            profile_path, _ = read(item[key], f"sources.{source}.{key}")
            target_key = target if key == "llm_nli_prompt" else f"{source}_{target}"
            if target_key in paths and paths[target_key] != str(profile_path):
                raise ConfigError("all branches of a study must select the same LLM NLI prompt")
            paths[target_key] = str(profile_path)
            if key == "taxonomy" and workflow == "repository_catalogue":
                paths[f"{source}_system_types"] = str(profile_path)
    for key in ("protocol", "taxonomy"):
        if key in manifest:
            profile_path, _ = read(manifest[key], key)
            paths[key] = str(profile_path)
        elif workflow == "systematic_review":
            raise ConfigError(f"systematic_review requires {key}")
    evaluation = copy.deepcopy(manifest.get("evaluation", {}))
    if not isinstance(evaluation, dict):
        raise ConfigError("evaluation must be a table")
    refs = evaluation.get("reference_sets", [])
    if not isinstance(refs, list) or any(not isinstance(ref, dict) for ref in refs):
        raise ConfigError("evaluation.reference_sets must be an array of tables")
    for ref in refs:
        if ref.get("source") not in names or not isinstance(ref.get("name"), str) or not ref["name"].strip():
            raise ConfigError("each evaluation reference requires a name and selected source")
        profile_path, _ = read(ref.get("profile"), "evaluation.reference_sets.profile")
        ref["profile"] = str(profile_path)
    if "references" in evaluation:
        profile_path, _ = read(evaluation["references"], "evaluation.references")
        evaluation["references"] = str(profile_path)
    values["evaluation"] = evaluation
    values["paths"] = paths
    values["study"] = {"id": study_id, "workflow": workflow, "sources": names,
        "manifest": str(config_path), "output_root": str(output), "input_fingerprints": fingerprints}
    for key in ("collection", "assessment"):
        value = manifest.get(key, {})
        if not isinstance(value, dict):
            raise ConfigError(f"{key} must be a table")
        values[key] = value
    ollama = values.setdefault("ollama", {})
    ollama["url"] = os.getenv("OLLAMA_URL", str(ollama.get("url", "http://127.0.0.1:11434")))
    if os.getenv("OLLAMA_MODEL"):
        ollama["models"] = [os.environ["OLLAMA_MODEL"]]
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
    taxonomy = _read_toml(settings.path(f"{source}_taxonomy"))
    if taxonomy.get("schema_version") != "1.0":
        raise ConfigError("taxonomy profile requires schema_version 1.0")
    tables = taxonomy.get("prompt_tables")
    if not isinstance(tables, dict) or not tables:
        raise ConfigError("taxonomy requires non-empty prompt_tables")
    for table, descriptions in tables.items():
        if not isinstance(descriptions, dict) or not descriptions or any(
            not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
            for k, v in descriptions.items()
        ):
            raise ConfigError(f"taxonomy prompt table {table!r} requires non-empty label descriptions")
        if table in profile:
            raise ConfigError(f"prompt table {table!r} belongs in the study taxonomy")
        profile[table] = descriptions
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
