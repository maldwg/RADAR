from __future__ import annotations

import os
import copy
import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ids_meta.common.screening import load_screening_profile


class ConfigError(ValueError):
    pass


DEFAULT_STUDY = Path("config/studies/ids_repositories/study.toml")

PIPELINE_OPTIONS = {
    "collection": {"collection_limit", "minimum_stars", "query_limit_factor", "search_sort", "search_order",
        "created_from", "max_surveys", "snowball_start_papers", "snowball_max_depth", "survey_reference_fraction",
        "survey_query_limit_factor", "reference_limit_factor", "survey_consensus_threshold", "retrieval_minimum_score",
        "retrieval_minimum_title_tokens", "retrieval_title_weight", "minimum_citations", "candidate_threshold", "borderline_threshold"},
    "evidence": {"max_docs_files", "enrich_uncertain_evidence", "uncertain_max_docs_files", "enrich_missing_abstracts",
        "enrich_uncertain_with_full_text", "evidence_lookup_timeout_seconds", "max_full_text_chars", "fetch_full_text"},
    "classification": {"classification_limit", "consecutive_runs", "consensus_threshold", "classify_borderline", "ensemble"},
    "reconciliation": {"reconcile_with_llm", "canonical_ambiguity_margin", "canonical_role_scores", "canonical_direct_release_score",
        "canonical_core_relevance_score", "canonical_earliest_year_score", "canonical_year_distance_penalty", "canonical_year_distance_cap",
        "canonical_citation_weight"},
}

REVIEW_PIPELINE_OPTIONS = {
    "collection": {"collection_limit", "query_limit_factor"},
    "evidence": {"fetch_full_text", "max_full_text_chars", "evidence_lookup_timeout_seconds"},
    "classification": {"classification_limit", "consecutive_runs", "consensus_threshold", "ensemble"},
    "reconciliation": set(),
}


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
        "output_root", "sources", "evaluation", "pipeline", "screening", "keyword", "nli", "llm_nli_prompt"}
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

    def read(value: Any, label: str, parent: Path = config_path.parent) -> tuple[Path, dict[str, Any]]:
        """Read and fingerprint a profile referenced by the study manifest."""
        profile_path = _reference(parent, value, label)
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
    pipeline_path, pipeline_raw = read(manifest.get("pipeline"), "pipeline")
    if set(pipeline_raw) - {"schema_version", *PIPELINE_OPTIONS}:
        raise ConfigError("pipeline may contain only collection, evidence, classification and reconciliation tables")
    pipeline: dict[str, Any] = {}
    execution: dict[str, Any] = {}
    for stage, allowed_options in PIPELINE_OPTIONS.items():
        options = pipeline_raw.get(stage, {})
        if not isinstance(options, dict) or set(options) - allowed_options:
            raise ConfigError(f"pipeline.{stage} contains unknown or misplaced settings")
        if workflow == "systematic_review" and set(options) - REVIEW_PIPELINE_OPTIONS[stage]:
            raise ConfigError(f"systematic_review does not support pipeline.{stage} settings: "
                + ", ".join(sorted(set(options) - REVIEW_PIPELINE_OPTIONS[stage])))
        pipeline[stage] = copy.deepcopy(options)
        native = {key: value for key, value in options.items() if key != "ensemble"}
        _compose(execution, native)
    ensemble = pipeline["classification"].get("ensemble")
    if ensemble is not None:
        if not isinstance(ensemble, dict):
            raise ConfigError("pipeline.classification.ensemble must be a table")
        _compose(values, {"ensemble": ensemble})
    values["pipeline"] = pipeline
    from ids_meta.common.limits import optional_limit, query_result_limit
    from ids_meta.common.study_consensus import validate_consensus

    query_result_limit(execution, optional_limit(execution.get("collection_limit"), "collection_limit"))
    validate_consensus(execution.get("consecutive_runs", 1), execution.get("consensus_threshold", 0.8))
    if workflow == "systematic_review" and values.get("baselines", {}).get("enabled", False):
        raise ConfigError("systematic_review does not support baseline execution")
    paths["pipeline"] = str(pipeline_path)
    screening_path, _ = read(manifest.get("screening"), "screening")
    screening = load_screening_profile(screening_path)
    expected_strategy = {"repository_catalogue": "repository_role_validation",
        "dataset_catalogue": "dataset_release_validation", "systematic_review": "criterion_assessment"}[workflow]
    if screening.strategy != expected_strategy:
        raise ConfigError(f"workflow {workflow!r} requires screening strategy {expected_strategy!r}")
    paths["screening"] = str(screening_path)
    taxonomy_path, _ = read(str(screening.taxonomy_path), "screening.taxonomy")
    prompt_path, _ = read(str(screening.prompt_path), "screening.prompt")
    paths["taxonomy"], paths["prompts"] = str(taxonomy_path), str(prompt_path)
    values["screening"] = screening.prompt_rules()
    for key in ("keyword", "nli", "llm_nli_prompt"):
        if key in manifest:
            profile_path, _ = read(manifest[key], key)
            paths[key] = str(profile_path)
        elif workflow != "systematic_review":
            raise ConfigError(f"catalogue study requires {key}")
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
        keys = {"name", "profile", "search"}
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
        _compose(source_settings, execution)
        values[source] = source_settings
        paths[f"{source}_cache"] = str(output / "raw" / source / "cache.sqlite3")
        search_path, _ = read(item.get("search"), f"sources.{source}.search")
        paths[f"{source}_terms"] = str(search_path)
        paths[f"{source}_taxonomy"], paths[f"{source}_prompts"] = str(taxonomy_path), str(prompt_path)
        paths[f"{source}_screening"] = str(screening_path)
        if workflow == "repository_catalogue":
            paths[f"{source}_system_types"] = str(taxonomy_path)
        for key in ("keyword", "nli"):
            if key in paths:
                paths[f"{source}_{key}"] = paths[key]
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
    screening = load_screening_profile(settings.path(f"{source}_screening"))
    if screening.prompt_path != settings.path(f"{source}_prompts") or screening.taxonomy_path != settings.path(f"{source}_taxonomy"):
        raise ConfigError("catalogue prompt and taxonomy must match the scientific screening profile")
    profile["screening"] = screening.prompt_rules()
    identifiers = set(re.findall(r"\$\{([^}]+)\}", profile["template"]))
    if "eligibility" not in identifiers:
        missing = {f"eligibility_{key}" for key in screening.eligibility} - identifiers
        if missing:
            raise ConfigError(f"classification prompt omits screening criteria: {', '.join(sorted(missing))}")
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
