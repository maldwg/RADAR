from __future__ import annotations

import argparse
from pathlib import Path

from ids_meta.common.cache import PipelineCache
from ids_meta.common.ensemble import ensemble_classifications
from ids_meta.common.evaluation import EvaluationResult, evaluate_results, load_rating_profile
from ids_meta.common.config import ConfigError, Settings, load_prompt_profile, load_settings, load_system_type_profile, load_term_profile
from ids_meta.common.run import RunManifest
from ids_meta.common.text import slug
from ids_meta.common.logging import configure_logging, logger
from ids_meta.github.collect import collect as collect_github
from ids_meta.github.pipeline import classify as classify_github
from ids_meta.github.pipeline import export as export_github
from ids_meta.github.pipeline import prefilter as prefilter_github
from ids_meta.github.reconcile import reconcile as reconcile_github
from ids_meta.github.schema import CLASSIFICATION_COLUMNS as GITHUB_CLASSIFICATION_COLUMNS
from ids_meta.openalex.collect import collect as collect_openalex
from ids_meta.openalex.pipeline import classify as classify_openalex
from ids_meta.openalex.pipeline import export as export_openalex
from ids_meta.openalex.pipeline import export_survey_sources as export_openalex_surveys
from ids_meta.openalex.pipeline import prefilter as prefilter_openalex
from ids_meta.openalex.reconcile import reconcile as reconcile_openalex
from ids_meta.openalex.schema import CLASSIFICATION_COLUMNS as OPENALEX_CLASSIFICATION_COLUMNS

LOG = logger("cli")


def _path_with_default(settings: Settings, key: str, default: str) -> Path:
    """Resolve an optional configured path while preserving older custom configs."""
    value = settings.section("paths").get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"paths.{key} must be a non-empty string")
    path = Path(value)
    return path if path.is_absolute() else settings.root / path


def _default_paths(settings: Settings, source: str) -> dict[str, Path]:
    """Build the default cache, CSV, plot, and comparison paths for a source."""
    interim = settings.path("interim_dir") / source
    processed = settings.path("processed_dir") / source
    plots = settings.path("plots_dir") / source
    evaluation = _path_with_default(settings, "evaluation_dir", "artifacts/evaluation") / source
    if source == "github":
        return {
            "cache": settings.path("github_cache"), "candidates": interim / "candidates.csv",
            "classifications": interim / "classifications.csv", "reconciled": interim / "reconciled.csv",
            "catalogue": processed / "repositories.csv",
            "plots": plots, "evaluation": evaluation,
            "comparison": settings.path("plots_dir") / "comparisons" / source,
        }
    return {
        "cache": settings.path("openalex_cache"), "candidates": interim / "paper_candidates.csv",
        "surveys": interim / "survey_sources.csv",
        "classifications": interim / "classifications.csv", "reconciled": interim / "reconciled.csv",
        "catalogue": processed / "base_papers.csv",
        "adjacent": processed / "adjacent_base_papers.csv", "plots": plots, "evaluation": evaluation,
        "comparison": settings.path("plots_dir") / "comparisons" / source,
    }


def _configured_models(settings: Settings) -> list[str]:
    """Return validated, de-duplicated model names from scalar or list configuration."""
    config = settings.section("ollama")
    raw = config.get("models")
    if raw in (None, []):
        raw = [config.get("model", "")]
    if not isinstance(raw, list) or any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ConfigError("ollama.models must be an array of non-empty strings")
    models = list(dict.fromkeys(value.strip() for value in raw))
    if not models:
        raise ConfigError("at least one Ollama model must be configured")
    return models


def _model_slugs(models: list[str]) -> dict[str, str]:
    """Create filesystem-safe model labels while rejecting ambiguous collisions."""
    labels = {model: slug(model) for model in models}
    if len(set(labels.values())) != len(labels):
        raise ConfigError("configured Ollama model names must have distinct filesystem slugs")
    return labels


def _model_ollama_config(settings: Settings, model: str) -> dict[str, object]:
    """Return one model-specific copy of the shared Ollama configuration."""
    config = dict(settings.section("ollama"))
    config.pop("models", None)
    config["model"] = model
    return config


def _ensemble_config(settings: Settings, models: list[str]) -> tuple[bool, list[str], float]:
    """Validate ensemble enablement, member subset, and agreement threshold."""
    config = settings.section("ensemble")
    enabled_value = config.get("enabled", False)
    if not isinstance(enabled_value, bool):
        raise ConfigError("ensemble.enabled must be true or false")
    enabled = enabled_value
    raw_members = config.get("models", [])
    if not isinstance(raw_members, list) or any(not isinstance(value, str) or not value.strip() for value in raw_members):
        raise ConfigError("ensemble.models must be an array of non-empty strings")
    members = list(dict.fromkeys(value.strip() for value in raw_members)) or list(models)
    unknown = [model for model in members if model not in models]
    if enabled and "ensemble" in models:
        raise ConfigError("ensemble is a reserved model name when ensemble mode is enabled")
    if unknown:
        raise ConfigError("ensemble.models contains models absent from ollama.models: " + ",".join(unknown))
    raw_threshold = config.get("threshold", 0.5)
    if isinstance(raw_threshold, bool):
        raise ConfigError("ensemble.threshold must be numeric")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ConfigError("ensemble.threshold must be numeric") from exc
    if not 0 < threshold <= 1:
        raise ConfigError("ensemble.threshold must be greater than 0 and at most 1")
    return enabled, members, threshold


def _variant_paths(settings: Settings, source: str, variant: Path) -> dict[str, Path]:
    """Build model- or ensemble-specific output paths while sharing source inputs."""
    paths = _default_paths(settings, source)
    interim = settings.path("interim_dir") / source / variant
    processed = settings.path("processed_dir") / source / variant
    plots = settings.path("plots_dir") / source / variant
    evaluation = _path_with_default(settings, "evaluation_dir", "artifacts/evaluation") / source / variant
    paths.update({"classifications": interim / "classifications.csv", "reconciled": interim / "reconciled.csv", "plots": plots, "evaluation": evaluation})
    if source == "github":
        paths["catalogue"] = processed / "repositories.csv"
    else:
        paths.update({"catalogue": processed / "base_papers.csv", "adjacent": processed / "adjacent_base_papers.csv"})
    return paths


def _execution_paths(settings: Settings, source: str, models: list[str]) -> dict[str, dict[str, Path]]:
    """Resolve backward-compatible single-model paths or isolated multi-model paths."""
    if len(models) == 1:
        return {models[0]: _default_paths(settings, source)}
    labels = _model_slugs(models)
    return {model: _variant_paths(settings, source, Path("models") / labels[model]) for model in models}


def _reject_ambiguous_custom_paths(args: argparse.Namespace) -> None:
    """Reject a single custom path when an action emits or consumes many model files."""
    output_is_ambiguous = getattr(args, "output", None) and args.action in {"classify", "evaluate", "reconcile", "export", "plot", "run"}
    input_is_ambiguous = getattr(args, "input", None) and args.action in {"evaluate", "reconcile", "export", "plot"}
    if output_is_ambiguous or input_is_ambiguous:
        raise ConfigError("custom paths for model-dependent outputs are unavailable in multi-model or ensemble mode")


def _manifest(settings: Settings, source: str, stage: str) -> RunManifest:
    """Create a run manifest initialized with the active configuration."""
    return RunManifest(settings.root, settings.path("runs_dir"), source, stage, settings.values)


def _pipeline_limits(args: argparse.Namespace, config: dict[str, object]) -> tuple[int, int | None]:
    """Resolve the finite collection cap and optional all-eligible classification cap."""
    legacy = getattr(args, "limit", None)
    collection = getattr(args, "collect_limit", None)
    classification = getattr(args, "classify_limit", None)
    if collection is None:
        collection = legacy if legacy is not None and args.action in {"collect", "run"} else config.get("collection_limit", 1000)
    classify_all = bool(getattr(args, "classify_all", False))
    if classify_all:
        classification = None
    elif classification is None:
        classification = legacy if legacy is not None and args.action in {"classify", "run"} else config.get("classification_limit", 100)
    try:
        collection_limit = int(collection)
    except (TypeError, ValueError) as exc:
        raise ConfigError("collection_limit must be an integer") from exc
    if collection_limit < 1:
        raise ConfigError("collection_limit must be at least 1")
    if isinstance(classification, str) and classification.strip().casefold() in {"all", "all_eligible", "unlimited"}:
        classification_limit = None
    else:
        try:
            classification_limit = int(classification) if classification is not None else None
        except (TypeError, ValueError) as exc:
            raise ConfigError("classification_limit must be an integer or 'all'") from exc
        if classification_limit is not None and classification_limit < 1:
            raise ConfigError("classification_limit must be at least 1 or 'all'")
    return collection_limit, classification_limit


def _limit_label(limit: int | None) -> str:
    """Format a classification cap for logs and run manifests."""
    return str(limit) if limit is not None else "all_eligible"


def _shortlist_count(eligible: int, limit: int | None) -> int:
    """Return how many eligible rows remain after applying an optional cap."""
    return eligible if limit is None else min(eligible, limit)


def _evaluation_enabled(settings: Settings) -> bool:
    """Return whether full pipeline runs should execute the evaluation stage."""
    enabled = settings.section("evaluation").get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("evaluation.enabled must be true or false")
    return enabled


def _evaluation_candidates(args: argparse.Namespace, default: Path) -> Path:
    """Resolve the candidate audit matching an explicit evaluation or full run."""
    if getattr(args, "candidates", None):
        return Path(args.candidates)
    if args.action == "run" and getattr(args, "input", None):
        return Path(args.input)
    return default


def _evaluate(
    settings: Settings, source: str, classifications: Path, candidates: Path, output_dir: Path,
) -> EvaluationResult:
    """Evaluate one source/model result against its centrally configured ratings."""
    profile = load_rating_profile(_path_with_default(settings, f"{source}_ratings", f"config/evaluation/{source}.toml"), source)
    result = evaluate_results(profile, classifications, output_dir, candidates)
    from ids_meta.plotting.evaluation import build as build_evaluation_plots

    plot_outputs = build_evaluation_plots(output_dir, settings.section("plots"))
    plot_tables = sorted((output_dir / "plot_data").glob("*.csv"))
    return EvaluationResult(
        result.metrics, result.instances, result.field_metrics, result.confusion,
        [*result.outputs, *plot_outputs, *plot_tables],
    )


def _record_evaluation(manifest: RunManifest, result: EvaluationResult, suffix: str = "") -> None:
    """Record compact operational evaluation counts in a run manifest."""
    label = f"_{suffix}" if suffix else ""
    operational = result.metrics["operational"]
    manifest.counts[f"evaluation_rated{label}"] = int(result.metrics["rated_entities"])
    for name in ("tp", "fp", "tn", "fn"):
        manifest.counts[f"evaluation_{name}{label}"] = int(operational[name])
    manifest.counts[f"evaluation_files{label}"] = len(result.outputs)


def _execute_github_models(args: argparse.Namespace, settings: Settings, models: list[str]) -> dict[str, int]:
    """Execute GitHub model variants plus an optional deterministic ensemble."""
    _reject_ambiguous_custom_paths(args)
    base_paths = _default_paths(settings, "github")
    model_paths = _execution_paths(settings, "github", models)
    ensemble_enabled, ensemble_members, threshold = _ensemble_config(settings, models)
    ensemble_paths = _variant_paths(settings, "github", Path("ensemble"))
    config, profile = dict(settings.section("github")), load_term_profile(settings, "github")
    prompt_config = load_prompt_profile(settings, "github")
    system_type_profile = load_system_type_profile(settings, "github")
    config.setdefault("progress_every", settings.section("logging").get("progress_every", 25))
    manifest = _manifest(settings, "github", args.action)
    collect_limit, classify_limit = _pipeline_limits(args, config)
    labels = _model_slugs(models)
    manifest.parameters.update({
        "collection_limit": collect_limit, "classification_limit": classify_limit,
        "classification_mode": "all_eligible" if classify_limit is None else "top_n",
        "classification_eligible_bands": ["candidate", "borderline"],
        "classification_minimum_score": config.get("borderline_threshold"),
        "models": models, "ensemble_enabled": ensemble_enabled,
        "ensemble_models": ensemble_members if ensemble_enabled else [],
        "ensemble_threshold": threshold if ensemble_enabled else None,
    })
    LOG.info("starting multi-model command source=github stage=%s run_id=%s models=%s ensemble=%s", args.action, manifest.run_id, ",".join(models), ensemble_enabled)
    with PipelineCache(base_paths["cache"]) as cache:
        if args.action in {"collect", "run"}:
            counts = {"cached": sum(1 for _ in cache.records("github.repository"))} if args.dry_run else collect_github(config, profile, cache, collect_limit, resume=args.resume)
            manifest.counts.update(counts)
        if args.action in {"prefilter", "run"}:
            output_path = Path(args.output) if args.output and args.action == "prefilter" else base_paths["candidates"]
            rows = prefilter_github(config, profile, cache, output_path, manifest.run_id)
            manifest.counts["candidates_all_bands"] = len(rows)
            eligible = sum(str(row.get("prefilter_band", "")) in {"candidate", "borderline"} for row in rows)
            manifest.counts.update({"classification_eligible": eligible, "classification_shortlist": _shortlist_count(eligible, classify_limit)})
        if args.action in {"classify", "run"}:
            input_path = Path(args.input) if args.input else base_paths["candidates"]
            for model in models:
                rows = classify_github(input_path, model_paths[model]["classifications"], cache, _model_ollama_config(settings, model), classify_limit, args.resume, args.dry_run, prompt_config=prompt_config, system_type_profile=system_type_profile, consecutive_runs=int(config.get("consecutive_runs", 1)), consensus_threshold=float(config.get("consensus_threshold", .5)), github_config=config)
                manifest.counts["classifications_" + labels[model]] = len(rows)
        if ensemble_enabled and args.action in {"classify", "reconcile", "run"}:
            selected = {model: model_paths[model]["classifications"] for model in ensemble_members}
            rows = ensemble_classifications("github", selected, ensemble_paths["classifications"], threshold, GITHUB_CLASSIFICATION_COLUMNS)
            manifest.counts["classifications_ensemble"] = len(rows)
        if args.action == "evaluate" or (args.action == "run" and _evaluation_enabled(settings) and not args.dry_run):
            candidates_path = _evaluation_candidates(args, base_paths["candidates"])
            for model in models:
                result = _evaluate(settings, "github", model_paths[model]["classifications"], candidates_path, model_paths[model]["evaluation"])
                _record_evaluation(manifest, result, labels[model])
            if ensemble_enabled:
                result = _evaluate(settings, "github", ensemble_paths["classifications"], candidates_path, ensemble_paths["evaluation"])
                _record_evaluation(manifest, result, "ensemble")
        if args.action in {"reconcile", "run"} and not (args.action == "run" and args.dry_run):
            for model in models:
                rows = reconcile_github(model_paths[model]["classifications"], model_paths[model]["reconciled"], cache, _model_ollama_config(settings, model), config, args.dry_run, prompt_config=prompt_config)
                manifest.counts["reconciled_" + labels[model]] = len(rows)
            if ensemble_enabled:
                rows = reconcile_github(ensemble_paths["classifications"], ensemble_paths["reconciled"], cache, _model_ollama_config(settings, ensemble_members[0]), config, True, prompt_config=prompt_config)
                manifest.counts["reconciled_ensemble"] = len(rows)
        if args.action in {"export", "run"} and not (args.action == "run" and args.dry_run):
            for model in models:
                rows = export_github(model_paths[model]["reconciled"], model_paths[model]["catalogue"], system_type_profile)
                manifest.counts["accepted_" + labels[model]] = len(rows)
            if ensemble_enabled:
                rows = export_github(ensemble_paths["reconciled"], ensemble_paths["catalogue"], system_type_profile)
                manifest.counts["accepted_ensemble"] = len(rows)
        if args.action in {"plot", "run"} and not (args.action == "run" and args.dry_run):
            from ids_meta.plotting.compare import build_models as build_model_comparison
            from ids_meta.plotting.github import build as build_github_plots

            comparison_inputs: dict[str, Path] = {}
            generated_count = 0
            for model in models:
                generated_count += len(build_github_plots(model_paths[model]["catalogue"], model_paths[model]["plots"], settings.section("plots"), system_type_profile, model_paths[model]["classifications"]))
                comparison_inputs[model] = model_paths[model]["classifications"]
            if ensemble_enabled:
                generated_count += len(build_github_plots(ensemble_paths["catalogue"], ensemble_paths["plots"], settings.section("plots"), system_type_profile, ensemble_paths["classifications"]))
                comparison_inputs["ensemble"] = ensemble_paths["classifications"]
            comparison_dir = base_paths["comparison"] / "models"
            generated_count += len(build_model_comparison("github", comparison_inputs, comparison_dir, settings.section("plots")))
            manifest.counts["plot_files"] = generated_count
        manifest.coverage = cache.coverage_rows("github")
    manifest.status = "complete"
    manifest.write()
    LOG.info("completed multi-model command source=github stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def _execute_github(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Execute the requested GitHub pipeline stage or complete workflow."""
    models = _configured_models(settings)
    ensemble_enabled, _members, _threshold = _ensemble_config(settings, models)
    if len(models) > 1 or ensemble_enabled:
        return _execute_github_models(args, settings, models)
    paths = _default_paths(settings, "github")
    config, profile = dict(settings.section("github")), load_term_profile(settings, "github")
    prompt_config = load_prompt_profile(settings, "github")
    system_type_profile = load_system_type_profile(settings, "github")
    config.setdefault("progress_every", settings.section("logging").get("progress_every", 25))
    manifest = _manifest(settings, "github", args.action)
    collect_limit, classify_limit = _pipeline_limits(args, config)
    classification_label = _limit_label(classify_limit)
    manifest.parameters.update({
        "collection_limit": collect_limit, "classification_limit": classify_limit,
        "classification_mode": "all_eligible" if classify_limit is None else "top_n",
        "classification_eligible_bands": ["candidate", "borderline"],
        "classification_minimum_score": config.get("borderline_threshold"),
    })
    LOG.info("starting command source=github stage=%s run_id=%s collection_limit=%d classification_limit=%s eligibility=candidate_or_borderline minimum_score=%s resume=%s dry_run=%s", args.action, manifest.run_id, collect_limit, classification_label, config.get("borderline_threshold", "unknown"), args.resume, args.dry_run)
    with PipelineCache(paths["cache"]) as cache:
        if args.action in {"collect", "run"}:
            counts = {"cached": sum(1 for _ in cache.records("github.repository"))} if args.dry_run else collect_github(config, profile, cache, collect_limit, resume=args.resume)
            manifest.counts.update(counts)
        if args.action in {"prefilter", "run"}:
            rows = prefilter_github(config, profile, cache, Path(args.output) if args.output and args.action == "prefilter" else paths["candidates"], manifest.run_id)
            manifest.counts["candidates_all_bands"] = len(rows)
            eligible = sum(str(row.get("prefilter_band", "")) in {"candidate", "borderline"} for row in rows)
            manifest.counts.update({"classification_eligible": eligible, "classification_shortlist": _shortlist_count(eligible, classify_limit)})
        if args.action in {"classify", "run"}:
            input_path = Path(args.input) if args.input else paths["candidates"]
            output_path = Path(args.output) if args.output and args.action == "classify" else paths["classifications"]
            rows = classify_github(input_path, output_path, cache, settings.section("ollama"), classify_limit, args.resume, args.dry_run, prompt_config=prompt_config, system_type_profile=system_type_profile, consecutive_runs=int(config.get("consecutive_runs", 1)), consensus_threshold=float(config.get("consensus_threshold", .5)), github_config=config)
            manifest.counts["classifications"] = len(rows)
        if args.action == "evaluate" or (args.action == "run" and _evaluation_enabled(settings) and not args.dry_run):
            classifications_path = Path(args.input) if args.input and args.action == "evaluate" else paths["classifications"]
            candidates_path = _evaluation_candidates(args, paths["candidates"])
            output_dir = Path(args.output) if args.output and args.action == "evaluate" else paths["evaluation"]
            _record_evaluation(manifest, _evaluate(settings, "github", classifications_path, candidates_path, output_dir))
        if args.action in {"reconcile", "run"} and not (args.action == "run" and args.dry_run):
            input_path = Path(args.input) if args.input else paths["classifications"]
            output_path = Path(args.output) if args.output and args.action == "reconcile" else paths["reconciled"]
            rows = reconcile_github(input_path, output_path, cache, settings.section("ollama"), config, args.dry_run, prompt_config=prompt_config)
            manifest.counts.update({"reconciled_rows": len(rows), "project_families": len({row.get("project_id") for row in rows if row.get("project_id")})})
        if args.action in {"export", "run"} and not (args.action == "run" and args.dry_run):
            input_path = Path(args.input) if args.input else paths["reconciled"]
            output_path = Path(args.output) if args.output and args.action == "export" else paths["catalogue"]
            rows = export_github(input_path, output_path, system_type_profile)
            manifest.counts["accepted"] = len(rows)
        if args.action in {"plot", "run"} and not (args.action == "run" and args.dry_run):
            from ids_meta.plotting.github import build as build_github_plots

            input_path = Path(args.input) if args.input else paths["catalogue"]
            output_path = Path(args.output) if args.output else paths["plots"]
            classifications_path = paths["classifications"] if paths["classifications"].is_file() else None
            generated = build_github_plots(input_path, output_path, settings.section("plots"), system_type_profile, classifications_path)
            manifest.counts["plot_files"] = len(generated)
        manifest.coverage = cache.coverage_rows("github")
    manifest.status = "complete"; manifest.write()
    LOG.info("completed command source=github stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def _execute_openalex_models(args: argparse.Namespace, settings: Settings, models: list[str]) -> dict[str, int]:
    """Execute OpenAlex model variants plus an optional deterministic ensemble."""
    _reject_ambiguous_custom_paths(args)
    base_paths = _default_paths(settings, "openalex")
    model_paths = _execution_paths(settings, "openalex", models)
    ensemble_enabled, ensemble_members, threshold = _ensemble_config(settings, models)
    ensemble_paths = _variant_paths(settings, "openalex", Path("ensemble"))
    config, profile = dict(settings.section("openalex")), load_term_profile(settings, "openalex")
    prompt_config = load_prompt_profile(settings, "openalex")
    config.setdefault("progress_every", settings.section("logging").get("progress_every", 25))
    manifest = _manifest(settings, "openalex", args.action)
    collect_limit, classify_limit = _pipeline_limits(args, config)
    labels = _model_slugs(models)
    manifest.parameters.update({
        "collection_limit": collect_limit, "classification_limit": classify_limit,
        "classification_mode": "all_eligible" if classify_limit is None else "top_n",
        "classification_eligible_bands": ["candidate", "borderline"],
        "classification_minimum_score": config.get("borderline_threshold"),
        "models": models, "ensemble_enabled": ensemble_enabled,
        "ensemble_models": ensemble_members if ensemble_enabled else [],
        "ensemble_threshold": threshold if ensemble_enabled else None,
    })
    LOG.info("starting multi-model command source=openalex stage=%s run_id=%s models=%s ensemble=%s", args.action, manifest.run_id, ",".join(models), ensemble_enabled)
    with PipelineCache(base_paths["cache"]) as cache:
        if args.action in {"collect", "run"}:
            counts = {"cached": sum(1 for _ in cache.records("openalex.work"))} if args.dry_run else collect_openalex(config, profile, cache, collect_limit)
            manifest.counts.update(counts)
        if args.action in {"collect", "prefilter", "run"}:
            survey_rows = export_openalex_surveys(cache, base_paths["surveys"], manifest.run_id)
            manifest.counts["auxiliary_survey_sources"] = len(survey_rows)
        if args.action in {"prefilter", "run"}:
            output_path = Path(args.output) if args.output and args.action == "prefilter" else base_paths["candidates"]
            rows = prefilter_openalex(config, profile, cache, output_path, manifest.run_id)
            manifest.counts["candidates_all_bands"] = len(rows)
            eligible = sum(str(row.get("prefilter_band", "")) in {"candidate", "borderline"} for row in rows)
            manifest.counts.update({"classification_eligible": eligible, "classification_shortlist": _shortlist_count(eligible, classify_limit)})
        if args.action in {"classify", "run"}:
            input_path = Path(args.input) if args.input else base_paths["candidates"]
            for model in models:
                rows = classify_openalex(input_path, model_paths[model]["classifications"], cache, _model_ollama_config(settings, model), classify_limit, args.resume, args.dry_run, prompt_config=prompt_config, openalex_config=config, consecutive_runs=int(config.get("consecutive_runs", 1)), consensus_threshold=float(config.get("consensus_threshold", .5)))
                manifest.counts["classifications_" + labels[model]] = len(rows)
        if ensemble_enabled and args.action in {"classify", "reconcile", "run"}:
            selected = {model: model_paths[model]["classifications"] for model in ensemble_members}
            rows = ensemble_classifications("openalex", selected, ensemble_paths["classifications"], threshold, OPENALEX_CLASSIFICATION_COLUMNS)
            manifest.counts["classifications_ensemble"] = len(rows)
        if args.action == "evaluate" or (args.action == "run" and _evaluation_enabled(settings) and not args.dry_run):
            candidates_path = _evaluation_candidates(args, base_paths["candidates"])
            for model in models:
                result = _evaluate(settings, "openalex", model_paths[model]["classifications"], candidates_path, model_paths[model]["evaluation"])
                _record_evaluation(manifest, result, labels[model])
            if ensemble_enabled:
                result = _evaluate(settings, "openalex", ensemble_paths["classifications"], candidates_path, ensemble_paths["evaluation"])
                _record_evaluation(manifest, result, "ensemble")
        if args.action in {"reconcile", "run"} and not (args.action == "run" and args.dry_run):
            for model in models:
                rows = reconcile_openalex(model_paths[model]["classifications"], model_paths[model]["reconciled"], cache, _model_ollama_config(settings, model), config, args.dry_run, prompt_config=prompt_config)
                manifest.counts["reconciled_" + labels[model]] = len(rows)
            if ensemble_enabled:
                rows = reconcile_openalex(ensemble_paths["classifications"], ensemble_paths["reconciled"], cache, _model_ollama_config(settings, ensemble_members[0]), config, True, prompt_config=prompt_config)
                manifest.counts["reconciled_ensemble"] = len(rows)
        if args.action in {"export", "run"} and not (args.action == "run" and args.dry_run):
            for model in models:
                main, adjacent = export_openalex(model_paths[model]["reconciled"], model_paths[model]["catalogue"], model_paths[model]["adjacent"])
                manifest.counts["accepted_core_" + labels[model]] = len(main)
                manifest.counts["accepted_adjacent_" + labels[model]] = len(adjacent)
            if ensemble_enabled:
                main, adjacent = export_openalex(ensemble_paths["reconciled"], ensemble_paths["catalogue"], ensemble_paths["adjacent"])
                manifest.counts["accepted_core_ensemble"] = len(main)
                manifest.counts["accepted_adjacent_ensemble"] = len(adjacent)
        if args.action in {"plot", "run"} and not (args.action == "run" and args.dry_run):
            from ids_meta.plotting.compare import build_models as build_model_comparison
            from ids_meta.plotting.openalex import build as build_openalex_plots

            comparison_inputs: dict[str, Path] = {}
            generated_count = 0
            for model in models:
                generated_count += len(build_openalex_plots(model_paths[model]["catalogue"], model_paths[model]["adjacent"], model_paths[model]["plots"], settings.section("plots"), model_paths[model]["classifications"], base_paths["surveys"]))
                comparison_inputs[model] = model_paths[model]["classifications"]
            if ensemble_enabled:
                generated_count += len(build_openalex_plots(ensemble_paths["catalogue"], ensemble_paths["adjacent"], ensemble_paths["plots"], settings.section("plots"), ensemble_paths["classifications"], base_paths["surveys"]))
                comparison_inputs["ensemble"] = ensemble_paths["classifications"]
            comparison_dir = base_paths["comparison"] / "models"
            generated_count += len(build_model_comparison("openalex", comparison_inputs, comparison_dir, settings.section("plots")))
            manifest.counts["plot_files"] = generated_count
        manifest.coverage = cache.coverage_rows("openalex")
    manifest.status = "complete"
    manifest.write()
    LOG.info("completed multi-model command source=openalex stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def _execute_openalex(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Execute the requested OpenAlex pipeline stage or complete workflow."""
    models = _configured_models(settings)
    ensemble_enabled, _members, _threshold = _ensemble_config(settings, models)
    if len(models) > 1 or ensemble_enabled:
        return _execute_openalex_models(args, settings, models)
    paths = _default_paths(settings, "openalex")
    config, profile = dict(settings.section("openalex")), load_term_profile(settings, "openalex")
    prompt_config = load_prompt_profile(settings, "openalex")
    config.setdefault("progress_every", settings.section("logging").get("progress_every", 25))
    manifest = _manifest(settings, "openalex", args.action)
    collect_limit, classify_limit = _pipeline_limits(args, config)
    classification_label = _limit_label(classify_limit)
    manifest.parameters.update({
        "collection_limit": collect_limit, "classification_limit": classify_limit,
        "classification_mode": "all_eligible" if classify_limit is None else "top_n",
        "classification_eligible_bands": ["candidate", "borderline"],
        "classification_minimum_score": config.get("borderline_threshold"),
    })
    LOG.info("starting command source=openalex stage=%s run_id=%s collection_limit=%d classification_limit=%s eligibility=candidate_or_borderline minimum_score=%s resume=%s dry_run=%s", args.action, manifest.run_id, collect_limit, classification_label, config.get("borderline_threshold", "unknown"), args.resume, args.dry_run)
    with PipelineCache(paths["cache"]) as cache:
        if args.action in {"collect", "run"}:
            counts = {"cached": sum(1 for _ in cache.records("openalex.work"))} if args.dry_run else collect_openalex(config, profile, cache, collect_limit)
            manifest.counts.update(counts)
        if args.action in {"collect", "prefilter", "run"}:
            survey_rows = export_openalex_surveys(cache, paths["surveys"], manifest.run_id)
            manifest.counts["auxiliary_survey_sources"] = len(survey_rows)
        if args.action in {"prefilter", "run"}:
            rows = prefilter_openalex(config, profile, cache, Path(args.output) if args.output and args.action == "prefilter" else paths["candidates"], manifest.run_id)
            manifest.counts["candidates_all_bands"] = len(rows)
            eligible = sum(str(row.get("prefilter_band", "")) in {"candidate", "borderline"} for row in rows)
            manifest.counts.update({"classification_eligible": eligible, "classification_shortlist": _shortlist_count(eligible, classify_limit)})
        if args.action in {"classify", "run"}:
            input_path = Path(args.input) if args.input else paths["candidates"]
            output_path = Path(args.output) if args.output and args.action == "classify" else paths["classifications"]
            rows = classify_openalex(input_path, output_path, cache, settings.section("ollama"), classify_limit, args.resume, args.dry_run, prompt_config=prompt_config, openalex_config=config, consecutive_runs=int(config.get("consecutive_runs", 1)), consensus_threshold=float(config.get("consensus_threshold", .5)))
            manifest.counts["classifications"] = len(rows)
        if args.action == "evaluate" or (args.action == "run" and _evaluation_enabled(settings) and not args.dry_run):
            classifications_path = Path(args.input) if args.input and args.action == "evaluate" else paths["classifications"]
            candidates_path = _evaluation_candidates(args, paths["candidates"])
            output_dir = Path(args.output) if args.output and args.action == "evaluate" else paths["evaluation"]
            _record_evaluation(manifest, _evaluate(settings, "openalex", classifications_path, candidates_path, output_dir))
        if args.action in {"reconcile", "run"} and not (args.action == "run" and args.dry_run):
            input_path = Path(args.input) if args.input else paths["classifications"]
            output_path = Path(args.output) if args.output and args.action == "reconcile" else paths["reconciled"]
            rows = reconcile_openalex(input_path, output_path, cache, settings.section("ollama"), config, args.dry_run, prompt_config=prompt_config)
            manifest.counts.update({"reconciled_rows": len(rows), "publication_families": len({row.get("publication_family_id") for row in rows if row.get("publication_family_id")})})
        if args.action in {"export", "run"} and not (args.action == "run" and args.dry_run):
            input_path = Path(args.input) if args.input else paths["reconciled"]
            main_path = Path(args.output) if args.output and args.action == "export" else paths["catalogue"]
            main, adjacent = export_openalex(input_path, main_path, paths["adjacent"])
            manifest.counts.update({"accepted_core": len(main), "accepted_adjacent": len(adjacent)})
        if args.action in {"plot", "run"} and not (args.action == "run" and args.dry_run):
            from ids_meta.plotting.openalex import build as build_openalex_plots

            input_path = Path(args.input) if args.input else paths["catalogue"]
            output_path = Path(args.output) if args.output else paths["plots"]
            generated = build_openalex_plots(input_path, paths["adjacent"], output_path, settings.section("plots"), paths["classifications"], paths["surveys"])
            manifest.counts["plot_files"] = len(generated)
        manifest.coverage = cache.coverage_rows("openalex")
    manifest.status = "complete"; manifest.write()
    LOG.info("completed command source=openalex stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser and all source-stage subcommands."""
    parser = argparse.ArgumentParser(prog="ids-meta", description="IDS meta-study discovery, classification, export, and plotting")
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="stdout logging verbosity; defaults to config logging.level")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for source in ("github", "openalex"):
        source_parser = subparsers.add_parser(source)
        actions = source_parser.add_subparsers(dest="action", required=True)
        for action in ("collect", "prefilter", "classify", "evaluate", "reconcile", "export", "plot", "run"):
            action_parser = actions.add_parser(action)
            action_parser.add_argument("--limit", type=int, help="legacy stage limit; for run, applies to both collection and classification")
            if action in {"collect", "run"}:
                action_parser.add_argument("--collect-limit", "--collection-limit", dest="collect_limit", type=int, help="maximum records collected from the source API")
            if action in {"classify", "run"}:
                classification_group = action_parser.add_mutually_exclusive_group()
                classification_group.add_argument("--classify-limit", "--classification-limit", dest="classify_limit", type=int, help="maximum top-ranked eligible candidates sent to the LLM")
                classification_group.add_argument("--classify-all", action="store_true", help="classify every candidate and borderline row above the configured rejection threshold")
            action_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
            action_parser.add_argument("--dry-run", action="store_true")
            action_parser.add_argument("--input", type=Path)
            action_parser.add_argument("--output", type=Path)
            if action == "evaluate":
                action_parser.add_argument("--candidates", type=Path, help="candidate CSV used to distinguish not retrieved from not classified")
    architecture = subparsers.add_parser("architecture")
    architecture.add_argument("--output", type=Path, help="output directory; defaults to the repository root")
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("source", choices=("github", "openalex"))
    comparison.add_argument("--input", type=Path)
    comparison.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, run the selected command, and return an exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("limit", "collect_limit", "classify_limit"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    try:
        configure_logging("INFO")
        settings = load_settings(args.config)
        configure_logging(args.log_level or str(settings.section("logging").get("level", "INFO")))
        LOG.info("loaded configuration %s", Path(args.config).resolve())
        if args.command == "github":
            counts = _execute_github(args, settings)
        elif args.command == "openalex":
            counts = _execute_openalex(args, settings)
        elif args.command == "architecture":
            from ids_meta.architecture import build as build_architecture

            files = build_architecture(args.output or settings.root)
            counts = {"architecture_files": len(files)}
        else:
            from ids_meta.plotting.compare import build as build_comparison
            from ids_meta.plotting.compare import build_models as build_model_comparison

            paths = _default_paths(settings, args.source)
            models = _configured_models(settings)
            ensemble_enabled, _members, _threshold = _ensemble_config(settings, models)
            if args.input or (len(models) == 1 and not ensemble_enabled):
                input_path = args.input or paths["classifications"]
                output_path = args.output or paths["comparison"]
                files = build_comparison(args.source, input_path, output_path, settings.section("plots"))
            else:
                model_paths = _execution_paths(settings, args.source, models)
                inputs = {model: model_paths[model]["classifications"] for model in models}
                if ensemble_enabled:
                    inputs["ensemble"] = _variant_paths(settings, args.source, Path("ensemble"))["classifications"]
                output_path = args.output or paths["comparison"] / "models"
                files = build_model_comparison(args.source, inputs, output_path, settings.section("plots"))
            counts = {"plot_files": len(files)}
        LOG.info("command summary %s", "; ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        return 0
    except (ConfigError, ValueError, OSError, RuntimeError) as exc:
        # Logging intentionally remains on stdout so redirected run logs stay complete.
        LOG.error("command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
