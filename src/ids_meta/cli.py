from __future__ import annotations

import argparse
import json
import tomllib
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from ids_meta.common.baselines import (
    classify_keyword_baseline, configured_baseline_methods, load_keyword_baseline_config,
    load_keyword_taxonomy_profile,
)
from ids_meta.common.cache import PipelineCache
from ids_meta.common.ensemble import ensemble_classifications
from ids_meta.common.evaluation import EvaluationResult, evaluate_results, load_rating_profile
from ids_meta.common.config import DEFAULT_STUDY, ConfigError, Settings, load_prompt_profile, load_settings, load_system_type_profile, load_term_profile
from ids_meta.common.csvio import read_csv
from ids_meta.common.limits import optional_limit, limit_label
from ids_meta.common.run import RunManifest
from ids_meta.common.text import slug
from ids_meta.common.logging import configure_logging, logger
from ids_meta.common.nli import classify_nli_baseline, load_nli_profile, load_nli_runtime_config
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
STUDY_STAGES = ("collect", "prefilter", "classify", "baseline", "evaluate", "reconcile", "export", "plot", "run", "screen")


def _evaluation_root(settings: Settings, source: str) -> Path:
    """Return the single source root containing every rated-evaluation artifact."""
    return settings.path("evaluation_dir") / source


def _evaluation_variant(settings: Settings, source: str, variant: Path) -> Path:
    """Map pipeline variants into a consistent classifier-oriented evaluation tree."""
    root = _evaluation_root(settings, source) / "classifiers"
    parts = variant.parts
    if parts[:1] == ("models",) and len(parts) > 1:
        return root / "llm" / Path(*parts[1:])
    if parts[:1] == ("ensemble",):
        return root / "ensemble" / Path(*parts[1:])
    if parts[:2] == ("baselines", "keyword"):
        return root / "keyword" / Path(*parts[2:])
    if parts[:2] == ("baselines", "nli"):
        return root / "nli" / Path(*parts[2:])
    if parts[:2] == ("baselines", "llm_nli"):
        return root / "llm_nli" / Path(*parts[2:])
    return root / Path(*parts)


def _default_paths(settings: Settings, source: str) -> dict[str, Path]:
    """Build the default cache, CSV, plot, and comparison paths for a source."""
    interim = settings.path("interim_dir") / source
    plots = settings.path("plots_dir") / source
    evaluation_root = _evaluation_root(settings, source)
    comparison = settings.path("plots_dir") / source / "comparison"
    if source == "github":
        return {
            "cache": settings.path("github_cache"), "candidates": interim / "candidates.csv",
            "plots": plots,
            "comparison": comparison, "evaluation_comparison": evaluation_root / "comparison",
        }
    return {
        "cache": settings.path("openalex_cache"), "candidates": interim / "paper_candidates.csv",
        "surveys": interim / "survey_sources.csv",
        "plots": plots,
        "comparison": comparison, "evaluation_comparison": evaluation_root / "comparison",
    }


def _configured_models(settings: Settings) -> list[str]:
    """Return validated, de-duplicated names from the required model list."""
    config = settings.section("ollama")
    raw = config.get("models")
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
    evaluation = _evaluation_variant(settings, source, variant)
    paths.update({"classifications": interim / "classifications.csv", "reconciled": interim / "reconciled.csv", "plots": plots, "evaluation": evaluation})
    if source == "github":
        paths["catalogue"] = processed / "repositories.csv"
    else:
        paths.update({"catalogue": processed / "base_papers.csv", "adjacent": processed / "adjacent_base_papers.csv"})
    return paths


def _execution_paths(settings: Settings, source: str, models: list[str]) -> dict[str, dict[str, Path]]:
    """Use the same model-specific paths for every model count."""
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


def _pipeline_limits(args: argparse.Namespace, config: dict[str, object]) -> tuple[int | None, int | None]:
    """Resolve independent optional collection and classification limits."""
    collection = getattr(args, "collect_limit", None)
    classification = getattr(args, "classify_limit", None)
    if getattr(args, "collect_all", False):
        collection = None
    elif collection is None:
        collection = config.get("collection_limit")
    if getattr(args, "classify_all", False):
        classification = None
    elif classification is None:
        classification = config.get("classification_limit")
    return optional_limit(collection, "collection_limit"), optional_limit(classification, "classification_limit")


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


def _baselines_enabled(settings: Settings) -> bool:
    """Return whether full pipeline runs should execute configured baselines."""
    enabled = settings.section("baselines").get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("baselines.enabled must be true or false")
    return enabled


def _evaluation_candidates(args: argparse.Namespace, default: Path) -> Path:
    """Resolve the candidate audit matching an explicit evaluation or full run."""
    if getattr(args, "candidates", None):
        return Path(args.candidates)
    if args.action == "run" and getattr(args, "input", None):
        return Path(args.input)
    return default


def _evaluation_output(args: argparse.Namespace, default: Path) -> Path:
    """Keep an explicitly selected rating profile separate from the default report."""
    if getattr(args, "output", None) and args.action == "evaluate":
        return Path(args.output)
    ratings = getattr(args, "ratings", None)
    return default / Path(ratings).stem if ratings else default


def _configured_reference_sets(settings: Settings, source: str) -> list[tuple[str, Path]]:
    """Validate and resolve every external reference profile configured for one source."""
    configured = settings.section("evaluation").get("reference_sets", [])
    if not isinstance(configured, list) or any(not isinstance(item, dict) for item in configured):
        raise ConfigError("evaluation.reference_sets must be an array of tables")
    selected: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(configured, 1):
        configured_source = str(item.get("source", "")).strip()
        if configured_source not in {"github", "openalex"}:
            raise ConfigError(f"evaluation.reference_sets entry {index} has unsupported source {configured_source!r}")
        if configured_source != source:
            continue
        name = str(item.get("name", "")).strip()
        profile_value = item.get("profile")
        if not name or not isinstance(profile_value, str) or not profile_value.strip():
            raise ConfigError(
                f"evaluation.reference_sets entry {index} requires non-empty name, source, and profile"
            )
        label = slug(name)
        if label in seen_names:
            raise ConfigError(f"duplicate evaluation reference-set name for {source}: {name}")
        seen_names.add(label)
        profile = Path(profile_value)
        selected.append((label, profile if profile.is_absolute() else settings.root / profile))
    if not selected:
        raise ConfigError(f"no evaluation.reference_sets entry configured for source {source!r}")
    return selected


def _evaluation_jobs(
    settings: Settings, source: str, args: argparse.Namespace, default_output: Path,
) -> list[tuple[Path, Path, str]]:
    """Resolve configured reference-set profiles and their output directories."""
    explicit = getattr(args, "ratings", None)
    if args.action in {"evaluate", "baseline"} and explicit:
        return [(Path(explicit), _evaluation_output(args, default_output), "")]
    selected = _configured_reference_sets(settings, source)
    output = _evaluation_output(args, default_output)
    if len(selected) == 1:
        label, profile = selected[0]
        return [(profile, output, label)]
    return [(profile, output / label, label) for label, profile in selected]


def _evaluation_confidence_level(settings: Settings) -> float:
    """Return the configured interval level without embedding a fixed z-score."""
    raw_confidence_level = settings.section("evaluation").get("confidence_level", .95)
    if isinstance(raw_confidence_level, bool):
        raise ConfigError("evaluation.confidence_level must be numeric")
    try:
        confidence_level = float(raw_confidence_level)
    except (TypeError, ValueError) as exc:
        raise ConfigError("evaluation.confidence_level must be numeric") from exc
    if not 0 < confidence_level < 1:
        raise ConfigError("evaluation.confidence_level must be greater than zero and smaller than one")
    return confidence_level


def _evaluate(
    settings: Settings, source: str, classifications: Path, candidates: Path, output_dir: Path,
    ratings_path: Path | None = None,
) -> EvaluationResult:
    """Evaluate one source/model result against its centrally configured ratings."""
    if ratings_path is None:
        jobs = _evaluation_jobs(
            settings, source, argparse.Namespace(action="run", ratings=None, output=None), output_dir,
        )
        if len(jobs) != 1:
            raise ConfigError("an explicit reference profile is required when multiple sets are configured")
        profile_path = jobs[0][0]
    else:
        profile_path = ratings_path
    profile = load_rating_profile(profile_path, source)
    result = evaluate_results(
        profile, classifications, output_dir, candidates,
        confidence_level=_evaluation_confidence_level(settings),
    )
    from ids_meta.plotting.evaluation import build as build_evaluation_plots

    plot_outputs = build_evaluation_plots(output_dir, settings.section("plots"))
    from ids_meta.common.reporting import ReportingDatabase

    with ReportingDatabase(output_dir / 'reporting.sqlite3') as database:
        database.add_figures(output_dir / 'figures')
    plot_tables = sorted((output_dir / "figures" / "data").glob("*.csv"))
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


def _classifier_entries(settings: Settings, source: str) -> dict[str, tuple[Path, Path]]:
    """Discover classifier CSVs and matching evaluation roots already present on disk."""
    interim = settings.path("interim_dir") / source
    evaluation = _evaluation_root(settings, source) / "classifiers"
    entries: dict[str, tuple[Path, Path]] = {}
    for path in sorted((interim / "models").glob("*/classifications.csv")):
        name = path.parent.name
        entries[f"LLM · {name}"] = (path, evaluation / "llm" / name)
    ensemble = interim / "ensemble" / "classifications.csv"
    if ensemble.is_file():
        entries["Ensemble"] = (ensemble, evaluation / "ensemble")
    keyword = interim / "baselines" / "keyword" / "classifications.csv"
    if keyword.is_file():
        entries["Keyword"] = (keyword, evaluation / "keyword")
    nli_root = interim / "baselines" / "nli"
    for path in sorted(nli_root.glob("*/classifications.csv")):
        name = path.parent.name
        entries[f"NLI · {name}"] = (path, evaluation / "nli" / name)
    for path in sorted((interim / "baselines/llm_nli").glob("*/classifications.csv")):
        name = path.parent.name
        entries[f"NLI - {name}"] = (path, evaluation / "llm_nli" / name)
    order = {"Keyword": 0, "NLI": 1, "LLM": 2, "Ensemble": 3}
    return dict(sorted(entries.items(), key=lambda item: (
        next((rank for prefix, rank in order.items() if item[0].startswith(prefix)), 9), item[0],
    )))


def _build_classifier_comparisons(settings: Settings, source: str) -> list[Path]:
    """Build classifier comparisons separately for every configured reference set."""
    from ids_meta.plotting.compare import build_classifiers

    entries = _classifier_entries(settings, source)
    if len(entries) < 2:
        LOG.warning("skipping all-classifier comparison source=%s available=%d", source, len(entries))
        return []
    reference_sets = _configured_reference_sets(settings, source)
    multiple = len(reference_sets) > 1
    comparison_root = _default_paths(settings, source)["evaluation_comparison"]
    outputs: list[Path] = []
    compared: list[tuple[str, Path, list[str]]] = []
    for reference_label, _profile in reference_sets:
        classifications: dict[str, Path] = {}
        evaluations: dict[str, Path] = {}
        classifier_labels: list[str] = []
        for classifier_label, (classification, evaluation_root) in entries.items():
            reference_root = evaluation_root / reference_label if multiple else evaluation_root
            if classification.is_file() and (reference_root / "metrics.json").is_file():
                classifications[classifier_label] = classification
                evaluations[classifier_label] = reference_root
                classifier_labels.append(classifier_label)
        if len(evaluations) < 2:
            LOG.warning(
                "skipping reference-set comparison source=%s reference=%s available=%d",
                source, reference_label, len(evaluations),
            )
            continue
        target = comparison_root / reference_label if multiple else comparison_root
        outputs.extend(build_classifiers(
            source, classifications, evaluations, target, settings.section("plots"),
        ))
        compared.append((reference_label, target, classifier_labels))
    if not compared:
        return []
    evaluation_root = _evaluation_root(settings, source)
    display_source = "OpenAlex" if source == "openalex" else "GitHub"
    lines = [
        f"# {display_source} external-reference evaluation", "", "## Start here", "",
    ]
    for reference_label, target, classifier_labels in compared:
        relative = target.relative_to(evaluation_root)
        lines.extend([
            f"- `{reference_label}`: open `{relative}/00_metric_overview.svg` ",
            f"  ({len(classifier_labels)} classifiers; exact values in `{relative}/data/00_metric_overview.csv`).",
            f"  Horizontal recovery overview: `{relative}/01_reference_recovery.svg`.",
            f"  Compact taxonomy comparison: `{relative}/taxonomy/04_taxonomy_comparison.svg`.",
            f"  Taxonomy accuracy among recovered items: `{relative}/taxonomy/05_recovered_taxonomy_comparison.svg`.",
        ])
    lines.extend([
        "", "## Directory guide", "",
        "- `comparison/`: cross-classifier reporting, split by reference-set name when several are configured.",
        "- `classifiers/`: detailed audits for one classifier, likewise split by reference set when needed.",
        "- `classifiers/.../figures/06_taxonomy_accuracy`: horizontal exact field-accuracy overview with percentage labels.",
        "- `classifiers/.../figures/07_recovered_taxonomy_accuracy`: field accuracy among recovered reference positives, with field-rated sample counts and recovery totals. No recovered comparisons means N/A; compare alongside recovery because approaches can recover different subsets.",
        "- `classifiers/nli/<model>/sensitivity/`: offline NLI cutoff robustness; not another model result.",
        "- Matching `.svg` and `.pdf` files are the same figure in screen and publication formats.",
        "", "## Available classifier details", "",
    ])
    for classifier_label, (_classification, directory) in entries.items():
        available = [
            label for label, _profile in reference_sets
            if ((directory / label) if multiple else directory).joinpath("metrics.json").is_file()
        ]
        if available:
            lines.append(
                f"- `{classifier_label}` → `{directory.relative_to(evaluation_root)}/` "
                f"(reference sets: {', '.join(available)})"
            )
    lines.append("")
    evaluation_root.mkdir(parents=True, exist_ok=True)
    (evaluation_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return outputs


def _execute_llm_nli_variants(args: argparse.Namespace, settings: Settings, source: str, cache: PipelineCache | None, manifest: RunManifest) -> None:
    """Run or re-evaluate each independently cached Ollama NLI model."""
    from ids_meta.common.llm_nli import OllamaNLIBackend, augmented_profile, load_llm_nli_config
    from ids_meta.common.nli import nli_profile_hash, rescore_nli_rows
    from ids_meta.common.csvio import atomic_write_csv
    from ids_meta.common.nli_diagnostics import build_nli_threshold_diagnostics, load_nli_threshold_grid

    config = load_llm_nli_config(settings)
    reference = load_nli_runtime_config(settings)
    base_profile = load_nli_profile(settings, source)
    fields = tuple(value.strip() for value in (getattr(args, "fields", None) or "").split(",") if value.strip())
    limit = getattr(args, "limit", None)
    adapter = {"system_type_profile": load_system_type_profile(settings, "github")} if source == "github" else {}
    if getattr(args, "output", None) and len(config.models) != 1:
        raise ConfigError("LLM NLI --output requires exactly one configured model")
    for model in config.models:
        label = "llm_nli_" + slug(model)
        variant = Path("baselines/llm_nli") / slug(model)
        if fields or limit is not None:
            # Partial runs must not replace the full result or enter comparisons.
            variant /= Path("pilots") / f"{slug('-'.join(fields) or 'all')}-{limit or 'all'}"
        paths = _variant_paths(settings, source, variant)
        output = Path(args.output) if args.output else paths["classifications"]
        candidates = Path(args.input) if args.input else _default_paths(settings, source)["candidates"]
        if getattr(args, "evaluate_only", False):
            if not output.is_file():
                raise ConfigError(f"--evaluate-only requires saved LLM NLI classifications: {output}")
            rows = read_csv(output)
            metadata_values = {row.get("llm_nli_metadata_json", "") for row in rows}
            if len(metadata_values) != 1 or not rows:
                raise ConfigError("Saved LLM NLI rows require one consistent model/scoring provenance")
            metadata = json.loads(next(iter(metadata_values)))
            if metadata.get("model") != model:
                raise ConfigError("Saved LLM NLI model does not match the configured model")
            runtime = replace(reference, model=model, revision=metadata["model_digest"], entailment_threshold=config.entailment_threshold, label_margin=config.label_margin)
            profile = augmented_profile(base_profile, config, reference, metadata, fields)
            for row in rows:
                stored = replace(runtime, entailment_threshold=float(row["nli_entailment_threshold"]), label_margin=float(row["nli_label_margin"]))
                if row.get("classifier_profile_hash") != nli_profile_hash(profile, stored):
                    raise ConfigError("Saved LLM NLI scores do not match the active hypotheses/chunking configuration")
            if any(float(row["nli_entailment_threshold"]) != runtime.entailment_threshold or float(row["nli_label_margin"]) != runtime.label_margin for row in rows):
                rows = rescore_nli_rows(source, rows, profile, runtime.entailment_threshold, runtime.label_margin, adapter)
                for row in rows:
                    row["classifier_profile_hash"] = row["profile_hash"] = nli_profile_hash(profile, runtime)
                    row["nli_entailment_threshold"] = str(runtime.entailment_threshold)
                    row["nli_label_margin"] = str(runtime.label_margin)
                atomic_write_csv(output, rows, list(rows[0]))
        else:
            backend = OllamaNLIBackend(model, config, reference, settings.section("ollama"), cache if args.resume else None)
            profile = augmented_profile(base_profile, config, reference, backend.metadata, fields)
            preflight = backend.preflight()
            LOG.info("LLM NLI capability check passed source=%s model=%s digest=%s", source, model, backend.revision)
            manifest.parameters[label] = backend.metadata
            if getattr(args, "check_only", False):
                manifest.parameters[label + "_preflight"] = preflight
                continue
            assert cache is not None
            runtime = backend.runtime_config()
            rows = classify_nli_baseline(
                source, candidates, output, cache, profile, runtime,
                method="llm_nli", limit=limit, backend=backend, resume=args.resume,
                progress_every=int(settings.section("logging").get("progress_every", 25)),
                adapter_config=adapter,
            )
        manifest.counts[label + "_rows"] = len(rows)
        for ratings_path, evaluation_dir, profile_label in _evaluation_jobs(settings, source, args, paths["evaluation"]):
            result = _evaluate(settings, source, output, candidates, evaluation_dir, ratings_path)
            _record_evaluation(manifest, result, "_".join(filter(None, (label, profile_label))))
            if any(row.get("nli_scores_json") not in {None, "", "{}"} for row in rows):
                diagnostic_outputs = build_nli_threshold_diagnostics(
                    source, output, candidates, load_rating_profile(ratings_path, source),
                    profile, runtime, load_nli_threshold_grid(settings), evaluation_dir / "sensitivity",
                    settings.section("plots"), confidence_level=_evaluation_confidence_level(settings), adapter_config=adapter,
                )
                manifest.counts[label + "_sensitivity_files"] = len(diagnostic_outputs)


def _execute_baselines(args: argparse.Namespace, settings: Settings, source: str) -> dict[str, int]:
    """Run configured source baselines and their rated evaluations."""
    paths = _default_paths(settings, source)
    methods = configured_baseline_methods(settings, getattr(args, "methods", None))
    evaluate_only = bool(getattr(args, "evaluate_only", False))
    check_only = bool(getattr(args, "check_only", False))
    if (check_only or getattr(args, "fields", None) or getattr(args, "limit", None) is not None) and methods != ["llm_nli"]:
        raise ConfigError("--check-only, --fields and baseline --limit require --methods llm_nli")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        raise ConfigError("Baseline --limit must be positive")
    if check_only and evaluate_only:
        raise ConfigError("--check-only and --evaluate-only cannot be combined")
    if getattr(args, "output", None) and len(methods) != 1:
        raise ConfigError("--output requires exactly one selected baseline method")
    candidates_path = Path(args.input) if args.input else paths["candidates"]
    manifest = _manifest(settings, source, "baseline")
    manifest.parameters.update({
        "baseline_methods": methods, "candidates_path": str(candidates_path),
        "evaluate_only": evaluate_only,
    })
    LOG.info(
        "starting baseline command source=%s methods=%s evaluate_only=%s run_id=%s",
        source, ",".join(methods), evaluate_only, manifest.run_id,
    )
    cache_context = nullcontext(None) if evaluate_only or check_only else PipelineCache(paths["cache"])
    with cache_context as cache:
        for method in methods:
            if method == "llm_nli":
                _execute_llm_nli_variants(args, settings, source, cache, manifest)
                continue
            nli_config = load_nli_runtime_config(settings) if method == "nli" else None
            nli_profile = load_nli_profile(settings, source) if method == "nli" else None
            adapter_config = (
                {"system_type_profile": load_system_type_profile(settings, "github")}
                if source == "github" else {}
            )
            variant = Path("baselines") / method
            if nli_config is not None:
                variant /= slug(nli_config.model)
            variant_paths = _variant_paths(settings, source, variant)
            output = Path(args.output) if args.output else variant_paths["classifications"]
            if evaluate_only:
                if not output.is_file():
                    raise ConfigError(
                        f"--evaluate-only requires an existing {method} classification file: {output}"
                    )
                rows = read_csv(output)
            elif method == "keyword":
                assert cache is not None
                term_profile = load_term_profile(settings, source)
                keyword_config = load_keyword_baseline_config(settings, source)
                keyword_profile = load_keyword_taxonomy_profile(settings, source)
                keyword_adapter_config = (
                    {"system_type_profile": load_system_type_profile(settings, "github")}
                    if source == "github" else {}
                )
                manifest.parameters["keyword_accept_threshold"] = keyword_config.accept_threshold
                manifest.parameters["keyword_uncertain_threshold"] = keyword_config.uncertain_threshold
                manifest.parameters["keyword_taxonomy_profile"] = str(keyword_profile.path)
                rows = classify_keyword_baseline(
                    source, candidates_path, output, cache, term_profile, keyword_config,
                    keyword_profile,
                    resume=args.resume,
                    progress_every=int(settings.section("logging").get("progress_every", 25)),
                    adapter_config=keyword_adapter_config,
                )
            elif method == "nli":
                assert cache is not None
                assert nli_config is not None
                assert nli_profile is not None
                manifest.parameters.update({
                    "nli_model": nli_config.model, "nli_revision": nli_config.revision,
                    "nli_entailment_threshold": nli_config.entailment_threshold,
                    "nli_label_margin": nli_config.label_margin,
                })
                rows = classify_nli_baseline(
                    source, candidates_path, output, cache, nli_profile, nli_config,
                    resume=args.resume,
                    progress_every=int(settings.section("logging").get("progress_every", 25)),
                    adapter_config=adapter_config,
                )
            else:  # pragma: no cover - validated before the loop
                raise ConfigError(f"baseline method is not implemented: {method}")
            manifest.counts[f"baseline_rows_{method}"] = len(rows)
            manifest.counts[f"baseline_{method}_candidate_evidence"] = sum(
                row.get("classifier_evidence_source") == "candidate_csv" for row in rows
            )
            for decision in ("accept", "uncertain", "exclude", "error"):
                manifest.counts[f"baseline_{method}_{decision}"] = sum(
                    row.get("decision") == decision for row in rows
                )
            for ratings_path, evaluation_dir, profile_label in _evaluation_jobs(
                settings, source, args, variant_paths["evaluation"],
            ):
                result = _evaluate(
                    settings, source, output, candidates_path, evaluation_dir, ratings_path,
                )
                _record_evaluation(
                    manifest, result, "_".join(filter(None, (method, profile_label))),
                )
                if method == "nli":
                    from ids_meta.common.nli_diagnostics import (
                        build_nli_threshold_diagnostics, load_nli_threshold_grid,
                    )

                    assert nli_config is not None and nli_profile is not None
                    rating_profile = load_rating_profile(ratings_path, source)
                    diagnostic_outputs = build_nli_threshold_diagnostics(
                        source, output, candidates_path, rating_profile, nli_profile,
                        nli_config, load_nli_threshold_grid(settings),
                        evaluation_dir / "sensitivity", settings.section("plots"),
                        confidence_level=_evaluation_confidence_level(settings),
                        adapter_config=adapter_config,
                    )
                    diagnostic_label = "_".join(filter(None, (method, profile_label)))
                    manifest.counts[f"nli_threshold_diagnostic_files_{diagnostic_label}"] = len(
                        diagnostic_outputs
                    )
        if cache is not None:
            manifest.coverage = cache.coverage_rows(source)
    comparison_outputs = [] if check_only else _build_classifier_comparisons(settings, source)
    manifest.counts["baseline_comparison_files"] = len(comparison_outputs)
    manifest.status = "complete"
    manifest.write()
    LOG.info("completed baseline command source=%s counts=%s", source, manifest.counts)
    return manifest.counts


def _execute_github_models(args: argparse.Namespace, settings: Settings, models: list[str]) -> dict[str, int]:
    """Execute GitHub model variants plus an optional deterministic ensemble."""
    base_paths = _default_paths(settings, "github")
    model_paths = _execution_paths(settings, "github", models)
    ensemble_enabled, ensemble_members, threshold = _ensemble_config(settings, models)
    ensemble_paths = _variant_paths(settings, "github", Path("ensemble"))
    if len(models) > 1 or ensemble_enabled:
        _reject_ambiguous_custom_paths(args)
    else:
        paths = model_paths[models[0]]
        if args.input and args.action in {"evaluate", "reconcile", "export", "plot"}:
            key = {"evaluate": "classifications", "reconcile": "classifications", "export": "reconciled", "plot": "catalogue"}[args.action]
            paths[key] = Path(args.input)
        if args.output and args.action in {"classify", "evaluate", "reconcile", "export", "plot"}:
            key = {"classify": "classifications", "evaluate": "evaluation", "reconcile": "reconciled", "export": "catalogue", "plot": "plots"}[args.action]
            paths[key] = Path(args.output)
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
            counts = {"cached": sum(1 for _ in cache.records("github.repository"))} if args.dry_run else collect_github({**config, "classification_limit": classify_limit}, profile, cache, collect_limit, resume=args.resume)
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
                if not model_paths[model]["classifications"].is_file():
                    LOG.warning(
                        "skipping evaluation for model without classifications source=github model=%s path=%s",
                        model, model_paths[model]["classifications"],
                    )
                    continue
                for ratings_path, output_dir, profile_label in _evaluation_jobs(settings, "github", args, model_paths[model]["evaluation"]):
                    result = _evaluate(settings, "github", model_paths[model]["classifications"], candidates_path, output_dir, ratings_path)
                    _record_evaluation(manifest, result, "_".join(filter(None, (labels[model], profile_label))))
            if ensemble_enabled and ensemble_paths["classifications"].is_file():
                for ratings_path, output_dir, profile_label in _evaluation_jobs(settings, "github", args, ensemble_paths["evaluation"]):
                    result = _evaluate(settings, "github", ensemble_paths["classifications"], candidates_path, output_dir, ratings_path)
                    _record_evaluation(manifest, result, "_".join(filter(None, ("ensemble", profile_label))))
            if not getattr(args, "output", None):
                manifest.counts["evaluation_comparison_files"] = len(
                    _build_classifier_comparisons(settings, "github")
                )
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
            if len(comparison_inputs) > 1:
                generated_count += len(build_model_comparison("github", comparison_inputs, comparison_dir, settings.section("plots")))
            manifest.counts["plot_files"] = generated_count
        manifest.coverage = cache.coverage_rows("github")
    manifest.status = "complete"
    manifest.write()
    LOG.info("completed multi-model command source=github stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def _execute_github(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Execute the source using the configured model list."""
    settings.require_source("github")
    if args.action == "baseline":
        return _execute_baselines(args, settings, "github")
    return _execute_github_models(args, settings, _configured_models(settings))


def _execute_openalex_models(args: argparse.Namespace, settings: Settings, models: list[str]) -> dict[str, int]:
    """Execute OpenAlex model variants plus an optional deterministic ensemble."""
    base_paths = _default_paths(settings, "openalex")
    model_paths = _execution_paths(settings, "openalex", models)
    ensemble_enabled, ensemble_members, threshold = _ensemble_config(settings, models)
    ensemble_paths = _variant_paths(settings, "openalex", Path("ensemble"))
    if len(models) > 1 or ensemble_enabled:
        _reject_ambiguous_custom_paths(args)
    else:
        paths = model_paths[models[0]]
        if args.input and args.action in {"evaluate", "reconcile", "export", "plot"}:
            key = {"evaluate": "classifications", "reconcile": "classifications", "export": "reconciled", "plot": "catalogue"}[args.action]
            paths[key] = Path(args.input)
        if args.output and args.action in {"classify", "evaluate", "reconcile", "export", "plot"}:
            key = {"classify": "classifications", "evaluate": "evaluation", "reconcile": "reconciled", "export": "catalogue", "plot": "plots"}[args.action]
            paths[key] = Path(args.output)
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
            counts = {"cached": sum(1 for _ in cache.records("openalex.work"))} if args.dry_run else collect_openalex({**config, "classification_limit": classify_limit}, profile, cache, collect_limit)
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
                if not model_paths[model]["classifications"].is_file():
                    LOG.warning(
                        "skipping evaluation for model without classifications source=openalex model=%s path=%s",
                        model, model_paths[model]["classifications"],
                    )
                    continue
                for ratings_path, output_dir, profile_label in _evaluation_jobs(settings, "openalex", args, model_paths[model]["evaluation"]):
                    result = _evaluate(settings, "openalex", model_paths[model]["classifications"], candidates_path, output_dir, ratings_path)
                    _record_evaluation(manifest, result, "_".join(filter(None, (labels[model], profile_label))))
            if ensemble_enabled and ensemble_paths["classifications"].is_file():
                for ratings_path, output_dir, profile_label in _evaluation_jobs(settings, "openalex", args, ensemble_paths["evaluation"]):
                    result = _evaluate(settings, "openalex", ensemble_paths["classifications"], candidates_path, output_dir, ratings_path)
                    _record_evaluation(manifest, result, "_".join(filter(None, ("ensemble", profile_label))))
            if not getattr(args, "output", None):
                manifest.counts["evaluation_comparison_files"] = len(
                    _build_classifier_comparisons(settings, "openalex")
                )
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
            if len(comparison_inputs) > 1:
                generated_count += len(build_model_comparison("openalex", comparison_inputs, comparison_dir, settings.section("plots")))
            manifest.counts["plot_files"] = generated_count
        manifest.coverage = cache.coverage_rows("openalex")
    manifest.status = "complete"
    manifest.write()
    LOG.info("completed multi-model command source=openalex stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def _execute_openalex(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Execute the source using the configured model list."""
    settings.require_source("openalex")
    if args.action == "baseline":
        return _execute_baselines(args, settings, "openalex")
    return _execute_openalex_models(args, settings, _configured_models(settings))


def _configured_review_protocol(settings: Settings):
    """Validate that review execution and reporting select the same profiles."""
    from ids_meta.common.study_screening import load_study_protocol
    from ids_meta.common.study_pipeline import load_assessment_template, load_assessment_fields

    protocol = load_study_protocol(settings.path("protocol"))
    load_assessment_fields(protocol)
    _, prompt_path, _ = load_assessment_template(protocol)
    with protocol.path.open("rb") as handle:
        protocol_config = tomllib.load(handle)
    if (protocol.path.parent / protocol_config["source_profile"]).resolve() != settings.path("taxonomy"):
        raise ConfigError("review protocol must reference the study taxonomy")
    source = settings.section("study")["sources"][0]
    if prompt_path != settings.path(f"{source}_prompts"):
        raise ConfigError("review protocol must reference the configured classification prompt")
    ratings = settings.section("evaluation").get("references")
    if ratings:
        load_rating_profile(Path(ratings), protocol.contract.name, contract=protocol.contract)
    return protocol


def _execute_review(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Run a configured evidence-backed review without named-study branches."""
    from ids_meta.common.study_pipeline import collect_study, classify_study
    from ids_meta.common.study_screening import screen_study
    from ids_meta.plotting.generic import build_report

    action = args.action
    if action not in {"collect", "classify", "screen", "plot", "evaluate", "run"}:
        raise ConfigError(f"systematic_review does not support stage {action!r}")
    if getattr(args, "collect_all", False):
        raise ConfigError("systematic_review requires an explicit bounded collection limit")
    models = _configured_models(settings)
    if args.output and action in {"classify", "plot", "evaluate", "run"} and len(models) > 1:
        raise ConfigError("a single custom output cannot represent multiple review models")
    source = settings.section("study")["sources"][0]
    candidates = settings.path("interim_dir") / source / "candidates.json"
    protocol = _configured_review_protocol(settings).path
    paths = _execution_paths(settings, source, models)
    manifest = _manifest(settings, source, action)
    manifest.write()
    counts: dict[str, int] = {}
    try:
        if action in {"collect", "run"} and not (action == "run" and args.input):
            if args.dry_run:
                raise ConfigError("review collection needs network access; use classify --dry-run --input for an offline preview")
            target = args.output if action == "collect" and args.output else candidates
            limit = args.collect_limit or settings.section("collection").get("limit", 200)
            rows = collect_study(settings.path(f"{source}_terms"), target, limit=limit,
                source_config=settings.section(source))
            counts["candidate_papers"] = len(rows)
        if action == "screen":
            if not args.input:
                raise ConfigError("screen requires --input containing criterion assessments")
            rows = screen_study(protocol, args.input, args.output or settings.path("interim_dir") / "screened.csv")
            counts["screened_records"] = len(rows)
        for model in models:
            classification = args.output if action == "classify" and args.output else paths[model]["classifications"]
            if action in {"classify", "run"}:
                assessment = settings.section("assessment")
                limit = None if args.classify_all else args.classify_limit or assessment.get("limit")
                rows = classify_study(protocol, args.input or candidates, classification,
                    _model_ollama_config(settings, model), limit=limit, resume=args.resume, dry_run=args.dry_run,
                    fetch_full_text=bool(assessment.get("fetch_full_text", False)),
                    max_chars=assessment.get("max_full_text_chars", 120000))
                counts[f"classified_{slug(model)}"] = len(rows)
                counts[f"errors_{slug(model)}"] = sum(r["decision"] == "error" for r in rows)
            if action in {"plot", "evaluate", "run"}:
                ratings = getattr(args, "ratings", None) or settings.section("evaluation").get("references")
                enabled = settings.section("evaluation").get("enabled", False)
                files = build_report(settings.path("taxonomy"),
                    args.input if action in {"plot", "evaluate"} and args.input else classification,
                    args.output if action in {"plot", "evaluate"} and args.output else paths[model]["plots"],
                    settings.section("plots"),
                    candidates_path=(getattr(args, "candidates", None) or
                        (args.input if action == "run" and args.input else candidates if candidates.is_file() else None)),
                    ratings_path=Path(ratings) if ratings and (enabled or action == "evaluate") else None,
                    confidence_level=_evaluation_confidence_level(settings))
                counts[f"report_files_{slug(model)}"] = len(files)
    except Exception:
        manifest.status = "failed"
        manifest.write()
        raise
    manifest.counts = counts
    manifest.status = "completed"
    manifest.write()
    return counts


def _add_stage_arguments(parser: argparse.ArgumentParser, action: str) -> None:
    """Declare stage options once for study and explicit-source commands."""
    parser.set_defaults(action=action)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
        help="explicitly reuse compatible current-format stage results")
    if action != "baseline":
        parser.add_argument("--dry-run", action="store_true")
    if action in {"collect", "run"}:
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--collect-limit", type=int, help="maximum records retained from discovery")
        group.add_argument("--collect-all", action="store_true")
    if action in {"collect", "classify", "run"}:
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--classify-limit", type=int, help="maximum candidates classified per model")
        group.add_argument("--classify-all", action="store_true")
    if action in {"baseline", "evaluate"}:
        parser.add_argument("--ratings", type=Path, help="explicit reference profile for evaluation")
    if action == "evaluate":
        parser.add_argument("--candidates", type=Path)
    if action == "baseline":
        parser.add_argument("--methods", help="comma-separated keyword,nli,llm_nli methods")
        parser.add_argument("--evaluate-only", action="store_true", help="rescore saved outputs without inference")
        parser.add_argument("--check-only", action="store_true", help="check LLM NLI class-token scoring")
        parser.add_argument("--fields", help="comma-separated LLM NLI pilot fields")
        parser.add_argument("--limit", type=int, help="maximum candidates for an isolated LLM NLI pilot")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser and all source-stage subcommands."""
    parser = argparse.ArgumentParser(prog="ids-meta", description="IDS meta-study discovery, classification, export, and plotting")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config", type=Path, help="study.toml manifest; references resolve relative to that file")
    selection.add_argument("--study", help="study directory name under config/studies")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="stdout logging verbosity; defaults to config logging.level")
    parser.add_argument("--subcaptions", action=argparse.BooleanOptionalAction, default=None, help="show or hide explanatory plot subcaptions; overrides plots.show_subcaptions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="validate the selected study configuration without API or model calls")
    for action in STUDY_STAGES:
        stage = subparsers.add_parser(action, help=f"{action} the selected study")
        stage.add_argument("--source", help="select a configured source branch")
        _add_stage_arguments(stage, action)
    for source in ("github", "openalex"):
        source_parser = subparsers.add_parser(source)
        actions = source_parser.add_subparsers(dest="action", required=True)
        for action in STUDY_STAGES:
            if action != "screen":
                _add_stage_arguments(actions.add_parser(action), action)
    architecture = subparsers.add_parser("architecture")
    architecture.add_argument("--output", type=Path, help="output directory; defaults to the repository root")
    architecture.add_argument("--formats", default="svg,pdf", help="comma-separated vector formats: svg,pdf")
    funnel = subparsers.add_parser("funnel")
    funnel.add_argument("source", choices=("github", "openalex"))
    funnel.add_argument("--output", type=Path, help="output directory; defaults to artifacts/plots/<source>/pipeline_funnel")
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("source", choices=("github", "openalex"))
    comparison.add_argument("--input", type=Path)
    comparison.add_argument("--output", type=Path)
    report = subparsers.add_parser("report", help="report on any CSV/JSON provider using a source profile")
    report.add_argument("--profile", required=True, type=Path)
    report.add_argument("--input", required=True, type=Path)
    report.add_argument("--candidates", type=Path)
    report.add_argument("--exports", type=Path)
    report.add_argument("--ratings", type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--database", type=Path)
    screening = subparsers.add_parser("study-screen", help="replay an external study's evidence-backed eligibility and quality rules")
    screening.add_argument("--protocol", required=True, type=Path)
    screening.add_argument("--input", required=True, type=Path, help="JSON records with criterion assessments and full text")
    screening.add_argument("--output", required=True, type=Path, help="classification CSV for report/evaluation")
    study_collect = subparsers.add_parser("study-collect", help="discover papers for an external review with OpenAlex")
    study_collect.add_argument("--search-profile", required=True, type=Path)
    study_collect.add_argument("--output", required=True, type=Path, help="candidate JSON and collection audit")
    study_collect.add_argument("--limit", type=int, default=200)
    study_collect.add_argument("--per-query-limit", type=int)
    study_classify = subparsers.add_parser("study-classify", help="assess external-review candidates with Ollama")
    study_classify.add_argument("--protocol", required=True, type=Path)
    study_classify.add_argument("--input", required=True, type=Path, help="candidate JSON, optionally containing full text")
    study_classify.add_argument("--output", required=True, type=Path, help="classification CSV and assessment checkpoint")
    study_classify.add_argument("--model", help="select one model for this explicit classification call")
    study_classify.add_argument("--limit", type=int)
    study_classify.add_argument("--resume", action="store_true")
    study_classify.add_argument("--dry-run", action="store_true", help="write uncertain previews without HTTP or model calls")
    study_classify.add_argument("--fetch-full-text", action="store_true", help="download open PDFs declared by candidate metadata")
    study_classify.add_argument("--max-full-text-chars", type=int, default=120000)
    verify = subparsers.add_parser("verify", help="independently check figure bundles or reporting databases")
    target = verify.add_mutually_exclusive_group(required=True)
    target.add_argument("--input", type=Path)
    target.add_argument("--database", type=Path)
    audit = subparsers.add_parser("audit-import", help="backfill existing evaluation audits without inference")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--database", type=Path, required=True)
    audit.add_argument("--profile", type=Path)
    report_compare = subparsers.add_parser("report-compare", help="compare compatible saved evaluations from any provider")
    report_compare.add_argument("--evaluation", action="append", required=True, help="LABEL=EVALUATION_DIRECTORY")
    report_compare.add_argument("--output", required=True, type=Path)
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
        if args.study and (Path(args.study).name != args.study or args.study in {".", ".."}):
            raise ConfigError("--study must be a study directory name; use --config for an explicit path")
        config_path = args.config or (Path("config/studies") / args.study / "study.toml" if args.study else DEFAULT_STUDY)
        standalone = {"report", "verify", "audit-import", "report-compare", "study-screen"}
        settings = (Settings(Path.cwd(), {"plots": {"formats": ["svg", "pdf"]}})
            if args.command in standalone and not args.config and not args.study else load_settings(config_path))
        if args.subcaptions is not None:
            settings.values.setdefault("plots", {})["show_subcaptions"] = args.subcaptions
        configure_logging(args.log_level or str(settings.section("logging").get("level", "INFO")))
        if args.command != 'verify':
            LOG.info("loaded configuration %s", config_path.resolve() if settings.section("study") else "standalone reporting")
        if args.command == "validate":
            if settings.section("study")["workflow"] == "systematic_review":
                _configured_review_protocol(settings)
            else:
                for source in settings.section("study")["sources"]:
                    load_term_profile(settings, source)
                    load_prompt_profile(settings, source)
                    load_keyword_taxonomy_profile(settings, source)
                    load_nli_profile(settings, source)
                    for _, profile_path in _configured_reference_sets(settings, source):
                        load_rating_profile(profile_path, source)
            _configured_models(settings)
            print(json.dumps(settings.section("study"), indent=2))
            return 0
        if args.command in STUDY_STAGES:
            args.action = args.command
            sources = settings.section("study")["sources"]
            source = args.source or (sources[0] if len(sources) == 1 else None)
            if source is None:
                raise ConfigError("this study has multiple source branches; select one with --source")
            settings.require_source(source)
            if settings.section("study")["workflow"] == "systematic_review":
                counts = _execute_review(args, settings)
                LOG.info("command summary %s", counts)
                return 0
            args.command = source
        if args.command in {"github", "openalex", "funnel", "compare"}:
            settings.require_source(args.command if args.command in {"github", "openalex"} else args.source)
            if settings.section("study").get("workflow") == "systematic_review":
                raise ConfigError("use study stage commands for systematic_review workflows")
        if args.command == "verify":
            from ids_meta.common.verification import verify_directory, verify_database

            result = verify_database(args.database) if args.database else verify_directory(args.input)
            print(json.dumps(result, sort_keys=True, indent=2))
            return 0 if result["ok"] else 2
        elif args.command == "study-screen":
            from ids_meta.common.study_screening import screen_study

            rows = screen_study(args.protocol, args.input, args.output)
            counts = {"screened_records": len(rows)}
        elif args.command == "study-collect":
            from ids_meta.common.study_pipeline import collect_study

            rows = collect_study(args.search_profile, args.output, limit=args.limit, per_query_limit=args.per_query_limit)
            counts = {"candidate_papers": len(rows)}
        elif args.command == "study-classify":
            from ids_meta.common.study_pipeline import classify_study

            config = dict(settings.section("ollama"))
            config["model"] = args.model or _configured_models(settings)[0]
            rows = classify_study(args.protocol, args.input, args.output, config,
                limit=args.limit, resume=args.resume, dry_run=args.dry_run,
                fetch_full_text=args.fetch_full_text, max_chars=args.max_full_text_chars)
            counts = {"classified_papers": len(rows), "assessment_errors": sum(r["decision"] == "error" for r in rows)}
        elif args.command == "report":
            from ids_meta.plotting.generic import build_report

            files = build_report(args.profile, args.input, args.output, settings.section("plots"),
                candidates_path=args.candidates, ratings_path=args.ratings, exports_path=args.exports, database_path=args.database)
            counts = {"report_files": len(files)}
        elif args.command == "audit-import":
            from ids_meta.common.reporting import ReportingDatabase
            from ids_meta.common.sources import SourceContract, load_source_contract

            directories = sorted({p.parent for p in args.input.rglob("metrics.json")})
            if not directories:
                raise ValueError("no saved evaluations found to import")
            with ReportingDatabase(args.database) as database:
                for directory in directories:
                    metrics = json.loads((directory / "metrics.json").read_text())
                    contract = load_source_contract(args.profile) if args.profile else SourceContract(**metrics["source_contract"])
                    database.ingest_evaluation(directory, contract)
                database.add_figures(args.input)
            counts = {"imported_evaluations": len(directories)}
        elif args.command == "report-compare":
            from ids_meta.plotting.compare import build_classifiers

            evaluations, classifications = {}, {}
            for value in args.evaluation:
                label, separator, directory = value.partition("=")
                if not separator or not label or label in evaluations:
                    raise ValueError("--evaluation requires distinct LABEL=DIRECTORY entries")
                evaluations[label] = Path(directory)
                metrics = json.loads((Path(directory) / "metrics.json").read_text())
                classifications[label] = Path(metrics["classifications_path"])
            files = build_classifiers(metrics["source"], classifications, evaluations, args.output, settings.section("plots"))
            counts = {"comparison_files": len(files)}
        elif args.command == "github":
            counts = _execute_github(args, settings)
        elif args.command == "openalex":
            counts = _execute_openalex(args, settings)
        elif args.command == "architecture":
            from ids_meta.architecture import build as build_architecture

            files = build_architecture(args.output or settings.root, [value.strip() for value in args.formats.split(",")])
            counts = {"architecture_files": len(files)}
        elif args.command == "funnel":
            from ids_meta.funnel import build as build_funnel

            files = build_funnel(settings, args.source, args.output)
            counts = {"funnel_files": len(files)}
        else:
            from ids_meta.plotting.compare import build as build_comparison

            paths = _default_paths(settings, args.source)
            if args.input:
                input_path = args.input
                output_path = args.output or paths["comparison"]
                files = build_comparison(args.source, input_path, output_path, settings.section("plots"))
            else:
                if args.output:
                    raise ConfigError("--output is available for compare only together with --input")
                files = _build_classifier_comparisons(settings, args.source)
            counts = {"plot_files": len(files)}
        if (
            args.command in {"github", "openalex"} and args.action == "run"
            and not args.dry_run and _baselines_enabled(settings)
        ):
            baseline_args = argparse.Namespace(
                action="baseline", input=args.input, output=None, methods=None, ratings=None,
                resume=args.resume, dry_run=False, limit=None, collect_limit=None,
                classify_limit=None, classify_all=False, evaluate_only=False,
            )
            baseline_counts = _execute_baselines(baseline_args, settings, args.command)
            counts.update(baseline_counts)
        if (
            args.command in {"github", "openalex"}
            and args.action in {"classify", "baseline", "run"}
            and not getattr(args, "dry_run", False)
            and not getattr(args, "check_only", False)
            and not getattr(args, "fields", None)
            and not (args.action == "baseline" and getattr(args, "limit", None) is not None)
            and not getattr(args, "input", None)
            and not getattr(args, "output", None)
        ):
            from ids_meta.funnel import build as build_funnel

            counts["funnel_files"] = len(build_funnel(settings, args.command))
        LOG.info("command summary %s", "; ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        return 0
    except (ConfigError, ValueError, OSError, RuntimeError) as exc:
        # Logging intentionally remains on stdout so redirected run logs stay complete.
        LOG.error("command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
