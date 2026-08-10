from __future__ import annotations

import argparse
from pathlib import Path

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import ConfigError, Settings, load_prompt_profile, load_settings, load_system_type_profile, load_term_profile
from ids_meta.common.run import RunManifest
from ids_meta.common.logging import configure_logging, logger
from ids_meta.github.collect import collect as collect_github
from ids_meta.github.pipeline import classify as classify_github
from ids_meta.github.pipeline import export as export_github
from ids_meta.github.pipeline import prefilter as prefilter_github
from ids_meta.github.reconcile import reconcile as reconcile_github
from ids_meta.openalex.collect import collect as collect_openalex
from ids_meta.openalex.pipeline import classify as classify_openalex
from ids_meta.openalex.pipeline import export as export_openalex
from ids_meta.openalex.pipeline import export_survey_sources as export_openalex_surveys
from ids_meta.openalex.pipeline import prefilter as prefilter_openalex
from ids_meta.openalex.reconcile import reconcile as reconcile_openalex

LOG = logger("cli")


def _default_paths(settings: Settings, source: str) -> dict[str, Path]:
    """Build the default cache, CSV, plot, and comparison paths for a source."""
    interim = settings.path("interim_dir") / source
    processed = settings.path("processed_dir") / source
    plots = settings.path("plots_dir") / source
    if source == "github":
        return {
            "cache": settings.path("github_cache"), "candidates": interim / "candidates.csv",
            "classifications": interim / "classifications.csv", "reconciled": interim / "reconciled.csv",
            "catalogue": processed / "repositories.csv",
            "plots": plots, "comparison": settings.path("plots_dir") / "comparisons" / source,
        }
    return {
        "cache": settings.path("openalex_cache"), "candidates": interim / "paper_candidates.csv",
        "surveys": interim / "survey_sources.csv",
        "classifications": interim / "classifications.csv", "reconciled": interim / "reconciled.csv",
        "catalogue": processed / "base_papers.csv",
        "adjacent": processed / "adjacent_base_papers.csv", "plots": plots,
        "comparison": settings.path("plots_dir") / "comparisons" / source,
    }


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


def _execute_github(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Execute the requested GitHub pipeline stage or complete workflow."""
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
            rows = classify_github(input_path, output_path, cache, settings.section("ollama"), classify_limit, args.resume, args.dry_run, prompt_config=prompt_config, system_type_profile=system_type_profile)
            manifest.counts["classifications"] = len(rows)
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
            generated = build_github_plots(input_path, output_path, settings.section("plots"), system_type_profile)
            manifest.counts["plot_files"] = len(generated)
        manifest.coverage = cache.coverage_rows("github")
    manifest.status = "complete"; manifest.write()
    LOG.info("completed command source=github stage=%s counts=%s", args.action, manifest.counts)
    return manifest.counts


def _execute_openalex(args: argparse.Namespace, settings: Settings) -> dict[str, int]:
    """Execute the requested OpenAlex pipeline stage or complete workflow."""
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
            rows = classify_openalex(input_path, output_path, cache, settings.section("ollama"), classify_limit, args.resume, args.dry_run, prompt_config=prompt_config)
            manifest.counts["classifications"] = len(rows)
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
        for action in ("collect", "prefilter", "classify", "reconcile", "export", "plot", "run"):
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
        else:
            from ids_meta.plotting.compare import build as build_comparison

            paths = _default_paths(settings, args.source)
            input_path = args.input or paths["classifications"]
            output_path = args.output or paths["comparison"]
            files = build_comparison(args.source, input_path, output_path, settings.section("plots"))
            counts = {"plot_files": len(files)}
        LOG.info("command summary %s", "; ".join(f"{key}={value}" for key, value in sorted(counts.items())))
        return 0
    except (ConfigError, ValueError, OSError, RuntimeError) as exc:
        # Logging intentionally remains on stdout so redirected run logs stay complete.
        LOG.error("command failed: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
