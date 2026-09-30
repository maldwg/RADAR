from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Polygon

from ids_meta.common.cache import PipelineCache
from ids_meta.common.config import Settings, load_settings, load_term_profile
from ids_meta.common.csvio import read_csv
from ids_meta.common.logging import configure_logging, logger
from ids_meta.common.text import slug
from ids_meta.plotting.common import add_heading, configure_style, save_figure, stamp, write_table


LOG = logger("funnel")
OBSERVED = "observed"
CONFIGURED = "configured_capacity"
FUNNEL_COLORS = ["#163D5C", "#235B7A", "#2F7391", "#3E8B9F", "#59A091", "#83AF73", "#C39A45", "#C96A4A"]


@dataclass(frozen=True)
class FunnelStage:
    """Represent one count-bearing step in a source-to-catalogue funnel."""

    key: str
    label: str
    count: int
    basis: str
    detail: str
    removed_label: str = "Records removed"


@dataclass(frozen=True)
class ClassifierVariant:
    """Describe one persisted classifier output and its downstream files."""

    label: str
    identifier: str
    classifications: Path
    reconciled: Path
    catalogue: Path
    adjacent: Path | None = None


def _int(value: object, default: int = 0) -> int:
    """Convert a configuration or cache value to a non-negative integer."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _current_after(path: Path, predecessor: Path | None) -> bool:
    """Return whether a persisted stage exists and is no older than its predecessor."""
    if not path.is_file():
        return False
    return predecessor is None or not predecessor.is_file() or path.stat().st_mtime >= predecessor.stat().st_mtime


def _written_after_snapshot(path: Path, snapshot_at: str) -> bool:
    """Return whether a stage file plausibly belongs to the active database snapshot."""
    if not path.is_file():
        return False
    if not snapshot_at:
        return True
    try:
        timestamp = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True
    return path.stat().st_mtime + 1 >= timestamp


def _source_paths(settings: Settings, source: str) -> dict[str, Path]:
    """Resolve source-level cache, interim, processed, and plotting directories."""
    interim = settings.path("interim_dir") / source
    processed = settings.path("processed_dir") / source
    return {
        "cache": settings.path(f"{source}_cache"),
        "interim": interim,
        "processed": processed,
        "candidates": interim / ("candidates.csv" if source == "github" else "paper_candidates.csv"),
        "output": settings.path("plots_dir") / source / "pipeline_funnel",
    }


def _variant_label(relative: Path, rows: list[dict[str, str]]) -> tuple[str, str]:
    """Build a readable and stable classifier label from its variant directory."""
    parts = relative.parts
    if parts[:1] == ("models",) and len(parts) > 1:
        identifier = "/".join(parts[1:])
        return identifier, slug(identifier)
    if parts[:1] == ("ensemble",):
        return "Ensemble", "ensemble"
    if parts[:2] == ("baselines", "keyword"):
        return "Keyword baseline", "keyword"
    if parts[:2] == ("baselines", "nli"):
        identifier = "/".join(parts[2:]) or "zero-shot NLI"
        return f"Zero-shot NLI — {identifier}", "nli-" + slug(identifier)
    classifier = ""
    if rows:
        classifier = rows[0].get("classifier_id", "") or rows[0].get("llm_model", "")
    classifier = classifier.strip() or "Configured LLM"
    return classifier, slug(classifier)


def _classifier_variants(settings: Settings, source: str, paths: dict[str, Path]) -> list[ClassifierVariant]:
    """Discover every available classifier output without assuming a fixed model list."""
    interim = paths["interim"]
    processed = paths["processed"]
    variants: list[ClassifierVariant] = []
    if not interim.is_dir():
        return variants
    for classifications in sorted(interim.rglob("classifications.csv")):
        relative = classifications.parent.relative_to(interim)
        rows = read_csv(classifications)
        label, identifier = _variant_label(relative, rows)
        output_root = processed / relative
        variants.append(ClassifierVariant(
            label=label,
            identifier=identifier,
            classifications=classifications,
            reconciled=classifications.parent / "reconciled.csv",
            catalogue=output_root / ("repositories.csv" if source == "github" else "base_papers.csv"),
            adjacent=None if source == "github" else output_root / "adjacent_base_papers.csv",
        ))
    return variants


def _github_shared_stages(
    settings: Settings, profile: dict[str, Any], paths: dict[str, Path],
) -> tuple[list[FunnelStage], dict[str, object]]:
    """Derive GitHub discovery and prefilter counts from the active cache snapshot."""
    config = settings.section("github")
    queries = [str(value).strip() for value in profile.get("search_queries", []) if str(value).strip()]
    sample_size = _int(config.get("query_sample_size"), 100)
    configured_capacity = len(queries) * sample_size
    collection_limit = _int(config.get("collection_limit"), configured_capacity)
    stages: list[FunnelStage] = []
    metadata: dict[str, object] = {
        "source": "github", "query_count": len(queries), "query_sample_size": sample_size,
        "collection_limit": collection_limit, "snapshot_at": "configuration only",
        "complete_through": "configuration",
    }
    if not paths["cache"].is_file():
        stages.extend([
            FunnelStage("initial_results", "Potential query-result capacity", configured_capacity, CONFIGURED,
                        f"{len(queries)} queries × {sample_size:,} configured results/query"),
            FunnelStage("active_collection", "Configured collection ceiling", min(configured_capacity, collection_limit), CONFIGURED,
                        f"github.collection_limit={collection_limit:,}", "Outside configured collection ceiling"),
        ])
        return stages, metadata

    active_ids: list[str] = []
    with PipelineCache(paths["cache"]) as cache:
        collection = cache.get_record("github.collection", "latest") or {}
        strategy = str(collection.get("search_strategy", ""))
        search_records = [
            (key, payload) for key, payload, _fetched_at in cache.records("github.search_candidate")
            if not strategy or str(payload.get("search_strategy", "")) == strategy
        ]
        raw_matches = sum(len(set(payload.get("matched_queries", []))) for _key, payload in search_records)
        unique_results = len(search_records)
        active_ids = [str(value) for value in collection.get("repository_ids", []) if value]
        selected_at = str(collection.get("selected_at", ""))
        stored_limit = collection.get("collection_limit")
        if stored_limit not in (None, ""):
            collection_limit = _int(stored_limit, collection_limit)
        metadata.update({
            "collection_limit": collection_limit, "snapshot_at": selected_at or "database snapshot",
            "search_strategy": strategy, "complete_through": "collection",
        })
        if raw_matches:
            stages.append(FunnelStage(
                "initial_results", "Retrieved query matches", raw_matches, OBSERVED,
                "Overlapping matches are counted once per query",
            ))
        elif configured_capacity:
            stages.append(FunnelStage(
                "initial_results", "Potential query-result capacity", configured_capacity, CONFIGURED,
                f"{len(queries)} queries × {sample_size:,} configured results/query",
            ))
        if unique_results:
            stages.append(FunnelStage(
                "deduplicated", "Unique repositories after query deduplication", unique_results, OBSERVED,
                "Repository IDs merge repeated hits across search queries", "Duplicate cross-query matches",
            ))
        if active_ids:
            stages.append(FunnelStage(
                "active_collection", "Repositories retained in the active collection", len(active_ids), OBSERVED,
                f"Observed selection; configured/run limit={collection_limit:,}", "Outside the active collection",
            ))
            active = set(active_ids)
            enriched = sum(cache.get_record("github.repository", key) is not None for key in active)
            stages.append(FunnelStage(
                "evidence_cached", "Repositories with cached evidence", enriched, OBSERVED,
                "Metadata, README, documentation, and commit evidence when available", "Awaiting evidence enrichment",
            ))
            metadata["collection_complete"] = enriched == len(active_ids)
            metadata["complete_through"] = "evidence" if enriched else "collection"
        else:
            metadata["collection_complete"] = False

    candidates = paths["candidates"]
    collection_complete = bool(metadata.get("collection_complete"))
    if collection_complete and _written_after_snapshot(candidates, str(metadata.get("snapshot_at", ""))):
        rows = read_csv(candidates)
        active = set(active_ids)
        rows = [row for row in rows if not active or row.get("github_repo_id", "") in active]
        bands = [str(row.get("prefilter_band", "")).strip().casefold() for row in rows]
        eligible = sum(value in {"candidate", "borderline"} for value in bands)
        rejected = sum(value == "reject" for value in bands)
        stages.append(FunnelStage(
            "prefilter_eligible", "Eligible after deterministic prefilter", eligible, OBSERVED,
            f"candidate + borderline; {rejected:,} explicit rejects", "Rejected by deterministic prefilter",
        ))
        metadata.update({"prefilter_rejected": rejected, "complete_through": "prefilter"})
    return stages, metadata


def _openalex_search_queries(profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return direct queries and the expanded survey-query plan used by collection."""
    direct = [str(value) for value in profile.get("direct_queries", []) if str(value).strip()]
    survey = [str(value) for value in profile.get("survey_queries", []) if str(value).strip()]
    publishers = profile.get("survey_publishers", {})
    publisher_queries = [
        f"{query} {aliases[0]}" for query in survey
        for aliases in publishers.values() if isinstance(aliases, list) and aliases
    ] if isinstance(publishers, dict) else []
    return direct, list(dict.fromkeys([*survey, *publisher_queries]))


def _openalex_shared_stages(
    settings: Settings, profile: dict[str, Any], paths: dict[str, Path],
) -> tuple[list[FunnelStage], dict[str, object]]:
    """Derive OpenAlex discovery and prefilter counts from its active cache snapshot."""
    config = settings.section("openalex")
    direct_queries, survey_queries = _openalex_search_queries(profile)
    direct_sample = _int(config.get("query_sample_size"), 100)
    survey_sample = _int(config.get("survey_query_sample_size"), direct_sample)
    reference_sample = _int(config.get("reference_sample_size"), 0)
    configured_capacity = (
        len(direct_queries) * direct_sample + len(survey_queries) * survey_sample + reference_sample
    )
    collection_limit = _int(config.get("collection_limit"), configured_capacity)
    metadata: dict[str, object] = {
        "source": "openalex", "query_count": len(direct_queries) + len(survey_queries),
        "direct_query_count": len(direct_queries), "survey_query_count": len(survey_queries),
        "query_sample_size": direct_sample, "survey_query_sample_size": survey_sample,
        "reference_sample_size": reference_sample, "collection_limit": collection_limit,
        "snapshot_at": "configuration only", "complete_through": "configuration",
    }
    if not paths["cache"].is_file():
        return [
            FunnelStage("initial_results", "Potential source-record capacity", configured_capacity, CONFIGURED,
                        "Direct + survey-query samples + backward-reference budget"),
            FunnelStage("active_collection", "Configured candidate-paper ceiling", min(configured_capacity, collection_limit), CONFIGURED,
                        f"openalex.collection_limit={collection_limit:,}", "Outside configured collection ceiling"),
        ], metadata

    with PipelineCache(paths["cache"]) as cache:
        collection = cache.get_record("openalex.collection", "latest") or {}
        active_ids = [str(value) for value in collection.get("work_ids", []) if value]
        survey_ids = [str(value) for value in collection.get("survey_ids", []) if value]
        selected_at = str(collection.get("selected_at", ""))
        stored_limit = collection.get("collection_limit")
        if stored_limit not in (None, ""):
            collection_limit = _int(stored_limit, collection_limit)
        coverage = cache.coverage_rows("openalex")
        direct_set, survey_set = set(direct_queries), set(survey_queries)
        reached_depth = _int(collection.get("snowball_reached_depth"), 0)
        direct_scanned = sum(_int(row.get("total_count")) for row in coverage if row.get("shard") == "direct_cursor" and row.get("query") in direct_set)
        survey_scanned = sum(_int(row.get("total_count")) for row in coverage if row.get("shard") == "survey_cursor" and row.get("query") in survey_set)
        references_scanned = sum(
            _int(row.get("total_count")) for row in coverage
            if row.get("shard") == "backward_references"
            and str(row.get("query", "")).startswith("snowball_depth:")
            and _int(str(row.get("query", "")).split(":")[-1]) <= reached_depth
        )
        scanned = direct_scanned + survey_scanned + references_scanned
        unique_retained = len(set(active_ids) | set(survey_ids))
        metadata.update({
            "collection_limit": collection_limit, "snapshot_at": selected_at or "database snapshot",
            "direct_results_scanned": direct_scanned, "survey_results_scanned": survey_scanned,
            "reference_results_scanned": references_scanned, "snowball_reached_depth": reached_depth,
            "complete_through": "collection", "collection_complete": bool(active_ids),
        })
        stages: list[FunnelStage] = []
        if scanned:
            stages.append(FunnelStage(
                "initial_results", "Source records inspected", scanned, OBSERVED,
                f"direct={direct_scanned:,}; survey={survey_scanned:,}; backward references={references_scanned:,}",
            ))
        elif configured_capacity:
            stages.append(FunnelStage(
                "initial_results", "Potential source-record capacity", configured_capacity, CONFIGURED,
                "Direct + survey-query samples + backward-reference budget",
            ))
        if unique_retained:
            stages.append(FunnelStage(
                "deduplicated", "Unique retained works after source deduplication", unique_retained, OBSERVED,
                "Includes auxiliary surveys used only for traversal", "Duplicate or ineligible source records",
            ))
        if active_ids:
            stages.append(FunnelStage(
                "active_collection", "Candidate papers in the active collection", len(active_ids), OBSERVED,
                f"Auxiliary surveys excluded; configured/run limit={collection_limit:,}", "Auxiliary traversal sources",
            ))

    candidates = paths["candidates"]
    if bool(metadata.get("collection_complete")) and _written_after_snapshot(candidates, str(metadata.get("snapshot_at", ""))):
        rows = read_csv(candidates)
        active = set(active_ids)
        rows = [row for row in rows if not active or row.get("openalex_id", "") in active]
        bands = [str(row.get("prefilter_band", "")).strip().casefold() for row in rows]
        eligible = sum(value in {"candidate", "borderline"} for value in bands)
        rejected = sum(value == "reject" for value in bands)
        stages.append(FunnelStage(
            "prefilter_eligible", "Eligible after deterministic prefilter", eligible, OBSERVED,
            f"candidate + borderline; {rejected:,} explicit rejects", "Rejected by deterministic prefilter",
        ))
        metadata.update({"prefilter_rejected": rejected, "complete_through": "prefilter"})
    return stages, metadata


def _variant_stages(
    source: str, shared: list[FunnelStage], variant: ClassifierVariant, candidates_path: Path,
) -> tuple[list[FunnelStage], dict[str, int]]:
    """Append classifier decisions and downstream catalogue counts to shared stages."""
    stages = list(shared)
    decisions: dict[str, int] = {}
    if not candidates_path.is_file():
        return stages, decisions
    if not _current_after(variant.classifications, candidates_path):
        return stages, decisions
    rows = read_csv(variant.classifications)
    decisions = {}
    for row in rows:
        decision = str(row.get("decision", "")).strip().casefold() or "unknown"
        decisions[decision] = decisions.get(decision, 0) + 1
    included_values = {"accept"} if source == "github" else {"accept", "accept_adjacent"}
    included = sum(count for decision, count in decisions.items() if decision in included_values)
    excluded = decisions.get("exclude", 0)
    unresolved = len(rows) - included - excluded
    stages.append(FunnelStage(
        "classified", f"Classified by {variant.label}", len(rows), OBSERVED,
        "Final consensus rows, including uncertain and error outcomes", "Eligible records not yet classified",
    ))
    stages.append(FunnelStage(
        "included", "Included by the classifier", included, OBSERVED,
        f"excluded={excluded:,}; unresolved/error={unresolved:,}", "Excluded or unresolved classifications",
    ))
    final_count = 0
    if _current_after(variant.catalogue, variant.classifications):
        final_count += len(read_csv(variant.catalogue))
    if variant.adjacent is not None and _current_after(variant.adjacent, variant.classifications):
        final_count += len(read_csv(variant.adjacent))
    if final_count or variant.catalogue.is_file() or (variant.adjacent is not None and variant.adjacent.is_file()):
        stages.append(FunnelStage(
            "final_catalogue", "Exported catalogue records", final_count, OBSERVED,
            "Core + adjacent exports" if source == "openalex" else "Validated accepted repository export",
            "Removed during reconciliation or export",
        ))
    return stages, {"included": included, "excluded": excluded, "unresolved": unresolved, **decisions}


def _query_context(source: str, metadata: dict[str, object]) -> str:
    """Render concise source-specific acquisition context for the figure subtitle."""
    if source == "github":
        return (
            f"{metadata.get('query_count', 0)} configured queries • "
            f"up to {int(metadata.get('query_sample_size', 0)):,} results/query • "
            f"active collection limit {int(metadata.get('collection_limit', 0)):,}"
        )
    return (
        f"{metadata.get('direct_query_count', 0)} direct + {metadata.get('survey_query_count', 0)} survey searches • "
        f"reference sampling up to {int(metadata.get('reference_sample_size', 0)):,} • "
        f"active collection limit {int(metadata.get('collection_limit', 0)):,}"
    )


def _stage_table(
    source: str, classifier: str, stages: list[FunnelStage], metadata: dict[str, object],
) -> pd.DataFrame:
    """Create the loss-accounting table used directly by the funnel figure."""
    rows: list[dict[str, object]] = []
    previous: int | None = None
    for order, stage in enumerate(stages, 1):
        removed = max(0, previous - stage.count) if previous is not None else 0
        rows.append({
            "source": source, "classifier": classifier, "stage_order": order,
            "stage_key": stage.key, "stage_label": stage.label, "count": stage.count,
            "basis": stage.basis, "basis_detail": stage.detail,
            "removed_from_previous": removed,
            "removed_percent_of_previous": removed / previous if previous else 0.0,
            "removed_label": stage.removed_label if previous is not None else "",
            "snapshot_at": metadata.get("snapshot_at", ""),
        })
        previous = stage.count
    return pd.DataFrame(rows)


def _draw_funnel(
    source: str, classifier: str, stages: list[FunnelStage], metadata: dict[str, object],
    out_dir: Path, stem: str, formats: list[str], dpi: int,
) -> list[Path]:
    """Render a publication-ready reverse-pyramid funnel with explicit losses."""
    table = _stage_table(source, classifier, stages, metadata)
    write_table(table, out_dir / "data", stem)
    configure_style()
    height = max(6.8, 1.08 * len(stages) + 2.5)
    fig, ax = plt.subplots(figsize=(12.5, height))
    source_name = "GitHub" if source == "github" else "OpenAlex"
    title_suffix = f" — {classifier}" if classifier else ""
    add_heading(
        fig, f"{source_name} evidence-to-catalogue funnel{title_suffix}",
        _query_context(source, metadata) + f" • snapshot: {metadata.get('snapshot_at', 'configuration only')}",
        top=.88,
    )
    ax.set_xlim(-1.06, 1.64)
    ax.set_ylim(-.12, len(stages) + .18)
    ax.axis("off")
    maximum = max([stage.count for stage in stages] or [1])
    widths = [max(.24, math.sqrt(stage.count / maximum) if maximum else .24) for stage in stages]
    for index, stage in enumerate(stages):
        y_top = len(stages) - index
        y_bottom = y_top - .82
        top_width = widths[index]
        bottom_width = widths[index + 1] if index + 1 < len(widths) else max(.20, top_width * .82)
        color = FUNNEL_COLORS[min(index, len(FUNNEL_COLORS) - 1)]
        patch = Polygon(
            [(-top_width, y_top), (top_width, y_top), (bottom_width, y_bottom), (-bottom_width, y_bottom)],
            closed=True, facecolor=color, edgecolor="white", linewidth=1.8,
            hatch="///" if stage.basis == CONFIGURED else None,
            alpha=.58 if stage.basis == CONFIGURED else .97,
        )
        ax.add_patch(patch)
        center_y = (y_top + y_bottom) / 2
        ax.text(0, center_y + .10, f"{stage.count:,}", ha="center", va="center", color="white",
                fontsize=16, fontweight="bold")
        ax.text(0, center_y - .15, textwrap.fill(stage.label, width=34), ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        ax.plot([top_width + .03, 1.08], [center_y, center_y], color="#8A99A3", linewidth=.8)
        ax.text(1.11, center_y + .08, "Observed" if stage.basis == OBSERVED else "Configured capacity",
                ha="left", va="bottom", fontsize=8, fontweight="bold", color="#334A5B")
        ax.text(1.11, center_y - .06, textwrap.fill(stage.detail, width=42), ha="left", va="top", fontsize=7.4, color="#5F6C76")
        if index:
            previous = stages[index - 1].count
            removed = max(0, previous - stage.count)
            rate = removed / previous if previous else 0.0
            boundary_y = y_top + .02
            ax.text(
                -1.03, boundary_y,
                f"−{removed:,}  ({rate:.1%})\n{stage.removed_label}",
                ha="left", va="center", fontsize=7.7, color="#8B3E35",
                bbox={"boxstyle": "round,pad=.28", "facecolor": "#FFF7F3", "edgecolor": "#D8A69D", "linewidth": .8},
            )
    legend = [
        Patch(facecolor=FUNNEL_COLORS[1], edgecolor="white", label="Observed persisted count"),
        Patch(facecolor=FUNNEL_COLORS[1], edgecolor="#6E7E89", hatch="///", alpha=.58,
              label="Configuration capacity; not an observed result"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(.34, -.02), ncol=2, frameon=False, fontsize=8)
    stamp(
        fig,
        "Exact counts and losses are printed and exported to CSV. Funnel width uses a square-root scale so small retained sets remain visible; area is not proportional to count.",
    )
    return save_figure(fig, out_dir, stem, formats, dpi)


def _write_readme(
    out_dir: Path, source: str, outputs: list[Path], metadata: dict[str, object], variants: list[ClassifierVariant],
) -> Path:
    """Write interpretation and provenance guidance beside the funnel figures."""
    source_name = "GitHub" if source == "github" else "OpenAlex"
    path = out_dir / "README.md"
    figures = [item.name for item in outputs if item.suffix == ".svg"]
    lines = [
        f"# {source_name} evidence-to-catalogue funnel", "",
        "The reverse pyramid follows records from source acquisition through deduplication, collection, prefiltering, classification, and export whenever those stages are available.", "",
        f"- Snapshot: `{metadata.get('snapshot_at', 'configuration only')}`",
        f"- Complete through: `{metadata.get('complete_through', 'configuration')}`",
        f"- Classifier variants found: `{len(variants)}`", "",
        "Solid segments are observed database or CSV counts. Hatched segments are capacities derived from the TOML configuration and must not be reported as observed sample sizes. Query counts are acquisition context, not funnel records. Width uses a square-root scale only for readability; use the printed values or companion CSV for exact reporting.", "",
        "## Figures", "",
    ]
    lines.extend(f"- `{name}`" for name in figures)
    lines.extend(["", "Exact stage counts, provenance, transition losses, and percentages are under `data/`.", ""])
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _clear_previous_bundle(out_dir: Path) -> None:
    """Remove only stale files owned by the dedicated funnel generator."""
    for directory in (out_dir, out_dir / "data"):
        if not directory.is_dir():
            continue
        for pattern in ("*_pipeline_funnel*.svg", "*_pipeline_funnel*.pdf", "*_pipeline_funnel*.csv"):
            for path in directory.glob(pattern):
                path.unlink()


def build(settings: Settings, source: str, output: Path | None = None) -> list[Path]:
    """Build database-aware funnel figures for one discovery source without network access."""
    if source not in {"github", "openalex"}:
        raise ValueError(f"unsupported funnel source: {source}")
    paths = _source_paths(settings, source)
    out_dir = output or paths["output"]
    profile = load_term_profile(settings, source)
    if source == "github":
        shared, metadata = _github_shared_stages(settings, profile, paths)
    else:
        shared, metadata = _openalex_shared_stages(settings, profile, paths)
    if not shared:
        raise ValueError(f"no configured or persisted {source} funnel stages are available")
    discovered_variants = _classifier_variants(settings, source, paths)
    formats = [str(value) for value in settings.section("plots").get("formats", ["svg", "pdf"])]
    dpi = _int(settings.section("plots").get("dpi"), 160)
    _clear_previous_bundle(out_dir)
    outputs: list[Path] = []
    active_variants: list[ClassifierVariant] = []
    variant_payloads: list[tuple[ClassifierVariant, list[FunnelStage], dict[str, int]]] = []
    for variant in discovered_variants:
        stages, decisions = _variant_stages(source, shared, variant, paths["candidates"])
        if len(stages) > len(shared):
            active_variants.append(variant)
            variant_payloads.append((variant, stages, decisions))
    if variant_payloads:
        all_keys = {stage.key for _variant, stages, _decisions in variant_payloads for stage in stages}
        if "final_catalogue" in all_keys:
            metadata["complete_through"] = "catalogue export"
        else:
            partial = any(
                next(stage.count for stage in stages if stage.key == "classified")
                < next(stage.count for stage in stages if stage.key == "prefilter_eligible")
                for _variant, stages, _decisions in variant_payloads
                if any(stage.key == "prefilter_eligible" for stage in stages)
            )
            metadata["complete_through"] = "classification (partial)" if partial else "classification"
    if variant_payloads:
        for number, (variant, stages, decisions) in enumerate(variant_payloads, 1):
            variant_metadata = dict(metadata)
            variant_metadata.update({f"decision_{key}": value for key, value in decisions.items()})
            stem = f"{number:02d}_pipeline_funnel_{variant.identifier}"
            outputs.extend(_draw_funnel(
                source, variant.label, stages, variant_metadata, out_dir, stem, formats, dpi,
            ))
    else:
        outputs.extend(_draw_funnel(source, "", shared, metadata, out_dir, "00_pipeline_funnel", formats, dpi))
    readme = _write_readme(out_dir, source, outputs, metadata, active_variants)
    outputs.append(readme)
    LOG.info(
        "pipeline funnel complete source=%s stages=%d classifiers=%d files=%d output=%s",
        source, len(shared), len(active_variants), len(outputs), out_dir,
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    """Generate source funnel figures independently from collection and classification."""
    parser = argparse.ArgumentParser(description="Build database-aware IDS meta-study pipeline funnels")
    parser.add_argument("source", choices=("github", "openalex"))
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    configure_logging(str(settings.section("logging").get("level", "INFO")))
    build(settings, args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
