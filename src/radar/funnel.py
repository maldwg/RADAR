from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Polygon

from radar.common.cache import PipelineCache
from radar.common.config import Settings, load_settings, load_term_profile
from radar.common.csvio import read_csv
from radar.common.logging import configure_logging, logger
from radar.common.text import slug
from radar.common.limits import optional_limit, query_result_limit, limit_label, capacity_min
from radar.common.provenance import audited_bundle, register_input, freeze_database
from radar.plotting.common import configure_style, save_figure, write_table


LOG = logger("funnel")
OBSERVED = "observed"
CONFIGURED = "configured_capacity"
NOT_OBSERVED = "not_observed"
FUNNEL_COLORS = ["#163D5C", "#3E8B9F", "#59A091", "#83AF73", "#C39A45"]


@dataclass(frozen=True)
class FunnelStage:
    """Represent one count-bearing step in a source-to-catalogue funnel."""

    key: str
    label: str
    count: int | None
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
        raise ValueError("funnel requires a collection timestamp")
    try:
        timestamp = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00")).timestamp()
    except ValueError as exc:
        raise ValueError("funnel requires a valid collection timestamp") from exc
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
    if parts[:2] == ("baselines", "llm_nli"):
        identifier = "/".join(parts[2:])
        return f"LLM NLI — {identifier}", "llm-nli-" + slug(identifier)
    if parts[:2] == ("baselines", "plain_llm"):
        identifier = "/".join(parts[2:])
        return f"Plain LLM — {identifier}", "plain-llm-" + slug(identifier)
    raise ValueError("funnel requires model, ensemble or named baseline directories")


def _classifier_variants(settings: Settings, source: str, paths: dict[str, Path]) -> list[ClassifierVariant]:
    """Discover every available classifier output without assuming a fixed model list."""
    interim = paths["interim"]
    processed = paths["processed"]
    variants: list[ClassifierVariant] = []
    if not interim.is_dir():
        return variants
    for classifications in sorted(interim.rglob("classifications.csv")):
        if "pilots" in classifications.relative_to(interim).parts:
            continue
        relative = classifications.parent.relative_to(interim)
        if not relative.parts or relative.parts[0] not in {"models", "ensemble", "baselines"}:
            continue
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
    collection_limit = optional_limit(config.get("collection_limit"), "collection_limit")
    sample_size = query_result_limit(config, collection_limit)
    configured_capacity = None if sample_size is None else len(queries) * sample_size
    stages: list[FunnelStage] = []
    metadata: dict[str, object] = {
        "source": "github", "query_count": len(queries), "query_sample_size": sample_size,
        "collection_limit": collection_limit, "snapshot_at": "configuration only",
        "complete_through": "configuration",
    }
    if not paths["cache"].is_file():
        stages.extend([
            FunnelStage("initial_results", "Discovery capacity", configured_capacity, CONFIGURED,
                        f"{len(queries)} queries × {limit_label(sample_size)} configured results/query"),
            FunnelStage("active_collection", "Collection capacity", capacity_min(configured_capacity, collection_limit), CONFIGURED,
                        f"github.collection_limit={limit_label(collection_limit)}", "Outside configured collection ceiling"),
        ])
        return stages, metadata

    active_ids: list[str] = []
    with PipelineCache(paths["cache"], read_only=True) as cache:
        collection = cache.get_record("github.collection", "latest")
        if collection is None or not collection.get("selected_at"):
            raise ValueError("github funnel requires current collection metadata")
        if "sample_size_per_query" in collection:
            sample_size = collection["sample_size_per_query"]
            configured_capacity = None if sample_size is None else len(queries) * sample_size
            metadata["query_sample_size"] = sample_size
        strategy = str(collection.get("search_strategy", ""))
        search_records = [
            (key, payload) for key, payload, _fetched_at in cache.records("github.search_candidate")
            if str(payload.get("search_strategy", "")) == strategy
        ]
        raw_matches = sum(len(set(payload.get("matched_queries", []))) for _key, payload in search_records)
        unique_results = len(search_records)
        active_ids = [str(value) for value in collection.get("repository_ids", []) if value]
        selected_at = str(collection.get("selected_at", ""))
        if "collection_limit" in collection:
            collection_limit = optional_limit(collection["collection_limit"], "collection_limit")
        metadata.update({
            "collection_limit": collection_limit, "snapshot_at": selected_at,
            "search_strategy": strategy, "complete_through": "collection",
        })
        if raw_matches:
            stages.append(FunnelStage(
                "initial_results", "Discovery: query matches", raw_matches, OBSERVED,
                "Overlapping matches are counted once per query",
            ))
        else:
            stages.append(FunnelStage(
                "initial_results", "Discovery capacity", configured_capacity, CONFIGURED,
                f"{len(queries)} queries × {limit_label(sample_size)} configured results/query",
            ))
        if unique_results:
            stages.append(FunnelStage(
                "deduplicated", "Deduplication", unique_results, OBSERVED,
                "Repository IDs merge repeated hits across search queries", "Duplicate cross-query matches",
            ))
        if active_ids:
            stages.append(FunnelStage(
                "active_collection", "Collection", len(active_ids), OBSERVED,
                f"Observed selection; configured/run limit={limit_label(collection_limit)}", "Outside the active collection",
            ))
            active = set(active_ids)
            enriched = sum(cache.get_record("github.repository", key) is not None for key in active)
            stages.append(FunnelStage(
                "evidence_cached", "Evidence enrichment", enriched, OBSERVED,
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
        rows = [row for row in rows if row.get("github_repo_id", "") in active]
        bands = [str(row.get("prefilter_band", "")).strip().casefold() for row in rows]
        eligible = sum(value in {"candidate", "borderline"} for value in bands)
        rejected = sum(value == "reject" for value in bands)
        stages.append(FunnelStage(
            "prefilter_eligible", "Prefilter", eligible, OBSERVED,
            f"candidate + borderline; {rejected:,} explicit rejects", "Rejected by deterministic prefilter",
        ))
        metadata.update({"prefilter_rejected": rejected, "complete_through": "prefilter"})
    return stages, metadata


def _openalex_search_queries(profile: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return direct queries and the expanded survey-query plan used by collection."""
    direct = [query["search"] for query in profile.get("queries", []) if query["search"].strip()]
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
    collection_limit = optional_limit(config.get("collection_limit"), "collection_limit")
    direct_sample = query_result_limit(config, collection_limit)
    survey_sample = query_result_limit(config, collection_limit, "survey_query_limit_factor")
    reference_sample = query_result_limit(config, collection_limit, "reference_limit_factor")
    configured_capacity = None if direct_sample is None else (
        len(direct_queries) * direct_sample + len(survey_queries) * survey_sample + reference_sample
    )
    metadata: dict[str, object] = {
        "source": "openalex", "query_count": len(direct_queries) + len(survey_queries),
        "direct_query_count": len(direct_queries), "survey_query_count": len(survey_queries),
        "query_sample_size": direct_sample, "survey_query_sample_size": survey_sample,
        "reference_sample_size": reference_sample, "collection_limit": collection_limit,
        "snapshot_at": "configuration only", "complete_through": "configuration",
    }
    if not paths["cache"].is_file():
        return [
            FunnelStage("initial_results", "Discovery capacity", configured_capacity, CONFIGURED,
                        "Direct + survey-query samples + backward-reference budget"),
            FunnelStage("active_collection", "Collection capacity", capacity_min(configured_capacity, collection_limit), CONFIGURED,
                        f"openalex.collection_limit={limit_label(collection_limit)}", "Outside configured collection ceiling"),
        ], metadata

    with PipelineCache(paths["cache"], read_only=True) as cache:
        collection = cache.get_record("openalex.collection", "latest")
        if collection is None or not collection.get("selected_at"):
            raise ValueError("openalex funnel requires current collection metadata")
        active_ids = [str(value) for value in collection.get("work_ids", []) if value]
        survey_ids = [str(value) for value in collection.get("survey_ids", []) if value]
        selected_at = str(collection.get("selected_at", ""))
        if "collection_limit" in collection:
            collection_limit = optional_limit(collection["collection_limit"], "collection_limit")
        if "sample_size_per_query" in collection:
            direct_sample = collection["sample_size_per_query"]
            survey_sample = collection.get("survey_sample_size_per_query", direct_sample)
            reference_sample = collection.get("reference_sample_size", direct_sample)
            configured_capacity = None if any(value is None for value in (direct_sample, survey_sample, reference_sample)) else (
                len(direct_queries) * direct_sample + len(survey_queries) * survey_sample + reference_sample
            )
            metadata.update({"query_sample_size": direct_sample, "survey_query_sample_size": survey_sample,
                             "reference_sample_size": reference_sample})
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
            "collection_limit": collection_limit, "snapshot_at": selected_at,
            "direct_results_scanned": direct_scanned, "survey_results_scanned": survey_scanned,
            "reference_results_scanned": references_scanned, "snowball_reached_depth": reached_depth,
            "survey_sources": len(set(survey_ids)),
            "auxiliary_only_surveys": len(set(survey_ids) - set(active_ids)),
            "unique_retained_with_surveys": unique_retained,
            "complete_through": "collection", "collection_complete": bool(active_ids),
        })
        stages: list[FunnelStage] = []
        if scanned:
            stages.append(FunnelStage(
                "initial_results", "Discovery: sampled hits", scanned, OBSERVED,
                f"direct={direct_scanned:,}; survey={survey_scanned:,}; backward references={references_scanned:,}",
            ))
        else:
            stages.append(FunnelStage(
                "initial_results", "Discovery capacity", configured_capacity, CONFIGURED,
                "Direct + survey-query samples + backward-reference budget",
            ))
        if unique_retained:
            stages.append(FunnelStage(
                "deduplicated", "Retention + deduplication", unique_retained, OBSERVED,
                "Unique retained candidates and auxiliary traversal surveys; not the full deduplicated discovery pool",
                "Duplicate, filtered or unselected hits",
            ))
        if active_ids:
            stages.append(FunnelStage(
                "active_collection", "Collection: candidate papers", len(active_ids), OBSERVED,
                f"Auxiliary surveys excluded; configured/run limit={limit_label(collection_limit)}", "Auxiliary traversal sources",
            ))

    candidates = paths["candidates"]
    if bool(metadata.get("collection_complete")) and _written_after_snapshot(candidates, str(metadata.get("snapshot_at", ""))):
        rows = read_csv(candidates)
        active = set(active_ids)
        rows = [row for row in rows if row.get("openalex_id", "") in active]
        bands = [str(row.get("prefilter_band", "")).strip().casefold() for row in rows]
        eligible = sum(value in {"candidate", "borderline"} for value in bands)
        rejected = sum(value == "reject" for value in bands)
        stages.append(FunnelStage(
            "prefilter_eligible", "Prefilter", eligible, OBSERVED,
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
        "classified", "Classification", len(rows), OBSERVED,
        "Final consensus rows, including uncertain and error outcomes", "Eligible records not yet classified",
    ))
    stages.append(FunnelStage(
        "included", "Inclusion", included, OBSERVED,
        f"excluded={excluded:,}; unresolved/error={unresolved:,}", "Excluded or unresolved classifications",
    ))
    final_count = 0
    export_observed = _current_after(variant.catalogue, variant.classifications)
    if export_observed:
        final_count += len(read_csv(variant.catalogue))
    if variant.adjacent is not None and _current_after(variant.adjacent, variant.classifications):
        export_observed = True
        final_count += len(read_csv(variant.adjacent))
    if export_observed:
        stages.append(FunnelStage(
            "final_catalogue", "Catalogue export", final_count, OBSERVED,
            "Core + adjacent exports" if source == "openalex" else "Validated accepted repository export",
            "Removed during reconciliation or export",
        ))
    return stages, {"included": included, "excluded": excluded, "unresolved": unresolved, **decisions}


def _aligned_stages(stages: list[FunnelStage]) -> list[FunnelStage]:
    """Prepend crawler hits to five common stages without inferring discovery losses."""
    lookup = {stage.key: stage for stage in stages}
    aligned: list[FunnelStage] = []
    for alternatives, key, label in (
        (("initial_results",), "initial_results", "Crawler hits (overlap)"),
        (("active_collection",), "candidate_selection", "Candidate selection"),
        (("prefilter_eligible", "evidence_cached"), "enrichment_prefilter", "Enrichment / prefilter"),
        (("classified",), "classified", "Classification"),
        (("included",), "included", "Inclusion"),
        (("final_catalogue",), "final_catalogue", "Export"),
    ):
        stage = next((lookup[name] for name in alternatives if name in lookup), None)
        if stage is None:
            aligned.append(FunnelStage(key, label, None, NOT_OBSERVED, "No compatible saved count for this stage"))
            continue
        detail = stage.detail
        if key == "initial_results" and stage.basis == CONFIGURED:
            label = "Crawler capacity"
        if key == "enrichment_prefilter":
            if stage.key == "evidence_cached":
                detail += "; keyword prefilter not yet observed"
            elif "evidence_cached" in lookup:
                detail = f"Evidence cached for {lookup['evidence_cached'].count:,} candidates; " + detail
        aligned.append(replace(stage, key=key, label=label, detail=detail))
    return aligned


def _discovery_context(
    source: str, stages: list[FunnelStage], metadata: dict[str, object],
) -> pd.DataFrame:
    """Preserve acquisition details in a table; only total hits appear in the figure."""
    lookup = {stage.key: stage for stage in stages}
    rows: list[dict[str, object]] = []

    def append(key: str, label: str, count: object, basis: str, detail: str, *, visible: bool = False) -> None:
        """Append one discovery count with its observation basis and display setting."""
        rows.append({
            "source": source, "context_key": key, "context_label": label, "count": count,
            "basis": basis, "basis_detail": detail, "visible_in_figure": visible,
            "snapshot_at": metadata.get("snapshot_at", ""),
        })

    initial = lookup.get("initial_results")
    if initial is not None:
        append("initial_results", "Sampled hits (overlap)" if initial.basis == OBSERVED else "Discovery capacity",
               initial.count, initial.basis, initial.detail, visible=True)
    if source == "github":
        unique = lookup.get("deduplicated")
        if unique is not None:
            append("deduplicated", "Unique search matches", unique.count, unique.basis, unique.detail)
        if initial is not None and unique is not None and initial.basis == unique.basis == OBSERVED:
            append("duplicate_matches", "Repeated query matches", max(0, initial.count - unique.count), OBSERVED,
                   "Duplicates across query matches, before candidate selection", visible=False)
    else:
        for key, label in (
            ("direct_results_scanned", "Direct-search hits"),
            ("survey_results_scanned", "Survey-search hits"),
            ("reference_results_scanned", "Reference hits → candidates"),
            ("auxiliary_only_surveys", "Auxiliary-only surveys"),
        ):
            if key in metadata:
                append(key, label, metadata[key], OBSERVED,
                       "Auxiliary discovery sources; outside the main candidate-loss sequence"
                       if key == "auxiliary_only_surveys" else "Sampled discovery hits; routes can overlap")
        append("unique_discovery_pool", "Unique discovery pool", None, "not_recorded",
               "The saved snapshot does not record the full unique pool before eligibility and selection")
        for key, label in (
            ("survey_sources", "Survey sources"),
            ("unique_retained_with_surveys", "Retained candidates and surveys"),
        ):
            if key in metadata:
                append(key, label, metadata[key], OBSERVED,
                       "May include surveys that also occur among selected candidates", visible=False)
    if "evidence_cached" in lookup:
        evidence = lookup["evidence_cached"]
        append("evidence_cached", "Evidence cached", evidence.count, evidence.basis, evidence.detail, visible=False)
    return pd.DataFrame(rows)


def _stage_table(
    source: str, classifier: str, stages: list[FunnelStage], metadata: dict[str, object],
) -> pd.DataFrame:
    """Create the loss-accounting table used directly by the funnel figure."""
    rows: list[dict[str, object]] = []
    previous: int | None = None
    for order, stage in enumerate(stages, 1):
        follows_crawler = order > 1 and stages[order - 2].key == "initial_results"
        known_transition = previous is not None and stage.count is not None and not follows_crawler
        removed = max(0, previous - stage.count) if known_transition else 0 if order == 1 else None
        rows.append({
            "source": source, "classifier": classifier, "stage_order": order,
            "stage_key": stage.key, "stage_label": stage.label, "count": stage.count,
            "basis": stage.basis, "basis_detail": stage.detail,
            "removed_from_previous": removed,
            "removed_percent_of_previous": removed / previous if known_transition and previous else 0.0 if known_transition or order == 1 else None,
            "removed_label": stage.removed_label if previous is not None and not follows_crawler else "",
            "snapshot_at": metadata.get("snapshot_at", ""),
        })
        previous = stage.count
    return pd.DataFrame(rows)


def _draw_funnel(
    source: str, classifier: str, stages: list[FunnelStage], metadata: dict[str, object],
    out_dir: Path, stem: str, formats: list[str], dpi: int,
) -> list[Path]:
    """Render counts with short pipeline labels and readable losses on the right."""
    stages = _aligned_stages(stages)
    table = _stage_table(source, classifier, stages, metadata)
    write_table(table, out_dir / "data", stem)
    height = max(5.6, .92 * len(stages) + 1.0)
    fig, ax = plt.subplots(figsize=(9.5, height))
    source_name = "GitHub" if source == "github" else "OpenAlex"
    fig.subplots_adjust(left=.025, right=.98, top=1.0 - .72 / height, bottom=.35 / height)
    fig.suptitle(
        f"{source_name} pipeline funnel",
        x=.5, y=1.0 - .10 / height, ha="center", va="top", fontsize=14, fontweight="bold",
    )
    ax.set_xlim(-1.06, 2.46)
    ax.set_ylim(.02, len(stages) + .10)
    ax.axis("off")
    maximum = max([stage.count for stage in stages if stage.count is not None] or [1])
    widths: list[float] = []
    for stage in stages:
        widths.append(widths[-1] if stage.count is None and widths else
                      max(.12, math.sqrt(stage.count / maximum) if stage.count is not None and maximum else .12))
    for index, stage in enumerate(stages):
        y_top = len(stages) - index
        y_bottom = y_top - .82
        top_width = widths[index]
        bottom_width = widths[index + 1] if index + 1 < len(widths) else max(.10, top_width * .82)
        color = "#E2E8EC" if stage.count is None else "#192F44" if index == 0 else FUNNEL_COLORS[min(index - 1, len(FUNNEL_COLORS) - 1)]
        patch = Polygon(
            [(-top_width, y_top), (top_width, y_top), (bottom_width, y_bottom), (-bottom_width, y_bottom)],
            closed=True, facecolor=color, edgecolor="white", linewidth=1.8,
            hatch="///" if stage.basis == CONFIGURED else None,
            alpha=.58 if stage.basis == CONFIGURED else .97,
        )
        ax.add_patch(patch)
        center_y = (y_top + y_bottom) / 2
        ax.text(0, center_y, "N/A" if stage.count is None else f"{stage.count:,}", ha="center", va="center",
                color="#5F6C76" if stage.count is None else "white",
                fontsize=17, fontweight="bold")
        ax.plot([top_width + .03, 1.08], [center_y, center_y], color="#8A99A3", linewidth=.8)
        ax.text(1.11, center_y, stage.label, ha="left", va="center",
                fontsize=13, fontweight="bold", color="#334A5B")
        if index > 1 and stage.count is not None and stages[index - 1].count is not None:
            previous = stages[index - 1].count
            removed = max(0, previous - stage.count)
            rate = removed / previous if previous else 0.0
            boundary_y = y_top + .02
            ax.text(
                1.11, boundary_y, f"−{removed:,} ({rate:.1%} removed)",
                ha="left", va="center", fontsize=12, color="#8B3E35",
            )
    if any(stage.basis in {CONFIGURED, NOT_OBSERVED} for stage in stages):
        legend = [
            Patch(facecolor=FUNNEL_COLORS[1], edgecolor="white", label="Observed count"),
        ]
        if any(stage.basis == CONFIGURED for stage in stages):
            legend.append(Patch(facecolor=FUNNEL_COLORS[1], edgecolor="#6E7E89", hatch="///", alpha=.58,
                                label="Configured capacity"))
        if any(stage.basis == NOT_OBSERVED for stage in stages):
            legend.append(Patch(facecolor="#E2E8EC", label="N/A: not observed"))
        ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, -.08),
                  ncol=len(legend), frameon=False, fontsize=9)
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
        "GitHub and OpenAlex use the same sequence: **Crawler hits (overlap) → Candidate selection → Enrichment / prefilter → Classification → Inclusion → Export**. Every figure keeps all six layers; unavailable counts are grey **N/A**, including baseline exports that have not been produced. No loss percentage is inferred across an unavailable stage or from overlapping crawler hits to selected candidates.", "",
        f"- Snapshot: `{metadata.get('snapshot_at', 'configuration only')}`",
        f"- Complete through: `{metadata.get('complete_through', 'configuration')}`",
        f"- Classifier variants found: `{len(variants)}`", "",
        "Solid segments are observed database or CSV counts. Hatched segments are capacities derived from the TOML configuration and must not be reported as observed sample sizes. Width uses a square-root scale only for readability; use the printed values or companion CSV for exact reporting.", "",
        "Labels name the pipeline steps. Right-side transition annotations show the number removed and its percentage of the preceding stage. Detailed stage explanations and loss reasons are preserved in the companion CSVs rather than printed as subcaptions.", "",
        "The top layer shows overlapping crawler hits, not unique records. Candidate-selection counts are the starting denominator for subsequent removal percentages. Enrichment and keyword-prefilter eligibility share one stage; when only evidence enrichment has completed, its count is shown and the missing prefilter observation is noted in the CSV.", "",
        "`data/discovery_context.csv` retains detailed search/deduplication information, source-route counts, auxiliary surveys and evidence-cache counts. These details do not appear in a separate figure panel. Neither collection ranking nor classifier behavior is changed by this reporting layout.", "",
    ]
    if source == "openalex":
        lines.extend([
            "## Discovery counts and limits", "",
            "Direct-search, survey-search, and backward-reference hit counts can overlap across queries and discovery routes. Their sum is not a count of unique papers or model inputs, and no removal percentage is drawn from these hits to selected candidates.", "",
            f"- Direct-search hits: `{metadata.get('direct_results_scanned', 'not observed')}`",
            f"- Survey-search hits: `{metadata.get('survey_results_scanned', 'not observed')}`",
            f"- Backward-reference hits: `{metadata.get('reference_results_scanned', 'not observed')}`",
            f"- Active collection limit recorded by the snapshot: `{metadata.get('collection_limit', 0)}`", "",
            "Discovery caps derive from the largest numeric `collection_limit` or `classification_limit`, multiplied by `query_limit_factor` (default 3). OpenAlex survey queries and reference traversal can override this using `survey_query_limit_factor` and `reference_limit_factor`. Without either numeric limit, query retrieval has no application cap. Pagination does not impose a separate absolute cap. Auxiliary surveys are traversal sources and do not consume either candidate limit.", "",
            "Auxiliary-only surveys support reference discovery and are not counted as papers removed from classification. Their counts and the old combined retained-candidates-and-surveys count remain in `data/discovery_context.csv` for auditing. The full unique discovery pool was not recorded in this snapshot, so no deduplication loss is inferred from crawler hits.", "",
            "Observed figures use stored cache coverage and compatible CSVs. Changing today's configuration or classification limit does not retroactively change the recorded discovery counts.", "",
        ])
    lines.extend(["## Figures", ""])
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


@audited_bundle
def build(settings: Settings, source: str, output: Path | None = None) -> list[Path]:
    """Build database-aware funnel figures for one discovery source without network access."""
    settings.require_source(source)
    if source not in {"github", "openalex"}:
        raise ValueError(f"unsupported funnel source: {source}")
    configure_style(settings.section("plots"))
    paths = _source_paths(settings, source)
    out_dir = output or paths['output']
    if paths['cache'].is_file():
        paths['cache'] = freeze_database(paths['cache'], out_dir)
    for path in paths["interim"].rglob("*.csv"):
        if 'pilots' not in path.parts:
            register_input(path)
    for path in paths["processed"].rglob("*.csv"):
        register_input(path)
    profile = load_term_profile(settings, source)
    if source == "github":
        shared, metadata = _github_shared_stages(settings, profile, paths)
    else:
        shared, metadata = _openalex_shared_stages(settings, profile, paths)
    if not shared:
        raise ValueError(f"no configured or persisted {source} funnel stages are available")
    if not any(stage.key == "active_collection" for stage in shared):
        initial = next((stage for stage in shared if stage.key == "initial_results"), None)
        capacity = capacity_min(initial.count if initial else None, optional_limit(metadata.get("collection_limit"), "collection_limit"))
        shared.append(FunnelStage(
            "active_collection", "Collection capacity", capacity, CONFIGURED,
            "No selected-candidate count recorded; configured capacity only",
        ))
    context = _discovery_context(source, shared, metadata)
    metadata["discovery_context"] = context.to_dict("records")
    discovered_variants = _classifier_variants(settings, source, paths)
    formats = [str(value) for value in settings.section("plots").get("formats", ["svg", "pdf"])]
    dpi = _int(settings.section("plots").get("dpi"), 160)
    _clear_previous_bundle(out_dir)
    outputs: list[Path] = [write_table(context, out_dir / "data", "discovery_context")]
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
    parser = argparse.ArgumentParser(description="Build database-aware RADAR pipeline funnels")
    parser.add_argument("source", choices=("github", "openalex"))
    parser.add_argument("--config", type=Path, default=Path("config/studies/ids_repositories/study.toml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    configure_logging(str(settings.section("logging").get("level", "INFO")))
    build(settings, args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
