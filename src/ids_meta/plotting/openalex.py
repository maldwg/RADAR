from __future__ import annotations

import math
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd

from ids_meta.common.logging import logger
from ids_meta.plotting.common import add_heading, configure_style, load_frame, metadata_caption, save_figure, write_index, write_table

LOG = logger("plotting.openalex")

TYPE_COLORS = {
    "dataset": "#2F6F9F",
    "generator_framework": "#D1842F",
    "dataset_and_generator": "#7B5EA7",
    "unknown": "#6F7782",
}
def artifact_label(value: object) -> str:
    """Format a normalized artifact type for display."""
    return str(value or "unknown").replace("_", " ")


def discovery_provenance(value: object) -> str:
    """Collapse detailed discovery relations into a readable source category."""
    relations = {item.strip() for item in str(value or "").split(";") if item.strip()}
    direct = "direct_search" in relations
    snowball = any(item == "survey_reference" or item.startswith("survey_reference:") for item in relations)
    if direct and snowball:
        return "direct + snowball"
    if snowball:
        return "survey snowball"
    if direct:
        return "direct search"
    if "survey_search" in relations:
        return "survey source"
    return "other"


def concise_text(value: object, limit: int = 64) -> str:
    """Collapse whitespace and shorten long display text without changing stored CSV values."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def resource_display_name(row: pd.Series) -> str:
    """Choose the LLM display name, a compact alias, or a conservative legacy fallback."""
    display = " ".join(str(row.get("resource_name", "") or row.get("artifact_display_name", "") or "").split())
    if display:
        return concise_text(display, 54)
    name = " ".join(str(row.get("artifact_name", "") or "").split())
    aliases = [" ".join(value.split()) for value in str(row.get("artifact_aliases", "") or "").split(";") if value.strip()]
    compact_names = [value for value in [name, *aliases] if value and len(value) <= 54]
    if compact_names:
        return min(compact_names, key=lambda value: (len(value), value.casefold()))
    parenthetical = re.findall(r"\(([^()]{2,20})\)", name)
    if parenthetical:
        return parenthetical[-1]
    if name and len(name) <= 54:
        return name
    artifact_type = artifact_label(row.get("artifact_type", "unknown"))
    year = str(row.get("publication_year", "") or "unknown year")
    work_id = str(row.get("openalex_id", "")).rsplit("/", 1)[-1]
    return f"Unnamed {artifact_type} · {year} · {work_id}"


def add_unique_resource_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Add concise resource labels and disambiguate any duplicates with stable work IDs."""
    view = frame.copy()
    view["resource_name"] = view.apply(resource_display_name, axis=1)
    duplicated = view["resource_name"].duplicated(keep=False)
    if duplicated.any():
        suffixes = view.loc[duplicated, "openalex_id"].astype(str).str.rsplit("/", n=1).str[-1]
        view.loc[duplicated, "resource_name"] = view.loc[duplicated, "resource_name"] + " · " + suffixes
    return view


def log_count_coordinate(value: object) -> float:
    """Transform a non-negative count to log10(count + 1) for zero-safe plotting."""
    try:
        return math.log10(max(0.0, float(value)) + 1.0)
    except (TypeError, ValueError):
        return 0.0


def set_count_log_ticks(ax: plt.Axes, maximum: float, axis: str = "x") -> None:
    """Label a log10(count + 1) axis with the corresponding untransformed counts."""
    candidates = [0, 1, 10, 100, 1_000, 10_000, 100_000]
    values = [value for value in candidates if value <= max(1.0, maximum * 1.1)]
    if maximum > values[-1] and log_count_coordinate(maximum) - log_count_coordinate(values[-1]) >= .35:
        values.append(int(maximum))
    positions = [log_count_coordinate(value) for value in values]
    labels = [f"{value:,}" for value in values]
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    setter(positions, labels)


def clean_bundle(out_dir: Path, formats: list[str]) -> None:
    """Remove stale OpenAlex figures and aggregate tables before replacement."""
    for extension in formats:
        for path in out_dir.glob(f"*.{extension}"):
            path.unlink()
    data_dir = out_dir / "data"
    if data_dir.is_dir():
        for path in data_dir.glob("*.csv"):
            path.unlink()
    index = out_dir / "README.md"
    if index.is_file():
        index.unlink()


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add numeric citation/year fields and normalized visual categories."""
    view = frame.copy()
    for column in ("resource_name", "artifact_name", "artifact_display_name", "artifact_aliases", "title", "discovery_relations", "prefilter_score", "cited_by_count"):
        if column not in view:
            view[column] = ""
    view["artifact_name"] = view["artifact_name"].where(view["artifact_name"] != "", view["title"])
    view["artifact_type"] = view["artifact_type"].replace("", "unknown")
    view["publication_year_numeric"] = pd.to_numeric(view["publication_year"], errors="coerce")
    view["citations"] = pd.to_numeric(view["cited_by_count"], errors="coerce").fillna(0).clip(lower=0)
    view["provenance"] = view["discovery_relations"].map(discovery_provenance)
    view["resource_name"] = view.apply(resource_display_name, axis=1)
    return view


def publication_timeline(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot one non-overlapping row per canonical resource on an exact-year timeline."""
    dated = frame.dropna(subset=["publication_year_numeric"]).copy()
    dated = dated.sort_values(["publication_year_numeric", "resource_name", "openalex_id"], kind="stable")
    dated["publication_year"] = dated["publication_year_numeric"].astype(int)
    columns = [
        "resource_name", "artifact_display_name", "artifact_name", "publication_year", "publication_date", "artifact_type", "paper_role", "catalogue_tier",
        "citations", "citation_as_of", "provenance", "discovery_relations", "prefilter_score", "title", "doi", "openalex_id",
    ]
    for column in columns:
        if column not in dated:
            dated[column] = ""
    write_table(dated[columns], out_dir / "data", "00_publication_timeline")
    height = max(6.0, 2.0 + .34 * max(1, len(dated)))
    fig, ax = plt.subplots(figsize=(13.5, height))
    caption = metadata_caption(frame, "one row per canonical resource; color identifies resource type")
    add_heading(fig, "IDS dataset and generator publication timeline", caption, top=.89)
    fig.subplots_adjust(left=.29, right=.91)
    if dated.empty:
        ax.axis("off")
        ax.text(.5, .5, "No dated canonical artifacts in the current catalogue", transform=ax.transAxes, ha="center", va="center")
    else:
        minimum_year = int(dated["publication_year"].min())
        maximum_year = int(dated["publication_year"].max())
        positions = list(range(len(dated)))
        ax.hlines(positions, minimum_year - .45, dated["publication_year"], color="#DCE3E8", linewidth=1.0, zorder=1)
        for artifact_type, group in dated.assign(_position=positions).groupby("artifact_type", sort=False):
            ax.scatter(
                group["publication_year"], group["_position"], s=52,
                color=TYPE_COLORS.get(str(artifact_type), TYPE_COLORS["unknown"]), edgecolor="white", linewidth=1.0,
                label=artifact_label(artifact_type), zorder=3,
            )
        ax.set_yticks(positions, dated["resource_name"])
        ax.invert_yaxis()
        ax.set_xlabel("Publication year")
        ax.set_ylabel("")
        ax.set_xlim(minimum_year - .6, maximum_year + 1.4)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
        ax.grid(axis="y", visible=False)
        for position, (_, row) in enumerate(dated.iterrows()):
            ax.annotate(str(int(row["publication_year"])), (row["publication_year"], position), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7, color="#56636F")
        type_handles = [Patch(facecolor=color, label=artifact_label(name)) for name, color in TYPE_COLORS.items() if name in set(dated["artifact_type"])]
        ax.legend(handles=type_handles, title="Resource type", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    return save_figure(fig, out_dir, "00_publication_timeline", formats, dpi)


def citation_landscape(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Rank canonical resources by citations using a zero-safe axis with count-labelled ticks."""
    ranked = frame.sort_values(["citations", "resource_name"], ascending=[True, True], kind="stable").copy()
    ranked["citation_coordinate"] = ranked["citations"].map(log_count_coordinate)
    columns = ["resource_name", "artifact_display_name", "artifact_name", "publication_year", "citations", "citation_as_of", "artifact_type", "provenance", "discovery_relations", "openalex_id", "doi"]
    write_table(ranked[columns], out_dir / "data", "01_citation_landscape")
    height = max(6.0, 2.0 + .31 * max(1, len(ranked)))
    fig, ax = plt.subplots(figsize=(12.8, height))
    caption = metadata_caption(frame, "citations are descriptive only; tick labels show actual citation counts")
    add_heading(fig, "Citation context for canonical resources", caption)
    fig.subplots_adjust(left=.29, right=.90)
    if ranked.empty:
        ax.axis("off")
        ax.text(.5, .5, "No canonical resources", transform=ax.transAxes, ha="center", va="center")
    else:
        positions = list(range(len(ranked)))
        ax.hlines(positions, 0, ranked["citation_coordinate"], color="#DCE3E8", linewidth=1.0, zorder=1)
        for artifact_type, group in ranked.assign(_position=positions).groupby("artifact_type", sort=False):
            ax.scatter(group["citation_coordinate"], group["_position"], s=46, color=TYPE_COLORS.get(str(artifact_type), TYPE_COLORS["unknown"]), edgecolor="white", linewidth=.9, zorder=3)
        ax.set_yticks(positions, ranked["resource_name"])
        ax.set_xlabel("Citation count (log-scaled spacing; labels are actual counts)")
        ax.set_ylabel("")
        ax.grid(axis="y", visible=False)
        maximum = float(ranked["citations"].max())
        set_count_log_ticks(ax, maximum)
        ax.set_xlim(-.05, max(1.0, log_count_coordinate(maximum)) + .42)
        for position, (_, row) in enumerate(ranked.iterrows()):
            ax.annotate(f"{int(row['citations']):,}", (row["citation_coordinate"], position), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7, color="#56636F")
        handles = [Patch(facecolor=color, label=artifact_label(name)) for name, color in TYPE_COLORS.items() if name in set(ranked["artifact_type"])]
        ax.legend(handles=handles, title="Resource type", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    return save_figure(fig, out_dir, "01_citation_landscape", formats, dpi)


def provenance_matrix(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Render artifact-level discovery paths alongside citation and score context."""
    view = frame.sort_values(["publication_year_numeric", "resource_name"], na_position="last", kind="stable").copy()
    relations = view["discovery_relations"].fillna("")
    view["direct_search"] = relations.str.split(";").map(lambda values: "direct_search" in values)
    view["survey_snowball"] = relations.map(lambda value: any(item == "survey_reference" or item.startswith("survey_reference:") for item in value.split(";") if item))
    view["survey_source"] = relations.str.split(";").map(lambda values: "survey_search" in values)
    columns = ["resource_name", "artifact_display_name", "artifact_name", "publication_year", "artifact_type", "citations", "citation_as_of", "prefilter_score", "direct_search", "survey_snowball", "survey_source", "provenance", "discovery_relations", "openalex_id"]
    write_table(view[columns], out_dir / "data", "02_discovery_provenance")
    height = max(5.8, 2.0 + .32 * max(1, len(view)))
    fig, ax = plt.subplots(figsize=(12.8, height))
    caption = metadata_caption(frame, "filled cells mark observed discovery paths; numeric columns show exact values")
    add_heading(fig, "Where each accepted base paper came from", caption, top=.89)
    fig.subplots_adjust(left=.29, right=.97)
    ax.set_xlim(-.5, 5.5)
    positions = list(range(len(view)))
    ax.set_ylim(len(view) - .4, -.7)
    ax.set_yticks(positions, view["resource_name"])
    ax.set_xticks([0, 1, 2, 3.45, 4.85], ["Direct", "Survey\nsnowball", "Survey\nsource", "Citations", "Prefilter\nscore"])
    ax.grid(axis="y", visible=False)
    for position, (_, row) in enumerate(view.iterrows()):
        color = TYPE_COLORS.get(str(row["artifact_type"]), TYPE_COLORS["unknown"])
        for x, field in ((0, "direct_search"), (1, "survey_snowball"), (2, "survey_source")):
            ax.scatter([x], [position], s=42, facecolor=color if bool(row[field]) else "white", edgecolor=color if bool(row[field]) else "#CDD5DB", linewidth=.9)
        ax.text(3.45, position, f"{int(row['citations']):,}", ha="center", va="center", fontsize=7.2, color="#43505C")
        ax.text(4.85, position, str(row.get("prefilter_score", "")) or "—", ha="center", va="center", fontsize=7.2, color="#43505C")
    type_handles = [Patch(facecolor=color, label=artifact_label(name)) for name, color in TYPE_COLORS.items() if name in set(view["artifact_type"])]
    ax.legend(handles=type_handles, title="Resource type", loc="lower right", bbox_to_anchor=(1, 1.005), ncol=max(1, len(type_handles)), frameon=False, fontsize=8)
    return save_figure(fig, out_dir, "02_discovery_provenance", formats, dpi)


def cumulative_publications(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot cumulative artifact-family growth over time by artifact type."""
    dated = frame.dropna(subset=["publication_year_numeric"]).copy()
    dated["publication_year"] = dated["publication_year_numeric"].astype(int)
    yearly = dated.groupby(["publication_year", "artifact_type"]).size().rename("new_publications").reset_index()
    if not yearly.empty:
        years = range(int(yearly["publication_year"].min()), int(yearly["publication_year"].max()) + 1)
        grid = pd.MultiIndex.from_product([years, sorted(yearly["artifact_type"].unique())], names=["publication_year", "artifact_type"]).to_frame(index=False)
        yearly = grid.merge(yearly, how="left").fillna({"new_publications": 0})
        yearly["new_publications"] = yearly["new_publications"].astype(int)
        yearly["cumulative_publications"] = yearly.groupby("artifact_type")["new_publications"].cumsum()
    else:
        yearly["cumulative_publications"] = pd.Series(dtype=int)
    write_table(yearly, out_dir / "data", "03_cumulative_publications")
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    caption = metadata_caption(frame, "canonical artifact families; versions and alternate manifestations excluded")
    add_heading(fig, "Growth of the IDS data-resource landscape", caption, top=.84)
    for artifact_type, group in yearly.groupby("artifact_type", sort=False):
        ax.step(group["publication_year"], group["cumulative_publications"], where="post", linewidth=2.3, color=TYPE_COLORS.get(str(artifact_type), TYPE_COLORS["unknown"]), label=artifact_label(artifact_type))
        ax.scatter(group["publication_year"], group["cumulative_publications"], s=18, color=TYPE_COLORS.get(str(artifact_type), TYPE_COLORS["unknown"]))
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Cumulative canonical artifacts")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if not yearly.empty:
        ax.legend(title="Artifact type", frameon=False)
    return save_figure(fig, out_dir, "03_cumulative_publications", formats, dpi)


def relationship_heatmap(frame: pd.DataFrame, row_field: str, column_field: str, title: str, stem: str, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot an annotated count heatmap between two categorical dimensions."""
    matrix = pd.crosstab(frame[row_field].replace("", "unknown"), frame[column_field].replace("", "unknown"))
    table = matrix.stack().rename("count").reset_index()
    write_table(table, out_dir / "data", stem)
    if matrix.empty:
        matrix = pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 2.4 + .55 * max(1, len(matrix.index)))))
    caption = metadata_caption(frame, "canonical resource families; each cell is an exact resource count")
    add_heading(fig, title, caption, top=.84)
    image = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0)
    for row_index, row_name in enumerate(matrix.index):
        for column_index, column_name in enumerate(matrix.columns):
            count = int(matrix.loc[row_name, column_name])
            maximum = max(1, int(matrix.to_numpy().max()))
            color = "white" if count > maximum * .52 else "#263238"
            ax.text(column_index, row_index, str(count), ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(matrix.columns)), [artifact_label(value) for value in matrix.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(matrix.index)), [artifact_label(value) for value in matrix.index])
    ax.set_xlabel(artifact_label(column_field))
    ax.set_ylabel(artifact_label(row_field))
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, pad=.02)
    colorbar.set_label("Canonical resources")
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return save_figure(fig, out_dir, stem, formats, dpi)


def metadata_lollipop(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Visualize catalogue metadata completeness as a lollipop chart."""
    fields = ["doi", "authors", "venue", "publication_year", "primary_url", "pdf_url", "citation_as_of", "discovery_relations", "domain", "data_form", "capture_environment", "labeling"]
    completeness = pd.DataFrame({
        "field": fields,
        "present": [((frame[field] != "") & (frame[field].str.casefold() != "unknown")).sum() if field in frame else 0 for field in fields],
    })
    completeness["percent"] = completeness["present"] / max(1, len(frame)) * 100
    completeness = completeness.sort_values("percent")
    write_table(completeness, out_dir / "data", "06_metadata_completeness")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    caption = metadata_caption(frame, "canonical base-paper metadata")
    add_heading(fig, "Metadata completeness", caption, top=.84)
    positions = range(len(completeness))
    ax.hlines(positions, 0, completeness["percent"], color="#CDD7DE", linewidth=2)
    ax.scatter(completeness["percent"], list(positions), s=58, color="#2A9D8F", edgecolor="white", zorder=3)
    ax.set_yticks(list(positions), [artifact_label(value) for value in completeness["field"]])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Rows populated (%)")
    ax.grid(axis="y", visible=False)
    for position, percent in zip(positions, completeness["percent"]):
        ax.text(percent + 1.5, position, f"{percent:.0f}%", va="center", fontsize=7.5, color="#43505C")
    return save_figure(fig, out_dir, "06_metadata_completeness", formats, dpi)


def classification_audit(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Show how deterministic prefilter bands map to final LLM decisions using exact counts."""
    view = frame.copy()
    bands = [value for value in ("candidate", "borderline", "reject") if value in set(view["prefilter_band"])]
    decisions = [value for value in ("accept", "accept_adjacent", "exclude", "uncertain", "error", "dry_run") if value in set(view["decision"])]
    matrix = pd.crosstab(view["prefilter_band"], view["decision"]).reindex(index=bands, columns=decisions, fill_value=0)
    totals = matrix.sum(axis=1).replace(0, 1)
    percentages = matrix.div(totals, axis=0) * 100
    table = matrix.stack().rename("count").reset_index()
    percentage_table = percentages.stack().rename("percent_within_prefilter_band").reset_index()
    table = table.merge(percentage_table, on=["prefilter_band", "decision"])
    write_table(table, out_dir / "data", "07_classification_audit")
    if matrix.empty:
        matrix = pd.DataFrame([[0]], index=["no data"], columns=["no data"])
        percentages = matrix.astype(float)
    fig, ax = plt.subplots(figsize=(10.8, max(5.5, 3.2 + .7 * len(matrix.index))))
    caption = metadata_caption(view, "rows are deterministic prefilter bands; cells show count and row percentage")
    add_heading(fig, "From keyword prefilter to LLM decision", caption)
    image = ax.imshow(percentages.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            count = int(matrix.iloc[row_index, column_index])
            percent = float(percentages.iloc[row_index, column_index])
            color = "white" if percent >= 55 else "#263238"
            ax.text(column_index, row_index, f"{count}\n{percent:.0f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(matrix.columns)), [artifact_label(value) for value in matrix.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), [artifact_label(value) for value in matrix.index])
    ax.set_xlabel("LLM decision")
    ax.set_ylabel("Prefilter band")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, pad=.02)
    colorbar.set_label("Share within prefilter band (%)")
    return save_figure(fig, out_dir, "07_classification_audit", formats, dpi)


def survey_source_landscape(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Compare selected survey sources in aligned score, citation, and reference-count panels."""
    relations = frame.get("discovery_relations", pd.Series("", index=frame.index)).fillna("")
    surveys = frame[relations.str.split(";").map(lambda values: "survey_search" in values)].copy()
    year_values = surveys.get("publication_year", pd.Series("", index=surveys.index))
    citation_values = surveys.get("cited_by_count", pd.Series("", index=surveys.index))
    score_values = surveys.get("survey_discovery_score", pd.Series("", index=surveys.index))
    reference_values = surveys.get("referenced_works_count", pd.Series("", index=surveys.index))
    surveys["publication_year_numeric"] = pd.to_numeric(year_values, errors="coerce")
    surveys["citations"] = pd.to_numeric(citation_values, errors="coerce").fillna(0).clip(lower=0)
    surveys["survey_score"] = pd.to_numeric(score_values, errors="coerce")
    surveys["reference_count"] = pd.to_numeric(reference_values, errors="coerce").fillna(0).clip(lower=0)
    columns = ["title", "openalex_id", "publication_year", "survey_discovery_score", "citations", "citation_as_of", "reference_count", "matched_queries", "discovery_relations"]
    for column in columns:
        if column not in surveys:
            surveys[column] = ""
    surveys = surveys.sort_values(["survey_score", "citations", "title"], ascending=[True, True, True], na_position="first", kind="stable")
    surveys["survey_title"] = surveys["title"].map(lambda value: concise_text(value, 68))
    surveys["citation_coordinate"] = surveys["citations"].map(log_count_coordinate)
    columns = ["survey_title", *columns]
    write_table(surveys[columns], out_dir / "data", "08_survey_sources")
    height = max(6.0, 2.1 + .34 * max(1, len(surveys)))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, height), sharey=True, gridspec_kw={"width_ratios": [1.2, 1.2, 1.2], "wspace": .13})
    caption = metadata_caption(surveys, "one row per selected survey; all three panels show exact source metadata")
    add_heading(fig, "Survey sources used for reference snowballing", caption, top=.84)
    fig.subplots_adjust(left=.35, right=.97)
    if surveys.empty:
        for ax in axes:
            ax.axis("off")
        axes[1].text(.5, .5, "No selected survey sources in this candidate audit", transform=axes[1].transAxes, ha="center", va="center")
    else:
        positions = list(range(len(surveys)))
        values = [
            (surveys["survey_score"].fillna(0), "Survey retrieval score", False, float(surveys["survey_score"].fillna(0).max())),
            (surveys["citation_coordinate"], "Citation count", True, float(surveys["citations"].max())),
            (surveys["reference_count"], "Referenced works", False, float(surveys["reference_count"].max())),
        ]
        for ax, (series, label, logarithmic, maximum) in zip(axes, values):
            ax.hlines(positions, 0, series, color="#DCE3E8", linewidth=1.0, zorder=1)
            ax.scatter(series, positions, s=42, color="#2F6F9F", edgecolor="white", linewidth=.8, zorder=3)
            ax.set_xlabel(label + (" (log-scaled spacing)" if logarithmic else ""))
            ax.grid(axis="y", visible=False)
            if logarithmic:
                set_count_log_ticks(ax, maximum)
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            raw_series = surveys["citations"] if logarithmic else series
            for position, raw_value in zip(positions, raw_series):
                ax.annotate(f"{float(raw_value):,.0f}", (series.iloc[position], position), xytext=(5, 0), textcoords="offset points", va="center", fontsize=6.5, color="#56636F")
        axes[0].set_yticks(positions, surveys["survey_title"])
        axes[0].set_ylabel("")
    return save_figure(fig, out_dir, "08_survey_sources", formats, dpi)


def build(
    main_path: Path, adjacent_path: Path | None, out_dir: Path, config: dict[str, object],
    classifications_path: Path | None = None, survey_sources_path: Path | None = None,
) -> list[Path]:
    """Generate the OpenAlex timeline, citation, provenance, and relationship bundle."""
    LOG.info("building OpenAlex plots core_input=%s adjacent_input=%s output=%s", main_path, adjacent_path or "none", out_dir)
    configure_style()
    required = ("openalex_id", "artifact_id", "publication_family_id", "paper_relationship", "artifact_type", "publication_year", "domain", "data_form", "capture_environment", "labeling")
    main = load_frame(main_path, required)
    main["catalogue_tier"] = "core"
    frames = [main]
    if adjacent_path and adjacent_path.is_file():
        adjacent = load_frame(adjacent_path, required)
        adjacent["catalogue_tier"] = "adjacent"
        frames.append(adjacent)
    frame = prepare_frame(pd.concat(frames, ignore_index=True))
    canonical = add_unique_resource_names(frame[frame["paper_relationship"] == "canonical_base"].copy())
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    clean_bundle(out_dir, formats)
    outputs: list[Path] = []
    outputs += publication_timeline(canonical, out_dir, formats, dpi)
    outputs += citation_landscape(canonical, out_dir, formats, dpi)
    outputs += provenance_matrix(canonical, out_dir, formats, dpi)
    outputs += cumulative_publications(canonical, out_dir, formats, dpi)
    outputs += relationship_heatmap(canonical, "domain", "data_form", "Domain and data-form relationships", "04_domain_data_form", out_dir, formats, dpi)
    outputs += relationship_heatmap(canonical, "capture_environment", "labeling", "Capture environment and label availability", "05_capture_environment_labeling", out_dir, formats, dpi)
    outputs += metadata_lollipop(canonical, out_dir, formats, dpi)
    if classifications_path and classifications_path.is_file():
        evaluated = load_frame(classifications_path, ("openalex_id", "prefilter_score", "prefilter_band", "decision", "paper_role", "cited_by_count", "discovery_relations"))
        outputs += classification_audit(evaluated, out_dir, formats, dpi)
    if survey_sources_path and survey_sources_path.is_file():
        surveys = load_frame(survey_sources_path, ("openalex_id", "title", "cited_by_count", "discovery_relations"))
        outputs += survey_source_landscape(surveys, out_dir, formats, dpi)
    outputs.append(write_index(out_dir, main_path, frame, outputs, [
        "Timeline labels prefer the LLM-extracted concise resource name and use one row per resource, so labels cannot collide.",
        "Citation counts are descriptive and never affect retrieval, inclusion, or canonical-base selection.",
        "Discovery provenance distinguishes direct searches, survey-reference snowballing, combined discovery, and survey-source records.",
        "All artifact-level figures use canonical base papers; versions and alternate manifestations remain available in the processed CSV.",
        "Every figure has an exact aggregate CSV under data/.",
        "When classifications.csv is available, the classification audit reports exact counts and within-band percentages.",
        "Survey-source scores, citations, and reference-list sizes use aligned panels rather than overloaded marker encodings.",
    ]))
    LOG.info("OpenAlex plot bundle complete canonical_artifacts=%d files=%d output=%s", len(canonical), len(outputs), out_dir)
    return outputs
