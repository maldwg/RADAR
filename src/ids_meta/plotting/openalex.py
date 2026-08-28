from __future__ import annotations

import math
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd

from ids_meta.common.logging import logger
from ids_meta.plotting.common import add_heading, configure_style, evidence_escalation_plot, load_frame, metadata_caption, save_figure, score_decision_plot, write_index, write_table

LOG = logger("plotting.openalex")

TYPE_COLORS = {
    "dataset": "#2F6F9F",
    "generator_framework": "#D1842F",
    "dataset_and_generator": "#7B5EA7",
    "unknown": "#6F7782",
}

DISCOVERY_COLORS = {
    "Direct search": "#2F6F9F",
    "Survey snowball": "#7B5EA7",
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
    return ""


def citing_survey_count(value: object) -> int:
    """Count distinct surveys citing a work, with a one-source fallback for legacy provenance."""
    relations = {item.strip() for item in str(value or "").split(";") if item.strip()}
    survey_ids = {item.split(":", 1)[1] for item in relations if item.startswith("survey_reference:") and item.split(":", 1)[1]}
    if survey_ids:
        return len(survey_ids)
    return int("survey_reference" in relations)


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
    """Render a true publication timeline; resources in the same year are stacked vertically."""
    dated = frame.dropna(subset=["publication_year_numeric"]).copy()
    dated = dated.sort_values(["publication_year_numeric", "resource_name", "openalex_id"], kind="stable")
    dated["publication_year"] = dated["publication_year_numeric"].astype(int)
    dated["timeline_stack"] = dated.groupby("publication_year", sort=False).cumcount()
    dated["timeline_stack_size"] = dated.groupby("publication_year", sort=False)["openalex_id"].transform("size")

    columns = [
        "resource_name", "artifact_display_name", "artifact_name", "publication_year", "publication_date",
        "artifact_type", "domain", "data_form", "capture_environment", "labeling", "paper_role",
        "catalogue_tier", "provenance", "discovery_relations", "prefilter_score", "title", "doi", "openalex_id",
        "timeline_stack", "timeline_stack_size",
    ]
    for column in columns:
        if column not in dated:
            dated[column] = ""
    # Row-level audit table: exactly the resources shown on the timeline and their metadata.
    write_table(dated[columns], out_dir / "data", "00_publication_timeline")

    max_stack = int(dated["timeline_stack"].max()) if not dated.empty else 0
    height = max(6.3, 5.2 + .72 * max_stack)

    # Give each calendar year substantially more horizontal room.  The timeline
    # remains proportional in time (missing years still occupy their real gap),
    # but long resource names no longer have to compete for the same few inches.
    if dated.empty:
        year_span = 1
    else:
        year_span = int(dated["publication_year"].max()) - int(dated["publication_year"].min()) + 1
    figure_width = max(15.2, min(36.0, 1.15 * year_span))
    fig, ax = plt.subplots(figsize=(figure_width, height))
    caption = metadata_caption(
        frame,
        "x = publication year; y has no metric meaning; resources sharing a year are stacked vertically",
    )
    add_heading(fig, "IDS dataset and generator publication timeline", caption, top=.86)
    fig.subplots_adjust(left=.07, right=.84, bottom=.16)

    if dated.empty:
        ax.axis("off")
        ax.text(.5, .5, "No dated canonical artifacts in the current catalogue", transform=ax.transAxes, ha="center", va="center")
        return save_figure(fig, out_dir, "00_publication_timeline", formats, dpi)

    # The vertical coordinate is deliberately only a layout coordinate. A small gap
    # between stacked entries leaves room for the mandatory resource labels.
    dated["timeline_y"] = dated["timeline_stack"].astype(float) * 1.35
    minimum_year = int(dated["publication_year"].min())
    maximum_year = int(dated["publication_year"].max())

    # Timeline baseline plus a light stem for every occupied year.
    ax.axhline(-.72, color="#AAB6BF", linewidth=1.2, zorder=1)
    for year, group in dated.groupby("publication_year", sort=True):
        ax.vlines(year, -.72, float(group["timeline_y"].max()) + .12, color="#DCE3E8", linewidth=1.1, zorder=1)

    for artifact_type, group in dated.groupby("artifact_type", sort=False):
        ax.scatter(
            group["publication_year"], group["timeline_y"], s=68,
            color=TYPE_COLORS.get(str(artifact_type), TYPE_COLORS["unknown"]),
            edgecolor="white", linewidth=1.0, label=artifact_label(artifact_type), zorder=4,
        )

    # Names remain directly beneath their own point. Because same-year resources
    # have different layout rows, these labels also stack instead of colliding.
    for _, row in dated.iterrows():
        ax.annotate(
            textwrap.fill(str(row["resource_name"]), width=22),
            (row["publication_year"], row["timeline_y"]),
            xytext=(0, -10), textcoords="offset points",
            ha="center", va="top", fontsize=6.7, color="#43505C", linespacing=.95,
            annotation_clip=False, zorder=3,
        )

    ax.set_xlim(minimum_year - .65, maximum_year + .65)
    ax.set_ylim(-1.0, float(dated["timeline_y"].max()) + .85)
    ax.set_xlabel("Publication year")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=14))
    ax.grid(axis="x", linewidth=.6, alpha=.45)
    ax.grid(axis="y", visible=False)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    handles = [
        Patch(facecolor=color, label=artifact_label(name))
        for name, color in TYPE_COLORS.items()
        if name in set(dated["artifact_type"])
    ]
    ax.legend(handles=handles, title="Resource type", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
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
    """Show discovery paths together with resource classification, domain, and provenance metadata."""
    view = frame.sort_values(["publication_year_numeric", "resource_name"], na_position="last", kind="stable").copy()
    relations = view["discovery_relations"].fillna("")
    view["direct_search"] = relations.str.split(";").map(lambda values: "direct_search" in values)
    view["survey_snowball"] = relations.map(
        lambda value: any(item == "survey_reference" or item.startswith("survey_reference:") for item in value.split(";") if item)
    )
    view["citing_surveys"] = relations.map(citing_survey_count)
    for column in ("domain", "artifact_type", "provenance", "prefilter_score", "paper_role", "catalogue_tier"):
        if column not in view:
            view[column] = ""

    columns = [
        "resource_name", "artifact_display_name", "artifact_name", "publication_year", "artifact_type", "domain",
        "paper_role", "catalogue_tier", "direct_search", "survey_snowball", "citing_surveys", "provenance",
        "discovery_relations", "prefilter_score", "data_form", "capture_environment", "labeling", "doi", "openalex_id",
    ]
    for column in columns:
        if column not in view:
            view[column] = ""
    # This CSV is intentionally row-level so every plotted paper can be audited.
    write_table(view[columns], out_dir / "data", "02_discovery_provenance")

    height = max(6.0, 2.1 + .36 * max(1, len(view)))
    fig, ax = plt.subplots(figsize=(16.8, height))
    caption = metadata_caption(
        frame,
        "filled cells mark discovery paths; text columns expose resource type, domain, normalized provenance, and score",
    )
    add_heading(fig, "Discovery provenance and resource classification", caption, top=.89)
    fig.subplots_adjust(left=.27, right=.985, top=.92)

    if view.empty:
        ax.axis("off")
        ax.text(.5, .5, "No discovery-provenance data", transform=ax.transAxes, ha="center", va="center")
        return save_figure(fig, out_dir, "02_discovery_provenance", formats, dpi)

    positions = list(range(len(view)))
    x_positions = {
        "direct_search": 0.0,
        "survey_snowball": 1.0,
        "citing_surveys": 2.0,
        "artifact_type": 3.65,
        "domain": 5.35,
        "provenance": 7.25,
        "prefilter_score": 8.8,
    }
    ax.set_xlim(-.55, 9.55)
    ax.set_ylim(len(view) - .4, -.7)
    ax.set_yticks(positions, view["resource_name"])
    ax.set_xticks(
        list(x_positions.values()),
        ["Direct search", "Survey\nsnowball", "Citing\nsurveys", "Resource\ntype", "Domain", "Provenance", "Prefilter\nscore"],
    )
    # Treat the x tick labels as table-style column headers and keep them at the
    # top of the provenance view.  On tall figures, bottom-only labels are easy
    # to miss and make the metadata columns appear unlabeled.
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="both", top=False, bottom=False, labeltop=True, labelbottom=False, pad=9)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("#43505C")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", linewidth=.5, alpha=.25)

    for position, (_, row) in enumerate(view.iterrows()):
        color = TYPE_COLORS.get(str(row["artifact_type"]), TYPE_COLORS["unknown"])
        for field in ("direct_search", "survey_snowball"):
            active = bool(row[field])
            ax.scatter(
                [x_positions[field]], [position], s=46,
                facecolor=color if active else "white",
                edgecolor=color if active else "#CDD5DB", linewidth=.9, zorder=3,
            )
        ax.text(x_positions["citing_surveys"], position, str(int(row["citing_surveys"])), ha="center", va="center", fontsize=7.1, color="#43505C", fontweight="bold")
        ax.text(x_positions["artifact_type"], position, concise_text(artifact_label(row["artifact_type"]), 24), ha="center", va="center", fontsize=7.1, color="#43505C")
        ax.text(x_positions["domain"], position, concise_text(row.get("domain", "") or "—", 28), ha="center", va="center", fontsize=7.1, color="#43505C")
        ax.text(x_positions["provenance"], position, concise_text(row.get("provenance", "") or "—", 28), ha="center", va="center", fontsize=7.1, color="#43505C")
        score = row.get("prefilter_score", "")
        ax.text(x_positions["prefilter_score"], position, str(score) if str(score).strip() else "—", ha="center", va="center", fontsize=7.1, color="#43505C")

    type_handles = [
        Patch(facecolor=color, label=artifact_label(name))
        for name, color in TYPE_COLORS.items()
        if name in set(view["artifact_type"])
    ]
    if type_handles:
        # Keep the legend away from the table-style column headers.
        ax.legend(
            handles=type_handles, title="Resource type", loc="upper center",
            bbox_to_anchor=(.5, -.035), ncol=max(1, len(type_handles)),
            frameon=False, fontsize=8,
        )
    return save_figure(fig, out_dir, "02_discovery_provenance", formats, dpi)

def type_domain_overview(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Show resource type (dataset/generator) against application domain as an annotated matrix."""
    view = frame.copy()
    for column in ("artifact_type", "domain"):
        if column not in view:
            view[column] = "unknown"
        view[column] = view[column].fillna("").astype(str)

    expanded = view[["artifact_type", "domain", "openalex_id"]].copy()
    expanded["artifact_type"] = expanded["artifact_type"].replace("", "unknown")
    expanded["domain"] = expanded["domain"].map(
        lambda value: [item.strip() for item in value.split(";") if item.strip()] or ["unknown"]
    )
    expanded = expanded.explode("domain")
    matrix = pd.crosstab(expanded["artifact_type"], expanded["domain"])

    preferred_rows = ["dataset", "generator_framework", "dataset_and_generator", "unknown"]
    row_order = [value for value in preferred_rows if value in matrix.index]
    row_order += [value for value in matrix.index if value not in row_order]
    if row_order:
        matrix = matrix.reindex(row_order)
    if not matrix.empty:
        matrix = matrix.loc[:, matrix.sum(axis=0).sort_values(ascending=False).index]

    table = matrix.stack().rename("count").reset_index() if not matrix.empty else pd.DataFrame(columns=["artifact_type", "domain", "count"])
    if not table.empty:
        row_totals = table.groupby("artifact_type")["count"].transform("sum").replace(0, 1)
        table["percent_within_type"] = table["count"] / row_totals * 100
    else:
        table["percent_within_type"] = pd.Series(dtype=float)
    write_table(table, out_dir / "data", "09_type_domain_overview")
    detail_columns = ["resource_name", "artifact_name", "publication_year", "artifact_type", "domain", "data_form", "capture_environment", "labeling", "provenance", "discovery_relations", "openalex_id", "doi"]
    detail = frame.copy()
    for column in detail_columns:
        if column not in detail:
            detail[column] = ""
    write_table(detail[detail_columns], out_dir / "data", "09_type_domain_overview_records")

    display_matrix = matrix if not matrix.empty else pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    row_percentages = display_matrix.div(display_matrix.sum(axis=1).replace(0, 1), axis=0) * 100
    fig_width = max(10.5, 6.5 + .75 * len(display_matrix.columns))
    fig_height = max(5.0, 3.0 + .9 * len(display_matrix.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    caption = metadata_caption(frame, "canonical resource families; cells show exact count and share within resource type")
    add_heading(fig, "Resource type by application domain", caption, top=.84)
    image = ax.imshow(display_matrix.values, cmap="Blues", aspect="auto", vmin=0)
    maximum = max(1, int(display_matrix.to_numpy().max()))
    for row_index in range(len(display_matrix.index)):
        for column_index in range(len(display_matrix.columns)):
            count = int(display_matrix.iloc[row_index, column_index])
            percent = float(row_percentages.iloc[row_index, column_index])
            color = "white" if count > maximum * .52 else "#263238"
            ax.text(column_index, row_index, f"{count}\n{percent:.0f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(display_matrix.columns)), [artifact_label(value) for value in display_matrix.columns], rotation=28, ha="right")
    ax.set_yticks(range(len(display_matrix.index)), [artifact_label(value) for value in display_matrix.index])
    ax.set_xlabel("Application domain")
    ax.set_ylabel("Resource type")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, pad=.02)
    colorbar.set_label("Canonical resources")
    colorbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    return save_figure(fig, out_dir, "09_type_domain_overview", formats, dpi)


def discovery_source_distribution(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Show the sheer number of works contributed by each discovery source as bars."""
    sources = ["Direct search", "Survey snowball"]
    records: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        relations = {item.strip() for item in str(row.get("discovery_relations", "") or "").split(";") if item.strip()}
        direct = "direct_search" in relations
        snowball = any(item == "survey_reference" or item.startswith("survey_reference:") for item in relations)
        memberships: list[str] = []
        if direct:
            memberships.append("Direct search")
        if snowball:
            memberships.append("Survey snowball")

        for source in memberships:
            records.append({
                "source": source,
                "resource_name": row.get("resource_name", ""),
                "artifact_name": row.get("artifact_name", ""),
                "publication_year": row.get("publication_year", ""),
                "artifact_type": row.get("artifact_type", ""),
                "domain": row.get("domain", ""),
                "provenance": row.get("provenance", ""),
                "discovery_relations": row.get("discovery_relations", ""),
                "prefilter_score": row.get("prefilter_score", ""),
                "openalex_id": row.get("openalex_id", ""),
                "doi": row.get("doi", ""),
            })

    contributions = pd.DataFrame(records)
    if contributions.empty:
        contributions = pd.DataFrame(columns=[
            "source", "resource_name", "artifact_name", "publication_year", "artifact_type", "domain", "provenance",
            "discovery_relations", "prefilter_score", "openalex_id", "doi",
        ])
    source_counts = contributions.groupby("source").size().reindex(sources, fill_value=0).rename("works_found").reset_index()
    source_counts["percent_of_canonical_works"] = source_counts["works_found"] / max(1, len(frame)) * 100
    source_counts["share_of_source_assignments_percent"] = source_counts["works_found"] / max(1, int(source_counts["works_found"].sum())) * 100

    # Aggregate CSV corresponds exactly to the bars; the records CSV makes every bar auditable.
    write_table(source_counts, out_dir / "data", "10_discovery_source_contribution")
    write_table(contributions, out_dir / "data", "10_discovery_source_contribution_records")

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    caption = metadata_caption(
        frame,
        "bar height = works found; in-bar percentage = share of all canonical works; multi-path papers can contribute to more than one bar",
    )
    add_heading(fig, "Discovery-source contribution", caption, top=.84)

    positions = list(range(len(source_counts)))
    bar_colors = [DISCOVERY_COLORS.get(source, TYPE_COLORS["unknown"]) for source in source_counts["source"]]
    bars = ax.bar(positions, source_counts["works_found"], width=.62, color=bar_colors, edgecolor="none")
    ax.set_xticks(positions, source_counts["source"])
    ax.set_ylabel("Works found")
    ax.set_xlabel("Discovery source")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", visible=False)

    maximum = max(1, int(source_counts["works_found"].max()))
    ax.set_ylim(0, maximum * 1.18 + .5)
    for bar, (_, source_row) in zip(bars, source_counts.iterrows()):
        count = int(source_row["works_found"])
        share = float(source_row["percent_of_canonical_works"])
        x_center = bar.get_x() + bar.get_width() / 2

        # Keep the exact count above the bar and place the relative contribution
        # inside it.  The percentage denominator is the number of unique canonical
        # works; therefore multi-path discovery can make percentages sum above 100%.
        if count > 0:
            ax.text(
                x_center, bar.get_height() + maximum * .025, f"n={count}",
                ha="center", va="bottom", fontsize=8, color="#43505C",
            )
            ax.text(
                x_center, bar.get_height() * .50, f"{share:.1f}%",
                ha="center", va="center", fontsize=9, fontweight="bold", color="white",
            )
        else:
            # A zero-height bar has no interior, so combine count and percentage
            # in one baseline annotation instead of overlapping two labels.
            ax.text(
                x_center, .06 * maximum, "n=0 · 0.0%",
                ha="center", va="bottom", fontsize=8, color="#56636F",
            )

    return save_figure(fig, out_dir, "10_discovery_source_contribution", formats, dpi)

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
    detail_columns = ["resource_name", "artifact_name", "publication_year", "artifact_type", "domain", "provenance", "discovery_relations", "openalex_id", "doi"]
    detail = dated.copy()
    for column in detail_columns:
        if column not in detail:
            detail[column] = ""
    write_table(detail[detail_columns], out_dir / "data", "03_cumulative_publications_records")
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
    detail_columns = ["resource_name", "artifact_name", "publication_year", "artifact_type", row_field, column_field, "domain", "provenance", "discovery_relations", "openalex_id", "doi"]
    detail_columns = list(dict.fromkeys(detail_columns))
    detail = frame.copy()
    for column in detail_columns:
        if column not in detail:
            detail[column] = ""
    write_table(detail[detail_columns], out_dir / "data", f"{stem}_records")
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
    """Visualize the percentage and exact count of papers with each metadata field available."""
    fields = [
        ("doi", "DOI"),
        ("authors", "Authors"),
        ("venue", "Venue"),
        ("publication_year", "Publication year"),
        ("primary_url", "Primary URL"),
        ("pdf_url", "PDF URL"),
        ("citation_as_of", "Citation timestamp"),
        ("discovery_relations", "Discovery provenance"),
        ("domain", "Application domain"),
        ("data_form", "Data form"),
        ("capture_environment", "Capture environment"),
        ("labeling", "Labeling"),
    ]
    missing_tokens = {"", "unknown", "none", "nan", "n/a", "na", "null", "[]", "{}"}
    rows: list[dict[str, object]] = []
    total = len(frame)
    for field, label in fields:
        if field not in frame:
            present = 0
        else:
            present = int(frame[field].map(lambda value: str(value).strip().casefold() not in missing_tokens).sum())
        rows.append({"field": field, "label": label, "present": present, "total": total, "percent": present / max(1, total) * 100})
    completeness = pd.DataFrame(rows).sort_values(["percent", "label"], ascending=[True, True], kind="stable")
    write_table(completeness, out_dir / "data", "06_metadata_completeness")
    metadata_records = frame.copy()
    audit_columns = ["resource_name", "artifact_name", "publication_year", "artifact_type", "domain", "openalex_id"]
    for field, _ in fields:
        if field not in metadata_records:
            metadata_records[field] = ""
        metadata_records[f"{field}_available"] = metadata_records[field].map(lambda value: str(value).strip().casefold() not in missing_tokens)
        audit_columns.extend([field, f"{field}_available"])
    audit_columns = list(dict.fromkeys(audit_columns))
    for column in audit_columns:
        if column not in metadata_records:
            metadata_records[column] = ""
    write_table(metadata_records[audit_columns], out_dir / "data", "06_metadata_completeness_records")

    fig, ax = plt.subplots(figsize=(11.2, 7.1))
    caption = metadata_caption(frame, "canonical base-paper metadata; availability means a non-empty, non-placeholder value")
    add_heading(fig, "Metadata availability across papers", caption, top=.84)
    positions = list(range(len(completeness)))
    ax.barh(positions, [100] * len(completeness), height=.58, color="#EEF2F4", edgecolor="none", zorder=1)
    ax.barh(positions, completeness["percent"], height=.58, color="#2F6F9F", edgecolor="none", zorder=2)
    ax.set_yticks(positions, completeness["label"])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Papers with metadata available (%)")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    for position, (_, row) in zip(positions, completeness.iterrows()):
        ax.text(
            min(109, float(row["percent"]) + 1.4), position,
            f"{float(row['percent']):.0f}%  ({int(row['present'])}/{int(row['total'])})",
            va="center", fontsize=7.6, color="#43505C",
        )
    return save_figure(fig, out_dir, "06_metadata_completeness", formats, dpi)

def classification_audit(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Map candidate/borderline/reject score bands to final acceptance/exclusion decisions."""
    view = frame.copy()
    for column in ("prefilter_band", "decision", "prefilter_score"):
        if column not in view:
            view[column] = ""
    view["prefilter_band"] = view["prefilter_band"].fillna("").astype(str).replace("", "unknown")
    view["decision"] = view["decision"].fillna("").astype(str).replace("", "unknown")
    view["prefilter_score_numeric"] = pd.to_numeric(view["prefilter_score"], errors="coerce")

    preferred_bands = ["candidate", "borderline", "reject", "unknown"]
    preferred_decisions = ["accept", "accept_adjacent", "exclude", "uncertain", "error", "dry_run", "unknown"]
    observed_bands = list(dict.fromkeys(view["prefilter_band"].tolist()))
    observed_decisions = list(dict.fromkeys(view["decision"].tolist()))
    bands = [value for value in preferred_bands if value in observed_bands] + [value for value in observed_bands if value not in preferred_bands]
    decisions = [value for value in preferred_decisions if value in observed_decisions] + [value for value in observed_decisions if value not in preferred_decisions]

    matrix = pd.crosstab(view["prefilter_band"], view["decision"]).reindex(index=bands, columns=decisions, fill_value=0)
    totals = matrix.sum(axis=1).replace(0, 1)
    percentages = matrix.div(totals, axis=0) * 100
    score_summary = view.groupby("prefilter_band", dropna=False)["prefilter_score_numeric"].agg(["min", "median", "max"]).reindex(bands)

    table = matrix.stack().rename("count").reset_index()
    percentage_table = percentages.stack().rename("percent_within_prefilter_band").reset_index()
    table = table.merge(percentage_table, on=["prefilter_band", "decision"])
    if not score_summary.empty:
        score_table = score_summary.reset_index().rename(columns={"min": "score_min", "median": "score_median", "max": "score_max"})
        table = table.merge(score_table, on="prefilter_band", how="left")
    write_table(table, out_dir / "data", "07_classification_audit")
    classification_columns = [
        "openalex_id", "resource_name", "title", "artifact_type", "domain", "paper_role", "prefilter_score",
        "prefilter_band", "decision", "discovery_relations", "cited_by_count",
    ]
    classification_detail = view.copy()
    for column in classification_columns:
        if column not in classification_detail:
            classification_detail[column] = ""
    write_table(classification_detail[classification_columns], out_dir / "data", "07_classification_audit_records")

    if matrix.empty:
        matrix = pd.DataFrame([[0]], index=["no data"], columns=["no data"])
        percentages = matrix.astype(float)
        score_summary = pd.DataFrame(index=matrix.index, columns=["min", "median", "max"], dtype=float)

    fig_width = max(10.8, 6.8 + .85 * len(matrix.columns))
    fig, ax = plt.subplots(figsize=(fig_width, max(5.5, 3.4 + .9 * len(matrix.index))))
    caption = metadata_caption(view, "rows are deterministic score bands; cells show exact result count and row percentage")
    add_heading(fig, "Acceptance / exclusion by prefilter score band", caption, top=.84)
    image = ax.imshow(percentages.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    for row_index in range(len(matrix.index)):
        for column_index in range(len(matrix.columns)):
            count = int(matrix.iloc[row_index, column_index])
            percent = float(percentages.iloc[row_index, column_index])
            color = "white" if percent >= 55 else "#263238"
            ax.text(column_index, row_index, f"{count}\n{percent:.0f}%", ha="center", va="center", color=color, fontsize=8, fontweight="bold")

    row_labels: list[str] = []
    for band in matrix.index:
        label = artifact_label(band)
        if band in score_summary.index:
            summary = score_summary.loc[band]
            if pd.notna(summary.get("min")) and pd.notna(summary.get("max")):
                label += f"\nscore {summary['min']:g}–{summary['max']:g}; med {summary['median']:g}"
        row_labels.append(label)
    ax.set_xticks(range(len(matrix.columns)), [artifact_label(value) for value in matrix.columns], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), row_labels)
    ax.set_xlabel("Final LLM / review decision")
    ax.set_ylabel("Prefilter classification and score range")
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
    outputs += type_domain_overview(canonical, out_dir, formats, dpi)
    outputs += provenance_matrix(canonical, out_dir, formats, dpi)
    outputs += discovery_source_distribution(canonical, out_dir, formats, dpi)
    outputs += cumulative_publications(canonical, out_dir, formats, dpi)
    outputs += relationship_heatmap(canonical, "domain", "data_form", "Domain and data-form relationships", "04_domain_data_form", out_dir, formats, dpi)
    outputs += relationship_heatmap(canonical, "capture_environment", "labeling", "Capture environment and label availability", "05_capture_environment_labeling", out_dir, formats, dpi)
    outputs += metadata_lollipop(canonical, out_dir, formats, dpi)
    if classifications_path and classifications_path.is_file():
        evaluated = load_frame(classifications_path, ("openalex_id", "prefilter_score", "prefilter_band", "decision", "paper_role", "cited_by_count", "discovery_relations"))
        outputs += classification_audit(evaluated, out_dir, formats, dpi)
        outputs += evidence_escalation_plot(evaluated, out_dir, formats, dpi, "09_evidence_escalation", "OpenAlex")
        outputs += score_decision_plot(evaluated, out_dir, formats, dpi, "11_keyword_score_vs_decision", "OpenAlex")
    if survey_sources_path and survey_sources_path.is_file():
        surveys = load_frame(survey_sources_path, ("openalex_id", "title", "cited_by_count", "discovery_relations"))
        outputs += survey_source_landscape(surveys, out_dir, formats, dpi)
    outputs.append(write_index(out_dir, main_path, frame, outputs, [
        "The timeline uses publication year on x only; the y coordinate is purely a stacking layout for resources published in the same year, and every dataset/generator is directly labelled.",
        "Citation counts are descriptive and never affect retrieval, inclusion, or canonical-base selection.",
        "Discovery provenance shows direct and survey-snowball paths plus the exact number of distinct citing surveys; source contribution is summarized for those two catalogue-producing routes only.",
        "All artifact-level figures use canonical base papers; versions and alternate manifestations remain available in the processed CSV.",
        "Every figure writes a corresponding CSV under data/; aggregate figures also write a *_records.csv audit table listing the contributing resources and metadata.",
        "When classifications.csv is available, the classification audit pairs candidate/borderline/reject score bands with final acceptance/exclusion decisions, exact counts, percentages, and observed score ranges.",
        "The type/domain matrix summarizes dataset vs generator coverage across application domains such as general, IoT, and automotive where present.",
        "Metadata completeness reports both percentage and exact n/N availability for each tracked field.",
        "Survey-source scores, citations, and reference-list sizes use aligned panels rather than overloaded marker encodings.",
        "09_evidence_escalation reports how often extra abstract/full-text evidence was required, found, and beneficial; every bar states its denominator.",
        "11_keyword_score_vs_decision directly compares every numeric keyword-prefilter score with its exact final LLM decision; the companion summary reports n, range, mean, median, and quartiles per decision.",
    ]))
    LOG.info("OpenAlex plot bundle complete canonical_artifacts=%d files=%d output=%s", len(canonical), len(outputs), out_dir)
    return outputs
