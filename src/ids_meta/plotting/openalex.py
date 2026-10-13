from __future__ import annotations

import math
from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import pandas as pd

from ids_meta.common.logging import logger
from ids_meta.common.provenance import audited_bundle
from ids_meta.plotting.common import add_heading, configure_style, count_plot, evidence_escalation_plot, load_frame, metadata_caption, save_figure, score_decision_plot, subcaption, write_index, write_table

LOG = logger("plotting.openalex")

TYPE_COLORS = {
    "dataset": "#2F6F9F",
    "generator_framework": "#D1842F",
    "dataset_and_generator": "#7B5EA7",
    "unknown": "#6F7782",
}

TYPE_MARKERS = {
    "dataset": "o",
    "generator_framework": "s",
    "dataset_and_generator": "D",
    "unknown": "X",
}

TYPE_LINESTYLES = {
    "dataset": "solid",
    "generator_framework": "dashed",
    "dataset_and_generator": "dotted",
    "unknown": "dashdot",
}

DOMAIN_COLORS = {
    "general_network": "#2F6F9F", "host_endpoint": "#B04A5A", "iot_iiot": "#D1842F",
    "cloud": "#7B5EA7", "mobile_android": "#3F8F83", "electrical_smartgrid": "#CCB974",
    "uav": "#8A6D3B", "vpn": "#C76EAF", "unknown": "#6F7782",
}

DOMAIN_LABELS = {
    "general_network": "General network", "host_endpoint": "Host / endpoint",
    "iot_iiot": "IoT / IIoT", "cloud": "Cloud", "mobile_android": "Mobile / Android",
    "electrical_smartgrid": "Electrical / smart grid", "uav": "UAV", "vpn": "VPN",
    "unknown": "Unknown",
}

DISCOVERY_COLORS = {
    "Direct search": "#2F6F9F",
    "Survey snowball": "#7B5EA7",
}
def artifact_label(value: object) -> str:
    """Format a normalized artifact type for display."""
    return str(value or "unknown").replace("_", " ")


def domain_label(value: object) -> str:
    """Format a normalized security domain for display."""
    normalized = str(value or "unknown")
    return DOMAIN_LABELS.get(normalized, normalized.replace("_", " "))


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
    """Count distinct surveys citing a work, from explicit survey identifiers."""
    relations = {item.strip() for item in str(value or "").split(";") if item.strip()}
    survey_ids = {item.split(":", 1)[1] for item in relations if item.startswith("survey_reference:") and item.split(":", 1)[1]}
    if survey_ids:
        return len(survey_ids)
    if "survey_reference" in relations:
        raise ValueError("survey discovery provenance requires explicit citing survey IDs")
    return 0


def concise_text(value: object, limit: int = 64) -> str:
    """Collapse whitespace and shorten long display text without changing stored CSV values."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def resource_display_name(row: pd.Series) -> str:
    """Use the current exported resource name as the plot label."""
    if "resource_name" not in row or not str(row["resource_name"]).strip():
        raise ValueError("paper plotting requires an explicit resource_name")
    return concise_text(row["resource_name"], 54)


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


def spread_label_coordinates(values: list[float], lower: float, upper: float, minimum_gap: float) -> list[float]:
    """Spread label centers vertically while preserving their order and staying inside the plot."""
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    available = max(0.0, upper - lower)
    gap = min(max(0.0, minimum_gap), available / max(1, len(values) - 1))
    placed: list[float] = []
    for _index, value in indexed:
        position = min(upper, max(lower, float(value)))
        if placed:
            position = max(position, placed[-1] + gap)
        placed.append(position)
    overflow = max(0.0, placed[-1] - upper)
    if overflow:
        placed = [value - overflow for value in placed]
    if placed[0] < lower:
        placed = [value + lower - placed[0] for value in placed]
    result = [0.0] * len(values)
    for (original_index, _value), position in zip(indexed, placed):
        result[original_index] = position
    return result


def spread_label_boxes(
    values: list[float], heights: list[float], lower: float, upper: float, gap: float,
) -> list[float]:
    """Pack measured callouts in citation order without overlapping or shrinking text."""
    if len(values) != len(heights):
        raise ValueError("every label must have a measured height")
    if not values:
        return []
    if sum(heights) + gap * (len(values) - 1) > upper - lower + 1e-9:
        raise ValueError("label boxes need more vertical space")
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    offsets = [0.0]
    for previous, index in zip(order, order[1:]):
        offsets.append(offsets[-1] + (heights[previous] + heights[index]) / 2 + gap)
    # Isotonic least-squares placement shares displacement around crowded points,
    # rather than pushing every later label upward into long leader-line fans.
    blocks: list[tuple[float, int]] = []
    for index, offset in zip(order, offsets):
        blocks.append((values[index] - offset, 1))
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right_sum, right_count = blocks.pop()
            left_sum, left_count = blocks.pop()
            blocks.append((left_sum + right_sum, left_count + right_count))
    minimum = lower + heights[order[0]] / 2
    maximum = upper - heights[order[-1]] / 2 - offsets[-1]
    fitted = [min(maximum, max(minimum, total / size)) for total, size in blocks for _ in range(size)]
    placed = [value + offset for value, offset in zip(fitted, offsets)]
    result = [0.0] * len(values)
    for index, center in zip(order, placed):
        result[index] = center
    return result


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
    view["artifact_type"] = view["artifact_type"].fillna("").replace("", "unknown")
    view["domain"] = view["domain"].fillna("").replace("", "unknown")
    view["publication_year_numeric"] = pd.to_numeric(view["publication_year"], errors="coerce")
    view["citations"] = pd.to_numeric(view["cited_by_count"], errors="coerce").fillna(0).clip(lower=0)
    view["provenance"] = view["discovery_relations"].map(discovery_provenance)
    view["resource_name"] = view.apply(resource_display_name, axis=1)
    return view


def timeline_citation_coordinate(value: object) -> float:
    """Use actual counts through 250, then 2.5% of linear upper-tail spacing."""
    count = max(0.0, float(value))
    return count if count <= 250 else 250 + .025 * (count - 250)


def publication_timeline(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Place styled labels on a linear citation axis with a compressed upper tail."""
    dated = frame.dropna(subset=["publication_year_numeric"]).copy()
    # Stacks grow upward: ascending counts put the most-cited resource on top.
    dated = dated.sort_values(["publication_year_numeric", "citations", "resource_name", "openalex_id"], kind="stable")
    dated["publication_year"] = dated["publication_year_numeric"].astype(int)
    dated["same_year_index"] = dated.groupby("publication_year", sort=False).cumcount()
    dated["same_year_count"] = dated.groupby("publication_year", sort=False)["openalex_id"].transform("size")
    dated["citation_coordinate"] = dated["citations"].map(timeline_citation_coordinate)
    dated["timeline_x"] = dated["publication_year"].astype(float)
    dated["timeline_x_offset"] = 0.0
    dated["label_x"] = dated["timeline_x"]
    dated["label_x_offset"] = 0.0
    dated["label_side"] = "above"
    dated["label_alignment"] = "center"
    dated["layout_y_unit"] = "citations through 250; 250 + 0.025 * (citations - 250) above 250"
    dated["citation_anchor_x"] = dated["timeline_x"] + .48
    dated["label_citation_offset"] = 0.0
    dated["label_adjusted_for_overlap"] = False
    dated["line_style"] = dated["artifact_type"].map(TYPE_LINESTYLES).fillna(TYPE_LINESTYLES["unknown"])
    for column in ("label_coordinate", "label_lower_coordinate", "label_upper_coordinate",
                   "label_left_coordinate", "label_right_coordinate", "panel_number",
                   "panel_start_year", "panel_end_year"):
        dated[column] = 0.0
    width, footer, header = 36.0, 1.90, .90
    fig = plt.figure(figsize=(width, 10.0))
    ax = fig.add_axes([.050, footer / 10, .925, (10 - footer - header) / 10])
    first_year = int(dated["publication_year"].min()) if not dated.empty else 0
    last_year = int(dated["publication_year"].max()) if not dated.empty else 0
    ax.set_xlim(first_year - .65, last_year + .65)
    ax.set_ylim(0, 1)
    annotations = {}
    for index, row in dated.iterrows():
        color = DOMAIN_COLORS.get(str(row["domain"]), DOMAIN_COLORS["unknown"])
        annotations[index] = ax.text(
            row["label_x"], .5, str(row["resource_name"]), ha="center", va="center",
            fontsize=10, color="#243746", linespacing=1.15, zorder=3,
            bbox={"boxstyle": "round,pad=.24", "facecolor": "white", "edgecolor": color,
                  "linewidth": 1.15, "linestyle": row["line_style"]})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # A label is narrower than one year interval, so neighboring year stacks
    # cannot collide. Measure padding and wrapped text rather than guessing.
    year_width = ax.bbox.width / (last_year - first_year + 1.30)
    label_width = min(1.50 * fig.dpi, year_width * .87)
    box_heights = {}
    box_widths = {}
    for index, annotation in annotations.items():
        wrap_width = 24
        while True:
            annotation.set_text(textwrap.fill(str(dated.loc[index, "resource_name"]), width=wrap_width))
            annotation.update_bbox_position_size(renderer)
            box = annotation.get_bbox_patch().get_window_extent(renderer)
            if box.width <= label_width or wrap_width == 1:
                break
            wrap_width -= 1
        box_heights[index] = box.height / fig.dpi
        box_widths[index] = box.width / year_width
    gap, baseline_gap = .08, .22
    stacks = dated.groupby("publication_year", sort=False).groups
    stack_heights = [baseline_gap + sum(box_heights[index] for index in indices) + gap * (len(indices) - 1)
                     for indices in stacks.values()]
    # More physical space makes the populated 0–250 citation band useful
    # for wrapped names while retaining a wide landscape canvas.
    plot_height = max(12.0, max(stack_heights, default=0) + .60)
    maximum_citations = float(dated["citations"].max()) if not dated.empty else 0.0
    citation_upper = max(1.0, timeline_citation_coordinate(maximum_citations) * 1.08)
    citation_lower = -.03 * citation_upper
    height = plot_height + footer + header
    fig.set_size_inches(width, height)
    ax.set_position([.050, footer / height, .925, plot_height / height])
    ax.set_ylim(citation_lower, citation_upper)
    fig.suptitle("IDS dataset and generator publication timeline", x=.5, y=1 - .14 / height,
                 ha="center", va="top", fontsize=15, fontweight="bold")
    subcaption(fig, .035, 1 - .50 / height,
               f"{len(dated):,} resources · linear citations through 250 · labels spaced to avoid overlap · color: domain · line style: resource type",
               fontsize=10, color="#5F6C76", ha="left", va="top")
    if dated.empty:
        ax.axis("off")
        ax.text(.5, .5, "No dated canonical artifacts in the current catalogue", transform=ax.transAxes,
                ha="center", va="center")
    else:
        dated["panel_number"] = 1
        dated["panel_start_year"] = first_year
        dated["panel_end_year"] = last_year
        # Pack measured boxes around their true citation coordinates, keeping
        # order and moving labels only when their boxes would collide. Short
        # styled leader lines end at exact citation ticks beside each column.
        fig.canvas.draw()
        pixels_per_coordinate = ax.bbox.height / (citation_upper - citation_lower)
        for year, indices in stacks.items():
            coordinates = [float(dated.loc[index, "citation_coordinate"]) for index in indices]
            centers = spread_label_boxes(
                coordinates, [box_heights[index] * fig.dpi / pixels_per_coordinate for index in indices],
                0.0, citation_upper, gap * fig.dpi / pixels_per_coordinate)
            for index, center in zip(indices, centers):
                annotations[index].set_position((float(year), center))
                dated.loc[index, "label_coordinate"] = center
                actual = float(dated.loc[index, "citation_coordinate"])
                dated.loc[index, "label_citation_offset"] = center - actual
                dated.loc[index, "label_adjusted_for_overlap"] = abs(center - actual) > 1e-9
                color = DOMAIN_COLORS.get(str(dated.loc[index, "domain"]), DOMAIN_COLORS["unknown"])
                style = dated.loc[index, "line_style"]
                anchor_x = float(dated.loc[index, "citation_anchor_x"])
                edge_x = float(year) + box_widths[index] / 2
                ax.plot([edge_x, anchor_x], [center, actual], color=color,
                        linestyle=style, linewidth=.9, alpha=.8, zorder=2)
                ax.plot([anchor_x - .025, anchor_x + .025], [actual, actual],
                        color=color, linestyle=style, linewidth=1.3, zorder=2)
        ax.set_xticks(range(first_year, last_year + 1))
        ax.set_xlabel("Publication year", fontsize=12, labelpad=12)
        tick_counts = [value for value in (0, 50, 100, 150, 200, 250, 1_000, 2_000, 3_000, 4_000, 5_000,
                                           10_000, 25_000, 50_000, 100_000)
                       if value <= maximum_citations]
        if maximum_citations > 0 and (timeline_citation_coordinate(maximum_citations) - timeline_citation_coordinate(tick_counts[-1])) > .12 * citation_upper:
            tick_counts.append(int(maximum_citations))
        ax.set_yticks([timeline_citation_coordinate(value) for value in tick_counts], [f"{value:,}" for value in tick_counts])
        ax.set_ylabel("Citations (linear to 250; uniformly compressed above)", fontsize=12, labelpad=12)
        if maximum_citations > 250:
            knee = timeline_citation_coordinate(250)
            ax.axhline(knee, color="#82909C", linewidth=.8, linestyle="--", alpha=.6, zorder=1)
            # Scale information must stay visible even with subcaptions disabled.
            ax.annotate("Uniform scale compression above 250", (.012, knee), xycoords=("axes fraction", "data"),
                        xytext=(0, 6), textcoords="offset points", fontsize=9,
                        color="#5F6C76", ha="left", va="bottom")
        ax.tick_params(axis="x", labelsize=10, length=6, width=1, color="#66727C")
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", linewidth=.5, alpha=.20)
        for side in ("right", "top"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#66727C")
        ax.spines["bottom"].set_color("#66727C")
        ax.spines["bottom"].set_linewidth(1.2)
    domain_handles = [Patch(facecolor=color, label=domain_label(name))
                      for name, color in DOMAIN_COLORS.items() if name in set(dated["domain"])]
    type_handles = [Line2D([0], [0], color="#435563", linestyle=style, linewidth=1.5,
                          label=artifact_label(name)) for name, style in TYPE_LINESTYLES.items()
                    if name in set(dated["artifact_type"])]
    if domain_handles:
        fig.legend(handles=domain_handles, title="Dataset domain (color)", loc="upper left",
                   bbox_to_anchor=(.035, 1.22 / height), ncol=5, frameon=False, fontsize=10, title_fontsize=10)
    if type_handles:
        fig.legend(handles=type_handles, title="Dataset / generator (line style)", loc="upper left",
                   bbox_to_anchor=(.035, .56 / height), ncol=4, frameon=False, fontsize=10,
                   title_fontsize=10, handlelength=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for index, annotation in annotations.items():
        box = annotation.get_bbox_patch().get_window_extent(renderer)
        left, bottom = ax.transData.inverted().transform((box.x0, box.y0))
        right, top = ax.transData.inverted().transform((box.x1, box.y1))
        dated.loc[index, ["label_left_coordinate", "label_right_coordinate",
                          "label_lower_coordinate", "label_upper_coordinate"]] = [left, right, bottom, top]
    columns = [
        "resource_name", "artifact_display_name", "artifact_name", "publication_year", "publication_date",
        "citations", "citation_as_of", "artifact_type", "domain", "data_form", "capture_environment", "labeling", "paper_role",
        "catalogue_tier", "provenance", "discovery_relations", "prefilter_score", "title", "doi", "openalex_id",
        "same_year_index", "same_year_count", "citation_coordinate", "timeline_x_offset", "timeline_x",
        "label_side", "label_alignment", "label_x_offset", "label_x", "label_coordinate", "layout_y_unit", "line_style",
        "citation_anchor_x", "label_citation_offset", "label_adjusted_for_overlap",
        "panel_number", "panel_start_year", "panel_end_year", "label_lower_coordinate", "label_upper_coordinate",
        "label_left_coordinate", "label_right_coordinate",
    ]
    for column in columns:
        if column not in dated:
            dated[column] = ""
    write_table(dated[columns], out_dir / "data", "00_publication_timeline")
    with plt.rc_context({"savefig.bbox": None}):
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

def _cumulative_dimension(
    dated: pd.DataFrame, dimension: str, preferred_order: list[str],
) -> tuple[pd.DataFrame, list[int], list[str], pd.DataFrame]:
    """Build complete annual and cumulative counts for one categorical dimension."""
    columns = [
        "publication_year", "dimension", "category", "category_label",
        "new_resources", "cumulative_resources", "cumulative_all_resources",
    ]
    if dated.empty:
        return pd.DataFrame(columns=columns), [], [], pd.DataFrame()
    first_year = int(dated["publication_year"].min())
    last_year = int(dated["publication_year"].max())
    years = list(range(first_year, last_year + 1))
    observed = set(dated[dimension].fillna("unknown").replace("", "unknown").astype(str))
    categories = [value for value in preferred_order if value in observed]
    categories.extend(sorted(observed - set(categories)))
    source = dated.assign(**{dimension: dated[dimension].fillna("unknown").replace("", "unknown")})
    annual = source.groupby(["publication_year", dimension]).size().unstack(fill_value=0)
    annual = annual.reindex(index=years, columns=categories, fill_value=0).astype(int)
    cumulative = annual.cumsum()
    labeler = artifact_label if dimension == "artifact_type" else domain_label
    rows = []
    for year in years:
        total = int(cumulative.loc[year].sum())
        for category in categories:
            rows.append({
                "publication_year": year,
                "dimension": "resource_type" if dimension == "artifact_type" else "security_domain",
                "category": category, "category_label": labeler(category),
                "new_resources": int(annual.at[year, category]),
                "cumulative_resources": int(cumulative.at[year, category]),
                "cumulative_all_resources": total,
            })
    return pd.DataFrame(rows, columns=columns), years, categories, cumulative


def cumulative_publications(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot cumulative canonical-resource growth by resource type and security domain."""
    dated = frame.dropna(subset=["publication_year_numeric"]).copy()
    dated["publication_year"] = dated["publication_year_numeric"].astype(int)
    type_order = list(TYPE_COLORS)
    domain_order = list(DOMAIN_COLORS)
    type_table, type_years, type_categories, type_cumulative = _cumulative_dimension(dated, "artifact_type", type_order)
    domain_table, domain_years, domain_categories, domain_cumulative = _cumulative_dimension(dated, "domain", domain_order)
    cumulative_table = pd.concat([type_table, domain_table], ignore_index=True)
    stem = "03_cumulative_growth_by_type_and_domain"
    write_table(cumulative_table, out_dir / "data", stem)
    detail_columns = ["resource_name", "artifact_name", "publication_year", "artifact_type", "domain", "provenance", "discovery_relations", "openalex_id", "doi"]
    detail = dated.copy()
    for column in detail_columns:
        if column not in detail:
            detail[column] = ""
    write_table(detail[detail_columns], out_dir / "data", stem + "_records")

    fig, axes = plt.subplots(2, 1, figsize=(14.2, 10.0), sharex=True)
    caption = metadata_caption(
        frame,
        "canonical resource families; each coloured band is cumulative and the full stacked height equals all resources",
    )
    add_heading(fig, "Cumulative catalogue growth by resource type and security domain", caption, top=.90)
    fig.subplots_adjust(left=.09, right=.76, bottom=.09, top=.82, hspace=.28)
    panels = (
        (axes[0], type_years, type_categories, type_cumulative, TYPE_COLORS, artifact_label, "Resource type"),
        (axes[1], domain_years, domain_categories, domain_cumulative, DOMAIN_COLORS, domain_label, "Security domain"),
    )
    for ax, years, categories, cumulative, colors, labeler, legend_title in panels:
        if not years or not categories:
            ax.axis("off")
            ax.text(.5, .5, "No dated canonical resources", transform=ax.transAxes, ha="center", va="center")
            continue
        values = [cumulative[category].to_numpy() for category in categories]
        legend_labels = [
            f"{labeler(category)} (n={int(cumulative[category].iloc[-1])})"
            for category in categories
        ]
        ax.stackplot(
            years, *values, labels=legend_labels,
            colors=[colors.get(category, colors["unknown"]) for category in categories],
            alpha=.82, step="post",
        )
        total = cumulative.sum(axis=1)
        ax.step(years, total, where="post", color="#26333D", linewidth=1.5, label="All resources")
        ax.scatter(years, total, color="#26333D", s=13, zorder=4)
        ax.set_title(legend_title, loc="center", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative canonical resources")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(title=legend_title, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    axes[1].set_xlabel("Publication year")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    return save_figure(fig, out_dir, stem, formats, dpi)


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

def _openalex_why_not_included(row: pd.Series) -> tuple[str, str, str]:
    """Explain one excluded or unresolved paper in reader-facing language."""
    decision = str(row.get("decision", "")).strip().casefold()
    raw_flags = ";".join(filter(None, (
        str(row.get("classification_validation_flags", "")),
        str(row.get("consensus_validation_flags", "")),
    )))
    flags = [value.strip() for value in raw_flags.split(";") if value.strip()]
    flag_meanings = {
        "role_artifact_type_mismatch": ("Paper role and resource type conflict", "The claimed paper contribution is incompatible with the classified dataset or generator type."),
        "tier_not_accepted_for_role": ("Relevance level does not qualify for this paper role", "The paper role and relevance level do not form an eligible catalogue contribution."),
        "direct_release_evidence_missing": ("No evidence that the paper releases a resource", "The evidence does not directly show that the paper creates or releases the dataset or generator."),
        "artifact_name_missing": ("Released resource could not be identified by name", "No concrete dataset or generator name could be established from the evidence."),
        "artifact_identity_evidence_missing": ("Resource name was not supported by the quoted evidence", "The classifier supplied a dataset or generator name that does not occur in its verified evidence quotation."),
        "artifact_identity_consensus_conflict": ("Repeated classifications identified different resources", "Repeated classifications of the paper assigned incompatible dataset or generator names."),
        "cross_model_artifact_identity_conflict": ("Models identified different resources", "The agreeing models assigned incompatible dataset or generator names, so no label was selected silently."),
        "verified_evidence_missing": ("No supporting quotation could be verified", "The classification lacks a quotation that can be matched to the collected paper evidence."),
        "model_role_uncertain": ("Paper contribution could not be determined", "The model could not determine whether the paper releases, studies, or merely uses a resource."),
        "decision_fields_conflict": ("Classification fields conflict", "The combined paper role and evidence fields do not support the ensemble decision."),
    }
    if decision == "error":
        return "Processing failed", "Final decision is error", "The classification request failed, so no inclusion decision was available."
    if decision == "uncertain":
        if flags:
            label, explanation = flag_meanings.get(flags[0], (flags[0].replace("_", " ").capitalize(), "A deterministic classification requirement was not satisfied."))
            return label, f"Automatic validation check: {flags[0]}", explanation
        vote_status = str(row.get("ensemble_vote_status", "")).strip()
        if vote_status in {"tie", "below_threshold"}:
            return "Models did not agree", f"Ensemble result: {vote_status}", "The configured models did not reach the agreement threshold for a definitive decision."
        return "Not enough evidence for a decision", "Final decision is uncertain", "The collected paper evidence did not support either inclusion or exclusion."
    role = str(row.get("paper_role", "")).strip().casefold()
    categories = {
        "survey": ("Survey only — no new dataset or generator", "The paper reviews existing resources but does not introduce an eligible new dataset or generator."),
        "downstream_application": ("Uses an existing resource — does not release one", "The paper applies an existing dataset or generator rather than creating or releasing it."),
        "dataset_analysis": ("Analyzes existing datasets — does not release one", "The paper studies or compares existing datasets without releasing an eligible resource."),
        "unrelated": ("No eligible IDS dataset or generator contribution", "The paper does not make an eligible intrusion-detection dataset or generator contribution."),
    }
    if role in categories:
        label, explanation = categories[role]
        return label, f"Structured paper role: {role}", explanation
    return "No eligible dataset or generator release was established", "Structured paper role did not match a narrower group", "The paper was excluded, but its structured role did not identify a more specific reason."


def why_papers_not_included(frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot plain-language paper exclusion reasons and preserve detailed evidence."""
    decisions = frame.get("decision", pd.Series("", index=frame.index)).fillna("").astype(str).str.casefold()
    view = frame[decisions.isin({"exclude", "uncertain", "error"})].copy()
    details = [_openalex_why_not_included(row) for _, row in view.iterrows()]
    view["why_not_included"] = [value[0] for value in details]
    view["how_reason_was_assigned"] = [value[1] for value in details]
    view["criterion_explanation"] = [value[2] for value in details]
    columns = [
        value for value in (
            "openalex_id", "doi", "title", "decision", "paper_role", "relevance_tier",
            "why_not_included", "criterion_explanation", "how_reason_was_assigned",
            "classification_validation_flags", "primary_reason", "model_evidence_json",
            "evidence_json", "prefilter_band", "prefilter_score", "discovery_relations",
            "ensemble_vote_status", "ensemble_agreement", "error",
        ) if value in view
    ]
    write_table(view[columns].copy(), out_dir / "data", "12_why_not_included_records")
    definitions = view[["why_not_included", "criterion_explanation"]].drop_duplicates().sort_values("why_not_included")
    write_table(definitions, out_dir / "data", "12_why_not_included_definitions")
    return count_plot(
        view, "why_not_included", "Why papers are not in the catalogue",
        metadata_caption(view, "excluded, unresolved, and failed classifications; definitions and exact explanations are exported below"),
        out_dir, "12_why_not_included", formats, dpi,
    )


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


@audited_bundle
def build(
    main_path: Path, adjacent_path: Path | None, out_dir: Path, config: dict[str, object],
    classifications_path: Path | None = None, survey_sources_path: Path | None = None,
) -> list[Path]:
    """Generate the OpenAlex timeline, citation, provenance, and relationship bundle."""
    LOG.info("building OpenAlex plots core_input=%s adjacent_input=%s output=%s", main_path, adjacent_path or "none", out_dir)
    configure_style(config)
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
        outputs += why_papers_not_included(evaluated, out_dir, formats, dpi)
    if survey_sources_path and survey_sources_path.is_file():
        surveys = load_frame(survey_sources_path, ("openalex_id", "title", "cited_by_count", "discovery_relations"))
        outputs += survey_source_landscape(surveys, out_dir, formats, dpi)
    outputs.append(write_index(out_dir, main_path, frame, outputs, [
        "The publication timeline uses a continuous horizontal year axis on a 36-inch-wide landscape canvas and a linear citation axis from zero through 250. Above 250, equal citation differences use 2.5% of the lower range’s vertical spacing (40-fold compression), so the upper range remains uniformly linear rather than logarithmic. The mapping is continuous at 250, the change is marked, and tick labels show actual citation counts. Resource labels form year columns in citation order, with the most-cited at the top; there are no scatter dots. Measured label boxes are placed as close to their true citation heights as possible and shifted locally when necessary to avoid overlaps. Short styled leaders end at exact citation-height ticks beside each year column. Domain colors identify application areas; solid borders/lines indicate datasets, dashed generators, dotted combined releases, and dash-dot unknown types. The companion CSV records true citation coordinates, displayed label coordinates, displacement flags/offsets, line styles, and measured box boundaries. SVG and PDF preserve the full wide landscape canvas.",
        "03_cumulative_growth_by_type_and_domain uses cumulative stacked bands for resource type and security domain; at every year the full stacked height and dark outline equal the number of canonical resources published up to that year.",
        "Citation counts are descriptive and never affect retrieval, inclusion, or canonical-base selection.",
        "Discovery provenance shows direct and survey-snowball paths plus the exact number of distinct citing surveys; source contribution is summarized for those two catalogue-producing routes only.",
        "All artifact-level figures use canonical base papers; versions and alternate manifestations remain available in the processed CSV.",
        "Every figure writes a corresponding CSV under data/; aggregate figures also write a *_records.csv audit table listing the contributing resources and metadata.",
        "When classifications.csv is available, the classification audit pairs candidate/borderline/reject score bands with final acceptance/exclusion decisions, exact counts, percentages, and observed score ranges.",
        "The type/domain matrix summarizes dataset vs generator coverage across application domains such as general, IoT, and automotive where present.",
        "Metadata completeness reports both percentage and exact n/N availability for each tracked field.",
        "Survey-source scores, citations, and reference-list sizes use aligned panels rather than overloaded marker encodings.",
        "09_evidence_escalation reports how often extra abstract/full-text evidence was required, found, and beneficial; every bar states its denominator.",
        "11_keyword_score_vs_decision shows every scored item as a dot by final decision; the box is the middle 50%, its line is the median, and whiskers reach the furthest non-outlier within 1.5 x IQR. Dots beyond the whiskers are valid outliers; the companion summary exports the fences, actual whisker endpoints, and outlier count.",
        "12_why_not_included uses plain-language reasons for papers absent from the catalogue; its definitions and records tables preserve each criterion, structured paper role, assignment basis, validation flag, exact model explanation, and evidence.",
    ]))
    LOG.info("OpenAlex plot bundle complete canonical_artifacts=%d files=%d output=%s", len(canonical), len(outputs), out_dir)
    return outputs
