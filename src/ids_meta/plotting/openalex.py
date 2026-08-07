from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ids_meta.plotting.common import COLORS, add_heading, configure_style, count_legend, count_plot, load_frame, metadata_caption, save_figure, scrollable_table, stamp, write_index, write_table
from ids_meta.common.logging import logger

LOG = logger("plotting.openalex")


def build(main_path: Path, adjacent_path: Path | None, out_dir: Path, config: dict[str, object]) -> list[Path]:
    """Generate the OpenAlex core/adjacent plot and aggregate-table bundle."""
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
    frame = pd.concat(frames, ignore_index=True)
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    canonical = frame[frame["paper_relationship"] == "canonical_base"].copy()
    caption = metadata_caption(canonical, "one row per reconciled artifact family")
    outputs: list[Path] = []
    overview_columns = [
        "artifact_name", "publication_year", "publication_date", "artifact_type", "paper_role",
        "catalogue_tier", "domain", "data_form", "title", "doi", "openalex_id",
    ]
    for column in overview_columns:
        if column not in canonical:
            canonical[column] = ""
    overview = canonical[overview_columns].sort_values(["publication_year", "artifact_name"], kind="stable")
    write_table(overview, out_dir / "data", "00_artifact_timeline_overview")
    outputs += scrollable_table(
        overview.drop(columns=["doi", "openalex_id"]), "Dataset and generator publication overview",
        metadata_caption(canonical, "canonical base papers; every reconciled artifact family is listed"),
        out_dir, "00_artifact_timeline_overview", formats, dpi,
    )
    outputs += count_plot(frame, "paper_relationship", "Paper relationships", metadata_caption(frame, "all accepted release-paper rows"), out_dir, "01_paper_relationships", formats, dpi)
    for number, column, title in (
        (2, "catalogue_tier", "Base-paper catalogue tiers"), (3, "artifact_type", "Artifact types"),
        (4, "domain", "Dataset domains"), (5, "data_form", "Data forms"),
        (6, "capture_environment", "Capture environments"), (7, "labeling", "Label availability"),
    ):
        outputs += count_plot(canonical, column, title, caption, out_dir, f"0{number}_{column}", formats, dpi)
    timeline_source = canonical.assign(publication_year_numeric=pd.to_numeric(canonical["publication_year"], errors="coerce")).dropna(subset=["publication_year_numeric"])
    timeline_source["publication_year"] = timeline_source["publication_year_numeric"].astype(int)
    years = timeline_source.groupby(["publication_year", "artifact_type"], dropna=False).size().rename("count").reset_index().sort_values(["publication_year", "artifact_type"])
    write_table(years, out_dir / "data", "08_publication_timeline")
    pivot = years.pivot(index="publication_year", columns="artifact_type", values="count").fillna(0)
    fig, ax = plt.subplots()
    if pivot.empty:
        ax.axis("off")
        ax.text(.5, .5, "No dated canonical artifacts in the current catalogue", transform=ax.transAxes, ha="center", va="center", color="#5f6c76")
    else:
        pivot.rename(columns=lambda value: str(value).replace("_", " ")).plot.bar(stacked=True, ax=ax, color=COLORS[:len(pivot.columns)], width=.82)
        ax.set_xlabel("Publication year"); ax.set_ylabel("Canonical base papers")
        ax.legend(title="Artifact type", frameon=False, fontsize=8)
    add_heading(fig, "Dataset and generator publication timeline", caption)
    stamp(fig, caption)
    outputs += save_figure(fig, out_dir, "08_publication_timeline", formats, dpi)
    matrix = pd.crosstab(canonical["domain"].replace("", "unknown"), canonical["data_form"].replace("", "unknown"))
    if matrix.empty:
        matrix = pd.DataFrame([[0]], index=["no data"], columns=["no data"])
    write_table(matrix.reset_index(), out_dir / "data", "09_domain_data_form")
    fig, ax = plt.subplots(figsize=(10, 6)); image = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [str(value).replace("_", " ") for value in matrix.columns], rotation=35, ha="right"); ax.set_yticks(range(len(matrix.index)), [str(value).replace("_", " ") for value in matrix.index])
    add_heading(fig, "Domain by data form", caption); fig.colorbar(image, ax=ax, label="Artifacts")
    stamp(fig, caption)
    outputs += save_figure(fig, out_dir, "09_domain_data_form", formats, dpi)
    metadata_fields = ["doi", "authors", "venue", "publication_year", "primary_url", "pdf_url", "domain", "data_form", "capture_environment", "labeling"]
    completeness = pd.DataFrame({"field": metadata_fields, "present": [((canonical[field] != "") & (canonical[field].str.casefold() != "unknown")).sum() if field in canonical else 0 for field in metadata_fields]})
    completeness["percent"] = completeness["present"] / max(1, len(canonical)) * 100
    write_table(completeness, out_dir / "data", "10_metadata_completeness")
    fig, ax = plt.subplots(); ordered = completeness.iloc[::-1]
    ax.barh(ordered["field"], ordered["percent"], color="#4E9A62"); ax.set_xlim(0, 100); ax.set_xlabel("Rows populated (%)")
    add_heading(fig, "Base-paper metadata completeness", caption)
    count_legend(ax, "Rows populated (%)", "#4E9A62")
    stamp(fig, caption)
    outputs += save_figure(fig, out_dir, "10_metadata_completeness", formats, dpi)
    outputs.append(write_index(out_dir, main_path, frame, outputs, [
        "Core includes security-intended dataset and generator releases.",
        "Adjacent contains original neutral foundational releases and is always shown as a separate tier.",
        "Artifact-level figures count canonical base papers only; relationship counts retain versions and alternate manifestations.",
        "Citations are descriptive metadata only and never affect selection or canonicalization.",
        "The complete chronological artifact overview is data/00_artifact_timeline_overview.csv; rendered pages list every canonical family.",
    ]))
    LOG.info("OpenAlex plot bundle complete files=%d output=%s", len(outputs), out_dir)
    return outputs
