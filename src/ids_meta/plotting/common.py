from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from ids_meta.common.logging import logger

LOG = logger("plotting")

COLORS = ["#2F6B9A", "#4E9A62", "#D9902F", "#7A5FA8", "#C94F4F", "#3B8C8C", "#B85C9E", "#8C6D31", "#6C7A89"]


def configure_style() -> None:
    """Apply the shared non-interactive matplotlib theme."""
    plt.rcParams.update({
        "figure.figsize": (9, 5.4), "figure.dpi": 120, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.alpha": .22, "font.size": 10,
    })


def load_frame(path: Path, required: Iterable[str]) -> pd.DataFrame:
    """Load a CSV as strings and validate its required plotting columns."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    LOG.info("loaded plotting input %s rows=%d columns=%d", path, len(frame), len(frame.columns))
    return frame


def explode_values(frame: pd.DataFrame, column: str, id_column: str) -> pd.DataFrame:
    """Expand semicolon-delimited values into one row per identifier and value."""
    view = frame[[id_column, column]].copy()
    view[column] = view[column].fillna("").str.split(";")
    view = view.explode(column)
    view[column] = view[column].str.strip()
    return view[view[column] != ""]


def metadata_caption(frame: pd.DataFrame, extra: str = "") -> str:
    """Summarize sample size, run identity, and metadata date for a figure."""
    run_ids = sorted(set(value for value in frame.get("run_id", pd.Series(dtype=str)).astype(str) if value))
    run = run_ids[0] if len(run_ids) == 1 else "multiple runs" if run_ids else "unknown run"
    fetched = sorted(set(value[:10] for value in frame.get("metadata_fetched_at", pd.Series(dtype=str)).astype(str) if value))
    as_of = fetched[-1] if fetched else "unknown date"
    return f"n={len(frame)}" + (f"; {extra}" if extra else "")


def stamp(fig: plt.Figure, caption: str) -> None:
    """Add a small provenance caption at the bottom of a figure."""
    fig.text(.01, .005, caption, fontsize=7.5, color="#5f6c76", ha="left")


def add_heading(fig: plt.Figure, title: str, subtitle: str, *, top: float | None = None) -> None:
    """Reserve fixed physical margins for a title and subtitle on short or tall figures."""
    height = max(1.0, float(fig.get_figheight()))
    title_y = 1.0 - .12 / height
    subtitle_y = 1.0 - .48 / height
    physical_top = 1.0 - .90 / height
    axes_top = max(top, physical_top) if top is not None else physical_top
    axes_bottom = max(.10, .95 / height)
    fig.subplots_adjust(top=axes_top, bottom=axes_bottom)
    fig.suptitle(title, x=.08, y=title_y, ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(.08, subtitle_y, textwrap.fill(subtitle, width=115), ha="left", va="top", fontsize=8.5, color="#5f6c76")


def count_legend(ax: plt.Axes, label: str, color: str = COLORS[0]) -> None:
    """Add a compact single-series legend to a count chart."""
    ax.legend(handles=[Patch(facecolor=color, label=label)], loc="best", frameon=False, fontsize=8)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str], dpi: int) -> list[Path]:
    """Save a figure in each requested format and close its resources."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for extension in formats:
        path = out_dir / f"{stem}.{extension}"
        fig.savefig(path, dpi=dpi, metadata={"Title": stem})
        paths.append(path)
        LOG.info("wrote figure %s", path)
    plt.close(fig)
    return paths


def write_table(frame: pd.DataFrame, out_dir: Path, stem: str) -> Path:
    """Export the exact aggregate table used to construct a plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.csv"
    frame.to_csv(path, index=False)
    LOG.debug("wrote plot data %s rows=%d", path, len(frame))
    return path


def scrollable_table(
    frame: pd.DataFrame, title: str, subtitle: str, out_dir: Path, stem: str,
    formats: list[str], dpi: int,
) -> list[Path]:
    """Render every table row on one tall, scrollable canvas; the companion CSV remains lossless."""
    display = frame.fillna("").astype(str)
    if display.empty:
        display = pd.DataFrame([
            {column: ("No accepted entries in the current catalogue" if index == 0 else "") for index, column in enumerate(display.columns)}
        ], columns=display.columns)
    wrapped = display.map(lambda value: "\n".join(textwrap.wrap(value, width=28)[:3]))
    height = max(4.5, 2.1 + .48 * (len(display) + 1))
    fig, ax = plt.subplots(figsize=(15.5, height))
    ax.axis("off")
    add_heading(fig, title, subtitle, top=.88)
    table = ax.table(cellText=wrapped.values, colLabels=list(wrapped.columns), cellLoc="left", colLoc="left", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d9e0e5")
        if row == 0:
            cell.set_facecolor("#2F6B9A")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f6f8")
    stamp(fig, subtitle)
    for stale in out_dir.glob(f"{stem}_p*.*"):
        stale.unlink()
    return save_figure(fig, out_dir, stem, formats, dpi)


def write_index(out_dir: Path, input_path: Path, frame: pd.DataFrame, outputs: list[Path], notes: list[str]) -> Path:
    """Write a human-readable index for a generated plot bundle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "README.md"
    lines = ["# Plot bundle", "", f"- Input: `{input_path}`", f"- {metadata_caption(frame)}", ""]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Figures", ""])
    lines.extend(f"- `{output.name}`" for output in outputs)
    lines.extend(["", "Aggregate CSVs under `data/` are the exact tables used by the figures.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("wrote plot bundle index %s", path)
    return path


def count_plot(frame: pd.DataFrame, column: str, title: str, subtitle: str, out_dir: Path, stem: str, formats: list[str], dpi: int, top_n: int | None = None) -> list[Path]:
    """Build and export a horizontal count chart plus its aggregate CSV."""
    counts = frame[column].replace("", "unknown").value_counts().rename_axis(column).reset_index(name="count")
    if top_n:
        counts = counts.head(top_n)
    write_table(counts, out_dir / "data", stem)
    fig, ax = plt.subplots()
    add_heading(fig, title, subtitle)
    if counts.empty:
        ax.axis("off")
        ax.text(.5, .5, "No entries in the current catalogue", transform=ax.transAxes, ha="center", va="center", color="#5f6c76")
    else:
        ordered = counts.iloc[::-1]
        labels = ordered[column].astype(str).str.replace("_", " ", regex=False)
        ax.barh(labels, ordered["count"], color=COLORS[0])
        ax.set_xlabel("Count")
        count_legend(ax, "Entries")
    stamp(fig, subtitle)
    return save_figure(fig, out_dir, stem, formats, dpi)
