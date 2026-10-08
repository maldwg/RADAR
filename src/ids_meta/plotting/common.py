from __future__ import annotations

import os
import textwrap
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Patch
import pandas as pd

from ids_meta.common.logging import logger

LOG = logger("plotting")

COLORS = ["#2F6B9A", "#4E9A62", "#D9902F", "#7A5FA8", "#C94F4F", "#3B8C8C", "#B85C9E", "#8C6D31", "#6C7A89"]
_SHOW_SUBCAPTIONS: ContextVar[bool] = ContextVar("show_subcaptions", default=True)

TAXONOMY_FIELD_LABELS = {
    "centralization": "IDS centralization",
    "role": "Repository role",
    "observation_scopes": "Observation scope",
    "observation_objects": "Observation object",
    "capabilities": "Capability",
    "detection_methods": "Detection method",
    "primary_system_type": "Primary security system type",
    "system_types": "Security system types",
    "paper_role": "Paper role",
    "relevance_tier": "Security relevance",
    "artifact_type": "Artifact type",
    "domain": "Domain",
    "data_form": "Data form",
    "capture_environment": "Capture environment",
    "labeling": "Label availability",
}


def taxonomy_field_label(field: object) -> str:
    """Return an explicit reader-facing label for a taxonomy field."""
    value = str(field)
    return TAXONOMY_FIELD_LABELS.get(value, value.replace("_", " ").title())


def taxonomy_overview_label(field: object) -> str:
    """Use the same plain-language field names in taxonomy overview figures."""
    return {
        "artifact_type": "Resource type", "domain": "Application area", "data_form": "Data format",
        "capture_environment": "How data was collected", "labeling": "Label availability",
    }.get(str(field), taxonomy_field_label(field))


def taxonomy_value_label(field: object, value: object) -> str:
    """Return a readable taxonomy value while preserving security-system acronyms."""
    raw = str(value)
    if str(field) == "role" and raw == "antivirus":
        return "antivirus / file scanner"
    if str(field) in {"primary_system_type", "system_types"}:
        return {
            "provids": "ProvIDS",
            "unknown": "Unknown",
        }.get(raw.casefold(), raw.upper())
    return raw.replace("_", " ")


def configure_style(config: dict[str, object] | None = None) -> None:
    """Apply the shared non-interactive matplotlib theme."""
    show_subcaptions = (config or {}).get("show_subcaptions", True)
    if not isinstance(show_subcaptions, bool):
        raise ValueError("plots.show_subcaptions must be true or false")
    _SHOW_SUBCAPTIONS.set(show_subcaptions)
    plt.rcParams.update({
        "figure.figsize": (9, 5.4), "figure.dpi": 120, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "axes.axisbelow": True, "grid.alpha": .22, "font.size": 10,
        "axes.titlelocation": "center",
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


def subcaption(target: plt.Figure | plt.Axes, x: float, y: float, text: str, **kwargs: Any) -> Text | None:
    """Add optional explanatory text without affecting titles, legends, or data labels."""
    if not _SHOW_SUBCAPTIONS.get():
        return None
    artist = target.text(x, y, text, **kwargs)
    artist.set_gid("ids-meta-subcaption")
    return artist


def stamp(fig: plt.Figure, caption: str) -> None:
    """Add a small provenance caption at the bottom of a figure."""
    subcaption(fig, .01, .005, caption, fontsize=7.5, color="#5f6c76", ha="left")


def add_heading(fig: plt.Figure, title: str, subtitle: str, *, top: float | None = None) -> None:
    """Reserve fixed physical margins for a title and subtitle on short or tall figures."""
    height = max(1.0, float(fig.get_figheight()))
    title_y = 1.0 - .12 / height
    subtitle_y = 1.0 - .48 / height
    physical_top = 1.0 - (.90 if _SHOW_SUBCAPTIONS.get() else .60) / height
    axes_top = max(top, physical_top) if top is not None else physical_top
    axes_bottom = max(.10, .95 / height)
    fig.subplots_adjust(top=axes_top, bottom=axes_bottom)
    fig.suptitle(title, x=.5, y=title_y, ha="center", va="top", fontsize=14, fontweight="bold")
    subcaption(fig, .08, subtitle_y, textwrap.fill(subtitle, width=115), ha="left", va="top", fontsize=8.5, color="#5f6c76")


def count_legend(ax: plt.Axes, label: str, color: str = COLORS[0]) -> None:
    """Add a compact single-series legend to a count chart."""
    ax.legend(handles=[Patch(facecolor=color, label=label)], loc="best", frameon=False, fontsize=8)


def percentage_bars(
    labels: list[str], values: list[float | None], *, title: str, xlabel: str,
    caption: str, colors: list[str] | None = None,
) -> plt.Figure:
    """Render annotated horizontal rates, preserving zero and unavailable values."""
    fig, ax = plt.subplots(figsize=(12.5, max(4.5, .70 * len(labels) + 1.7)))
    fig.subplots_adjust(left=.32, right=.97, top=.85, bottom=.23)
    if not labels:
        ax.axis("off")
        ax.text(.5, .5, "No rated comparisons", transform=ax.transAxes, ha="center", va="center")
    else:
        for index, value in enumerate(values):
            if value is None or pd.isna(value):
                ax.text(1.5, index, "N/A", va="center", fontsize=13, color="#5f6c76")
                continue
            percent = float(value) * 100
            color = colors[index] if colors else "#286f9b"
            ax.barh(index, percent, height=.65, color=color)
            inside = percent >= 90
            ax.text(
                percent - 1.5 if inside else percent + 1.5, index, f"{percent:.1f}%",
                ha="right" if inside else "left", va="center", fontsize=14,
                fontweight="bold", color="white" if inside else "#24323d",
            )
        ax.set_yticks(range(len(labels)), labels=[textwrap.fill(label, 38) for label in labels])
        ax.set_ylim(len(labels) - .4, -.6)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(xlabel)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.grid(axis="x", alpha=.16)
    ax.set_title(title, loc="center", fontweight="bold", fontsize=14, pad=18)
    subcaption(fig, .32, .035, textwrap.fill(caption, 110), fontsize=8.5, color="#5f6c76", va="bottom")
    return fig


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


def score_decision_plot(
    frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int, stem: str, source_label: str,
) -> list[Path]:
    """Compare each numeric keyword-prefilter score with the exact final LLM decision."""
    view = frame.copy()
    if "prefilter_score" in view:
        view["keyword_score"] = pd.to_numeric(view["prefilter_score"], errors="coerce")
    else:
        view["keyword_score"] = float("nan")
    if "decision" in view:
        view["llm_decision"] = view["decision"].fillna("").astype(str).str.strip().str.casefold().replace("", "unknown")
    else:
        view["llm_decision"] = "unknown"

    preferred_order = ["accept", "accept_adjacent", "include", "exclude", "reject", "uncertain", "error", "dry_run", "unknown"]
    observed = set(view["llm_decision"])
    decisions = [value for value in preferred_order if value in observed]
    decisions.extend(sorted(observed - set(decisions)))

    summary_rows: list[dict[str, object]] = []
    for decision in decisions:
        rows = view[view["llm_decision"] == decision]
        scores = rows["keyword_score"].dropna()
        if scores.empty:
            q1 = q3 = iqr = lower_fence = upper_fence = lower_whisker = upper_whisker = ""
            outlier_count = 0
        else:
            q1 = scores.quantile(.25)
            q3 = scores.quantile(.75)
            iqr = q3 - q1
            lower_fence = q1 - 1.5 * iqr
            upper_fence = q3 + 1.5 * iqr
            within_whiskers = scores[scores.between(lower_fence, upper_fence)]
            lower_whisker = within_whiskers.min()
            upper_whisker = within_whiskers.max()
            outlier_count = int(((scores < lower_fence) | (scores > upper_fence)).sum())
        summary_rows.append({
            "llm_decision": decision,
            "count_total": len(rows),
            "count_with_score": len(scores),
            "score_min": scores.min() if not scores.empty else "",
            "score_q1": q1,
            "score_median": scores.median() if not scores.empty else "",
            "score_mean": scores.mean() if not scores.empty else "",
            "score_q3": q3,
            "score_max": scores.max() if not scores.empty else "",
            "score_iqr": iqr,
            "lower_outlier_fence": lower_fence,
            "upper_outlier_fence": upper_fence,
            "lower_whisker": lower_whisker,
            "upper_whisker": upper_whisker,
            "outlier_count": outlier_count,
        })
    summary = pd.DataFrame(summary_rows, columns=[
        "llm_decision", "count_total", "count_with_score", "score_min", "score_q1",
        "score_median", "score_mean", "score_q3", "score_max", "score_iqr",
        "lower_outlier_fence", "upper_outlier_fence", "lower_whisker", "upper_whisker",
        "outlier_count",
    ])
    write_table(summary, out_dir / "data", stem)

    identity_columns = [value for value in ("github_repo_id", "full_name", "openalex_id", "title") if value in view]
    audit_columns = identity_columns + [
        value for value in ("prefilter_score", "keyword_score", "prefilter_band", "decision", "llm_decision", "role", "paper_role", "primary_reason")
        if value in view and value not in identity_columns
    ]
    write_table(view[audit_columns].copy(), out_dir / "data", stem + "_records")

    scored = view.dropna(subset=["keyword_score"])
    plotted_decisions = [value for value in decisions if not scored[scored["llm_decision"] == value].empty]
    height = max(4.8, 2.8 + .7 * len(plotted_decisions))
    fig, ax = plt.subplots(figsize=(10.8, height))
    subtitle = metadata_caption(
        frame,
        f"{source_label}; dots are all classified entities; box = middle 50%, line = median; "
        "whiskers end at the furthest score within 1.5 x IQR, so dots beyond them are valid outliers",
    )
    add_heading(fig, "Keyword-score distribution by final classification decision", subtitle)
    if not plotted_decisions:
        ax.axis("off")
        ax.text(.5, .5, "No rows contain both a numeric keyword score and a final decision", transform=ax.transAxes, ha="center", va="center", color="#5f6c76")
    else:
        distributions = [scored.loc[scored["llm_decision"] == decision, "keyword_score"].sort_values().tolist() for decision in plotted_decisions]
        positions = list(range(len(plotted_decisions)))
        boxes = ax.boxplot(distributions, orientation="horizontal", positions=positions, widths=.48, patch_artist=True, showfliers=False)
        for index, box in enumerate(boxes["boxes"]):
            box.set_facecolor(COLORS[index % len(COLORS)])
            box.set_alpha(.20)
            box.set_edgecolor(COLORS[index % len(COLORS)])
        for index, scores in enumerate(distributions):
            offsets = [((point % 9) - 4) * .025 for point in range(len(scores))]
            ax.scatter(scores, [index + offset for offset in offsets], s=28, alpha=.68, color=COLORS[index % len(COLORS)], edgecolors="white", linewidths=.35, zorder=3)
        counts = {row["llm_decision"]: int(row["count_with_score"]) for row in summary_rows}
        ax.set_yticks(positions, [f"{decision.replace('_', ' ')} (n={counts[decision]})" for decision in plotted_decisions])
        ax.set_xlabel("Deterministic keyword-prefilter score")
        ax.set_ylabel("Final LLM decision")
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
    stamp(fig, subtitle)
    return save_figure(fig, out_dir, stem, formats, dpi)


def evidence_escalation_plot(
    frame: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int, stem: str, source_label: str,
) -> list[Path]:
    """Plot evidence-expansion demand, retrieval, improvement, and resolution rates."""
    view = frame.copy()
    flag_names = (
        "evidence_expansion_required", "evidence_expansion_attempted", "evidence_expansion_changed",
        "evidence_expansion_improved", "evidence_expansion_resolved",
    )
    audit_available = all(name in view for name in flag_names)
    for name in flag_names:
        values = view.get(name, pd.Series("false", index=view.index)).fillna("").astype(str).str.casefold()
        view[name + "_flag"] = values.isin({"true", "1", "yes"})
    total = len(view)
    required = int(view["evidence_expansion_required_flag"].sum())
    metrics = [
        ("Required additional evidence", int(view["evidence_expansion_required_flag"].sum()), total, "all classified entities"),
        ("Retrieval attempted", int((view["evidence_expansion_required_flag"] & view["evidence_expansion_attempted_flag"]).sum()), required, "entities requiring evidence"),
        ("New evidence found", int((view["evidence_expansion_required_flag"] & view["evidence_expansion_changed_flag"]).sum()), required, "entities requiring evidence"),
        ("Outcome / agreement improved", int((view["evidence_expansion_required_flag"] & view["evidence_expansion_improved_flag"]).sum()), required, "entities requiring evidence"),
        ("Became definitive", int((view["evidence_expansion_required_flag"] & view["evidence_expansion_resolved_flag"]).sum()), required, "entities requiring evidence"),
    ]
    summary = pd.DataFrame([
        {
            "metric": label, "numerator": numerator, "denominator": denominator,
            "percent": (100 * numerator / denominator) if denominator else 0.0,
            "denominator_scope": scope,
            "audit_available": audit_available,
        }
        for label, numerator, denominator, scope in metrics
    ])
    write_table(summary, out_dir / "data", stem)
    record_columns = [
        value for value in (
            "github_repo_id", "full_name", "openalex_id", "title", "decision",
            "evidence_expansion_required", "evidence_expansion_attempted", "evidence_expansion_changed",
            "evidence_expansion_improved", "evidence_expansion_resolved", "evidence_initial_decision",
            "evidence_initial_vote_status", "evidence_initial_agreement", "evidence_initial_votes",
            "evidence_initial_uncertain_votes",
            "ensemble_vote_status", "ensemble_agreement",
        ) if value in view
    ]
    write_table(view[record_columns].copy(), out_dir / "data", stem + "_records")

    plotted = summary.iloc[::-1]
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    add_heading(
        fig, "Additional-evidence effectiveness",
        metadata_caption(frame, f"{source_label} classifications; percentages use the denominator printed on each bar"),
        top=.84,
    )
    if not audit_available:
        ax.axis("off")
        ax.text(
            .5, .5, "Evidence-escalation audit unavailable\nRerun classification with the current pipeline version.",
            transform=ax.transAxes, ha="center", va="center", color="#5f6c76",
        )
        return save_figure(fig, out_dir, stem, formats, dpi)
    bars = ax.barh(plotted["metric"], plotted["percent"], color=[COLORS[1], COLORS[1], COLORS[0], COLORS[2], COLORS[2]])
    ax.set_xlim(0, 112)
    ax.set_xlabel("Entities (%)")
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", visible=False)
    for bar, (_, row) in zip(bars, plotted.iterrows()):
        ax.text(
            min(109, float(row["percent"]) + 1.5), bar.get_y() + bar.get_height() / 2,
            f"{row['percent']:.1f}% ({int(row['numerator'])}/{int(row['denominator'])})",
            va="center", fontsize=8, color="#43505C",
        )
    return save_figure(fig, out_dir, stem, formats, dpi)
