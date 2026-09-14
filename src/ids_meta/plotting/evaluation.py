from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ids_meta.common.logging import logger
from ids_meta.plotting.common import COLORS, add_heading, configure_style, save_figure, stamp, write_table


LOG = logger("plotting.evaluation")
VIEW_LABELS = {"operational": "Operational", "strict": "Strict", "selective": "Selective"}
STATUS_LABELS = {
    "classified": "Definitive", "uncertain": "Uncertain", "error": "Error",
    "not_classified": "Not classified", "not_retrieved": "Not retrieved",
    "survey_reference_classified_nonrelease": "Survey reference classified as non-release",
    "survey_reference_classification_uncertain": "Survey reference classification uncertain",
    "survey_reference_classification_error": "Survey reference classification error",
    "survey_reference_retrieved_not_classified": "Survey reference retrieved, not classified",
    "survey_reference_not_retrieved": "Survey reference not retrieved",
    "survey_reference_not_observable": "Survey-reference retrieval unavailable",
    "survey_reference_unresolved": "Survey reference has no OpenAlex ID",
    "unavailable": "Unavailable",
}


def _annotate_bars(ax: plt.Axes, bars: Any, *, percent: bool = True) -> None:
    """Place compact values above vertical bars without obscuring the axes."""
    for bar in bars:
        value = float(bar.get_height())
        label = f"{value:.0%}" if percent else f"{value:g}"
        ax.text(bar.get_x() + bar.get_width() / 2, value + (.025 if percent else .08), label, ha="center", va="bottom", fontsize=7.5)


def _decision_performance(metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot quality and error rates for operational, strict, and selective views."""
    positive_only = metrics.get("benchmark_scope") == "positive_reference_set"
    quality = [("DR", "Dataset recall")] if positive_only else [
        ("DR", "Detection rate"), ("PREC", "Precision"), ("ACC", "Accuracy"), ("f1", "F1"),
    ]
    errors = [("FNR", "Miss rate")] if positive_only else [
        ("FPR", "False-positive rate"), ("FNR", "False-negative rate"),
    ]
    rows: list[dict[str, object]] = []
    for view in VIEW_LABELS:
        values = metrics.get(view, {})
        for key, label in quality + errors:
            rows.append({"view": VIEW_LABELS[view], "metric": label, "metric_key": key, "value": float(values.get(key, 0.0))})
    table = pd.DataFrame(rows)
    write_table(table, out_dir / "plot_data", "01_decision_performance")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), gridspec_kw={"width_ratios": [2, 1]})
    add_heading(
        fig, "Dataset coverage performance" if positive_only else "Rated decision performance",
        f"n={metrics.get('rated_entities', 0)} positive reference datasets; FPR requires a separate negative set"
        if positive_only else f"n={metrics.get('rated_entities', 0)} rated entities; higher is better on the left, lower is better on the right",
        top=.83,
    )
    colors = COLORS[:3]
    quality_frame = table[table["metric_key"].isin([key for key, _label in quality])]
    error_frame = table[table["metric_key"].isin([key for key, _label in errors])]
    for ax, frame, title in ((axes[0], quality_frame, "Quality metrics ↑"), (axes[1], error_frame, "Error rates ↓")):
        metrics_order = list(dict.fromkeys(frame["metric"]))
        x_values = list(range(len(metrics_order)))
        width = .23
        for index, view in enumerate(VIEW_LABELS.values()):
            values = frame[frame["view"] == view].set_index("metric")["value"].reindex(metrics_order).fillna(0)
            positions = [value + (index - 1) * width for value in x_values]
            bars = ax.bar(positions, values, width=width, label=view, color=colors[index])
            _annotate_bars(ax, bars)
        ax.set_xticks(x_values, labels=metrics_order, rotation=18, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Rate")
        ax.set_title(title, loc="left", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .875), ncol=3, frameon=False, fontsize=8)
    stamp(fig, "Operational includes all outcomes; strict penalizes abstentions; selective uses definitive predictions only.")
    return save_figure(fig, out_dir, "01_decision_performance", formats, dpi)


def _decision_confusion(input_dir: Path, metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot the exact expected-to-actual disposition confusion matrix."""
    source = input_dir / "decision_confusion.csv"
    frame = pd.read_csv(source, dtype={"expected": str, "actual": str, "count": int}, keep_default_na=False)
    preferred = ["include", "exclude", "uncertain", "unavailable"]
    expected_values = [value for value in preferred if value in set(frame.get("expected", []))]
    expected_values += sorted(set(frame.get("expected", [])) - set(expected_values))
    actual_values = [value for value in preferred if value in set(frame.get("actual", []))]
    actual_values += sorted(set(frame.get("actual", [])) - set(actual_values))
    matrix = frame.pivot_table(index="expected", columns="actual", values="count", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(index=expected_values, columns=actual_values, fill_value=0)
    write_table(matrix.reset_index(), out_dir / "plot_data", "02_decision_confusion")

    fig, ax = plt.subplots(figsize=(max(7, 1.25 * max(1, len(actual_values))), max(4.8, 1.05 * max(1, len(expected_values)))))
    add_heading(
        fig, "Expected versus actual decisions",
        f"n={metrics.get('rated_entities', 0)} rated entities; rows are configured ratings and columns are pipeline results",
        top=.80,
    )
    if matrix.empty:
        ax.axis("off")
        ax.text(.5, .5, "No rated decisions", transform=ax.transAxes, ha="center", va="center")
    else:
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=max(1.0, float(values.max())))
        ax.set_xticks(range(len(actual_values)), labels=[value.replace("_", " ").title() for value in actual_values], rotation=25, ha="right")
        ax.set_yticks(range(len(expected_values)), labels=[value.replace("_", " ").title() for value in expected_values])
        ax.set_xlabel("Actual pipeline disposition")
        ax.set_ylabel("Expected rated disposition")
        threshold = float(values.max()) * .55
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = int(values[row_index, column_index])
                ax.text(column_index, row_index, str(value), ha="center", va="center", color="white" if value > threshold else "#253746", fontweight="bold")
        fig.colorbar(image, ax=ax, label="Rated entities")
    stamp(fig, "Uncertain and unavailable outcomes remain visible rather than being collapsed into binary decisions.")
    return save_figure(fig, out_dir, "02_decision_confusion", formats, dpi)


def _field_accuracy(input_dir: Path, metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot exact metadata accuracy and readable multi-label F1 by field."""
    source = input_dir / "field_metrics.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    for column in ("comparisons", "exact_matches", "exact_accuracy", "label_f1"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    plot_frame = frame[frame["comparisons"] > 0].copy()
    plot_frame["field_label"] = plot_frame["field"].astype(str).str.replace("_", " ").str.title()
    write_table(plot_frame.drop(columns=["field_label"]), out_dir / "plot_data", "03_field_accuracy")

    fig, ax = plt.subplots(figsize=(12.5, max(5.2, .72 * max(1, len(plot_frame)) + 2.0)))
    add_heading(
        fig, "Classification metadata accuracy",
        f"{metrics.get('classification_field_comparisons', 0)} rated field comparisons; exact agreement is shown for every configured field",
        top=.83,
    )
    if plot_frame.empty:
        ax.axis("off")
        ax.text(.5, .5, "Classifier does not emit rated metadata fields", transform=ax.transAxes, ha="center", va="center")
    else:
        ordered = plot_frame.iloc[::-1].reset_index(drop=True)
        positions = list(range(len(ordered)))
        multilabel = ordered["label_f1"].notna()
        exact_positions = [position + .17 if multilabel.iloc[index] else position for index, position in enumerate(positions)]
        exact_heights = [.28 if multilabel.iloc[index] else .52 for index in ordered.index]
        exact_bars = ax.barh(
            exact_positions, ordered["exact_accuracy"].fillna(0), height=exact_heights,
            color=COLORS[0], label="Exact accuracy",
        )
        if multilabel.any():
            f1_positions = [positions[index] - .17 for index in ordered.index[multilabel]]
            f1_bars = ax.barh(
                f1_positions, ordered.loc[multilabel, "label_f1"], height=.28,
                color=COLORS[2], label="Label F1",
            )
            for bar in f1_bars:
                value = float(bar.get_width())
                ax.text(
                    value + .015, bar.get_y() + bar.get_height() / 2, f"{value:.0%}",
                    va="center", fontsize=8, fontweight="bold", color="#253746",
                )
        ax.set_yticks(positions, labels=ordered["field_label"])
        ax.set_xlim(0, 1.25)
        ax.set_xlabel("Score")
        for bar in exact_bars:
            value = float(bar.get_width())
            ax.text(value + .015, bar.get_y() + bar.get_height() / 2, f"{value:.0%}", va="center", fontsize=8)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    stamp(fig, "Multi-label fields additionally show label-level F1; percentages are printed beside their bars.")
    return save_figure(fig, out_dir, "03_field_accuracy", formats, dpi)


def _coverage(metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot pipeline coverage percentages and mutually exclusive entity statuses."""
    artifact_profile = metrics.get("evaluation_unit") == "artifact"
    coverage_specs = (
        [
            ("classification_coverage", "Dataset identified"),
            ("definitive_coverage", "Definitive artifact decision"),
            ("abstention_rate", "Classifier abstention"),
            ("unresolved_artifact_rate", "Dataset unresolved"),
        ]
        if artifact_profile else [
            ("retrieval_coverage", "Retrieved"), ("classification_coverage", "Classification present"),
            ("definitive_coverage", "Definitive decision"), ("abstention_rate", "Abstention"),
        ]
    )
    coverage_rows = [
        {"measure": label, "key": key, "value": metrics.get(key)}
        for key, label in coverage_specs if metrics.get(key) is not None
    ]
    statuses = metrics.get("status_counts", {})
    status_rows = [
        {"status": STATUS_LABELS.get(str(key), str(key).replace("_", " ").title()), "count": int(value)}
        for key, value in statuses.items()
    ]
    table = pd.DataFrame(coverage_rows)
    write_table(table, out_dir / "plot_data", "04_coverage_rates")
    write_table(pd.DataFrame(status_rows, columns=["status", "count"]), out_dir / "plot_data", "04_status_counts")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    add_heading(
        fig, "Evaluation coverage and abstentions",
        f"n={metrics.get('rated_entities', 0)} rated entities; actual output also contains {metrics.get('unrated_actual_entities', 0)} unrated entities",
        top=.82,
    )
    if table.empty:
        axes[0].axis("off")
    else:
        bars = axes[0].bar(table["measure"], pd.to_numeric(table["value"]), color=[COLORS[1], COLORS[0], COLORS[3], COLORS[4]][:len(table)])
        axes[0].set_ylim(0, 1.12)
        axes[0].set_ylabel("Share of rated entities")
        axes[0].tick_params(axis="x", rotation=22)
        _annotate_bars(axes[0], bars)
    axes[0].set_title("Pipeline coverage", loc="left", fontweight="bold")
    status_frame = pd.DataFrame(status_rows)
    if status_frame.empty:
        axes[1].axis("off")
        axes[1].text(.5, .5, "No rated entities", transform=axes[1].transAxes, ha="center", va="center")
    else:
        status_frame = status_frame.sort_values(["count", "status"], kind="stable")
        bars = axes[1].barh(status_frame["status"], status_frame["count"], color=COLORS[0])
        axes[1].set_xlabel("Rated entities")
        for bar in bars:
            axes[1].text(bar.get_width() + .06, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va="center", fontsize=8)
    axes[1].set_title("Final status", loc="left", fontweight="bold")
    stamp(
        fig,
        "Survey-reference diagnostics distinguish name matches, accepted release fallbacks, retrieval gaps, and unresolved references."
        if artifact_profile else "Candidate input distinguishes not retrieved from retrieved-but-not-classified entities.",
    )
    return save_figure(fig, out_dir, "04_coverage_and_status", formats, dpi)


def _append_readme(input_dir: Path, outputs: list[Path]) -> None:
    """Add generated figure links and interpretation guidance to the evaluation README."""
    path = input_dir / "README.md"
    current = path.read_text(encoding="utf-8") if path.is_file() else "# Classification evaluation\n"
    marker = "\n## Figures\n"
    if marker in current:
        current = current.split(marker, 1)[0].rstrip() + "\n"
    lines = ["", "## Figures", ""]
    lines.extend(f"- `{output.name}`" for output in outputs)
    lines.extend(["", "Companion aggregate tables are stored under `plot_data/`. Metadata accuracy is shown separately from binary inclusion/exclusion metrics.", ""])
    path.write_text(current.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


def build(input_dir: Path, config: dict[str, object]) -> list[Path]:
    """Build the complete evaluation visualization bundle from persisted audit artifacts."""
    metrics_path = input_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"evaluation metrics do not exist: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    configure_style()
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    outputs: list[Path] = []
    outputs += _decision_performance(metrics, input_dir, formats, dpi)
    outputs += _decision_confusion(input_dir, metrics, input_dir, formats, dpi)
    outputs += _field_accuracy(input_dir, metrics, input_dir, formats, dpi)
    outputs += _coverage(metrics, input_dir, formats, dpi)
    _append_readme(input_dir, outputs)
    LOG.info("evaluation plot bundle complete source=%s files=%d output=%s", metrics.get("source", "unknown"), len(outputs), input_dir)
    return outputs
