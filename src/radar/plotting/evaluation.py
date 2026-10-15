from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from radar.common.logging import logger
from radar.plotting.common import COLORS, add_heading, configure_style, percentage_bars, save_figure, stamp, taxonomy_field_label, taxonomy_overview_label, taxonomy_value_label, write_table
from radar.common.provenance import audited_bundle
from radar.plotting.recovery import recovered_taxonomy_table
from radar.plotting.uncertainty import reference_uncertainty


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


def _decision_performance(metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot quality and error rates for operational, strict, and selective views."""
    positive_only = metrics.get("benchmark_scope") == "positive_reference_set"
    quality = [("DR", "Dataset recall")] if positive_only else [
        ("DR", "Detection rate"), ("PREC", "Precision"), ("ACC", "Accuracy"), ("f1", "F1"),
    ]
    errors = [("FNR", "Miss rate")] if positive_only else [
        ("FPR", "False-positive rate"), ("FDR", "False-discovery rate"),
        ("FNR", "False-negative rate"),
    ]
    rows: list[dict[str, object]] = []
    for view in VIEW_LABELS:
        values = metrics.get(view, {})
        for key, label in quality + errors:
            rows.append({"view": VIEW_LABELS[view], "metric": label, "metric_key": key, "value": float(values.get(key, 0.0))})
    table = pd.DataFrame(rows)
    write_table(table, out_dir / "data", "01_decision_views")

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
            ax.bar(positions, values, width=width, label=view, color=colors[index])
        ax.set_xticks(x_values, labels=metrics_order, rotation=18, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Rate")
        ax.set_title(title, loc="center", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .875), ncol=3, frameon=False, fontsize=8)
    stamp(fig, "Operational includes all outcomes; strict penalizes abstentions; selective uses definitive predictions only.")
    return save_figure(fig, out_dir, "01_decision_views", formats, dpi)


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
    write_table(matrix.reset_index(), out_dir / "data", "02_decision_confusion")

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
    plot_frame["field_label"] = plot_frame["field"].map(taxonomy_field_label)
    write_table(plot_frame.drop(columns=["field_label"]), out_dir / "data", "03_taxonomy_fields")

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
        ax.barh(
            exact_positions, ordered["exact_accuracy"].fillna(0), height=exact_heights,
            color=COLORS[0], label="Exact accuracy",
        )
        if multilabel.any():
            f1_positions = [positions[index] - .17 for index in ordered.index[multilabel]]
            ax.barh(
                f1_positions, ordered.loc[multilabel, "label_f1"], height=.28,
                color=COLORS[2], label="Label F1",
            )
        ax.set_yticks(positions, labels=ordered["field_label"])
        ax.set_xlim(0, 1.25)
        ax.set_xlabel("Score")
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    stamp(fig, "Multi-label fields additionally show label-level F1; exact values are in the companion CSV.")
    return save_figure(fig, out_dir, "03_taxonomy_fields", formats, dpi)


def _taxonomy_accuracy_overview(
    input_dir: Path, metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int,
) -> list[Path]:
    """Summarize exact field matches against each field's full rated denominator."""
    frame = pd.read_csv(input_dir / "field_metrics.csv", keep_default_na=False)
    for column in ("comparisons", "exact_matches", "exact_accuracy"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    supported = frame["supported"].astype(str).str.casefold().eq("true")
    estimable = supported & frame["comparisons"].gt(0)
    frame["value"] = (frame["exact_matches"] / frame["comparisons"].where(estimable)).where(estimable)
    frame["percentage"] = frame["value"] * 100
    write_table(frame, out_dir / "data", "06_taxonomy_accuracy")
    labels = [taxonomy_overview_label(field) for field in frame["field"]]
    counts = sorted(set(int(value) for value in frame.loc[estimable, "comparisons"]))
    source = metrics.get("source_display_name") or str(metrics.get("source", "")).title()
    denominators = (
        f"{counts[0]:,} rated entities per field" if len(counts) == 1
        else "Rated denominator varies by field; see companion CSV"
    )
    fig = percentage_bars(
        labels, frame["value"].tolist(),
        title=f"{source} taxonomy accuracy",
        xlabel="Correct labels across the reference inventory (%)",
        caption=f"{denominators}. Missing and unresolved results remain in each rated denominator. Multi-label fields require an exact set match; N/A means no supported comparisons.",
    )
    return save_figure(fig, out_dir, "06_taxonomy_accuracy", formats, dpi)


def _recovered_taxonomy_accuracy(
    input_dir: Path, metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int,
) -> list[Path]:
    """Show labeling accuracy conditional on successful reference recovery."""
    frame = recovered_taxonomy_table(input_dir)
    if frame is None or frame.empty:
        return []
    write_table(frame, out_dir / "data", "07_recovered_taxonomy_accuracy")
    recovered = int(frame.iloc[0]["recovered"])
    expected = int(frame.iloc[0]["expected_positive_entities"])
    source = metrics.get("source_display_name") or str(metrics.get("source", "")).title()
    labels = [
        f"{taxonomy_overview_label(row['field'])}\n(n={row['comparisons']:,})"
        for row in frame.to_dict("records")
    ]
    fig = percentage_bars(
        labels, frame["value"].tolist(),
        title=f"{source} taxonomy accuracy among recovered items",
        xlabel=f"Correct labels among recovered reference items (%) · recovered {recovered:,}/{expected:,}",
        caption="Only expected relevant items accepted by the classifier are evaluated. n=field-rated recovered items; missing field predictions are incorrect. Multi-label fields require an exact set match. N/A=no recovered field comparisons or unsupported field.",
    )
    return save_figure(fig, out_dir, "07_recovered_taxonomy_accuracy", formats, dpi)


def _reference_uncertainty(
    input_dir: Path, metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int,
) -> list[Path]:
    """Show uncertainty conditional on an attempted reference classification."""
    counts = reference_uncertainty(input_dir)
    if counts is None:
        return []
    write_table(pd.DataFrame([counts]), out_dir / "data", "08_reference_uncertainty")
    source = metrics.get("source_display_name") or str(metrics.get("source", "")).title()
    fig = percentage_bars(
        [f"Uncertain ({counts['uncertain']:,}/{counts['attempted']:,} attempts)"], [counts["value"]],
        title=f"{source} classifier uncertainty on reference items",
        xlabel="Uncertain decisions among attempted reference classifications (%)",
        caption="All rated reference items; missing retrieval/classification is excluded. Errors stay in the attempted denominator but do not count as uncertain. OpenAlex uses a classified cited paper when no artifact match exists. N/A=no attempts.",
        colors=[COLORS[2]],
    )
    return save_figure(fig, out_dir, "08_reference_uncertainty", formats, dpi)


def _field_categories(input_dir: Path, metrics: dict[str, Any], out_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    """Plot precision, recall, and F1 for every rated taxonomy category."""
    source = input_dir / "field_label_metrics.csv"
    frame = pd.read_csv(source, keep_default_na=False)
    for column in ("precision", "recall", "f1"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame["supported"].astype(str).str.casefold() == "true"].copy()
    field_order = list(dict.fromkeys(frame["field"].astype(str)))
    order = {field: index for index, field in enumerate(field_order)}
    frame["field_order"] = frame["field"].astype(str).map(order)
    frame = frame.sort_values(["field_order", "label"], kind="stable").reset_index(drop=True)
    frame["category_label"] = frame.apply(
        lambda row: f"{taxonomy_field_label(row['field'])} — {taxonomy_value_label(row['field'], row['label'])}",
        axis=1,
    )
    write_table(
        frame.drop(columns=["field_order", "category_label"]),
        out_dir / "data", "04_taxonomy_categories",
    )

    fig, ax = plt.subplots(figsize=(11.5, max(6.0, .34 * max(1, len(frame)) + 2.6)))
    add_heading(
        fig, "Taxonomy performance by category",
        f"{metrics.get('rated_entities', 0)} rated entities; each row is one controlled-vocabulary value",
    )
    if frame.empty:
        ax.axis("off")
        ax.text(.5, .5, "No supported taxonomy categories", transform=ax.transAxes, ha="center", va="center")
    else:
        values = frame[["precision", "recall", "f1"]].to_numpy(dtype=float)
        image = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), labels=["Precision", "Recall", "F1"])
        ax.set_yticks(range(len(frame)), labels=frame["category_label"], fontsize=7)
        boundaries = [index - .5 for index in range(1, len(frame)) if frame.loc[index, "field"] != frame.loc[index - 1, "field"]]
        for boundary in boundaries:
            ax.axhline(boundary, color="white", linewidth=2.0)
        fig.colorbar(image, ax=ax, label="One-vs-rest score")
        fig.subplots_adjust(left=.38)
    stamp(fig, "Rows are grouped by taxonomy dimension; exact TP/FP/FN and scores are in the companion CSV.")
    return save_figure(fig, out_dir, "04_taxonomy_categories", formats, dpi)


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
        {"status": STATUS_LABELS.get(str(key), str(key).replace("_", " ").title()), "status_key": str(key), "count": int(value)}
        for key, value in statuses.items()
    ]
    table = pd.DataFrame(coverage_rows)
    write_table(table, out_dir / "data", "05_coverage_rates")
    write_table(pd.DataFrame(status_rows, columns=["status", "status_key", "count"]), out_dir / "data", "05_status_counts")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.25, 1]})
    add_heading(
        fig, "Evaluation coverage and abstentions",
        f"n={metrics.get('rated_entities', 0)} rated entities; actual output also contains {metrics.get('unrated_actual_entities', 0)} unrated entities",
        top=.82,
    )
    # Reserve room between panels for the wrapped status labels.
    fig.subplots_adjust(left=.09, right=.97, wspace=.85)
    if table.empty:
        axes[0].axis("off")
    else:
        axes[0].bar(table["measure"], pd.to_numeric(table["value"]), color=[COLORS[1], COLORS[0], COLORS[3], COLORS[4]][:len(table)])
        axes[0].set_ylim(0, 1.12)
        axes[0].set_ylabel("Share of rated entities")
        axes[0].set_xticks(range(len(table)), labels=[textwrap.fill(label, 18) for label in table["measure"]])
        axes[0].tick_params(axis="x", labelsize=9)
    axes[0].set_title("Pipeline coverage", loc="center", fontweight="bold")
    status_frame = pd.DataFrame(status_rows)
    if status_frame.empty:
        axes[1].axis("off")
        axes[1].text(.5, .5, "No rated entities", transform=axes[1].transAxes, ha="center", va="center")
    else:
        status_frame = status_frame.sort_values(["count", "status"], kind="stable")
        positions = range(len(status_frame))
        bars = axes[1].barh(positions, status_frame["count"], color=COLORS[0])
        axes[1].set_yticks(positions, labels=[textwrap.fill(label, 26) for label in status_frame["status"]])
        axes[1].tick_params(axis="y", labelsize=9)
        axes[1].set_xlabel("Rated entities")
        axes[1].set_xlim(0, max(1, float(status_frame["count"].max()) * 1.15))
        for bar in bars:
            axes[1].text(bar.get_width() + .06, bar.get_y() + bar.get_height() / 2, str(int(bar.get_width())), va="center", fontsize=8)
    axes[1].set_title("Final status", loc="center", fontweight="bold")
    stamp(
        fig,
        "Survey-reference diagnostics distinguish name matches, accepted release fallbacks, retrieval gaps, and unresolved references."
        if artifact_profile else "Candidate input distinguishes not retrieved from retrieved-but-not-classified entities.",
    )
    return save_figure(fig, out_dir, "05_pipeline_coverage", formats, dpi)


def _append_readme(input_dir: Path, outputs: list[Path]) -> None:
    """Add generated figure links and interpretation guidance to the evaluation README."""
    path = input_dir / "README.md"
    current = path.read_text(encoding="utf-8") if path.is_file() else "# Classification evaluation\n"
    marker = "\n## Figures\n"
    if marker in current:
        current = current.split(marker, 1)[0].rstrip() + "\n"
    lines = ["", "## Figures", ""]
    lines.extend(
        f"- `{output.relative_to(input_dir)}`" for output in outputs
        if output.suffix == ".svg"
    )
    lines.extend(["", "Each listed SVG has a matching PDF for publication; they are two formats of the same figure. These are single-classifier diagnostics. Use the source-level `comparison/` directory for cross-classifier reporting. Companion aggregate tables are stored under `figures/data/`.", "", "`06_taxonomy_accuracy` is a horizontal overview of exact field accuracy with percentage labels. Each field uses its exported rated comparison count, retaining unresolved results; unsupported or empty comparisons are N/A. `03_taxonomy_fields` additionally reports multi-label F1.", "", "`07_recovered_taxonomy_accuracy` evaluates taxonomy fields only among expected relevant reference items matched to accepted outputs (`accept` or `accept_adjacent`). Its denominator excludes missed references and negative controls. Each field displays its own rated recovered count, and the axis reports recovery against all reference positives. Missing field predictions count as incorrect; multi-label fields require an exact set match. N/A means an unsupported field or no recovered field comparisons. Read this conditional quality alongside reference recovery: different classifiers may recover different subsets. Matching and field checks reuse `instances.csv`, including the OpenAlex cited-paper fallback. Older bundles without that audit cannot render this view.", ""])
    lines.extend(["`08_reference_uncertainty` shows uncertain decisions divided by attempted classifications of rated reference items. Accept and exclude are definitive outcomes. Errors and unavailable outputs remain in the attempted denominator but are counted separately in the CSV; references without a classification attempt are excluded. A matched artifact classification takes priority over the OpenAlex cited-paper fallback, so each reference entity contributes once. N/A means zero attempts. This differs from the overall abstention metric and from end-to-end unresolved artifact coverage.", ""])
    path.write_text(current.rstrip() + "\n" + "\n".join(lines), encoding="utf-8")


@audited_bundle
def build(input_dir: Path, config: dict[str, object]) -> list[Path]:
    """Build the complete evaluation visualization bundle from persisted audit artifacts."""
    metrics_path = input_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError(f"evaluation metrics do not exist: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = {**config, "field_labels": metrics.get("source_contract", {}).get("field_labels", config.get("field_labels", {}))}
    configure_style(config)
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    figure_dir = input_dir / "figures"
    outputs: list[Path] = []
    outputs += _decision_performance(metrics, figure_dir, formats, dpi)
    outputs += _decision_confusion(input_dir, metrics, figure_dir, formats, dpi)
    outputs += _field_accuracy(input_dir, metrics, figure_dir, formats, dpi)
    outputs += _field_categories(input_dir, metrics, figure_dir, formats, dpi)
    outputs += _coverage(metrics, figure_dir, formats, dpi)
    outputs += _taxonomy_accuracy_overview(input_dir, metrics, figure_dir, formats, dpi)
    outputs += _recovered_taxonomy_accuracy(input_dir, metrics, figure_dir, formats, dpi)
    outputs += _reference_uncertainty(input_dir, metrics, figure_dir, formats, dpi)
    _append_readme(input_dir, outputs)
    LOG.info("evaluation plot bundle complete source=%s files=%d output=%s", metrics.get("source", "unknown"), len(outputs), figure_dir)
    return outputs
