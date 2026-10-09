from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ids_meta.plotting.common import COLORS, configure_style, load_frame, metadata_caption, percentage_bars, save_figure, stamp, subcaption, taxonomy_field_label, taxonomy_overview_label, taxonomy_value_label, write_table
from ids_meta.common.logging import logger
from ids_meta.plotting.recovery import recovered_taxonomy_table
from ids_meta.plotting.uncertainty import reference_uncertainty

LOG = logger("plotting.compare")


def _decision_distribution(
    source: str, combined: pd.DataFrame, out_dir: Path, formats: list[str], dpi: int,
) -> list[Path]:
    """Show horizontal disposition shares on a common source-union denominator."""
    dispositions = {
        "accept": "Accept", "accept_adjacent": "Accept", "exclude": "Exclude",
        "uncertain": "Uncertain", "error": "Error", "": "Missing",
    }
    colors = {
        "Accept": "#286f9b", "Exclude": "#afbac4", "Uncertain": "#dc9a42",
        "Error": "#C94F4F", "Missing": "#e3e8ec", "Other outcomes": "#7A5FA8",
    }
    normalized = combined.apply(lambda column: column.str.strip().str.casefold())
    grouped = normalized.apply(lambda column: column.map(dispositions).fillna("Other outcomes"))
    names = list(combined.columns)
    total = len(combined)
    present = set(grouped.stack())
    categories = [category for category in colors if category in present]
    rows = [
        {
            "model": model, "disposition": category,
            "count": int(grouped[model].eq(category).sum()), "total_sources": total,
            "share_percent": 100 * grouped[model].eq(category).sum() / total,
        }
        for model in names for category in categories
    ]
    write_table(
        pd.DataFrame(rows, columns=["model", "disposition", "count", "total_sources", "share_percent"]),
        out_dir / "data", "01_model_decision_shares",
    )
    fig, ax = plt.subplots(figsize=(12, max(4.2, .85 * len(names) + 1.8)))
    fig.subplots_adjust(left=.26, right=.97, bottom=.26, top=.88)
    if not total:
        ax.axis("off")
        ax.text(.5, .5, "No classification results", transform=ax.transAxes, ha="center", va="center")
    else:
        table = pd.DataFrame(rows).set_index(["model", "disposition"])
        left = [0.0] * len(names)
        for category in categories:
            values = [float(table.loc[(name, category), "share_percent"]) for name in names]
            ax.barh(range(len(names)), values, left=left, color=colors[category], height=.60, label=category)
            for index, (name, width) in enumerate(zip(names, values)):
                if width >= 6:
                    count = int(table.loc[(name, category), "count"])
                    ax.text(
                        left[index] + width / 2, index, f"{count:,}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if category in {"Accept", "Error", "Other outcomes"} else "#24323d",
                    )
                left[index] += width
        ax.set_yticks(range(len(names)), labels=[_display_label(name, 46) for name in names])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        unit = "papers" if source == "openalex" else "repositories"
        ax.set_xlabel(f"Share of {total:,} {unit} (%) · numbers inside bars are record counts")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.legend(
            loc="upper left", bbox_to_anchor=(0, -.18), ncol=min(3, len(categories)),
            frameon=False, fontsize=10,
        )
    ax.set_title(f"{_source_name(source)} decisions by model", loc="center", fontweight="bold", pad=16)
    accepted = "accept + accept_adjacent" if source == "openalex" else "accept"
    stamp(fig, f"Accept={accepted}; Exclude=exclude; common source union; distributions describe decisions, not correctness. Exact counts, including small segments, are in CSV.")
    return save_figure(fig, out_dir, "01_model_decision_counts", formats, dpi)


def _source_name(source: str) -> str:
    """Return the publication-ready display name for a discovery source."""
    return "OpenAlex" if source == "openalex" else "GitHub" if source == "github" else source.title()


def build(source: str, input_path: Path, out_dir: Path, config: dict[str, object]) -> list[Path]:
    """Build plots comparing prefilter scores with subsequent LLM decisions."""
    LOG.info("building %s classifier comparison input=%s output=%s", source, input_path, out_dir)
    configure_style(config)
    frame = load_frame(input_path, ("prefilter_band", "prefilter_score", "decision"))
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    matrix = pd.crosstab(frame["prefilter_band"], frame["decision"])
    write_table(matrix.reset_index(), out_dir / "data", "01_prefilter_decisions")
    fig, ax = plt.subplots(); matrix.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"{_source_name(source)} prefilter bands and LLM decisions", loc="center", fontweight="bold")
    ax.set_ylabel("Classified candidates"); ax.legend(title="LLM decision", fontsize=8)
    stamp(fig, metadata_caption(frame, "universe: records evaluated by the LLM"))
    outputs = save_figure(fig, out_dir, "01_prefilter_decisions", formats, dpi)
    scores = frame.copy(); scores["prefilter_score"] = pd.to_numeric(scores["prefilter_score"], errors="coerce")
    table = scores.groupby("decision")["prefilter_score"].agg(["count", "min", "median", "max"]).reset_index()
    write_table(table, out_dir / "data", "02_score_summary")
    fig, ax = plt.subplots()
    groups = [group["prefilter_score"].dropna().values for _name, group in scores.groupby("decision")]
    labels = [name for name, _group in scores.groupby("decision")]
    if groups:
        ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_title(f"{_source_name(source)} prefilter scores by LLM decision", loc="center", fontweight="bold"); ax.set_ylabel("Prefilter score")
    stamp(fig, metadata_caption(frame, "prefilter scores select candidates; they are not final labels"))
    outputs += save_figure(fig, out_dir, "02_score_summary", formats, dpi)
    LOG.info("classifier comparison complete source=%s files=%d", source, len(outputs))
    return outputs


def build_models(source: str, model_inputs: dict[str, Path], out_dir: Path, config: dict[str, object]) -> list[Path]:
    """Build cross-model decision, agreement, and disagreement comparison plots."""
    if len(model_inputs) < 2:
        raise ValueError("model comparison requires at least two classification inputs")
    if source not in {"github", "openalex"}:
        raise ValueError(f"unsupported comparison source: {source}")
    configure_style(config)
    identity = "github_repo_id" if source == "github" else "openalex_id"
    label_column = "full_name" if source == "github" else "title"
    decisions: list[pd.Series] = []
    labels: dict[str, str] = {}
    for model, path in model_inputs.items():
        frame = load_frame(path, (identity, "decision"))
        if label_column not in frame:
            frame[label_column] = frame[identity]
        view = frame.drop_duplicates(identity, keep="last").set_index(identity)
        decisions.append(view["decision"].rename(model))
        labels.update({str(key): str(value) for key, value in view[label_column].items()})
    combined = pd.concat(decisions, axis=1).fillna("")
    names = list(model_inputs)
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    top_n = int(config.get("top_n", 15))
    outputs: list[Path] = []

    long_counts = combined.reset_index().melt(id_vars=identity, var_name="model", value_name="decision")
    long_counts = long_counts[long_counts["decision"] != ""]
    count_table = long_counts.groupby(["model", "decision"]).size().reset_index(name="count")
    write_table(count_table, out_dir / "data", "01_model_decision_counts")
    outputs += _decision_distribution(source, combined, out_dir, formats, dpi)

    agreement = pd.DataFrame(index=names, columns=names, dtype=float)
    agreement_rows: list[dict[str, object]] = []
    for left in names:
        for right in names:
            shared = combined[left].ne("") & combined[right].ne("")
            value = float((combined.loc[shared, left] == combined.loc[shared, right]).mean()) if shared.any() else float("nan")
            agreement.loc[left, right] = value
            agreement_rows.append({"left_model": left, "right_model": right, "shared_sources": int(shared.sum()), "exact_agreement": value})
    write_table(pd.DataFrame(agreement_rows), out_dir / "data", "02_pairwise_agreement")
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.25), max(5, len(names) * 1.05)))
    image = ax.imshow(agreement.astype(float).values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)), labels=names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), labels=names)
    for row_index, left in enumerate(names):
        for column_index, right in enumerate(names):
            value = agreement.loc[left, right]
            text_value = "n/a" if pd.isna(value) else f"{value:.0%}"
            ax.text(column_index, row_index, text_value, ha="center", va="center", color="white" if not pd.isna(value) and value > .65 else "black")
    fig.colorbar(image, ax=ax, label="Exact agreement")
    ax.set_title(f"{_source_name(source)} pairwise classification agreement", loc="center", fontweight="bold")
    stamp(fig, "agreement uses only sources classified by both outputs; overlap counts are in the companion CSV")
    outputs += save_figure(fig, out_dir, "02_pairwise_agreement", formats, dpi)

    unique_counts = combined.apply(lambda row: len(set(row)), axis=1)
    disagreement = combined.loc[unique_counts > 1].copy()
    disagreement.insert(0, label_column, [labels.get(str(key), "") for key in disagreement.index])
    disagreement.insert(1, "distinct_decisions", unique_counts.loc[disagreement.index].astype(int))
    disagreement_table = disagreement.reset_index()
    write_table(disagreement_table, out_dir / "data", "03_disagreement_records")
    if disagreement.empty:
        patterns = pd.DataFrame(columns=["classification_pattern", "count"])
    else:
        pattern_values = combined.loc[disagreement.index].apply(lambda row: " | ".join(f"{model}={row[model] or chr(8709)}" for model in names), axis=1)
        patterns = pattern_values.value_counts().rename_axis("classification_pattern").reset_index(name="count").head(top_n)
    write_table(patterns, out_dir / "data", "03_disagreement_patterns")
    fig, ax = plt.subplots(figsize=(10, max(4.8, .55 * max(1, len(patterns)))))
    if patterns.empty:
        ax.axis("off")
        ax.text(.5, .5, "No cross-model classification differences", transform=ax.transAxes, ha="center", va="center")
    else:
        ordered = patterns.iloc[::-1]
        ax.barh(ordered["classification_pattern"], ordered["count"], color="#C94F4F")
        ax.set_xlabel("Sources")
    ax.set_title(f"{_source_name(source)} cross-model disagreement patterns", loc="center", fontweight="bold")
    stamp(fig, f"disagreements={len(disagreement)} of {len(combined)} sources; top {top_n} patterns shown")
    outputs += save_figure(fig, out_dir, "03_disagreement_patterns", formats, dpi)
    LOG.info("model comparison complete source=%s models=%d disagreements=%d files=%d", source, len(names), len(disagreement), len(outputs))
    return outputs


METRIC_SPECS = (
    ("detection_rate", "Detection rate (recall)", "operational", "DR"),
    ("false_positive_rate", "False-positive rate", "operational", "FPR"),
    ("accuracy", "Accuracy", "operational", "ACC"),
    ("precision", "Precision", "operational", "PREC"),
    ("f1", "F1 score", "operational", "f1"),
    ("definitive_coverage", "Definitive-decision coverage", "root", "definitive_coverage"),
    ("abstention_rate", "Abstention rate", "root", "abstention_rate"),
    ("field_accuracy", "Taxonomy-field exact accuracy", "root", "classification_field_accuracy"),
    ("field_macro_f1", "Mean taxonomy-field macro-F1", "root", "classification_field_macro_f1"),
    ("false_discovery_rate", "False-discovery rate", "operational", "FDR"),
)


OVERVIEW_SPECS = (
    ("detection_rate", "DR (recall) ↑", "higher", "operational", "DR"),
    ("false_negative_rate", "FNR ↓", "lower", "operational", "FNR"),
    ("false_positive_rate", "FPR ↓", "lower", "operational", "FPR"),
    ("false_discovery_rate", "FDR ↓", "lower", "operational", "FDR"),
    ("precision", "Precision ↑", "higher", "operational", "PREC"),
    ("accuracy", "Accuracy ↑", "higher", "operational", "ACC"),
    ("f1", "Decision F1 ↑", "higher", "operational", "f1"),
    ("definitive_coverage", "Definitive-decision coverage ↑", "higher", "root", "definitive_coverage"),
    ("abstention_rate", "Abstention rate ↓", "lower", "root", "abstention_rate"),
    ("field_accuracy", "Taxonomy exact accuracy ↑", "higher", "root", "classification_field_accuracy"),
    ("field_macro_f1", "Taxonomy macro-F1 ↑", "higher", "root", "classification_field_macro_f1"),
)


def _method_color(method: str) -> str:
    """Return a stable color for a classifier family."""
    return {
        "keyword": COLORS[2], "nli": COLORS[1], "llm_nli": COLORS[5], "llm": COLORS[0],
        "ensemble": COLORS[3],
    }.get(method, COLORS[8])


def _metric_value(metrics: dict[str, object], section: str, key: str) -> float | None:
    """Read a numeric metric while preserving explicit N/A values."""
    container = metrics if section == "root" else metrics.get(section, {})
    if not isinstance(container, dict):
        return None
    value = container.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _wilson(successes: int, total: int, confidence_level: float) -> list[float] | None:
    """Calculate a Wilson interval from exported counts and a declared confidence level."""
    if total <= 0:
        return None
    z = NormalDist().inv_cdf(.5 + confidence_level / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _fallback_interval(metrics: dict[str, object], stem: str, value: float | None) -> list[float] | None:
    """Derive a binomial interval from counts when a report lacks the requested interval."""
    confidence_level = float(metrics.get("confidence_level", .95))
    operational = metrics.get("operational", {})
    if isinstance(operational, dict):
        tp, fp = int(operational.get("tp", 0)), int(operational.get("fp", 0))
        tn, fn = int(operational.get("tn", 0)), int(operational.get("fn", 0))
        pairs = {
            "detection_rate": (tp, tp + fn), "false_positive_rate": (fp, fp + tn),
            "false_discovery_rate": (fp, tp + fp),
            "accuracy": (tp + tn, tp + fp + tn + fn), "precision": (tp, tp + fp),
        }
        if stem in pairs:
            return _wilson(*pairs[stem], confidence_level)
    denominator = (
        int(metrics.get("classification_field_comparisons", 0))
        if stem == "field_accuracy" else int(metrics.get("rated_entities", 0))
    )
    if stem in {"definitive_coverage", "abstention_rate", "field_accuracy"} and value is not None:
        return _wilson(round(value * denominator), denominator, confidence_level)
    return None


def _display_label(label: str, maximum: int = 34) -> str:
    """Keep long model identifiers readable on an axis while retaining them in CSV data."""
    if label.casefold().startswith("nli") and "deberta" in label.casefold():
        return "NLI · deberta-v3"
    return label if len(label) <= maximum else label[: maximum - 1] + "…"


def _reference_recovery_overview(
    source: str, loaded: dict[str, dict[str, object]], out_dir: Path,
    formats: list[str], dpi: int,
) -> list[Path]:
    """Summarize operational recall over expected positive reference entities."""
    rows: list[dict[str, object]] = []
    for label, metrics in loaded.items():
        operational = metrics.get("operational", {})
        if not isinstance(operational, dict):
            operational = {}
        recovered = int(operational.get("tp", 0))
        missed = int(operational.get("fn", 0))
        expected = recovered + missed
        confidence = float(metrics.get("confidence_level", .95))
        interval = _wilson(recovered, expected, confidence)
        method = str(metrics.get("classifier_method", "")) or (
            "ensemble" if label.casefold() == "ensemble" else "llm"
        )
        rows.append({
            "classifier": label, "classifier_method": method,
            "recovered": recovered, "missed": missed, "expected_positive_entities": expected,
            "rated_entities": metrics.get("rated_entities", 0),
            "value": recovered / expected if expected else None,
            "percentage": 100 * recovered / expected if expected else None,
            "confidence_level": confidence,
            "ci_low": interval[0] if interval else None, "ci_high": interval[1] if interval else None,
            "rating_profile": metrics.get("ratings_path", ""),
        })
    table = pd.DataFrame(rows)
    write_table(table, out_dir / "data", "01_reference_recovery")
    artifact_units = all(metrics.get("evaluation_unit") == "artifact" for metrics in loaded.values())
    unit = "datasets" if source == "openalex" and artifact_units else "papers" if source == "openalex" else "repositories"
    denominators = sorted(set(table["expected_positive_entities"]))
    denominator = (
        f"{denominators[0]:,} expected positive reference {unit}"
        if len(denominators) == 1 else "Expected positive reference count varies by classifier; see CSV"
    )
    fig = percentage_bars(
        [_display_label(str(label), 42) for label in table["classifier"]], table["value"].tolist(),
        title=f"{_source_name(source)} reference recovery by classifier",
        xlabel=f"Known reference {unit} recovered (%)",
        caption=f"{denominator}. Missing, uncertain and excluded positives count as misses. N/A means no expected positives. Confidence intervals are in the CSV and detailed detection-rate plot.",
        colors=[_method_color(str(method)) for method in table["classifier_method"]],
    )
    return save_figure(fig, out_dir, "01_reference_recovery", formats, dpi)


def _reference_uncertainty_overview(
    source: str, evaluation_inputs: dict[str, Path], out_dir: Path, formats: list[str], dpi: int,
) -> list[Path]:
    """Compare uncertainty on attempted reference classifications, preserving N/A."""
    rows = []
    for label, directory in evaluation_inputs.items():
        counts = reference_uncertainty(directory)
        if counts is not None:
            rows.append({"classifier": label, **counts})
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    write_table(frame, out_dir / "data", "02_reference_uncertainty")
    fig = percentage_bars(
        [f"{_display_label(row['classifier'])} ({row['uncertain']:,}/{row['attempted']:,} attempts)" for row in rows],
        frame["value"].tolist(),
        title=f"{_source_name(source)} classifier uncertainty on reference items",
        xlabel="Uncertain decisions among attempted reference classifications (%)",
        caption="All rated reference items; missing retrieval/classification is excluded. Errors stay in the attempted denominator but do not count as uncertain. OpenAlex uses a classified cited paper when no artifact match exists. N/A=no attempts.",
        colors=[COLORS[2]] * len(rows),
    )
    return save_figure(fig, out_dir, "02_reference_uncertainty", formats, dpi)


def _taxonomy_comparison_overview(
    source: str, field_table: pd.DataFrame, labels: list[str], out_dir: Path,
    formats: list[str], dpi: int, *, recovered_only: bool = False,
) -> list[Path]:
    """Compare all approaches in compact field groups on one percentage axis."""
    fields = list(dict.fromkeys(field_table["field"]))
    table = field_table.set_index(["field", "classifier"]).reindex(
        pd.MultiIndex.from_product([fields, labels], names=["field", "classifier"]),
    ).reset_index()
    for column in ("comparisons", "exact_matches"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    estimable = table["supported"].astype(str).str.casefold().eq("true") & table["comparisons"].gt(0)
    table["value"] = (table["exact_matches"] / table["comparisons"].where(estimable)).where(estimable)
    table["percentage"] = table["value"] * 100
    table["estimable"] = estimable
    stem = "05_recovered_taxonomy_comparison" if recovered_only else "04_taxonomy_comparison"
    write_table(table, out_dir / "data", stem)

    base_colors = {"keyword": "#b6a064", "nli": "#559373", "llm_nli": "#3b8c8c", "llm": "#286f9b", "ensemble": "#8171a4"}
    extra_colors = iter(["#3b8c8c", "#C94F4F", "#B85C9E", "#8C6D31", "#D9902F", "#6C7A89"])
    colors: dict[str, object] = {}
    for index, label in enumerate(labels):
        method_rows = table.loc[table["classifier"].eq(label), "classifier_method"].dropna()
        method = str(method_rows.iloc[0]) if not method_rows.empty else ""
        color = base_colors.get(method)
        if color is None or color in colors.values():
            color = next(extra_colors, None)
        if color is None:
            color = plt.get_cmap("tab20")(index % 20)
        colors[label] = color

    fig, ax = plt.subplots(figsize=(12.5, max(6.0, .75 * len(fields) + 1.8)))
    fig.subplots_adjust(left=.25, right=.97, top=.86, bottom=.23)
    group_height = .76
    step = group_height / len(labels)
    bar_height = step * .83
    lookup = table.set_index(["field", "classifier"])
    for field_index, field in enumerate(fields):
        if field_index % 2 == 0:
            ax.axhspan(field_index - .47, field_index + .47, color="#f3f6f8", zorder=0)
        for model_index, label in enumerate(labels):
            y = field_index + (model_index - (len(labels) - 1) / 2) * step
            value = lookup.loc[(field, label), "percentage"]
            count = lookup.loc[(field, label), "comparisons"]
            count_label = f" (n={int(count):,})" if recovered_only and pd.notna(count) else ""
            if pd.isna(value):
                ax.text(1.2, y, "N/A" + count_label, va="center", fontsize=7.5, color="#5f6c76")
                continue
            ax.barh(y, value, height=bar_height, color=colors[label], zorder=3)
            if value == 0:
                ax.plot(0, y, marker="|", color=colors[label], markersize=6, markeredgewidth=2, clip_on=False, zorder=4)
            inside = value >= (80 if recovered_only else 93)
            ax.text(
                value - 1.2 if inside else value + 1.2, y, f"{value:.1f}%" + count_label,
                ha="right" if inside else "left", va="center", fontsize=7.5,
                color="white" if inside else "#24323d", zorder=4,
            )
    field_labels = []
    for field in fields:
        counts = sorted(set(int(value) for value in table.loc[table["field"].eq(field) & estimable, "comparisons"]))
        denominator = f"n={counts[0]:,}" if len(counts) == 1 else "n varies" if counts else "N/A"
        field_labels.append(textwrap.fill(taxonomy_overview_label(field), 27) + ("" if recovered_only else f"\n({denominator})"))
    ax.set_yticks(range(len(fields)), labels=field_labels)
    ax.set_ylim(len(fields) - .5, -.55)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Correct labels among recovered reference items (%)" if recovered_only else "Correct labels across the reference inventory (%)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(False)
    ax.grid(axis="x", alpha=.16)
    title = "taxonomy accuracy among recovered items" if recovered_only else "taxonomy accuracy by approach"
    ax.set_title(f"{_source_name(source)} {title}", loc="center", fontweight="bold", pad=18)
    legend_labels = {}
    for label in labels:
        display = _display_label(label, 38)
        if recovered_only:
            rows = table.loc[table["classifier"].eq(label)]
            recovered = rows["recovered"].dropna()
            expected = rows["expected_positive_entities"].dropna()
            display += f"\n{int(recovered.iloc[0]):,}/{int(expected.iloc[0]):,} recovered" if not recovered.empty and not expected.empty else "\naudit unavailable"
        legend_labels[label] = display
    fig.legend(
        handles=[Patch(facecolor=colors[label], label=legend_labels[label]) for label in labels],
        loc="lower center", bbox_to_anchor=(.59, .085), ncol=min(5, len(labels)),
        frameon=False, fontsize=8.5,
    )
    caption = (
        "Bars follow legend order. Only accepted reference positives are evaluated; n=field-rated recovered items.\nMissing field predictions are incorrect; multi-label fields require an exact set match. N/A=unsupported or no recovered comparisons."
        if recovered_only else
        "Bars follow legend order within each field. n=rated field comparisons; unresolved results remain in the denominator.\nMulti-label fields require an exact set match; N/A=unsupported or no comparisons. Exact counts are in the companion CSV."
    )
    subcaption(fig, .25, .018, caption, fontsize=7.5, color="#5f6c76")
    return save_figure(fig, out_dir, stem, formats, dpi)


def build_classifiers(
    source: str, classification_inputs: dict[str, Path], evaluation_inputs: dict[str, Path],
    out_dir: Path, config: dict[str, object],
) -> list[Path]:
    """Compare keyword, NLI, LLM, and ensemble outputs on shared metric axes."""
    if len(evaluation_inputs) < 2:
        raise ValueError("classifier comparison requires at least two evaluated outputs")
    configure_style(config)
    loaded: dict[str, dict[str, object]] = {}
    rating_paths: set[str] = set()
    evaluation_units: set[str] = set()
    for label, directory in evaluation_inputs.items():
        path = directory / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        loaded[label] = metrics
        rating_path = str(metrics.get("ratings_path", "")).strip()
        rating_paths.add(str(Path(rating_path).resolve()) if rating_path else "")
        evaluation_units.add(str(metrics.get("evaluation_unit", "")))
    if len(loaded) < 2:
        raise ValueError("classifier comparison requires at least two readable metrics.json files")
    if len(rating_paths) != 1 or len(evaluation_units) != 1:
        raise ValueError("classifier metrics must use the same rating profile and evaluation unit")
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    outputs: list[Path] = []
    labels = list(loaded)
    metrics_dir = out_dir / "metrics"
    taxonomy_dir = out_dir / "taxonomy"
    overview_rows: list[dict[str, object]] = []
    for stem, metric_label, direction, section, key in OVERVIEW_SPECS:
        for label, metrics in loaded.items():
            not_estimable = set(metrics.get("not_estimable_metrics", []))
            value = None if stem in not_estimable else _metric_value(metrics, section, key)
            method = str(metrics.get("classifier_method", "")) or (
                "ensemble" if label.casefold() == "ensemble" else "llm"
            )
            overview_rows.append({
                "classifier": label,
                "classifier_method": method,
                "metric": stem,
                "metric_label": metric_label,
                "preferred_direction": direction,
                "value": value,
                "estimable": value is not None,
                "rated_entities": metrics.get("rated_entities", ""),
                "rating_profile": metrics.get("ratings_path", ""),
            })
    overview_table = pd.DataFrame(overview_rows)
    write_table(overview_table, out_dir / "data", "00_metric_overview")
    metric_labels = [spec[1] for spec in OVERVIEW_SPECS]
    overview_pivot = overview_table.pivot(
        index="metric_label", columns="classifier", values="value",
    ).reindex(index=metric_labels, columns=labels)
    fig, ax = plt.subplots(figsize=(
        max(8.5, 1.25 * len(labels) + 3.5),
        max(7.2, .52 * len(metric_labels) + 2.5),
    ))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#E6E8EB")
    image = ax.imshow(
        overview_pivot.astype(float).values, vmin=0, vmax=1,
        cmap=cmap, aspect="auto",
    )
    ax.set_xticks(
        range(len(labels)), labels=[_display_label(label) for label in labels],
        rotation=25, ha="right",
    )
    ax.set_yticks(range(len(metric_labels)), labels=metric_labels)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for row_index in range(len(metric_labels)):
        for column_index in range(len(labels)):
            if pd.isna(overview_pivot.iloc[row_index, column_index]):
                ax.text(column_index, row_index, "N/A", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Observed rate (0–1)")
    ax.set_title(
        f"{_source_name(source)} — All headline metrics by classification approach",
        loc="center", fontweight="bold",
    )
    stamp(
        fig,
        "color encodes the raw rate; arrows show the preferred direction; "
        "grey N/A=not estimable; exact values are in the companion CSV",
    )
    fig.subplots_adjust(bottom=.25, left=.30)
    outputs += save_figure(fig, out_dir, "00_metric_overview", formats, dpi)
    outputs += _reference_recovery_overview(source, loaded, out_dir, formats, dpi)
    outputs += _reference_uncertainty_overview(
        source, {label: evaluation_inputs[label] for label in loaded}, out_dir, formats, dpi,
    )
    for number, (stem, title, section, key) in enumerate(METRIC_SPECS, 1):
        rows: list[dict[str, object]] = []
        for label, metrics in loaded.items():
            not_estimable = set(metrics.get("not_estimable_metrics", []))
            value = None if stem in not_estimable else _metric_value(metrics, section, key)
            method = str(metrics.get("classifier_method", "")) or (
                "ensemble" if label.casefold() == "ensemble" else "llm"
            )
            operational = metrics.get("operational", {})
            intervals = metrics.get("confidence_intervals", {})
            interval = intervals.get(stem) if isinstance(intervals, dict) else None
            if value is None:
                interval = None
            elif not (isinstance(interval, list) and len(interval) == 2):
                interval = _fallback_interval(metrics, stem, value)
            rows.append({
                "classifier": label, "classifier_method": method, "metric": stem,
                "classifier_id": metrics.get("classifier_id", ""),
                "value": value, "rated_entities": metrics.get("rated_entities", ""),
                "confidence_level": metrics.get("confidence_level", .95),
                "ci_low": interval[0] if isinstance(interval, list) and len(interval) == 2 else None,
                "ci_high": interval[1] if isinstance(interval, list) and len(interval) == 2 else None,
                "tp": operational.get("tp", "") if isinstance(operational, dict) else "",
                "fp": operational.get("fp", "") if isinstance(operational, dict) else "",
                "tn": operational.get("tn", "") if isinstance(operational, dict) else "",
                "fn": operational.get("fn", "") if isinstance(operational, dict) else "",
                "rating_profile": metrics.get("ratings_path", ""),
            })
        table = pd.DataFrame(rows)
        write_table(table, metrics_dir / "data", f"{number:02d}_{stem}")
        visible = table.dropna(subset=["value"])
        fig, ax = plt.subplots(figsize=(max(8, .9 * len(visible) + 3), 5.4))
        if visible.empty:
            ax.axis("off")
            ax.text(
                .5, .5, "Not estimable from this rating profile",
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
            )
            ax.set_title(f"{_source_name(source)} — {title} by classification approach", loc="center", fontweight="bold")
            stamp(fig, "the positive-only reference set contains no negative examples for this metric")
            outputs += save_figure(fig, metrics_dir, f"{number:02d}_{stem}", formats, dpi)
            continue
        error = None
        if visible["ci_low"].notna().all() and visible["ci_high"].notna().all():
            values = visible["value"].astype(float)
            error = [
                (values - visible["ci_low"].astype(float)).clip(lower=0),
                (visible["ci_high"].astype(float) - values).clip(lower=0),
            ]
        ax.bar(
            range(len(visible)), visible["value"].astype(float),
            color=[_method_color(str(value)) for value in visible["classifier_method"]],
            yerr=error, capsize=5,
            error_kw={"ecolor": "#202830", "elinewidth": 1.6, "capthick": 1.6},
        )
        ax.set_xticks(
            range(len(visible)), labels=[_display_label(str(value)) for value in visible["classifier"]],
            rotation=25, ha="right",
        )
        fig.subplots_adjust(bottom=.30)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(title)
        ax.set_xlabel("")
        interval_caption = ""
        if error is not None:
            confidence_levels = sorted(set(visible["confidence_level"].astype(float)))
            interval_label = (
                f"{confidence_levels[0]:.0%} Wilson interval"
                if len(confidence_levels) == 1 else "Configured Wilson intervals"
            )
            ax.legend(
                handles=[Line2D(
                    [0], [0], color="#202830", linewidth=1.6, marker="_",
                    markersize=12, markeredgewidth=1.6, label=interval_label,
                )],
                loc="upper right", frameon=True, fontsize=8,
            )
            interval_caption = f"; capped vertical line={interval_label}"
        ax.set_title(f"{_source_name(source)} — {title} by classification approach", loc="center", fontweight="bold")
        stamp(
            fig,
            f"n={visible['rated_entities'].iloc[0]}; bar height=observed metric"
            f"{interval_caption}; missing bar=not estimable",
        )
        outputs += save_figure(fig, metrics_dir, f"{number:02d}_{stem}", formats, dpi)
    field_rows: list[dict[str, object]] = []
    for label, metrics in loaded.items():
        directory = evaluation_inputs[label]
        path = directory / "field_metrics.csv"
        if not path.is_file():
            continue
        method = str(metrics.get("classifier_method", "")) or (
            "ensemble" if label.casefold() == "ensemble" else "llm"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            field_rows.append({
                "classifier": label,
                "classifier_method": method,
                "field": row.get("field", ""),
                "supported": row.get("supported", ""),
                "comparisons": row.get("comparisons", ""),
                "exact_matches": row.get("exact_matches", ""),
                "exact_accuracy": row.get("exact_accuracy"),
                "macro_f1": row.get("macro_f1"),
                "per_label_metrics_json": row.get("per_label_metrics_json", ""),
            })
    field_table = pd.DataFrame(field_rows)
    if not field_table.empty:
        outputs += _taxonomy_comparison_overview(source, field_table, labels, taxonomy_dir, formats, dpi)
        recovered_tables = []
        for label, metrics in loaded.items():
            recovered_table = recovered_taxonomy_table(evaluation_inputs[label])
            if recovered_table is None or recovered_table.empty:
                continue
            recovered_table["classifier"] = label
            recovered_table["classifier_method"] = str(metrics.get("classifier_method", "")) or (
                "ensemble" if label.casefold() == "ensemble" else "llm"
            )
            recovered_tables.append(recovered_table)
        if recovered_tables:
            outputs += _taxonomy_comparison_overview(
                source, pd.concat(recovered_tables, ignore_index=True), labels, taxonomy_dir,
                formats, dpi, recovered_only=True,
            )
        for number, value_column, title in (
            (1, "exact_accuracy", "Exact accuracy by taxonomy field"),
            (2, "macro_f1", "Macro-F1 by taxonomy field"),
        ):
            table = field_table.copy()
            table[value_column] = pd.to_numeric(table[value_column], errors="coerce")
            write_table(table, taxonomy_dir / "data", f"{number:02d}_{value_column}_by_field")
            pivot = table.pivot(index="field", columns="classifier", values=value_column)
            pivot = pivot.reindex(columns=labels)
            fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(labels) + 3), max(4.8, .62 * len(pivot) + 2)))
            cmap = plt.get_cmap("Blues").copy()
            cmap.set_bad("#E6E8EB")
            image = ax.imshow(pivot.astype(float).values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
            ax.set_xticks(
                range(len(labels)), labels=[_display_label(label) for label in labels],
                rotation=25, ha="right",
            )
            ax.set_yticks(range(len(pivot)), labels=[taxonomy_field_label(field) for field in pivot.index])
            ax.set_xlabel("")
            ax.set_ylabel("")
            fig.colorbar(image, ax=ax, label=title)
            ax.set_title(f"{_source_name(source)} — {title}", loc="center", fontweight="bold")
            stamp(fig, "color encodes the metric; grey=unsupported or not estimable; exact values are in the companion CSV")
            fig.subplots_adjust(bottom=.30)
            outputs += save_figure(
                fig, taxonomy_dir, f"{number:02d}_{value_column}_by_field", formats, dpi,
            )
    label_rows: list[dict[str, object]] = []
    for label, metrics in loaded.items():
        path = evaluation_inputs[label] / "field_label_metrics.csv"
        if not path.is_file():
            continue
        method = str(metrics.get("classifier_method", "")) or (
            "ensemble" if label.casefold() == "ensemble" else "llm"
        )
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            label_rows.append({
                "classifier": label, "classifier_method": method,
                "field": row.get("field", ""), "label": row.get("label", ""),
                "tp": row.get("tp", ""), "fp": row.get("fp", ""), "fn": row.get("fn", ""),
                "precision": row.get("precision"), "recall": row.get("recall"),
                "f1": row.get("f1"),
            })
    label_table = pd.DataFrame(label_rows)
    if not label_table.empty:
        label_table["f1"] = pd.to_numeric(label_table["f1"], errors="coerce")
        write_table(label_table, taxonomy_dir / "data", "03_category_f1")
        label_table["field_label"] = label_table.apply(
            lambda row: f"{taxonomy_field_label(row['field'])} — {taxonomy_value_label(row['field'], row['label'])}",
            axis=1,
        )
        pivot = label_table.pivot(index="field_label", columns="classifier", values="f1")
        pivot = pivot.reindex(columns=labels)
        fig, ax = plt.subplots(figsize=(
            max(8, 1.25 * len(labels) + 3), max(6, .34 * len(pivot) + 2),
        ))
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#E6E8EB")
        image = ax.imshow(pivot.astype(float).values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(
            range(len(labels)), labels=[_display_label(label) for label in labels],
            rotation=25, ha="right",
        )
        ax.set_yticks(
            range(len(pivot)),
            labels=[str(value).replace("_", " ") for value in pivot.index],
            fontsize=7,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        fig.colorbar(image, ax=ax, label="One-vs-rest category F1")
        ax.set_title(
            f"{_source_name(source)} — F1 for every rated taxonomy category",
            loc="center", fontweight="bold",
        )
        stamp(fig, "one-vs-rest F1; grey=not represented or unsupported; exact TP/FP/FN values are in the companion CSV")
        fig.subplots_adjust(bottom=.18, left=.28)
        outputs += save_figure(fig, taxonomy_dir, "03_category_f1", formats, dpi)
    available_classifications = {
        label: path for label, path in classification_inputs.items()
        if label in loaded and path.is_file()
    }
    if len(available_classifications) >= 2:
        outputs += build_models(
            source, available_classifications, out_dir / "agreement", config,
        )
    readme = out_dir / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text(
        f"# {_source_name(source)} classifier comparison\n\n"
        "## Start here\n\n"
        "Open `00_metric_overview.svg`. Rows are metrics and columns are classifiers. "
        "The color is the raw rate, so use the arrow on each row to see whether higher "
        "or lower is preferable. Grey N/A cells mean the benchmark cannot estimate that metric.\n\n"
        "Open `01_reference_recovery.svg` for a horizontal overview of recovered positive "
        "reference entities by classifier, with percentage labels. Exact recovered/missed "
        "counts, denominators and confidence intervals are in `data/01_reference_recovery.csv`. "
        "The sibling classifier `figures/06_taxonomy_accuracy.svg` charts show exact field "
        "accuracy as horizontal percentage bars.\n\n"
        "Open `02_reference_uncertainty.svg` for uncertain decisions divided by attempted "
        "classifications of rated reference items. Each bar shows its uncertain/attempted "
        "counts; retrieval gaps and unclassified references are excluded. Errors remain in "
        "the denominator and are reported separately in `data/02_reference_uncertainty.csv`. "
        "OpenAlex uses a classified cited paper if no artifact match exists. Each reference "
        "entity contributes once, and N/A means no attempts. This is distinct from the legacy "
        "abstention metric.\n\n"
        "## Drill-down directories\n\n"
        "- `metrics/`: one enlarged operational-metric comparison per figure.\n"
        "- `taxonomy/`: field and individual-category comparisons. "
        "Start with `04_taxonomy_comparison.svg` for compact grouped horizontal bars "
        "comparing every approach within each field. Colours identify individual classifiers; "
        "percentages and rated denominators are printed, with exact values in its companion CSV.\n"
        "  `05_recovered_taxonomy_comparison.svg` compares field accuracy only among each "
        "approach's recovered reference positives. Bar annotations show field-rated sample counts; "
        "the legend shows recovered/expected totals. Negative controls and missed references "
        "are excluded, while missing field predictions are incorrect. Multi-label fields require "
        "an exact set match. N/A means unsupported fields, no recovered comparisons, or a missing "
        "entity audit. Different approaches can recover different subsets, so read with recovery. "
        "The per-classifier equivalent is `figures/07_recovered_taxonomy_accuracy.svg`.\n"
        "- `agreement/`: where classifiers make the same decision; agreement is not correctness.\n"
        "  `01_model_decision_counts` uses horizontal stacked shares of the full source union, "
        "with record counts inside sufficiently wide segments. Accept combines accept and "
        "accept_adjacent; Exclude means exclude. Errors and missing results stay visible. "
        "Exact decisions remain in `data/01_model_decision_counts.csv`; grouped counts and "
        "percentage denominators are in `data/01_model_decision_shares.csv`.\n"
        "- `data/00_metric_overview.csv`: exact values behind the overview.\n\n"
        "Matching SVG and PDF files are the same figure in screen and publication formats.\n\n"
        "Single-classifier operational/strict/selective views, confusion matrices, and coverage "
        "diagnostics are under the sibling `classifiers/` directory. Those diagnose one output; "
        "the figures here compare approaches on the common rated set.\n",
        encoding="utf-8",
    )
    outputs.append(readme)
    LOG.info(
        "classifier performance comparison complete source=%s classifiers=%d files=%d labels=%s",
        source, len(labels), len(outputs), ",".join(labels),
    )
    return outputs
