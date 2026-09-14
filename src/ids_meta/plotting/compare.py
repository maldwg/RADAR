from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ids_meta.plotting.common import COLORS, configure_style, load_frame, metadata_caption, save_figure, stamp, write_table
from ids_meta.common.logging import logger

LOG = logger("plotting.compare")


def build(source: str, input_path: Path, out_dir: Path, config: dict[str, object]) -> list[Path]:
    """Build plots comparing prefilter scores with subsequent LLM decisions."""
    LOG.info("building %s classifier comparison input=%s output=%s", source, input_path, out_dir)
    configure_style()
    frame = load_frame(input_path, ("prefilter_band", "prefilter_score", "decision"))
    formats = [str(value) for value in config.get("formats", ["svg", "pdf"])]
    dpi = int(config.get("dpi", 160))
    matrix = pd.crosstab(frame["prefilter_band"], frame["decision"])
    write_table(matrix.reset_index(), out_dir / "data", "01_prefilter_decisions")
    fig, ax = plt.subplots(); matrix.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title(f"{source.title()} prefilter bands and LLM decisions", loc="left", fontweight="bold")
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
    ax.set_title(f"{source.title()} prefilter scores by LLM decision", loc="left", fontweight="bold"); ax.set_ylabel("Prefilter score")
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
    configure_style()
    identity = "github_repo_id" if source == "github" else "openalex_id"
    label_column = "full_name" if source == "github" else "title"
    decisions: list[pd.Series] = []
    labels: dict[str, str] = {}
    for model, path in model_inputs.items():
        frame = load_frame(path, (identity, label_column, "decision"))
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
    pivot = count_table.pivot(index="model", columns="decision", values="count").fillna(0)
    fig, ax = plt.subplots()
    if pivot.empty:
        ax.axis("off")
    else:
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Classified sources")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Decision", fontsize=8)
    ax.set_title(f"{source.title()} decisions by model", loc="left", fontweight="bold")
    stamp(fig, f"models={len(names)}; union of sources={len(combined)}")
    outputs += save_figure(fig, out_dir, "01_model_decision_counts", formats, dpi)

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
    ax.set_title(f"{source.title()} pairwise classification agreement", loc="left", fontweight="bold")
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
    ax.set_title(f"{source.title()} cross-model disagreement patterns", loc="left", fontweight="bold")
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
)


def _method_color(method: str) -> str:
    """Return a stable color for a classifier family."""
    return {
        "keyword": COLORS[2], "nli": COLORS[1], "llm": COLORS[0],
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


def _wilson(successes: int, total: int) -> list[float] | None:
    """Calculate a 95% Wilson interval for comparison inputs created before interval export."""
    if total <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _fallback_interval(metrics: dict[str, object], stem: str, value: float | None) -> list[float] | None:
    """Derive a binomial interval from exported counts when an older report lacks one."""
    operational = metrics.get("operational", {})
    if isinstance(operational, dict):
        tp, fp = int(operational.get("tp", 0)), int(operational.get("fp", 0))
        tn, fn = int(operational.get("tn", 0)), int(operational.get("fn", 0))
        pairs = {
            "detection_rate": (tp, tp + fn), "false_positive_rate": (fp, fp + tn),
            "accuracy": (tp + tn, tp + fp + tn + fn), "precision": (tp, tp + fp),
        }
        if stem in pairs:
            return _wilson(*pairs[stem])
    denominator = (
        int(metrics.get("classification_field_comparisons", 0))
        if stem == "field_accuracy" else int(metrics.get("rated_entities", 0))
    )
    if stem in {"definitive_coverage", "abstention_rate", "field_accuracy"} and value is not None:
        return _wilson(round(value * denominator), denominator)
    return None


def _display_label(label: str, maximum: int = 34) -> str:
    """Keep long model identifiers readable on an axis while retaining them in CSV data."""
    return label if len(label) <= maximum else label[: maximum - 1] + "…"


def build_classifiers(
    source: str, classification_inputs: dict[str, Path], evaluation_inputs: dict[str, Path],
    out_dir: Path, config: dict[str, object],
) -> list[Path]:
    """Compare keyword, NLI, LLM, and ensemble outputs on shared metric axes."""
    if len(evaluation_inputs) < 2:
        raise ValueError("classifier comparison requires at least two evaluated outputs")
    configure_style()
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
    for number, (stem, title, section, key) in enumerate(METRIC_SPECS, 1):
        rows: list[dict[str, object]] = []
        for label, metrics in loaded.items():
            not_estimable = set(metrics.get("not_estimable_metrics", []))
            if metrics.get("evaluation_unit") == "artifact":
                not_estimable.update({"false_positive_rate", "accuracy", "precision", "f1"})
            value = None if stem in not_estimable else _metric_value(metrics, section, key)
            method = str(metrics.get("classifier_method", "")) or (
                "ensemble" if label.casefold() == "ensemble" else "llm"
            )
            operational = metrics.get("operational", {})
            intervals = metrics.get("confidence_intervals_95", {})
            interval = intervals.get(stem) if isinstance(intervals, dict) else None
            if not (isinstance(interval, list) and len(interval) == 2):
                interval = _fallback_interval(metrics, stem, value)
            rows.append({
                "classifier": label, "classifier_method": method, "metric": stem,
                "classifier_id": metrics.get("classifier_id", ""),
                "value": value, "rated_entities": metrics.get("rated_entities", ""),
                "ci95_low": interval[0] if isinstance(interval, list) and len(interval) == 2 else None,
                "ci95_high": interval[1] if isinstance(interval, list) and len(interval) == 2 else None,
                "tp": operational.get("tp", "") if isinstance(operational, dict) else "",
                "fp": operational.get("fp", "") if isinstance(operational, dict) else "",
                "tn": operational.get("tn", "") if isinstance(operational, dict) else "",
                "fn": operational.get("fn", "") if isinstance(operational, dict) else "",
                "rating_profile": metrics.get("ratings_path", ""),
            })
        table = pd.DataFrame(rows)
        write_table(table, out_dir / "data", f"{number:02d}_{stem}")
        visible = table.dropna(subset=["value"])
        fig, ax = plt.subplots(figsize=(max(8, .9 * len(visible) + 3), 5.4))
        if visible.empty:
            ax.axis("off")
            ax.text(
                .5, .5, "Not estimable from this rating profile",
                transform=ax.transAxes, ha="center", va="center", fontsize=13,
            )
            ax.set_title(f"{source.title()} — {title} by classification approach", loc="left", fontweight="bold")
            stamp(fig, "the positive-only reference set contains no negative examples for this metric")
            outputs += save_figure(fig, out_dir, f"{number:02d}_{stem}", formats, dpi)
            continue
        error = None
        if visible["ci95_low"].notna().all() and visible["ci95_high"].notna().all():
            values = visible["value"].astype(float)
            error = [
                (values - visible["ci95_low"].astype(float)).clip(lower=0),
                (visible["ci95_high"].astype(float) - values).clip(lower=0),
            ]
        bars = ax.bar(
            range(len(visible)), visible["value"].astype(float),
            color=[_method_color(str(value)) for value in visible["classifier_method"]],
            yerr=error, capsize=4,
        )
        ax.set_xticks(
            range(len(visible)), labels=[_display_label(str(value)) for value in visible["classifier"]],
            rotation=25, ha="right",
        )
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(title)
        ax.set_xlabel("")
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in visible["value"]], padding=3, fontsize=8)
        ax.set_title(f"{source.title()} — {title} by classification approach", loc="left", fontweight="bold")
        stamp(
            fig,
            f"all bars use the same rating profile; n={visible['rated_entities'].iloc[0]}; "
            "missing bars mean the metric is not estimable or the classifier does not emit that field",
        )
        outputs += save_figure(fig, out_dir, f"{number:02d}_{stem}", formats, dpi)
    available_classifications = {
        label: path for label, path in classification_inputs.items()
        if label in loaded and path.is_file()
    }
    if len(available_classifications) >= 2:
        outputs += build_models(
            source, available_classifications, out_dir / "decision_agreement", config,
        )
    LOG.info(
        "classifier performance comparison complete source=%s classifiers=%d files=%d labels=%s",
        source, len(labels), len(outputs), ",".join(labels),
    )
    return outputs
