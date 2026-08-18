from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ids_meta.plotting.common import configure_style, load_frame, metadata_caption, save_figure, stamp, write_table
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
