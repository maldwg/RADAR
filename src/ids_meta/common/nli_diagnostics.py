from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from ids_meta.common.config import ConfigError, Settings
from ids_meta.common.csvio import atomic_write_csv, read_csv
from ids_meta.common.evaluation import RatingProfile, evaluate_results
from ids_meta.common.logging import logger
from ids_meta.common.nli import (
    NLIProfile, NLIRuntimeConfig, nli_profile_hash, rescore_nli_rows,
)
from ids_meta.plotting.common import configure_style, save_figure, stamp, write_table


LOG = logger("nli.diagnostics")


@dataclass(frozen=True)
class NLIThresholdGrid:
    """Predeclared score-replay grid for descriptive threshold diagnostics."""

    enabled: bool
    entailment_thresholds: tuple[float, ...]
    label_margins: tuple[float, ...]


def load_nli_threshold_grid(settings: Settings) -> NLIThresholdGrid:
    """Load and validate a finite, explicitly configured offline sensitivity grid."""
    raw_nli = settings.section("baselines").get("nli", {})
    if not isinstance(raw_nli, dict):
        raise ConfigError("baselines.nli must be a table")
    raw = raw_nli.get("sensitivity", {})
    if not isinstance(raw, dict):
        raise ConfigError("baselines.nli.sensitivity must be a table")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ConfigError("baselines.nli.sensitivity.enabled must be true or false")

    def values(name: str, default: list[float], *, allow_zero: bool) -> tuple[float, ...]:
        """Normalize one threshold axis into unique ascending numeric values."""
        supplied = raw.get(name, default)
        if not isinstance(supplied, list) or not supplied or any(isinstance(item, bool) for item in supplied):
            raise ConfigError(f"baselines.nli.sensitivity.{name} must be a non-empty numeric array")
        try:
            parsed = tuple(dict.fromkeys(float(item) for item in supplied))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"baselines.nli.sensitivity.{name} must be a non-empty numeric array"
            ) from exc
        lower_ok = (lambda value: value >= 0) if allow_zero else (lambda value: value > 0)
        if any(not lower_ok(value) or value >= 1 for value in parsed):
            interval = "[0, 1)" if allow_zero else "(0, 1)"
            raise ConfigError(f"baselines.nli.sensitivity.{name} values must lie in {interval}")
        return tuple(sorted(parsed))

    return NLIThresholdGrid(
        enabled=enabled,
        entailment_thresholds=values(
            "entailment_thresholds", [.35, .40, .45, .50, .55, .60, .65],
            allow_zero=False,
        ),
        label_margins=values(
            "label_margins", [0, .05, .10, .15, .20], allow_zero=True,
        ),
    )


def _columns(rows: Sequence[dict[str, str]]) -> list[str]:
    """Retain source schema order and append any replay-only fields deterministically."""
    ordered: list[str] = []
    for row in rows:
        for name in row:
            if name not in ordered:
                ordered.append(name)
    return ordered


def plot_nli_threshold_sensitivity(
    source: str, frame: pd.DataFrame, out_dir: Path, plot_config: dict[str, object],
) -> list[Path]:
    """Render persisted sensitivity values without rescoring or running inference."""
    configure_style(plot_config)
    formats = [str(value) for value in plot_config.get("formats", ["svg", "pdf"])]
    dpi = int(plot_config.get("dpi", 160))
    panels = [
        ("detection_rate", "Detection rate ↑", "higher"),
        (
            "false_positive_rate" if source == "github" else "definitive_coverage",
            "False-positive rate ↓" if source == "github" else "Definitive coverage ↑",
            "lower" if source == "github" else "higher",
        ),
        ("definitive_coverage", "Definitive coverage ↑", "higher"),
        ("abstention_rate", "Abstention rate ↓", "lower"),
        ("field_accuracy", "Taxonomy exact accuracy ↑", "higher"),
        ("field_macro_f1", "Taxonomy macro-F1 ↑", "higher"),
    ]
    if source != "github":
        panels[2] = ("classification_coverage", "Dataset identified ↑", "higher")
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), constrained_layout=False)
    threshold_values = sorted(frame["entailment_threshold"].unique())
    margin_values = sorted(frame["label_margin"].unique())
    operating_points = frame[frame["configured_operating_point"].astype(str).str.casefold().eq("true")]
    for ax, (metric, title, direction) in zip(axes.flat, panels):
        pivot = frame.pivot(
            index="label_margin", columns="entailment_threshold", values=metric,
        ).reindex(index=margin_values, columns=threshold_values)
        cmap = plt.get_cmap("Blues" if direction == "higher" else "Reds").copy()
        cmap.set_bad("#E6E8EB")
        image = ax.imshow(
            pivot.astype(float).values, vmin=0, vmax=1,
            cmap=cmap, aspect="auto", origin="lower",
        )
        ax.set_xticks(range(len(threshold_values)), labels=[f"{value:g}" for value in threshold_values])
        ax.set_yticks(range(len(margin_values)), labels=[f"{value:g}" for value in margin_values])
        ax.set_xlabel("Minimum label evidence (entailment)")
        ax.set_ylabel("Minimum lead over runner-up")
        ax.set_title(title, loc="center", fontweight="bold", fontsize=10)
        for row_index in range(len(margin_values)):
            for column_index in range(len(threshold_values)):
                if pd.isna(pivot.iloc[row_index, column_index]):
                    ax.text(column_index, row_index, "N/A", ha="center", va="center", fontsize=7)
        for point in operating_points.itertuples():
            column_index = threshold_values.index(point.entailment_threshold)
            row_index = margin_values.index(point.label_margin)
            ax.add_patch(Rectangle(
                (column_index - .5, row_index - .5), 1, 1,
                fill=False, edgecolor="#111111", linewidth=2.2,
            ))
        fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.suptitle(
        f"{'OpenAlex' if source == 'openalex' else 'GitHub'} — What changes when NLI abstention cutoffs move?",
        x=.5, ha="center", fontsize=16, fontweight="bold",
    )
    stamp(
        fig,
        "move right/up = demand stronger, less ambiguous NLI evidence; black outline=configured cutoffs; "
        "offline sensitivity check only—not another classifier and not automatic tuning",
    )
    fig.subplots_adjust(left=.08, right=.97, top=.89, bottom=.11, wspace=.32, hspace=.38)
    return save_figure(fig, out_dir, "00_threshold_sensitivity", formats, dpi)


def build_nli_threshold_diagnostics(
    source: str, classifications_path: Path, candidates_path: Path,
    rating_profile: RatingProfile, nli_profile: NLIProfile,
    runtime_config: NLIRuntimeConfig, grid: NLIThresholdGrid,
    out_dir: Path, plot_config: dict[str, object], *, confidence_level: float = .95,
    adapter_config: dict[str, Any] | None = None,
) -> list[Path]:
    """Evaluate a threshold grid by replaying saved NLI scores without inference."""
    if not grid.enabled:
        return []
    source_rows = read_csv(classifications_path)
    if not source_rows:
        raise ValueError(f"NLI sensitivity requires non-empty classifications: {classifications_path}")
    if not any(str(row.get("nli_scores_json", "")).strip() not in {"", "{}"} for row in source_rows):
        raise ValueError(f"NLI sensitivity requires persisted nli_scores_json: {classifications_path}")
    incompatible = 0
    for row in source_rows:
        if str(row.get("nli_scores_json", "")).strip() in {"", "{}"}:
            continue
        try:
            stored_config = replace(
                runtime_config,
                entailment_threshold=float(row["nli_entailment_threshold"]),
                label_margin=float(row["nli_label_margin"]),
            )
        except (KeyError, TypeError, ValueError):
            incompatible += 1
            continue
        incompatible += int(row.get("classifier_profile_hash") != nli_profile_hash(
            nli_profile, stored_config,
        ))
    if incompatible:
        raise ValueError(
            f"NLI sensitivity found {incompatible} score rows incompatible with the active "
            "model, hypotheses, or chunking configuration"
        )
    result_rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ids-meta-nli-replay-") as directory:
        temporary_root = Path(directory)
        for index, (threshold, margin) in enumerate(product(
            grid.entailment_thresholds, grid.label_margins,
        )):
            rescored = rescore_nli_rows(
                source, source_rows, nli_profile, threshold, margin, adapter_config,
            )
            replay_path = temporary_root / f"classifications-{index}.csv"
            atomic_write_csv(replay_path, rescored, _columns(rescored))
            evaluated = evaluate_results(
                rating_profile, replay_path, temporary_root / f"evaluation-{index}",
                candidates_path, confidence_level=confidence_level,
            )
            operational = evaluated.metrics["operational"]
            unavailable = set(evaluated.metrics.get("not_estimable_metrics", []))
            result_rows.append({
                "entailment_threshold": threshold,
                "label_margin": margin,
                "configured_operating_point": (
                    threshold == runtime_config.entailment_threshold
                    and margin == runtime_config.label_margin
                ),
                "rated_entities": evaluated.metrics["rated_entities"],
                "accept": sum(row.get("decision") == "accept" for row in rescored),
                "uncertain": sum(row.get("decision") == "uncertain" for row in rescored),
                "exclude": sum(row.get("decision") == "exclude" for row in rescored),
                "error": sum(row.get("decision") == "error" for row in rescored),
                "detection_rate": None if "detection_rate" in unavailable else operational["DR"],
                "false_positive_rate": None if "false_positive_rate" in unavailable else operational["FPR"],
                "false_discovery_rate": None if "false_discovery_rate" in unavailable else operational["FDR"],
                "accuracy": None if "accuracy" in unavailable else operational["ACC"],
                "precision": None if "precision" in unavailable else operational["PREC"],
                "f1": None if "f1" in unavailable else operational["f1"],
                "definitive_coverage": evaluated.metrics["definitive_coverage"],
                "classification_coverage": evaluated.metrics["classification_coverage"],
                "abstention_rate": evaluated.metrics["abstention_rate"],
                "field_accuracy": evaluated.metrics["classification_field_accuracy"],
                "field_macro_f1": evaluated.metrics["classification_field_macro_f1"],
            })

    frame = pd.DataFrame(result_rows)
    data_dir = out_dir / "data"
    table_path = write_table(frame, data_dir, "nli_threshold_sensitivity")
    outputs: list[Path] = [table_path]
    outputs += plot_nli_threshold_sensitivity(source, frame, out_dir, plot_config)
    readme = out_dir / "README.md"
    readme.write_text(
        "# NLI cutoff sensitivity\n\n"
        "This directory does **not** contain another classifier result. It replays the NLI "
        "scores already saved for this one model and asks how its results would change under "
        "different abstention cutoffs. No model inference or network access occurs.\n\n"
        "## How to read the figure\n\n"
        "- An entailment score estimates how strongly the text supports a hypothesis such as "
        "‘this repository is a detector’. It is not a probability that the final classification is correct.\n"
        "- Columns are the minimum entailment score required before emitting a label.\n"
        "- Rows are the minimum lead required between the best and second-best labels.\n"
        "- Moving right or upward is stricter and usually creates more abstentions.\n"
        f"- The black outline is the reported setting: score at least "
        f"{runtime_config.entailment_threshold:g} and lead at least "
        f"{runtime_config.label_margin:g}.\n"
        "- Blue panels prefer darker cells; red error-rate panels prefer lighter cells.\n"
        "- The figure shows robustness and trade-offs. It must not be used to tune thresholds on "
        "the same evaluation fixture and then report the tuned score as an unbiased result.\n\n"
        "Exact values for every tested pair are in `data/nli_threshold_sensitivity.csv`.\n",
        encoding="utf-8",
    )
    outputs.append(readme)
    LOG.info(
        "NLI threshold diagnostics complete source=%s combinations=%d files=%d output=%s",
        source, len(frame), len(outputs), out_dir,
    )
    return outputs
