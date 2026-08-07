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
