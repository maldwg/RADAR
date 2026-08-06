#!/usr/bin/env python3
"""Compare keyword and LLM classifications of OpenAlex IDS dataset findings.

The keyword catalogue is the accepted output of
``openalex_dataset_discovery.py``.  The LLM input must be the *wide* audit
output of ``openalex-llm-dataset-classifier.py`` so exclusions and uncertain
decisions are represented honestly.

Example:
    python3 figure_creation/dataset-classifier-comparison/compare_keyword_llm_dataset_classifiers.py

    python3 figure_creation/dataset-classifier-comparison/compare_keyword_llm_dataset_classifiers.py \
      --llm qwen=data-gathering/ids_datasets_openalex.llm_classified.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEYWORD = ROOT / "data-gathering" / "ids_datasets_openalex.csv"
DEFAULT_LLM = ROOT / "data-gathering" / "ids_datasets_openalex.llm_classified.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "output"
UNKNOWN = "Unknown"
PALETTE = ["#2F6B9A", "#4E9A62", "#D9902F", "#7A5FA8", "#C94F4F", "#3B8C8C", "#B85C9E", "#8C6D31"]
DECISION_COLORS = {"accept": "#2F8F5B", "exclude": "#C94F4F", "uncertain": "#D9902F", "error": "#6C7A89", "dry_run": "#AAB3BC", "not_accepted": "#AAB3BC"}
OVERLAP_COLORS = {"both_accept": "#238B57", "keyword_only_accept": "#7055A1", "llm_only_accept": "#D05A35", "neither_or_uncertain": "#D9902F"}


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def clean(value: Any, default: str = "") -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip() or default


def key(value: Any) -> str:
    return clean(value).casefold()


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def shorten(value: Any, limit: int = 44) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:max(1, limit - 1)] + "…"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg(width: int, height: int, title: str, body: Iterable[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{esc(title)}</title><style>
text{{font-family:Arial,Helvetica,sans-serif;fill:#263238}} .title{{font-size:21px;font-weight:700}} .sub{{font-size:12px;fill:#5f6c76}} .axis{{font-size:11px;fill:#43505c}} .small{{font-size:10px;fill:#5f6c76}} .value{{font-size:11px;font-weight:700}} .grid{{stroke:#d8dee6;stroke-width:1}}
</style><rect x="5" y="5" width="{width - 10}" height="{height - 10}" rx="8" fill="#fff" stroke="#d8dee6"/>{''.join(body)}</svg>'''


def write_svg(out_dir: Path, name: str, width: int, height: int, title: str, body: Iterable[str]) -> None:
    (out_dir / name).write_text(svg(width, height, title, body), encoding="utf-8")


def parse_llm_argument(value: str, position: int) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if label.strip() and path.strip():
            return label.strip(), Path(path.strip())
    path = Path(value)
    return path.stem.replace("ids_datasets_openalex.", "").replace("_classified", "") or f"llm_{position}", path


def load_keyword(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for raw in load_csv(path):
        identifier = key(raw.get("dataset_id"))
        if identifier:
            result[identifier] = {"dataset_id": raw.get("dataset_id", ""), "name": raw.get("name", ""), "keyword_decision": "accept", **raw}
    return result


def load_llm(path: Path) -> dict[str, dict[str, str]]:
    result = {}
    for raw in load_csv(path):
        identifier = key(raw.get("dataset_id"))
        if not identifier:
            continue
        decision = key(raw.get("decision"))
        if decision not in DECISION_COLORS:
            decision = "error"
        result[identifier] = {"dataset_id": raw.get("dataset_id", ""), "name": raw.get("name", ""), "llm_decision": decision, **raw}
    return result


def comparison_rows(keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    rows = []
    for identifier in sorted(set(keyword) | set(llm)):
        old, new = keyword.get(identifier, {}), llm.get(identifier, {})
        keyword_decision = "accept" if old else "not_accepted"
        llm_decision = new.get("llm_decision", "not_accepted")
        if keyword_decision == "accept" and llm_decision == "accept":
            overlap = "both_accept"
        elif keyword_decision == "accept":
            overlap = "keyword_only_accept"
        elif llm_decision == "accept":
            overlap = "llm_only_accept"
        else:
            overlap = "neither_or_uncertain"
        rows.append({
            "dataset_id": old.get("dataset_id") or new.get("dataset_id", ""), "name": old.get("name") or new.get("name", ""),
            "keyword_decision": keyword_decision, "llm_decision": llm_decision, "overlap": overlap,
            "keyword_artifact_type": old.get("artifact_type", ""), "llm_artifact_type": new.get("artifact_type", ""),
            "keyword_domain": old.get("domain", ""), "llm_domain": new.get("domain", ""),
            "keyword_data_form": old.get("data_form", ""), "llm_data_form": new.get("data_form", ""),
            "keyword_temporal_type": old.get("temporal_type", ""), "llm_temporal_type": new.get("temporal_type", ""),
            "keyword_environment": old.get("environment", ""), "llm_environment": new.get("environment", ""),
            "keyword_labeling": old.get("labeling", ""), "llm_labeling": new.get("labeling", ""),
            "llm_confidence": new.get("dataset_confidence", ""), "llm_reason": new.get("primary_reason", ""),
            "llm_evidence_quote_verified": new.get("evidence_quote_verified", ""),
        })
    return rows


def bar_figure(out_dir: Path, name: str, title: str, subtitle: str, series: list[tuple[str, Counter[str]]], order: list[str], colors: dict[str, str]) -> None:
    width, left, top, row_height = 920, 235, 112, 66
    height = top + len(series) * row_height + 58
    maximum = max((sum(counter.values()) for _, counter in series), default=1)
    body = [f'<text class="title" x="{width / 2}" y="33" text-anchor="middle">{esc(title)}</text>', f'<text class="sub" x="{width / 2}" y="54" text-anchor="middle">{esc(subtitle)}</text>']
    for index, (label, counts) in enumerate(series):
        y, x = top + index * row_height, left
        body.append(f'<text class="axis" x="{left - 12}" y="{y + 22}" text-anchor="end">{esc(label)}</text>')
        for category in order:
            count = counts.get(category, 0)
            if not count:
                continue
            width_part = count * 620 / maximum
            body.append(f'<rect x="{x}" y="{y}" width="{width_part}" height="30" fill="{colors.get(category, "#6C7A89")}"/>')
            if width_part > 25:
                body.append(f'<text class="small" x="{x + width_part / 2}" y="{y + 20}" text-anchor="middle" fill="#fff">{count}</text>')
            x += width_part
        body.append(f'<text class="value" x="{left + 635}" y="{y + 21}">{sum(counts.values())}</text>')
    x = left
    for category in order:
        body.append(f'<rect x="{x}" y="{height - 28}" width="12" height="12" fill="{colors.get(category, "#6C7A89")}"/>')
        body.append(f'<text class="small" x="{x + 17}" y="{height - 18}">{esc(category.replace("_", " "))}</text>')
        x += max(105, 22 + len(category) * 6)
    write_svg(out_dir, name, width, height, title, body)


def matrix_figure(out_dir: Path, name: str, title: str, subtitle: str, pairs: list[tuple[str, str]]) -> None:
    row_values = sorted({left or UNKNOWN for left, _ in pairs})
    col_values = sorted({right or UNKNOWN for _, right in pairs})
    cell, left, top = 83, 180, 116
    width, height = left + cell * len(col_values) + 42, top + cell * len(row_values) + 58
    counts = Counter((a or UNKNOWN, b or UNKNOWN) for a, b in pairs)
    maximum = max(counts.values(), default=1)
    body = [f'<text class="title" x="{width / 2}" y="33" text-anchor="middle">{esc(title)}</text>', f'<text class="sub" x="{width / 2}" y="54" text-anchor="middle">{esc(subtitle)}</text>']
    for j, value in enumerate(col_values):
        body.append(f'<text class="axis" x="{left + j * cell + cell / 2}" y="{top - 14}" text-anchor="middle">{esc(shorten(value, 13))}</text>')
    for i, value in enumerate(row_values):
        y = top + i * cell
        body.append(f'<text class="axis" x="{left - 9}" y="{y + cell / 2 + 4}" text-anchor="end">{esc(shorten(value, 23))}</text>')
        for j, other in enumerate(col_values):
            count = counts.get((value, other), 0)
            intensity = count / maximum
            color = f"rgb({int(239 - 155 * intensity)},{int(244 - 106 * intensity)},{int(247 - 59 * intensity)})"
            x = left + j * cell
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="3" fill="{color}" stroke="#d8dee6"/>')
            body.append(f'<text class="value" x="{x + (cell - 3) / 2}" y="{y + cell / 2 + 4}" text-anchor="middle">{count}</text>')
    write_svg(out_dir, name, width, height, title, body)


def overlap_figure(out_dir: Path, label: str, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["overlap"] for row in rows)
    width, height, x, y, full = 920, 300, 150, 128, 650
    body = [f'<text class="title" x="{width / 2}" y="33" text-anchor="middle">Keyword–LLM Accepted-set Overlap</text>', f'<text class="sub" x="{width / 2}" y="54" text-anchor="middle">{esc(label)}; rows are all entities seen by either classifier</text>']
    total = sum(counts.values()) or 1
    for index, category in enumerate(("both_accept", "keyword_only_accept", "llm_only_accept", "neither_or_uncertain")):
        value = counts[category]
        w = value * full / total
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="48" fill="{OVERLAP_COLORS[category]}"/>')
        if w > 55:
            body.append(f'<text class="value" x="{x + w / 2}" y="{y + 29}" text-anchor="middle" fill="#fff">{value}</text>')
        body.append(f'<rect x="{150 + index * 185}" y="218" width="12" height="12" fill="{OVERLAP_COLORS[category]}"/>')
        body.append(f'<text class="small" x="{168 + index * 185}" y="228">{esc(category.replace("_", " "))} ({value})</text>')
        x += w
    write_svg(out_dir, f"01_accepted_overlap_{slug(label)}.svg", width, height, "Keyword–LLM Accepted-set Overlap", body)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "llm"


def run_one(label: str, keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]], out_dir: Path) -> None:
    rows = comparison_rows(keyword, llm)
    llm_counts = Counter(row["llm_decision"] for row in rows if row["llm_decision"] != "not_accepted")
    keyword_counts = Counter(row["keyword_decision"] for row in rows)
    suffix = slug(label)
    bar_figure(out_dir, f"00_decision_overview_{suffix}.svg", "Classifier Decision Overview", f"{label}; LLM coverage: {sum(llm_counts.values())}/{len(rows)} entities", [("Keyword catalogue", keyword_counts), (f"LLM: {label}", llm_counts)], ["accept", "exclude", "uncertain", "error", "dry_run", "not_accepted"], DECISION_COLORS)
    overlap_figure(out_dir, label, rows)
    both = [row for row in rows if row["overlap"] == "both_accept"]
    for index, (title, old_key, new_key) in enumerate((
        ("Artifact Type Agreement", "keyword_artifact_type", "llm_artifact_type"),
        ("Domain Agreement", "keyword_domain", "llm_domain"),
        ("Data-form Agreement", "keyword_data_form", "llm_data_form"),
        ("Temporal Type Agreement", "keyword_temporal_type", "llm_temporal_type"),
        ("Environment Agreement", "keyword_environment", "llm_environment"),
        ("Labeling Agreement", "keyword_labeling", "llm_labeling"),
    ), start=2):
        matrix_figure(out_dir, f"{index:02d}_{slug(title)}_{suffix}.svg", title, f"Keyword rows (vertical) vs LLM rows (horizontal); both accepted only, n={len(both)}", [(clean(row[old_key], UNKNOWN), clean(row[new_key], UNKNOWN)) for row in both])
    disagreements = [row for row in rows if row["overlap"] in {"keyword_only_accept", "llm_only_accept"}]
    write_csv(out_dir / f"dataset_classifier_disagreements_{suffix}.csv", disagreements, list(rows[0]) if rows else ["dataset_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare keyword and LLM OpenAlex IDS-dataset classification results.")
    parser.add_argument("--keyword", type=Path, default=DEFAULT_KEYWORD, help="Keyword accepted catalogue from openalex_dataset_discovery.py.")
    parser.add_argument("--llm", action="append", default=None, help="LLM audit file, optionally label=path. Repeatable.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    configure_csv_limit()
    args = parse_args()
    if not args.keyword.is_file():
        print(f"Keyword catalogue does not exist: {args.keyword}", file=sys.stderr)
        return 2
    llm_args = args.llm or [str(DEFAULT_LLM)]
    parsed = [parse_llm_argument(value, i + 1) for i, value in enumerate(llm_args)]
    missing = [str(path) for _, path in parsed if not path.is_file()]
    if missing:
        print("LLM audit CSV does not exist: " + ", ".join(missing), file=sys.stderr)
        return 2
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    keyword = load_keyword(args.keyword)
    for label, path in parsed:
        run_one(label, keyword, load_llm(path), out_dir)
    (out_dir / "README.md").write_text(
        "# Keyword–LLM OpenAlex dataset comparison\n\n"
        "`00_decision_overview_*.svg` reports classifier coverage and decisions. "
        "`01_accepted_overlap_*.svg` compares accepted sets. The remaining matrices compare taxonomy labels only where both approaches accepted the same entity. "
        "`dataset_classifier_disagreements_*.csv` lists one-sided accepted decisions for manual review.\n",
        encoding="utf-8",
    )
    print(f"Saved comparison figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
