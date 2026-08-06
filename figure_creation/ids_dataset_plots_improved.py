#!/usr/bin/env python3
"""Create readable SVG figures from an OpenAlex IDS dataset CSV.

Dependency-light by design: standard library only, SVG written directly.
Improvements over the original version:
- wrapped/adaptive heatmap labels to avoid overlap
- scatter plot point labels with simple collision avoidance
- heuristic-vs-citation diagnostic plots
- citation-bin score plot, residual plot, method distribution plot, overview dashboard
- Pearson/Spearman summary in README
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data-gathering" / "ids_datasets_openalex.csv"
DEFAULT_OUT_DIR = ROOT / "figure_creation" / "dataset_plots"

UNKNOWN = "Unknown"
PLOT_DATE = "Data acquired from input CSV"

PALETTE = [
    "#2F6B9A", "#C94F4F", "#4E9A62", "#D9902F", "#7A5FA8",
    "#3B8C8C", "#B85C9E", "#6C7A89", "#8C6D31", "#4F78C4",
]

ARTIFACT_COLORS = {
    "Dataset": "#2F6B9A",
    "Generator/framework": "#D9902F",
    "Dataset + generator/framework": "#7A5FA8",
    UNKNOWN: "#6C7A89",
}

DISCOVERY_METHOD_COLORS = {
    "Dataset query": "#2F6B9A",
    "Trusted references": "#4E9A62",
    "Alias query": "#D9902F",
    "Survey PDF table": "#7A5FA8",
    "Survey PDF text": "#3B8C8C",
    "Survey metadata": "#C94F4F",
    "Trusted input": "#8C6D31",
    "Survey query": "#4F78C4",
    "Trusted title search": "#5E7CE2",
    "Candidate audit": "#6C7A89",
    UNKNOWN: "#6C7A89",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def as_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(str(value))
    except (TypeError, ValueError):
        return default


def clean_category(value: object) -> str:
    text = str(value or "").strip()
    if text in {"IoT", "IIoT"}:
        return "IoT/IIoT"
    return text if text else UNKNOWN


def shorten(value: object, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def wrap_text(value: object, limit: int = 14, max_lines: int = 3) -> list[str]:
    words = str(value or "").replace("/", "/ ").replace("-", "- ").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = shorten(lines[-1], max(4, limit - 1))
    return lines


def row_name(row: dict[str, str]) -> str:
    return row.get("name") or row.get("canonical_title") or row.get("dataset_id") or UNKNOWN


def row_description(row: dict[str, str], limit: int = 280) -> str:
    return shorten(
        row.get("description", "")
        or row.get("canonical_title", "")
        or row.get("mention_contexts", ""),
        limit,
    )


def row_year(row: dict[str, str]) -> int:
    return as_int(row.get("release_year") or row.get("year") or row.get("canonical_publication_year"), -1)


def row_citations(row: dict[str, str]) -> int:
    return max(0, as_int(row.get("citations"), 0))


def row_score(row: dict[str, str], default: float = -1.0) -> float:
    return as_float(row.get("canonical_score"), default)


def readable_discovery_method(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN
    key = text.casefold().replace("-", "_")
    labels = {
        "survey_metadata": "Survey metadata",
        "survey_pdf_table": "Survey PDF table",
        "survey_pdf_text": "Survey PDF text",
        "trusted_reference": "Trusted references",
        "dataset_publication_search": "Dataset query",
        "candidate_resolution": "Alias query",
        "survey_search": "Survey query",
        "trusted_input": "Trusted input",
        "trusted_title_search": "Trusted title search",
        "candidate_reconstruction": "Candidate audit",
    }
    return labels.get(key, text)


def row_discovery_methods(row: dict[str, str]) -> list[str]:
    raw = row.get("discovery_methods") or row.get("discovery_method") or row.get("mention_sources") or ""
    methods = [
        readable_discovery_method(part)
        for part in raw.replace(",", ";").split(";")
        if part.strip()
    ]
    return list(dict.fromkeys(method for method in methods if method and method != UNKNOWN))


def row_discovery_method(row: dict[str, str]) -> str:
    explicit = readable_discovery_method(row.get("discovery_method"))
    if explicit != UNKNOWN:
        return explicit
    methods = row_discovery_methods(row)
    priority = {
        "Dataset query": 0,
        "Trusted references": 1,
        "Alias query": 2,
        "Survey PDF table": 3,
        "Survey PDF text": 4,
        "Survey metadata": 5,
        "Trusted input": 6,
        "Survey query": 7,
        "Trusted title search": 8,
        "Candidate audit": 9,
    }
    return sorted(methods, key=lambda method: (priority.get(method, 99), method))[0] if methods else UNKNOWN


def count_values(rows: Iterable[dict[str, str]], column: str) -> Counter[str]:
    return Counter(clean_category(row.get(column)) for row in rows)


def top_categories(counter: Counter[str], limit: int | None = None) -> list[tuple[str, int]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return items if limit is None else items[:limit]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = rank
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(rank_values(xs), rank_values(ys))


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return 0.0, statistics.fmean(ys) if ys else 0.0
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return slope, my - slope * mx


# ---------------------------------------------------------------------------
# SVG helper
# ---------------------------------------------------------------------------

class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            "<style>",
            "text{font-family:Arial,Helvetica,sans-serif;fill:#17212b}",
            ".title{font-size:22px;font-weight:700}",
            ".subtitle{font-size:12px;fill:#53616f}",
            ".axis{stroke:#9aa5b1;stroke-width:1}",
            ".grid{stroke:#d7dde5;stroke-width:1;stroke-dasharray:4 4}",
            ".label{font-size:11px}",
            ".small{font-size:10px;fill:#53616f}",
            ".tiny{font-size:9px;fill:#53616f}",
            ".tick{font-size:10px;fill:#53616f}",
            ".number{font-size:11px;font-weight:700;fill:#17212b}",
            "</style>",
        ]

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", opacity: float = 1.0, rx: float = 0) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
            f'rx="{rx:.2f}" fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str = "axis", color: str | None = None, width: float | None = None) -> None:
        color_attr = f' stroke="{color}"' if color else ""
        width_attr = f' stroke-width="{width:.2f}"' if width is not None else ""
        self.items.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{cls}"{color_attr}{width_attr}/>' )

    def polyline(self, points: Sequence[tuple[float, float]], color: str, width: float = 2.0, opacity: float = 1.0) -> None:
        if len(points) < 2:
            return
        point_text = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.items.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{width:.2f}" opacity="{opacity:.3f}"/>')

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "#ffffff", opacity: float = 0.9, title: str = "") -> None:
        title_node = f"<title>{esc(title)}</title>" if title else ""
        self.items.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1" opacity="{opacity:.3f}">{title_node}</circle>'
        )

    def square(self, x: float, y: float, size: float, fill: str, stroke: str = "#ffffff", opacity: float = 0.9, title: str = "") -> None:
        half = size / 2
        title_node = f"<title>{esc(title)}</title>" if title else ""
        self.items.append(
            f'<rect x="{x - half:.2f}" y="{y - half:.2f}" width="{size:.2f}" height="{size:.2f}" '
            f'fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}">{title_node}</rect>'
        )

    def triangle(self, x: float, y: float, size: float, fill: str, stroke: str = "#ffffff", opacity: float = 0.9, title: str = "") -> None:
        half = size / 2
        points = f"{x:.2f},{(y - half):.2f} {(x - half):.2f},{(y + half):.2f} {(x + half):.2f},{(y + half):.2f}"
        title_node = f"<title>{esc(title)}</title>" if title else ""
        self.items.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="1" opacity="{opacity:.3f}">{title_node}</polygon>'
        )

    def text(self, x: float, y: float, text: object, cls: str = "label", anchor: str = "start", rotate: float | None = None) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.items.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{transform}>{esc(text)}</text>')

    def multiline_text(self, x: float, y: float, lines: Sequence[str], cls: str = "small", anchor: str = "middle", line_h: float = 11.0, rotate: float | None = None) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.items.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{transform}>')
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else line_h
            self.items.append(f'<tspan x="{x:.2f}" dy="{dy:.2f}">{esc(line)}</tspan>')
        self.items.append("</text>")

    def label_box(self, x: float, y: float, text: str, cls: str = "tiny", anchor: str = "start") -> None:
        w = min(172, max(42, len(text) * 5.3))
        x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
        self.rect(x0 - 3, y - 11, w + 6, 14, "#ffffff", stroke="#d7dde5", opacity=0.92, rx=3)
        self.text(x, y, text, cls=cls, anchor=anchor)

    def title(self, title: str, subtitle: str | None = None) -> None:
        self.text(28, 34, title, cls="title")
        if subtitle:
            self.text(28, 54, subtitle, cls="subtitle")

    def footer(self, source: str) -> None:
        self.text(28, self.height - 18, source, cls="small")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.items, "</svg>\n"]), encoding="utf-8")


class LabelPlacer:
    def __init__(self, min_x: float, max_x: float, min_y: float, max_y: float) -> None:
        self.bounds = (min_x, max_x, min_y, max_y)
        self.boxes: list[tuple[float, float, float, float]] = []

    def _box(self, x: float, y: float, text: str, anchor: str) -> tuple[float, float, float, float]:
        w = min(172, max(42, len(text) * 5.3))
        if anchor == "end":
            return (x - w - 3, y - 12, x + 3, y + 3)
        if anchor == "middle":
            return (x - w / 2 - 3, y - 12, x + w / 2 + 3, y + 3)
        return (x - 3, y - 12, x + w + 3, y + 3)

    def _fits(self, box: tuple[float, float, float, float]) -> bool:
        min_x, max_x, min_y, max_y = self.bounds
        if box[0] < min_x or box[2] > max_x or box[1] < min_y or box[3] > max_y:
            return False
        return all(box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3] for old in self.boxes)

    def _penalty(self, box: tuple[float, float, float, float]) -> float:
        min_x, max_x, min_y, max_y = self.bounds
        penalty = 0.0
        if box[0] < min_x:
            penalty += (min_x - box[0]) ** 2
        if box[2] > max_x:
            penalty += (box[2] - max_x) ** 2
        if box[1] < min_y:
            penalty += (min_y - box[1]) ** 2
        if box[3] > max_y:
            penalty += (box[3] - max_y) ** 2
        for old in self.boxes:
            overlap_x = max(0.0, min(box[2], old[2]) - max(box[0], old[0]))
            overlap_y = max(0.0, min(box[3], old[3]) - max(box[1], old[1]))
            penalty += overlap_x * overlap_y * 10
        return penalty

    def place(self, x: float, y: float, text: str, *, force: bool = False) -> tuple[float, float, str] | None:
        candidates = [
            (x + 9, y - 8, "start"), (x + 9, y + 15, "start"),
            (x - 9, y - 8, "end"), (x - 9, y + 15, "end"),
            (x, y - 17, "middle"), (x, y + 25, "middle"),
            (x + 16, y + 3, "start"), (x - 16, y + 3, "end"),
        ]
        for cx, cy, anchor in candidates:
            box = self._box(cx, cy, text, anchor)
            if self._fits(box):
                self.boxes.append(box)
                return cx, cy, anchor
        if force:
            cx, cy, anchor = min(candidates, key=lambda candidate: self._penalty(self._box(*candidate[:2], text, candidate[2])))
            self.boxes.append(self._box(cx, cy, text, anchor))
            return cx, cy, anchor
        return None


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def candidate_path_for(path: Path) -> Path:
    stem = path.with_suffix("")
    suffix = path.suffix or ".csv"
    return stem.with_name(f"{stem.name}_candidates").with_suffix(suffix)


def fallback_input_path(path: Path) -> Path | None:
    if path.exists() and path.stat().st_size > 0:
        return path
    if path == DEFAULT_INPUT:
        test_path = path.with_name("ids_datasets_openalex_test.csv")
        if test_path.exists() and test_path.stat().st_size > 0:
            return test_path
    return None


def reconstruct_from_candidates(path: Path) -> list[dict[str, str]]:
    candidates = candidate_path_for(path)
    if not candidates.exists():
        return []
    rows = read_csv_rows(candidates)
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        if clean_category(row.get("status")) != "canonical_resolved":
            continue
        dataset_id = row.get("dataset_id", "")
        if not dataset_id:
            continue
        old = best.get(dataset_id)
        if old is None or as_float(row.get("score")) > as_float(old.get("canonical_score")):
            best[dataset_id] = {
                "dataset_id": dataset_id,
                "name": row.get("name", ""),
                "aliases": row.get("aliases", ""),
                "artifact_type": row.get("artifact_type", UNKNOWN),
                "status": row.get("status", ""),
                "canonical_openalex_id": row.get("candidate_openalex_id", ""),
                "canonical_title": row.get("candidate_title", ""),
                "description": row.get("candidate_title", ""),
                "release_year": row.get("release_year_hint") or row.get("candidate_year", ""),
                "canonical_publication_year": row.get("candidate_year", ""),
                "year": row.get("candidate_year", ""),
                "release_year_source": "candidate reconstruction",
                "doi": row.get("candidate_doi", ""),
                "url": "",
                "citations": row.get("candidate_citations", ""),
                "survey_count": "1",
                "survey_sources": "",
                "discovery_method": readable_discovery_method(row.get("discovery_method") or "candidate_reconstruction"),
                "discovery_methods": readable_discovery_method(row.get("discovery_method") or "candidate_reconstruction"),
                "mention_sources": "candidate_reconstruction",
                "domain": UNKNOWN,
                "data_form": UNKNOWN,
                "temporal_type": UNKNOWN,
                "environment": UNKNOWN,
                "labeling": UNKNOWN,
                "confidence": "Medium",
                "canonical_score": row.get("score", ""),
                "evidence": row.get("reason", ""),
                "mention_contexts": "",
                "candidate_count": "1",
            }
    return list(best.values())


def load_rows(path: Path) -> tuple[list[dict[str, str]], Path, str]:
    concrete = fallback_input_path(path)
    if concrete:
        return read_csv_rows(concrete), concrete, "accepted_csv"
    reconstructed = reconstruct_from_candidates(path)
    if reconstructed:
        return reconstructed, candidate_path_for(path), "reconstructed_from_candidates"
    raise SystemExit(f"No usable input CSV found at {path} and no candidate fallback found.")


# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

def point_title(row: dict[str, str], extra: Sequence[str] = ()) -> str:
    parts = [
        row_name(row),
        f"artifact={clean_category(row.get('artifact_type'))}",
        f"domain={clean_category(row.get('domain'))}",
        f"citations={row_citations(row)}",
        f"score={row_score(row, 0):.2f}",
        f"survey_count={as_int(row.get('survey_count'))}",
        f"candidate_count={as_int(row.get('candidate_count'))}",
        *extra,
        row_description(row),
    ]
    return " | ".join(str(part) for part in parts if str(part).strip())


def draw_point(svg: Svg, row: dict[str, str], x: float, y: float, radius: float, color: str, title: str, opacity: float = 0.78) -> None:
    artifact = clean_category(row.get("artifact_type"))
    if artifact == "Generator/framework":
        svg.triangle(x, y, radius * 2.3, color, opacity=opacity, title=title)
    elif artifact == "Dataset + generator/framework":
        svg.square(x, y, radius * 1.8, color, opacity=opacity, title=title)
    else:
        svg.circle(x, y, radius, color, opacity=opacity, title=title)


def scatter_axes(svg: Svg, *, left: float, right: float, top: float, bottom: float, max_log_citations: float, min_score: float, max_score: float, x_label: str, y_label: str) -> None:
    for i in range(6):
        x = left + i * (right - left) / 5
        raw = 10 ** (i * max_log_citations / 5) - 1
        label = f"{raw:.0f}" if raw < 1000 else f"{raw / 1000:.1f}k"
        svg.line(x, top, x, bottom, cls="grid")
        svg.text(x, bottom + 18, label, cls="tick", anchor="middle")
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = min_score + i * (max_score - min_score) / 5
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{value:.1f}", cls="tick", anchor="end")
    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text((left + right) / 2, svg.height - 32, x_label, cls="label", anchor="middle")
    svg.text(22, (top + bottom) / 2, y_label, cls="label", anchor="middle", rotate=-90)


def draw_point_label(svg: Svg, placer: LabelPlacer, x: float, y: float, label: str, *, force: bool = True) -> None:
    placed = placer.place(x, y, label, force=force)
    if not placed:
        return
    lx, ly, anchor = placed
    svg.line(x, y, lx, ly - 5, cls="grid", color="#a8b2bd")
    svg.label_box(lx, ly, label, anchor=anchor)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def draw_overview(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    width, height = 1120, 640
    svg = Svg(width, height)
    svg.title("IDS Dataset Discovery Overview", subtitle)

    total = len(rows)
    scored = sum(1 for row in rows if row_score(row) >= 0)
    with_year = sum(1 for row in rows if row_year(row) > 0)
    citations = [row_citations(row) for row in rows]
    scores = [row_score(row) for row in rows if row_score(row) >= 0]

    cards = [
        ("Entities", total),
        ("With score", scored),
        ("With release year", with_year),
        ("Total citations", sum(citations)),
        ("Median citations", f"{percentile(citations, 0.5):.0f}" if citations else "0"),
        ("Median score", f"{percentile(scores, 0.5):.1f}" if scores else "n/a"),
    ]

    card_w, card_h = 160, 72
    for i, (label, value) in enumerate(cards):
        x = 40 + (i % 3) * (card_w + 24)
        y = 84 + (i // 3) * (card_h + 24)
        svg.rect(x, y, card_w, card_h, "#f3f6fa", stroke="#d7dde5", rx=7)
        svg.text(x + 14, y + 24, label, cls="small")
        svg.text(x + 14, y + 55, value, cls="title")

    def small_bar(counter: Counter[str], title: str, x: float, y: float, color_map: dict[str, str] | None = None) -> None:
        items = top_categories(counter, 8)
        svg.text(x, y, title, cls="label")
        max_value = max((value for _, value in items), default=1)
        for j, (label, value) in enumerate(items):
            yy = y + 22 + j * 25
            color = color_map.get(label, PALETTE[j % len(PALETTE)]) if color_map else PALETTE[j % len(PALETTE)]
            svg.text(x, yy + 13, shorten(label, 28), cls="tiny")
            svg.rect(x + 145, yy, 260 * value / max_value, 15, color, opacity=0.85)
            svg.text(x + 410, yy + 12, value, cls="tiny")

    small_bar(count_values(rows, "domain"), "Top domains", 600, 92)
    small_bar(count_values(rows, "artifact_type"), "Artifact types", 600, 330, ARTIFACT_COLORS)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_horizontal_bar_chart(rows: list[dict[str, str]], column: str, title: str, out: Path, *, subtitle: str, limit: int | None = None) -> None:
    items = top_categories(count_values(rows, column), limit)
    max_label_len = max((len(label) for label, _ in items), default=8)
    width = max(980, 500 + max_label_len * 7)
    height = max(260, 100 + len(items) * 34)
    svg = Svg(width, height)
    svg.title(title, subtitle)

    left = min(330, max(170, max_label_len * 7 + 42))
    right = width - 56
    top = 82
    bar_h = 20
    gap = 14
    max_value = max((value for _, value in items), default=1)

    for index, (label, value) in enumerate(items):
        y = top + index * (bar_h + gap)
        color = PALETTE[index % len(PALETTE)]
        svg.text(left - 12, y + 14, shorten(label, 40), cls="label", anchor="end")
        svg.rect(left, y, (right - left) * value / max_value, bar_h, color, opacity=0.9)
        svg.text(left + (right - left) * value / max_value + 8, y + 14, value, cls="label")

    svg.line(left, top - 8, left, top + len(items) * (bar_h + gap) - gap + 8)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_heatmap(rows: list[dict[str, str]], row_col: str, col_col: str, title: str, out: Path, *, subtitle: str, max_rows: int = 18, max_cols: int = 16) -> None:
    row_labels = [item for item, _ in top_categories(count_values(rows, row_col), max_rows)]
    col_labels = [item for item, _ in top_categories(count_values(rows, col_col), max_cols)]
    if not row_labels or not col_labels:
        return

    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        r = clean_category(row.get(row_col))
        c = clean_category(row.get(col_col))
        if r in row_labels and c in col_labels:
            counts[(r, c)] += 1

    row_label_width = min(310, max(160, max(len(label) for label in row_labels) * 6 + 34))
    cell_w = max(72, min(112, 840 / max(1, len(col_labels))))
    cell_h = 48
    header_h = 58
    left = row_label_width
    top = 140
    header_y = top - header_h
    width = int(max(980, left + cell_w * len(col_labels) + 72))
    height = int(max(380, top + cell_h * len(row_labels) + 88))
    svg = Svg(width, height)
    svg.title(title, subtitle)
    max_value = max(counts.values(), default=1)

    svg.rect(left - row_label_width + 14, header_y, row_label_width - 16, header_h - 2, "#e8eef5", stroke="#ffffff")
    svg.text(left - 12, header_y + header_h / 2 + 4, row_col, cls="small", anchor="end")
    for j, label in enumerate(col_labels):
        x = left + j * cell_w
        lines = wrap_text(label, limit=max(8, int(cell_w / 7)), max_lines=4)
        first_line_y = header_y + (header_h - (len(lines) - 1) * 10) / 2 + 3
        svg.rect(x, header_y, cell_w - 2, header_h - 2, "#e8eef5", stroke="#ffffff")
        svg.multiline_text(x + cell_w / 2, first_line_y, lines, cls="tiny", anchor="middle", line_h=10)

    for i, label in enumerate(row_labels):
        y = top + i * cell_h
        row_fill = "#f8fafc" if i % 2 == 0 else "#ffffff"
        svg.rect(left - row_label_width + 14, y, row_label_width - 16, cell_h - 2, "#e8eef5" if i % 2 == 0 else "#eef3f8", stroke="#ffffff")
        svg.rect(left, y, cell_w * len(col_labels), cell_h - 2, row_fill)
        svg.text(left - 12, y + cell_h / 2 + 4, shorten(label, 40), cls="label", anchor="end")
        for j, col in enumerate(col_labels):
            value = counts.get((label, col), 0)
            shade = 0.08 + 0.84 * (value / max_value if max_value else 0)
            color = f"rgb({int(255 - 208 * shade)},{int(255 - 148 * shade)},{int(255 - 101 * shade)})"
            x = left + j * cell_w
            svg.rect(x, y, cell_w - 2, cell_h - 2, color, stroke="#ffffff")
            if value:
                svg.text(x + cell_w / 2, y + cell_h / 2 + 4, value, cls="number", anchor="middle")

    svg.text(left + cell_w * len(col_labels) / 2, header_y - 10, col_col, cls="small", anchor="middle")
    svg.text(26, top + cell_h * len(row_labels) / 2, row_col, cls="small", anchor="middle", rotate=-90)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_year_histogram(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    year_rows = [row for row in rows if row_year(row) > 0]
    if not year_rows:
        return
    years = [row_year(row) for row in year_rows]
    min_year, max_year = min(years), max(years)
    artifacts = [item for item, _ in top_categories(count_values(year_rows, "artifact_type"))]
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in year_rows:
        counts[(row_year(row), clean_category(row.get("artifact_type")))] += 1

    width = max(1080, 96 + (max_year - min_year + 1) * 28)
    height = 480
    left, right, top, bottom = 76, width - 42, 78, height - 78
    svg = Svg(width, height)
    svg.title("Dataset Release Years Over Time", subtitle)
    max_stack = max(sum(counts.get((year, artifact), 0) for artifact in artifacts) for year in range(min_year, max_year + 1))
    plot_h = bottom - top
    step = (right - left) / max(1, max_year - min_year + 1)
    bar_w = min(20, step * 0.72)

    tick_step = max(1, math.ceil(max_stack / 6))
    for tick in range(0, max_stack + 1, tick_step):
        y = bottom - plot_h * tick / max(1, max_stack)
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, tick, cls="tick", anchor="end")

    for year in range(min_year, max_year + 1):
        x = left + (year - min_year + 0.5) * step
        y_cursor = bottom
        for artifact in artifacts:
            value = counts.get((year, artifact), 0)
            if not value:
                continue
            h = plot_h * value / max(1, max_stack)
            y_cursor -= h
            svg.rect(x - bar_w / 2, y_cursor, bar_w, h, ARTIFACT_COLORS.get(artifact, "#6C7A89"), stroke="#ffffff")
        if year % max(1, math.ceil((max_year - min_year + 1) / 14)) == 0 or year in {min_year, max_year}:
            svg.text(x, bottom + 19, year, cls="tick", anchor="middle", rotate=-35)

    legend_x = left
    for artifact in artifacts:
        color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
        svg.rect(legend_x, height - 44, 14, 14, color)
        svg.text(legend_x + 20, height - 32, artifact, cls="small")
        legend_x += max(170, len(artifact) * 7 + 40)

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_timeline(rows: list[dict[str, str]], out: Path, *, subtitle: str, label_top: int = 28) -> None:
    year_rows = [row for row in rows if row_year(row) > 0]
    if not year_rows:
        svg = Svg(980, 300)
        svg.title("Dataset Release Timeline By Domain", subtitle)
        svg.text(80, 145, "No usable release_year, year, or canonical_publication_year values were found.", cls="label")
        svg.text(80, 170, "Regenerate the discovery CSV or inspect release_year_source for missing year evidence.", cls="small")
        svg.footer(PLOT_DATE)
        svg.save(out)
        return
    no_year_count = len(rows) - len(year_rows)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sorted(year_rows, key=lambda item: (clean_category(item.get("domain")), row_year(item), row_name(item))):
        grouped[clean_category(row.get("domain"))].append(row)
    domains = [domain for domain, _ in top_categories(count_values(year_rows, "domain")) if domain in grouped]
    years = [row_year(row) for row in year_rows]
    min_year, max_year = min(years), max(years)
    width = max(1420, 330 + (max_year - min_year + 1) * 38)
    row_gap = 26
    group_gap = 18
    plot_rows = sum(len(grouped[domain]) for domain in domains)
    height = max(620, 154 + plot_rows * row_gap + len(domains) * group_gap + 92)
    left, right, top, bottom = 310, width - 82, 96, height - 104
    svg = Svg(width, height)
    svg.title("Dataset Release Timeline By Domain", subtitle)

    def x_for_year(year: int) -> float:
        return left + (year - min_year) / max(1, max_year - min_year) * (right - left)

    year_tick_step = max(1, math.ceil((max_year - min_year) / 12))
    for year in range(min_year, max_year + 1):
        if year % year_tick_step == 0 or year in {min_year, max_year}:
            x = x_for_year(year)
            svg.line(x, top - 10, x, bottom + 8, cls="grid")
            svg.text(x, bottom + 24, year, cls="tick", anchor="middle", rotate=-35)

    y_cursor = top
    row_index = 0
    for domain in domains:
        domain_rows = grouped[domain]
        domain_top = y_cursor - 13
        domain_height = max(row_gap, len(domain_rows) * row_gap)
        svg.rect(28, domain_top, right - 20, domain_height + 6, "#f8fafc" if row_index % 2 == 0 else "#ffffff")
        svg.text(28, y_cursor + 4, shorten(domain, 28), cls="label")
        for row in domain_rows:
            year = row_year(row)
            artifact = clean_category(row.get("artifact_type"))
            x = x_for_year(year)
            y = y_cursor
            support = as_int(row.get("survey_count")) + as_int(row.get("candidate_count"))
            size = 5.5 + min(8.5, math.log10(row_citations(row) + 1) * 3.0 + min(3.0, math.log(support + 1, 4)))
            color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
            svg.text(left - 14, y + 4, shorten(row_name(row), 34), cls="label", anchor="end")
            svg.line(left, y, right, y, cls="grid")
            draw_point(svg, row, x, y, size, color, point_title(row, [f"year={year}"]))
            svg.text(x + 10 if x < right - 150 else x - 10, y + 4, year, cls="tiny", anchor="start" if x < right - 150 else "end")
            y_cursor += row_gap
            row_index += 1
        y_cursor += group_gap

    if no_year_count:
        svg.text(left, height - 48, f"{no_year_count} accepted row(s) are not shown because no release year is available.", cls="small")

    legend_y = height - 70
    legend_x = left
    for artifact in ("Dataset", "Dataset + generator/framework", "Generator/framework"):
        color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
        if artifact == "Generator/framework":
            svg.triangle(legend_x + 7, legend_y - 5, 14, color)
        elif artifact == "Dataset + generator/framework":
            svg.square(legend_x + 7, legend_y - 5, 13, color)
        else:
            svg.circle(legend_x + 7, legend_y - 5, 7, color)
        svg.text(legend_x + 20, legend_y - 1, artifact, cls="small")
        legend_x += 230

    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_score_citation_scatter(rows: list[dict[str, str]], out: Path, *, subtitle: str, label_top: int = 45) -> None:
    plot_rows = [row for row in rows if row_score(row) >= 0]
    if not plot_rows:
        return
    width, height = 1250, 760
    left, right, top, bottom = 96, width - 80, 90, height - 98
    svg = Svg(width, height)
    svg.title("Heuristic Score Vs Citation Count", subtitle)

    logs = [math.log10(row_citations(row) + 1) for row in plot_rows]
    scores = [row_score(row) for row in plot_rows]
    max_x = max(logs) if logs else 1
    min_y, max_y = min(scores), max(scores)
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    def x_for(row: dict[str, str]) -> float:
        return left + math.log10(row_citations(row) + 1) / max(1e-9, max_x) * (right - left)

    def y_for(row: dict[str, str]) -> float:
        return bottom - (row_score(row) - min_y) / max(1e-9, max_y - min_y) * (bottom - top)

    scatter_axes(svg, left=left, right=right, top=top, bottom=bottom, max_log_citations=max_x, min_score=min_y, max_score=max_y, x_label="Citations, log scale", y_label="Heuristic/canonical score")

    slope, intercept = linear_regression(logs, scores)
    trend_points = []
    for i in range(51):
        xlog = i * max_x / 50
        x = left + xlog / max(1e-9, max_x) * (right - left)
        y_value = slope * xlog + intercept
        y = bottom - (y_value - min_y) / max(1e-9, max_y - min_y) * (bottom - top)
        trend_points.append((x, max(top, min(bottom, y))))
    svg.polyline(trend_points, "#17212b", width=2.0, opacity=0.75)

    placer = LabelPlacer(left, right, top - 20, bottom + 55)
    collisions: dict[tuple[int, int], int] = defaultdict(int)
    point_labels: list[tuple[float, float, str]] = []

    for row in sorted(plot_rows, key=lambda item: (row_citations(item), row_score(item), row_name(item))):
        x, y = x_for(row), y_for(row)
        bucket = (round(x / 18), round(y / 18))
        offset = collisions[bucket]
        collisions[bucket] += 1
        if offset:
            angle = (offset % 10) * 2 * math.pi / 10
            dist = 3.0 * min(5, math.ceil(offset / 10))
            x += math.cos(angle) * dist
            y += math.sin(angle) * dist
        artifact = clean_category(row.get("artifact_type"))
        color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
        radius = 4.8 + min(8, math.sqrt(max(1, as_int(row.get("survey_count")) + as_int(row.get("candidate_count")))))
        draw_point(svg, row, x, y, radius, color, point_title(row))
        point_labels.append((x, y, shorten(row_name(row), 28)))

    for x, y, label in point_labels:
        draw_point_label(svg, placer, x, y, label)

    svg.rect(right - 245, top + 12, 220, 58, "#ffffff", stroke="#d7dde5", opacity=0.95, rx=5)
    svg.text(right - 232, top + 34, f"Pearson r(log citations, score): {pearson(logs, scores):.2f}", cls="small")
    svg.text(right - 232, top + 54, f"Spearman ρ: {spearman(logs, scores):.2f}", cls="small")
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_score_citation_bins(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    plot_rows = [row for row in rows if row_score(row) >= 0]
    if not plot_rows:
        return

    bins = [("0", 0, 0), ("1-9", 1, 9), ("10-49", 10, 49), ("50-99", 50, 99), ("100-249", 100, 249), ("250-499", 250, 499), ("500-999", 500, 999), ("1000+", 1000, 10**12)]
    grouped = [(label, [row_score(row) for row in plot_rows if lo <= row_citations(row) <= hi]) for label, lo, hi in bins]
    grouped = [(label, values) for label, values in grouped if values]
    if not grouped:
        return

    all_scores = [row_score(row) for row in plot_rows]
    min_y, max_y = min(all_scores), max(all_scores)
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    width, height = 1120, 620
    left, right, top, bottom = 90, width - 52, 82, height - 92
    svg = Svg(width, height)
    svg.title("Heuristic Scaling Against Citation Count", subtitle)

    def y_for(value: float) -> float:
        return bottom - (value - min_y) / max(1e-9, max_y - min_y) * (bottom - top)

    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = min_y + i * (max_y - min_y) / 5
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{value:.1f}", cls="tick", anchor="end")

    step = (right - left) / len(grouped)
    med_points: list[tuple[float, float]] = []
    for i, (label, values) in enumerate(grouped):
        x = left + (i + 0.5) * step
        box_w = min(72, step * 0.55)
        q1, med, q3 = percentile(values, 0.25), percentile(values, 0.50), percentile(values, 0.75)
        lo, hi = min(values), max(values)
        y_lo, y_hi = y_for(lo), y_for(hi)
        y_q1, y_q3 = y_for(q1), y_for(q3)
        y_med = y_for(med)
        med_points.append((x, y_med))
        svg.line(x, y_hi, x, y_lo, color="#6C7A89", width=1.2)
        svg.rect(x - box_w / 2, min(y_q1, y_q3), box_w, abs(y_q3 - y_q1), "#DDE7F0", stroke="#2F6B9A", opacity=0.95, rx=3)
        svg.line(x - box_w / 2, y_med, x + box_w / 2, y_med, color="#17212b", width=2)
        svg.text(x, bottom + 22, label, cls="tick", anchor="middle", rotate=-25)
        svg.text(x, bottom + 44, f"n={len(values)}", cls="tiny", anchor="middle")

    svg.polyline(med_points, "#C94F4F", width=2.4, opacity=0.95)
    for x, y in med_points:
        svg.circle(x, y, 4.5, "#C94F4F", stroke="#ffffff")

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text((left + right) / 2, height - 26, "Citation-count bins", cls="label", anchor="middle")
    svg.text(20, (top + bottom) / 2, "Heuristic/canonical score", cls="label", anchor="middle", rotate=-90)
    svg.text(right - 350, top + 22, "Boxes show IQR, red line shows median score.", cls="small")
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_score_citation_residuals(rows: list[dict[str, str]], out: Path, *, subtitle: str, label_top: int = 35) -> None:
    plot_rows = [row for row in rows if row_score(row) >= 0]
    if len(plot_rows) < 3:
        return

    logs = [math.log10(row_citations(row) + 1) for row in plot_rows]
    scores = [row_score(row) for row in plot_rows]
    slope, intercept = linear_regression(logs, scores)
    residuals = [score - (slope * log_cit + intercept) for score, log_cit in zip(scores, logs)]
    max_abs = max(abs(value) for value in residuals) or 1.0
    max_x = max(logs) if logs else 1

    width, height = 1220, 680
    left, right, top, bottom = 96, width - 80, 86, height - 92
    svg = Svg(width, height)
    svg.title("Heuristic Residuals After Citation Trend", subtitle)

    def x_for(log_cit: float) -> float:
        return left + log_cit / max(1e-9, max_x) * (right - left)

    def y_for(residual: float) -> float:
        return bottom - (residual + max_abs) / (2 * max_abs) * (bottom - top)

    for i in range(6):
        x = left + i * (right - left) / 5
        raw = 10 ** (i * max_x / 5) - 1
        label = f"{raw:.0f}" if raw < 1000 else f"{raw / 1000:.1f}k"
        svg.line(x, top, x, bottom, cls="grid")
        svg.text(x, bottom + 18, label, cls="tick", anchor="middle")
    for i in range(5):
        value = -max_abs + i * max_abs / 2
        y = y_for(value)
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{value:.1f}", cls="tick", anchor="end")
    svg.line(left, y_for(0), right, y_for(0), color="#17212b", width=1.4)

    ranked = sorted(zip(plot_rows, logs, residuals), key=lambda item: abs(item[2]), reverse=True)
    placer = LabelPlacer(left, right, top - 20, bottom + 50)
    point_labels: list[tuple[float, float, str]] = []

    for row, log_cit, residual in ranked:
        x, y = x_for(log_cit), y_for(residual)
        color = "#4E9A62" if residual >= 0 else "#C94F4F"
        radius = 5 + min(7, math.sqrt(max(1, row_citations(row))) / 12)
        svg.circle(x, y, radius, color, opacity=0.74, title=point_title(row, [f"residual={residual:.2f}"]))
        point_labels.append((x, y, shorten(row_name(row), 28)))

    for x, y, label in point_labels:
        draw_point_label(svg, placer, x, y, label)

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text((left + right) / 2, height - 28, "Citations, log scale", cls="label", anchor="middle")
    svg.text(20, (top + bottom) / 2, "Score minus expected score from citations", cls="label", anchor="middle", rotate=-90)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_discovery_method_scatter(rows: list[dict[str, str]], out: Path, *, subtitle: str, label_top: int = 45) -> None:
    plot_rows = [row for row in rows if row_score(row) >= 0]
    if not plot_rows:
        return

    methods = [method for method, _ in top_categories(Counter(row_discovery_method(row) for row in plot_rows))]
    width, height = 1320, 780
    left, right, top, bottom = 96, width - 270, 90, height - 98
    svg = Svg(width, height)
    svg.title("Discovery Method Vs Score And Citations", subtitle)

    logs = [math.log10(row_citations(row) + 1) for row in plot_rows]
    scores = [row_score(row) for row in plot_rows]
    max_x = max(logs) if logs else 1
    min_y, max_y = min(scores), max(scores)
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    def x_for(row: dict[str, str]) -> float:
        return left + math.log10(row_citations(row) + 1) / max(1e-9, max_x) * (right - left)

    def y_for(row: dict[str, str]) -> float:
        return bottom - (row_score(row) - min_y) / max(1e-9, max_y - min_y) * (bottom - top)

    scatter_axes(svg, left=left, right=right, top=top, bottom=bottom, max_log_citations=max_x, min_score=min_y, max_score=max_y, x_label="Citations, log scale", y_label="Heuristic/canonical score")

    placer = LabelPlacer(left, right + 150, top - 20, bottom + 50)
    collisions: dict[tuple[int, int], int] = defaultdict(int)
    point_labels: list[tuple[float, float, str]] = []

    for row in sorted(plot_rows, key=lambda item: (row_discovery_method(item), row_name(item))):
        x, y = x_for(row), y_for(row)
        bucket = (round(x / 18), round(y / 18))
        offset = collisions[bucket]
        collisions[bucket] += 1
        if offset:
            angle = (offset % 10) * 2 * math.pi / 10
            radius_offset = 3 * min(5, math.ceil(offset / 10))
            x += math.cos(angle) * radius_offset
            y += math.sin(angle) * radius_offset

        method = row_discovery_method(row)
        color = DISCOVERY_METHOD_COLORS.get(method, "#6C7A89")
        support = max(1, as_int(row.get("survey_count")) + as_int(row.get("candidate_count")))
        radius = 4.8 + min(8, math.log(support + 1, 2) * 2)
        all_methods = "; ".join(row_discovery_methods(row)) or method
        svg.circle(x, y, radius, color, opacity=0.76, title=point_title(row, [f"method={method}", f"all_methods={all_methods}"]))
        point_labels.append((x, y, shorten(row_name(row), 28)))

    for x, y, label in point_labels:
        draw_point_label(svg, placer, x, y, label)

    legend_x = right + 28
    legend_y = top + 5
    svg.text(legend_x, legend_y, "Discovery method", cls="label")
    for index, method in enumerate(methods[:17]):
        y = legend_y + 24 + index * 24
        svg.circle(legend_x + 7, y - 5, 6, DISCOVERY_METHOD_COLORS.get(method, "#6C7A89"))
        svg.text(legend_x + 20, y - 1, shorten(method, 27), cls="small")
    if len(methods) > 17:
        svg.text(legend_x, legend_y + 24 + 17 * 24, f"+ {len(methods) - 17} more", cls="small")

    svg.text(legend_x, height - 90, "Point size", cls="label")
    svg.text(legend_x, height - 70, "survey_count + candidate_count", cls="small")
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_method_score_distribution(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    plot_rows = [row for row in rows if row_score(row) >= 0]
    if not plot_rows:
        return
    method_counts: Counter[str] = Counter()
    for row in plot_rows:
        methods_for_row = row_discovery_methods(row) or [row_discovery_method(row)]
        method_counts.update(method for method in methods_for_row if method != UNKNOWN)
    methods = [method for method, _ in top_categories(method_counts, 14)]
    grouped = [
        (
            method,
            [
                row_score(row)
                for row in plot_rows
                if method in (row_discovery_methods(row) or [row_discovery_method(row)])
            ],
        )
        for method in methods
    ]
    grouped = [(method, values) for method, values in grouped if values]
    if not grouped:
        return

    scores = [row_score(row) for row in plot_rows]
    min_y, max_y = min(scores), max(scores)
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    width = max(1120, 130 + len(grouped) * 82)
    height = 650
    left, right, top, bottom = 86, width - 56, 82, height - 150
    svg = Svg(width, height)
    svg.title("Score Distribution By Discovery Method", subtitle)

    def y_for(value: float) -> float:
        return bottom - (value - min_y) / max(1e-9, max_y - min_y) * (bottom - top)

    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = min_y + i * (max_y - min_y) / 5
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{value:.1f}", cls="tick", anchor="end")

    step = (right - left) / len(grouped)
    for i, (method, values) in enumerate(grouped):
        x = left + (i + 0.5) * step
        q1, med, q3 = percentile(values, 0.25), percentile(values, 0.5), percentile(values, 0.75)
        lo, hi = min(values), max(values)
        box_w = min(58, step * 0.52)
        color = DISCOVERY_METHOD_COLORS.get(method, "#6C7A89")
        svg.line(x, y_for(hi), x, y_for(lo), color="#6C7A89", width=1.1)
        svg.rect(x - box_w / 2, min(y_for(q1), y_for(q3)), box_w, abs(y_for(q3) - y_for(q1)), color, stroke="#ffffff", opacity=0.55, rx=3)
        svg.line(x - box_w / 2, y_for(med), x + box_w / 2, y_for(med), color="#17212b", width=2)
        for k, value in enumerate(sorted(values)):
            jitter = ((k % 7) - 3) * min(3.0, box_w / 14)
            svg.circle(x + jitter, y_for(value), 2.5, color, opacity=0.5)
        svg.multiline_text(x, bottom + 28, wrap_text(method, 12, 4), cls="tiny", anchor="middle", line_h=10)
        svg.text(x, bottom + 78, f"n={len(values)}", cls="tiny", anchor="middle")

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text(20, (top + bottom) / 2, "Heuristic/canonical score", cls="label", anchor="middle", rotate=-90)
    svg.text(right - 360, top + 22, "Rows may appear in multiple method buckets.", cls="small")
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_missingness(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    columns = [
        "release_year", "canonical_publication_year", "domain", "data_form", "temporal_type",
        "environment", "labeling", "confidence", "artifact_type", "canonical_score",
        "citations", "survey_count", "candidate_count", "discovery_method",
    ]
    width, height = 980, max(390, 116 + len(columns) * 26)
    left, right, top = 210, 890, 84
    row_h = 25
    svg = Svg(width, height)
    svg.title("Metadata Completeness", subtitle)
    total = max(1, len(rows))
    for i, column in enumerate(columns):
        known = sum(1 for row in rows if clean_category(row.get(column)) != UNKNOWN and str(row.get(column, "")).strip() != "")
        y = top + i * row_h
        svg.text(left - 10, y + 15, column, cls="label", anchor="end")
        svg.rect(left, y, right - left, 16, "#edf1f5")
        svg.rect(left, y, (right - left) * known / total, 16, "#4E9A62")
        svg.text(right + 8, y + 13, f"{known}/{total}", cls="small")
    svg.footer(PLOT_DATE)
    svg.save(out)


# ---------------------------------------------------------------------------
# Summary and orchestration
# ---------------------------------------------------------------------------

def write_summary(rows: list[dict[str, str]], out_dir: Path, input_path: Path, mode: str) -> None:
    score_rows = [row for row in rows if row_score(row) >= 0]
    unscored_rows = [row for row in rows if row_score(row) < 0]
    logs = [math.log10(row_citations(row) + 1) for row in score_rows]
    scores = [row_score(row) for row in score_rows]
    lines = [
        "# Dataset Plot Summary", "",
        f"Input: `{input_path}`",
        f"Load mode: `{mode}`",
        f"Rows: {len(rows)}",
        f"Rows with canonical score: {len(score_rows)}", "",
        "## Heuristic-vs-citation diagnostics", "",
    ]
    if score_rows:
        lines.extend([
            f"- Pearson r(log10(citations+1), canonical_score): `{pearson(logs, scores):.3f}`",
            f"- Spearman rho(log10(citations+1), canonical_score): `{spearman(logs, scores):.3f}`",
        ])
    else:
        lines.append("- No scored rows available.")
    if unscored_rows:
        omitted = ", ".join(shorten(row_name(row), 40) for row in unscored_rows[:12])
        suffix = "" if len(unscored_rows) <= 12 else f", ... +{len(unscored_rows) - 12} more"
        lines.extend([
            "",
            f"- Score/citation plots omit `{len(unscored_rows)}` row(s) without `canonical_score`: {omitted}{suffix}",
        ])
    lines.extend([
        "",
        "Interpretation hint: a strong positive correlation means the heuristic largely follows citation count; a weak correlation means the heuristic is using other signals. The residual plot highlights entries that score unusually high or low for their citation count.",
        "", "## Generated Figures", "",
    ])
    for path in sorted(out_dir.glob("*.svg")):
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_plots(rows: list[dict[str, str]], out_dir: Path, input_path: Path, mode: str, *, label_scatter_top: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"{len(rows)} entities from {input_path.name}"

    paths = {
        "overview": out_dir / "00_dataset_overview.svg",
        "domain_counts": out_dir / "01_dataset_domain_counts.svg",
        "artifact_counts": out_dir / "02_dataset_artifact_counts.svg",
        "timeline": out_dir / "03_dataset_timeline_by_domain.svg",
        "years": out_dir / "04_dataset_publications_by_year.svg",
        "domain_artifact": out_dir / "05_dataset_domain_artifact_heatmap.svg",
        "domain_dataform": out_dir / "06_dataset_domain_dataform_heatmap.svg",
        "temporal_environment": out_dir / "07_dataset_temporal_environment_heatmap.svg",
        "confidence_artifact": out_dir / "08_dataset_confidence_artifact_heatmap.svg",
        "score_citations": out_dir / "09_heuristic_score_vs_citations.svg",
        "score_bins": out_dir / "10_heuristic_score_citation_bins.svg",
        "score_residuals": out_dir / "11_heuristic_score_citation_residuals.svg",
        "method_scatter": out_dir / "12_discovery_method_score_citations.svg",
        "method_distribution": out_dir / "13_discovery_method_score_distribution.svg",
        "missingness": out_dir / "14_dataset_metadata_completeness.svg",
    }

    draw_overview(rows, paths["overview"], subtitle=subtitle)
    draw_horizontal_bar_chart(rows, "domain", "Datasets By Domain", paths["domain_counts"], subtitle=subtitle)
    draw_horizontal_bar_chart(rows, "artifact_type", "Datasets By Artifact Type", paths["artifact_counts"], subtitle=subtitle)
    draw_timeline(rows, paths["timeline"], subtitle=subtitle, label_top=max(24, label_scatter_top // 2))
    draw_year_histogram(rows, paths["years"], subtitle=subtitle)
    draw_heatmap(rows, "domain", "artifact_type", "Domain By Artifact Type", paths["domain_artifact"], subtitle=subtitle)
    draw_heatmap(rows, "domain", "data_form", "Domain By Data Form", paths["domain_dataform"], subtitle=subtitle)
    draw_heatmap(rows, "temporal_type", "environment", "Temporal Type By Environment", paths["temporal_environment"], subtitle=subtitle)
    draw_heatmap(rows, "confidence", "artifact_type", "Confidence By Artifact Type", paths["confidence_artifact"], subtitle=subtitle)
    draw_score_citation_scatter(rows, paths["score_citations"], subtitle=subtitle, label_top=label_scatter_top)
    draw_score_citation_bins(rows, paths["score_bins"], subtitle=subtitle)
    draw_score_citation_residuals(rows, paths["score_residuals"], subtitle=subtitle, label_top=max(25, label_scatter_top - 10))
    draw_discovery_method_scatter(rows, paths["method_scatter"], subtitle=subtitle, label_top=label_scatter_top)
    draw_method_score_distribution(rows, paths["method_distribution"], subtitle=subtitle)
    draw_missingness(rows, paths["missingness"], subtitle=subtitle)
    write_summary(rows, out_dir, input_path, mode)

    return [path for path in paths.values() if path.exists()] + [out_dir / "README.md"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create readable SVG plots from the OpenAlex IDS dataset CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Accepted OpenAlex dataset CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for SVG figures.")
    parser.add_argument("--label-scatter-top", type=int, default=45, help="Deprecated compatibility option; scatter/residual plots now label every point.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_arg = args.input if args.input.is_absolute() else args.input.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else args.out_dir.resolve()
    rows, input_path, mode = load_rows(input_arg)
    generated = create_plots(rows, out_dir, input_path, mode, label_scatter_top=max(0, args.label_scatter_top))
    for path in generated:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
