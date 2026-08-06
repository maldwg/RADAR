#!/usr/bin/env python3
"""Create SVG figures from the OpenAlex IDS dataset CSV.

The script is intentionally dependency-light: it uses only the Python standard
library and writes SVG directly. This keeps the figure generation usable even in
minimal Python environments without pandas/matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data-gathering" / "ids_datasets_openalex.csv"
DEFAULT_OUT_DIR = ROOT / "figure_creation" / "dataset_plots"

UNKNOWN = "Unknown"
PLOT_DATE = "Data acquired from input CSV"

PALETTE = [
    "#2F6B9A",
    "#C94F4F",
    "#4E9A62",
    "#D9902F",
    "#7A5FA8",
    "#3B8C8C",
    "#B85C9E",
    "#6C7A89",
    "#8C6D31",
    "#4F78C4",
]

ARTIFACT_COLORS = {
    "Dataset": "#2F6B9A",
    "Generator/framework": "#D9902F",
    "Dataset + generator/framework": "#7A5FA8",
    UNKNOWN: "#6C7A89",
}

CONFIDENCE_COLORS = {
    "High": "#2F6B9A",
    "Medium": "#D9902F",
    "Low": "#C94F4F",
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
    "Candidate audit": "#6C7A89",
    UNKNOWN: "#6C7A89",
}


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
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "..."


def row_description(row: dict[str, str], limit: int = 280) -> str:
    return shorten(
        row.get("description", "")
        or row.get("canonical_title", "")
        or row.get("mention_contexts", ""),
        limit,
    )


def row_year(row: dict[str, str]) -> int:
    return as_int(row.get("release_year") or row.get("year"), -1)


def readable_discovery_method(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return UNKNOWN
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
    return labels.get(text.casefold(), text)


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
    }
    return sorted(methods, key=lambda method: (priority.get(method, 99), method))[0] if methods else UNKNOWN


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
            ".tick{font-size:10px;fill:#53616f}",
            "</style>",
        ]

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", opacity: float = 1.0) -> None:
        self.items.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" '
            f'fill="{fill}" stroke="{stroke}" opacity="{opacity:.3f}"/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str = "axis", color: str | None = None) -> None:
        color_attr = f' stroke="{color}"' if color else ""
        self.items.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{cls}"{color_attr}/>')

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
        points = f"{x:.2f},{(y-half):.2f} {(x-half):.2f},{(y+half):.2f} {(x+half):.2f},{(y+half):.2f}"
        title_node = f"<title>{esc(title)}</title>" if title else ""
        self.items.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="1" opacity="{opacity:.3f}">{title_node}</polygon>'
        )

    def text(
        self,
        x: float,
        y: float,
        text: object,
        cls: str = "label",
        anchor: str = "start",
        rotate: float | None = None,
    ) -> None:
        transform = f' transform="rotate({rotate:.1f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.items.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}"{transform}>{esc(text)}</text>')

    def title(self, title: str, subtitle: str | None = None) -> None:
        self.text(28, 34, title, cls="title")
        if subtitle:
            self.text(28, 54, subtitle, cls="subtitle")

    def footer(self, source: str) -> None:
        self.text(28, self.height - 18, source, cls="small")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.items, "</svg>\n"]), encoding="utf-8")


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


def count_values(rows: Iterable[dict[str, str]], column: str) -> Counter[str]:
    return Counter(clean_category(row.get(column)) for row in rows)


def top_categories(counter: Counter[str], limit: int | None = None) -> list[tuple[str, int]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return items if limit is None else items[:limit]


def draw_horizontal_bar_chart(
    rows: list[dict[str, str]],
    column: str,
    title: str,
    out: Path,
    *,
    subtitle: str,
    limit: int | None = None,
) -> None:
    items = top_categories(count_values(rows, column), limit)
    width = 980
    height = max(260, 92 + len(items) * 34)
    svg = Svg(width, height)
    svg.title(title, subtitle)

    left = 210
    right = 930
    top = 82
    bar_h = 20
    gap = 14
    max_value = max((value for _, value in items), default=1)

    for index, (label, value) in enumerate(items):
        y = top + index * (bar_h + gap)
        color = PALETTE[index % len(PALETTE)]
        svg.text(left - 12, y + 14, label, cls="label", anchor="end")
        svg.rect(left, y, (right - left) * value / max_value, bar_h, color, opacity=0.9)
        svg.text(left + (right - left) * value / max_value + 8, y + 14, value, cls="label")

    svg.line(left, top - 8, left, top + len(items) * (bar_h + gap) - gap + 8)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_heatmap(
    rows: list[dict[str, str]],
    row_col: str,
    col_col: str,
    title: str,
    out: Path,
    *,
    subtitle: str,
) -> None:
    row_labels = [item for item, _ in top_categories(count_values(rows, row_col))]
    col_labels = [item for item, _ in top_categories(count_values(rows, col_col))]
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        counts[(clean_category(row.get(row_col)), clean_category(row.get(col_col)))] += 1

    cell = 56
    left = 230
    top = 124
    width = max(940, left + cell * len(col_labels) + 90)
    height = max(380, top + cell * len(row_labels) + 92)
    svg = Svg(width, height)
    svg.title(title, subtitle)
    max_value = max(counts.values(), default=1)

    for j, label in enumerate(col_labels):
        x = left + j * cell + cell / 2
        svg.text(x - 10, top - 16, shorten(label, 18), cls="tick", anchor="end", rotate=-38)

    for i, label in enumerate(row_labels):
        y = top + i * cell
        svg.text(left - 12, y + cell / 2 + 4, shorten(label, 26), cls="label", anchor="end")
        for j, col in enumerate(col_labels):
            value = counts.get((label, col), 0)
            shade = 0.12 + 0.82 * (value / max_value if max_value else 0)
            color = f"rgb({int(47 + (255 - 47) * (1 - shade))},{int(107 + (255 - 107) * (1 - shade))},{int(154 + (255 - 154) * (1 - shade))})"
            x = left + j * cell
            svg.rect(x, y, cell - 2, cell - 2, color, stroke="#ffffff")
            if value:
                svg.text(x + cell / 2 - 1, y + cell / 2 + 4, value, cls="label", anchor="middle")

    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_year_histogram(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    year_rows = [row for row in rows if row_year(row) > 0]
    years = [row_year(row) for row in year_rows]
    if not years:
        return
    min_year = min(years)
    max_year = max(years)
    artifacts = [item for item, _ in top_categories(count_values(year_rows, "artifact_type"))]
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in year_rows:
        counts[(row_year(row), clean_category(row.get("artifact_type")))] += 1

    width = max(1050, 70 + (max_year - min_year + 1) * 26)
    height = 460
    left, right, top, bottom = 72, width - 36, 76, height - 74
    svg = Svg(width, height)
    svg.title("Dataset Release Years Over Time", subtitle)
    max_stack = max(sum(counts.get((year, artifact), 0) for artifact in artifacts) for year in range(min_year, max_year + 1))
    plot_h = bottom - top
    step = (right - left) / max(1, max_year - min_year + 1)
    bar_w = min(18, step * 0.72)

    for tick in range(0, max_stack + 1):
        if max_stack > 8 and tick % max(1, math.ceil(max_stack / 6)) != 0:
            continue
        y = bottom - plot_h * tick / max_stack
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, tick, cls="tick", anchor="end")

    for year in range(min_year, max_year + 1):
        x = left + (year - min_year + 0.5) * step
        y_cursor = bottom
        for artifact in artifacts:
            value = counts.get((year, artifact), 0)
            if not value:
                continue
            h = plot_h * value / max_stack
            y_cursor -= h
            svg.rect(x - bar_w / 2, y_cursor, bar_w, h, ARTIFACT_COLORS.get(artifact, "#6C7A89"), stroke="#ffffff")
        if year % max(1, math.ceil((max_year - min_year + 1) / 14)) == 0 or year in {min_year, max_year}:
            svg.text(x, bottom + 18, year, cls="tick", anchor="middle", rotate=-35)

    legend_x = left
    for artifact in artifacts:
        color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
        svg.rect(legend_x, height - 42, 14, 14, color)
        svg.text(legend_x + 20, height - 30, artifact, cls="small")
        legend_x += 205

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_timeline(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    year_rows = [row for row in rows if row_year(row) > 0]
    if not year_rows:
        return
    domains = [item for item, _ in top_categories(count_values(year_rows, "domain"))]
    years = [row_year(row) for row in year_rows]
    min_year, max_year = min(years), max(years)
    width = max(1380, 260 + (max_year - min_year + 1) * 34)
    height = max(520, 132 + len(domains) * 52)
    left, right, top, bottom = 190, width - 92, 92, height - 104
    svg = Svg(width, height)
    svg.title("Dataset Release Timeline By Domain", subtitle)

    def x_for_year(year: int) -> float:
        return left + (year - min_year) / max(1, max_year - min_year) * (right - left)

    domain_y = {domain: top + i * ((bottom - top) / max(1, len(domains) - 1)) for i, domain in enumerate(domains)}
    for i, domain in enumerate(domains):
        y = domain_y[domain]
        if i % 2 == 0:
            svg.rect(left - 8, y - 16, right - left + 16, 32, "#f3f6fa")
        svg.text(left - 14, y + 4, shorten(domain, 22), cls="label", anchor="end")
        svg.line(left, y, right, y, cls="grid")

    for year in range(min_year, max_year + 1):
        if year % max(1, math.ceil((max_year - min_year) / 12)) == 0 or year in {min_year, max_year}:
            x = x_for_year(year)
            svg.line(x, top - 10, x, bottom + 8, cls="grid")
            svg.text(x, bottom + 24, year, cls="tick", anchor="middle", rotate=-35)

    label_rows = sorted(
        year_rows,
        key=lambda row: (as_int(row.get("survey_count")), as_int(row.get("citations")), as_float(row.get("canonical_score"))),
        reverse=True,
    )[:16]
    label_ids = {row.get("dataset_id") for row in label_rows}
    collisions: dict[tuple[int, str], int] = defaultdict(int)
    label_boxes: list[tuple[float, float, float, float]] = []

    def label_fits(x: float, y: float, text: str) -> bool:
        w = min(150, max(45, len(text) * 5.5))
        box = (x, y - 11, x + w, y + 2)
        for old in label_boxes:
            if not (box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]):
                return False
        label_boxes.append(box)
        return True

    for row in sorted(year_rows, key=lambda item: (row_year(item), clean_category(item.get("domain")), item.get("name", ""))):
        year = row_year(row)
        domain = clean_category(row.get("domain"))
        artifact = clean_category(row.get("artifact_type"))
        key = (year, domain)
        offset = (collisions[key] % 5 - 2) * 4
        collisions[key] += 1
        x = x_for_year(year)
        y = domain_y.get(domain, top) + offset
        size = 7 + min(8, math.log(as_int(row.get("citations")) + 1, 10) * 3)
        color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
        title = " | ".join(
            item
            for item in (
                row.get("name", ""),
                artifact,
                domain,
                str(year),
                row_description(row),
            )
            if item
        )
        if artifact == "Generator/framework":
            svg.triangle(x, y, size * 1.7, color, title=title)
        elif artifact == "Dataset + generator/framework":
            svg.square(x, y, size * 1.5, color, title=title)
        else:
            svg.circle(x, y, size, color, title=title)
        label = shorten(row.get("name", ""), 24)
        if row.get("dataset_id") in label_ids and label_fits(x + 10, y - 8, label):
            svg.text(x + 10, y - 8, label, cls="small")

    legend_y = height - 64
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
        legend_x += 220

    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_citation_score_scatter(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    plot_rows = [row for row in rows if as_float(row.get("canonical_score"), -1) >= 0]
    if not plot_rows:
        return
    width, height = 1120, 680
    left, right, top, bottom = 92, width - 72, 86, height - 92
    svg = Svg(width, height)
    svg.title("Canonical Score Vs Citation Count", subtitle)
    max_x = max(math.log10(as_int(row.get("citations")) + 1) for row in plot_rows)
    min_y = min(as_float(row.get("canonical_score")) for row in plot_rows)
    max_y = max(as_float(row.get("canonical_score")) for row in plot_rows)

    def x_for(row: dict[str, str]) -> float:
        return left + math.log10(as_int(row.get("citations")) + 1) / max(1e-9, max_x) * (right - left)

    def y_for(row: dict[str, str]) -> float:
        return bottom - (as_float(row.get("canonical_score")) - min_y) / max(1e-9, max_y - min_y) * (bottom - top)

    for i in range(6):
        x = left + i * (right - left) / 5
        label = f"{10 ** (i * max_x / 5) - 1:.0f}"
        svg.line(x, top, x, bottom, cls="grid")
        svg.text(x, bottom + 18, label, cls="tick", anchor="middle")
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = min_y + i * (max_y - min_y) / 5
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{value:.1f}", cls="tick", anchor="end")

    label_rows = sorted(plot_rows, key=lambda row: (as_int(row.get("citations")), as_float(row.get("canonical_score"))), reverse=True)[:14]
    label_ids = {row.get("dataset_id") for row in label_rows}
    label_boxes: list[tuple[float, float, float, float]] = []

    def can_place_label(x: float, y: float, text: str) -> bool:
        w = min(145, max(45, len(text) * 5.8))
        box = (x, y - 12, x + w, y + 2)
        for old in label_boxes:
            if not (box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]):
                return False
        label_boxes.append(box)
        return True

    for row in plot_rows:
        x, y = x_for(row), y_for(row)
        artifact = clean_category(row.get("artifact_type"))
        color = ARTIFACT_COLORS.get(artifact, "#6C7A89")
        radius = 5 + min(8, math.sqrt(max(1, as_int(row.get("survey_count")))))
        title = " | ".join(
            item
            for item in (
                row.get("name", ""),
                artifact,
                f"citations={as_int(row.get('citations'))}",
                f"score={as_float(row.get('canonical_score')):.2f}",
                row_description(row),
            )
            if item
        )
        svg.circle(x, y, radius, color, opacity=0.78, title=title)
        label = shorten(row.get("name", ""), 24)
        if row.get("dataset_id") in label_ids and can_place_label(x + 8, y - 8, label):
            svg.text(x + 8, y - 8, label, cls="small")

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text((left + right) / 2, height - 28, "Citations, log scale", cls="label", anchor="middle")
    svg.text(20, (top + bottom) / 2, "Canonical score", cls="label", anchor="middle", rotate=-90)
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_discovery_method_scatter(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    plot_rows = [row for row in rows if as_float(row.get("canonical_score"), -1) >= 0]
    if not plot_rows:
        return
    methods = [method for method, _ in top_categories(Counter(row_discovery_method(row) for row in plot_rows))]
    width, height = 1250, 720
    left, right, top, bottom = 96, width - 250, 90, height - 96
    svg = Svg(width, height)
    svg.title("Discovery Method Vs Score And Citations", subtitle)
    max_x = max(math.log10(as_int(row.get("citations")) + 1) for row in plot_rows)
    min_y = min(as_float(row.get("canonical_score")) for row in plot_rows)
    max_y = max(as_float(row.get("canonical_score")) for row in plot_rows)

    def x_for(row: dict[str, str]) -> float:
        return left + math.log10(as_int(row.get("citations")) + 1) / max(1e-9, max_x) * (right - left)

    def y_for(row: dict[str, str]) -> float:
        return bottom - (as_float(row.get("canonical_score")) - min_y) / max(1e-9, max_y - min_y) * (bottom - top)

    for i in range(6):
        x = left + i * (right - left) / 5
        label = f"{10 ** (i * max_x / 5) - 1:.0f}"
        svg.line(x, top, x, bottom, cls="grid")
        svg.text(x, bottom + 18, label, cls="tick", anchor="middle")
    for i in range(6):
        y = bottom - i * (bottom - top) / 5
        value = min_y + i * (max_y - min_y) / 5
        svg.line(left, y, right, y, cls="grid")
        svg.text(left - 10, y + 4, f"{value:.1f}", cls="tick", anchor="end")

    label_rows = sorted(
        plot_rows,
        key=lambda row: (
            as_float(row.get("canonical_score")),
            as_int(row.get("citations")),
            as_int(row.get("survey_count")),
        ),
        reverse=True,
    )[:18]
    label_ids = {row.get("dataset_id") for row in label_rows}
    label_boxes: list[tuple[float, float, float, float]] = []

    def can_place_label(x: float, y: float, text: str) -> bool:
        w = min(150, max(45, len(text) * 5.7))
        box = (x, y - 12, x + w, y + 2)
        if box[2] > right + 160 or box[0] < left:
            return False
        for old in label_boxes:
            if not (box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]):
                return False
        label_boxes.append(box)
        return True

    collisions: dict[tuple[int, int], int] = defaultdict(int)
    for row in sorted(plot_rows, key=lambda item: (row_discovery_method(item), item.get("name", ""))):
        x, y = x_for(row), y_for(row)
        bucket = (round(x / 18), round(y / 18))
        offset = collisions[bucket]
        collisions[bucket] += 1
        if offset:
            angle = (offset % 8) * math.pi / 4
            radius = 2.5 * min(4, math.ceil(offset / 8))
            x += math.cos(angle) * radius
            y += math.sin(angle) * radius

        method = row_discovery_method(row)
        color = DISCOVERY_METHOD_COLORS.get(method, "#6C7A89")
        support = max(1, as_int(row.get("survey_count")) + as_int(row.get("candidate_count")))
        radius = 5 + min(8, math.log(support + 1, 2) * 2)
        all_methods = "; ".join(row_discovery_methods(row)) or method
        title = " | ".join(
            item
            for item in (
                row.get("name", ""),
                f"method={method}",
                f"all methods={all_methods}",
                f"citations={as_int(row.get('citations'))}",
                f"score={as_float(row.get('canonical_score')):.2f}",
                f"survey_count={as_int(row.get('survey_count'))}",
                f"candidate_count={as_int(row.get('candidate_count'))}",
                row_description(row),
            )
            if item
        )
        svg.circle(x, y, radius, color, opacity=0.76, title=title)
        label = shorten(row.get("name", ""), 24)
        if row.get("dataset_id") in label_ids and can_place_label(x + 8, y - 8, label):
            svg.text(x + 8, y - 8, label, cls="small")

    svg.line(left, top, left, bottom)
    svg.line(left, bottom, right, bottom)
    svg.text((left + right) / 2, height - 30, "Citations, log scale", cls="label", anchor="middle")
    svg.text(22, (top + bottom) / 2, "Heuristic/canonical score", cls="label", anchor="middle", rotate=-90)

    legend_x = right + 26
    legend_y = top + 4
    svg.text(legend_x, legend_y, "Discovery method", cls="label")
    for index, method in enumerate(methods[:14]):
        y = legend_y + 24 + index * 24
        svg.circle(legend_x + 7, y - 5, 6, DISCOVERY_METHOD_COLORS.get(method, "#6C7A89"))
        svg.text(legend_x + 20, y - 1, shorten(method, 24), cls="small")
    if len(methods) > 14:
        svg.text(legend_x, legend_y + 24 + 14 * 24, f"+ {len(methods) - 14} more", cls="small")

    svg.text(legend_x, height - 90, "Point size", cls="label")
    svg.text(legend_x, height - 70, "survey_count + candidate_count", cls="small")
    svg.footer(PLOT_DATE)
    svg.save(out)


def draw_missingness(rows: list[dict[str, str]], out: Path, *, subtitle: str) -> None:
    columns = ["release_year", "canonical_publication_year", "domain", "data_form", "temporal_type", "environment", "labeling", "confidence", "artifact_type", "canonical_score"]
    width, height = 930, 360
    left, right, top = 170, 860, 84
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


def write_summary(rows: list[dict[str, str]], out_dir: Path, input_path: Path, mode: str) -> None:
    lines = [
        "# Dataset Plot Summary",
        "",
        f"Input: `{input_path}`",
        f"Load mode: `{mode}`",
        f"Rows: {len(rows)}",
        "",
        "## Columns Used",
        "",
        "`name`, `description`, `artifact_type`, `release_year`, `canonical_publication_year`, `citations`, `survey_count`, `candidate_count`, `discovery_method`, `discovery_methods`, `domain`, `data_form`, `temporal_type`, `environment`, `labeling`, `confidence`, `canonical_score`",
        "",
        "## Generated Figures",
        "",
    ]
    for path in sorted(out_dir.glob("*.svg")):
        lines.append(f"- `{path.name}`")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_plots(rows: list[dict[str, str]], out_dir: Path, input_path: Path, mode: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    subtitle = f"{len(rows)} entities from {input_path.name}"
    paths = [
        out_dir / "dataset_domain_counts.svg",
        out_dir / "dataset_artifact_counts.svg",
        out_dir / "dataset_timeline_by_domain.svg",
        out_dir / "dataset_publications_by_year.svg",
        out_dir / "dataset_domain_artifact_heatmap.svg",
        out_dir / "dataset_domain_dataform_heatmap.svg",
        out_dir / "dataset_temporal_environment_heatmap.svg",
        out_dir / "dataset_confidence_artifact_heatmap.svg",
        out_dir / "dataset_score_citations.svg",
        out_dir / "dataset_discovery_method_score_citations.svg",
        out_dir / "dataset_metadata_completeness.svg",
    ]

    draw_horizontal_bar_chart(rows, "domain", "Datasets By Domain", paths[0], subtitle=subtitle)
    draw_horizontal_bar_chart(rows, "artifact_type", "Datasets By Artifact Type", paths[1], subtitle=subtitle)
    draw_timeline(rows, paths[2], subtitle=subtitle)
    draw_year_histogram(rows, paths[3], subtitle=subtitle)
    draw_heatmap(rows, "domain", "artifact_type", "Domain By Artifact Type", paths[4], subtitle=subtitle)
    draw_heatmap(rows, "domain", "data_form", "Domain By Data Form", paths[5], subtitle=subtitle)
    draw_heatmap(rows, "temporal_type", "environment", "Temporal Type By Environment", paths[6], subtitle=subtitle)
    draw_heatmap(rows, "confidence", "artifact_type", "Confidence By Artifact Type", paths[7], subtitle=subtitle)
    draw_citation_score_scatter(rows, paths[8], subtitle=subtitle)
    draw_discovery_method_scatter(rows, paths[9], subtitle=subtitle)
    draw_missingness(rows, paths[10], subtitle=subtitle)
    write_summary(rows, out_dir, input_path, mode)
    return [path for path in paths if path.exists()] + [out_dir / "README.md"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SVG plots from the OpenAlex IDS dataset CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Accepted OpenAlex dataset CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for SVG figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, input_path, mode = load_rows(args.input.resolve() if not args.input.is_absolute() else args.input)
    generated = create_plots(rows, args.out_dir.resolve() if not args.out_dir.is_absolute() else args.out_dir, input_path, mode)
    for path in generated:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
