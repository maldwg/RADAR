#!/usr/bin/env python3
"""Create improved SVG figures for IDS GitHub repository crawl results.

The script is intentionally dependency-light: it reads the compact
``ids_repos.csv`` emitted by ``data-gathering/github-crawler.py`` and writes
plain SVG files directly. It mirrors the improved OpenAlex dataset plotter's
style: readable labels, diagnostic views, heatmaps, scatter plots with simple
label placement, and a README with short data-quality notes.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data-gathering" / "ids_repos.csv"
DEFAULT_OUT_DIR = ROOT / "figure_creation" / "github-plots"

UNKNOWN = "Unknown"

PALETTE = [
    "#2F6B9A",
    "#4E9A62",
    "#D9902F",
    "#7A5FA8",
    "#C94F4F",
    "#3B8C8C",
    "#B85C9E",
    "#8C6D31",
    "#4F78C4",
    "#6C7A89",
]

FAMILY_COLORS = {
    "NIDS": "#2F6B9A",
    "NIPS": "#3B8C8C",
    "HIDS": "#4E9A62",
    "HIPS": "#78A653",
    "IPS": "#D9902F",
    "WIDS": "#5E7CE2",
    "WIPS": "#7A5FA8",
    "DBIDS": "#8C6D31",
    "DBIPS": "#B85C9E",
    "PIDS": "#9A7B22",
    "NDR": "#C94F4F",
    "EDR": "#B85C9E",
    "XDR": "#6C7A89",
    "IDS": "#4F78C4",
    UNKNOWN: "#6C7A89",
}

ROLE_COLORS = {
    "deployable_ids": "#2F6B9A",
    "platform_framework": "#7A5FA8",
    "classifier_model": "#D9902F",
    "rules_content": "#3B8C8C",
    "support_tooling": "#6C7A89",
    "research_demo": "#8C6D31",
    UNKNOWN: "#6C7A89",
}

ROLE_LABELS = {
    "deployable_ids": "Deployable IDS",
    "platform_framework": "Platform/framework",
    "classifier_model": "Classifier/model",
    "rules_content": "Rules/content",
    "support_tooling": "Support tooling",
    "research_demo": "Research/demo",
    UNKNOWN: UNKNOWN,
}

ROLE_ORDER = [
    "deployable_ids",
    "platform_framework",
    "classifier_model",
    "rules_content",
    "support_tooling",
    "research_demo",
    UNKNOWN,
]

ACTIVITY_ORDER = [
    "Very active",
    "Active",
    "Quiet",
    "Stale",
    "Dormant",
    "Archived",
    UNKNOWN,
]

ACTIVITY_LABELS = {
    "Very active": "<=90d",
    "Active": "<=1y",
    "Quiet": "1-3y",
    "Stale": "3-5y",
    "Dormant": ">5y",
    "Archived": "arch.",
    UNKNOWN: "?",
}

ACTIVITY_COLORS = {
    "Very active": "#3B8C8C",
    "Active": "#4E9A62",
    "Quiet": "#D9902F",
    "Stale": "#C94F4F",
    "Dormant": "#8D3B4F",
    "Archived": "#6C7A89",
    UNKNOWN: "#A0A8B0",
}


# ---------------------------------------------------------------------------
# General helpers
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


def as_bool(value: object) -> bool:
    return str(value or "").strip().casefold() == "true"


def parse_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def clean(value: object, default: str = UNKNOWN) -> str:
    text = str(value or "").strip()
    return text if text else default


def shorten(value: object, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "..."


def wrap_text(value: object, limit: int = 16, max_lines: int = 3) -> list[str]:
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


def split_values(value: object) -> list[str]:
    parts = [
        clean(part, "")
        for part in str(value or "").replace(",", ";").split(";")
        if clean(part, "")
    ]
    return list(dict.fromkeys(parts))


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, clean(role))


def repo_name(row: dict[str, object]) -> str:
    return clean(row.get("full_name"))


def repo_short_name(row: dict[str, object]) -> str:
    full_name = repo_name(row)
    return full_name.split("/", 1)[-1] if "/" in full_name else full_name


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


def log1p(value: float) -> float:
    return math.log10(max(0.0, value) + 1.0)


def format_int(value: int | float) -> str:
    value = int(round(value))
    return f"{value:,}"


def format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def color_for_index(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rows(path: Path, as_of: datetime | None = None) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))

    dates = [
        dt
        for row in rows
        for dt in (parse_dt(row.get("pushed_at")), parse_dt(row.get("updated_at")))
        if dt is not None
    ]
    reference_date = as_of or (max(dates) if dates else datetime.now(timezone.utc))
    if reference_date.tzinfo is None:
        reference_date = reference_date.replace(tzinfo=timezone.utc)

    for row in rows:
        row["stars_i"] = as_int(row.get("stars"))
        row["forks_i"] = as_int(row.get("forks"))
        row["open_issues_i"] = as_int(row.get("open_issues"))
        row["rank_i"] = as_int(row.get("rank"))
        row["archived_b"] = as_bool(row.get("archived"))
        row["pushed_dt"] = parse_dt(row.get("pushed_at")) or parse_dt(row.get("updated_at"))
        row["updated_dt"] = parse_dt(row.get("updated_at"))
        pushed_dt = row["pushed_dt"]
        if isinstance(pushed_dt, datetime):
            row["age_days"] = max(0, (reference_date - pushed_dt).days)
        else:
            row["age_days"] = None
        row["activity_bucket"] = activity_bucket(row)
        row["primary_family_clean"] = clean(row.get("primary_family"))
        row["role_clean"] = clean(row.get("role"))
        row["role_label_clean"] = clean(row.get("role_label"), role_label(str(row.get("role_clean"))))
        row["language_clean"] = clean(row.get("language"))
        row["type_values"] = split_values(row.get("type"))
        row["topic_values"] = split_values(row.get("topics"))
    return rows


def activity_bucket(row: dict[str, object]) -> str:
    if as_bool(row.get("archived")):
        return "Archived"
    age = row.get("age_days")
    if age is None:
        return UNKNOWN
    age_i = int(age)
    if age_i <= 90:
        return "Very active"
    if age_i <= 365:
        return "Active"
    if age_i <= 365 * 3:
        return "Quiet"
    if age_i <= 365 * 5:
        return "Stale"
    return "Dormant"


def count_by(rows: Iterable[dict[str, object]], column: str) -> Counter[str]:
    return Counter(clean(row.get(column)) for row in rows)


def count_multi(rows: Iterable[dict[str, object]], column: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        for value in row.get(column, []) or []:
            counter[clean(value)] += 1
    return counter


def sorted_counts(counter: Counter[str], limit: int | None = None) -> list[tuple[str, int]]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return items if limit is None else items[:limit]


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------


def svg_document(width: int, height: int, content: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{esc(title)}</title>\n"
        "<style>\n"
        "text { font-family: Arial, Helvetica, sans-serif; fill: #263238; }\n"
        ".title { font-size: 23px; font-weight: 700; }\n"
        ".subtitle { font-size: 12px; fill: #64707d; }\n"
        ".axis { font-size: 11px; fill: #59636f; }\n"
        ".label { font-size: 12px; }\n"
        ".small { font-size: 10px; fill: #59636f; }\n"
        ".tiny { font-size: 8px; fill: #43505c; }\n"
        ".value { font-size: 11px; font-weight: 700; fill: #263238; }\n"
        ".tile-value { font-size: 24px; font-weight: 700; }\n"
        ".tile-label { font-size: 11px; fill: #59636f; }\n"
        "</style>\n"
        f"{content}\n"
        "</svg>\n"
    )


def write_svg(out_dir: Path, name: str, width: int, height: int, content: str, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(svg_document(width, height, content, title), encoding="utf-8")
    return path


def add_title(parts: list[str], title: str, subtitle: str, width: int) -> None:
    parts.append(f'<text class="title" x="{width / 2}" y="34" text-anchor="middle">{esc(title)}</text>')
    if subtitle:
        parts.append(f'<text class="subtitle" x="{width / 2}" y="55" text-anchor="middle">{esc(subtitle)}</text>')


def legend(parts: list[str], items: Sequence[tuple[str, str]], x: int, y: int, columns: int = 4) -> None:
    col_w = 150
    row_h = 18
    for index, (label, color) in enumerate(items):
        col = index % columns
        row = index // columns
        lx = x + col * col_w
        ly = y + row * row_h
        parts.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" rx="2" fill="{color}"/>')
        parts.append(f'<text class="small" x="{lx + 15}" y="{ly}">{esc(shorten(label, 19))}</text>')


def axis_ticks(parts: list[str], x: int, y: int, width: int, max_value: float, ticks: int = 5) -> None:
    if max_value <= 0:
        return
    for i in range(ticks + 1):
        value = max_value * i / ticks
        tx = x + width * i / ticks
        parts.append(f'<line x1="{tx:.1f}" y1="{y}" x2="{tx:.1f}" y2="{y + 5}" stroke="#AEB7C2"/>')
        parts.append(
            f'<text class="tiny" x="{tx:.1f}" y="{y + 18}" text-anchor="middle">{esc(format_int(value))}</text>'
        )


def horizontal_bar_chart(
    rows: Sequence[tuple[str, int]],
    out_dir: Path,
    filename: str,
    title: str,
    subtitle: str,
    color: str | dict[str, str] = "#2F6B9A",
    width: int = 1040,
    label_width: int = 280,
    row_height: int = 32,
) -> Path:
    rows = [(label, value) for label, value in rows if value > 0]
    height = 104 + max(1, len(rows)) * row_height + 34
    chart_w = width - label_width - 92
    max_value = max([value for _, value in rows] or [1])
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(parts, title, subtitle, width)
    top = 82
    for index, (label, value) in enumerate(rows):
        y = top + index * row_height
        bar_w = chart_w * value / max_value
        bar_color = color.get(label, color_for_index(index)) if isinstance(color, dict) else color
        parts.append(f'<text class="label" x="{label_width - 14}" y="{y + 15}" text-anchor="end">{esc(shorten(label, 34))}<title>{esc(label)}</title></text>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{chart_w}" height="19" rx="4" fill="#EEF2F5"/>')
        parts.append(f'<rect x="{label_width}" y="{y}" width="{bar_w:.1f}" height="19" rx="4" fill="{bar_color}"/>')
        parts.append(f'<text class="value" x="{label_width + bar_w + 8:.1f}" y="{y + 15}">{value}</text>')
    axis_ticks(parts, label_width, top + len(rows) * row_height + 2, chart_w, max_value)
    return write_svg(out_dir, filename, width, height, "\n".join(parts), title)


def heatmap(
    matrix: dict[tuple[str, str], int],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_dir: Path,
    filename: str,
    title: str,
    subtitle: str,
    row_label_width: int = 190,
    cell_w: int = 92,
    cell_h: int = 48,
) -> Path:
    row_labels = list(row_labels)
    col_labels = list(col_labels)
    top = 120
    left = row_label_width
    width = left + max(1, len(col_labels)) * cell_w + 48
    height = top + max(1, len(row_labels)) * cell_h + 58
    max_value = max(matrix.values() or [1])
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(parts, title, subtitle, width)

    for ci, col in enumerate(col_labels):
        x = left + ci * cell_w + cell_w / 2
        for li, line in enumerate(wrap_text(col, 12, 3)):
            parts.append(f'<text class="tiny" x="{x:.1f}" y="{84 + li * 10}" text-anchor="middle">{esc(line)}</text>')

    for ri, row in enumerate(row_labels):
        y = top + ri * cell_h
        parts.append(f'<text class="label" x="{left - 12}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end">{esc(shorten(row, 24))}</text>')
        for ci, col in enumerate(col_labels):
            value = matrix.get((row, col), 0)
            intensity = value / max_value if max_value else 0.0
            fill = blend("#F2F5F8", "#2F6B9A", intensity)
            x = left + ci * cell_w
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="5" fill="{fill}"/>')
            if value:
                parts.append(f'<text class="value" x="{x + cell_w / 2 - 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="middle">{value}</text>')
    return write_svg(out_dir, filename, width, height, "\n".join(parts), title)


def blend(low: str, high: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    low_rgb = tuple(int(low[i : i + 2], 16) for i in (1, 3, 5))
    high_rgb = tuple(int(high[i : i + 2], 16) for i in (1, 3, 5))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(low_rgb, high_rgb))
    return "#" + "".join(f"{value:02x}" for value in rgb)


def scatter_plot(
    points: Sequence[dict[str, object]],
    out_dir: Path,
    filename: str,
    title: str,
    subtitle: str,
    x_get,
    y_get,
    x_label: str,
    y_label: str,
    color_get,
    label_get=None,
    width: int = 1100,
    height: int = 700,
) -> Path:
    left, right, top, bottom = 86, 52, 82, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(x_get(row)) for row in points]
    ys = [float(y_get(row)) for row in points]
    x_max = max(xs or [1.0])
    y_max = max(ys or [1.0])
    x_max = x_max * 1.05 if x_max else 1.0
    y_max = y_max * 1.05 if y_max else 1.0

    def sx(value: float) -> float:
        return left + plot_w * value / x_max

    def sy(value: float) -> float:
        return top + plot_h - plot_h * value / y_max

    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(parts, title, subtitle, width)
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#FBFCFD" stroke="#D8DEE6"/>')

    for i in range(6):
        x = left + plot_w * i / 5
        y = top + plot_h * i / 5
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#E3E8EE"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#E3E8EE"/>')
        parts.append(f'<text class="tiny" x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle">{esc(format_int(10 ** (x_max * i / 5) - 1 if "log" in x_label.lower() else x_max * i / 5))}</text>')
        parts.append(f'<text class="tiny" x="{left - 8}" y="{top + plot_h - plot_h * i / 5 + 3:.1f}" text-anchor="end">{esc(format_int(10 ** (y_max * i / 5) - 1 if "log" in y_label.lower() else y_max * i / 5))}</text>')

    parts.append(f'<text class="axis" x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">{esc(x_label)}</text>')
    parts.append(f'<text class="axis" x="22" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2})">{esc(y_label)}</text>')

    labeled: list[tuple[float, float, str]] = []
    for row in points:
        x = sx(float(x_get(row)))
        y = sy(float(y_get(row)))
        role = str(row.get("role_clean") or UNKNOWN)
        radius = 4.0 + min(6.0, log1p(as_int(row.get("stars_i"))) * 0.75)
        color = color_get(row)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.78" stroke="white" stroke-width="1.1"><title>{esc(repo_name(row))}: {format_int(as_int(row.get("stars_i")))} stars, {format_int(as_int(row.get("forks_i")))} forks, {role_label(role)}</title></circle>')
        if label_get:
            label = label_get(row)
            if label:
                labeled.append((x, y, label))

    used: list[tuple[float, float]] = []
    for x, y, label in labeled:
        ly = y - 9
        while any(abs(ly - uy) < 12 and abs(x - ux) < 90 for ux, uy in used):
            ly -= 12
        used.append((x, ly))
        parts.append(f'<text class="tiny" x="{x + 8:.1f}" y="{ly:.1f}">{esc(shorten(label, 22))}</text>')

    legend_items = [(role_label(role), ROLE_COLORS.get(role, ROLE_COLORS[UNKNOWN])) for role in ROLE_ORDER if any(str(row.get("role_clean")) == role for row in points)]
    legend(parts, legend_items, left, height - 50, columns=4)
    return write_svg(out_dir, filename, width, height, "\n".join(parts), title)


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------


def plot_overview(rows: list[dict[str, object]], out_dir: Path) -> Path:
    width, height = 1120, 640
    total = len(rows)
    stars = [as_int(row.get("stars_i")) for row in rows]
    forks = [as_int(row.get("forks_i")) for row in rows]
    active = sum(1 for row in rows if row.get("activity_bucket") in {"Very active", "Active"})
    deployable = sum(1 for row in rows if row.get("role_clean") == "deployable_ids")
    archived = sum(1 for row in rows if row.get("archived_b"))
    top_repo = max(rows, key=lambda row: as_int(row.get("stars_i")), default={})
    tiles = [
        ("Repositories", format_int(total)),
        ("Total stars", format_int(sum(stars))),
        ("Median stars", format_int(statistics.median(stars) if stars else 0)),
        ("Total forks", format_int(sum(forks))),
        ("Active <= 1y", format_pct(active / total) if total else "0%"),
        ("Deployable IDS", format_pct(deployable / total) if total else "0%"),
        ("Archived", format_int(archived)),
        ("Top repo", shorten(repo_name(top_repo), 18)),
    ]

    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(parts, "GitHub IDS Repository Overview", "Compact crawl output: size, roles, families, and maintenance signal", width)
    tile_w, tile_h = 245, 86
    start_x, start_y = 58, 88
    for index, (label, value) in enumerate(tiles):
        col = index % 4
        row = index // 4
        x = start_x + col * (tile_w + 24)
        y = start_y + row * (tile_h + 18)
        parts.append(f'<rect x="{x}" y="{y}" width="{tile_w}" height="{tile_h}" rx="7" fill="#F6F8FA" stroke="#D8DEE6"/>')
        parts.append(f'<text class="tile-value" x="{x + 18}" y="{y + 38}">{esc(value)}</text>')
        parts.append(f'<text class="tile-label" x="{x + 18}" y="{y + 64}">{esc(label)}</text>')

    role_counts = sorted_counts(count_by(rows, "role_clean"))
    family_counts = sorted_counts(count_by(rows, "primary_family_clean"), 8)
    activity_counts = [(label, count_by(rows, "activity_bucket").get(label, 0)) for label in ACTIVITY_ORDER]

    def mini_bars(data: Sequence[tuple[str, int]], x: int, y: int, title: str, color_map: dict[str, str] | None = None) -> None:
        parts.append(f'<text class="label" x="{x}" y="{y - 16}" font-weight="700">{esc(title)}</text>')
        max_value = max([value for _, value in data] or [1])
        for i, (label, value) in enumerate(data):
            by = y + i * 24
            color = (color_map or {}).get(label, color_for_index(i))
            parts.append(f'<text class="small" x="{x}" y="{by + 12}">{esc(shorten(role_label(label), 22))}</text>')
            parts.append(f'<rect x="{x + 150}" y="{by}" width="210" height="15" rx="3" fill="#EEF2F5"/>')
            parts.append(f'<rect x="{x + 150}" y="{by}" width="{210 * value / max_value:.1f}" height="15" rx="3" fill="{color}"/>')
            parts.append(f'<text class="value" x="{x + 368}" y="{by + 12}">{value}</text>')

    mini_bars([(role_label(k), v) for k, v in role_counts], 74, 315, "Role mix", {role_label(k): ROLE_COLORS.get(k, ROLE_COLORS[UNKNOWN]) for k, _ in role_counts})
    mini_bars(family_counts, 468, 315, "Primary family", FAMILY_COLORS)
    mini_bars(activity_counts, 820, 315, "Maintenance", ACTIVITY_COLORS)
    return write_svg(out_dir, "00_github_repo_overview.svg", width, height, "\n".join(parts), "GitHub IDS Repository Overview")


def plot_counts(rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    paths = []
    paths.append(
        horizontal_bar_chart(
            sorted_counts(count_by(rows, "primary_family_clean")),
            out_dir,
            "01_primary_family_counts.svg",
            "Repositories By Primary Family",
            "One primary IDS/IPS family per accepted repository",
            FAMILY_COLORS,
        )
    )
    role_data = [(role_label(role), count) for role, count in sorted_counts(count_by(rows, "role_clean"))]
    role_color = {role_label(role): ROLE_COLORS.get(role, ROLE_COLORS[UNKNOWN]) for role in ROLE_ORDER}
    paths.append(
        horizontal_bar_chart(
            role_data,
            out_dir,
            "02_repository_role_counts.svg",
            "Repositories By Role",
            "Deployable tools separated from frameworks, classifiers, rules, support, and demos",
            role_color,
        )
    )
    paths.append(
        horizontal_bar_chart(
            sorted_counts(count_multi(rows, "type_values"), 14),
            out_dir,
            "03_detection_type_counts.svg",
            "Detection Type Evidence",
            "Multi-label counts from the crawler type column",
            "#3B8C8C",
        )
    )
    paths.append(
        horizontal_bar_chart(
            sorted_counts(count_by(rows, "language_clean"), 16),
            out_dir,
            "04_language_counts.svg",
            "Implementation Languages",
            "Primary GitHub language among accepted repositories",
            "#7A5FA8",
        )
    )
    return paths


def plot_heatmaps(rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    paths = []
    families = [label for label, _count in sorted_counts(count_by(rows, "primary_family_clean"), 12)]
    roles = [role for role in ROLE_ORDER if any(row.get("role_clean") == role for row in rows)]
    role_labels = [role_label(role) for role in roles]
    family_role_matrix: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        family = str(row.get("primary_family_clean") or UNKNOWN)
        role = role_label(str(row.get("role_clean") or UNKNOWN))
        family_role_matrix[(family, role)] += 1
    paths.append(
        heatmap(
            family_role_matrix,
            families,
            role_labels,
            out_dir,
            "05_family_role_heatmap.svg",
            "Family x Role Heatmap",
            "Where accepted repositories concentrate by detector family and public-facing role",
        )
    )

    activity_labels = [label for label in ACTIVITY_ORDER if any(row.get("activity_bucket") == label for row in rows)]
    family_activity_matrix: dict[tuple[str, str], int] = defaultdict(int)
    for row in rows:
        family_activity_matrix[(str(row.get("primary_family_clean") or UNKNOWN), str(row.get("activity_bucket") or UNKNOWN))] += 1
    paths.append(
        heatmap(
            family_activity_matrix,
            families,
            activity_labels,
            out_dir,
            "06_family_activity_heatmap.svg",
            "Family x Maintenance Heatmap",
            "Activity buckets are based on pushed_at or updated_at relative to the newest CSV timestamp",
        )
    )
    return paths


def plot_top_repos(rows: list[dict[str, object]], out_dir: Path) -> Path:
    top = sorted(rows, key=lambda row: as_int(row.get("stars_i")), reverse=True)[:20]
    data = [(repo_name(row), as_int(row.get("stars_i"))) for row in top]
    return horizontal_bar_chart(
        data,
        out_dir,
        "07_top_repositories_by_stars.svg",
        "Top Repositories By Stars",
        "Popularity is log-free here so long-tail dominance is visible",
        {repo_name(row): ROLE_COLORS.get(str(row.get("role_clean")), ROLE_COLORS[UNKNOWN]) for row in top},
        width=1160,
        label_width=330,
    )


def plot_scatters(rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    top_label_names = {
        repo_name(row)
        for row in sorted(rows, key=lambda item: as_int(item.get("stars_i")), reverse=True)[:12]
    }
    paths = []
    paths.append(
        scatter_plot(
            rows,
            out_dir,
            "08_stars_vs_forks.svg",
            "Stars vs Forks",
            f"Log-scale popularity/fork relationship; Spearman rho={spearman([log1p(as_int(r.get('stars_i'))) for r in rows], [log1p(as_int(r.get('forks_i'))) for r in rows]):.2f}",
            lambda row: log1p(as_int(row.get("stars_i"))),
            lambda row: log1p(as_int(row.get("forks_i"))),
            "Stars (log10)",
            "Forks (log10)",
            lambda row: ROLE_COLORS.get(str(row.get("role_clean")), ROLE_COLORS[UNKNOWN]),
            lambda row: repo_short_name(row) if repo_name(row) in top_label_names else "",
        )
    )
    paths.append(
        scatter_plot(
            rows,
            out_dir,
            "09_popularity_vs_activity.svg",
            "Popularity vs Repository Age",
            "Older x-position means longer since last push/update; labels show the most-starred repositories",
            lambda row: log1p(as_int(row.get("age_days") if row.get("age_days") is not None else 0)),
            lambda row: log1p(as_int(row.get("stars_i"))),
            "Days since last activity (log10)",
            "Stars (log10)",
            lambda row: ACTIVITY_COLORS.get(str(row.get("activity_bucket")), ACTIVITY_COLORS[UNKNOWN]),
            lambda row: repo_short_name(row) if repo_name(row) in top_label_names else "",
        )
    )
    issue_rows = [row for row in rows if as_int(row.get("open_issues_i")) > 0 or as_int(row.get("stars_i")) > 0]
    paths.append(
        scatter_plot(
            issue_rows,
            out_dir,
            "10_open_issues_vs_stars.svg",
            "Open Issues vs Stars",
            "Diagnostic view for issue load relative to popularity",
            lambda row: log1p(as_int(row.get("stars_i"))),
            lambda row: log1p(as_int(row.get("open_issues_i"))),
            "Stars (log10)",
            "Open issues (log10)",
            lambda row: FAMILY_COLORS.get(str(row.get("primary_family_clean")), FAMILY_COLORS[UNKNOWN]),
            lambda row: repo_short_name(row) if repo_name(row) in top_label_names else "",
        )
    )
    return paths


def plot_activity_and_quality(rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    paths = []
    activity_counts = [(label, count_by(rows, "activity_bucket").get(label, 0)) for label in ACTIVITY_ORDER]
    paths.append(
        horizontal_bar_chart(
            activity_counts,
            out_dir,
            "11_activity_counts.svg",
            "Repository Maintenance Status",
            "Buckets from pushed_at/updated_at; archived repositories are separated",
            ACTIVITY_COLORS,
        )
    )
    confidence_counts = sorted_counts(count_by(rows, "role_confidence"))
    paths.append(
        horizontal_bar_chart(
            confidence_counts,
            out_dir,
            "12_role_confidence_counts.svg",
            "Role Confidence Distribution",
            "Crawler confidence in public role classification",
            "#4E9A62",
        )
    )
    topic_counts = sorted_counts(count_multi(rows, "topic_values"), 20)
    paths.append(
        horizontal_bar_chart(
            topic_counts,
            out_dir,
            "13_top_github_topics.svg",
            "Top GitHub Topics",
            "Topic metadata from accepted repositories; many older projects have sparse topics",
            "#D9902F",
            width=1120,
            label_width=300,
        )
    )
    return paths


def plot_role_popularity(rows: list[dict[str, object]], out_dir: Path) -> Path:
    roles = [role for role in ROLE_ORDER if any(row.get("role_clean") == role for row in rows)]
    stats: list[tuple[str, int, int, int, int]] = []
    for role in roles:
        values = sorted(as_int(row.get("stars_i")) for row in rows if row.get("role_clean") == role)
        if not values:
            continue
        stats.append(
            (
                role_label(role),
                len(values),
                int(percentile(values, 0.25)),
                int(percentile(values, 0.50)),
                int(percentile(values, 0.75)),
            )
        )

    width = 1050
    height = 112 + len(stats) * 48 + 52
    left = 260
    chart_w = width - left - 96
    max_value = max([q3 for *_rest, q3 in stats] or [1])
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(parts, "Popularity Distribution By Role", "Interquartile range and median stars per role", width)
    top = 84
    for index, (label, count, q1, median, q3) in enumerate(stats):
        y = top + index * 48
        x1 = left + chart_w * q1 / max_value
        xm = left + chart_w * median / max_value
        x3 = left + chart_w * q3 / max_value
        color = ROLE_COLORS.get(roles[index], color_for_index(index))
        parts.append(f'<text class="label" x="{left - 14}" y="{y + 18}" text-anchor="end">{esc(label)} ({count})</text>')
        parts.append(f'<line x1="{left}" y1="{y + 12}" x2="{left + chart_w}" y2="{y + 12}" stroke="#EEF2F5" stroke-width="10"/>')
        parts.append(f'<rect x="{x1:.1f}" y="{y + 4}" width="{max(2, x3 - x1):.1f}" height="17" rx="4" fill="{color}" fill-opacity="0.55"/>')
        parts.append(f'<line x1="{xm:.1f}" y1="{y + 1}" x2="{xm:.1f}" y2="{y + 25}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text class="tiny" x="{xm + 6:.1f}" y="{y + 36}">med {format_int(median)}</text>')
    axis_ticks(parts, left, top + len(stats) * 48 + 2, chart_w, max_value)
    return write_svg(out_dir, "14_role_popularity_distribution.svg", width, height, "\n".join(parts), "Popularity Distribution By Role")


def write_readme(rows: list[dict[str, object]], out_dir: Path, paths: Sequence[Path], input_path: Path) -> Path:
    total = len(rows)
    stars = [as_int(row.get("stars_i")) for row in rows]
    forks = [as_int(row.get("forks_i")) for row in rows]
    active = sum(1 for row in rows if row.get("activity_bucket") in {"Very active", "Active"})
    deployable = sum(1 for row in rows if row.get("role_clean") == "deployable_ids")
    rho = spearman([log1p(value) for value in stars], [log1p(value) for value in forks])
    lines = [
        "# GitHub IDS Repository Plots",
        "",
        f"Input CSV: `{input_path}`",
        f"Rows: `{total}`",
        "",
        "## Summary",
        "",
        f"- Total stars: `{format_int(sum(stars))}`",
        f"- Median stars: `{format_int(statistics.median(stars) if stars else 0)}`",
        f"- Total forks: `{format_int(sum(forks))}`",
        f"- Active within one year: `{format_pct(active / total) if total else '0%'}`",
        f"- Deployable IDS share: `{format_pct(deployable / total) if total else '0%'}`",
        f"- Spearman correlation between log-stars and log-forks: `{rho:.2f}`",
        "",
        "## Figures",
        "",
    ]
    for path in paths:
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Plots use only the compact accepted repository CSV, so they visualize what the crawler kept after filtering and deduplication.",
            "- Activity buckets use `pushed_at` first and fall back to `updated_at`; archived repositories are separated regardless of timestamp.",
            "- Detection type counts are multi-label, so totals can exceed the number of repositories.",
            "- The output directory intentionally uses `github_plots` to avoid overwriting the older `github-plots` figures.",
        ]
    )
    path = out_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_plots(input_path: Path, out_dir: Path) -> list[Path]:
    rows = load_rows(input_path)
    paths: list[Path] = []
    paths.append(plot_overview(rows, out_dir))
    paths.extend(plot_counts(rows, out_dir))
    paths.extend(plot_heatmaps(rows, out_dir))
    paths.append(plot_top_repos(rows, out_dir))
    paths.extend(plot_scatters(rows, out_dir))
    paths.extend(plot_activity_and_quality(rows, out_dir))
    paths.append(plot_role_popularity(rows, out_dir))
    paths.append(write_readme(rows, out_dir, paths, input_path))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create improved SVG plots for GitHub IDS repository crawl output.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to ids_repos.csv.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for SVG outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV does not exist: {args.input}")
    paths = build_plots(args.input, args.out_dir)
    print(f"Wrote {len(paths)} files to {args.out_dir}")
    for path in paths:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
