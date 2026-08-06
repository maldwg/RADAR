#!/usr/bin/env python3

"""Create SVG summary plots for the curated IDS repository list.

The script intentionally uses only the Python standard library so it can run in
the lightweight thesis environment without pandas/matplotlib.
"""
# last execution: python3 data-gathering/github-crawler.py   --max-repos 200   --min-stars 100 --max-docs-pages 5 --include-archived --min-score 5 --anonymous-friend
# ly --max-pages-per-query 10

from __future__ import annotations

import csv
import html
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT / "data-gathering" / "ids_repos.llm.csv"
OUTDIR = Path(f"{Path(__file__).resolve().parent}/github-plots")
os.makedirs(OUTDIR, exist_ok=True)
AS_OF = datetime(2026, 5, 21, tzinfo=timezone.utc)
REMOVED_OUTPUTS = (
    "ids_repo_solution_groups.svg",
    "ids_repo_family_activity_heatmap.svg",
    "ids_repo_solution_family_heatmap.svg",
)

PALETTE = {
    "blue": "#2f6f9f",
    "teal": "#2a9d8f",
    "green": "#5b8f3a",
    "orange": "#d1842f",
    "red": "#bc4b51",
    "purple": "#7b5ea7",
    "gray": "#6f7782",
    "light": "#eef2f5",
    "ink": "#263238",
    "grid": "#d8dee6",
}

FAMILY_COLORS = {
    "NIDS": "#2f6f9f",
    "HIDS": "#5b8f3a",
    "HIPS": "#bc4b51",
    "IPS": "#d1842f",
    "NIPS": "#c96c2f",
    "NDR": "#7b5ea7",
    "PIDS": "#9a7b22",
    "WIDS": "#2a9d8f",
    "WIPS": "#287f75",
    "DBIDS": "#8f6a3a",
    "Unclassified": "#6f7782",
}

ACTIVITY_ORDER = [
    "Very active (<=90d)",
    "Active (<=1y)",
    "Quiet (1-3y)",
    "Stale (3-5y)",
    "Dormant (>5y)",
    "Archived",
    "Unknown",
]

ACTIVITY_LABELS = {
    "Very active (<=90d)": "<=90d",
    "Active (<=1y)": "<=1y",
    "Quiet (1-3y)": "1-3y",
    "Stale (3-5y)": "3-5y",
    "Dormant (>5y)": ">5y",
    "Archived": "arch.",
    "Unknown": "?",
}

ACTIVITY_COLORS = {
    "Very active (<=90d)": PALETTE["teal"],
    "Active (<=1y)": PALETTE["green"],
    "Quiet (1-3y)": PALETTE["orange"],
    "Stale (3-5y)": PALETTE["red"],
    "Dormant (>5y)": "#8d3b4f",
    "Archived": PALETTE["gray"],
    "Unknown": "#9aa3ad",
}

ROLE_LABELS = {
    "deployable_ids": "Deployable IDS",
    "classifier_model": "Classifier/model",
    "platform_framework": "Platform/framework",
    "rules_content": "Rules/content",
    "support_tooling": "Support tooling",
    "research_demo": "Research/demo",
}

ROLE_SHORT_LABELS = {
    "deployable_ids": "Dep.",
    "platform_framework": "Plat.",
    "classifier_model": "Class.",
    "rules_content": "Rules",
    "support_tooling": "Supp.",
    "research_demo": "Demo",
}

ROLE_COLORS = {
    "deployable_ids": PALETTE["teal"],
    "classifier_model": PALETTE["orange"],
    "platform_framework": PALETTE["purple"],
    "rules_content": PALETTE["blue"],
    "support_tooling": PALETTE["gray"],
    "research_demo": "#9a7b22",
}

ROLE_RADII = {
    "deployable_ids": 4.4,
    "platform_framework": 3.8,
    "classifier_model": 3.3,
    "rules_content": 2.9,
    "support_tooling": 2.5,
    "research_demo": 2.2,
}

ROLE_ORDER = [
    "deployable_ids",
    "platform_framework",
    "classifier_model",
    "rules_content",
    "support_tooling",
    "research_demo",
]


def esc(value):
    return html.escape(str(value), quote=True)


def parse_bool(value):
    return str(value).strip().lower() == "true"


def parse_int(value, default=0):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_dt(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_rows():
    with INFILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["stars_i"] = parse_int(row.get("stars"))
        row["forks_i"] = parse_int(row.get("forks"))
        row["watchers_i"] = parse_int(row.get("watchers"))
        row["open_issues_i"] = parse_int(row.get("open_issues"))
        row["archived_b"] = parse_bool(row.get("archived"))
        row["pushed_dt"] = parse_dt(row.get("pushed_at")) or parse_dt(row.get("updated_at"))
        if row["pushed_dt"]:
            row["age_days"] = max(0, (AS_OF - row["pushed_dt"]).days)
        else:
            row["age_days"] = None
    return rows


def svg_document(width, height, content, title=None):
    title_node = f"<title>{esc(title)}</title>" if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"{title_node}\n"
        '<style>\n'
        "text { font-family: Arial, Helvetica, sans-serif; fill: #263238; }\n"
        ".title { font-size: 22px; font-weight: 700; }\n"
        ".subtitle { font-size: 12px; fill: #6f7782; }\n"
        ".axis { font-size: 11px; fill: #59636f; }\n"
        ".label { font-size: 12px; }\n"
        ".small { font-size: 10px; fill: #59636f; }\n"
        ".tiny { font-size: 7px; fill: #43505c; }\n"
        ".tiny-strong { font-size: 8px; font-weight: 700; fill: #263238; }\n"
        ".value { font-size: 11px; font-weight: 700; fill: #263238; }\n"
        "</style>\n"
        f"{content}\n"
        "</svg>\n"
    )


def write_svg(name, width, height, content, title):
    path = OUTDIR / name
    path.write_text(svg_document(width, height, content, title), encoding="utf-8")
    return path


def remove_deprecated_outputs():
    for name in REMOVED_OUTPUTS:
        (OUTDIR / name).unlink(missing_ok=True)


def add_title(parts, title, subtitle, width):
    parts.append(f'<text class="title" x="{width / 2}" y="34" text-anchor="middle">{esc(title)}</text>')
    if subtitle:
        parts.append(f'<text class="subtitle" x="{width / 2}" y="54" text-anchor="middle">{esc(subtitle)}</text>')


def horizontal_bar_chart(
    data,
    output,
    title,
    subtitle,
    color="#2f6f9f",
    width=1000,
    row_height=31,
    label_width=260,
    right_pad=84,
):
    data = [(label, value) for label, value in data if value > 0]
    height = 92 + max(1, len(data)) * row_height + 28
    top = 82
    bar_h = 18
    chart_w = width - label_width - right_pad
    max_value = max([value for _, value in data] or [1])

    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(parts, title, subtitle, width)

    for i, (label, value) in enumerate(data):
        y = top + i * row_height
        bar_w = chart_w * value / max_value
        max_chars = max(14, int(label_width / 8.2))
        shown_label = shorten(label, max_chars)
        bar_color = color.get(label, PALETTE["blue"]) if isinstance(color, dict) else color
        parts.append(
            f'<text class="label" x="{label_width - 12}" y="{y + 14}" text-anchor="end">'
            f"{esc(shown_label)}"
            f"<title>{esc(label)}</title>"
            "</text>"
        )
        parts.append(
            f'<rect x="{label_width}" y="{y}" width="{chart_w}" height="{bar_h}" '
            f'rx="3" fill="{PALETTE["light"]}"/>'
        )
        parts.append(f'<rect x="{label_width}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" rx="3" fill="{bar_color}"/>')
        parts.append(f'<text class="value" x="{label_width + bar_w + 8:.1f}" y="{y + 14}">{value}</text>')

    return write_svg(output, width, height, "\n".join(parts), title)


def shorten(label, max_chars):
    label = str(label)
    if len(label) <= max_chars:
        return label
    return label[: max_chars - 1] + "…"


def hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(channel))):02x}" for channel in rgb)


def mix_with_white(color, strength):
    strength = max(0.0, min(1.0, strength))
    rgb = hex_to_rgb(color)
    mixed = tuple(255 - (255 - channel) * strength for channel in rgb)
    return rgb_to_hex(mixed)


def primary_family(row):
    return row.get("primary_family") or "Unclassified"


def ordered_primary_families(rows):
    counts = Counter(primary_family(row) for row in rows)
    return sorted(
        counts,
        key=lambda family: (
            family == "Unclassified",
            -counts[family],
            family.lower(),
        ),
    )


def repo_role(row):
    role = (row.get("role") or "").strip()
    if role in ROLE_LABELS:
        return role

    # Backward-compatible fallback for older ids_repos.csv snapshots.
    if str(row.get("fully_fledged_ids", "")).strip():
        return "deployable_ids"
    if str(row.get("classifier_only", "")).strip():
        return "classifier_model"
    if str(row.get("framework_only", "")).strip():
        return "platform_framework"

    kind = (row.get("repo_kind") or "").lower()
    text = " ".join(
        str(row.get(key) or "")
        for key in ("type", "description", "matched_families", "evidence", "topics", "role_evidence")
    ).lower()

    if kind in {"dataset", "paper", "list", "tutorial", "research-code"}:
        return "research_demo"
    if "ruleset" in text or "detection rules" in text or "threat-hunting content" in text:
        return "rules_content"
    if any(term in text for term in ("classifier", "dnn-based", "ml-based", "ann-based", "deep learning", "neural network", "training script")):
        return "classifier_model"
    if kind == "framework" or any(term in text for term in ("distribution", "suite/wrapper", "framework", "platform", "workbench")):
        return "platform_framework"
    if "base detector evidence" in text or "runtime detector" in text or "standalone" in text:
        return "deployable_ids"
    return "support_tooling"


def repo_role_label(row):
    return row.get("role_label") or ROLE_LABELS.get(repo_role(row), "Unclassified")


def format_stars(value):
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def format_age(days):
    if days is None:
        return "?"
    if days < 14:
        return f"{days}d"
    if days < 365:
        return f"{round(days / 30)}mo"
    return f"{days / 365:.1f}y"


def family_coverage_plot(rows):
    families = ordered_primary_families(rows)
    primary = Counter(primary_family(row) for row in rows)
    total = len(rows)

    width = 920
    left = 170
    top = 92
    row_h = 36
    height = top + max(1, len(families)) * row_h + 72
    chart_w = 600
    max_value = max(primary.values() or [1])
    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(
        parts,
        "Primary IDS Families",
        "Each repository is counted once using the curated `primary_family` column",
        width,
    )

    for i, family in enumerate(families):
        y = top + i * row_h
        value = primary[family]
        bar_w = chart_w * value / max_value
        pct = value / total * 100 if total else 0
        c = FAMILY_COLORS.get(family, FAMILY_COLORS["Unclassified"])
        parts.append(f'<text class="label" x="{left - 14}" y="{y + 17}" text-anchor="end">{family}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{chart_w}" height="20" rx="4" fill="{PALETTE["light"]}"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="20" rx="4" fill="{c}"/>')
        parts.append(f'<text class="value" x="{left + bar_w + 10:.1f}" y="{y + 15}">{value} ({pct:.0f}%)</text>')

    note_y = top + len(families) * row_h + 26
    parts.append(
        f'<text class="subtitle" x="{left}" y="{note_y}">'
        "Unclassified means the row is retained in the repository list but has no single primary IDS family assigned."
        "</text>"
    )

    return write_svg("ids_repo_family_coverage.svg", width, height, "\n".join(parts), "Primary IDS Families")


def activity_bucket(row):
    if row["archived_b"]:
        return "Archived"
    days = row.get("age_days")
    if days is None:
        return "Unknown"
    if days <= 90:
        return "Very active (<=90d)"
    if days <= 365:
        return "Active (<=1y)"
    if days <= 3 * 365:
        return "Quiet (1-3y)"
    if days <= 5 * 365:
        return "Stale (3-5y)"
    return "Dormant (>5y)"


def activity_plot(rows):
    counts = Counter(activity_bucket(row) for row in rows)
    data = [(label, counts[label]) for label in ACTIVITY_ORDER if counts[label]]
    return horizontal_bar_chart(
        data,
        "ids_repo_activity_status.svg",
        "Repository Activity",
        "Buckets use pushed_at relative to 2026-05-19; archived repos are separated",
        color=ACTIVITY_COLORS,
        label_width=230,
        width=900,
    )


def popularity_activity_scatter(rows):
    width = 1180
    height = 640
    left = 96
    right = 46
    top = 82
    bottom = 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_age = max((row["age_days"] or 0 for row in rows), default=1)
    max_age = max(365, max_age)
    max_age_log = math.log10(max_age + 1)
    max_stars = max((row["stars_i"] for row in rows), default=1)
    min_stars = max(1, min((row["stars_i"] for row in rows), default=1))
    min_log = math.log10(max(1, min_stars * 0.82))
    max_log = math.log10(max_stars * 1.12)

    def x_for_age(days):
        value = math.log10((days or 0) + 1)
        return left + value / max_age_log * plot_w

    def y_for_stars(stars):
        value = math.log10(max(1, stars))
        return top + plot_h - ((value - min_log) / (max_log - min_log) * plot_h)

    def radius_for_repo(row):
        return ROLE_RADII.get(repo_role(row), 2.4)

    def jitter(row, salt, spread):
        seed = sum((idx + 1 + salt) * ord(ch) for idx, ch in enumerate(row["full_name"]))
        return ((seed % 1000) / 999 - 0.5) * spread

    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(
        parts,
        "Popularity vs. Recent Activity",
        "Color shows primary family; point size shows inferred repository role; both axes are log-scaled",
        width,
    )

    # Subtle age bands keep the transformed x-axis readable.
    for start, end, fill in [
        (0, 90, "#f2faf8"),
        (90, 365, "#f6faf1"),
        (365, 3 * 365, "#fff8ed"),
        (3 * 365, max_age, "#fbf0f2"),
    ]:
        x1 = x_for_age(start)
        x2 = x_for_age(min(end, max_age))
        if x2 > x1:
            parts.append(f'<rect x="{x1:.1f}" y="{top}" width="{x2 - x1:.1f}" height="{plot_h}" fill="{fill}"/>')

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{PALETTE["ink"]}"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{PALETTE["ink"]}"/>')

    for days, label in [(0, "0d"), (7, "7d"), (30, "30d"), (90, "90d"), (365, "1y"), (3 * 365, "3y"), (5 * 365, "5y"), (max_age, f"{round(max_age / 365, 1)}y")]:
        if days <= max_age:
            x = x_for_age(days)
            parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{PALETTE["grid"]}"/>')
            parts.append(f'<text class="axis" x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle">{label}</text>')

    y_ticks = [100, 200, 500, 1000, 3000, 10000, 20000]
    for tick in y_ticks:
        if min_stars * 0.75 <= tick <= max_stars * 1.2:
            y = y_for_stars(tick)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{PALETTE["grid"]}"/>')
            parts.append(f'<text class="axis" x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{tick:,}</text>')

    parts.append(f'<text class="axis" x="{left + plot_w / 2}" y="{height - 28}" text-anchor="middle">days since last push (log scale; recent projects on the left)</text>')
    parts.append(
        f'<text class="axis" x="24" y="{top + plot_h / 2}" text-anchor="middle" '
        'transform="rotate(-90 24 '
        f'{top + plot_h / 2})">GitHub stars (log scale, fitted range)</text>'
    )

    strong_labels = {row["full_name"] for row in sorted(rows, key=lambda item: item["stars_i"], reverse=True)[:10]}
    plotted = []

    def label_offset(row, index):
        seed = sum((idx + 5) * ord(ch) for idx, ch in enumerate(row["full_name"])) + index * 37
        angle = math.radians(seed % 360)
        distance = 8 + seed % 12
        return math.cos(angle) * distance, math.sin(angle) * distance

    for row in sorted(rows, key=lambda item: item["stars_i"]):
        age = row["age_days"] or 0
        spread_x = 20 if age <= 90 else 12
        x = max(left + 4, min(left + plot_w - 4, x_for_age(age) + jitter(row, 3, spread_x)))
        y = max(top + 4, min(top + plot_h - 4, y_for_stars(row["stars_i"]) + jitter(row, 17, 12)))
        family = row.get("primary_family") or "Unclassified"
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["Unclassified"])
        radius = radius_for_repo(row)
        opacity = "0.35" if activity_bucket(row).startswith(("Stale", "Dormant")) else "0.78"
        role_label = repo_role_label(row)
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" '
            f'opacity="{opacity}" stroke="white" stroke-width="0.8">'
            f"<title>{esc(row['full_name'])}: {row['stars_i']:,} stars, "
            f"{role_label}, {age} days since push</title>"
            "</circle>"
        )
        plotted.append((row, x, y))

    for index, (row, x, y) in enumerate(plotted):
        short = shorten(row["full_name"].split("/")[-1], 18)
        dx, dy = label_offset(row, index)
        label_x = max(left + 6, min(left + plot_w - 6, x + dx))
        label_y = max(top + 8, min(top + plot_h - 4, y + dy))
        anchor = "start" if dx >= 0 else "end"
        line_x2 = label_x - 2 if anchor == "start" else label_x + 2
        text_class = "tiny-strong" if row["full_name"] in strong_labels else "tiny"
        parts.append(
            f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{line_x2:.1f}" y2="{label_y - 3:.1f}" '
            'stroke="#c3ccd6" stroke-width="0.45" opacity="0.72"/>'
        )
        parts.append(
            f'<text class="{text_class}" x="{label_x:.1f}" y="{label_y:.1f}" '
            f'text-anchor="{anchor}">{esc(short)}</text>'
        )

    families = ordered_primary_families(rows)
    legend_x = max(left + 16, left + plot_w - len(families) * 76)
    legend_y = top + 18
    for i, family in enumerate(families):
        x = legend_x + i * 76
        color = FAMILY_COLORS.get(family, FAMILY_COLORS["Unclassified"])
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(f'<text class="small" x="{x + 9}" y="{legend_y + 4}">{esc(family)}</text>')

    size_y = top + 48
    parts.append(f'<text class="small" x="{legend_x}" y="{size_y}">Size: role</text>')
    x = legend_x + 74
    for role_key in ROLE_ORDER[:4]:
        radius = ROLE_RADII[role_key]
        parts.append(
            f'<circle cx="{x}" cy="{size_y - 3:.1f}" r="{radius:.1f}" '
            f'fill="{ROLE_COLORS[role_key]}" opacity="0.72" stroke="white" stroke-width="0.8"/>'
        )
        parts.append(f'<text class="small" x="{x + 10}" y="{size_y + 1}">{esc(ROLE_LABELS[role_key])}</text>')
        x += 132 if role_key != "platform_framework" else 150

    return write_svg("ids_repo_popularity_activity.svg", width, height, "\n".join(parts), "Popularity vs. Activity")


def top_projects_plot(rows):
    data = [(row["full_name"], row["stars_i"]) for row in sorted(rows, key=lambda item: item["stars_i"], reverse=True)[:15]]
    return horizontal_bar_chart(
        data,
        "ids_repo_top_projects.svg",
        "Most Starred IDS-Related Repositories",
        "Top 15 entries from ids_repos.csv",
        color=PALETTE["blue"],
        width=1100,
        label_width=360,
        row_height=30,
    )


def repo_feature_matrix(rows):
    grouped = []
    for family in ordered_primary_families(rows):
        family_rows = [
            row for row in rows
            if primary_family(row) == family
        ]
        if not family_rows:
            continue
        grouped.append(("header", family, len(family_rows)))
        for row in sorted(family_rows, key=lambda item: (-item["stars_i"], item["full_name"].lower())):
            grouped.append(("row", row, None))

    width = 1500
    row_h = 23
    header_h = 28
    top = 104
    body_h = sum(header_h if item[0] == "header" else row_h for item in grouped)
    height = top + body_h + 74
    max_stars = max((row["stars_i"] for row in rows), default=1)
    max_forks = max((row["forks_i"] for row in rows), default=1)
    max_issues = max((row["open_issues_i"] for row in rows), default=1)

    cols = {
        "repo": 28,
        "type": 300,
        "deployable_ids": 520,
        "platform_framework": 560,
        "classifier_model": 602,
        "rules_content": 646,
        "support_tooling": 688,
        "research_demo": 732,
        "activity": 786,
        "stars": 884,
        "forks": 1040,
        "issues": 1168,
        "language": 1298,
        "confidence": 1402,
    }

    parts = [f'<rect width="{width}" height="{height}" fill="white"/>']
    add_title(
        parts,
        "Repository Feature Matrix",
        "Grouped by primary family; filled role dots separate deployable IDS from classifiers, wrappers, content, and support tooling",
        width,
    )

    headers = [
        ("Repository", "repo"),
        ("Type", "type"),
        ("Activity", "activity"),
        ("Stars", "stars"),
        ("Forks", "forks"),
        ("Issues", "issues"),
        ("Lang.", "language"),
        ("Conf.", "confidence"),
    ]
    for label, key in headers:
        parts.append(f'<text class="axis" x="{cols[key]}" y="{top - 14}">{label}</text>')
    for role_key in ROLE_ORDER:
        parts.append(
            f'<text class="axis" x="{cols[role_key]}" y="{top - 14}" text-anchor="middle">'
            f'{ROLE_SHORT_LABELS[role_key]}<title>{esc(ROLE_LABELS[role_key])}</title></text>'
        )
    parts.append(f'<line x1="24" y1="{top - 5}" x2="{width - 24}" y2="{top - 5}" stroke="{PALETTE["grid"]}"/>')

    y = top
    row_index = 0
    for kind, payload, count in grouped:
        if kind == "header":
            family = payload
            color = FAMILY_COLORS.get(family, FAMILY_COLORS["Unclassified"])
            parts.append(f'<rect x="18" y="{y - 1}" width="{width - 36}" height="{header_h - 6}" rx="4" fill="{mix_with_white(color, 0.2)}"/>')
            parts.append(f'<rect x="18" y="{y - 1}" width="9" height="{header_h - 6}" rx="4" fill="{color}"/>')
            parts.append(f'<text class="value" x="36" y="{y + 16}">{family}</text>')
            parts.append(f'<text class="small" x="95" y="{y + 16}">{count} repositories</text>')
            y += header_h
            continue

        row = payload
        if row_index % 2 == 0:
            parts.append(f'<rect x="18" y="{y - 2}" width="{width - 36}" height="{row_h}" fill="#fbfcfd"/>')

        activity = activity_bucket(row)
        role = repo_role(row)
        star_w = math.log10(row["stars_i"] + 1) / math.log10(max_stars + 1) * 120
        fork_w = math.log10(row["forks_i"] + 1) / math.log10(max_forks + 1) * 78 if max_forks else 0
        issue_w = math.log10(row["open_issues_i"] + 1) / math.log10(max_issues + 1) * 78 if max_issues else 0

        parts.append(
            f'<text class="small" x="{cols["repo"]}" y="{y + 12}">'
            f'{esc(shorten(row["full_name"], 37))}<title>{esc(row["full_name"])}</title></text>'
        )
        parts.append(
            f'<text class="small" x="{cols["type"]}" y="{y + 12}">'
            f'{esc(shorten(row.get("type") or "", 27))}<title>{esc(row.get("type") or "")}</title></text>'
        )

        for role_key in ROLE_ORDER:
            is_match = role == role_key
            fill = ROLE_COLORS[role_key] if is_match else "white"
            stroke = ROLE_COLORS[role_key] if is_match else "#c9d0d8"
            opacity = "1" if is_match else "0.85"
            title = row.get("role_evidence") if is_match else ROLE_LABELS[role_key]
            parts.append(
                f'<circle cx="{cols[role_key]}" cy="{y + 8}" r="4.7" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1" opacity="{opacity}">'
                f'<title>{esc(title or ROLE_LABELS[role_key])}</title></circle>'
            )

        activity_color = ACTIVITY_COLORS.get(activity, PALETTE["gray"])
        parts.append(f'<rect x="{cols["activity"]}" y="{y + 2}" width="64" height="13" rx="3" fill="{activity_color}"/>')
        parts.append(
            f'<text class="small" x="{cols["activity"] + 32}" y="{y + 12}" text-anchor="middle" fill="white">'
            f'{esc(ACTIVITY_LABELS.get(activity, "?"))}</text>'
        )

        parts.append(f'<rect x="{cols["stars"]}" y="{y + 4}" width="120" height="8" rx="2" fill="{PALETTE["light"]}"/>')
        parts.append(f'<rect x="{cols["stars"]}" y="{y + 4}" width="{star_w:.1f}" height="8" rx="2" fill="{PALETTE["blue"]}"/>')
        parts.append(f'<text class="small" x="{cols["stars"] + 126}" y="{y + 12}" text-anchor="start">{format_stars(row["stars_i"])}</text>')

        parts.append(f'<rect x="{cols["forks"]}" y="{y + 4}" width="78" height="8" rx="2" fill="{PALETTE["light"]}"/>')
        parts.append(f'<rect x="{cols["forks"]}" y="{y + 4}" width="{fork_w:.1f}" height="8" rx="2" fill="{PALETTE["purple"]}"/>')
        parts.append(f'<text class="small" x="{cols["forks"] + 84}" y="{y + 12}">{format_stars(row["forks_i"])}</text>')

        parts.append(f'<rect x="{cols["issues"]}" y="{y + 4}" width="78" height="8" rx="2" fill="{PALETTE["light"]}"/>')
        parts.append(f'<rect x="{cols["issues"]}" y="{y + 4}" width="{issue_w:.1f}" height="8" rx="2" fill="{PALETTE["orange"]}"/>')
        parts.append(f'<text class="small" x="{cols["issues"] + 84}" y="{y + 12}">{row["open_issues_i"]}</text>')

        parts.append(f'<text class="small" x="{cols["language"]}" y="{y + 12}">{esc(shorten(row.get("language") or "-", 12))}</text>')
        confidence = row.get("role_confidence") or "-"
        parts.append(f'<text class="small" x="{cols["confidence"]}" y="{y + 12}">{esc(confidence)}</text>')

        y += row_h
        row_index += 1

    legend_y = height - 42
    legend_x = 28
    parts.append(f'<text class="small" x="{legend_x}" y="{legend_y}">Activity colors: </text>')
    x = legend_x + 82
    for bucket in ACTIVITY_ORDER[:5]:
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="34" height="10" rx="3" fill="{ACTIVITY_COLORS[bucket]}"/>')
        parts.append(f'<text class="small" x="{x + 40}" y="{legend_y}">{esc(ACTIVITY_LABELS[bucket])}</text>')
        x += 84

    role_x = x + 16
    parts.append(f'<text class="small" x="{role_x}" y="{legend_y}">Role dots:</text>')
    role_x += 70
    for role_key in ROLE_ORDER:
        parts.append(f'<circle cx="{role_x + 5}" cy="{legend_y - 5}" r="4.5" fill="{ROLE_COLORS[role_key]}"/>')
        parts.append(f'<text class="small" x="{role_x + 18}" y="{legend_y}">{esc(ROLE_LABELS[role_key])}</text>')
        role_x += 126 if role_key != "platform_framework" else 148

    parts.append(
        f'<text class="small" x="{width - 28}" y="{height - 18}" text-anchor="end">'
        f'Data acquired: {AS_OF.date().isoformat()}</text>'
    )

    return write_svg("ids_repo_feature_matrix.svg", width, height, "\n".join(parts), "Repository Feature Matrix")


def main():
    remove_deprecated_outputs()
    rows = load_rows()

    exact_types = Counter(row.get("type") or "Unclassified" for row in rows)
    type_data = sorted(exact_types.items(), key=lambda item: (-item[1], item[0].lower()))
    horizontal_bar_chart(
        type_data,
        "ids_repo_type_counts.svg",
        "Inferred IDS Type Labels",
        "Counts by the inferred `type` column in ids_repos.csv",
        color=PALETTE["blue"],
        width=1050,
        label_width=310,
        row_height=29,
    )

    family_coverage_plot(rows)

    activity_plot(rows)
    popularity_activity_scatter(rows)
    top_projects_plot(rows)
    repo_feature_matrix(rows)

    print("Generated:")
    for path in sorted(OUTDIR.glob("ids_repo_*.svg")):
        try:
            shown_path = path.relative_to(ROOT)
        except ValueError:
            shown_path = path
        print(f"  {shown_path}")


if __name__ == "__main__":
    main()
