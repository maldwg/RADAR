#!/usr/bin/env python3
"""Compare keyword-crawler and zero-shot-LLM GitHub repository classifications.

The script uses only the standard library and writes portable SVG figures plus
CSV audit tables.  It accepts more than one LLM output so separate models,
prompts, or runs can be compared without changing the code.

Examples:
  python3 figure_creation/github-classifier-comparison/compare_keyword_llm_classifiers.py

  python3 figure_creation/github-classifier-comparison/compare_keyword_llm_classifiers.py \
    --llm qwen=data-gathering/ids_repos.llm_accepted.csv \
    --llm llama=data-gathering/ids_repos.llama_accepted.csv

Input LLM files are produced by ``data-gathering/github-llm-classifier.py``.
They may be partial runs: every figure states its matched coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
except ImportError:  # The set-overlap map still works without NumPy.
    np = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEYWORD_EVALUATED = ROOT / "data-gathering" / "ids_repos.evaluated.csv"
DEFAULT_KEYWORD_ACCEPTED = ROOT / "data-gathering" / "ids_repos.csv"
DEFAULT_LLM = ROOT / "data-gathering" / "ids_repos.llm_accepted.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "output"

UNKNOWN = "Unknown"
PALETTE = ["#2F6B9A", "#4E9A62", "#D9902F", "#7A5FA8", "#C94F4F", "#3B8C8C", "#B85C9E", "#8C6D31", "#4F78C4"]
DECISION_COLORS = {"accept": "#2F8F5B", "exclude": "#C94F4F", "uncertain": "#D9902F", "error": "#6C7A89", "missing": "#AAB3BC"}
OVERLAP_COLORS = {
    "both_accept": "#238B57",
    "both_exclude": "#82909B",
    "llm_only_accept": "#D05A35",
    "keyword_only_accept": "#7055A1",
    "uncertain_or_error": "#D9902F",
}
ROLE_ORDER = ["deployable_ids", "platform_framework", "classifier_model", "support_tooling", "rules_content", "research_demo", "out_of_scope", "uncertain", "rejected"]
FAMILY_ORDER = ["NIDS", "NIPS", "HIDS", "HIPS", "IDS", "IPS", "WIDS", "WIPS", "DBIDS", "DBIPS", "PIDS", "UNKNOWN"]
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+-]{2,}", re.I)
STOPWORDS = {"and", "are", "for", "from", "into", "its", "not", "the", "this", "that", "with", "you", "your", "using", "repository", "github", "project", "tool"}


def configure_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def clean(value: Any, default: str = UNKNOWN) -> str:
    text = str(value or "").strip()
    return text if text else default


def key(value: Any) -> str:
    return str(value or "").strip().casefold()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def split_values(value: Any) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in re.split(r"[;,]", str(value or "")) if part.strip()))


def primary_family(value: Any) -> str:
    values = [item.upper() for item in split_values(value)]
    return next((item for item in values if item != "UNKNOWN"), values[0] if values else "UNKNOWN")


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def shorten(value: Any, limit: int = 56) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def svg(width: int, height: int, title: str, body: Iterable[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{esc(title)}</title>
<style>
text {{ font-family: Arial, Helvetica, sans-serif; fill: #263238; }}
.title {{ font-size: 21px; font-weight: 700; }} .sub {{ font-size: 12px; fill: #5f6c76; }}
.axis {{ font-size: 11px; fill: #43505c; }} .small {{ font-size: 10px; fill: #5f6c76; }}
.value {{ font-size: 11px; font-weight: 700; }} .grid {{ stroke: #d8dee6; stroke-width: 1; }}
</style>
<!-- The outer SVG remains transparent; this card keeps content readable on dark viewers. -->
<rect x="5" y="5" width="{width - 10}" height="{height - 10}" rx="8" fill="#ffffff" stroke="#d8dee6"/>
{''.join(body)}</svg>'''


def write_svg(out_dir: Path, name: str, width: int, height: int, title: str, body: Iterable[str]) -> None:
    (out_dir / name).write_text(svg(width, height, title, body), encoding="utf-8")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "run"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def keyword_role(row: dict[str, str]) -> str:
    try:
        score_card = json.loads(row.get("score_card_json") or "{}")
    except json.JSONDecodeError:
        score_card = {}
    role = str(score_card.get("role") or "").strip()
    if role:
        return role
    repo_role = str(row.get("repo_role") or "").strip()
    return {"full": "deployable_ids", "framework": "platform_framework", "classifier": "classifier_model"}.get(repo_role, "rejected")


def load_keyword(evaluated_path: Path, accepted_path: Path | None) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw in load_csv(evaluated_path):
        name = key(raw.get("full_name"))
        if not name:
            continue
        rows[name] = {
            "full_name": raw.get("full_name", ""),
            "html_url": raw.get("html_url", ""),
            "description": raw.get("description", ""), "topics": raw.get("topics", ""),
            "keyword_decision": "accept" if raw.get("accepted") == "." else "exclude",
            "keyword_family": primary_family(raw.get("primary_family")),
            "keyword_role": keyword_role(raw),
            "keyword_score": raw.get("score", ""),
        }
    if accepted_path and accepted_path.exists():
        for raw in load_csv(accepted_path):
            name = key(raw.get("full_name"))
            if name not in rows:
                continue
            rows[name].update({
                "keyword_decision": "accept", "keyword_family": primary_family(raw.get("primary_family")),
                "keyword_role": raw.get("role") or rows[name]["keyword_role"],
            })
    return rows


def parse_llm_argument(value: str, position: int) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        if label.strip() and path.strip():
            return label.strip(), Path(path.strip())
    path = Path(value)
    return path.stem.replace("ids_repos.", "").replace("_classified", "") or f"llm_{position}", path


def load_llm(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for raw in load_csv(path):
        name = key(raw.get("full_name"))
        if not name:
            continue
        decision = clean(raw.get("decision"), "error").casefold()
        if decision not in {"accept", "exclude", "uncertain", "error"}:
            decision = "error"
        rows[name] = {
            "full_name": raw.get("full_name", ""), "html_url": raw.get("html_url", ""),
            "description": raw.get("description", ""), "topics": raw.get("topics", ""),
            "llm_model": raw.get("llm_model", ""), "llm_decision": decision,
            "llm_scope": clean(raw.get("scope_gate"), "missing").casefold(),
            "scope_gate_confidence": raw.get("scope_gate_confidence", ""),
            "llm_role": clean(raw.get("llm_role"), "missing").casefold(),
            "llm_family": primary_family(raw.get("families")),
            "llm_methods": ";".join(split_values(raw.get("detection_methods"))) or "UNKNOWN",
            "role_confidence": raw.get("role_confidence", ""),
            "family_confidence": raw.get("family_confidence", ""),
            "method_confidence": raw.get("method_confidence", ""),
            "reason": raw.get("primary_reason", ""),
            "evidence_quote_verified": raw.get("evidence_quote_verified", ""),
        }
    return rows


def comparison_view(keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, str]], list[str], bool]:
    """Expand an accepted-only LLM file into an accepted-set comparison view.

    A missing row is intentionally labelled ``not_accepted`` rather than
    ``exclude``: an accepted file may come from an incomplete LLM run.
    """
    accepted_only = bool(llm) and all(row["llm_decision"] == "accept" for row in llm.values())
    if not accepted_only:
        return llm, sorted(set(keyword) & set(llm)), False
    expanded = dict(llm)
    for name, keyword_row in keyword.items():
        if keyword_row["keyword_decision"] != "accept" or name in expanded:
            continue
        expanded[name] = {
            "full_name": keyword_row["full_name"], "html_url": keyword_row["html_url"],
            "description": keyword_row.get("description", ""), "topics": keyword_row.get("topics", ""),
            "llm_model": "", "llm_decision": "not_accepted", "llm_scope": "missing",
            "scope_gate_confidence": "", "llm_role": "missing", "llm_family": "UNKNOWN",
            "llm_methods": "UNKNOWN", "role_confidence": "", "family_confidence": "",
            "method_confidence": "", "reason": "Not present in LLM accepted output.",
            "evidence_quote_verified": "",
        }
    return expanded, sorted(expanded), True


def ordered(values: Iterable[str], preferred: list[str], max_items: int = 14) -> list[str]:
    counts = Counter(values)
    result = [item for item in preferred if item in counts]
    remainder = sorted((item for item in counts if item not in result), key=lambda item: (-counts[item], item))
    return (result + remainder)[:max_items]


def matrix_figure(out_dir: Path, name: str, title: str, subtitle: str, matrix: dict[tuple[str, str], int], rows: list[str], cols: list[str]) -> None:
    cell, left, top = 74, 170, 112
    width, height = left + cell * len(cols) + 45, top + cell * len(rows) + 60
    max_value = max(matrix.values(), default=1)
    body = [f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">{esc(title)}</text>', f'<text class="sub" x="{width / 2}" y="51" text-anchor="middle">{esc(subtitle)}</text>']
    for col_index, col in enumerate(cols):
        x = left + col_index * cell + cell / 2
        body.append(f'<text class="axis" x="{x}" y="{top - 11}" text-anchor="middle">{esc(shorten(col, 11))}</text>')
    for row_index, row in enumerate(rows):
        y = top + row_index * cell
        body.append(f'<text class="axis" x="{left - 9}" y="{y + cell / 2 + 4}" text-anchor="end">{esc(shorten(row, 20))}</text>')
        for col_index, col in enumerate(cols):
            x = left + col_index * cell
            value = matrix.get((row, col), 0)
            intensity = value / max_value if max_value else 0
            color = f"rgb({int(239 - 155 * intensity)},{int(244 - 106 * intensity)},{int(247 - 59 * intensity)})"
            body.append(f'<rect x="{x}" y="{y}" width="{cell - 3}" height="{cell - 3}" rx="3" fill="{color}" stroke="#d8dee6"/>')
            body.append(f'<text class="value" x="{x + (cell - 3) / 2}" y="{y + cell / 2 + 4}" text-anchor="middle">{value}</text>')
    write_svg(out_dir, name, width, height, title, body)


def bar_figure(out_dir: Path, name: str, title: str, subtitle: str, series: list[tuple[str, Counter[str]]], order: list[str], colors: dict[str, str] | None = None) -> None:
    width, left, top, row_height = 900, 210, 108, 62
    height = top + row_height * len(series) + 55
    maximum = max((sum(counter.values()) for _, counter in series), default=1)
    scale = 610 / maximum
    body = [f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">{esc(title)}</text>', f'<text class="sub" x="{width / 2}" y="51" text-anchor="middle">{esc(subtitle)}</text>']
    for index, (label, counts) in enumerate(series):
        y = top + index * row_height
        body.append(f'<text class="axis" x="{left - 12}" y="{y + 22}" text-anchor="end">{esc(shorten(label, 28))}</text>')
        x = left
        for order_index, category in enumerate(order):
            count = counts.get(category, 0)
            if not count:
                continue
            w = count * scale
            color = (colors or {}).get(category, PALETTE[order_index % len(PALETTE)])
            body.append(f'<rect x="{x}" y="{y}" width="{w}" height="28" fill="{color}"/>')
            if w >= 28:
                body.append(f'<text class="small" x="{x + w / 2}" y="{y + 18}" text-anchor="middle" fill="#fff">{count}</text>')
            x += w
        body.append(f'<text class="value" x="{left + 620}" y="{y + 20}">{sum(counts.values())}</text>')
    legend_x, legend_y = left, height - 23
    for index, category in enumerate(order):
        x = legend_x + index * max(92, int(630 / max(1, len(order))))
        color = (colors or {}).get(category, PALETTE[index % len(PALETTE)])
        body.append(f'<rect x="{x}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        body.append(f'<text class="small" x="{x + 14}" y="{legend_y}">{esc(shorten(category, 15))}</text>')
    write_svg(out_dir, name, width, height, title, body)


def outlier_figure(out_dir: Path, name: str, label: str, rows: list[dict[str, str]]) -> None:
    shown = rows[:16]
    width, height = 1220, 110 + max(1, len(shown)) * 38
    body = [f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">{esc(label)}: notable keyword/LLM disagreements</text>', '<text class="sub" x="610" y="51" text-anchor="middle">Ordered by disagreement type and lowest LLM confidence; full details are in classifier_outliers.csv.</text>']
    headers = [(20, "Repository"), (230, "Keyword"), (350, "LLM"), (470, "Family change"), (650, "Method"), (810, "Reason")]
    for x, text in headers:
        body.append(f'<text class="axis" x="{x}" y="77">{esc(text)}</text>')
    for index, row in enumerate(shown):
        y = 100 + index * 38
        if index % 2 == 0:
            body.append(f'<rect x="15" y="{y - 15}" width="1188" height="32" fill="#f5f7f9"/>')
        body.append(f'<text class="small" x="20" y="{y}">{esc(shorten(row["full_name"], 31))}</text>')
        body.append(f'<text class="small" x="230" y="{y}">{esc(shorten(row["keyword_decision"], 14))}</text>')
        body.append(f'<text class="small" x="350" y="{y}">{esc(shorten(row["llm_decision"], 14))}</text>')
        body.append(f'<text class="small" x="470" y="{y}">{esc(shorten(row["keyword_family"] + " → " + row["llm_family"], 26))}</text>')
        body.append(f'<text class="small" x="650" y="{y}">{esc(shorten(row["llm_methods"], 22))}</text>')
        body.append(f'<text class="small" x="810" y="{y}">{esc(shorten(row["reason"], 58))}</text>')
    write_svg(out_dir, name, width, height, f"{label} outliers", body)


def stable_jitter(value: str, amplitude: float) -> float:
    """Stable point separation: reruns retain the same repository placement."""
    fraction = int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return (fraction - 0.5) * 2 * amplitude


def llm_confidence(row: dict[str, str]) -> float:
    values = [as_float(row.get(name), 0.0) for name in ("role_confidence", "family_confidence")]
    if row.get("llm_scope") in {"yes", "no"}:
        values.append(as_float(row.get("scope_gate_confidence"), 0.0))
    usable = [value for value in values if value > 0]
    return min(usable) if usable else 0.5


def overlap_kind(keyword: dict[str, str], llm: dict[str, str]) -> str:
    if keyword["keyword_decision"] == "accept" and llm["llm_decision"] == "accept":
        return "both_accept"
    if keyword["keyword_decision"] == "exclude" and llm["llm_decision"] == "exclude":
        return "both_exclude"
    if llm["llm_decision"] == "accept":
        return "llm_only_accept"
    if keyword["keyword_decision"] == "accept":
        return "keyword_only_accept"
    return "uncertain_or_error"


def repository_overlap_figure(out_dir: Path, name: str, label: str, keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]], matched: list[str]) -> None:
    """Plot every repository, preserving the two classifiers' actual decisions.

    Horizontal position is the keyword score.  Vertical position is the signed
    LLM confidence: accepted rows appear above zero; excluded rows below;
    uncertain/error rows cluster around zero.  The representation makes every
    disagreement visible while retaining the keyword score information.
    """
    width, height = 1160, 710
    left, right, top, bottom = 90, 80, 92, 92
    plot_width, plot_height = width - left - right, height - top - bottom
    scores = [as_float(keyword[name].get("keyword_score"), 0.0) for name in matched]
    extent = max(10.0, math.ceil(max((abs(score) for score in scores), default=10.0) / 10) * 10)
    def project_x(score: float) -> float:
        return left + ((score + extent) / (2 * extent)) * plot_width
    def project_y(value: float) -> float:
        return top + ((1.12 - value) / 2.24) * plot_height

    title = f"Repository-level keyword/LLM overlap map — {label}"
    body = [f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">{esc(title)}</text>', '<text class="sub" x="580" y="51" text-anchor="middle">One dot per matched repository. Above: LLM accepts; below: LLM excludes; centre: uncertain/error. Hover points for repository details.</text>']
    for tick in (-1, -0.5, 0, 0.5, 1):
        y = project_y(tick)
        body.append(f'<line class="grid" x1="{left}" y1="{y}" x2="{width - right}" y2="{y}"/>')
        label_text = {1: "LLM accept (high confidence)", 0.5: "LLM accept", 0: "LLM uncertain / error", -0.5: "LLM exclude", -1: "LLM exclude (high confidence)"}[tick]
        body.append(f'<text class="axis" x="{left - 8}" y="{y + 4}" text-anchor="end">{esc(label_text)}</text>')
    for tick in range(-int(extent), int(extent) + 1, max(10, int(extent // 4) or 10)):
        x = project_x(tick)
        body.append(f'<line class="grid" x1="{x}" y1="{top}" x2="{x}" y2="{height - bottom}"/>')
        body.append(f'<text class="axis" x="{x}" y="{height - bottom + 20}" text-anchor="middle">{tick}</text>')
    body.append(f'<text class="axis" x="{width / 2}" y="{height - 28}" text-anchor="middle">Keyword heuristic score (higher = stronger keyword evidence)</text>')

    labels: list[tuple[float, float, str, str]] = []
    for repo_name in matched:
        keyword_row, llm_row = keyword[repo_name], llm[repo_name]
        confidence = llm_confidence(llm_row)
        if llm_row["llm_decision"] == "accept":
            vertical = confidence
        elif llm_row["llm_decision"] == "exclude":
            vertical = -confidence
        else:
            vertical = stable_jitter(repo_name + "-vertical", 0.12)
        x = project_x(as_float(keyword_row.get("keyword_score"), 0.0) + stable_jitter(repo_name + "-horizontal", min(1.8, extent / 25)))
        y = project_y(vertical + stable_jitter(repo_name + "-vertical-small", 0.035))
        kind = overlap_kind(keyword_row, llm_row)
        tooltip = f"{keyword_row['full_name']} | keyword: {keyword_row['keyword_decision']} (score {keyword_row['keyword_score']}, {keyword_row['keyword_family']}) | LLM: {llm_row['llm_decision']} ({llm_row['llm_role']}, {llm_row['llm_family']}, confidence {confidence:.2f})"
        body.append(f'<g><title>{esc(tooltip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="{OVERLAP_COLORS[kind]}" fill-opacity="0.84" stroke="#fff" stroke-width="0.8"/></g>')
        if kind in {"llm_only_accept", "keyword_only_accept"}:
            labels.append((x, y, keyword_row["full_name"], kind))
    for index, (x, y, repo, kind) in enumerate(sorted(labels, key=lambda item: item[2])[:12]):
        offset = -9 if index % 2 else 14
        body.append(f'<text class="small" x="{x + 7:.1f}" y="{y + offset:.1f}" fill="{OVERLAP_COLORS[kind]}">{esc(shorten(repo, 24))}</text>')
    legend_x = left + 5
    for index, (kind, text) in enumerate((
        ("both_accept", "accepted by both"), ("both_exclude", "excluded by both"),
        ("llm_only_accept", "LLM-only accept"), ("keyword_only_accept", "keyword-only accept"),
        ("uncertain_or_error", "LLM uncertain/error"),
    )):
        x = legend_x + index * 180
        body.append(f'<circle cx="{x}" cy="70" r="5" fill="{OVERLAP_COLORS[kind]}"/><text class="small" x="{x + 9}" y="74">{esc(text)}</text>')
    write_svg(out_dir, name, width, height, title, body)


def set_overlap_figure(out_dir: Path, name: str, label: str, keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]], matched: list[str]) -> None:
    """A direct no-axis overlap view: one dot per repository in its set."""
    groups: dict[str, list[str]] = {kind: [] for kind in OVERLAP_COLORS}
    for repo_name in matched:
        groups[overlap_kind(keyword[repo_name], llm[repo_name])].append(repo_name)
    layout = {
        "both_accept": (20, 92, 360, 250, "Accepted by both"),
        "keyword_only_accept": (400, 92, 360, 250, "Keyword-only accept"),
        "llm_only_accept": (780, 92, 360, 250, "LLM-only accept"),
        "both_exclude": (210, 375, 360, 250, "Excluded by both"),
        "uncertain_or_error": (590, 375, 360, 250, "LLM uncertain / error"),
    }
    width, height = 1160, 660
    title = f"Repository-set overlap — {label}"
    body = [f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">{esc(title)}</text>', '<text class="sub" x="580" y="51" text-anchor="middle">Every dot is one repository. Its card shows exactly how the keyword and LLM approaches treated it; no artificial numeric axes.</text>']
    for kind, (x, y, card_width, card_height, heading) in layout.items():
        names = sorted(groups[kind], key=lambda value: keyword[value]["full_name"].casefold())
        color = OVERLAP_COLORS[kind]
        body.append(f'<rect x="{x}" y="{y}" width="{card_width}" height="{card_height}" rx="8" fill="{color}" fill-opacity="0.08" stroke="{color}" stroke-opacity="0.55"/>')
        body.append(f'<text class="value" x="{x + 14}" y="{y + 24}" fill="{color}">{esc(heading)} ({len(names)})</text>')
        for index, repo_name in enumerate(names):
            cols = 16
            dot_x = x + 18 + (index % cols) * 20
            dot_y = y + 48 + (index // cols) * 20
            if dot_y > y + card_height - 14:
                break
            keyword_row, llm_row = keyword[repo_name], llm[repo_name]
            tooltip = f"{keyword_row['full_name']} | keyword: {keyword_row['keyword_decision']} ({keyword_row['keyword_family']}, score {keyword_row['keyword_score']}) | LLM: {llm_row['llm_decision']} ({llm_row['llm_role']}, {llm_row['llm_family']})"
            body.append(f'<g><title>{esc(tooltip)}</title><circle cx="{dot_x}" cy="{dot_y}" r="6.1" fill="{color}" fill-opacity="0.88" stroke="#fff" stroke-width="1"/></g>')
            if kind in {"keyword_only_accept", "llm_only_accept"} and index < 8:
                body.append(f'<text class="small" x="{dot_x + 8}" y="{dot_y + 3}">{esc(shorten(keyword_row["full_name"], 20))}</text>')
    write_svg(out_dir, name, width, height, title, body)


def repository_profile_coordinates(keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]], matched: list[str]):
    """PCA of repository text and classifier labels; returns two coordinates per repo."""
    if np is None or len(matched) < 3:
        return None
    documents = []
    for name in matched:
        text = " ".join((keyword[name].get("full_name", ""), llm[name].get("description", ""), llm[name].get("topics", ""))).casefold()
        documents.append([token for token in TOKEN_RE.findall(text) if token not in STOPWORDS])
    document_frequency = Counter(token for tokens in documents for token in set(tokens))
    vocabulary = [token for token, count in document_frequency.most_common(500) if count >= 2]
    if not vocabulary:
        vocabulary = [token for tokens in documents for token in tokens][:100]
    vocab_index = {token: index for index, token in enumerate(dict.fromkeys(vocabulary))}
    text_features = np.zeros((len(matched), len(vocab_index)), dtype=float)
    for row_index, tokens in enumerate(documents):
        counts = Counter(tokens)
        for token, count in counts.items():
            if token in vocab_index:
                idf = math.log((1 + len(matched)) / (1 + document_frequency[token])) + 1
                text_features[row_index, vocab_index[token]] = (1 + math.log(count)) * idf
    label_sets: list[list[str]] = []
    for name in matched:
        keyword_row, llm_row = keyword[name], llm[name]
        label_sets.append([
            f"keyword decision:{keyword_row['keyword_decision']}", f"keyword family:{keyword_row['keyword_family']}", f"keyword role:{keyword_row['keyword_role']}",
            f"llm decision:{llm_row['llm_decision']}", f"llm family:{llm_row['llm_family']}", f"llm role:{llm_row['llm_role']}",
            *[f"llm method:{method}" for method in split_values(llm_row['llm_methods'])],
        ])
    label_vocab = {label: index for index, label in enumerate(sorted({label for labels in label_sets for label in labels}))}
    label_features = np.zeros((len(matched), len(label_vocab)), dtype=float)
    for row_index, labels in enumerate(label_sets):
        for value in labels:
            label_features[row_index, label_vocab[value]] = 1.0
    def normalize_rows(values):
        lengths = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(lengths, 1e-12)
    features = np.hstack((normalize_rows(text_features) * 0.75, normalize_rows(label_features) * 0.60))
    features -= features.mean(axis=0, keepdims=True)
    try:
        coordinates = np.linalg.svd(features, full_matrices=False)[0][:, :2]
    except np.linalg.LinAlgError:
        return None
    if coordinates.shape[1] < 2:
        coordinates = np.pad(coordinates, ((0, 0), (0, 2 - coordinates.shape[1])))
    return coordinates


def repository_profile_map(out_dir: Path, name: str, label: str, keyword: dict[str, dict[str, str]], llm: dict[str, dict[str, str]], matched: list[str]) -> None:
    coordinates = repository_profile_coordinates(keyword, llm, matched)
    title = f"2D repository-profile map — {label}"
    if coordinates is None:
        body = ['<text class="title" x="500" y="45" text-anchor="middle">2D repository-profile map unavailable</text>', '<text class="sub" x="500" y="75" text-anchor="middle">Install NumPy and provide at least three matched repositories to create this PCA map.</text>']
        write_svg(out_dir, name, 1000, 140, title, body)
        return
    width, height, left, top, right, bottom = 1160, 710, 80, 86, 70, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    max_x, max_y = max(0.05, float(np.max(np.abs(coordinates[:, 0])))), max(0.05, float(np.max(np.abs(coordinates[:, 1]))))
    def px(value: float) -> float: return left + (value / max_x + 1) * 0.5 * plot_width
    def py(value: float) -> float: return top + (1 - (value / max_y + 1) * 0.5) * plot_height
    body = [f'<text class="title" x="{width / 2}" y="31" text-anchor="middle">{esc(title)}</text>', '<text class="sub" x="580" y="51" text-anchor="middle">PCA of repository description/topics plus keyword and LLM classifications. Nearby points have similar profiles; axes have no standalone semantic meaning.</text>']
    body.extend((f'<line class="grid" x1="{left}" y1="{py(0)}" x2="{width - right}" y2="{py(0)}"/>', f'<line class="grid" x1="{px(0)}" y1="{top}" x2="{px(0)}" y2="{height - bottom}"/>'))
    body.append(f'<text class="axis" x="{width / 2}" y="{height - 23}" text-anchor="middle">PCA dimension 1 — relative repository-profile similarity</text>')
    body.append(f'<text class="axis" x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle">PCA dimension 2 — relative repository-profile similarity</text>')
    labels = []
    for index, repo_name in enumerate(matched):
        keyword_row, llm_row = keyword[repo_name], llm[repo_name]
        x, y = px(float(coordinates[index, 0])), py(float(coordinates[index, 1]))
        kind = overlap_kind(keyword_row, llm_row)
        tooltip = f"{keyword_row['full_name']} | keyword: {keyword_row['keyword_decision']} ({keyword_row['keyword_family']}) | LLM: {llm_row['llm_decision']} ({llm_row['llm_role']}, {llm_row['llm_family']})"
        body.append(f'<g><title>{esc(tooltip)}</title><circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{OVERLAP_COLORS[kind]}" fill-opacity="0.84" stroke="#fff" stroke-width="0.9"/></g>')
        if kind in {"keyword_only_accept", "llm_only_accept"}:
            labels.append((x, y, keyword_row["full_name"], kind))
    for index, (x, y, repo, kind) in enumerate(sorted(labels, key=lambda item: item[2])[:14]):
        body.append(f'<text class="small" x="{x + 8:.1f}" y="{y + (-8 if index % 2 else 14):.1f}" fill="{OVERLAP_COLORS[kind]}">{esc(shorten(repo, 24))}</text>')
    write_svg(out_dir, name, width, height, title, body)


def comparison_type(keyword: dict[str, str], llm: dict[str, str]) -> str:
    if keyword["keyword_decision"] != llm["llm_decision"]:
        return "decision mismatch"
    if keyword["keyword_decision"] == "accept" and llm["llm_decision"] == "accept" and keyword["keyword_family"] != llm["llm_family"]:
        return "family mismatch"
    if llm["llm_decision"] == "uncertain":
        return "LLM uncertain"
    return "agreement"


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pairwise_figures(out_dir: Path, models: list[tuple[str, dict[str, dict[str, str]]]]) -> list[dict[str, str]]:
    labels = [label for label, _ in models]
    accept_matrix: dict[tuple[str, str], int] = {}
    decision_matrix: dict[tuple[str, str], int] = {}
    summary: list[dict[str, str]] = []
    for left_label, left_rows in models:
        for right_label, right_rows in models:
            shared = set(left_rows) & set(right_rows)
            left_accept = {name for name in shared if left_rows[name]["llm_decision"] == "accept"}
            right_accept = {name for name in shared if right_rows[name]["llm_decision"] == "accept"}
            union = left_accept | right_accept
            jaccard = len(left_accept & right_accept) / len(union) if union else 1.0
            agreement = sum(left_rows[name]["llm_decision"] == right_rows[name]["llm_decision"] for name in shared) / len(shared) if shared else 0.0
            accept_matrix[(left_label, right_label)] = round(jaccard * 100)
            decision_matrix[(left_label, right_label)] = round(agreement * 100)
            summary.append({"left_model": left_label, "right_model": right_label, "shared_repositories": str(len(shared)), "accepted_set_jaccard_pct": f"{jaccard * 100:.1f}", "decision_agreement_pct": f"{agreement * 100:.1f}"})
    matrix_figure(out_dir, "07_model_accept_jaccard.svg", "LLM-model accepted-set overlap", "Cell = Jaccard similarity (%) of accepted repositories on shared input rows.", accept_matrix, labels, labels)
    matrix_figure(out_dir, "08_model_decision_agreement.svg", "LLM-model decision agreement", "Cell = identical accept/exclude/uncertain/error decision (%) on shared input rows.", decision_matrix, labels, labels)
    return summary


def main() -> int:
    configure_csv_limit()
    parser = argparse.ArgumentParser(description="Create SVG comparison figures for keyword and LLM GitHub repository classifiers.")
    parser.add_argument("--keyword-evaluated", type=Path, default=DEFAULT_KEYWORD_EVALUATED)
    parser.add_argument("--keyword-accepted", type=Path, default=DEFAULT_KEYWORD_ACCEPTED)
    parser.add_argument("--llm", action="append", default=[], metavar="LABEL=CSV", help="LLM accepted CSV for set comparison; repeat for multiple models. A classified audit CSV is also accepted for full decision comparison.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not args.keyword_evaluated.is_file():
        parser.error(f"keyword evaluated cache not found: {args.keyword_evaluated}")
    llm_inputs = [parse_llm_argument(value, index) for index, value in enumerate(args.llm, 1)]
    if not llm_inputs and DEFAULT_LLM.is_file():
        llm_inputs = [("default", DEFAULT_LLM)]
    if not llm_inputs:
        parser.error("provide at least one --llm LABEL=CSV (no default LLM file was found).")
    labels = [label for label, _ in llm_inputs]
    if len(labels) != len(set(labels)):
        parser.error("each --llm label must be unique.")
    for label, path in llm_inputs:
        if not path.is_file():
            parser.error(f"LLM CSV for {label!r} not found: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    keyword = load_keyword(args.keyword_evaluated, args.keyword_accepted if args.keyword_accepted.is_file() else None)
    models = [(label, load_llm(path)) for label, path in llm_inputs]
    all_outliers: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    expanded_models: list[tuple[str, dict[str, dict[str, str]]]] = []
    for label, llm_raw in models:
        llm, matched, accepted_only = comparison_view(keyword, llm_raw)
        expanded_models.append((label, llm))
        decisions = Counter(llm[name]["llm_decision"] for name in matched)
        keyword_decisions = Counter(keyword[name]["keyword_decision"] for name in matched)
        coverage = (f"Accepted-set comparison: {len(llm_raw):,} LLM accepted rows versus {sum(row['keyword_decision'] == 'accept' for row in keyword.values()):,} keyword accepted rows. " if accepted_only else "") + f"{len(matched):,} compared repositories; keyword audit contains {len(keyword):,}."
        safe_label = slug(label)
        bar_figure(args.out_dir, f"00_decision_overview_{safe_label}.svg", f"Keyword vs LLM decision overview — {label}", coverage, [("Keyword", keyword_decisions), ("LLM", decisions)], ["accept", "exclude", "not_accepted", "uncertain", "error"], DECISION_COLORS)

        decision_matrix = Counter((keyword[name]["keyword_decision"], llm[name]["llm_decision"]) for name in matched)
        matrix_subtitle = "Rows: keyword crawler. Columns: LLM classifier. `not accepted` means absent from an accepted-only LLM file, not a confirmed exclusion." if accepted_only else "Rows: keyword crawler. Columns: LLM classifier. Values are repository counts."
        matrix_figure(args.out_dir, f"01_decision_matrix_{safe_label}.svg", f"Decision comparison — {label}", matrix_subtitle, dict(decision_matrix), ["accept", "exclude"], ["accept", "exclude", "not_accepted", "uncertain", "error"])
        set_overlap_figure(args.out_dir, f"01b_repository_overlap_map_{safe_label}.svg", label, keyword, llm, matched)
        repository_profile_map(args.out_dir, f"01c_repository_profile_pca_{safe_label}.svg", label, keyword, llm, matched)

        accepted_both = [name for name in matched if keyword[name]["keyword_decision"] == "accept" and llm[name]["llm_decision"] == "accept"]
        families = Counter((keyword[name]["keyword_family"], llm[name]["llm_family"]) for name in accepted_both)
        family_rows = ordered((left for left, _ in families), FAMILY_ORDER)
        family_cols = ordered((right for _, right in families), FAMILY_ORDER)
        matrix_figure(args.out_dir, f"02_family_matrix_{safe_label}.svg", f"Accepted-family comparison — {label}", f"Only repositories accepted by both approaches ({len(accepted_both):,}).", dict(families), family_rows or ["none"], family_cols or ["none"])

        role_matrix = Counter((keyword[name]["keyword_role"], llm[name]["llm_role"]) for name in matched)
        role_rows = ordered((left for left, _ in role_matrix), ROLE_ORDER)
        role_cols = ordered((right for _, right in role_matrix), ROLE_ORDER)
        matrix_figure(args.out_dir, f"03_role_matrix_{safe_label}.svg", f"Role comparison — {label}", "Keyword role is recovered from its audit score card; LLM role is its semantic judgement.", dict(role_matrix), role_rows, role_cols)

        methods = Counter(method for name in matched for method in split_values(llm[name]["llm_methods"]))
        bar_figure(args.out_dir, f"04_detection_methods_{safe_label}.svg", f"LLM detection-method classifications — {label}", "Methods are model-decided; multi-label totals may exceed repository count.", [(label, methods)], ordered(methods, []), None)

        outliers: list[dict[str, str]] = []
        for name in matched:
            row = {"model_label": label, **keyword[name], **llm[name]}
            row["comparison_type"] = comparison_type(keyword[name], llm[name])
            if row["comparison_type"] != "agreement":
                outliers.append(row)
        outliers.sort(key=lambda row: (row["comparison_type"] == "LLM uncertain", as_float(row["role_confidence"], 1.0), row["full_name"]))
        all_outliers.extend(outliers)
        outlier_figure(args.out_dir, f"05_outliers_{safe_label}.svg", label, outliers)
        summary_rows.append({"model_label": label, "model_name": next((row["llm_model"] for row in llm.values() if row["llm_model"]), ""), "llm_rows": str(len(llm)), "matched_keyword_rows": str(len(matched)), "keyword_accepts_in_matched": str(keyword_decisions["accept"]), "llm_accepts": str(decisions["accept"]), "llm_excludes": str(decisions["exclude"]), "llm_uncertain": str(decisions["uncertain"]), "llm_errors": str(decisions["error"]), "decision_mismatches": str(sum(row["comparison_type"] == "decision mismatch" for row in outliers)), "family_mismatches": str(sum(row["comparison_type"] == "family mismatch" for row in outliers))})

    if len(expanded_models) >= 2:
        pairwise_rows = pairwise_figures(args.out_dir, expanded_models)
        write_csv(args.out_dir / "model_pairwise_agreement.csv", pairwise_rows, ["left_model", "right_model", "shared_repositories", "accepted_set_jaccard_pct", "decision_agreement_pct"])

    outlier_columns = ["model_label", "comparison_type", "full_name", "html_url", "keyword_decision", "keyword_family", "keyword_role", "keyword_score", "llm_model", "llm_decision", "llm_scope", "scope_gate_confidence", "llm_role", "llm_family", "llm_methods", "role_confidence", "family_confidence", "method_confidence", "evidence_quote_verified", "reason"]
    write_csv(args.out_dir / "classifier_outliers.csv", all_outliers, outlier_columns)
    write_csv(args.out_dir / "comparison_summary.csv", summary_rows, list(summary_rows[0]) if summary_rows else ["model_label"])
    readme = ["# Keyword vs LLM classifier comparison", "", "Generated by `compare_keyword_llm_classifiers.py`.", "", "## Inputs", "", f"- Keyword evaluated cache: `{args.keyword_evaluated}`", *[f"- LLM run `{label}`: `{path}`" for label, path in llm_inputs], "", "## Interpretation", "", "- The default input is `ids_repos.llm_accepted.csv`, the comparison-oriented accepted output. Missing rows are labelled `not_accepted`; that means absent from this accepted list, not necessarily confirmed excluded by an incomplete LLM run.", "- `01b_repository_overlap_map_*.svg` is the direct set-overlap view: every dot is one repository in the exact keyword-only, LLM-only, both-accept, both-exclude, or LLM-uncertain set. It deliberately has no numeric axes.", "- `01c_repository_profile_pca_*.svg` reduces repository descriptions/topics and both classifiers' labels to two PCA dimensions. Nearby points have similar overall profiles; the PCA axes do not have a standalone semantic meaning.", "- To compare all evaluated decisions rather than accepted sets, explicitly pass `ids_repos.llm_classified.csv` with `--llm`.", "- Family matrices use only repositories accepted by both approaches.", "- `classifier_outliers.csv` is the audit table for decision mismatches, family mismatches, and LLM uncertainty.", "- Method labels are LLM outputs, not ground truth. Compare model runs with the pairwise figures when more than one `--llm` input is supplied.", ""]
    (args.out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Saved comparison figures and audit tables to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
