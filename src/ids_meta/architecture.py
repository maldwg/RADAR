from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

WIDTH, HEIGHT = 1800, 1240


def _text(x: float, y: float, value: str, css: str = "body", anchor: str = "start") -> str:
    """Return one escaped SVG text element."""
    return f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">{escape(value)}</text>'


def _card(x: float, y: float, width: float, title: str, input_data: str, operation: str, output_data: str, color: str) -> str:
    """Render a process card naming its input, operation, and output data."""
    return "\n".join((
        f'<rect x="{x}" y="{y}" width="{width}" height="118" rx="12" class="card" fill="{color}"/>',
        _text(x + 14, y + 25, title, "card-title"),
        _text(x + 14, y + 51, "INPUT", "tag"), _text(x + 68, y + 51, input_data, "card-text"),
        _text(x + 14, y + 76, "DO", "tag"), _text(x + 68, y + 76, operation, "card-text"),
        _text(x + 14, y + 101, "OUTPUT", "tag"), _text(x + 68, y + 101, output_data, "card-text"),
    ))


def _data_card(x: float, y: float, width: float, title: str, lines: list[str]) -> str:
    """Render a dashed boundary for persisted or transferred records."""
    height = 82 + 20 * len(lines)
    parts = [f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="12" class="data-card"/>', _text(x + width / 2, y + 26, title, "data-title", "middle")]
    parts.extend(_text(x + 18, y + 53 + index * 20, f"• {line}", "data-text") for index, line in enumerate(lines))
    return "\n".join(parts)


def _flow(points: list[tuple[float, float]], label: str = "", label_at: tuple[float, float] | None = None) -> str:
    """Render an orthogonal labelled data-flow connector."""
    coordinates = " ".join(f"{x},{y}" for x, y in points)
    parts = [f'<polyline points="{coordinates}" class="flow" marker-end="url(#arrow)"/>']
    if label and label_at:
        x, y = label_at
        width = max(90, len(label) * 6.2 + 18)
        parts.extend((f'<rect x="{x - width / 2}" y="{y - 14}" width="{width}" height="20" rx="7" class="flow-label-bg"/>', _text(x, y + 1, label, "flow-label", "middle")))
    return "\n".join(parts)


def _section(x: float, y: float, width: float, height: float, title: str, subtitle: str, color: str) -> str:
    """Render one labelled source or shared-processing section."""
    return "\n".join((f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{color}" class="section"/>', _text(x + 20, y + 29, title, "section-title"), _text(x + 20, y + 50, subtitle, "section-subtitle")))


def _header() -> str:
    """Return SVG metadata, marker definitions, and shared styles."""
    return "\n".join((
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title description">',
        '<title id="title">IDS meta-study pipeline architecture</title>',
        '<desc id="description">Data-flow architecture for OpenAlex and GitHub discovery, classification consensus, validation, rated evaluation, reconciliation, and export.</desc>',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526979"/></marker>',
        '<style>',
        'text { font-family: Inter, Segoe UI, Arial, sans-serif; fill: #253746; }',
        '.title { font-size: 27px; font-weight: 700; } .subtitle { font-size: 14px; fill: #60717d; }',
        '.section { stroke: #c4d0d8; stroke-width: 1.4; } .section-title { font-size: 18px; font-weight: 700; } .section-subtitle { font-size: 12px; fill: #647682; }',
        '.card { stroke: #607786; stroke-width: 1.35; } .card-title { font-size: 14px; font-weight: 700; } .card-text { font-size: 11.5px; }',
        '.tag { font-size: 9px; font-weight: 700; fill: #5b7180; letter-spacing: .4px; }',
        '.data-card { fill: #fffaf0; stroke: #b78a3c; stroke-width: 1.5; stroke-dasharray: 7 5; } .data-title { font-size: 14px; font-weight: 700; fill: #76551f; } .data-text { font-size: 11.5px; fill: #5b4a2d; }',
        '.flow { fill: none; stroke: #526979; stroke-width: 1.8; stroke-linejoin: round; } .flow-label-bg { fill: #ffffff; stroke: #d5dde2; stroke-width: .8; } .flow-label { font-size: 10.5px; fill: #526979; }',
        '.legend { font-size: 11px; fill: #60717d; }',
        '</style></defs>',
    ))


def render_svg() -> str:
    """Build the complete architecture as a native dependency-free SVG."""
    items = [
        _header(),
        _text(48, 45, "IDS meta-study pipeline architecture", "title"),
        _text(48, 72, "Components transform data; connectors name transferred records; dashed boxes mark persistent data boundaries.", "subtitle"),
        _section(40, 105, 1260, 400, "OpenAlex discovery", "Publisher-diverse survey and paper seeds drive convergent backward snowballing (configurable depth ≤ 10).", "#f5f9fc"),
        _text(65, 166, "DIRECT WORK BRANCH", "tag"),
        _card(65, 180, 245, "Direct query sampler", "direct query strings", "request top x per query", "raw work metadata", "#e8f1f7"),
        _card(365, 180, 265, "Direct eligibility + rank", "raw work metadata", "citation floor, score, dedupe", "ranked direct work pool", "#e8f1f7"),
        _text(65, 324, "SURVEY / REFERENCE BRANCH", "tag"),
        _card(65, 338, 245, "Diverse seed selector", "queries + publisher targets", "select surveys + top start papers", "survey/paper seed frontier", "#e8f1f7"),
        _card(365, 338, 265, "Backward snowball traversal", "frontier reference IDs", "batch fetch; score; repeat to convergence", "relevant papers + auxiliary surveys", "#e8f1f7"),
        _card(700, 258, 270, "OpenAlex candidate selector", "direct + reference work pools", "reserve share, dedupe, select y", "active work IDs + cached evidence", "#dcebf5"),
        _card(1020, 258, 245, "OpenAlex prefilter", "active cached work evidence", "score and assign candidate band", "paper_candidates.csv", "#dcebf5"),
        _flow([(310, 239), (365, 239)]),
        _flow([(310, 397), (365, 397)]),
        _flow([(630, 239), (665, 239), (665, 300), (700, 300)]),
        _flow([(630, 397), (665, 397), (665, 334), (700, 334)]),
        _flow([(970, 317), (1020, 317)]),
        _section(40, 530, 1260, 200, "GitHub discovery", "Native best-match results are selected before repository evidence is enriched and scored.", "#f5faf6"),
        _card(65, 590, 245, "Repository query sampler", "queries + minimum-star floor", "request native top x/query", "repository stubs + query ranks", "#e8f4eb"),
        _card(365, 590, 245, "GitHub candidate selector", "stubs from every query", "dedupe, rank, select best y", "active repository IDs", "#e8f4eb"),
        _card(665, 590, 275, "Evidence enricher + cache", "active repository IDs", "fetch metadata, README, docs, commit", "cached repository evidence", "#dff0e3"),
        _card(995, 590, 270, "GitHub prefilter", "active cached repo evidence", "score and assign candidate band", "candidates.csv", "#dff0e3"),
        _flow([(310, 649), (365, 649)]),
        _flow([(610, 649), (665, 649)]),
        _flow([(940, 649), (995, 649)]),
        _section(1335, 105, 425, 1085, "Shared classification, reconciliation, and export", "Both sources cross the same auditable data boundaries.", "#faf9f5"),
        _data_card(1380, 165, 335, "Candidate evidence boundary", ["candidate CSV row", "stable cache key", "metadata / README / abstract sections"]),
        _card(1380, 345, 335, "Repeated LLM classification", "candidate row + evidence sections", "vote; enrich + restart unresolved round", "consensus + evidence-escalation audit", "#fff2de"),
        _card(1380, 525, 335, "Decision + field consensus", "all normalized run rows", "threshold-vote scalar + multi-label fields", "voted tuple + complete vote audit", "#fff2de"),
        _card(1380, 705, 335, "Cross-field validator", "voted tuple + taxonomy rules", "check coherence; use audited fallback", "validated or uncertain classification", "#fff2de"),
        _card(1380, 885, 335, "Reconcile + export", "validated classification rows", "group families and select canonical base", "accepted catalogues + audit rows", "#eeeaf7"),
        _data_card(1380, 1050, 335, "Persisted outputs", ["interim classifications / reconciled CSV", "processed OpenAlex and GitHub catalogues"]),
        _section(40, 760, 1260, 430, "Rated classification evaluation", "A separate offline branch resolves readable source selectors and compares results with versioned human ratings.", "#f8f6fb"),
        _data_card(65, 835, 300, "Central rating profiles", ["config/evaluation/<source>.toml", "expected include / exclude decision", "expected normalized field values"]),
        _card(430, 845, 350, "Result evaluator", "ratings + classifications + candidates", "resolve readable selectors; score three metric views", "confusions + coverage + field scores", "#eeeaf7"),
        _data_card(850, 825, 410, "Evaluation artifacts per model / ensemble", ["metrics.json: DR, FPR, ACC, PREC, F1", "instances + confusion + field metric CSVs", "4 SVG/PDF figures + exact plot data"]),
        _flow([(1265, 317), (1325, 317), (1325, 215), (1380, 215)]),
        _flow([(1265, 649), (1325, 649), (1325, 255), (1380, 255)]),
        _flow([(1548, 307), (1548, 345)], "candidate row + evidence", (1548, 330)),
        _flow([(1548, 463), (1548, 525)], "normalized run rows", (1548, 498)),
        _flow([(1548, 643), (1548, 705)], "voted tuple + field audit", (1548, 678)),
        _flow([(1548, 823), (1548, 885)], "validated classification", (1548, 858)),
        _flow([(1380, 775), (1320, 775), (1320, 904), (780, 904)], "classification + candidate rows", (1095, 899)),
        _flow([(365, 904), (430, 904)], "human ratings", (397, 899)),
        _flow([(780, 904), (850, 904)], "auditable scores", (815, 899)),
        _flow([(1548, 1003), (1548, 1050)], "families + accepted rows", (1548, 1029)),
        _text(48, 1218, "Solid cards transform data  •  Dashed cards persist or transfer data  •  x = per-query sample size  •  y = selected collection size", "legend"),
        "</svg>",
    ]
    return "\n".join(items)


def build(root: Path, formats: list[str] | None = None, dpi: int = 160) -> list[Path]:
    """Write the combined architecture SVG directly in the chosen repository root."""
    del dpi
    requested = formats or ["svg"]
    unsupported = [value for value in requested if value != "svg"]
    if unsupported:
        raise ValueError("architecture output currently supports SVG only: " + ", ".join(unsupported))
    root.mkdir(parents=True, exist_ok=True)
    output = root / "pipeline_architecture.svg"
    output.write_text(render_svg(), encoding="utf-8")
    return [output]


def main(argv: list[str] | None = None) -> int:
    """Generate the architecture SVG independently from catalogue plotting."""
    parser = argparse.ArgumentParser(description="Build the combined IDS meta-study architecture SVG")
    parser.add_argument("--output", type=Path, default=Path.cwd(), help="output directory; defaults to the current repository root")
    args = parser.parse_args(argv)
    build(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
