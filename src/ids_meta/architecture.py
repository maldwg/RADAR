from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

WIDTH, HEIGHT = 1900, 1470


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
        '<desc id="description">Data-flow architecture for OpenAlex and GitHub discovery, keyword, zero-shot NLI, LLM and ensemble classification, rated evaluation, reconciliation, plotting, and export.</desc>',
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
        _text(48, 72, "Discovery feeds comparable classifier families; rated evaluation and catalogue reporting remain separate outputs.", "subtitle"),
        _section(40, 105, 1260, 400, "OpenAlex discovery", "Direct queries and survey references form one deduplicated paper pool before evidence-based screening.", "#f5f9fc"),
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
        _section(1335, 105, 525, 1305, "Classification and catalogue reporting", "Comparable classifiers; only LLM and Ensemble feed catalogues.", "#faf9f5"),
        _data_card(1370, 165, 455, "Candidate evidence boundary", ["candidate CSV + stable source ID", "cached metadata and README / abstract evidence", "source taxonomy and classifier configuration"]),
        _card(1370, 335, 455, "Comparable classifier runners", "candidate evidence + source taxonomy", "keyword rules | zero-shot NLI | repeated LLM", "method predictions + saved score / vote audits", "#fff2de"),
        _card(1370, 500, 455, "Normalize, vote, and ensemble", "method predictions + repeated LLM runs", "normalize all; vote LLM runs; combine models", "per-approach decision + taxonomy rows", "#fff2de"),
        _card(1370, 665, 455, "Validation and audit", "classifier rows + deterministic field rules", "normalize; check coherence; record evidence", "validated rows + reasons + saved NLI scores", "#fff2de"),
        _data_card(1370, 830, 455, "Classifier result families", ["classifiers/keyword", "classifiers/nli/<model>", "classifiers/llm/<model> and classifiers/ensemble"]),
        _card(1370, 1015, 455, "Reconcile production results", "validated LLM / ensemble classifications", "group releases or forks; rank canonical member", "reconciled families + accepted rows", "#eeeaf7"),
        _card(1370, 1180, 455, "Catalogue plotter and export", "reconciled families + accepted catalogues", "build descriptive role, domain, and timeline views", "processed CSV + artifacts/plots/<source>", "#e9eef8"),
        _section(40, 760, 1260, 650, "Rated classification evaluation", "Offline scoring compares Keyword, NLI, every LLM, and Ensemble on the same versioned source benchmark.", "#f8f6fb"),
        _data_card(65, 835, 320, "Central rating profiles", ["GitHub: stratified positive + negative repos", "OpenAlex: 105 positive reference datasets", "expected decision + rated taxonomy fields"]),
        _data_card(65, 1050, 320, "Comparable classifier inputs", ["Keyword", "zero-shot NLI", "each LLM model", "Ensemble"]),
        _card(445, 845, 365, "Result evaluator", "ratings + classifier rows + candidates", "resolve fixtures; score operational / strict / selective", "confusions + coverage + taxonomy metrics", "#eeeaf7"),
        _card(445, 1030, 365, "NLI sensitivity replay", "saved NLI scores + configured cutoff grid", "replay score / winner-gap thresholds offline", "six-panel robustness heatmap + exact CSV", "#e9eef8"),
        _card(865, 845, 390, "Cross-approach plotter", "all compatible evaluator outputs", "align metrics, taxonomy, and agreement", "overview + drill-down comparisons", "#e9eef8"),
        _data_card(865, 1030, 390, "artifacts/evaluation/<source>", ["comparison/{metrics,taxonomy,agreement}", "classifiers/{keyword,nli,llm,ensemble}", "per-classifier figures + auditable CSVs", "OpenAlex unsupported binary metrics = N/A"]),
        _flow([(1265, 317), (1320, 317), (1320, 220), (1370, 220)]),
        _flow([(1265, 649), (1320, 649), (1320, 260), (1370, 260)]),
        _flow([(1598, 307), (1598, 335)], "candidate evidence", (1598, 327)),
        _flow([(1598, 453), (1598, 500)], "method predictions", (1598, 484)),
        _flow([(1598, 618), (1598, 665)], "all classifier rows", (1598, 649)),
        _flow([(1598, 783), (1598, 830)], "validated variants", (1598, 814)),
        _flow([(1370, 902), (1320, 902), (1320, 1245), (810, 1245), (810, 904)], "classifier + candidate rows", (1080, 1240)),
        _flow([(385, 904), (445, 904)], "human ratings", (415, 899)),
        _flow([(385, 1140), (415, 1140), (415, 1089), (445, 1089)], "saved NLI result", (415, 1128)),
        _flow([(810, 904), (865, 904)], "rated metrics", (838, 899)),
        _flow([(1058, 963), (1058, 1030)], "comparison bundle", (1058, 1002)),
        _flow([(810, 1089), (835, 1089), (835, 1170), (865, 1170)], "sensitivity", (835, 1145)),
        _flow([(1598, 972), (1598, 1015)], "production variants only", (1598, 1002)),
        _flow([(1598, 1133), (1598, 1180)], "reconciled catalogue", (1598, 1164)),
        _text(48, 1442, "Solid cards transform data  •  Dashed cards persist or transfer data  •  x = per-query sample size  •  y = retained collection size", "legend"),
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
