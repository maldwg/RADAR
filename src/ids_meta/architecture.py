from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, OffsetImage
from matplotlib.patches import (
    Arc, Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle,
)
from matplotlib.path import Path as MplPath
from matplotlib.text import Text


INK = "#24323A"
MUTED = "#5D6A71"
EDGE = "#3F515B"
GUIDE = "#CBD3D7"
OPENALEX = "#EDF3F7"
GITHUB = "#EDF4EE"
CLASSIFY = "#F7F1E5"
EVALUATE = "#F1EEF5"
REPORT = "#F0F2F2"
WHITE = "#FFFFFF"
EVALUATION_EDGE = "#65547E"
CATALOGUE_EDGE = "#345F74"
BRAND_ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "architecture"


def _card_label(
    ax: Axes, bounds: tuple[float, float, float, float], label: str, *,
    fontsize: float, color: str = INK, fontweight: str = "bold",
) -> Text:
    """Wrap a card label to its padded text area using rendered font widths."""
    x, y, width, height = bounds
    artist = ax.text(
        x + width / 2, y + height / 2, label, ha="center", va="center",
        fontsize=fontsize, color=color, fontweight=fontweight,
        linespacing=1.12, zorder=5,
    )
    artist.set_gid("architecture-card-label:" + ":".join(str(value) for value in bounds))
    renderer = ax.figure.canvas.get_renderer()
    lower = ax.transData.transform((x, y))
    upper = ax.transData.transform((x + width, y + height))
    available_width, available_height = upper - lower
    while True:
        lines = []
        for paragraph in label.split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}".strip()
                measured_width = renderer.get_text_width_height_descent(
                    candidate, artist.get_fontproperties(), False,
                )[0]
                if line and measured_width > available_width:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        artist.set_text("\n".join(lines))
        extent = artist.get_window_extent(renderer)
        if extent.width <= available_width and extent.height <= available_height:
            return artist
        if artist.get_fontsize() <= 5.5:
            raise ValueError(f"architecture label does not fit its padded card area: {label!r}")
        artist.set_fontsize(max(5.5, artist.get_fontsize() - .25))


def _line(area: DrawingArea, points: list[tuple[float, float]], color: str, linewidth: float = 1.15) -> None:
    """Draw one crisp polyline inside a vector pictogram."""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    area.add_artist(PathPatch(
        path, fill=False, edgecolor=color, linewidth=linewidth,
        capstyle="round", joinstyle="round",
    ))


def _icon_artist(kind: str, color: str = EDGE, size: float = 18) -> DrawingArea:
    """Return a small literal line pictogram that remains crisp in SVG/PDF."""
    area = DrawingArea(size, size, 0, 0)
    scale = size / 20.0

    def p(value: float) -> float:
        """Scale one pictogram coordinate to the requested icon size."""
        return value * scale

    def rect(
        x: float, y: float, width: float, height: float, *,
        fill: str = "none", radius: float = 0.0,
    ) -> None:
        """Add one optional-rounded rectangle to the pictogram."""
        if radius:
            patch = FancyBboxPatch(
                (p(x), p(y)), p(width), p(height),
                boxstyle=f"round,pad=0,rounding_size={p(radius)}",
                facecolor=fill, edgecolor=color, linewidth=1.1,
            )
        else:
            patch = Rectangle(
                (p(x), p(y)), p(width), p(height),
                facecolor=fill, edgecolor=color, linewidth=1.1,
            )
        area.add_artist(patch)

    def circle(x: float, y: float, radius: float, *, fill: str = WHITE) -> None:
        """Add one circle to the pictogram."""
        area.add_artist(Circle(
            (p(x), p(y)), p(radius),
            facecolor=fill, edgecolor=color, linewidth=1.1,
        ))

    def line(points: list[tuple[float, float]], linewidth: float = 1.1) -> None:
        """Add one scaled polyline to the pictogram."""
        _line(area, [(p(x), p(y)) for x, y in points], color, linewidth)

    if kind == "repository":
        path = MplPath(
            [
                (p(2), p(5)), (p(7), p(5)), (p(9), p(7)), (p(18), p(7)),
                (p(18), p(17)), (p(2), p(17)), (p(2), p(5)),
            ],
            [MplPath.MOVETO] + [MplPath.LINETO] * 5 + [MplPath.CLOSEPOLY],
        )
        area.add_artist(PathPatch(
            path, facecolor="none", edgecolor=color, linewidth=1.1,
            joinstyle="round",
        ))
        line([(6, 10), (6, 14), (13, 14)])
        circle(6, 10, 1.05)
        circle(6, 14, 1.05)
        circle(13, 14, 1.05)
    elif kind == "paper":
        rect(4, 4, 11, 13, fill=WHITE)
        line([(11, 17), (15, 13), (11, 13), (11, 17)])
        line([(6.5, 10.5), (12.5, 10.5)])
        line([(6.5, 7.5), (12.5, 7.5)])
        line([(2, 2), (13, 2), (13, 4)])
    elif kind == "search":
        circle(8, 11, 5, fill="none")
        line([(11.6, 7.4), (17, 2)])
        line([(5.5, 11), (10.5, 11)])
    elif kind == "rank":
        rect(1.5, 7, 6, 9, fill=WHITE)
        rect(4, 4, 6, 9, fill=WHITE)
        line([(5.5, 9), (7, 7.5), (9, 10.5)])
        line([(11.5, 14), (18, 14)])
        line([(11.5, 10), (16.5, 10)])
        line([(11.5, 6), (15, 6)])
        circle(10.5, 14, 0.55, fill=color)
        circle(10.5, 10, 0.55, fill=color)
        circle(10.5, 6, 0.55, fill=color)
    elif kind == "merge":
        line([(2, 15), (7, 15), (10, 11)])
        line([(2, 5), (7, 5), (10, 9)])
        line([(10, 10), (18, 10)])
        area.add_artist(Polygon(
            [(p(17), p(12)), (p(19), p(10)), (p(17), p(8))],
            closed=True, facecolor=color, edgecolor=color,
        ))
    elif kind == "documents":
        rect(2, 5, 10, 12, fill=WHITE)
        rect(7, 2, 10, 12, fill=WHITE)
        line([(9, 10), (15, 10)])
        line([(9, 7), (15, 7)])
    elif kind == "survey":
        line([
            (2, 15), (7, 16.5), (10, 14.5), (13, 16.5), (18, 15),
            (18, 4), (13, 5.5), (10, 3.5), (7, 5.5), (2, 4), (2, 15),
        ])
        line([(10, 14.5), (10, 3.5)])
    elif kind == "diverse":
        rect(2, 4, 4, 12, fill=WHITE, radius=0.5)
        rect(7, 6, 4, 10, fill=WHITE, radius=0.5)
        rect(12, 3, 5, 13, fill=WHITE, radius=0.5)
        line([(4, 6), (4, 14)])
        line([(9, 8), (9, 14)])
        line([(14.5, 5), (14.5, 14)])
        line([(1.5, 2.5), (18, 2.5)])
    elif kind == "snowball":
        circle(13, 7, 4.5, fill=WHITE)
        line([(13, 4.2), (13, 9.8)])
        line([(10.2, 7), (15.8, 7)])
        line([(11, 5), (15, 9)])
        line([(11, 9), (15, 5)])
        line([(2, 14), (7, 14)])
        line([(3.5, 10.5), (8, 10.5)])
        circle(5, 17, 0.7, fill=WHITE)
        circle(9, 15.5, 0.55, fill=WHITE)
    elif kind == "classifier":
        rect(5, 5, 10, 10, fill=WHITE, radius=1.0)
        for position in (7, 10, 13):
            line([(position, 2.5), (position, 5)])
            line([(position, 15), (position, 17.5)])
            line([(2.5, position), (5, position)])
            line([(15, position), (17.5, position)])
        line([(8, 12), (12, 12)])
        line([(8, 9), (12, 9)])
    elif kind == "key":
        circle(6, 12, 3.5, fill="none")
        line([(8.6, 9.4), (16.5, 2.5), (16.5, 6), (14, 6), (14, 8)])
    elif kind == "language":
        rect(2, 6, 12, 10, fill=WHITE, radius=1.5)
        line([(6, 6), (4, 3), (9, 6)])
        line([(5, 12), (11, 12)])
        line([(5, 9), (9, 9)])
        circle(16, 6, 1.5, fill=color)
    elif kind == "model":
        rect(4, 4, 12, 12, fill=WHITE, radius=1.2)
        circle(8, 11.5, 1.2, fill="none")
        circle(12, 11.5, 1.2, fill="none")
        line([(7, 7.5), (13, 7.5)])
        line([(10, 16), (10, 18)])
    elif kind == "ensemble":
        rect(2, 8, 9, 9, fill=WHITE)
        rect(5, 5, 9, 9, fill=WHITE)
        rect(8, 2, 9, 9, fill=WHITE)
        line([(10.5, 6), (12.5, 4), (15.5, 8)])
    elif kind == "people":
        circle(7, 13.5, 2.2, fill="none")
        circle(14, 13, 1.8, fill="none")
        area.add_artist(Arc(
            (p(7), p(6)), p(10), p(9), theta1=15, theta2=165,
            edgecolor=color, linewidth=1.1,
        ))
        area.add_artist(Arc(
            (p(14), p(6)), p(8), p(7), theta1=20, theta2=160,
            edgecolor=color, linewidth=1.1,
        ))
    elif kind == "checklist":
        rect(3, 2, 14, 15, fill=WHITE, radius=1.0)
        rect(7, 15, 6, 3, fill=WHITE, radius=0.8)
        line([(6, 11), (7.5, 9.5), (10, 13)])
        line([(11, 11), (15, 11)])
        line([(6, 6), (7.5, 4.5), (10, 8)])
        line([(11, 6), (15, 6)])
    elif kind == "chart":
        line([(3, 17), (3, 3), (18, 3)])
        rect(6, 3, 2.5, 5, fill="none")
        rect(10.5, 3, 2.5, 9, fill="none")
        rect(15, 3, 2.5, 13, fill="none")
    elif kind == "family":
        rect(2, 8, 7, 9, fill=WHITE)
        rect(11, 8, 7, 9, fill=WHITE)
        rect(6.5, 2, 7, 9, fill=WHITE)
        line([(6, 8), (8, 6.5)])
        line([(14, 8), (12, 6.5)])
    elif kind == "catalogue":
        rect(3, 3, 14, 14, fill=WHITE, radius=0.8)
        line([(7, 3), (7, 17)])
        line([(10, 13), (15, 13)])
        line([(10, 10), (15, 10)])
        line([(10, 7), (14, 7)])
    elif kind == "database":
        area.add_artist(Arc((p(10), p(15)), p(14), p(5), theta1=0, theta2=360, edgecolor=color, linewidth=1.1))
        area.add_artist(Arc((p(10), p(5)), p(14), p(5), theta1=180, theta2=360, edgecolor=color, linewidth=1.1))
        line([(3, 5), (3, 15)])
        line([(17, 5), (17, 15)])
        area.add_artist(Arc((p(10), p(10)), p(14), p(5), theta1=180, theta2=360, edgecolor=color, linewidth=1.1))
    elif kind == "web":
        for radius in (3.0, 5.8, 8.0):
            circle(10, 10, radius, fill="none")
        for endpoint in ((10, 1.5), (16, 4), (18.5, 10), (16, 16), (10, 18.5), (4, 16), (1.5, 10), (4, 4)):
            line([(10, 10), endpoint], linewidth=0.9)
        circle(10, 10, 0.7, fill=color)
    elif kind == "server":
        rect(3, 3, 14, 14, fill=WHITE, radius=0.8)
        line([(3, 8), (17, 8)])
        line([(3, 12), (17, 12)])
        circle(6, 5.5, 0.7, fill=color)
        circle(6, 10, 0.7, fill=color)
        circle(6, 14.5, 0.7, fill=color)
        line([(9, 5.5), (14.5, 5.5)])
        line([(9, 10), (14.5, 10)])
        line([(9, 14.5), (14.5, 14.5)])
    else:
        rect(3, 3, 14, 14, fill=WHITE)
    return area


def _add_icon(
    ax: Axes, x: float, y: float, kind: str, *,
    color: str = EDGE, size: float = 18,
) -> None:
    """Place a centered vector pictogram on the architecture axes."""
    ax.add_artist(AnnotationBbox(
        _icon_artist(kind, color, size),
        (x, y), frameon=False, box_alignment=(0.5, 0.5), zorder=5,
    ))


def _read_brand_image(path: Path) -> np.ndarray:
    """Decode raster lockups and render genuine SVG lockups to an RGBA array."""
    raw = path.read_bytes()
    prefix = raw.lstrip()[:512].lower()
    if prefix.startswith(b"<?xml") or b"<svg" in prefix:
        try:
            from cairosvg import svg2png
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "SVG architecture assets require CairoSVG; install the project dependencies"
            ) from exc
        raw = svg2png(bytestring=raw, output_width=512)
        image_format = "png"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        image_format = "png"
    elif raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        image_format = "webp"
    elif raw.startswith(b"\xff\xd8\xff"):
        image_format = "jpeg"
    else:
        image_format = path.suffix.lower().lstrip(".") or None
    return imread(BytesIO(raw), format=image_format)


def _trim_brand_image(path: Path) -> np.ndarray:
    """Load an official lockup and remove only transparent/white canvas margins."""
    image = _read_brand_image(path)
    if image.ndim != 3:
        return image
    alpha = image[..., 3] if image.shape[2] >= 4 else None
    if alpha is not None and np.any(alpha < 0.98):
        mask = alpha > 0.02
    else:
        mask = np.any(image[..., :3] < 0.97, axis=2)
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return image
    (y0, x0), (y1, x1) = coordinates.min(axis=0), coordinates.max(axis=0)
    padding = 2
    y0, x0 = max(0, y0 - padding), max(0, x0 - padding)
    y1 = min(image.shape[0], y1 + padding + 1)
    x1 = min(image.shape[1], x1 + padding + 1)
    cropped = image[y0:y1, x0:x1]
    if alpha is None or not np.any(alpha < 0.98):
        transparent = np.ones((*cropped.shape[:2], 4), dtype=cropped.dtype)
        transparent[..., :3] = cropped[..., :3]
        white_canvas = np.all(cropped[..., :3] > 0.97, axis=2)
        transparent[..., 3] = np.where(white_canvas, 0.0, 1.0)
        return transparent
    return cropped


def _add_brand(
    ax: Axes,
    x: float,
    y: float,
    filename: str,
    *,
    target_width: float = 40,
) -> None:
    """Place one unmodified official lockup at a controlled print width."""
    image = _trim_brand_image(BRAND_ASSET_DIR / filename)
    zoom = target_width / max(1, image.shape[1])
    ax.add_artist(AnnotationBbox(
        OffsetImage(image, zoom=zoom, interpolation="antialiased"),
        (x, y),
        frameon=False,
        box_alignment=(0.5, 0.5),
        zorder=5,
    ))


def _brand_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    filename: str,
    *,
    facecolor: str,
    target_width: float,
    label: str = "",
) -> None:
    """Draw a source card around an official GitHub or OpenAlex lockup."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        linewidth=0.9,
        edgecolor=EDGE,
        facecolor=facecolor,
        zorder=3,
    ))
    _add_brand(
        ax, x + width / 2, y + height * (0.60 if label else 0.50),
        filename, target_width=target_width,
    )
    if label:
        _card_label(
            ax, (x + width * .08, y + height * .08, width * .84, height * .32),
            label, color=MUTED, fontsize=6.6,
        )


def _logo_box(
    ax: Axes, x: float, y: float, width: float, height: float,
    filename: str, title: str, *, facecolor: str, target_width: float,
    edgecolor: str = EDGE, title_size: float = 6.6,
) -> None:
    """Draw a process card with an unmodified technology logo and text label."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        linewidth=0.9, edgecolor=edgecolor, facecolor=facecolor, zorder=3,
    ))
    _add_brand(
        ax, x + width / 2, y + height * (0.74 if title else 0.50), filename,
        target_width=target_width,
    )
    _card_label(
        ax, (x + width * .08, y + height * .08, width * .84, height * .43),
        title, fontsize=title_size,
    )


def _model_badge(
    ax: Axes, x: float, y: float, width: float, height: float,
    filename: str, label: str, *, target_width: float,
) -> None:
    """Draw one compact, labelled local-model badge inside the inference group."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.002,rounding_size=0.004",
        linewidth=0.7, edgecolor="#C9C1B2", facecolor=WHITE, zorder=4,
    ))
    _add_brand(
        ax, x + width / 2, y + height * 0.67, filename,
        target_width=target_width,
    )
    _card_label(
        ax, (x + width * .10, y + height * .08, width * .80, height * .28),
        label, fontsize=5.5,
    )


def _box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    *,
    icon: str,
    facecolor: str = WHITE,
    edgecolor: str = EDGE,
    linestyle: str = "-",
    title_size: float = 6.8,
    icon_size: float = 17,
    secondary_logo: str = "",
    secondary_logo_width: float = 11,
) -> None:
    """Draw a process card with a pictogram and optional technology mark."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        linewidth=0.9,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linestyle=linestyle,
        zorder=3,
    ))
    icon_y = y + height * 0.76
    icon_x = x + width * (0.40 if secondary_logo else 0.50)
    _add_icon(
        ax, icon_x, icon_y, icon,
        color=edgecolor, size=min(icon_size, 15),
    )
    if secondary_logo:
        _add_brand(
            ax, x + width * 0.62, icon_y, secondary_logo,
            target_width=secondary_logo_width,
        )
    _card_label(
        ax, (x + width * .08, y + height * .08, width * .84, height * .43),
        title, fontsize=title_size,
    )


def _arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = EDGE,
    linestyle: str = "-",
    linewidth: float = 1.25,
) -> None:
    """Draw a high-contrast connector whose head stops before the target border."""
    ax.add_patch(FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8.0,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        shrinkA=0,
        shrinkB=0,
        capstyle="round",
        joinstyle="round",
        zorder=7,
    ))


def _elbow_arrow(
    ax: Axes,
    points: list[tuple[float, float]],
    *,
    color: str = EDGE,
    linestyle: str = "-",
    linewidth: float = 1.25,
) -> None:
    """Draw an orthogonal routed connector with a visible arrowhead."""
    if len(points) < 2:
        return
    if len(points) > 2:
        path = MplPath(
            points[:-1],
            [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 2),
        )
        ax.add_patch(PathPatch(
            path, fill=False, edgecolor=color, linewidth=linewidth,
            linestyle=linestyle, capstyle="round", joinstyle="round", zorder=6,
        ))
    _arrow(
        ax, points[-2], points[-1],
        color=color, linestyle=linestyle, linewidth=linewidth,
    )


def _phase(ax: Axes, x0: float, x1: float, label: str) -> None:
    """Add an editorial phase heading and rule aligned to its column group."""
    y = 0.905
    ax.text(
        x0, y, label, ha="left", va="center",
        color=MUTED, fontsize=6.3, fontweight="bold",
    )
    ax.plot(
        [x0, x1], [y - 0.022, y - 0.022],
        color=GUIDE, linewidth=0.8, zorder=1,
    )


def _classifier_box(ax: Axes) -> dict[str, float]:
    """Draw the shared classifier family with identifiable methods."""
    x, y, width, height = 0.660, 0.145, 0.140, 0.715
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        linewidth=1.0, edgecolor=EDGE, facecolor=CLASSIFY, zorder=3,
    ))
    _add_icon(ax, x + width / 2, y + height - 0.035, "classifier", size=15)
    ax.text(
        x + width / 2, y + height - 0.082, "Classification",
        ha="center", va="center", color=INK,
        fontsize=7.2, fontweight="bold", zorder=5,
    )

    labels = (
        ("key", "Keyword rules"),
        ("language", "Zero-shot NLI"),
        ("model", "Repeated LLMs"),
        ("ensemble", "LLM ensemble"),
    )
    centers = (0.695, 0.555, 0.415, 0.275)
    for (icon, label), center_y in zip(labels, centers, strict=True):
        ax.add_patch(FancyBboxPatch(
            (x + 0.011, center_y - 0.050), width - 0.022, 0.100,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor=WHITE, edgecolor="#C9C1B2", linewidth=0.65, zorder=4,
        ))
        _add_icon(ax, x + 0.035, center_y, icon, size=14)
        _card_label(
            ax, (x + .059, center_y - .039, width - .080, .078),
            label, fontsize=6.4, fontweight="normal",
        )
    return {
        "left": x, "right": x + width, "keyword": centers[0],
        "nli": centers[1], "llm": centers[2], "ensemble": centers[3],
    }


def _branch_label(
    ax: Axes, x: float, y: float, label: str, color: str,
) -> None:
    """Place a small label over a branching connector."""
    ax.text(
        x, y, label, ha="center", va="center",
        fontsize=5.6, color=color, fontweight="bold",
        bbox={
            "boxstyle": "square,pad=0.12",
            "facecolor": WHITE,
            "edgecolor": "none",
        },
        zorder=8,
    )


def _color_legend(
    ax: Axes,
    items: tuple[tuple[str, str, str], ...],
    *,
    x: float,
    y: float,
    item_width: float,
) -> None:
    """Draw a compact fill-color legend using the same borders as the cards."""
    for index, (label, facecolor, edgecolor) in enumerate(items):
        left = x + index * item_width
        ax.add_patch(FancyBboxPatch(
            (left, y - 0.07), 0.22, 0.14,
            boxstyle="round,pad=0.002,rounding_size=0.015",
            linewidth=0.8, edgecolor=edgecolor, facecolor=facecolor, zorder=4,
        ))
        ax.text(
            left + 0.30, y, label, ha="left", va="center",
            color=MUTED, fontsize=6.2, zorder=5,
        )


def _draw_pipeline_figure() -> Figure:
    """Create the detailed five-stage pipeline and evaluation architecture."""
    figure, ax = plt.subplots(figsize=(14.6, 5.8))
    figure.subplots_adjust(left=0.008, right=0.992, bottom=0.020, top=0.992)
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    stage_headings = (
        (0.15, 6.10, "(i)  Source-specific discovery", GITHUB),
        (6.45, 7.95, "(ii)  Deterministic prefilter", CLASSIFY),
        (8.20, 9.90, "(iii)  LLM-based adjudication", CLASSIFY),
        (10.10, 13.75, "Evaluation path", EVALUATE),
    )
    for left, right, label, color in stage_headings:
        center = (left + right) / 2
        ax.text(
            center, 5.88, label, ha="center", va="center",
            color=INK, fontsize=8.0, fontweight="bold",
        )
        ax.plot(
            [left, right], [5.68, 5.68], color=color, linewidth=3.0,
            solid_capstyle="round", zorder=1,
        )
    # Discovery contains one lane per source; show both source colors.
    ax.plot([3.125, 6.10], [5.68, 5.68], color=OPENALEX, linewidth=3.0, solid_capstyle="round", zorder=2)

    github_y = 4.96
    openalex_source_y = 2.56
    direct_y = 3.41
    survey_y = 1.69
    card_height = 0.82
    ax.plot([0.14, 6.08], [github_y, github_y], color="#DCE9DF", linewidth=0.8, zorder=0)
    ax.plot([0.14, 6.08], [openalex_source_y, openalex_source_y], color="#DCE7EF", linewidth=0.8, zorder=0)

    _brand_box(
        ax, 0.15, github_y - card_height / 2, 1.10, card_height,
        "github-lockup-black.svg", facecolor=GITHUB, target_width=28,
    )
    _box(
        ax, 1.60, github_y - card_height / 2, 1.20, card_height,
        "Repository search", icon="search", facecolor=GITHUB, title_size=7.0,
    )
    _box(
        ax, 3.15, github_y - card_height / 2, 1.25, card_height,
        "Deduplicate & rank", icon="rank", facecolor=GITHUB, title_size=6.8,
    )
    _box(
        ax, 4.75, github_y - card_height / 2, 1.30, card_height,
        "README + docs", icon="documents", facecolor=GITHUB, title_size=7.0,
    )
    _arrow(ax, (1.33, github_y), (1.52, github_y), linewidth=1.45)
    _arrow(ax, (2.88, github_y), (3.07, github_y), linewidth=1.45)
    _arrow(ax, (4.48, github_y), (4.67, github_y), linewidth=1.45)

    _brand_box(
        ax, 0.15, openalex_source_y - card_height / 2, 1.10, card_height,
        "openalex-lockup-black.svg", facecolor=OPENALEX, target_width=44,
    )
    _box(
        ax, 1.60, direct_y - card_height / 2, 1.20, card_height,
        "Direct paper search", icon="search", facecolor=OPENALEX, title_size=6.7,
    )
    _box(
        ax, 1.60, survey_y - card_height / 2, 1.20, card_height,
        "Survey search", icon="survey", facecolor=OPENALEX, title_size=7.0,
    )
    _box(
        ax, 3.15, survey_y - card_height / 2, 1.25, card_height,
        "Diverse survey seeds", icon="diverse", facecolor=OPENALEX, title_size=6.4,
    )
    _box(
        ax, 4.75, survey_y - card_height / 2, 1.25, card_height,
        "Backward snowballing", icon="snowball", facecolor=OPENALEX, title_size=6.3,
    )
    _box(
        ax, 4.725, direct_y - 0.46, 1.30, 0.92,
        "Merge, dedupe & rank", icon="merge", facecolor=OPENALEX, title_size=6.5,
    )
    _elbow_arrow(
        ax, [(1.33, openalex_source_y), (1.33, direct_y), (1.52, direct_y)],
        linewidth=1.45,
    )
    _elbow_arrow(
        ax, [(1.33, openalex_source_y), (1.33, survey_y), (1.52, survey_y)],
        linewidth=1.45,
    )
    _arrow(ax, (2.88, direct_y), (4.645, direct_y), linewidth=1.45)
    _arrow(ax, (2.88, survey_y), (3.07, survey_y), linewidth=1.45)
    _arrow(ax, (4.48, survey_y), (4.67, survey_y), linewidth=1.45)
    _arrow(ax, (5.375, 2.18), (5.375, 2.87), linewidth=1.45)

    _box(
        ax, 6.55, 2.885, 1.30, 1.05,
        "Deterministic prefilter", icon="rank", facecolor=CLASSIFY,
        title_size=6.6, icon_size=18,
    )
    ax.plot(
        [6.13, 6.30, 6.30], [github_y, github_y, direct_y],
        color=EDGE, linewidth=1.50, solid_capstyle="round", solid_joinstyle="round", zorder=6,
    )
    _arrow(ax, (6.105, direct_y), (6.47, direct_y), linewidth=1.50)

    classifier_x, classifier_y = 8.20, 1.00
    classifier_width, classifier_height = 1.65, 4.20
    ax.add_patch(FancyBboxPatch(
        (classifier_x, classifier_y), classifier_width, classifier_height,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        linewidth=1.0, edgecolor=EDGE, facecolor=CLASSIFY, zorder=3,
    ))
    _add_icon(ax, 9.05, 4.92, "classifier", size=16)
    ax.text(
        9.05, 4.62, "Classification", ha="center", va="center",
        color=INK, fontsize=8.0, fontweight="bold", zorder=5,
    )
    classifier_rows = (
        (3.95, "Structured prompts", "language", EDGE),
        (3.10, "Repeated LLM runs", "model", EDGE),
        (2.25, "Consensus decision", "ensemble", EDGE),
    )
    for center_y, label, icon, edge in classifier_rows:
        ax.add_patch(FancyBboxPatch(
            (8.35, center_y - 0.29), 1.35, 0.58,
            boxstyle="round,pad=0.002,rounding_size=0.004",
            facecolor=WHITE, edgecolor=edge, linewidth=0.75, zorder=4,
        ))
        _add_icon(ax, 8.56, center_y, icon, color=edge, size=13)
        _card_label(
            ax, (8.80, center_y - .22, .80, .44), label, fontsize=6.4,
        )
    _arrow(ax, (7.93, direct_y), (8.12, direct_y), linewidth=1.50)

    _box(
        ax, 10.15, 4.85, 1.50, 0.75,
        "Human-labelled reference", icon="people", facecolor=WHITE,
        edgecolor=EVALUATION_EDGE, linestyle="--", title_size=6.2, icon_size=15,
    )
    _box(
        ax, 12.15, 4.85, 1.50, 0.75,
        "Keyword + NLI baselines", icon="key", facecolor=EVALUATE,
        edgecolor=EVALUATION_EDGE, title_size=6.2, icon_size=15,
    )
    _box(
        ax, 10.15, 3.55, 1.50, 0.85,
        "Compare all classifiers", icon="checklist", facecolor=EVALUATE,
        edgecolor=EVALUATION_EDGE, title_size=6.2, icon_size=16,
    )
    _box(
        ax, 12.15, 3.55, 1.50, 0.85,
        "Metrics & figures", icon="chart", facecolor=EVALUATE,
        edgecolor=EVALUATION_EDGE, title_size=6.6, icon_size=16,
    )
    _arrow(ax, (9.93, 3.975), (10.07, 3.975), color=EVALUATION_EDGE, linewidth=1.50)
    _arrow(ax, (10.90, 4.77), (10.90, 4.48), color=EVALUATION_EDGE, linestyle="--", linewidth=1.35)
    _elbow_arrow(
        ax, [(12.90, 4.77), (12.90, 4.70), (11.45, 4.70), (11.45, 4.48)],
        color=EVALUATION_EDGE, linewidth=1.35,
    )
    _arrow(ax, (11.73, 3.975), (12.07, 3.975), color=EVALUATION_EDGE, linewidth=1.50)

    output_groups = (
        (10.15, "Reconciliation", "Repository families", "Publication families", "family"),
        (12.15, "Final catalogues", "Repository catalogue", "Dataset catalogue", "catalogue"),
    )
    for x, title, github_label, openalex_label, icon in output_groups:
        ax.add_patch(FancyBboxPatch(
            (x, 0.50), 1.50, 1.90,
            boxstyle="round,pad=0.004,rounding_size=0.007",
            linewidth=1.0, edgecolor=CATALOGUE_EDGE, facecolor=REPORT, zorder=3,
        ))
        ax.text(
            x + 0.75, 2.16, title, ha="center", va="center",
            color=INK, fontsize=7.1, fontweight="bold", zorder=5,
        )
        for center_y, label, fill in (
            (1.58, github_label, GITHUB), (0.88, openalex_label, OPENALEX),
        ):
            ax.add_patch(FancyBboxPatch(
                (x + 0.10, center_y - 0.275), 1.30, 0.55,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=fill, edgecolor=CATALOGUE_EDGE, linewidth=0.70, zorder=4,
            ))
            _add_icon(ax, x + 0.27, center_y, icon, color=CATALOGUE_EDGE, size=13)
            _card_label(
                ax, (x + .47, center_y - .20, .83, .40), label, fontsize=5.5,
            )

    ax.text(
        10.90, 2.72, "(iv)  Source-specific reconciliation",
        ha="center", va="center", color=INK, fontsize=7.0, fontweight="bold",
    )
    ax.text(
        12.90, 2.72, "(v)  Catalogue export",
        ha="center", va="center", color=INK, fontsize=7.0, fontweight="bold",
    )
    for left, middle, right in ((10.15, 10.90, 11.65), (12.15, 12.90, 13.65)):
        ax.plot([left, middle], [2.53, 2.53], color=GITHUB, linewidth=3.0, solid_capstyle="round")
        ax.plot([middle, right], [2.53, 2.53], color=OPENALEX, linewidth=3.0, solid_capstyle="round")
    _arrow(ax, (9.93, 1.40), (10.07, 1.40), color=CATALOGUE_EDGE, linewidth=1.50)
    _arrow(ax, (11.73, 1.40), (12.07, 1.40), color=CATALOGUE_EDGE, linewidth=1.50)

    _color_legend(
        ax,
        (
            ("GitHub-specific", GITHUB, EDGE),
            ("OpenAlex-specific", OPENALEX, EDGE),
            ("Shared processing", CLASSIFY, EDGE),
            ("Evaluation only", EVALUATE, EVALUATION_EDGE),
            ("Reporting container", REPORT, CATALOGUE_EDGE),
        ),
        x=0.35, y=0.20, item_width=2.65,
    )

    return figure


def _draw_system_figure() -> Figure:
    """Create the high-level, grid-aligned pipeline architecture."""
    figure, ax = plt.subplots(figsize=(11.2, 4.8))
    figure.subplots_adjust(left=0.008, right=0.992, bottom=0.025, top=0.990)
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 13.8)
    ax.set_ylim(-0.40, 5.25)
    ax.axis("off")
    headings = (
        (2.72, "Source adapters", GITHUB, 2.04, 3.40),
        (4.90, "Evidence store", REPORT, 4.22, 5.58),
        (6.92, "Prefilter", CLASSIFY, 6.24, 7.60),
        (8.92, "Inference", CLASSIFY, 8.24, 9.60),
        (11.90, "Outputs", REPORT, 10.27, 13.60),
    )
    for center, label, color, left, right in headings:
        ax.text(
            center, 4.65, label, ha="center", va="center",
            fontsize=7.7, color=INK, fontweight="bold",
        )
        ax.plot(
            [left, right], [4.48, 4.48],
            color=color if color != WHITE else GUIDE, linewidth=3.0,
            solid_capstyle="round", zorder=1,
        )

    github_y = 3.85
    openalex_y = 2.45
    card_height = 0.84
    ax.plot([0.14, 13.63], [github_y, github_y], color="#DCE9DF", linewidth=0.8, zorder=0)
    ax.plot([0.14, 13.63], [openalex_y, openalex_y], color="#DCE7EF", linewidth=0.8, zorder=0)

    _brand_box(
        ax, 0.15, github_y - card_height / 2, 1.20, card_height,
        "github-lockup-black.svg", facecolor=GITHUB,
        target_width=24, label="",
    )
    _brand_box(
        ax, 0.15, openalex_y - card_height / 2, 1.20, card_height,
        "openalex-lockup-black.svg", facecolor=OPENALEX,
        target_width=45, label="",
    )

    _box(
        ax, 1.85, github_y - card_height / 2, 1.75, card_height,
        "GitHub crawler", icon="web", facecolor=GITHUB,
        title_size=7.2, icon_size=18,
        secondary_logo="python.svg", secondary_logo_width=11,
    )
    _box(
        ax, 1.85, openalex_y - card_height / 2, 1.75, card_height,
        "OpenAlex crawler", icon="web", facecolor=OPENALEX,
        title_size=7.2, icon_size=18,
        secondary_logo="python.svg", secondary_logo_width=11,
    )

    _logo_box(
        ax, 4.20, 2.72, 1.40, 0.90,
        "sqlite.svg", "Evidence + LLM results",
        facecolor=REPORT, target_width=42, title_size=5.8,
    )
    _logo_box(
        ax, 4.20, 1.32, 1.40, 0.88,
        "python.svg", "Evidence enrichment",
        facecolor=CLASSIFY, target_width=17, title_size=5.9,
    )
    _box(
        ax, 6.15, 2.72, 1.55, 0.90,
        "Deterministic prefilter", icon="rank", facecolor=CLASSIFY,
        title_size=6.7, icon_size=18,
        secondary_logo="python.svg", secondary_logo_width=11,
    )
    _logo_box(
        ax, 8.20, 2.72, 1.45, 0.90,
        "python.svg", "Adjudication client",
        facecolor=CLASSIFY, target_width=18, title_size=6.1,
    )

    ax.add_patch(FancyBboxPatch(
        (7.80, 0.30), 2.24, 1.90,
        boxstyle="round,pad=0.004,rounding_size=0.007",
        linewidth=0.9, edgecolor=EDGE, facecolor=CLASSIFY, zorder=3,
    ))
    _logo_box(
        ax, 8.12, 1.40, 1.60, 0.62,
        "ollama.svg", "Ollama server",
        facecolor=WHITE, target_width=11, title_size=6.4,
    )
    ax.text(
        8.92, 1.21, "Local model families", ha="center", va="center",
        color=MUTED, fontsize=6.0, fontweight="bold", zorder=5,
    )
    _model_badge(ax, 7.94, 0.49, 0.60, 0.58, "qwen.svg", "Qwen", target_width=20)
    _model_badge(ax, 8.62, 0.49, 0.60, 0.58, "openai.svg", "gpt-oss", target_width=12)
    _model_badge(ax, 9.30, 0.49, 0.60, 0.58, "gemma.png", "Gemma", target_width=13)

    _box(
        ax, 10.15, github_y - card_height / 2, 1.60, card_height,
        "Repository families", icon="family", facecolor=GITHUB,
        edgecolor=CATALOGUE_EDGE, title_size=6.8, icon_size=17,
    )
    _box(
        ax, 10.15, openalex_y - card_height / 2, 1.60, card_height,
        "Publication families", icon="family", facecolor=OPENALEX,
        edgecolor=CATALOGUE_EDGE, title_size=6.8, icon_size=17,
    )
    _box(
        ax, 12.20, github_y - card_height / 2, 1.45, card_height,
        "Repository catalogue", icon="catalogue", facecolor=GITHUB,
        edgecolor=CATALOGUE_EDGE, title_size=6.4, icon_size=17,
    )
    _box(
        ax, 12.20, openalex_y - card_height / 2, 1.45, card_height,
        "Dataset catalogue", icon="catalogue", facecolor=OPENALEX,
        edgecolor=CATALOGUE_EDGE, title_size=6.7, icon_size=17,
    )

    _arrow(ax, (1.77, github_y), (1.43, github_y), linewidth=1.45)
    _arrow(ax, (1.77, openalex_y), (1.43, openalex_y), linewidth=1.45)
    _elbow_arrow(
        ax, [(3.68, github_y), (3.90, github_y), (3.90, 3.33), (4.12, 3.33)],
        linewidth=1.45,
    )
    _elbow_arrow(
        ax, [(3.68, openalex_y), (3.90, openalex_y), (3.90, 2.97), (4.12, 2.97)],
        linewidth=1.45,
    )
    _arrow(ax, (5.68, 3.17), (6.07, 3.17), linewidth=1.45)
    _arrow(ax, (7.78, 3.17), (8.12, 3.17), linewidth=1.45)
    _arrow(ax, (4.90, 2.28), (4.90, 2.64), linewidth=1.20)
    _elbow_arrow(
        ax, [(6.925, 2.64), (6.925, 1.76), (5.68, 1.76)],
        linewidth=1.20,
    )
    ax.plot(
        [8.40, 8.40], [2.64, 2.42], color=EDGE, linewidth=1.20,
        solid_capstyle="round", zorder=6,
    )
    ax.plot(
        [6.925, 8.40], [2.42, 2.42], color=EDGE, linewidth=1.20,
        solid_capstyle="round", zorder=6,
    )
    _elbow_arrow(
        ax, [(8.92, 3.70), (8.92, 4.14), (4.90, 4.14), (4.90, 3.70)],
        linewidth=1.35,
    )
    _elbow_arrow(
        ax, [(9.73, 3.17), (9.90, 3.17), (9.90, github_y), (10.07, github_y)],
        linewidth=1.45,
    )
    _elbow_arrow(
        ax, [(9.73, 3.17), (9.90, 3.17), (9.90, openalex_y), (10.07, openalex_y)],
        linewidth=1.45,
    )
    _arrow(ax, (11.83, github_y), (12.12, github_y), color=CATALOGUE_EDGE, linewidth=1.50)
    _arrow(ax, (11.83, openalex_y), (12.12, openalex_y), color=CATALOGUE_EDGE, linewidth=1.50)
    _arrow(ax, (8.92, 2.64), (8.92, 2.28), color=EVALUATION_EDGE, linewidth=1.45)

    _color_legend(
        ax,
        (
            ("GitHub path", GITHUB, EDGE),
            ("OpenAlex path", OPENALEX, EDGE),
            ("Shared state", REPORT, EDGE),
            ("Shared processing", CLASSIFY, EDGE),
        ),
        x=1.10, y=-0.19, item_width=3.05,
    )

    return figure


def build(
    root: Path, formats: list[str] | None = None, dpi: int = 160,
) -> list[Path]:
    """Write detailed-pipeline and high-level-system figures in vector formats."""
    requested = [
        value.lower().strip()
        for value in (formats or ["svg", "pdf"])
        if value.strip()
    ]
    unsupported = sorted(set(requested) - {"svg", "pdf"})
    if unsupported:
        raise ValueError(
            "architecture output supports SVG and PDF only: "
            + ", ".join(unsupported)
        )
    if not requested:
        raise ValueError("at least one architecture output format is required")

    root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    ):
        figures = (
            ("pipeline_architecture", _draw_pipeline_figure),
            ("system_architecture", _draw_system_figure),
        )
        for stem, draw in figures:
            figure = draw()
            try:
                for output_format in requested:
                    output = root / f"{stem}.{output_format}"
                    figure.savefig(
                        output,
                        format=output_format,
                        dpi=dpi,
                        bbox_inches="tight",
                        pad_inches=0.02,
                    )
                    if output_format == "svg":
                        lines = output.read_text(encoding="utf-8").splitlines()
                        output.write_text(
                            "\n".join(line.rstrip() for line in lines) + "\n",
                            encoding="utf-8",
                        )
                    outputs.append(output)
            finally:
                plt.close(figure)
    return outputs


def main(argv: list[str] | None = None) -> int:
    """Generate both architecture figures independently from catalogue plotting."""
    parser = argparse.ArgumentParser(
        description="Build the pipeline and system architecture figures",
    )
    parser.add_argument(
        "--output", type=Path, default=Path.cwd(),
        help="output directory; defaults to the current repository root",
    )
    parser.add_argument(
        "--formats", default="svg,pdf",
        help="comma-separated vector formats: svg,pdf",
    )
    args = parser.parse_args(argv)
    build(
        args.output,
        [value.strip() for value in args.formats.split(",")],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
