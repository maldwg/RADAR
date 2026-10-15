from __future__ import annotations
from radar.common.provenance import audited_bundle, register_input

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
    register_input(path)
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
            label, color=MUTED, fontsize=9.0,
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
        label, fontsize=9.0,
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
            color=MUTED, fontsize=9.0, zorder=5,
        )


def _component(
    ax: Axes, x: float, y: float, width: float, height: float,
    title: str, body: str = "", *, icon: str = "documents", facecolor: str = WHITE,
    edgecolor: str = EDGE, linestyle: str = "-",
) -> None:
    """Draw a readable component with separate heading and explanatory text."""
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.025",
        linewidth=0.95, edgecolor=edgecolor, facecolor=facecolor,
        linestyle=linestyle, zorder=3,
    ))
    heading_height = min(.40, height * .33)
    heading_bottom = y + height - heading_height - .12
    _add_icon(ax, x + width * .15, heading_bottom + heading_height / 2,
        icon, color=edgecolor, size=14)
    _card_label(ax, (x + width * .27, heading_bottom, width * .65, heading_height),
        title, fontsize=11.0)
    if body:
        _card_label(ax, (x + width * .07, y + .12, width * .86, height - heading_height - .36),
            body, fontsize=10.0, fontweight="normal", color=MUTED)


def _section_heading(ax: Axes, left: float, right: float, y: float, label: str, color: str) -> None:
    """Align a phase or subsystem heading with its component column."""
    ax.text((left + right) / 2, y, label, ha="center", va="center",
        color=INK, fontsize=10.5, fontweight="bold")
    ax.plot([left, right], [y - .24, y - .24], color=color, linewidth=3,
        solid_capstyle="round", zorder=1)


def _study_configuration(ax: Axes, y: float) -> None:
    """Show three independent studies and their shared paper execution schema."""
    for left, title, detail in (
        (.25, "IDS repositories", "Repository criteria, taxonomy and prompts"),
        (4.95, "IDS datasets", "Dataset criteria, taxonomy and prompts"),
        (9.65, "Conformal treatment effects", "Eligibility, quality and assumption extraction"),
    ):
        _component(ax, left, y, 4.30, .95, title, detail,
            icon="checklist", facecolor=REPORT)
    _component(ax, 14.65, y, 3.75, .95, "Shared paper config schema",
        "Same operational keys; study-specific values", icon="merge", facecolor=OPENALEX)


def _draw_pipeline_figure() -> Figure:
    """Show shared acquisition and separate scientific strategies for all three studies."""
    figure, ax = plt.subplots(figsize=(18.8, 10.0))
    figure.subplots_adjust(left=.01, right=.99, bottom=.02, top=.99)
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 18.65)
    ax.set_ylim(0, 10.65)
    ax.axis("off")

    ax.text(.25, 10.30, "RADAR · Study-owned configuration", color=INK, fontsize=11, fontweight="bold")
    _study_configuration(ax, 9.08)
    ax.text(9.32, 8.80,
        "Search profiles + pipeline values + screening + taxonomy + prompts  |  Shared runtime and model profiles",
        ha="center", color=MUTED, fontsize=9.5)
    for left, right, label, color in (
        (.25, 5.35, "(i)  Provider discovery", OPENALEX),
        (5.70, 8.10, "(ii)  Evidence + prefilter", CLASSIFY),
        (8.55, 11.40, "(iii)  Scientific screening", CLASSIFY),
        (11.80, 14.50, "(iv)  Consensus + reconciliation", CLASSIFY),
        (14.95, 18.40, "(v)  Outputs + evaluation", REPORT),
    ):
        _section_heading(ax, left, right, 8.42, label, color)

    _brand_box(ax, .25, 6.10, 1.90, 1.10, "github-lockup-black.svg",
        facecolor=GITHUB, target_width=36, label="Repository API")
    _brand_box(ax, .25, 4.48, 1.90, 1.10, "openalex-lockup-black.svg",
        facecolor=OPENALEX, target_width=64, label="Both paper studies")
    _component(ax, .25, 3.17, 1.90, 1.05, "Semantic Scholar", "Review discovery",
        icon="web", facecolor=OPENALEX)
    _component(ax, .25, 1.86, 1.90, 1.05, "Local CSV / JSON", "Review discovery",
        icon="documents", facecolor=OPENALEX)

    _component(ax, 2.55, 5.98, 2.80, 1.34, "Repository search",
        "Date sharding, deduplication, README + docs", icon="repository", facecolor=GITHUB)
    _component(ax, 2.55, 1.84, 2.80, 3.76, "Shared paper collector",
        "Direct paper search\nDiverse survey seeds (optional)\nBackward snowballing (optional)\nUnion, dedupe + configurable ranking",
        icon="merge", facecolor=OPENALEX)
    _arrow(ax, (2.23, 6.65), (2.47, 6.65))
    for center in (5.03, 3.695, 2.385):
        _arrow(ax, (2.23, center), (2.47, center))

    _component(ax, 5.70, 5.98, 2.40, 1.34, "Repository prefilter",
        "Deterministic evidence scores", icon="rank", facecolor=GITHUB)
    _component(ax, 5.70, 2.60, 2.40, 2.65, "Shared paper evidence",
        "DOI / title lookup + bounded PDF\nDeterministic prefilter\nCandidate / borderline gate",
        icon="documents", facecolor=OPENALEX)
    _arrow(ax, (5.43, 6.65), (5.62, 6.65))
    _arrow(ax, (5.43, 3.72), (5.62, 3.72))
    ax.text(5.30, 1.45, "Native query syntax · proportional budgets · raw audits · frozen cohorts",
        ha="center", color=MUTED, fontsize=9.5)

    _component(ax, 8.55, 5.98, 2.85, 1.34, "Repository role validation",
        "IDS repositories", icon="repository", facecolor=GITHUB)
    _component(ax, 8.55, 4.03, 2.85, 1.34, "Dataset release validation",
        "IDS datasets", icon="paper", facecolor=OPENALEX)
    _component(ax, 8.55, 2.28, 2.85, 1.34, "Criterion assessment",
        "Conformal treatment effects\nEligibility + quality + literal quotes", icon="checklist", facecolor=OPENALEX)
    _arrow(ax, (8.18, 6.65), (8.47, 6.65))
    _elbow_arrow(ax, [(8.18, 4.36), (8.33, 4.36), (8.33, 4.70), (8.47, 4.70)])
    _elbow_arrow(ax, [(8.18, 3.19), (8.33, 3.19), (8.33, 2.95), (8.47, 2.95)])

    _component(ax, 11.80, 5.98, 2.70, 1.34, "Consensus + ensemble",
        "Structured prompts\nRepeated assessments per model", icon="ensemble", facecolor=CLASSIFY)
    _arrow(ax, (11.48, 6.65), (11.72, 6.65))
    # The scientific strategy is selected per run; its result enters the same voting stage.
    _elbow_arrow(ax, [(11.48, 4.70), (11.59, 4.70), (11.59, 6.29), (11.72, 6.29)])
    _elbow_arrow(ax, [(11.48, 2.95), (11.64, 2.95), (11.64, 6.12), (11.72, 6.12)])
    for bottom, height, title, body, fill, icon in (
        (4.37, 1.10, "Repository families", "Repository reconciliation", GITHUB, "family"),
        (2.93, 1.10, "Publication families", "Dataset release reconciliation", OPENALEX, "family"),
        (1.56, 1.05, "Review assessments", "Release reconciliation disabled", OPENALEX, "checklist"),
    ):
        _component(ax, 11.80, bottom, 2.70, height, title, body, icon=icon, facecolor=fill)
    # Route each configured reconciliation path outside its cards, never through their labels.
    for target_y, target_top, route_x in ((4.92, 5.47, 14.66), (3.48, 4.03, 14.72), (2.085, 2.61, 14.78)):
        _elbow_arrow(ax, [(14.50, 6.20), (route_x, 6.20), (route_x, target_top + .15),
            (13.15, target_top + .15), (13.15, target_top + .07)], color=CATALOGUE_EDGE, linewidth=.95)

    for bottom, height, title, body, fill, icon in (
        (4.37, 1.10, "Repository catalogue", "Accepted repository records", GITHUB, "catalogue"),
        (2.93, 1.10, "Dataset catalogue", "Canonical releases and versions", OPENALEX, "catalogue"),
        (1.56, 1.05, "Review classifications", "Criteria, assumptions and audit quotes", OPENALEX, "checklist"),
    ):
        _component(ax, 14.95, bottom, 3.45, height, title, body, icon=icon,
            facecolor=fill, edgecolor=CATALOGUE_EDGE)
        _arrow(ax, (14.58, bottom + height / 2), (14.87, bottom + height / 2), color=CATALOGUE_EDGE)

    _component(ax, 14.95, 6.00, 3.45, 1.25, "Evaluation + reporting",
        "Recovery + extraction metrics\nReview: per-provider and union", icon="chart",
        facecolor=EVALUATE, edgecolor=EVALUATION_EDGE)
    _arrow(ax, (14.58, 6.65), (14.87, 6.65), color=EVALUATION_EDGE)
    # Reference labels feed evaluation only, not retrieval, evidence or scientific prompts.
    ax.add_patch(FancyBboxPatch((14.95, 7.51), 3.45, .49,
        boxstyle="round,pad=0.004,rounding_size=0.025", facecolor=WHITE,
        edgecolor=EVALUATION_EDGE, linestyle="--", linewidth=.9, zorder=3))
    _card_label(ax, (15.06, 7.58, 3.23, .33), "Reference profiles · evaluation only", fontsize=8.0)
    _arrow(ax, (16.68, 7.43), (16.68, 7.33), color=EVALUATION_EDGE, linestyle="--")
    _component(ax, 8.55, .70, 2.85, 1.07, "Catalogue baselines",
        "Keyword + NLI + LLM NLI\nRepository + dataset studies", icon="key",
        facecolor=EVALUATE, edgecolor=EVALUATION_EDGE)
    _elbow_arrow(ax, [(8.10, 2.63), (8.24, 2.63), (8.24, 1.24), (8.47, 1.24)],
        color=EVALUATION_EDGE, linewidth=.9)
    _elbow_arrow(ax, [(11.48, 1.20), (18.56, 1.20), (18.56, 6.65), (18.48, 6.65)],
        color=EVALUATION_EDGE, linewidth=.95)
    ax.text(.25, .96, "Select one study per run.\nPaper studies share acquisition code and config keys.",
        color=MUTED, fontsize=9.5, va="center")
    _color_legend(ax, (
        ("Repository strategy", GITHUB, EDGE), ("Shared paper code", OPENALEX, EDGE),
        ("Shared voting", CLASSIFY, EDGE), ("Evaluation only", EVALUATE, EVALUATION_EDGE),
        ("Study configuration", REPORT, EDGE),
    ), x=.35, y=.28, item_width=3.68)
    return figure


def _draw_system_figure() -> Figure:
    """Show provider contracts, shared paper modules, inference and audited study outputs."""
    figure, ax = plt.subplots(figsize=(18.8, 9.5))
    figure.subplots_adjust(left=.01, right=.99, bottom=.02, top=.99)
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0, 19.65)
    ax.set_ylim(0, 10.1)
    ax.axis("off")
    ax.text(.25, 9.80, "RADAR · Study manifests + shared runtime / models", color=INK,
        fontsize=11, fontweight="bold")
    _study_configuration(ax, 8.57)
    for left, right, label, color in (
        (.25, 2.35, "Providers", OPENALEX), (2.85, 5.50, "Adapter contracts", OPENALEX),
        (6.10, 9.25, "Collection runners", REPORT), (9.75, 12.85, "Evidence + scientific rules", CLASSIFY),
        (13.30, 15.65, "Inference", CLASSIFY), (16.05, 19.40, "Outputs", REPORT),
    ):
        _section_heading(ax, left, right, 8.08, label, color)

    _brand_box(ax, .25, 6.35, 2.10, 1.10, "github-lockup-black.svg",
        facecolor=GITHUB, target_width=36, label="Repository API")
    _brand_box(ax, .25, 4.85, 2.10, 1.10, "openalex-lockup-black.svg",
        facecolor=OPENALEX, target_width=65, label="Paper API")
    _component(ax, .25, 3.35, 2.10, 1.10, "Semantic Scholar",
        "Review discovery + shared evidence", icon="web", facecolor=OPENALEX)
    _component(ax, .25, 1.85, 2.10, 1.10, "Local CSV / JSON",
        "Frozen review paper input", icon="documents", facecolor=OPENALEX)
    _component(ax, 2.85, 1.85, 2.65, 5.60, "Provider registry",
        "adapters/registry.py\nGitHubAdapter\nOpenAlexAdapter\nSemanticScholarAdapter\nLocalDiscoveryAdapter\nDiscoveryAdapter + SourceRecord\nOptional lookup, evidence, references and catalogue protocols",
        icon="merge", facecolor=OPENALEX)
    for center in (6.90, 5.40, 3.90, 2.40):
        _arrow(ax, (2.43, center), (2.77, center))

    _component(ax, 6.10, 6.05, 3.15, 1.60, "Repository collection",
        "github/collect.py\nDate sharding + repository docs", icon="repository", facecolor=GITHUB)
    _component(ax, 6.10, 4.05, 3.15, 1.45, "Shared paper collector",
        "common/paper_collection.py\nIDS datasets + conformal", icon="paper", facecolor=OPENALEX)
    _component(ax, 6.10, 2.10, 3.15, 1.25, "Review provider union",
        "common/discovery.py\nStrong-ID deduplication + memberships", icon="merge", facecolor=OPENALEX)
    _arrow(ax, (5.58, 6.85), (6.02, 6.85))
    _arrow(ax, (5.58, 4.775), (6.02, 4.775))
    _arrow(ax, (5.58, 2.725), (6.02, 2.725))
    _arrow(ax, (7.675, 3.43), (7.675, 3.97))
    ax.text(7.89, 3.70, "uses", fontsize=7.6, color=MUTED, va="center")

    _component(ax, 9.75, 6.05, 3.10, 1.60, "Repository prefilter",
        "github/pipeline.py\nRepository document evidence", icon="rank", facecolor=GITHUB)
    _component(ax, 9.75, 4.05, 3.10, 1.45, "Shared paper evidence",
        "paper_evidence.py + paper_collection.py\nLookup, bounded PDF + prefilter", icon="documents", facecolor=OPENALEX)
    _component(ax, 9.75, 2.10, 3.10, 1.25, "Scientific strategies",
        "Repository roles · dataset releases\nReview eligibility + quality", icon="checklist", facecolor=CLASSIFY)
    _arrow(ax, (9.33, 6.85), (9.67, 6.85))
    _arrow(ax, (9.33, 4.775), (9.67, 4.775))
    _elbow_arrow(ax, [(9.33, 2.725), (9.48, 2.725), (9.48, 4.39), (9.67, 4.39)])
    _elbow_arrow(ax, [(12.85, 6.14), (13.01, 6.14), (13.01, 2.725), (12.93, 2.725)])
    _arrow(ax, (11.30, 3.97), (11.30, 3.43))

    _component(ax, 13.30, 4.05, 2.35, 1.45, "Adjudication client",
        "common/ollama.py\nStructured prompts + repeated votes", icon="classifier", facecolor=CLASSIFY)
    _elbow_arrow(ax, [(12.93, 2.40), (13.14, 2.40), (13.14, 4.775), (13.22, 4.775)])
    ax.add_patch(FancyBboxPatch((13.30, 1.60), 2.35, 2.05,
        boxstyle="round,pad=0.004,rounding_size=0.025", facecolor=CLASSIFY,
        edgecolor=EDGE, linewidth=.95, zorder=3))
    _logo_box(ax, 13.48, 2.84, 1.99, .61, "ollama.svg", "Ollama server",
        facecolor=WHITE, target_width=13, title_size=7.6)
    ax.text(14.475, 2.61, "Configured local models", ha="center", color=MUTED,
        fontsize=7.5, fontweight="bold", zorder=5)
    _model_badge(ax, 13.48, 1.78, .89, .66, "qwen.svg", "Qwen", target_width=23)
    _model_badge(ax, 14.58, 1.78, .89, .66, "gemma.png", "Gemma", target_width=18)
    _arrow(ax, (14.26, 3.97), (14.26, 3.73), color=EDGE)
    _arrow(ax, (14.69, 3.73), (14.69, 3.97), color=EDGE)

    _component(ax, 16.05, 5.86, 3.35, 1.79, "Study results",
        "Consensus + configured reconciliation\nRepository / dataset catalogues\nReview classifications", icon="catalogue", facecolor=REPORT)
    _component(ax, 16.05, 3.19, 3.35, 1.77, "Reporting + verification",
        "Tables, plots, reporting database\nFingerprints + figure manifests\nReview source + union recovery", icon="chart",
        facecolor=EVALUATE, edgecolor=EVALUATION_EDGE)
    _component(ax, 16.05, 1.43, 3.35, 1.14, "Reference profiles",
        "Evaluation only", icon="people", edgecolor=EVALUATION_EDGE, linestyle="--")
    _elbow_arrow(ax, [(15.73, 4.775), (15.86, 4.775), (15.86, 6.755), (15.97, 6.755)])
    _arrow(ax, (17.725, 5.78), (17.725, 5.04), color=EVALUATION_EDGE)
    _arrow(ax, (17.725, 2.65), (17.725, 3.11), color=EVALUATION_EDGE, linestyle="--")

    _logo_box(ax, .25, .48, 1.60, .88, "sqlite.svg", "Per-study state",
        facecolor=REPORT, target_width=42, title_size=7.0)
    _component(ax, 2.12, .48, 10.73, .88, "Run state + provenance",
        "Raw responses · SQLite caches · frozen JSON / CSV cohorts · assessment audits · run manifests",
        icon="database", facecolor=REPORT)
    ax.text(7.70, 1.58, "Stored per study and model; shared components do not mix studies",
        ha="center", color=MUTED, fontsize=8.0)
    _color_legend(ax, (
        ("Repository strategy", GITHUB, EDGE), ("Shared paper code", OPENALEX, EDGE),
        ("Scientific / inference", CLASSIFY, EDGE), ("Evaluation only", EVALUATE, EVALUATION_EDGE),
    ), x=1.70, y=.18, item_width=4.30)
    return figure


@audited_bundle
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
