"""A tiny drawing toolkit on top of Pillow.

Scenes draw in a fixed *design space* of 800 x 450 units. The canvas scales that
to the requested pixel size (letter-boxed when the aspect ratio differs) and
renders at ``supersample`` times the resolution before shrinking, which gives
smooth edges and readable text without any other dependency.
"""
from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from PIL import Image, ImageDraw

from animgen import fonts, theme
from animgen.theme import RGB

DESIGN_W, DESIGN_H = 800, 450
LEFT, RIGHT = 24, 776  # horizontal margins of the content area
CONTENT_TOP, CONTENT_BOTTOM = 76, 384  # vertical extent of the content area
Point = tuple[float, float]


class Canvas:
    """One frame under construction."""

    def __init__(self, width: int, height: int, supersample: int = 2) -> None:
        self.width, self.height, self.ss = width, height, supersample
        self.k = min(width / DESIGN_W, height / DESIGN_H) * supersample  # design unit -> pixel
        self.ox = (width * supersample - DESIGN_W * self.k) / 2
        self.oy = (height * supersample - DESIGN_H * self.k) / 2
        self.img = Image.new("RGB", (width * supersample, height * supersample), theme.BG)
        self.d = ImageDraw.Draw(self.img)

    # ----- coordinates ------------------------------------------------------ #
    def px(self, x: float, y: float) -> Point:
        return (self.ox + x * self.k, self.oy + y * self.k)

    def _w(self, width: float) -> int:
        return max(1, round(width * self.k))

    # ----- text --------------------------------------------------------------- #
    def font(self, size: float, kind: str = fonts.SANS):
        return fonts.load(kind, max(6, round(size * self.k)))

    def text_width(self, s: str, size: float = 14, kind: str = fonts.SANS) -> float:
        """Width of ``s`` in design units."""
        return self.d.textlength(fonts.safe_text(kind, s), font=self.font(size, kind)) / self.k

    def text(self, x: float, y: float, s: str, size: float = 14, kind: str = fonts.SANS,
             fill: RGB = theme.INK, anchor: str = "la") -> None:
        """Draw ``s``; ``anchor`` is a Pillow anchor such as ``la`` (left/top), ``mm`` or ``ra``."""
        self.d.text(self.px(x, y), fonts.safe_text(kind, s), font=self.font(size, kind), fill=fill, anchor=anchor)

    # ----- shapes ------------------------------------------------------------- #
    def rect(self, x0: float, y0: float, x1: float, y1: float, fill: Optional[RGB] = None,
             outline: Optional[RGB] = None, width: float = 1.5, radius: float = 0) -> None:
        a, b = self.px(x0, y0), self.px(x1, y1)
        if radius > 0:
            self.d.rounded_rectangle([a, b], radius=radius * self.k, fill=fill, outline=outline, width=self._w(width))
        else:
            self.d.rectangle([a, b], fill=fill, outline=outline, width=self._w(width))

    def line(self, points: Sequence[Point], fill: RGB = theme.INK, width: float = 2) -> None:
        if len(points) >= 2:
            self.d.line([self.px(*p) for p in points], fill=fill, width=self._w(width), joint="curve")

    def circle(self, cx: float, cy: float, r: float, fill: Optional[RGB] = None,
               outline: Optional[RGB] = None, width: float = 1.5) -> None:
        a, b = self.px(cx - r, cy - r), self.px(cx + r, cy + r)
        self.d.ellipse([a, b], fill=fill, outline=outline, width=self._w(width))

    def polygon(self, points: Sequence[Point], fill: RGB) -> None:
        self.d.polygon([self.px(*p) for p in points], fill=fill)

    def arrow(self, p0: Point, p1: Point, fill: RGB = theme.MUTED, width: float = 2, head: float = 8) -> None:
        """Line from ``p0`` to ``p1`` with a filled arrow head at ``p1``."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            return
        ux, uy = dx / length, dy / length
        head = min(head, length)
        base = (p1[0] - ux * head, p1[1] - uy * head)
        self.line([p0, base], fill, width)
        nx, ny = -uy, ux
        self.polygon([p1, (base[0] + nx * head * 0.5, base[1] + ny * head * 0.5),
                      (base[0] - nx * head * 0.5, base[1] - ny * head * 0.5)], fill)

    def dashed(self, p0: Point, p1: Point, fill: RGB = theme.MUTED, width: float = 1.5, dash: float = 6) -> None:
        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        n = max(1, int(length / dash))
        for i in range(0, n, 2):
            a, b = i / n, min(1.0, (i + 1) / n)
            self.line([(theme.lerp(p0[0], p1[0], a), theme.lerp(p0[1], p1[1], a)),
                       (theme.lerp(p0[0], p1[0], b), theme.lerp(p0[1], p1[1], b))], fill, width)

    # ----- composite widgets ------------------------------------------------- #
    def box(self, x: float, y: float, w: float, h: float, label: str = "", fill: RGB = theme.PANEL,
            outline: RGB = theme.GRID, size: float = 14, kind: str = fonts.SANS, text_fill: RGB = theme.INK,
            radius: float = 8, width: float = 1.5) -> None:
        """Rounded box with a centred label."""
        self.rect(x, y, x + w, y + h, fill=fill, outline=outline, width=width, radius=radius)
        if label:
            self.text(x + w / 2, y + h / 2, label, size, kind, text_fill, "mm")

    def bar(self, x: float, y: float, w: float, h: float, frac: float, fill: RGB = theme.TOKEN,
            back: RGB = theme.GRID) -> None:
        self.rect(x, y, x + w, y + h, fill=back, radius=h / 2)
        if frac > 0.002:
            self.rect(x, y, x + max(h, w * theme.clamp01(frac)), y + h, fill=fill, radius=h / 2)

    def matrix(self, x: float, y: float, values: Sequence[Sequence[Optional[float]]], cell: tuple[float, float] = (22, 22),
               colour: Callable[[float], RGB] = theme.heat, fmt: Optional[str] = None, text_size: float = 10,
               gap: float = 1, outline: Optional[RGB] = None, masked: RGB = (232, 235, 240)) -> None:
        """Heat-map grid; ``None`` entries are drawn as a grey masked cell."""
        cw, ch = cell
        for r, row in enumerate(values):
            for c, v in enumerate(row):
                x0, y0 = x + c * cw, y + r * ch
                if v is None:
                    self.rect(x0, y0, x0 + cw - gap, y0 + ch - gap, fill=masked)
                    continue
                fill = colour(v)
                self.rect(x0, y0, x0 + cw - gap, y0 + ch - gap, fill=fill, outline=outline, width=0.8)
                if fmt:
                    dark = sum(fill) < 330
                    self.text(x0 + (cw - gap) / 2, y0 + (ch - gap) / 2, format(v, fmt), text_size, fonts.MONO,
                              theme.PANEL if dark else theme.INK, "mm")

    def token(self, x: float, y: float, label: str, fill: RGB = theme.TOKEN, size: float = 15, pad: float = 7,
              height: float = 28, min_w: float = 28, text_fill: RGB = theme.PANEL, kind: str = fonts.MONO) -> float:
        """Filled token chip; returns its width so rows can be laid out."""
        label = fonts.visible_token(label)
        w = max(min_w, self.text_width(label, size, kind) + 2 * pad)
        self.rect(x, y, x + w, y + height, fill=fill, radius=6)
        self.text(x + w / 2, y + height / 2, label, size, kind, text_fill, "mm")
        return w

    def polyline(self, xs: Sequence[float], ys: Sequence[float], box: tuple[float, float, float, float],
                 xlim: tuple[float, float], ylim: tuple[float, float], fill: RGB = theme.TOKEN, width: float = 2,
                 upto: Optional[float] = None) -> Optional[Point]:
        """Plot a series into ``box`` = (x0, y0, x1, y1); ``upto`` (0..1) reveals it progressively.

        Returns the last drawn point (handy for a moving marker).
        """
        x0, y0, x1, y1 = box
        pts = []
        for xv, yv in zip(xs, ys):
            fx = (xv - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.0
            fy = (yv - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] != ylim[0] else 0.0
            pts.append((x0 + (x1 - x0) * fx, y1 - (y1 - y0) * theme.clamp01(fy)))
        if upto is not None:
            pts = pts[: max(1, round(len(pts) * theme.clamp01(upto)))]
        self.line(pts, fill, width)
        return pts[-1] if pts else None

    def axes(self, box: tuple[float, float, float, float], title: str = "", xlabel: str = "", ylabel: str = "") -> None:
        """Frame, light grid and labels for a plot."""
        x0, y0, x1, y1 = box
        self.rect(x0, y0, x1, y1, fill=theme.PANEL, outline=theme.GRID, radius=4)
        for i in range(1, 4):
            gy = y0 + (y1 - y0) * i / 4
            self.line([(x0, gy), (x1, gy)], theme.GRID, 0.8)
        if title:
            self.text(x0, y0 - 6, title, 12, fonts.BOLD, theme.INK, "ls")
        if xlabel:
            self.text((x0 + x1) / 2, y1 + 4, xlabel, 11, fonts.SANS, theme.MUTED, "ma")
        if ylabel:
            self.text(x0 - 5, y0, ylabel, 11, fonts.SANS, theme.MUTED, "ra")

    # ----- frame furniture ---------------------------------------------------- #
    def chrome(self, kicker: str, title: str, step: int, steps: int, caption: str) -> None:
        """Header (sub-task and title), progress dots and the caption bar of the current step."""
        self.text(LEFT, 14, kicker, 12, fonts.SANS, theme.MUTED)
        self.text(LEFT, 30, title, 23, fonts.BOLD, theme.INK)
        for i in range(steps):
            cx = RIGHT - (steps - 1 - i) * 16 - 5
            self.circle(cx, 28, 5, fill=theme.TOKEN if i <= step else theme.GRID)
        self.line([(LEFT, 66), (RIGHT, 66)], theme.GRID, 1.2)
        self.rect(LEFT, 396, RIGHT, 438, fill=theme.PANEL, outline=theme.GRID, radius=8)
        self.text(LEFT + 12, 417, f"{step + 1}/{steps}", 13, fonts.BOLD, theme.TOKEN, "lm")
        self.text(LEFT + 52, 417, caption, 14, fonts.SANS, theme.INK, "lm")

    def finish(self) -> Image.Image:
        """The finished frame at the requested pixel size."""
        if self.ss == 1:
            return self.img
        return self.img.resize((self.width, self.height), Image.Resampling.LANCZOS)
