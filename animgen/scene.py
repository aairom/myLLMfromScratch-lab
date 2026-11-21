"""Scene model and frame rendering.

A :class:`Scene` is one animation (one GIF). It has a list of captions, one per
*step*, and a single ``draw(canvas, ctx, step, t)`` function. ``step`` is the
index of the step being shown and ``t`` its eased progress (0 to 1). Earlier
steps must be drawn complete, later ones not at all; :func:`phase` turns
``(step, t)`` into that 0..1 progress for any step index, so a scene reads like::

    a = phase(step, t, 0)      # progress of step 0
    b = phase(step, t, 1)      # progress of step 1
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PIL import Image

from animgen import theme
from animgen.canvas import Canvas
from animgen.options import Options

DrawFn = Callable[[Canvas, "object", int, float], None]


def phase(step: int, t: float, index: int) -> float:
    """Progress (0..1) of step ``index`` while step ``step`` is at ``t``."""
    if step > index:
        return 1.0
    if step == index:
        return t
    return 0.0


@dataclass(frozen=True)
class Scene:
    """One animation: metadata, a caption per step and the drawing function."""

    key: str  # file stem, e.g. ``02_tokenization``
    number: int  # sub-task number (1-10)
    kicker: str  # small line above the title
    title: str
    captions: tuple[str, ...]
    draw: DrawFn
    frames_per_step: int = 12

    @property
    def filename(self) -> str:
        return f"{self.key}.gif"

    @property
    def steps(self) -> int:
        return len(self.captions)

    def total_frames(self) -> int:
        return self.steps * self.frames_per_step


@dataclass(frozen=True)
class Frame:
    image: Image.Image
    duration_ms: int


def render_scene(scene: Scene, ctx, options: Options) -> list[Frame]:
    """All frames of ``scene``; the last frame of every step is held for ``options.hold_ms``."""
    frames: list[Frame] = []
    n = max(2, scene.frames_per_step)
    for step, caption in enumerate(scene.captions):
        for k in range(n):
            t = theme.ease(k / (n - 1))
            canvas = Canvas(options.width, options.height, options.supersample)
            canvas.chrome(scene.kicker, scene.title, step, scene.steps, caption)
            scene.draw(canvas, ctx, step, t)
            last = k == n - 1
            frames.append(Frame(canvas.finish(), options.frame_ms + (options.hold_ms if last else 0)))
    return frames


def shared_palette(images: list[Image.Image], colours: int = 255, samples: int = 12) -> Image.Image:
    """One palette for the whole animation, so colours do not flicker between frames."""
    picked = images[:: max(1, len(images) // samples)][:samples]
    w, h = picked[0].size
    sheet = Image.new("RGB", (w, h * len(picked)))
    for i, im in enumerate(picked):
        sheet.paste(im, (0, i * h))
    return sheet.quantize(colors=colours, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)


def save_gif(frames: list[Frame], path, loop: int = 0) -> int:
    """Write ``frames`` as an endlessly looping GIF; returns the file size in bytes."""
    if not frames:
        raise ValueError("no frames to save")
    palette = shared_palette([f.image for f in frames])
    paletted = [f.image.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    path.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=paletted[1:],
        duration=[f.duration_ms for f in frames],
        loop=loop,
        optimize=False,
        disposal=1,
    )
    return path.stat().st_size


def preview(scene: Scene, ctx, options: Options, step: Optional[int] = None, t: float = 1.0) -> Image.Image:
    """One still frame (for tests and quick looks): step ``step`` at progress ``t``."""
    step = scene.steps - 1 if step is None else step
    canvas = Canvas(options.width, options.height, options.supersample)
    canvas.chrome(scene.kicker, scene.title, step, scene.steps, scene.captions[step])
    scene.draw(canvas, ctx, step, t)
    return canvas.finish()
