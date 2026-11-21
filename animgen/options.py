"""Run-time options of the animation generator.

Defaults come from ``.env`` (documented in ``.env.example``); the command line may
override them. Every value is validated, so a typo fails early with a readable
message instead of producing a broken GIF.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import config

DEFAULT_OUT = "Docs/subtasks/animations"
DEFAULT_PHRASE = "The robot read the book"
MIN_WIDTH, MIN_HEIGHT, MAX_SIZE = 480, 270, 1920


@dataclass(frozen=True)
class Options:
    """Everything that changes how the animations look and what they are computed from."""

    out_dir: Path
    width: int = 800
    height: int = 450
    frame_ms: int = 60  # duration of one tween frame (60 ms is about 16 frames per second)
    hold_ms: int = 1400  # how long a finished "picture" stays on screen
    supersample: int = 2  # draw at this multiple, then shrink: smooth edges and text
    train_steps: int = 300  # length of the training run behind the training/inference GIFs
    seed: int = 42
    phrase: str = DEFAULT_PHRASE
    attn_tokens: int = 8  # how many characters of the phrase the attention scenes show
    font: Optional[str] = None  # optional TrueType files; None = search the system
    font_bold: Optional[str] = None
    font_mono: Optional[str] = None

    def problems(self) -> list[str]:
        """Human-readable validation errors (empty list = valid)."""
        out: list[str] = []
        for name, low in (("width", MIN_WIDTH), ("height", MIN_HEIGHT)):
            value = getattr(self, name)
            if not low <= value <= MAX_SIZE:
                out.append(f"{name} must be between {low} and {MAX_SIZE} pixels, got {value}.")
        if not 20 <= self.frame_ms <= 1000:
            out.append("frame_ms must be between 20 and 1000.")
        if self.hold_ms < 0:
            out.append("hold_ms must be >= 0.")
        if self.supersample not in (1, 2, 3):
            out.append("supersample must be 1, 2 or 3.")
        if self.train_steps < 1:
            out.append("train_steps must be >= 1.")
        if not self.phrase.strip():
            out.append("phrase must not be empty.")
        if not 3 <= self.attn_tokens <= 10:
            out.append("attn_tokens must be between 3 and 10.")
        return out

    def validated(self) -> "Options":
        """Return ``self`` or raise ``ValueError`` listing every problem."""
        problems = self.problems()
        if problems:
            raise ValueError(" ".join(problems))
        return self

    def quick(self) -> "Options":
        """A fast preview: shorter training, smaller canvas, no supersampling."""
        return replace(self, train_steps=min(self.train_steps, 30), supersample=1, width=640, height=360)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _opt(name: str) -> Optional[str]:
    return os.environ.get(name, "").strip() or None


def resolve_out_dir(value: str) -> Path:
    """Relative folders are taken from the project root."""
    path = Path(value)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def from_env() -> Options:
    """Build the options from the environment (``.env`` is loaded by ``config``)."""
    return Options(
        out_dir=resolve_out_dir(os.environ.get("ANIM_OUTPUT_DIR", "").strip() or DEFAULT_OUT),
        width=_int("ANIM_WIDTH", 800),
        height=_int("ANIM_HEIGHT", 450),
        frame_ms=_int("ANIM_FRAME_MS", 60),
        train_steps=_int("ANIM_TRAIN_STEPS", 300),
        seed=config.load_settings().default_seed,
        phrase=os.environ.get("ANIM_PHRASE", "").strip() or DEFAULT_PHRASE,
        font=_opt("ANIM_FONT"),
        font_bold=_opt("ANIM_FONT_BOLD"),
        font_mono=_opt("ANIM_FONT_MONO"),
    ).validated()
