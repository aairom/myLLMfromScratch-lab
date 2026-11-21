"""Font lookup for the animations.

Pillow's built-in font lacks many symbols, so the generator looks for a system
TrueType font (sans, bold sans, monospace) and falls back to the built-in one.
Text is always passed through :func:`safe_text`, which replaces characters the
chosen font cannot draw, so a missing glyph never shows up as an empty box.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import ImageFont

SANS = "sans"
BOLD = "bold"
MONO = "mono"

# Common locations on macOS, Linux and Windows; the first existing file wins.
CANDIDATES: dict[str, tuple[str, ...]] = {
    SANS: (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ),
    BOLD: (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ),
    MONO: (
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ),
}

_overrides: dict[str, Optional[str]] = {SANS: None, BOLD: None, MONO: None}


def configure(sans: Optional[str] = None, bold: Optional[str] = None, mono: Optional[str] = None) -> None:
    """Use explicit font files (from ``.env``) instead of searching."""
    _overrides.update({SANS: sans, BOLD: bold, MONO: mono})
    load.cache_clear()
    _missing.cache_clear()
    has_glyph.cache_clear()


def find_font_file(kind: str) -> Optional[Path]:
    """Path of the font file used for ``kind``, or ``None`` when only the built-in font is left."""
    explicit = _overrides.get(kind)
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"font file not found: {explicit}")
        return path
    for candidate in CANDIDATES[kind]:
        if Path(candidate).is_file():
            return Path(candidate)
    return None


@lru_cache(maxsize=64)
def load(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """The font of ``kind`` at ``size`` pixels."""
    path = find_font_file(kind)
    if path is not None:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:  # unreadable or unsupported file: fall through to the built-in font
            pass
    return ImageFont.load_default(size=size)


@lru_cache(maxsize=64)
def _missing(kind: str) -> bytes:
    """Bitmap the font draws for a character it does not have (a 'notdef' box)."""
    return bytes(load(kind, 24).getmask("\ue000"))


@lru_cache(maxsize=4096)
def has_glyph(kind: str, char: str) -> bool:
    if char.isspace():
        return True
    font = load(kind, 24)
    return bytes(font.getmask(char)) != _missing(kind)


def safe_text(kind: str, text: str) -> str:
    """Replace characters ``kind`` cannot draw by ``?``."""
    return "".join(c if has_glyph(kind, c) else "?" for c in text)


def visible(char: str) -> str:
    """Make whitespace readable inside a token box (``' '`` -> ``'_'``)."""
    return {" ": "_", "\n": "/n", "\t": "/t", "\r": "/r"}.get(char, char)


def visible_token(token: str) -> str:
    return "".join(visible(c) for c in token)
