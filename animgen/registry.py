"""All animations in teaching order, and how to pick some of them."""
from __future__ import annotations

from typing import Iterable, Optional

from animgen import scenes_a, scenes_b, scenes_c, scenes_d
from animgen.scene import Scene

ALL_SCENES: tuple[Scene, ...] = tuple(
    sorted(
        (*scenes_a.SCENES, *scenes_b.SCENES, *scenes_c.SCENES, *scenes_d.SCENES),
        key=lambda s: s.key,
    )
)


def find(token: str) -> list[Scene]:
    """Scenes matching ``token``: a sub-task number (``5``), a key or a part of it (``attention``)."""
    token = token.strip().lower()
    if token.isdigit():
        return [s for s in ALL_SCENES if s.number == int(token)]
    return [s for s in ALL_SCENES if token in s.key.lower()]


def select(tokens: Optional[Iterable[str]] = None) -> list[Scene]:
    """The scenes to render: all of them, or the union of what each token matches.

    Raises ``ValueError`` for a token that matches nothing, so a typo is reported
    instead of silently producing no GIF.
    """
    if not tokens:
        return list(ALL_SCENES)
    picked: list[Scene] = []
    for token in tokens:
        found = find(token)
        if not found:
            raise ValueError(f"no animation matches {token!r}; try --list")
        picked.extend(s for s in found if s not in picked)
    return sorted(picked, key=lambda s: s.key)
