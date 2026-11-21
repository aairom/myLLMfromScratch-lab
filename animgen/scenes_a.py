"""Scenes 1-3: scaffolding, tokenization, embeddings and positional encoding."""
from __future__ import annotations

import math

import torch

from animgen import fonts, parts, theme
from animgen.canvas import LEFT, RIGHT, Canvas
from animgen.context import MAX_TRACE_STEPS, Context
from animgen.scene import Scene, phase

K1 = "Sub-task 1 - Project scaffolding"
K2 = "Sub-task 2 - Tokenization"
K3 = "Sub-task 3 - Embeddings and positional encoding"


# --------------------------------------------------------------------------- #
# 1. Scaffolding
# --------------------------------------------------------------------------- #
def draw_scaffolding(c: Canvas, ctx: Context, step: int, t: float) -> None:
    facts = ctx.project
    # Step 0: environment (.venv, requirements); 1: .env -> config; 2: tests; 3: git hygiene
    a, b, d, e = (phase(step, t, i) for i in range(4))

    # column 1: environment
    x = LEFT
    parts.panel_title(c, x, 84, "1  Environment (uv)", theme.TOKEN)
    for i, name in enumerate(facts.requirements[:9]):
        show = theme.segment(a, i / 12, i / 12 + 0.35)
        if show > 0:
            c.box(x, 108 + i * 28, 150, 22, name, fill=parts.fade(theme.tint(theme.TOKEN, 0.18), show), outline=parts.fade(theme.TOKEN, show),
                  size=12, kind=fonts.MONO, text_fill=parts.fade(theme.INK, show), radius=5)
    parts.note(c, x, 108 + 9 * 28 + 4, f"{len(facts.requirements)} packages in requirements.txt", 11)

    # column 2: .env -> config.py
    x = 214
    parts.panel_title(c, x, 84, "2  Configuration", theme.QUERY)
    shown_keys = facts.env_keys[:7]
    for i, (key, value) in enumerate(shown_keys):
        show = theme.segment(b, i / 10, i / 10 + 0.3)
        if show > 0:
            c.rect(x, 108 + i * 30, x + 176, 108 + i * 30 + 24, fill=parts.fade(theme.tint(theme.QUERY, 0.18), show),
                   outline=parts.fade(theme.QUERY, show), radius=5)
            c.text(x + 7, 120 + i * 30, key, 11, fonts.MONO, parts.fade(theme.INK, show), "lm")
            if c.text_width(key, 11, fonts.MONO) + c.text_width(value, 10, fonts.MONO) < 150:  # skip values that would collide
                c.text(x + 169, 120 + i * 30, value, 10, fonts.MONO, parts.fade(theme.MUTED, show), "rm")
    if b > 0.6:
        c.arrow((x + 90, 108 + len(shown_keys) * 30 + 2), (x + 90, 108 + len(shown_keys) * 30 + 26), theme.QUERY, 2)
        c.box(x, 108 + len(shown_keys) * 30 + 28, 176, 26, "config.Settings", fill=theme.PANEL, outline=theme.QUERY,
              size=12, kind=fonts.MONO, radius=5)
        parts.note(c, x, 108 + len(shown_keys) * 30 + 58, "validated once, used everywhere", 11)

    # column 3: tests
    x = 420
    parts.panel_title(c, x, 84, "3  Tests", theme.VALUE)
    total = facts.test_modules
    cols = 8
    for i in range(total):
        r_, c_ = divmod(i, cols)
        show = theme.segment(d, i / (total + 6), i / (total + 6) + 0.25)
        if show > 0:
            c.rect(x + c_ * 22, 108 + r_ * 22, x + c_ * 22 + 18, 108 + r_ * 22 + 18, fill=parts.fade(theme.VALUE, show), radius=4)
    parts.note(c, x, 108 + math.ceil(total / cols) * 22 + 8, f"{total} test modules", 11)
    if d > 0.9:
        parts.check(c, x + 82, 91, 8, theme.VALUE, theme.segment(d, 0.9, 1.0))

    # column 4: git hygiene
    x = 606
    parts.panel_title(c, x, 84, "4  Git hygiene", theme.LOSS)
    for i, rule in enumerate(facts.gitignore[:11]):
        show = theme.segment(e, i / 14, i / 14 + 0.3)
        if show > 0:
            c.line([(x, 118 + i * 22), (x + 7, 118 + i * 22)], parts.fade(theme.LOSS, show), 2)
            c.text(x + 14, 110 + i * 22, rule, 12, fonts.MONO, parts.fade(theme.INK, show))
    parts.note(c, x, 110 + 11 * 22 + 6, "ignored: secrets, caches, data", 11)


SCAFFOLDING = Scene(
    "01_scaffolding", 1, K1, "From an empty folder to a reproducible lab",
    ("uv builds a project-local .venv from requirements.txt",
     "Settings come from .env, validated once in config.py",
     "Every source module gets a unit test",
     ".gitignore keeps secrets, caches and data out of git"),
    draw_scaffolding,
)


# --------------------------------------------------------------------------- #
# 2. Tokenization
# --------------------------------------------------------------------------- #
def draw_tokenization(c: Canvas, ctx: Context, step: int, t: float) -> None:
    text = ctx.phrase
    chars = list(text)
    ids = ctx.phrase_ids
    a, b, d = phase(step, t, 0), phase(step, t, 1), phase(step, t, 2)

    # Row A: characters -> ids (steps 0 and 1); both rows share the same chip positions
    x0 = parts.centred_x(c, chars, 15, 3, 24)
    c.text(LEFT, 84, "Text", 12, fonts.BOLD, theme.MUTED)
    boxes = parts.chip_row(c, x0, 100, chars, theme.TOKEN, 15, 28, 3, 24, shown=a * (len(chars) + 1) if step == 0 else None)
    if step >= 1:
        reveal = b * (len(ids) + 1) if step == 1 else len(ids) + 1
        for i, (bx, bw) in enumerate(boxes):
            show = theme.clamp01(reveal - i)
            if show > 0:
                c.line([(bx + bw / 2, 128), (bx + bw / 2, 140)], parts.fade(theme.GRID, show), 1.2)
                c.rect(bx, 140, bx + bw, 164, fill=parts.fade(theme.ACCENT, show), radius=5)
                c.text(bx + bw / 2, 152, str(ids[i]), 11, fonts.MONO, parts.fade(theme.PANEL, 0.4 + 0.6 * show), "mm")
        c.text(LEFT, 172, "Character ids (vocabulary of %d symbols)" % ctx.char_tok.vocab_size, 12, fonts.BOLD, theme.MUTED)
        if step == 1:
            c.text(LEFT, 204, "The vocabulary: every distinct character of the corpus gets the next free id", 12, fonts.BOLD, theme.MUTED)
            per_row = 17
            for i, tok_str in enumerate(ctx.char_tok.itos[:51]):
                r_, c_ = divmod(i, per_row)
                used = i in ids
                gx, gy = LEFT + c_ * 44, 228 + r_ * 38
                fill = theme.ACCENT if used else theme.tint(theme.ACCENT, 0.15)
                show = theme.clamp01(b * 2.2 - i / 51)
                c.rect(gx, gy, gx + 40, gy + 32, fill=parts.fade(fill, show), radius=5)
                c.text(gx + 20, gy + 11, fonts.visible_token(tok_str) if tok_str != "<unk>" else "?", 12, fonts.MONO,
                       parts.fade(theme.PANEL if used else theme.INK, show), "mm")
                c.text(gx + 20, gy + 24, str(i), 9, fonts.MONO, parts.fade(theme.PANEL if used else theme.MUTED, show), "mm")

    # Row B: BPE replay (step 2)
    if step >= 2:
        trace = ctx.bpe.trace[:MAX_TRACE_STEPS]
        n = len(trace)
        k = min(n, int(d * (n + 0.999)))  # merges applied so far
        c.line([(LEFT, 196), (RIGHT, 196)], theme.GRID, 1)
        c.text(LEFT, 204, "Byte-pair encoding: merge the most frequent neighbouring pair, over and over", 12, fonts.BOLD, theme.MUTED)
        toks = ctx.bpe_stage(k)
        x1 = parts.centred_x(c, toks, 14, 3, 20)
        colours = [theme.QUERY if len(tk) > 1 else theme.TOKEN for tk in toks]
        parts.chip_row(c, x1, 232, toks, colours, 14, 28, 3, 20)
        if k > 0:
            s = trace[k - 1]
            c.box(LEFT, 282, 360, 54, "", fill=theme.PANEL, outline=theme.QUERY, radius=8)
            c.text(LEFT + 12, 292, f"Merge {s.step}: ", 13, fonts.BOLD, theme.INK)
            c.text(LEFT + 92, 292, f"'{fonts.visible_token(s.pair[0])}' + '{fonts.visible_token(s.pair[1])}'  ->  '{fonts.visible_token(s.new_token)}'", 13, fonts.MONO, theme.QUERY)
            c.text(LEFT + 12, 314, f"seen {s.count} times in the corpus", 12, fonts.SANS, theme.MUTED)
        c.text(RIGHT, 292, f"{len(chars)} characters", 13, fonts.SANS, theme.MUTED, "ra")
        c.text(RIGHT, 314, f"{len(toks)} tokens", 20, fonts.BOLD, theme.QUERY if len(toks) < len(chars) else theme.TOKEN, "ra")
        # merge ladder
        for i, s in enumerate(trace):
            done = i < k
            c.rect(LEFT + i * 46, 350, LEFT + i * 46 + 40, 372, fill=theme.QUERY if done else theme.GRID, radius=4)
            c.text(LEFT + i * 46 + 20, 361, fonts.visible_token(s.new_token), 10, fonts.MONO, theme.PANEL if done else theme.MUTED, "mm")
    if step == 0:
        c.text(400, 200, "Each character becomes one token", 16, fonts.SANS, theme.MUTED, "mm")


TOKENIZATION = Scene(
    "02_tokenization", 2, K2, "From text to integer ids",
    ("Split the text into tokens (here: single characters)",
     "Look every token up in the vocabulary to get its integer id",
     "BPE learns merges, so frequent character groups become single tokens"),
    draw_tokenization, frames_per_step=18,
)


# --------------------------------------------------------------------------- #
# 3. Embeddings and positional encoding
# --------------------------------------------------------------------------- #
def _position_table(c: Canvas, ctx: Context, progress: float) -> None:
    """Step 4: the whole sinusoidal table as a heat map, plus three of its dimensions as curves."""
    table = ctx.pos_table
    rows, cols_n = table.shape
    c.text(LEFT, 84, f"Position signal: {rows} positions (rows) x {cols_n} dimensions (columns)", 12, fonts.BOLD, theme.ACCENT)
    px0, py0, pw, ph = LEFT, 104, 440, 250
    cw, rh = pw / cols_n, ph / rows
    for r in range(int(progress * rows)):
        for col in range(cols_n):
            c.rect(px0 + col * cw, py0 + r * rh, px0 + (col + 1) * cw, py0 + (r + 1) * rh,
                   fill=theme.diverging(float(table[r, col]), 1.0))
    c.rect(px0, py0, px0 + pw, py0 + ph, outline=theme.GRID, radius=0)
    c.text(px0 + pw / 2, py0 + ph + 6, "dimension  (fast waves left, slow waves right)", 11, fonts.SANS, theme.MUTED, "ma")
    box = (520, 104, 776, 328)
    c.axes(box, "Three dimensions along the position", "position")
    for k, (dim, col) in enumerate(((4, theme.TOKEN), (12, theme.QUERY), (28, theme.VALUE))):
        c.polyline(list(range(rows)), [(float(table[r, dim]) + 1) / 2 for r in range(rows)], box, (0, rows - 1), (0, 1),
                   col, 2, upto=progress)
        c.line([(box[0] + 4 + k * 86, 362), (box[0] + 22 + k * 86, 362)], col, 3)
        c.text(box[0] + 28 + k * 86, 362, f"dim {dim}", 11, fonts.MONO, theme.INK, "lm")


def draw_embeddings(c: Canvas, ctx: Context, step: int, t: float) -> None:
    if step == 3:
        _position_table(c, ctx, t)
        return
    chars = list(ctx.attn_text)
    ids = ctx.phrase_ids[: len(chars)]
    n, dims = len(chars), 12  # only the first 12 numbers of every vector are drawn
    d_model = ctx.model.cfg.d_model
    emb = ctx.details["token_embeddings"][0][:, :dims]
    pos = ctx.pos_table[:n, :dims]
    total = ctx.details["with_position"][0][:, :dims]
    a, b, d = (phase(step, t, i) for i in range(3))
    cell = (14.0, 24.0)
    x_emb, x_pos, x_sum, top = 130, 332, 534, 110

    def reveal(progress: float, i: int) -> float:
        return theme.segment(progress, i / (n + 2), i / (n + 2) + 0.4)

    def row(x: float, y: float, source, i: int, vmax: float, show: float) -> None:
        c.matrix(x, y, [[float(v) for v in source[i]]], cell, lambda v: parts.fade(theme.diverging(v, vmax), show))

    c.text(x_emb, 90, "token vector", 12, fonts.BOLD, theme.TOKEN)
    if step >= 1:
        c.text(x_pos, 90, "position signal", 12, fonts.BOLD, theme.ACCENT)
    if step >= 2:
        c.text(x_sum, 90, "input of block 1", 12, fonts.BOLD, theme.INK)

    for i in range(n):
        y = top + i * 30
        c.token(24, y, chars[i], theme.TOKEN, 14, 7, 24, 40)
        c.text(70, y + 12, f"id {ids[i]}", 10, fonts.MONO, theme.MUTED, "lm")
        show = reveal(a, i)
        if show > 0:
            c.arrow((106, y + 13), (x_emb - 6, y + 13), parts.fade(theme.MUTED, show), 1.5, 6)
            row(x_emb, y, emb, i, 0.08, show)
        if step >= 1 and reveal(b, i) > 0:
            c.text(x_pos - 17, y + 13, "+", 15, fonts.BOLD, theme.ACCENT, "mm")
            row(x_pos, y, pos, i, 1.0, reveal(b, i))
        if step >= 2 and reveal(d, i) > 0:
            c.text(x_sum - 17, y + 13, "=", 15, fonts.BOLD, theme.INK, "mm")
            row(x_sum, y, total, i, 1.0, reveal(d, i))

    parts.note(c, x_emb, 352, f"Each row shows the first {dims} of {d_model} numbers. Blue = negative, orange = positive.", 11)
    notes = (
        "An id selects one row of a (vocabulary x d_model) table of learnable numbers.",
        "Sine and cosine waves tell the model where each token stands; without them 'ab' and 'ba' look alike.",
        f"Token vector + position signal: shape {tuple(ctx.details['with_position'].shape)} = (batch, tokens, d_model).",
    )
    parts.note(c, x_emb, 369, notes[step], 11, theme.INK)


EMBEDDINGS = Scene(
    "03_embeddings_positional", 3, K3, "Ids become vectors, and vectors learn where they stand",
    ("An embedding table turns every token id into a vector of learnable numbers",
     "A sine/cosine signal encodes each position (blue = negative, orange = positive)",
     "Token vector plus position signal is what enters the first transformer block",
     "The signal is a fixed pattern of waves: fast ones for neighbours, slow ones for distance"),
    draw_embeddings, frames_per_step=14,
)

SCENES = (SCAFFOLDING, TOKENIZATION, EMBEDDINGS)
