"""Scenes 4-5: attention and layer internals, the full GPT model, and text generation."""
from __future__ import annotations

import math

import torch

from animgen import fonts, parts, theme
from animgen.canvas import LEFT, RIGHT, Canvas
from animgen.context import Context, top_k
from animgen.scene import Scene, phase
from llm.layers import gelu, relu
from llm.transformer import next_token_distribution
from viz.training import block_stages, model_stages

K4 = "Sub-task 4 - Attention, LayerNorm and feed-forward"
K5 = "Sub-task 5 - Full transformer (GPT) and generation"


def _labels(c: Canvas, x: float, y: float, chars: list[str], cw: float, ch: float, top: bool = True, left: bool = True) -> None:
    """Token names above the columns and left of the rows of a grid whose top-left corner is ``(x, y)``."""
    for i, ch_ in enumerate(chars):
        name = fonts.visible_token(ch_)
        if top:
            c.text(x + i * cw + cw / 2, y - 8, name, 11, fonts.MONO, theme.INK, "mm")
        if left:
            c.text(x - 6, y + i * ch + ch / 2, name, 11, fonts.MONO, theme.INK, "rm")


def _rows(values, upto: float):
    """First ``upto`` (0..1) share of the rows of a matrix as nested lists."""
    n = len(values)
    k = max(0, min(n, int(upto * n + 0.999)))
    return [[float(v) for v in row] for row in values[:k]]


def _vmax(t: torch.Tensor) -> float:
    finite = t[torch.isfinite(t)]
    return float(finite.abs().max()) if finite.numel() else 1.0


# --------------------------------------------------------------------------- #
# 4. Attention, LayerNorm, feed-forward
# --------------------------------------------------------------------------- #
def draw_attention(c: Canvas, ctx: Context, step: int, t: float) -> None:
    chars = list(ctx.attn_text)
    n = len(chars)
    d = ctx.trained_details["blocks"][0]["attention"]
    q, k, v = d["q"][0, 0], d["k"][0, 0], d["v"][0, 0]  # head 0: (T, d_k)
    d_k = q.size(-1)
    raw = q @ k.T / math.sqrt(d_k)
    weights = d["weights"][0, 0]
    a = phase(step, t, step)  # progress of the current step
    c.text(RIGHT, 376, f"trained model ({ctx.opt.train_steps} steps), block 1" + (", head 1" if step < 4 else ""), 11, fonts.SANS, theme.MUTED, "ra")

    if step == 0:  # Q, K, V
        cell = (11.0, 22.0)
        for idx, (name, mat, col, role) in enumerate((("Q  query", q, theme.QUERY, "what am I looking for?"),
                                                      ("K  key", k, theme.TOKEN, "what do I contain?"),
                                                      ("V  value", v, theme.VALUE, "what do I give away?"))):
            x = 60 + idx * 250
            part = theme.segment(a, idx / 3.2, idx / 3.2 + 0.45)
            c.text(x, 96, name, 14, fonts.BOLD, col)
            c.text(x, 114, role, 11, fonts.SANS, theme.MUTED)
            if part > 0:
                _labels(c, x, 150, chars, cell[0], cell[1], top=False)
                c.matrix(x, 150, _rows(mat, part), cell, lambda val, vm=_vmax(mat), cc=col: theme.mix((255, 255, 255), cc, abs(val) / vm))
        parts.note(c, 60, 346, f"Three learned projections of the same input: every token gets a query, a key and a value of {d_k} numbers (one head).", 12, theme.INK)
    elif step in (1, 2, 3):
        cell = (34.0, 28.0)
        gx, gy = 150, 112
        _labels(c, gx, gy, chars, cell[0], cell[1])
        c.text(gx + n * cell[0] / 2, gy - 28, "looks at (key)", 11, fonts.SANS, theme.MUTED, "mm")
        c.text(gx - 58, gy + n * cell[1] / 2, "token", 11, fonts.SANS, theme.MUTED, "mm")
        mask_t = theme.segment(t, 0.0, 1.0) if step == 2 else (1.0 if step == 3 else 0.0)
        upper = [[j > i for j in range(n)] for i in range(n)]
        if step == 1:
            cells = int(a * n * n)
            vals = [[float(raw[i, j]) if i * n + j < cells else None for j in range(n)] for i in range(n)]
            vm = _vmax(raw)
            c.matrix(gx, gy, vals, cell, lambda val: theme.diverging(val, vm), ".1f", 9, masked=(244, 246, 249))
        elif step == 2:
            vm = _vmax(raw)
            vals = [[float(raw[i, j]) if not (upper[i][j] and mask_t > 0.5) else None for j in range(n)] for i in range(n)]
            c.matrix(gx, gy, vals, cell, lambda val: theme.diverging(val, vm), ".1f", 9, masked=(200, 205, 212))
            if mask_t > 0.5:
                for i in range(n):
                    for j in range(i + 1, n):
                        c.text(gx + j * cell[0] + cell[0] / 2 - 0.5, gy + i * cell[1] + cell[1] / 2 - 0.5, "-inf", 8, fonts.MONO, theme.PANEL, "mm")
        else:
            c.matrix(gx, gy, [[float(weights[i, j]) if not upper[i][j] else None for j in range(n)] for i in range(n)], cell,
                     lambda val: theme.heat(val, 0.6), ".2f", 8, masked=(200, 205, 212))
            # the last row as bars on the right
            last = [float(weights[n - 1, j]) for j in range(n)]
            bx = 470
            c.text(bx, 100, f"Token {n - 1} ('{fonts.visible_token(chars[-1])}'): where does it look?", 12, fonts.BOLD, theme.INK)
            parts.bars(c, bx, 120, [f"{j}:{fonts.visible_token(ch)}" for j, ch in enumerate(chars)], last, 150, 22, theme.QUERY, 0.6, a, 12, ".2f", 36)
            c.text(bx, 120 + n * 22 + 6, f"row sum = {sum(last):.2f}", 12, fonts.MONO, theme.VALUE)
        right = {
            1: ["score = (q . k) / sqrt(d_k)", f"example: q[{n - 1}] . k[0] / {math.sqrt(d_k):.0f} = {float(raw[n - 1, 0]):+.2f}", "orange = high, blue = low"],
            2: ["a token may not look into the future", "masked cells become -inf", "so softmax gives them weight 0"],
        }.get(step)
        if right:
            for i, line in enumerate(right):
                c.text(470, 130 + i * 24, line, 13 if i == 0 else 12, fonts.BOLD if i == 0 else fonts.SANS, theme.INK if i == 0 else theme.MUTED)
    elif step == 4:  # heads and output
        cell = (17.0, 15.0)
        for h in range(4):
            x = 34 + h * 150
            part = theme.segment(a, h / 5, h / 5 + 0.4)
            c.text(x, 100, f"head {h + 1}", 12, fonts.BOLD, theme.INK)
            if part > 0:
                w = ctx.trained_details["blocks"][0]["attention"]["weights"][0, h]
                c.matrix(x, 118, [[float(w[i, j]) if j <= i else None for j in range(n)] for i in range(n)], cell,
                         lambda val: theme.heat(val, 0.6), masked=(226, 230, 236))
        if a > 0.7:
            e = theme.segment(a, 0.7, 1.0)
            c.arrow((640, 180), (680, 180), parts.fade(theme.MUTED, e), 2)
            c.box(684, 150, 92, 60, "concat, W_o", fill=parts.fade(theme.tint(theme.VALUE, 0.2), e), outline=parts.fade(theme.VALUE, e), size=12, kind=fonts.MONO)
        parts.note(c, 34, 280, f"{ctx.model.cfg.n_heads} heads attend in parallel, each with its own Q, K and V.", 12, theme.INK)
        parts.note(c, 34, 300, "After this short training the four patterns are still similar; with more training heads can specialise.", 12, theme.MUTED)
        parts.note(c, 34, 320, "Their outputs are concatenated and mixed by one more linear layer (W_o).", 12, theme.MUTED)
    elif step == 5:  # LayerNorm
        det = ctx.trained_details["blocks"][0]
        idx = n - 1
        before, after = det["input"][0, idx], det["ln1"][0, idx]
        c.text(LEFT, 92, f"One token vector ('{fonts.visible_token(chars[idx])}', {before.numel()} numbers) before and after LayerNorm", 13, fonts.BOLD, theme.INK)
        for which, (vec, label, col, x0) in enumerate(((before, "before", theme.TOKEN, 40), (after, "after", theme.VALUE, 420))):
            box = (x0, 130, x0 + 340, 270)
            c.axes(box, label)
            shown = vec if which == 0 else vec * theme.ease_out(a) + before * (1 - theme.ease_out(a))
            vm = max(float(before.abs().max()), float(after.abs().max()))
            mid = (box[1] + box[3]) / 2
            c.line([(box[0], mid), (box[2], mid)], theme.MUTED, 1)
            bw = (box[2] - box[0]) / shown.numel()
            for i, val in enumerate(shown.tolist()):
                hh = (box[3] - box[1]) / 2 * val / vm * 0.95
                c.rect(box[0] + i * bw + 0.5, min(mid, mid - hh), box[0] + (i + 1) * bw - 0.5, max(mid, mid - hh), fill=col)
            m_ = float(vec.mean()) if which == 0 else float(shown.mean())
            s_ = float(vec.std(unbiased=False)) if which == 0 else float(shown.std(unbiased=False))
            c.text(x0, 282, f"mean {m_:+.3f}    std {s_:.3f}", 13, fonts.MONO, col)
        c.arrow((384, 200), (414, 200), theme.MUTED, 2)
        parts.note(c, LEFT, 318, "(x - mean) / std, then a learned scale and shift. The result is close to mean 0 and spread 1, so every token enters attention on the same scale.", 12, theme.INK)
    else:  # FFN
        det = ctx.trained_details["blocks"][0]
        idx = n - 1
        c.text(LEFT, 92, "Feed-forward network, applied to every token on its own", 13, fonts.BOLD, theme.INK)
        grids = ((det["ln2"][0, idx], 8, 8, 40, 128, 12.0, "input", f"{ctx.model.cfg.d_model}"),
                 (det["ffn_hidden"][0, idx], 16, 16, 232, 108, 11.0, "after GELU", f"{ctx.model.cfg.ffn_mult * ctx.model.cfg.d_model}"),
                 (det["ffn_out"][0, idx], 8, 8, 470, 128, 12.0, "output", f"{ctx.model.cfg.d_model}"))
        for gi, (vec, rows, cols, gx, gy, cs, name, size) in enumerate(grids):
            part = theme.segment(a, gi / 3.2, gi / 3.2 + 0.4)
            if part <= 0:
                continue
            vm = float(vec.abs().max()) or 1.0
            vals = [[float(vec[r * cols + cc]) for cc in range(cols)] for r in range(rows)]
            c.matrix(gx, gy, vals, (cs, cs), lambda val, vm=vm, p=part: parts.fade(theme.diverging(val, vm), p), gap=0.5)
            c.text(gx + cols * cs / 2, gy + rows * cs + 10, f"{name} ({size})", 12, fonts.BOLD, theme.INK, "mm")
        for x0, label in ((150, "Linear + GELU"), (410, "Linear")):
            if a > 0.3:
                c.arrow((x0, 190), (x0 + 60, 190), theme.MUTED, 2)
                c.text(x0 + 30, 176, label, 10, fonts.SANS, theme.MUTED, "mm")
        gx0, gy0, gx1, gy1 = 620, 112, 776, 220
        c.axes((gx0, gy0, gx1, gy1), "GELU vs ReLU")
        xs = torch.linspace(-2, 2, 80)
        box = (gx0, gy0, gx1, gy1)
        c.polyline(xs.tolist(), relu(xs).tolist(), box, (-2, 2), (-0.4, 2), theme.MUTED, 4)
        c.polyline(xs.tolist(), gelu(xs).tolist(), box, (-2, 2), (-0.4, 2), theme.QUERY, 2, upto=a)
        c.text(gx0 + 6, gy1 - 8, "grey ReLU, orange GELU", 10, fonts.SANS, theme.MUTED, "ls")
        parts.note(c, LEFT, 322, "Widen to 4x the model width, bend with GELU, narrow back. This is where most parameters of a block live.", 12, theme.INK)


ATTENTION = Scene(
    "04_attention_layers", 4, K4, "How a token looks at the others",
    ("Every token is projected into a query, a key and a value",
     "Scores: how well does each query match each key?",
     "The causal mask hides every token that comes later",
     "Softmax turns each row of scores into weights that sum to 1",
     "Several heads read the text in parallel; their results are concatenated",
     "LayerNorm rescales every token vector to about mean 0 and spread 1",
     "The feed-forward network widens, bends with GELU, and narrows again"),
    draw_attention, frames_per_step=10,
)


# --------------------------------------------------------------------------- #
# 5. The full model
# --------------------------------------------------------------------------- #
BOX_X = 56.0


def _stack(c: Canvas, stages, lit: float, residual: bool = False) -> None:
    n = len(stages)
    h, gap, y0 = 32.0, 8.0, 84.0
    for i, (name, desc, shape) in enumerate(stages):
        y = y0 + i * (h + gap)
        on = theme.clamp01(lit - i)
        fill = theme.mix(theme.PANEL, theme.tint(theme.TOKEN, 0.35), on)
        c.box(BOX_X, y, 280, h, name, fill=fill, outline=theme.mix(theme.GRID, theme.TOKEN, on), size=13, radius=7)
        c.text(BOX_X + 292, y + h / 2, str(tuple(shape)), 12, fonts.MONO, theme.ACCENT, "lm")
        c.text(BOX_X + 410, y + h / 2, desc, 11, fonts.SANS, theme.MUTED if on < 1 else theme.INK, "lm")
        if i < n - 1:
            c.arrow((BOX_X + 140, y + h), (BOX_X + 140, y + h + gap), theme.MUTED, 1.5, 5)
    if residual and n >= 7:  # skip connections around the two sub-layers
        for src, dst in ((0, 3), (3, 6)):
            ys, yd = y0 + src * (h + gap) + h / 2, y0 + dst * (h + gap) + h / 2
            if lit > dst:
                c.line([(BOX_X, ys), (30, ys), (30, yd), (BOX_X - 5, yd)], theme.ACCENT, 2)
                c.polygon([(BOX_X, yd), (BOX_X - 9, yd - 4), (BOX_X - 9, yd + 4)], theme.ACCENT)


def draw_gpt(c: Canvas, ctx: Context, step: int, t: float) -> None:
    cfg = ctx.model.cfg
    a = phase(step, t, step)
    if step == 0:
        stages = model_stages(cfg, 1, len(ctx.attn_text))
        _stack(c, stages, a * (len(stages) + 0.2))
        c.text(RIGHT, 84, f"shapes for the {len(ctx.attn_text)}-token example", 11, fonts.SANS, theme.MUTED, "ra")
    elif step == 1:
        stages = block_stages(cfg, 1, len(ctx.attn_text))
        _stack(c, stages, a * (len(stages) + 0.2) + 0.0, residual=True)
        c.text(RIGHT, 84, "purple: the residual stream", 11, fonts.BOLD, theme.ACCENT, "ra")
    elif step == 2:
        bd = {k: v for k, v in ctx.breakdown.items() if k != "total"}
        total = ctx.breakdown["total"]
        c.text(LEFT, 90, f"Where do the {total:,} parameters live?", 14, fonts.BOLD, theme.INK)
        peak = max(bd.values())
        for i, (name, count) in enumerate(bd.items()):
            y = 124 + i * 36
            frac = theme.ease_out(theme.segment(a, i / 8, i / 8 + 0.5))
            c.text(LEFT, y + 10, name.replace("_", " "), 13, fonts.SANS, theme.INK, "lm")
            c.bar(190, y, 440, 20, count / peak * frac, theme.TOKEN)
            c.text(640, y + 10, f"{int(count * frac):,}  ({count / total:.1%})", 12, fonts.MONO, theme.MUTED, "lm")
        if cfg.tie_weights:
            parts.note(c, LEFT, 352, "output head = 0: it re-uses the token embedding matrix (tied weights), so it is counted once.", 12, theme.INK)
    else:  # loss
        probs = ctx.untrained_probs
        tgt = ctx.target_id
        c.text(LEFT, 90, f"After '{ctx.phrase[-12:]}' the untrained model rates every next character:", 13, fonts.BOLD, theme.INK)
        best = top_k(probs, 6)
        labels = [ctx.char_tok.itos[i] for i, _ in best] + [ctx.char_tok.itos[tgt]]
        values = [p for _, p in best] + [float(probs[tgt])]
        parts.bars(c, 40, 118, labels, values, 220, 30, theme.TOKEN, max(values) * 1.05, theme.ease_out(a), 13, ".3f", 40, highlight=len(values) - 1)
        c.text(40, 118 + 7 * 30 + 4, "(orange: the character that really follows - here a space)", 11, fonts.SANS, theme.MUTED)
        p_t = float(probs[tgt])
        uniform = math.log(cfg.vocab_size)
        x = 470
        e = theme.segment(a, 0.4, 1.0)
        c.box(x, 118, 306, 72, "", fill=theme.PANEL, outline=theme.LOSS, radius=8)
        c.text(x + 12, 132, "loss = -ln p(correct)", 13, fonts.BOLD, theme.INK)
        c.text(x + 12, 158, f"-ln({p_t:.3f}) = {-math.log(p_t) * e:.2f}", 18, fonts.MONO, theme.LOSS)
        c.box(x, 204, 306, 68, "", fill=theme.PANEL, outline=theme.GRID, radius=8)
        c.text(x + 12, 216, "pure guessing among all characters", 12, fonts.SANS, theme.MUTED)
        c.text(x + 12, 240, f"ln({cfg.vocab_size}) = {uniform:.2f}", 16, fonts.MONO, theme.INK)
        parts.note(c, x, 288, "Training pushes the loss below this line.", 12, theme.INK)


GPT_SCENE = Scene(
    "05_gpt_model", 5, K5, "Assembling the transformer",
    ("Ids flow through embedding, position, blocks and an output head",
     "Inside a block: attention and feed-forward, each added back onto the residual stream",
     "Count the parameters: the output head shares the embedding matrix",
     "Cross-entropy compares the predicted distribution with the true next character"),
    draw_gpt, frames_per_step=12,
)


# --------------------------------------------------------------------------- #
# 5b. Generation
# --------------------------------------------------------------------------- #
def draw_generation(c: Canvas, ctx: Context, step: int, t: float) -> None:
    logits = ctx.trained_logits
    itos = ctx.char_tok.itos
    a = phase(step, t, step)
    shown = 10
    if step == 0:
        probs = torch.softmax(logits, dim=-1)
        best = top_k(probs, shown)
        c.text(LEFT, 90, f"Model reads '{ctx.phrase}' and scores every one of the {len(itos)} characters", 13, fonts.BOLD, theme.INK)
        parts.bars(c, 60, 116, [itos[i] for i, _ in best], [p for _, p in best], 360, 24, theme.TOKEN, best[0][1] * 1.05, theme.ease_out(a), 13, ".3f", 40)
        parts.note(c, 60, 116 + shown * 24 + 8, "logits -> softmax -> a probability for every possible next character", 12, theme.MUTED)
    elif step == 1:
        c.text(LEFT, 90, "Temperature divides the logits before softmax", 13, fonts.BOLD, theme.INK)
        temps = (0.5, 1.0, 2.0)
        base = top_k(torch.softmax(logits, dim=-1), 6)
        ids = [i for i, _ in base]
        for col, temp in enumerate(temps):
            p = next_token_distribution(logits, temp)
            x = 40 + col * 250
            c.text(x, 116, f"T = {temp}", 14, fonts.BOLD, (theme.TOKEN, theme.VALUE, theme.QUERY)[col])
            c.text(x, 134, ("sharper: safe, repetitive", "as trained", "flatter: surprising")[col], 11, fonts.SANS, theme.MUTED)
            show = theme.segment(a, col / 3.2, col / 3.2 + 0.45)
            parts.bars(c, x, 154, [itos[i] for i in ids], [float(p[i]) for i in ids], 130, 28, (theme.TOKEN, theme.VALUE, theme.QUERY)[col], 0.35, show, 12, ".2f", 28)
    elif step == 2:
        c.text(LEFT, 90, "Top-k keeps only the k best characters, then renormalises", 13, fonts.BOLD, theme.INK)
        kk = 5
        full = torch.softmax(logits, dim=-1)
        cut = next_token_distribution(logits, 1.0, top_k=kk)
        order = top_k(full, 9)
        ids = [i for i, _ in order]
        parts.bars(c, 40, 120, [itos[i] for i in ids], [float(full[i]) for i in ids], 220, 26, theme.GRID, 0.4, 1.0, 12, ".2f", 28, show_values=False)
        parts.bars(c, 40, 120, [itos[i] for i in ids], [float(cut[i]) for i in ids], 220, 26, theme.TOKEN, 0.4, theme.ease_out(a), 12, ".2f", 28)
        c.text(470, 130, f"k = {kk}", 18, fonts.BOLD, theme.TOKEN)
        c.text(470, 160, "grey: before", 12, fonts.SANS, theme.MUTED)
        c.text(470, 180, "blue: after the cut", 12, fonts.SANS, theme.TOKEN)
        c.text(470, 210, "sample from what is left", 13, fonts.BOLD, theme.INK)
    else:
        gen_ids = ctx.generated
        n_new = max(1, int(a * len(gen_ids) + 0.5))
        c.text(LEFT, 90, "Loop: predict, sample, append, repeat", 13, fonts.BOLD, theme.INK)
        prompt = list(ctx.phrase)
        new = [itos[i] for i in gen_ids[:n_new]]
        chips = prompt + new
        per_line = 22
        for line in range(math.ceil(len(chips) / per_line)):
            seg = chips[line * per_line:(line + 1) * per_line]
            x = LEFT + 4
            for j, ch in enumerate(seg):
                idx = line * per_line + j
                is_new = idx >= len(prompt)
                w = c.token(x, 118 + line * 36, ch, theme.QUERY if is_new else theme.TOKEN, 14, 5, 28, 24)
                x += w + 3
        c.text(LEFT, 118 + 36 * 3 + 12, f"{n_new} of {len(gen_ids)} new characters", 12, fonts.MONO, theme.MUTED)
        parts.note(c, LEFT, 118 + 36 * 3 + 36, f"After only {ctx.opt.train_steps} steps the text is noise with realistic letter frequencies - the model has not learned words yet.", 12, theme.INK)


GENERATION = Scene(
    "05_gpt_model_generation", 5, K5, "Inference: from probabilities to text",
    ("The last position's logits become a probability for every character",
     "Temperature sharpens or flattens the distribution",
     "Top-k removes the unlikely characters",
     "Sample one character, append it, and run the model again"),
    draw_generation, frames_per_step=12,
)

SCENES = (ATTENTION, GPT_SCENE, GENERATION)
