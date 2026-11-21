"""Scenes 6-7: the training pipeline and the visualisation layer."""
from __future__ import annotations

import math

from animgen import fonts, parts, theme
from animgen.canvas import LEFT, RIGHT, Canvas
from animgen.context import Context
from animgen.scene import Scene, phase
from training.schedulers import warmup_cosine_lr
from viz.embeddings import pca_2d

K6 = "Sub-task 6 - Training pipeline"
K7 = "Sub-task 7 - Visualization layer"


def clip_text(c: Canvas, s: str, width: float, size: float = 12, kind: str = fonts.SANS) -> str:
    """``s`` shortened with ``...`` so that it fits ``width`` design units."""
    if c.text_width(s, size, kind) <= width:
        return s
    while s and c.text_width(s + "...", size, kind) > width:
        s = s[:-1]
    return s + "..."


def _loss_range(history: list[dict]) -> tuple[float, float]:
    values = [h["train_loss"] for h in history] + [h["val_loss"] for h in history if h["val_loss"] is not None]
    return min(values) - 0.1, max(values) + 0.1


# --------------------------------------------------------------------------- #
# 6. Training pipeline
# --------------------------------------------------------------------------- #
def _flow(c: Canvas, labels: list[str], colours: list, y: float, shown: float, reverse: bool = False,
          w: float = 118, h: float = 44) -> None:
    """A row of boxes joined by arrows; ``shown`` (0..n) reveals them; ``reverse`` flows right to left."""
    n = len(labels)
    gap = (RIGHT - LEFT - n * w) / (n - 1)
    order = range(n - 1, -1, -1) if reverse else range(n)
    for rank, i in enumerate(order):
        amount = theme.clamp01(shown - rank)
        if amount <= 0:
            continue
        x = LEFT + i * (w + gap)
        c.box(x, y, w, h, labels[i], fill=parts.fade(theme.tint(colours[i], 0.2), amount), outline=parts.fade(colours[i], amount),
              size=12, text_fill=parts.fade(theme.INK, amount))
        if rank > 0:
            nxt = i + 1 if reverse else i - 1
            xa = LEFT + nxt * (w + gap)
            if reverse:
                c.arrow((xa - 4, y + h / 2), (x + w + 4, y + h / 2), parts.fade(theme.LOSS, amount), 2.4)
            else:
                c.arrow((xa + w + 4, y + h / 2), (x - 4, y + h / 2), parts.fade(theme.MUTED, amount), 2.4)


def draw_training(c: Canvas, ctx: Context, step: int, t: float) -> None:
    a = phase(step, t, step)
    hist = ctx.history
    itos = ctx.char_tok.itos
    cfg = ctx.train_cfg
    if step == 0:  # data loading: windows and the shifted target
        x, y = ctx.batch
        c.text(LEFT, 90, "Cut random windows from the corpus; the target is the input shifted left by one", 13, fonts.BOLD, theme.INK)
        j = 5
        for row in range(2):
            top = 122 + row * 104
            xs = [itos[i] for i in x[row].tolist()]
            ys = [itos[i] for i in y[row].tolist()]
            c.text(LEFT, top + 14, "input x", 13, fonts.BOLD, theme.TOKEN, "lm")
            c.text(LEFT, top + 54, "target y", 13, fonts.BOLD, theme.VALUE, "lm")
            xb = parts.chip_row(c, 130, top, xs, theme.TOKEN, 15, 28, 4, 30, a * (len(xs) + 1))
            parts.chip_row(c, 130, top + 40, ys, theme.VALUE, 15, 28, 4, 30, a * (len(ys) + 1))
            if row == 0 and a > 0.7:
                x0, w0 = xb[j]
                c.rect(x0 - 2, top - 2, x0 + w0 + 2, top + 70, outline=theme.QUERY, width=2.2, radius=6)
                seen = "".join(xs[: j + 1])
                c.text(130, top + 84, f"seeing '{fonts.visible_token(seen)}' the model must predict '{fonts.visible_token(ys[j])}'",
                       12, fonts.MONO, theme.QUERY, "lm")
        parts.note(c, LEFT, 342, f"drawn: 12-character windows. A real step uses {cfg.batch_size} windows x {cfg.block_size} characters; every position is its own exercise", 12)
    elif step in (1, 2):  # forward and loss, then backpropagation
        labels = ["x  (B, T)", "GPT model", "logits (B, T, V)", "cross-entropy", "loss"]
        cols = [theme.TOKEN, theme.ACCENT, theme.TOKEN, theme.LOSS, theme.LOSS]
        if step == 1:
            c.text(LEFT, 90, "Forward pass: ids in, one score per character out, then one number: the loss", 13, fonts.BOLD, theme.INK)
            _flow(c, labels, cols, 130, a * 5.5)
            if a > 0.2:
                cx = LEFT + 3 * (118 + (RIGHT - LEFT - 5 * 118) / 4) + 59  # centre of the cross-entropy box
                c.text(cx, 214, "y (targets)", 12, fonts.BOLD, theme.VALUE, "mm")
                c.arrow((cx, 204), (cx, 178), theme.VALUE, 2)
            loss0 = hist[0]["train_loss"]
            show = theme.segment(a, 0.7, 1.0)
            c.rect(LEFT, 250, RIGHT, 340, fill=theme.PANEL, outline=parts.fade(theme.LOSS, show), radius=8, width=1.8)
            c.text(LEFT + 20, 275, "loss of the very first step", 13, fonts.BOLD, parts.fade(theme.INK, show), "lm")
            c.text(LEFT + 20, 308, f"{loss0:.2f}", 26, fonts.BOLD, parts.fade(theme.LOSS, show), "lm")
            c.text(LEFT + 130, 298, f"pure guessing among {len(itos)} characters would give ln({len(itos)}) = {math.log(len(itos)):.2f}",
                   12, fonts.SANS, parts.fade(theme.MUTED, show), "lm")
            c.text(LEFT + 130, 318, "an untrained model knows nothing, so its loss starts near that value", 12, fonts.SANS,
                   parts.fade(theme.MUTED, show), "lm")
        else:
            d = ctx.adam_demo
            c.text(LEFT, 90, "Backpropagation: the chain rule sends the blame back through every layer", 13, fonts.BOLD, theme.INK)
            _flow(c, labels, cols, 130, a * 5.5, reverse=True)
            show = theme.segment(a, 0.7, 1.0)
            c.rect(LEFT, 220, RIGHT, 340, fill=theme.PANEL, outline=parts.fade(theme.LOSS, show), radius=8, width=1.8)
            c.text(LEFT + 20, 244, "every weight receives a gradient: dL/dw", 13, fonts.BOLD, parts.fade(theme.INK, show), "lm")
            c.text(LEFT + 20, 276, d["where"], 13, fonts.MONO, parts.fade(theme.MUTED, show), "lm")
            c.text(LEFT + 20, 308, f"dL/dw = {d['grad']:.2e}", 22, fonts.MONO, parts.fade(theme.LOSS, show), "lm")
            c.text(LEFT + 330, 300, "negative: raising w would lower the loss", 12, fonts.SANS, parts.fade(theme.MUTED, show), "lm")
            c.text(LEFT + 330, 320, f"{ctx.n_params:,} weights, each with its own gradient", 12, fonts.SANS, parts.fade(theme.MUTED, show), "lm")
    elif step == 3:  # one hand-checked Adam update
        d = ctx.adam_demo
        b1, b2 = d["beta1"], d["beta2"]
        m_hat, v_hat = d["m"] / (1 - b1), d["v"] / (1 - b2)
        rows = [
            ("gradient", f"g  = {d['grad']:.3e}", theme.LOSS),
            ("momentum", f"m  = {b1}*0 + {1 - b1:.1f}*g = {d['m']:.3e}", theme.TOKEN),
            ("scale", f"v  = {b2}*0 + {1 - b2:.3f}*g^2 = {d['v']:.3e}", theme.ACCENT),
            ("bias fix", f"m^ = m/(1-{b1}) = {m_hat:.3e}     v^ = v/(1-{b2}) = {v_hat:.3e}", theme.MUTED),
            ("update", f"w  = {d['before']:.6f} - {d['lr']} * m^ / (sqrt(v^) + eps)", theme.QUERY),
            ("new weight", f"w  = {d['after']:.6f}", theme.VALUE),
        ]
        c.text(LEFT, 90, f"One real Adam update of {d['where']}", 13, fonts.BOLD, theme.INK)
        for i, (name, formula, col) in enumerate(rows):
            show = theme.segment(a, i / 6.5, i / 6.5 + 0.2)
            y = 118 + i * 38
            c.rect(LEFT, y, RIGHT, y + 30, fill=parts.fade(theme.tint(col, 0.14), show), outline=parts.fade(col, show), radius=6)
            c.text(LEFT + 12, y + 15, name, 12, fonts.BOLD, parts.fade(col, show), "lm")
            c.text(LEFT + 100, y + 15, clip_text(c, formula, RIGHT - LEFT - 112, 13, fonts.MONO), 13, fonts.MONO, parts.fade(theme.INK, show), "lm")
        parts.note(c, LEFT, 352, f"On step 1 Adam moves the weight by about lr = {d['lr']}, against the sign of its gradient.", 12, theme.INK)
    elif step == 4:  # learning-rate schedule and clipping
        box = (70, 112, 500, 330)
        c.axes(box, "learning rate per step", "training step", "lr")
        steps = list(range(cfg.steps))
        lrs = [warmup_cosine_lr(i, cfg.lr, cfg.warmup_steps, cfg.steps, cfg.min_lr_ratio) for i in steps]
        end = c.polyline(steps, lrs, box, (0, cfg.steps - 1), (0, cfg.lr * 1.05), theme.TOKEN, 2.6, a)
        if end:
            c.circle(end[0], end[1], 5, fill=theme.QUERY)
        wx = box[0] + (box[2] - box[0]) * cfg.warmup_steps / cfg.steps
        c.dashed((wx, box[1]), (wx, box[3]), theme.ACCENT)
        c.text(wx + 5, box[3] - 30, "warm-up ends", 11, fonts.SANS, theme.ACCENT)
        c.text(box[2] - 6, box[1] + 12, f"peak lr = {cfg.lr:g}", 11, fonts.MONO, theme.MUTED, "rm")
        c.text(540, 118, "Warm-up", 13, fonts.BOLD, theme.TOKEN)
        parts.paragraph(c, 540, 138, f"The rate climbs for {cfg.warmup_steps} steps so early, noisy gradients cannot throw the weights off.", 230, 12)
        c.text(540, 214, "Cosine decay", 13, fonts.BOLD, theme.TOKEN)
        parts.paragraph(c, 540, 234, f"Then it glides down to {cfg.min_lr_ratio:g} x the peak for fine adjustments.", 230, 12)
        c.text(540, 292, "Gradient clipping", 13, fonts.BOLD, theme.LOSS)
        biggest = max(h["grad_norm"] for h in hist)
        parts.paragraph(c, 540, 312, f"Gradients longer than {cfg.grad_clip:g} are scaled down (largest seen: {biggest:.2f}).", 230, 12)
    else:  # live loss curve
        box = (70, 108, 520, 336)
        lo, hi = _loss_range(hist)
        c.axes(box, "loss (lower is better)", "training step", "loss")
        n = len(hist)
        xs = [h["step"] for h in hist]
        for value, label, col in ((math.log(len(itos)), f"ln({len(itos)}) = random guessing", theme.MUTED),
                                  (ctx.unigram_entropy, "letter frequencies only", theme.ACCENT)):
            if lo < value < hi:
                yy = box[3] - (box[3] - box[1]) * (value - lo) / (hi - lo)
                c.dashed((box[0], yy), (box[2], yy), col)
        end = c.polyline(xs, [h["train_loss"] for h in hist], box, (1, max(2, xs[-1])), (lo, hi), theme.LOSS, 2.4, a)
        upto = max(1, round(n * a))
        for h in hist[:upto]:
            if h["val_loss"] is not None:
                px = box[0] + (box[2] - box[0]) * (h["step"] - 1) / max(1, xs[-1] - 1)
                py = box[3] - (box[3] - box[1]) * theme.clamp01((h["val_loss"] - lo) / (hi - lo))
                c.circle(px, py, 3.5, fill=theme.VALUE)
        if end:
            c.circle(end[0], end[1], 5, fill=theme.QUERY)
        cur = hist[upto - 1]
        c.text(550, 114, "live metrics", 13, fonts.BOLD, theme.INK)
        c.text(550, 138, f"step        {cur['step']}", 13, fonts.MONO, theme.INK)
        c.text(550, 160, f"train loss  {cur['train_loss']:.3f}", 13, fonts.MONO, theme.LOSS)
        last_val = next((h["val_loss"] for h in reversed(hist[:upto]) if h["val_loss"] is not None), None)
        c.text(550, 182, "val loss    " + (f"{last_val:.3f}" if last_val is not None else "-"), 13, fonts.MONO, theme.VALUE)
        c.text(550, 204, f"lr          {cur['lr']:.5f}", 13, fonts.MONO, theme.TOKEN)
        c.text(550, 226, f"grad norm   {cur['grad_norm']:.2f}", 13, fonts.MONO, theme.MUTED)
        c.text(550, 248, f"tokens/s    {cur['tokens_per_sec']:.0f}", 13, fonts.MONO, theme.MUTED)
        parts.paragraph(c, 550, 272, "red line: training loss; green dots: validation loss on unseen text", 220, 11, fill=theme.MUTED)
        for k, (value, label, col) in enumerate(((math.log(len(itos)), f"ln({len(itos)}) = {math.log(len(itos)):.2f}: random guessing", theme.MUTED),
                                                 (ctx.unigram_entropy, f"{ctx.unigram_entropy:.2f}: letter frequencies only", theme.ACCENT))):
            c.dashed((550, 318 + k * 18), (576, 318 + k * 18), col, 1.8)
            c.text(584, 318 + k * 18, label, 11, fonts.SANS, col, "lm")
        parts.note(c, 550, 360, "the curve settles near the letter-frequency floor", 11, theme.INK)


TRAINING = Scene(
    "06_training_pipeline", 6, K6, "Training: data, loss, gradients, Adam",
    ("Data loader: random windows, with the target shifted by one character",
     "Forward pass and cross-entropy turn predictions into one loss number",
     "Backpropagation gives every weight a gradient",
     "Adam turns a gradient into a small, scaled weight update",
     "The learning rate warms up, decays, and gradients are clipped",
     "Repeat for every step: the loss curve falls as the model learns"),
    draw_training, frames_per_step=12,
)


# --------------------------------------------------------------------------- #
# 7. Visualisation layer
# --------------------------------------------------------------------------- #
def _vowel_class(ch: str) -> tuple[str, theme.RGB]:
    if ch in "aeiouAEIOU":
        return "vowel", theme.VALUE
    if ch.isalpha():
        return "consonant", theme.TOKEN
    return "space / punctuation", theme.ACCENT


def draw_visualization(c: Canvas, ctx: Context, step: int, t: float) -> None:
    a = phase(step, t, step)
    if step == 0:  # tensors -> viz functions -> figures
        rows = [("token ids", "viz.tokens.tokens_html", "coloured token chips"),
                ("embedding matrix", "viz.embeddings.embedding_scatter", "2-D map of the vocabulary"),
                ("attention weights", "viz.attention.heads_grid", "one heat-map per head"),
                ("training history", "viz.training.loss_figure", "live loss curves")]
        cols = [theme.TOKEN, theme.ACCENT, theme.QUERY, theme.LOSS]
        c.text(LEFT, 90, "Pure functions turn model tensors into Plotly figures; the pages only display them", 13, fonts.BOLD, theme.INK)
        for i, ((src, fn, out), col) in enumerate(zip(rows, cols)):
            show = theme.segment(a, i / 4.5, i / 4.5 + 0.3)
            y = 122 + i * 62
            f = parts.fade
            c.box(LEFT, y, 190, 42, src, fill=f(theme.tint(col, 0.18), show), outline=f(col, show), size=13, text_fill=f(theme.INK, show))
            c.arrow((LEFT + 196, y + 21), (LEFT + 256, y + 21), f(theme.MUTED, show), 2.2)
            c.box(LEFT + 262, y, 280, 42, fn, fill=f(theme.PANEL, show), outline=f(theme.MUTED, show), size=12, kind=fonts.MONO,
                  text_fill=f(theme.INK, show))
            c.arrow((LEFT + 548, y + 21), (LEFT + 600, y + 21), f(theme.MUTED, show), 2.2)
            c.box(LEFT + 606, y, 146, 42, out, fill=f(theme.tint(col, 0.18), show), outline=f(col, show), size=11, text_fill=f(theme.INK, show))
        parts.note(c, LEFT, 372, "No Streamlit inside viz/: each function returns a figure, so it can be tested without a browser.", 12)
    elif step == 1:  # PCA of the trained embedding matrix
        emb = ctx.trained.tok_emb.weight
        coords, expl = pca_2d(emb)
        box = (60, 104, 470, 350)
        c.axes(box, "token embeddings, projected to 2-D (PCA)", "PC1", "PC2")
        lim_x = float(coords[:, 0].abs().max()) * 1.1 or 1.0
        lim_y = float(coords[:, 1].abs().max()) * 1.1 or 1.0  # each axis has its own scale so the points spread out
        itos = ctx.char_tok.itos
        n = len(itos)
        for i, ch in enumerate(itos):
            show = theme.segment(a, i / (n + 4), i / (n + 4) + 0.15)
            if show <= 0:
                continue
            _, col = _vowel_class(ch)
            px = (box[0] + box[2]) / 2 + coords[i, 0].item() / lim_x * (box[2] - box[0]) / 2
            py = (box[1] + box[3]) / 2 - coords[i, 1].item() / lim_y * (box[3] - box[1]) / 2
            c.text(px, py, fonts.visible_token(ch), 15, fonts.MONO, parts.fade(col, show), "mm")
        c.text(500, 118, "colour = kind of character", 12, fonts.BOLD, theme.INK)
        for k, (name, col) in enumerate((("vowel", theme.VALUE), ("consonant", theme.TOKEN), ("space / punctuation", theme.ACCENT))):
            c.circle(510, 148 + k * 24, 5, fill=col)
            c.text(522, 148 + k * 24, name, 12, fonts.SANS, theme.INK, "lm")
        parts.paragraph(c, 500, 232, f"PC1 and PC2 keep {100 * float(expl.sum()):.0f}% of the variance of the {emb.size(1)}-dimensional vectors.", 270, 12)
        parts.paragraph(c, 500, 268, "Each axis is stretched to fill the plot, so distances are only roughly comparable.", 270, 12, fill=theme.MUTED)
        parts.paragraph(c, 500, 310, f"After only {ctx.opt.train_steps} steps the groups are still faint; longer training separates them.", 270, 12, fill=theme.MUTED)
    elif step == 2:  # attention heads
        w = ctx.trained_details["blocks"][0]["attention"]["weights"][0]
        chars = list(ctx.attn_text)
        n = len(chars)
        cell = (17.0, 17.0)
        c.text(LEFT, 90, "One heat-map per head: row = who looks, column = who is looked at", 13, fonts.BOLD, theme.INK)
        for h in range(w.size(0)):
            x0 = 62 + h * 180
            show = theme.segment(a, h / 5, h / 5 + 0.35)
            c.text(x0 + n * cell[0] / 2, 118, f"head {h + 1}", 12, fonts.BOLD, parts.fade(theme.QUERY, show), "mm")
            rows = [[(float(w[h, r, k]) if k <= r else None) for k in range(n)] for r in range(n)]
            shown_rows = [[(None if v is None else v * show) for v in row] for row in rows]
            c.matrix(x0, 150, shown_rows, cell, lambda v: theme.heat(v, 1.0, high=theme.QUERY))
            for i, ch in enumerate(chars):
                name = fonts.visible_token(ch)
                c.text(x0 + i * cell[0] + cell[0] / 2, 142, name, 10, fonts.MONO, theme.INK, "mm")
                c.text(x0 - 4, 150 + i * cell[1] + cell[1] / 2, name, 10, fonts.MONO, theme.INK, "rm")
        parts.paragraph(c, LEFT, 312, "All four heads show the same smooth fan after so little training; with longer runs they start to differ. "
                        "The blank upper triangle is the causal mask: no peeking ahead.", RIGHT - LEFT, 12, fill=theme.INK)
    else:  # training curves
        hist = ctx.history
        xs = [h["step"] for h in hist]
        panels = (("loss", [h["train_loss"] for h in hist], theme.LOSS), ("learning rate", [h["lr"] for h in hist], theme.TOKEN),
                  ("gradient norm", [h["grad_norm"] for h in hist], theme.ACCENT))
        c.text(LEFT, 90, "The training lab redraws these figures while the model learns", 13, fonts.BOLD, theme.INK)
        for i, (name, ys, col) in enumerate(panels):
            box = (50 + i * 250, 130, 250 + i * 250 - 14, 320)
            c.axes(box, name, "step")
            lo, hi = min(ys), max(ys)
            pad = (hi - lo) * 0.08 or 1.0
            end = c.polyline(xs, ys, box, (1, max(2, xs[-1])), (lo - pad, hi + pad), col, 2.4, a)
            if end:
                c.circle(end[0], end[1], 4.5, fill=theme.QUERY)
        parts.note(c, LEFT, 358, "Loss, learning rate and gradient norm are all taken from the metrics dictionaries yielded by the trainer.", 12)


VISUALIZATION = Scene(
    "07_visualization", 7, K7, "Turning tensors into pictures",
    ("Each viz function takes tensors and returns a figure",
     "PCA squeezes the embedding vectors into a 2-D map",
     "A grid of heat-maps shows what every attention head looks at",
     "Training curves update live from the trainer's metrics"),
    draw_visualization, frames_per_step=12,
)

SCENES = (TRAINING, VISUALIZATION)
