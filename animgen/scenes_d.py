"""Scenes 8-10: the educational content layer, the Streamlit app, and scripts / docs / compliance."""
from __future__ import annotations

from animgen import fonts, parts, theme
from animgen.canvas import LEFT, RIGHT, Canvas
from animgen.context import Context
from animgen.scene import Scene, phase
from animgen.scenes_c import clip_text
from edu.lessons import all_lessons, get_lesson
from edu.snippets import get_source

K8 = "Sub-task 8 - Educational content layer"
K9 = "Sub-task 9 - Streamlit application"
K10 = "Sub-task 10 - Scripts, documentation and compliance"

SCRIPT_ROLES = {
    "start.sh": "launch the app, print its URL", "stop.sh": "shut it down gracefully",
    "cleanup.sh": "remove .venv, caches", "build-docs.sh": "rebuild the 30 sub-task documents",
    "generate-animations.sh": "re-create these GIFs", "github-push.sh": "publish to GitHub",
}


def source_lines(path: str, count: int) -> list[str]:
    """The first ``count`` non-empty source lines of a symbol (read live, like the app does)."""
    return [ln for ln in get_source(path).splitlines() if ln.strip()][:count]


# --------------------------------------------------------------------------- #
# 8. Educational content
# --------------------------------------------------------------------------- #
def draw_education(c: Canvas, ctx: Context, step: int, t: float) -> None:
    a = phase(step, t, step)
    lessons = all_lessons()
    lesson = get_lesson("attention")
    if step == 0:
        c.text(LEFT, 90, f"{len(lessons)} lessons, in teaching order, are plain data (no Streamlit)", 13, fonts.BOLD, theme.INK)
        for i, ls in enumerate(lessons):
            r, col = divmod(i, 3)
            show = theme.segment(a, i / (len(lessons) + 3), i / (len(lessons) + 3) + 0.2)
            x, y = LEFT + col * 254, 118 + r * 50
            f = parts.fade
            c.box(x, y, 240, 40, "", fill=f(theme.tint(theme.TOKEN, 0.15), show), outline=f(theme.TOKEN, show), radius=6)
            c.text(x + 10, y + 20, f"{i + 1:>2}", 13, fonts.MONO, f(theme.TOKEN, show), "lm")
            c.text(x + 36, y + 20, clip_text(c, ls.title, 196, 13), 13, fonts.SANS, f(theme.INK, show), "lm")
    elif step == 1:
        c.text(LEFT, 90, f"Every lesson has the same parts - here: '{lesson.title}'", 13, fonts.BOLD, theme.INK)
        facts = [("intuition", "the idea in plain words", theme.TOKEN), ("steps", f"{len(lesson.steps)} one-sentence stages", theme.VALUE),
                 ("formulas", f"{len(lesson.formulas)} LaTeX formulas, each with a plain reading", theme.ACCENT),
                 ("shapes", f"{len(lesson.shapes)} tensor shapes", theme.QUERY),
                 ("pitfalls", f"{len(lesson.pitfalls)} common mistakes", theme.LOSS),
                 ("snippets", f"{len(lesson.snippets)} real functions of this project", theme.MUTED)]
        for i, (name, what, col) in enumerate(facts):
            r, k = divmod(i, 2)
            show = theme.segment(a, i / 7, i / 7 + 0.25)
            x, y = LEFT + k * 380, 122 + r * 80
            c.rect(x, y, x + 366, y + 66, fill=parts.fade(theme.tint(col, 0.15), show), outline=parts.fade(col, show), radius=8)
            c.text(x + 14, y + 22, name, 15, fonts.BOLD, parts.fade(col, show), "lm")
            c.text(x + 14, y + 46, what, 12, fonts.SANS, parts.fade(theme.INK, show), "lm")
    elif step == 2:
        c.text(LEFT, 90, "Math tab: the formula in words, and the shape of every tensor", 13, fonts.BOLD, theme.INK)
        f0 = lesson.formulas[0]
        c.rect(LEFT, 114, RIGHT, 190, fill=theme.PANEL, outline=theme.ACCENT, radius=8)
        c.text(400, 142, "softmax( Q K^T / sqrt(d_k) + M ) V", 22, fonts.MONO, theme.ACCENT, "mm")
        parts.paragraph(c, LEFT + 14, 164, "In words: " + f0.meaning, RIGHT - LEFT - 28, 12, fill=theme.INK)
        c.text(LEFT, 214, "tensor", 12, fonts.BOLD, theme.MUTED)
        c.text(LEFT + 130, 214, "shape", 12, fonts.BOLD, theme.MUTED)
        c.text(LEFT + 260, 214, "meaning", 12, fonts.BOLD, theme.MUTED)
        for i, (name, shape, meaning) in enumerate(lesson.shapes[:5]):
            show = theme.segment(a, i / 6, i / 6 + 0.25)
            y = 236 + i * 28
            c.line([(LEFT, y - 6), (RIGHT, y - 6)], parts.fade(theme.GRID, show), 1)
            c.text(LEFT, y + 6, name, 13, fonts.MONO, parts.fade(theme.TOKEN, show), "lm")
            c.text(LEFT + 130, y + 6, shape, 13, fonts.MONO, parts.fade(theme.QUERY, show), "lm")
            c.text(LEFT + 260, y + 6, clip_text(c, meaning, 480, 12), 12, fonts.SANS, parts.fade(theme.INK, show), "lm")
    else:
        path = "llm.layers.gelu"
        lines = source_lines(path, 9)
        c.text(LEFT, 90, f"Code tab: the real source of {path}, read with inspect", 13, fonts.BOLD, theme.INK)
        c.rect(LEFT, 112, RIGHT, 350, fill=(246, 248, 250), outline=theme.GRID, radius=8)
        shown = a * len(lines)
        for i, line in enumerate(lines):
            amount = theme.clamp01(shown - i)
            if amount <= 0:
                continue
            col = theme.MUTED if line.strip().startswith(("#", '"""')) else theme.INK
            c.text(LEFT + 16, 134 + i * 24, clip_text(c, line.rstrip(), RIGHT - LEFT - 40, 13, fonts.MONO), 13, fonts.MONO, parts.fade(col, amount), "lm")
        parts.note(c, LEFT, 362, "Because it is read live, the code shown can never drift away from the code that runs.", 12, theme.INK)


EDUCATION = Scene(
    "08_educational_content", 8, K8, "Lessons, formulas and live code",
    ("Thirteen lessons form the teaching path",
     "Each lesson has the same six parts",
     "The Math tab pairs every formula with a plain reading and tensor shapes",
     "The Code tab shows the project's own source, read at run time"),
    draw_education, frames_per_step=12,
)


# --------------------------------------------------------------------------- #
# 9. Streamlit application
# --------------------------------------------------------------------------- #
def _app_pages() -> list[tuple[str, str]]:
    import app  # the entry script only builds the page list on import

    return [(title, url) for _, title, _, url in app.PAGES]


def draw_streamlit(c: Canvas, ctx: Context, step: int, t: float) -> None:
    a = phase(step, t, step)
    facts = ctx.project
    if step == 0:
        pages = _app_pages()
        c.text(LEFT, 90, f"app.py registers {len(pages)} pages with st.navigation", 13, fonts.BOLD, theme.INK)
        c.rect(LEFT, 112, 260, 372, fill=theme.PANEL, outline=theme.GRID, radius=8)
        active = min(len(pages) - 1, int(a * len(pages)))
        for i, (title, _) in enumerate(pages):
            y = 120 + i * 24
            if i == active:
                c.rect(LEFT + 6, y, 254, y + 22, fill=theme.tint(theme.TOKEN, 0.2), radius=5)
            c.text(LEFT + 16, y + 11, title, 13, fonts.BOLD if i == active else fonts.SANS, theme.TOKEN if i == active else theme.INK, "lm")
        c.rect(290, 112, RIGHT, 372, fill=theme.PANEL, outline=theme.GRID, radius=8)
        c.text(310, 136, pages[active][0], 18, fonts.BOLD, theme.INK, "lm")
        c.text(310, 162, f"url path:  /{pages[active][1]}", 12, fonts.MONO, theme.MUTED, "lm")
        for k, tab in enumerate(("Explain", "Math", "Code")):
            c.box(310 + k * 90, 186, 80, 28, tab, fill=theme.tint(theme.TOKEN, 0.15) if k == 0 else theme.PANEL,
                  outline=theme.TOKEN, size=12, radius=6)
        parts.note(c, 310, 236, "Each page is a small script; the logic lives in ui/, llm/, training/ and viz/.", 12)
    elif step == 1:
        c.text(LEFT, 90, "The sidebar rebuilds the model; every page reads the shared session state", 13, fonts.BOLD, theme.INK)
        chain = [("sidebar form", theme.QUERY), ("ModelSettings", theme.QUERY), ("apply_settings", theme.ACCENT), ("session_state", theme.VALUE)]
        w = 150
        gap = (RIGHT - LEFT - 4 * w) / 3
        for i, (name, col) in enumerate(chain):
            show = theme.segment(a, i / 5, i / 5 + 0.25)
            x = LEFT + i * (w + gap)
            c.box(x, 118, w, 44, name, fill=parts.fade(theme.tint(col, 0.18), show), outline=parts.fade(col, show), size=13,
                  kind=fonts.MONO, text_fill=parts.fade(theme.INK, show))
            if i:
                c.arrow((x - gap + 4, 140), (x - 4, 140), parts.fade(theme.MUTED, show), 2.2)
        parts.note(c, LEFT, 174, "validate (ModelSettings.problems)  ->  build tokenizer + fresh GPT  ->  reset history", 12)
        c.text(LEFT, 210, "keys held in st.session_state:", 12, fonts.BOLD, theme.INK)
        for i, key in enumerate(facts.state_keys):
            show = theme.segment(a, 0.5 + i / 30, 0.5 + i / 30 + 0.2)
            r, k = divmod(i, 5)
            c.box(LEFT + k * 150, 234 + r * 40, 140, 30, key, fill=parts.fade(theme.tint(theme.VALUE, 0.15), show),
                  outline=parts.fade(theme.VALUE, show), size=12, kind=fonts.MONO, text_fill=parts.fade(theme.INK, show), radius=5)
    elif step == 2:
        c.text(LEFT, 90, "Move a control, and the figure is recomputed from the real model", 13, fonts.BOLD, theme.INK)
        c.rect(LEFT, 112, 250, 360, fill=theme.PANEL, outline=theme.GRID, radius=8)
        parts.panel_title(c, LEFT + 14, 128, "Temperature", theme.INK, 12)
        c.rect(LEFT + 14, 154, 230, 160, fill=None)
        c.rect(LEFT + 14, 154, 220, 158, fill=theme.GRID, radius=2)
        knob = LEFT + 14 + 206 * a
        c.rect(LEFT + 14, 154, knob, 158, fill=theme.TOKEN, radius=2)
        c.circle(knob, 156, 8, fill=theme.TOKEN)
        temp = 0.3 + 1.7 * a
        c.text(LEFT + 14, 182, f"T = {temp:.2f}", 14, fonts.MONO, theme.TOKEN, "lm")
        parts.paragraph(c, LEFT + 14, 212, "Sidebar and page widgets each have a stable key, so state survives reruns.", 210, 11, fill=theme.MUTED)
        c.rect(270, 112, RIGHT, 360, fill=theme.PANEL, outline=theme.GRID, radius=8)
        import torch
        from llm.transformer import next_token_distribution

        probs = next_token_distribution(ctx.trained_logits, temp)
        top = torch.topk(probs, 7)
        itos = ctx.char_tok.itos
        parts.bars(c, 290, 132, [itos[int(i)] for i in top.indices], [float(v) for v in top.values], 300, 28, theme.TOKEN, 0.5, 1.0, 12, ".2f", 34)
        parts.note(c, 290, 338, f"next-character probabilities at temperature {temp:.2f}", 11)
    else:
        hist = ctx.history
        c.text(LEFT, 90, "Training lab: the loop yields metrics, the page redraws the charts", 13, fonts.BOLD, theme.INK)
        c.box(LEFT, 114, 130, 34, "Start (from scratch)", fill=theme.tint(theme.TOKEN, 0.2), outline=theme.TOKEN, size=12, radius=6)
        c.box(LEFT + 142, 114, 70, 34, "Stop", fill=theme.PANEL, outline=theme.LOSS, size=12, text_fill=theme.LOSS, radius=6)
        box = (60, 176, 520, 352)
        c.axes(box, "", "step")
        xs = [h["step"] for h in hist]
        ys = [h["train_loss"] for h in hist]
        end = c.polyline(xs, ys, box, (1, max(2, xs[-1])), (min(ys) - 0.1, max(ys) + 0.1), theme.LOSS, 2.4, a)
        if end:
            c.circle(end[0], end[1], 5, fill=theme.QUERY)
        n = max(1, round(len(hist) * a))
        c.text(550, 190, "train(model, ...)", 13, fonts.MONO, theme.ACCENT)
        c.text(550, 214, "yields one dict per step", 12, fonts.SANS, theme.INK)
        c.text(550, 246, f"step {hist[n - 1]['step']} / {len(hist)}", 14, fonts.MONO, theme.TOKEN)
        c.text(550, 270, f"loss {hist[n - 1]['train_loss']:.3f}", 14, fonts.MONO, theme.LOSS)
        parts.paragraph(c, 550, 300, "The Stop button sets a flag that the loop checks before every step.", 220, 11, fill=theme.MUTED)


STREAMLIT = Scene(
    "09_streamlit_app", 9, K9, "The Streamlit application",
    ("Ten pages are registered with st.navigation",
     "The sidebar rebuilds the model; pages share one session state",
     "Widgets recompute figures from the real model",
     "The training lab streams metrics into live charts"),
    draw_streamlit, frames_per_step=12,
)


# --------------------------------------------------------------------------- #
# 10. Scripts, documentation, compliance
# --------------------------------------------------------------------------- #
def draw_compliance(c: Canvas, ctx: Context, step: int, t: float) -> None:
    a = phase(step, t, step)
    facts = ctx.project
    if step == 0:
        c.text(LEFT, 90, "One command each: scripts/ keeps every routine task repeatable", 13, fonts.BOLD, theme.INK)
        for i, name in enumerate(facts.scripts):
            show = theme.segment(a, i / (len(facts.scripts) + 2), i / (len(facts.scripts) + 2) + 0.25)
            y = 120 + i * 42
            f = parts.fade
            c.box(LEFT, y, 200, 34, name, fill=f(theme.tint(theme.QUERY, 0.15), show), outline=f(theme.QUERY, show), size=13,
                  kind=fonts.MONO, text_fill=f(theme.INK, show), radius=6)
            c.arrow((LEFT + 208, y + 17), (LEFT + 250, y + 17), f(theme.MUTED, show), 2)
            c.text(LEFT + 262, y + 17, SCRIPT_ROLES.get(name, "project helper"), 13, fonts.SANS, f(theme.INK, show), "lm")
    elif step == 1:
        c.text(LEFT, 90, "Documentation: a front door, a quick start, and the architecture", 13, fonts.BOLD, theme.INK)
        docs = [(d, theme.TOKEN) for d in facts.documents] + [("Docs/subtasks/  (30 documents + index)", theme.VALUE),
                                                              ("Docs/subtasks/animations/  (these GIFs)", theme.ACCENT)]
        for i, (name, col) in enumerate(docs):
            show = theme.segment(a, i / (len(docs) + 1), i / (len(docs) + 1) + 0.3)
            y = 120 + i * 46
            c.rect(LEFT, y, RIGHT, y + 36, fill=parts.fade(theme.tint(col, 0.14), show), outline=parts.fade(col, show), radius=6)
            c.text(LEFT + 14, y + 18, name, 14, fonts.MONO, parts.fade(theme.INK, show), "lm")
        parts.note(c, LEFT, 120 + len(docs) * 46 + 4, "Mermaid diagrams in README and Architecture show the same flow as these animations.", 12)
    elif step == 2:
        c.text(LEFT, 90, "Compliance gates run before anything is delivered", 13, fonts.BOLD, theme.INK)
        gates = [("unit tests", f"{facts.test_modules} test modules, every source file covered", theme.VALUE),
                 ("secrets scan", "no keys, e-mails or URLs in code; settings come from .env", theme.LOSS),
                 ("git hygiene", f"{len(facts.gitignore)} .gitignore rules keep secrets and caches out", theme.ACCENT),
                 ("project audit", "README, Quickstart, Architecture, scripts, layout", theme.TOKEN)]
        for i, (name, what, col) in enumerate(gates):
            show = theme.segment(a, i / 6, i / 6 + 0.3)
            y = 120 + i * 62
            c.rect(LEFT, y, RIGHT, y + 50, fill=parts.fade(theme.PANEL, show), outline=parts.fade(col, show), radius=8, width=1.8)
            c.text(LEFT + 60, y + 17, name, 15, fonts.BOLD, parts.fade(col, show), "lm")
            c.text(LEFT + 60, y + 37, what, 12, fonts.SANS, parts.fade(theme.INK, show), "lm")
            parts.check(c, LEFT + 28, y + 25, 11, col, theme.segment(a, i / 6 + 0.2, i / 6 + 0.5))
    else:
        from docsgen.catalog import SUBTASKS

        c.text(LEFT, 90, "The whole path, from empty folder to generated text", 13, fonts.BOLD, theme.INK)
        n = len(SUBTASKS)
        per = 5
        w = (RIGHT - LEFT - (per - 1) * 14) / per
        for i, sub in enumerate(SUBTASKS):
            num, title = sub.number, sub.title
            show = theme.segment(a, i / (n + 1), i / (n + 1) + 0.2)
            r, k = divmod(i, per)
            x, y = LEFT + k * (w + 14), 126 + r * 110
            col = theme.TOKEN if i < 5 else theme.VALUE
            c.rect(x, y, x + w, y + 90, fill=parts.fade(theme.tint(col, 0.15), show), outline=parts.fade(col, show), radius=8)
            c.text(x + 10, y + 18, f"{num}", 18, fonts.BOLD, parts.fade(col, show), "lm")
            parts.paragraph(c, x + 10, y + 36, title, w - 20, 11, fill=parts.fade(theme.INK, show))
        parts.note(c, LEFT, 360, "Open index.html in this folder for all animations next to the written guides.", 12, theme.INK)


COMPLIANCE = Scene(
    "10_scripts_docs_compliance", 10, K10, "Scripts, documents and quality gates",
    ("Small scripts make start, stop, clean-up and rebuilds one command each",
     "README, Quickstart and Architecture explain the project; Docs/ holds the rest",
     "Tests, secret scan, git hygiene and a project audit guard every delivery",
     "Ten sub-tasks, one working lab"),
    draw_compliance, frames_per_step=12,
)

SCENES = (EDUCATION, STREAMLIT, COMPLIANCE)
