"""Turn the Markdown sources into HTML fragments.

The documents are written in ordinary Markdown plus a few extensions:

``$...$`` and ``$$...$$``
    LaTeX, converted to MathML at build time (no JavaScript, no external fonts).
``{{fact:NAME}}``
    A number computed from the live code or the stored runs (see ``docsgen.facts``).
``{{source:dotted.path}}``
    The real source of a class, function or method, read with ``inspect`` so the
    excerpt can never drift away from the code. The caption shows file and lines.
``{{file:path|lang|first-last}}``
    A file (or a line range of it) from the repository, highlighted.
``{{chart:loss|lr|gradnorm}}``
    An accessible inline-SVG chart built from the stored reference run.
fenced blocks
    ```` ```python ```` is highlighted with Pygments; ```` ```flow Caption ```` is an
    ASCII flow chart shown in a captioned figure.

Pieces that Markdown must not touch (code, MathML, SVG) are *stashed* behind
alphanumeric tokens before conversion and put back afterwards.
"""
from __future__ import annotations

import html
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import markdown
from latex2mathml.converter import convert as latex_to_mathml
from markdown.extensions.toc import slugify
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

from docsgen import charts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_EXTENSIONS = ["tables", "fenced_code", "toc", "attr_list", "admonition", "sane_lists"]

FENCE_RE = re.compile(
    r"^(?P<indent>[ ]*)```(?P<info>[^\n`]*)\n(?P<body>.*?)^(?P=indent)```[ ]*$",
    re.DOTALL | re.MULTILINE,
)
FACT_RE = re.compile(r"\{\{fact:([A-Za-z0-9_]+)\}\}")
DIRECTIVE_RE = re.compile(r"^(?P<indent>[ ]*)\{\{(?P<kind>source|file|chart):(?P<arg>[^}\n]+)\}\}[ ]*$", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"(`+)(?!`)(.+?)(?<!`)\1(?!`)")
BLOCK_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<![\\\w$])\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?![\w$])")
TOKEN_RE = re.compile(r"ZXSTASH(\d+)ZX")
BLOCK_TOKEN_RE = re.compile(r"<p>\s*(ZXSTASH\d+ZX)\s*</p>")
H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>\s*", re.DOTALL)
MD_LINK_RE = re.compile(r'href="([^":#]+?)\.md(#[^"]*)?"')
TABLE_RE = re.compile(r"<table>(.*?)</table>", re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


class RenderError(ValueError):
    """Raised when a Markdown source uses an unknown directive or an invalid formula."""


@dataclass
class TocItem:
    id: str
    title: str
    level: int
    children: list["TocItem"] = field(default_factory=list)


@dataclass
class Rendered:
    title: str          # text of the first ``# heading``
    body: str           # HTML of everything after it
    toc: list[TocItem]  # h2 entries with h3 children


# --------------------------------------------------------------------------- #
# Code highlighting
# --------------------------------------------------------------------------- #
def highlight_code(code: str, lang: str = "text") -> str:
    """Highlight ``code`` as a keyboard-scrollable ``<pre>`` inside ``div.code``."""
    try:
        lexer = get_lexer_by_name(lang or "text")
    except ClassNotFound:
        lexer = TextLexer()
    formatter = HtmlFormatter(cssclass="code", wrapcode=True)
    out = highlight(code.rstrip("\n") + "\n", lexer, formatter)
    label = html.escape(f"{lexer.name} code", quote=True)
    return out.replace("<pre>", f'<pre tabindex="0" role="region" aria-label="{label}">', 1)


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def source_figure(dotted: str) -> str:
    """Excerpt of the live source of ``dotted`` with a file and line caption."""
    from edu import snippets  # imported late: pulls in the whole project

    try:
        obj = snippets.resolve(dotted)
        code = snippets.get_source(dotted)
        lines, first = inspect.getsourcelines(obj)
        where = _relative(Path(inspect.getsourcefile(obj) or ""))
    except (snippets.SnippetError, OSError, TypeError) as exc:
        raise RenderError(f"{{{{source:{dotted}}}}}: {exc}") from exc
    caption = f"{where}, lines {first}-{first + len(lines) - 1}: {dotted}"
    return (
        f'<figure class="source"><figcaption>{html.escape(caption)}</figcaption>'
        f"{highlight_code(code, 'python')}</figure>"
    )


def file_figure(arg: str) -> str:
    """``path|lang|first-last`` -> highlighted excerpt of a repository file."""
    parts = [p.strip() for p in arg.split("|")]
    rel = parts[0]
    lang = parts[1] if len(parts) > 1 and parts[1] else Path(rel).suffix.lstrip(".") or "text"
    path = (PROJECT_ROOT / rel).resolve()
    if PROJECT_ROOT not in path.parents or not path.is_file():
        raise RenderError(f"{{{{file:{arg}}}}}: not a file inside the project")
    all_lines = path.read_text(encoding="utf-8").splitlines()
    first, last = 1, len(all_lines)
    if len(parts) > 2 and parts[2]:
        match = re.fullmatch(r"(\d+)-(\d+)", parts[2])
        if not match or not 1 <= int(match[1]) <= int(match[2]) <= len(all_lines):
            raise RenderError(f"{{{{file:{arg}}}}}: bad line range {parts[2]!r} (file has {len(all_lines)} lines)")
        first, last = int(match[1]), int(match[2])
    caption = f"{rel}, lines {first}-{last}" if (first, last) != (1, len(all_lines)) else f"{rel} ({last} lines)"
    code = "\n".join(all_lines[first - 1:last])
    return f'<figure class="source"><figcaption>{html.escape(caption)}</figcaption>{highlight_code(code, lang)}</figure>'


def flow_figure(body: str, caption: str) -> str:
    """ASCII flow chart as a captioned, keyboard-scrollable figure."""
    cap = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return (
        f'<figure class="flow">{cap}<pre tabindex="0" role="region" aria-label="Flow diagram">'
        f"<code>{html.escape(body.rstrip())}</code></pre></figure>"
    )


# --------------------------------------------------------------------------- #
# Stash helpers
# --------------------------------------------------------------------------- #
class _Stash:
    """Park HTML behind tokens Markdown leaves alone."""

    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, value: str) -> str:
        self.items.append(value)
        return f"ZXSTASH{len(self.items) - 1}ZX"

    def restore(self, text: str) -> str:
        text = BLOCK_TOKEN_RE.sub(lambda m: self.items[int(TOKEN_RE.fullmatch(m.group(1)).group(1))], text)
        return TOKEN_RE.sub(lambda m: self.items[int(m.group(1))], text)


def _math(latex: str, display: bool) -> str:
    try:
        mathml = latex_to_mathml(latex.strip(), display="block" if display else "inline")
    except Exception as exc:  # latex2mathml raises assorted exception types
        raise RenderError(f"cannot convert formula {latex.strip()!r}: {exc}") from exc
    return f'<div class="eq">{mathml}</div>' if display else mathml


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def expand_facts(text: str, lookup: Callable[[str], str]) -> str:
    """Replace every ``{{fact:NAME}}`` using ``lookup``."""

    def sub(match: re.Match) -> str:
        try:
            return lookup(match.group(1))
        except KeyError as exc:
            raise RenderError(str(exc.args[0])) from exc

    return FACT_RE.sub(sub, text)


def _unescape_dollar(text: str) -> str:
    return text.replace("\\$", "$")


def render(text: str, lookup: Optional[Callable[[str], str]] = None) -> Rendered:
    """Render one Markdown source to HTML (see the module docstring for the dialect)."""
    if lookup is None:
        from docsgen import facts

        lookup = facts.get
    stash = _Stash()
    text = expand_facts(text, lookup)

    def fence(match: re.Match) -> str:
        indent, info = match["indent"], match["info"].strip()
        body = "\n".join(line[len(indent):] if line.startswith(indent) else line for line in match["body"].split("\n"))
        lang, _, caption = info.partition(" ")
        html_block = flow_figure(body, caption.strip()) if lang == "flow" else highlight_code(body, lang or "text")
        return f"\n\n{indent}{stash.add(html_block)}\n\n"

    text = FENCE_RE.sub(fence, text)

    def directive(match: re.Match) -> str:
        kind, arg = match["kind"], match["arg"].strip()
        if kind == "source":
            block = source_figure(arg)
        elif kind == "file":
            block = file_figure(arg)
        else:
            if arg not in charts.CHARTS:
                raise RenderError(f"unknown chart {arg!r}; known: {', '.join(charts.CHARTS)}")
            block = charts.CHARTS[arg]()
        return f"\n\n{match['indent']}{stash.add(block)}\n\n"

    text = DIRECTIVE_RE.sub(directive, text)

    # Keep inline code (and its dollars) away from the math pass.
    codes: list[str] = []

    def protect(match: re.Match) -> str:
        codes.append(match.group(0))
        return f"ZXCODE{len(codes) - 1}ZX"

    text = INLINE_CODE_RE.sub(protect, text)
    text = BLOCK_MATH_RE.sub(lambda m: f"\n\n{stash.add(_math(m.group(1), True))}\n\n", text)
    text = INLINE_MATH_RE.sub(lambda m: stash.add(_math(m.group(1), False)), text)
    text = _unescape_dollar(text)
    text = re.sub(r"ZXCODE(\d+)ZX", lambda m: codes[int(m.group(1))], text)

    def plain(value: str) -> str:
        """Text of a heading with stashed formulas replaced by their plain letters."""
        return TOKEN_RE.sub(lambda m: html.unescape(TAG_RE.sub("", stash.items[int(m.group(1))])), value)

    def slug(value: str, separator: str) -> str:
        return slugify(plain(value), separator)

    md = markdown.Markdown(
        extensions=MD_EXTENSIONS, extension_configs={"toc": {"slugify": slug}}, output_format="html"
    )
    body = md.convert(text)
    body = stash.restore(body)
    body = MD_LINK_RE.sub(lambda m: f'href="{m.group(1)}.html{m.group(2) or ""}"', body)
    body = TABLE_RE.sub(
        lambda m: f'<div class="scroll" tabindex="0" role="region" aria-label="Scrollable table"><table>{m.group(1)}</table></div>',
        body,
    )

    h1 = H1_RE.search(body)
    title = html.unescape(TAG_RE.sub("", h1.group(1))).strip() if h1 else ""
    if h1:
        body = body[: h1.start()] + body[h1.end():]
    return Rendered(title=title, body=body, toc=_collect_toc(getattr(md, "toc_tokens", []), plain))


def _collect_toc(tokens: list[dict], plain: Callable[[str], str] = lambda value: value) -> list[TocItem]:
    """Flatten the ``toc`` extension's tree to h2 items with their h3 children."""
    items: list[TocItem] = []

    def clean(name: str) -> str:
        return html.unescape(TAG_RE.sub("", plain(name))).strip()

    def walk(nodes: list[dict]) -> None:
        for node in nodes:
            level = node["level"]
            if level == 2:
                item = TocItem(node["id"], clean(node["name"]), 2)
                item.children = [
                    TocItem(c["id"], clean(c["name"]), 3) for c in node["children"] if c["level"] == 3
                ]
                items.append(item)
            elif level == 1:
                walk(node["children"])

    walk(tokens)
    return items


def toc_html(items: list[TocItem]) -> str:
    """Nested ``<ul>`` for the sidebar; empty string when there is nothing to list."""
    if not items:
        return ""
    out = ["<ul>"]
    for item in items:
        out.append(f'<li><a href="#{item.id}">{html.escape(item.title)}</a>')
        if item.children:
            out.append("<ul>" + "".join(f'<li><a href="#{c.id}">{html.escape(c.title)}</a></li>' for c in item.children) + "</ul>")
        out.append("</li>")
    out.append("</ul>")
    return "".join(out)


def word_count(fragment: str) -> int:
    """Approximate number of words of prose in an HTML fragment (code and SVG excluded)."""
    cleaned = re.sub(r"<(pre|svg|math)\b.*?</\1>", " ", fragment, flags=re.DOTALL)
    return len(html.unescape(TAG_RE.sub(" ", cleaned)).split())
