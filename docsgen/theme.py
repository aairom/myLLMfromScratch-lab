"""Styling for the generated pages: one inline stylesheet plus an accessible code palette.

Everything is embedded in every page so each HTML file works on its own, offline.
Colours were chosen for WCAG AA contrast (4.5:1 for text) in both light and dark mode.
"""
from __future__ import annotations

from pygments.formatters import HtmlFormatter
from pygments.style import Style
from pygments.token import (
    Comment, Error, Generic, Keyword, Name, Number, Operator, String, Token,
)


class DocLight(Style):
    """Syntax colours with at least 4.5:1 contrast on #f6f8fa."""

    background_color = "#f6f8fa"
    styles = {
        Token: "#1f2328",
        Comment: "italic #4f5b66",
        Keyword: "bold #8a1c7c",
        Keyword.Constant: "bold #0b5cad",
        Operator: "#1f2328",
        Operator.Word: "bold #8a1c7c",
        Name.Builtin: "#0b5cad",
        Name.Function: "bold #6b3fa0",
        Name.Class: "bold #6b3fa0",
        Name.Decorator: "#6b3fa0",
        Name.Exception: "bold #b3261e",
        String: "#0a6b3b",
        String.Doc: "italic #4f5b66",
        Number: "#0b5cad",
        Generic.Heading: "bold",
        Generic.Prompt: "#4f5b66",
        Error: "#b3261e",
    }


class DocDark(Style):
    """Counterpart for dark mode (contrast at least 4.5:1 on #161b22)."""

    background_color = "#161b22"
    styles = {
        Token: "#e6edf3",
        Comment: "italic #9fb0c0",
        Keyword: "bold #ff9bd9",
        Keyword.Constant: "bold #79c0ff",
        Operator: "#e6edf3",
        Operator.Word: "bold #ff9bd9",
        Name.Builtin: "#79c0ff",
        Name.Function: "bold #d2a8ff",
        Name.Class: "bold #d2a8ff",
        Name.Decorator: "#d2a8ff",
        Name.Exception: "bold #ffa198",
        String: "#7ee787",
        String.Doc: "italic #9fb0c0",
        Number: "#79c0ff",
        Generic.Heading: "bold",
        Generic.Prompt: "#9fb0c0",
        Error: "#ffa198",
    }


def code_css() -> str:
    """Pygments rules for light mode and, inside a media query, for dark mode."""
    light = HtmlFormatter(style=DocLight, cssclass="code").get_style_defs(".code")
    dark = HtmlFormatter(style=DocDark, cssclass="code").get_style_defs(".code")
    dark = "\n".join("  " + line for line in dark.splitlines())
    return f"{light}\n@media (prefers-color-scheme: dark) {{\n{dark}\n}}"


BASE_CSS = """
:root {
  --bg: #ffffff; --surface: #f6f8fa; --border: #d0d7de; --text: #1f2328; --muted: #4f5b66;
  --accent: #0b5cad; --accent-bg: #eaf2fb; --note-bg: #fff8e1; --note-border: #9a6700;
  --ok-bg: #e8f5ee; --ok-border: #0a6b3b; --focus: #b3261e;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d; --text: #e6edf3; --muted: #9fb0c0;
    --accent: #79c0ff; --accent-bg: #10243a; --note-bg: #2a2310; --note-border: #d4a72c;
    --ok-bg: #0f2a1a; --ok-border: #7ee787; --focus: #ffa198;
  }
}
* { box-sizing: border-box; }
html { font-size: 100%; -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--text); line-height: 1.65;
  font-family: -apple-system, "Segoe UI", system-ui, Roboto, "Helvetica Neue", Arial, sans-serif;
  font-size: 1.0625rem;
}
.skip-link { position: absolute; left: -999px; top: 0; background: var(--accent); color: var(--bg);
  padding: .5rem 1rem; z-index: 10; }
.skip-link:focus { left: .5rem; top: .5rem; }
a { color: var(--accent); text-underline-offset: .15em; }
a:focus-visible, summary:focus-visible, pre:focus-visible, .scroll:focus-visible {
  outline: 3px solid var(--focus); outline-offset: 2px; }
header.site { border-bottom: 1px solid var(--border); background: var(--surface); }
.wrap { max-width: 62rem; margin: 0 auto; padding: 0 1.25rem; }
header.site .wrap { padding-top: 1.25rem; padding-bottom: 1rem; }
.kicker { margin: 0; color: var(--muted); font-size: .9rem; letter-spacing: .02em; }
h1 { font-size: 2rem; line-height: 1.25; margin: .25rem 0 .5rem; }
.lede { margin: 0; color: var(--muted); max-width: 46rem; }
.badges { display: flex; flex-wrap: wrap; gap: .5rem; margin: .9rem 0 0; padding: 0; list-style: none; }
.badges li { border: 1px solid var(--border); border-radius: 999px; padding: .1rem .7rem;
  font-size: .85rem; background: var(--bg); }
nav.subnav { display: flex; flex-wrap: wrap; gap: .5rem 1.25rem; margin: .9rem 0 0; font-size: .95rem; }
nav.subnav a[aria-current="page"] { font-weight: 700; color: var(--text); text-decoration: none; }
.layout { display: grid; grid-template-columns: 15rem minmax(0, 1fr); gap: 2.5rem; padding-top: 1.5rem; }
nav.toc { position: sticky; top: 1rem; align-self: start; font-size: .92rem;
  max-height: calc(100vh - 2rem); overflow: auto; }
nav.toc h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 .5rem; }
nav.toc ul { list-style: none; margin: 0; padding: 0; }
nav.toc ul ul { padding-left: .9rem; }
nav.toc li { margin: .25rem 0; }
main { min-width: 0; padding-bottom: 3rem; }
main h2 { font-size: 1.5rem; margin: 2.4rem 0 .6rem; padding-top: .6rem; border-top: 1px solid var(--border); }
main h3 { font-size: 1.2rem; margin: 1.8rem 0 .4rem; }
main h4 { font-size: 1.05rem; margin: 1.4rem 0 .3rem; }
main p, main li { max-width: 46rem; }
main li { margin: .25rem 0; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .92em;
  background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: .05em .3em; }
pre { margin: 1rem 0; padding: .9rem 1rem; overflow: auto; background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; line-height: 1.5; tab-size: 4; }
pre code { background: none; border: 0; padding: 0; font-size: .86rem; }
figure.source { margin: 1.25rem 0; }
figure.source figcaption { font-size: .85rem; color: var(--muted); margin-bottom: .25rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
figure.source pre { margin: 0; }
pre.flow { background: var(--accent-bg); border-color: var(--accent); }
.scroll { overflow-x: auto; margin: 1rem 0; }
table { border-collapse: collapse; width: 100%; font-size: .95rem; }
th, td { border: 1px solid var(--border); padding: .4rem .65rem; text-align: left; vertical-align: top; }
thead th { background: var(--surface); }
caption { text-align: left; color: var(--muted); font-size: .85rem; padding-bottom: .3rem; }
math { font-size: 1.1em; }
math[display="block"] { display: block; margin: 1rem 0; overflow-x: auto; overflow-y: hidden; }
.eq { margin: 1rem 0; padding: .4rem .8rem; border-left: 4px solid var(--accent); background: var(--accent-bg); overflow-x: auto; }
.callout { margin: 1.25rem 0; padding: .7rem 1rem; border-left: 4px solid var(--note-border);
  background: var(--note-bg); border-radius: 0 6px 6px 0; }
.callout.ok { border-color: var(--ok-border); background: var(--ok-bg); }
.callout > :first-child { margin-top: 0; } .callout > :last-child { margin-bottom: 0; }
.callout .label { font-weight: 700; display: block; }
ol.path { list-style: none; counter-reset: step; display: grid; gap: .35rem; padding: 0; margin: 1rem 0;
  grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); }
ol.path li { counter-increment: step; border: 1px solid var(--border); border-radius: 6px;
  padding: .45rem .7rem; font-size: .9rem; background: var(--surface); max-width: none; }
ol.path li::before { content: counter(step) ". "; font-weight: 700; }
ol.path li[aria-current="step"] { border: 2px solid var(--accent); background: var(--accent-bg); }
footer.site { border-top: 1px solid var(--border); padding: 1.5rem 0 2.5rem; color: var(--muted); font-size: .9rem; }
.cards { display: grid; gap: 1rem; grid-template-columns: repeat(auto-fit, minmax(17rem, 1fr)); padding: 0; list-style: none; }
.cards li { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; background: var(--surface); max-width: none; }
.cards h3 { margin: 0 0 .3rem; font-size: 1.05rem; }
.cards p { margin: .3rem 0; }
@media (max-width: 52rem) {
  .layout { grid-template-columns: minmax(0, 1fr); }
  nav.toc { position: static; max-height: none; }
}
@media print {
  .skip-link, nav.subnav, nav.toc { display: none; }
  .layout { display: block; }
  pre { white-space: pre-wrap; }
  a { color: inherit; }
}
"""


EXTRA_CSS = """
:root { --c0: #5b6b7b; --c1: #0b5cad; --c2: #8a1c7c; --grid: #d8dee4; --plot: #ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --c0: #9fb0c0; --c1: #79c0ff; --c2: #ff9bd9; --grid: #30363d; --plot: #0d1117; }
}
.admonition { margin: 1.25rem 0; padding: .7rem 1rem; border-left: 4px solid var(--note-border);
  background: var(--note-bg); border-radius: 0 6px 6px 0; }
.admonition.tip, .admonition.success { border-color: var(--ok-border); background: var(--ok-bg); }
.admonition.info, .admonition.example { border-color: var(--accent); background: var(--accent-bg); }
.admonition > :last-child { margin-bottom: 0; }
.admonition-title { margin: 0 0 .25rem; font-weight: 700; }
figure.chart { margin: 1.5rem 0; }
figure.chart figcaption { max-width: 46rem; margin-bottom: .4rem; font-size: .95rem; }
.chart-svg { width: 100%; height: auto; max-width: 46rem; display: block; }
.chart-svg .plot-bg { fill: var(--plot); stroke: var(--border); }
.chart-svg .grid { stroke: var(--grid); stroke-width: 1; }
.chart-svg .axis { stroke: var(--muted); stroke-width: 1.5; }
.chart-svg .tick { fill: var(--muted); font-size: 11px; }
.chart-svg .label { fill: var(--text); font-size: 12px; }
.chart-svg .ref, svg .ref { stroke: var(--muted); stroke-width: 1.5; stroke-dasharray: 2 4; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.line.s0 { stroke: var(--c0); } .line.s1 { stroke: var(--c1); stroke-width: 2.5; }
.line.s2 { stroke: var(--c2); stroke-dasharray: 7 4; }
.line.thin { stroke-width: 1; opacity: .8; }
.dot.s2 { fill: var(--c2); }
ul.legend { display: flex; flex-wrap: wrap; gap: .3rem 1.2rem; list-style: none; margin: .4rem 0 0; padding: 0; font-size: .88rem; }
ul.legend li { margin: 0; max-width: none; }
ul.legend svg { vertical-align: middle; }
details.data { margin-top: .5rem; font-size: .92rem; }
details.data summary { cursor: pointer; }
figure.flow { margin: 1.25rem 0; }
figure.flow figcaption { font-size: .9rem; color: var(--muted); margin-bottom: .25rem; }
figure.flow pre { margin: 0; background: var(--accent-bg); border-color: var(--accent); }
.fact-note { color: var(--muted); font-size: .9rem; }
"""


def stylesheet() -> str:
    """The complete inline stylesheet for a page."""
    return BASE_CSS + "\n" + EXTRA_CSS + "\n" + code_css()
