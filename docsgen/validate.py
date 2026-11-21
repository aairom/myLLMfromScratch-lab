"""Quality gate for the generated pages: standalone, accessible, and free of leftovers."""
from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

VOID = {"meta", "link", "br", "hr", "img", "input", "col", "area", "base", "source", "track", "wbr"}
URL_ATTRS = ("src", "href", "poster", "action", "data", "srcset")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)
PLACEHOLDER = re.compile(r"\{\{|\}\}")
INTERNAL_TOKEN = re.compile(r"ZXSTASH|ZXCODE")
CODE_BLOCK = re.compile(r"<pre\b.*?</pre>|<code\b.*?</code>", re.DOTALL | re.IGNORECASE)


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.ids: set[str] = set()
        self.dup_ids: set[str] = set()
        self.fragments: list[str] = []
        self.links: list[str] = []
        self.headings: list[int] = []
        self.lang = ""
        self.title = ""
        self.in_title = False
        self.has_viewport = False
        self.has_main = False
        self.has_skip = False
        self.svg_names: list[tuple[bool, bool]] = []  # (has <title>, has role=img)
        self._svg_stack: list[list[bool]] = []
        self.pre_without_label = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag not in VOID:
            self.stack.append(tag)
        if tag == "html":
            self.lang = a.get("lang") or ""
        if tag == "title":
            self.in_title = True
        if tag == "meta" and a.get("name") == "viewport":
            self.has_viewport = True
        if tag == "main":
            self.has_main = True
        if tag == "a" and (a.get("class") or "") == "skip-link":
            self.has_skip = True
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append(int(tag[1]))
        if "id" in a:
            if a["id"] in self.ids:
                self.dup_ids.add(a["id"])
            self.ids.add(a["id"])
        for name in URL_ATTRS:
            value = a.get(name)
            if value is None:
                continue
            if name == "href" and value.startswith("#"):
                self.fragments.append(value[1:])
            elif name == "href" and tag == "a":
                self.links.append(value)
            else:
                self.errors.append(f"<{tag} {name}={value!r}> is not allowed in a standalone page")
        if tag == "svg" and a.get("aria-hidden") != "true":
            self._svg_stack.append([False, a.get("role") == "img"])
        elif tag == "svg":
            self._svg_stack.append([True, True])
        if tag == "title" and self._svg_stack:
            self._svg_stack[-1][0] = True
        if tag == "pre" and not (a.get("aria-label") or a.get("tabindex")):
            self.pre_without_label += 1

    def handle_endtag(self, tag):
        if tag == "title":
            self.in_title = False
        if tag == "svg" and self._svg_stack:
            self.svg_names.append(tuple(self._svg_stack.pop()))  # type: ignore[arg-type]
        if tag in VOID:
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:
                self.errors.append(f"<{self.stack.pop()}> is not closed before </{tag}>")
            self.stack.pop()
        else:
            self.errors.append(f"unexpected </{tag}>")

    def handle_data(self, data):
        if self.in_title:
            self.title += data


def check_html(text: str) -> list[str]:
    """Return a list of problems (empty means the page passes)."""
    p = _Collector()
    p.feed(text)
    p.close()
    problems = list(p.errors)
    problems += [f"<{t}> never closed" for t in p.stack]
    if not text.lstrip().lower().startswith("<!doctype html>"):
        problems.append("missing <!DOCTYPE html>")
    if p.lang != "en":
        problems.append(f"html lang is {p.lang!r}, expected 'en'")
    if not p.title.strip():
        problems.append("empty <title>")
    if not p.has_viewport:
        problems.append("missing viewport meta tag")
    if not p.has_main:
        problems.append("missing <main> landmark")
    if not p.has_skip:
        problems.append("missing skip link")
    h1 = p.headings.count(1)
    if h1 != 1:
        problems.append(f"expected exactly one <h1>, found {h1}")
    for before, after in zip(p.headings, p.headings[1:]):
        if after > before + 1:
            problems.append(f"heading level jumps from h{before} to h{after}")
            break
    for ident in sorted(p.dup_ids):
        problems.append(f"duplicate id {ident!r}")
    for frag in p.fragments:
        if frag and frag not in p.ids:
            problems.append(f"link to missing anchor #{frag}")
    for href in p.links:
        if EXTERNAL.match(href):
            problems.append(f"external link {href!r}; pages must be self-contained")
    for has_title, has_role in p.svg_names:
        if not (has_title and has_role):
            problems.append("an <svg> has no accessible name (needs role=img and <title>) and is not aria-hidden")
            break
    if p.pre_without_label:
        problems.append(f"{p.pre_without_label} <pre> block(s) without tabindex or aria-label")
    if "<script" in text.lower():
        problems.append("<script> found")
    prose = CODE_BLOCK.sub("", text)  # displayed code may legitimately mention url( or braces
    if "@import" in prose or "url(" in prose:
        problems.append("stylesheet references an external resource")
    if PLACEHOLDER.search(prose) or INTERNAL_TOKEN.search(text):
        problems.append("unresolved placeholder ({{ }} or internal token) left in the page")
    return problems


def check_file(path: Path, check_links: bool = True) -> list[str]:
    """Check one file; optionally verify that relative links point to existing siblings."""
    text = path.read_text(encoding="utf-8")
    problems = check_html(text)
    if check_links:
        for href in re.findall(r'<a [^>]*href="([^"#:]+)(?:#[^"]*)?"', text):
            if not (path.parent / href).exists():
                problems.append(f"broken link to {href}")
    return [f"{path.name}: {p}" for p in problems]
