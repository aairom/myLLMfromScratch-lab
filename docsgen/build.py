"""Build every deliverable in ``Docs/subtasks/``.

For each of the ten sub-tasks::

    NN_<slug>_explanation.md                 hand-written source (input)
    guide_src/NN_<slug>_guide.md             hand-written source (input)
    NN_<slug>_explanation.html               generated
    NN_<slug>_implementation_guide.html      generated
    index.html                               generated

Run ``scripts/build-docs.sh`` (or ``python -m docsgen.build``). The exit code is
non-zero when a source is missing, a directive fails, or a page fails validation.
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Optional, Sequence

from docsgen import catalog, facts, page, validate
from docsgen.render import Rendered, render, word_count

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "Docs" / "subtasks"


class BuildError(RuntimeError):
    """Raised for missing sources or pages that fail validation."""


def _read(path: Path) -> str:
    if not path.is_file():
        raise BuildError(f"missing source: {path}")
    return path.read_text(encoding="utf-8")


def build_document(sub: catalog.SubTask, kind: str, docs_dir: Path = DOCS_DIR) -> Path:
    """Render one sub-task document and return the written path."""
    if kind == page.KIND_EXPLANATION:
        src, dest = docs_dir / sub.explanation_md, docs_dir / sub.explanation_html
    else:
        src, dest = docs_dir / sub.guide_src, docs_dir / sub.guide_html
    rendered = render(_read(src))
    words = word_count(rendered.body)
    badges = (f"{words:,} words", f"{len(rendered.toc)} sections")
    meta = page.subtask_meta(sub, kind, rendered, badges)
    note = "All numbers are produced from the running code and the recorded reference run, not typed by hand."
    dest.write_text(page.build_page(meta, rendered, note), encoding="utf-8")
    return dest


ANIMATIONS_INDEX = "animations/index.html"  # written by animgen; linked from the index when present


def index_markdown(animations: bool = False) -> str:
    """Markdown for the index page (the cards are plain HTML kept inside it).

    ``animations`` adds a link per sub-task to the GIF gallery next to the pages.
    """
    lines = [
        "# Documentation of the ten sub-tasks",
        "",
        "Each sub-task has two standalone documents: a **scientific explanation** (concepts, mathematics, "
        "significance and limits) and an **implementation guide** (where the code sits, how data flows, "
        "which structures exist, how it is configured and how it behaves at run time). "
        "Every page is one self-contained HTML file: no network, no scripts, readable offline.",
        "",
        "## Reference facts used throughout",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Default model | {{fact:d_model}}-dimensional, {{fact:n_heads}} heads, {{fact:n_layers}} layers, "
        "context {{fact:context}}, **{{fact:params}}** parameters |",
        "| Corpus | {{fact:corpus_chars}} characters, {{fact:char_vocab}}-symbol character vocabulary |",
        "| Reference run | {{fact:ref_steps}} steps; training loss {{fact:ref_first_train}} to {{fact:ref_last_train}}, "
        "validation loss {{fact:ref_last_val}} |",
        "| Honest reading | The unigram entropy of the data is {{fact:unigram_val}} nats, so after "
        "{{fact:ref_steps}} steps the model has learned letter frequencies, not yet context |",
        "",
        "## Documents",
        "",
    ]
    for sub in catalog.SUBTASKS:
        lines += [
            f"### {sub.number}. {sub.title}",
            "",
            sub.summary,
            "",
            f"- [Scientific explanation]({sub.explanation_html})",
            f"- [Implementation guide]({sub.guide_html})",
        ]
        if animations:
            lines.append(f"- [Animation](animations/index.html#{sub.stem})")
        lines.append("")
    lines += [
        "## How these pages are made",
        "",
        "The Markdown sources live next to the pages. `scripts/build-docs.sh` converts them with the "
        "`docsgen` package: formulas become MathML, code excerpts are read from the live source with "
        "`inspect`, and every number is looked up from the code or the recorded run. The build then "
        "validates each page (one `h1`, heading order, accessible SVG, no external resources, no "
        "unresolved placeholders).",
        "",
    ]
    return "\n".join(lines)


def build_index(docs_dir: Path = DOCS_DIR) -> Path:
    rendered = render(index_markdown((docs_dir / ANIMATIONS_INDEX).is_file()))
    meta = page.PageMeta(
        title="Sub-task documentation",
        kicker=f"{page.PROJECT_NAME} \u00b7 Index",
        lede="Scientific explanations and implementation guides for all ten sub-tasks.",
        badges=(f"{len(catalog.SUBTASKS)} sub-tasks", f"{2 * len(catalog.SUBTASKS)} documents"),
        path_html=page.path_html(None, page.KIND_EXPLANATION),
    )
    dest = docs_dir / "index.html"
    dest.write_text(page.build_page(meta, rendered), encoding="utf-8")
    return dest


def build_all(docs_dir: Path = DOCS_DIR, only: Optional[Sequence[int]] = None) -> list[Path]:
    """Build everything (or only the given sub-task numbers), then validate all written files."""
    facts.all_facts()  # fail early if a fact cannot be computed
    written: list[Path] = []
    for sub in catalog.SUBTASKS:
        if only and sub.number not in only:
            continue
        written.append(build_document(sub, page.KIND_EXPLANATION, docs_dir))
        written.append(build_document(sub, page.KIND_GUIDE, docs_dir))
    written.append(build_index(docs_dir))
    problems: list[str] = []
    for path in written:
        problems += validate.check_file(path, check_links=not only)
    if problems:
        raise BuildError("validation failed:\n  " + "\n  ".join(problems))
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the sub-task documentation pages.")
    parser.add_argument("--only", type=int, nargs="*", help="sub-task numbers to build (default: all)")
    args = parser.parse_args(argv)
    try:
        written = build_all(only=args.only)
    except (BuildError, ValueError, KeyError) as exc:
        print(f"build failed: {html.unescape(str(exc))}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
