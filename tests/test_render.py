"""Unit tests for docsgen/render.py."""
import pytest

from docsgen import render as r


def lookup(name):
    return {"a": "42"}[name]


def test_inline_and_display_math_become_mathml():
    out = r.render("# T\n\nInline $x^2$ here.\n\n$$y = x + 1$$\n", lookup).body
    assert out.count("<math") == 2
    assert 'class="eq"' in out


def test_escaped_dollar_stays_literal():
    out = r.render("# T\n\nCosts \\$5 and \\$6.\n", lookup).body
    assert "$5" in out and "<math" not in out


def test_inline_code_dollars_are_not_math():
    out = r.render("# T\n\nUse `$a$ and $b$` verbatim.\n", lookup).body
    assert "<math" not in out and "$a$" in out


def test_facts_expand_and_unknown_raises():
    assert "42" in r.render("# T\n\nValue {{fact:a}}.\n", lookup).body
    with pytest.raises(r.RenderError):
        r.render("# T\n\n{{fact:zzz}}\n", lookup)


def test_expand_facts_direct():
    assert r.expand_facts("x={{fact:a}}", lookup) == "x=42"


def test_title_is_taken_from_h1_and_removed():
    rendered = r.render("# Hello\n\n## Part\n\ntext\n", lookup)
    assert rendered.title == "Hello"
    assert "<h1" not in rendered.body


def test_flow_fence_becomes_captioned_figure():
    out = r.render("# T\n\n```flow The caption\nA --> B\n```\n", lookup).body
    assert '<figure class="flow">' in out and "The caption" in out and "A --&gt; B" in out


def test_code_fence_is_highlighted_and_focusable():
    out = r.render("# T\n\n```python\nx = 1\n```\n", lookup).body
    assert 'tabindex="0"' in out and "highlight" in out or 'class="code"' in out


def test_unknown_language_falls_back_to_text():
    out = r.highlight_code("hello", "no-such-lang")
    assert "hello" in out and 'tabindex="0"' in out


def test_admonition_renders():
    out = r.render('# T\n\n!!! note "Remember"\n    Body text.\n', lookup).body
    assert "admonition" in out and "Remember" in out


def test_md_links_are_rewritten_to_html():
    out = r.render("# T\n\n[x](other.md#frag) and [y](https://e.org/a.md)\n", lookup).body
    assert 'href="other.html#frag"' in out


def test_tables_are_wrapped_for_scrolling():
    out = r.render("# T\n\n| a | b |\n|---|---|\n| 1 | 2 |\n", lookup).body
    assert 'class="scroll"' in out and "<table>" in out


def test_toc_has_h2_with_h3_children_and_math_headings():
    rendered = r.render("# T\n\n## One\n\n### Sub\n\n## Two with $x$\n", lookup)
    assert [i.title for i in rendered.toc][0] == "One"
    assert [c.title for c in rendered.toc[0].children] == ["Sub"]
    assert rendered.toc[1].title.startswith("Two with")
    assert "<math" not in rendered.toc[1].title


def test_toc_html_lists_links_and_escapes():
    items = [r.TocItem("a", "A & B", 2, [r.TocItem("c", "Child", 3)])]
    out = r.toc_html(items)
    assert 'href="#a"' in out and "A &amp; B" in out and 'href="#c"' in out
    assert r.toc_html([]) == ""


def test_unknown_chart_raises():
    with pytest.raises(r.RenderError, match="unknown chart"):
        r.render("# T\n\n{{chart:nonsense}}\n", lookup)


def test_known_chart_renders_svg():
    assert "<svg" in r.render("# T\n\n{{chart:loss}}\n", lookup).body


def test_file_directive_excerpt_and_errors():
    out = r.file_figure("config.py||1-3")
    assert "config.py, lines 1-3" in out
    assert "(" in r.file_figure("config.py|python")
    with pytest.raises(r.RenderError):
        r.file_figure("missing/nowhere.py")
    with pytest.raises(r.RenderError):
        r.file_figure("config.py||9-1")
    with pytest.raises(r.RenderError):
        r.file_figure("../outside.py")


def test_source_directive_caption_and_errors():
    out = r.source_figure("llm.tokenizer.CharTokenizer")
    assert "llm/tokenizer.py, lines" in out and "CharTokenizer" in out
    with pytest.raises(r.RenderError):
        r.source_figure("llm.tokenizer.NoSuchThing")


def test_word_count_ignores_code_svg_and_math():
    fragment = "<p>one two three</p><pre>a b c d</pre><svg><text>x y</text></svg><math>m n</math>"
    assert r.word_count(fragment) == 3


def test_bad_formula_raises_render_error(monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("bad")

    monkeypatch.setattr(r, "latex_to_mathml", boom)
    with pytest.raises(r.RenderError, match="cannot convert"):
        r.render("# T\n\n$x$\n", lookup)
