"""Unit tests for docsgen/page.py."""
from docsgen import catalog, page, validate
from docsgen.render import Rendered, TocItem


def make_rendered():
    return Rendered(
        title="Body title",
        body="<h2 id=\"one\">One</h2><p>text</p>",
        toc=[TocItem("one", "One", 2)],
    )


def test_path_html_marks_current_step():
    out = page.path_html(3, page.KIND_EXPLANATION)
    assert out.count('aria-current="step"') == 1
    assert catalog.get(3).explanation_html in out
    assert out.count("<li") == 10


def test_path_html_without_current_and_guide_links():
    out = page.path_html(None, page.KIND_GUIDE)
    assert "aria-current" not in out
    assert catalog.get(1).guide_html in out


def test_subtask_meta_neighbours():
    first = page.subtask_meta(catalog.get(1), page.KIND_EXPLANATION, make_rendered())
    labels = [label for label, _h, _c in first.subnav]
    assert not any(label.startswith("\u2190") for label in labels)
    assert any(label.endswith("\u2192") for label in labels)
    last = page.subtask_meta(catalog.get(10), page.KIND_GUIDE, make_rendered(), ("badge",))
    labels = [label for label, _h, _c in last.subnav]
    assert any(label.startswith("\u2190") for label in labels)
    assert not any(label.endswith("\u2192") for label in labels)
    assert last.badges == ("badge",)
    assert last.title.startswith("10. ")


def test_subtask_meta_marks_current_document():
    meta = page.subtask_meta(catalog.get(2), page.KIND_GUIDE, make_rendered())
    current = [label for label, _h, is_current in meta.subnav if is_current]
    assert current == ["Implementation guide"]


def test_build_page_is_valid_standalone_html():
    meta = page.subtask_meta(catalog.get(2), page.KIND_EXPLANATION, make_rendered(), ("1 word",))
    out = page.build_page(meta, make_rendered(), "A note")
    assert out.startswith("<!DOCTYPE html>")
    assert "A note" in out and "wrap layout" in out and "layout single" not in out
    assert validate.check_html(out) == []


def test_build_page_without_toc_uses_single_column():
    meta = page.PageMeta(title="T", kicker="K", lede="L")
    out = page.build_page(meta, Rendered(title="T", body="<p>x</p>", toc=[]))
    assert "layout single" in out and "On this page" not in out
