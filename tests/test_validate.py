"""Unit tests for docsgen/validate.py."""
import pytest

from docsgen import validate


def doc(body="<h2 id=\"a\">A</h2>", head_extra="", lang="en", title="T"):
    return (
        f'<!DOCTYPE html><html lang="{lang}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width"><title>{title}</title>{head_extra}</head>'
        f'<body><a class="skip-link" href="#content">Skip</a><main id="content"><h1>H</h1>{body}</main></body></html>'
    )


def test_good_page_passes():
    assert validate.check_html(doc()) == []


@pytest.mark.parametrize(
    "html, expected",
    [
        (doc(lang=""), "lang"),
        (doc(title=""), "empty <title>"),
        (doc("<h1>Two</h1>"), "exactly one <h1>"),
        (doc("<h3>Jump</h3>"), "heading level jumps"),
        (doc('<p id="x"></p><p id="x"></p>'), "duplicate id"),
        (doc('<a href="#nowhere">x</a>'), "missing anchor"),
        (doc('<a href="https://example.org">x</a>'), "external link"),
        (doc("<script>1</script>"), "<script>"),
        (doc(head_extra="<style>a{background:url(x.png)}</style>"), "external resource"),
        (doc("<svg><rect/></svg>"), "accessible name"),
        (doc("<pre>raw</pre>"), "without tabindex"),
        (doc("<p>{{fact:x}}</p>"), "placeholder"),
        (doc("<p>ZXSTASH3ZX</p>"), "placeholder"),
        (doc('<img src="a.png">'), "not allowed"),
        (doc("<div>"), "not closed before"),
        (doc("</span>"), "unexpected"),
    ],
)
def test_problems_are_reported(html, expected):
    problems = validate.check_html(html)
    assert any(expected in p for p in problems), problems


def test_missing_doctype_viewport_main_and_skip_link():
    problems = validate.check_html('<html lang="en"><head><title>T</title></head><body><h1>x</h1></body></html>')
    joined = " ".join(problems)
    for needle in ("DOCTYPE", "viewport", "<main>", "skip link"):
        assert needle in joined


def test_code_blocks_may_mention_url_and_braces():
    body = '<pre tabindex="0"><code>a{background:url(x)} {{fact:x}} @import</code></pre>'
    assert validate.check_html(doc(body)) == []


def test_accessible_svg_passes():
    body = '<svg role="img"><title>Chart</title></svg><svg aria-hidden="true"></svg>'
    assert validate.check_html(doc(body)) == []


def test_check_file_reports_broken_links(tmp_path):
    (tmp_path / "ok.html").write_text(doc(), encoding="utf-8")
    page = tmp_path / "p.html"
    page.write_text(doc('<a href="ok.html">ok</a><a href="gone.html#x">bad</a>'), encoding="utf-8")
    problems = validate.check_file(page)
    assert problems == ["p.html: broken link to gone.html"]
    assert validate.check_file(page, check_links=False) == []
