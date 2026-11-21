"""Unit tests for docsgen/build.py."""
import pytest

from docsgen import build, catalog, page


@pytest.fixture
def tiny_docs(tmp_path):
    """A docs dir with a minimal source pair for every sub-task."""
    (tmp_path / "guide_src").mkdir()
    for sub in catalog.SUBTASKS:
        (tmp_path / sub.explanation_md).write_text(
            f"# Explanation {sub.number}\n\n## Part\n\nValue {{{{fact:d_model}}}}.\n", encoding="utf-8"
        )
        (tmp_path / sub.guide_src).write_text(f"# Guide {sub.number}\n\n## Part\n\ntext\n", encoding="utf-8")
    return tmp_path


def test_build_document_writes_valid_page(tiny_docs):
    sub = catalog.get(1)
    dest = build.build_document(sub, page.KIND_EXPLANATION, tiny_docs)
    text = dest.read_text(encoding="utf-8")
    assert dest.name == sub.explanation_html
    assert "Explanation 1" in text and "words" in text


def test_build_document_guide_kind(tiny_docs):
    dest = build.build_document(catalog.get(2), page.KIND_GUIDE, tiny_docs)
    assert dest.name == catalog.get(2).guide_html


def test_missing_source_raises(tmp_path):
    with pytest.raises(build.BuildError, match="missing source"):
        build.build_document(catalog.get(1), page.KIND_EXPLANATION, tmp_path)


def test_index_markdown_links_all_twenty_pages():
    md = build.index_markdown()
    for sub in catalog.SUBTASKS:
        assert sub.explanation_html in md and sub.guide_html in md


def test_build_all_with_only_writes_selected_pages_and_index(tiny_docs):
    written = build.build_all(tiny_docs, only=[3])
    names = {p.name for p in written}
    assert names == {catalog.get(3).explanation_html, catalog.get(3).guide_html, "index.html"}


def test_build_all_builds_everything(tiny_docs):
    assert len(build.build_all(tiny_docs)) == 21


def test_build_all_reports_validation_problems(tiny_docs):
    (tiny_docs / catalog.get(1).explanation_md).write_text("# T\n\n[x](#nowhere)\n", encoding="utf-8")
    with pytest.raises(build.BuildError, match="validation failed"):
        build.build_all(tiny_docs, only=[1])


def test_main_returns_one_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(build, "DOCS_DIR", build.DOCS_DIR / "does_not_exist")
    monkeypatch.setattr(build, "build_all", lambda only=None: (_ for _ in ()).throw(build.BuildError("boom")))
    assert build.main([]) == 1
    assert "build failed: boom" in capsys.readouterr().err


def test_main_success_prints_written_paths(monkeypatch, capsys):
    target = build.PROJECT_ROOT / "Docs" / "subtasks" / "index.html"
    monkeypatch.setattr(build, "build_all", lambda only=None: [target])
    assert build.main(["--only", "1"]) == 0
    assert "Docs/subtasks/index.html" in capsys.readouterr().out


def test_committed_pages_validate():
    from docsgen import validate

    pages = sorted(build.DOCS_DIR.glob("*.html"))
    if not pages:
        pytest.skip("pages not built yet")
    problems = [p for path in pages for p in validate.check_file(path)]
    assert problems == []


def test_index_links_animations_only_when_asked_and_anchors_exist():
    from animgen import registry

    assert "animations/index.html" not in build.index_markdown()
    md = build.index_markdown(animations=True)
    keys = {s.key for s in registry.ALL_SCENES}
    for sub in catalog.SUBTASKS:
        assert f"(animations/index.html#{sub.stem})" in md and sub.stem in keys


def test_build_index_links_the_gallery_when_it_exists(tiny_docs):
    plain = build.build_index(tiny_docs).read_text(encoding="utf-8")
    assert "animations/index.html" not in plain
    (tiny_docs / "animations").mkdir()
    (tiny_docs / "animations" / "index.html").write_text("<html></html>")
    assert "animations/index.html#01_scaffolding" in build.build_index(tiny_docs).read_text(encoding="utf-8")
