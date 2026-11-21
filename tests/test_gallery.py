"""Unit tests for animgen/gallery.py."""
import re

from animgen import gallery, registry


def test_gallery_links_every_existing_gif_with_alt_text():
    page = gallery.gallery_html({s.filename for s in registry.ALL_SCENES})
    for s in registry.ALL_SCENES:
        assert f'src="{s.filename}"' in page and f'id="{s.key}"' in page
    assert page.count("<img ") == len(registry.ALL_SCENES)
    assert all(alt.strip() for alt in re.findall(r'alt="([^"]*)"', page))


def test_gallery_is_self_contained_and_links_the_written_guides():
    page = gallery.gallery_html(set())
    assert "<script" not in page and "http://" not in page and "https://" not in page and "@import" not in page
    assert "../01_scaffolding_explanation.html" in page and "../10_scripts_docs_compliance_implementation_guide.html" in page


def test_missing_gifs_get_a_hint_instead_of_a_broken_image():
    page = gallery.gallery_html({"01_scaffolding.gif"})
    assert page.count("<img ") == 1 and "generate-animations.sh" in page


def test_text_is_escaped(monkeypatch):
    from dataclasses import replace

    nasty = replace(registry.ALL_SCENES[0], title="<b>x</b>")
    monkeypatch.setattr(gallery, "ALL_SCENES", (nasty,))
    page = gallery.gallery_html(set())
    assert "<b>x</b>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page


def test_write_gallery_lists_the_gifs_found_in_the_folder(tmp_path):
    (tmp_path / "02_tokenization.gif").write_bytes(b"GIF89a")
    path = gallery.write_gallery(tmp_path / "new")
    assert path.name == "index.html" and "<img " not in path.read_text(encoding="utf-8")
    path = gallery.write_gallery(tmp_path)
    assert 'src="02_tokenization.gif"' in path.read_text(encoding="utf-8")
