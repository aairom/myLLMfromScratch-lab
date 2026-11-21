"""Unit tests for lesson content integrity."""
import re

import pytest

from edu import lessons, snippets

EXPECTED_KEYS = {
    "tokenization", "embeddings", "positional", "attention", "multihead", "layernorm",
    "ffn", "residual", "model", "loss", "optimizer", "schedule", "sampling",
}


def test_all_expected_lessons_exist_in_order():
    keys = [l.key for l in lessons.all_lessons()]
    assert set(keys) == EXPECTED_KEYS and len(keys) == len(set(keys))
    assert keys.index("tokenization") < keys.index("embeddings") < keys.index("attention") < keys.index("model")


@pytest.mark.parametrize("lesson", lessons.all_lessons(), ids=lambda l: l.key)
def test_lesson_has_every_section(lesson):
    assert lesson.title and lesson.summary and len(lesson.intuition) > 100
    assert len(lesson.steps) >= 3 and lesson.formulas and lesson.shapes and lesson.pitfalls and lesson.snippets
    for f in lesson.formulas:
        assert f.latex.strip() and f.meaning.strip()
    for row in lesson.shapes:
        assert len(row) == 3 and all(row)


@pytest.mark.parametrize("lesson", lessons.all_lessons(), ids=lambda l: l.key)
def test_every_snippet_path_resolves_to_real_code(lesson):
    for path in lesson.snippets:
        assert snippets.get_source(path).strip(), path


@pytest.mark.parametrize("lesson", lessons.all_lessons(), ids=lambda l: l.key)
def test_related_keys_exist(lesson):
    assert set(lesson.related) <= EXPECTED_KEYS and lesson.key not in lesson.related


@pytest.mark.parametrize("lesson", lessons.all_lessons(), ids=lambda l: l.key)
def test_latex_is_balanced(lesson):
    for f in lesson.formulas:
        tex = f.latex
        assert tex.count("{") == tex.count("}"), tex
        # match whole commands only: \leftarrow must not count as \left
        assert len(re.findall(r"\\left(?![a-zA-Z])", tex)) == len(re.findall(r"\\right(?![a-zA-Z])", tex)), tex
        assert not re.search(r"(?<!\\)\$", tex), "no dollar signs inside st.latex input"


def test_get_lesson():
    assert lessons.get_lesson("attention").title == "Self-attention"
    with pytest.raises(KeyError, match="unknown lesson"):
        lessons.get_lesson("nope")
