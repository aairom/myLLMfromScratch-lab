"""Unit tests for docsgen/catalog.py."""
import pytest

from docsgen import catalog


def test_ten_sub_tasks_numbered_one_to_ten():
    assert [s.number for s in catalog.SUBTASKS] == list(range(1, 11))
    assert len({s.slug for s in catalog.SUBTASKS}) == 10


def test_file_names_follow_the_convention():
    sub = catalog.get(6)
    assert sub.stem == "06_training_pipeline"
    assert sub.explanation_md == "06_training_pipeline_explanation.md"
    assert sub.explanation_html == "06_training_pipeline_explanation.html"
    assert sub.guide_src == "guide_src/06_training_pipeline_guide.md"
    assert sub.guide_html == "06_training_pipeline_implementation_guide.html"


def test_get_unknown_number_raises():
    with pytest.raises(KeyError, match="no sub-task 11"):
        catalog.get(11)


def test_neighbours_at_both_ends_and_in_the_middle():
    assert catalog.neighbours(1) == (None, catalog.get(2))
    assert catalog.neighbours(10) == (catalog.get(9), None)
    prev, nxt = catalog.neighbours(5)
    assert (prev.number, nxt.number) == (4, 6)
