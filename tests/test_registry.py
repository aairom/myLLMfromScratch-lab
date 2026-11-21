"""Unit tests for animgen/registry.py."""
import pytest

from animgen import registry


def test_all_scenes_are_unique_and_in_teaching_order():
    keys = [s.key for s in registry.ALL_SCENES]
    assert keys == sorted(keys) and len(set(keys)) == len(keys) == 11
    assert sorted({s.number for s in registry.ALL_SCENES}) == list(range(1, 11))


def test_every_sub_task_has_an_animation_and_a_catalog_entry():
    from docsgen import catalog

    assert {s.number for s in registry.ALL_SCENES} == {s.number for s in catalog.SUBTASKS}
    for s in registry.ALL_SCENES:
        assert s.key.startswith(f"{s.number:02d}_") and s.filename.endswith(".gif")


def test_find_by_number_and_by_name():
    assert [s.key for s in registry.find("5")] == ["05_gpt_model", "05_gpt_model_generation"]
    assert [s.key for s in registry.find("Attention")] == ["04_attention_layers"]
    assert registry.find("zzz") == []


def test_select_without_tokens_returns_everything():
    assert registry.select() == list(registry.ALL_SCENES) and registry.select([]) == list(registry.ALL_SCENES)


def test_select_unions_tokens_without_duplicates_in_order():
    keys = [s.key for s in registry.select(["9", "tokenization", "9", "2"])]
    assert keys == ["02_tokenization", "09_streamlit_app"]


def test_select_reports_a_token_that_matches_nothing():
    with pytest.raises(ValueError, match="nothing|no animation"):
        registry.select(["99"])
