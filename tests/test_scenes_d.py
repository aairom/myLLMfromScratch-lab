"""Unit tests for animgen/scenes_d.py (educational content, Streamlit app, compliance)."""
import pytest

from animgen import scenes_d
from animgen.scene import preview


@pytest.mark.parametrize("scene", scenes_d.SCENES, ids=lambda s: s.key)
def test_every_step_draws_a_distinct_frame(scene, anim_ctx, anim_opt):
    shots = [preview(scene, anim_ctx, anim_opt, step).tobytes() for step in range(scene.steps)]
    assert len(set(shots)) == scene.steps


@pytest.mark.parametrize("scene", scenes_d.SCENES, ids=lambda s: s.key)
def test_first_frame_differs_from_last_frame_of_a_step(scene, anim_ctx, anim_opt):
    assert preview(scene, anim_ctx, anim_opt, 0, 0.1).tobytes() != preview(scene, anim_ctx, anim_opt, 0, 1.0).tobytes()


def test_sub_task_numbers_and_keys():
    assert [s.number for s in scenes_d.SCENES] == [8, 9, 10]
    assert [s.key for s in scenes_d.SCENES][0] == "08_educational_content"


def test_source_lines_come_from_the_real_code():
    lines = scenes_d.source_lines("llm.layers.gelu", 3)
    assert len(lines) == 3 and lines[0].startswith("def gelu")
    assert all(ln.strip() for ln in lines)


def test_app_pages_are_read_from_app_py():
    pages = scenes_d._app_pages()
    assert len(pages) == 10 and pages[0] == ("Home", "home")
