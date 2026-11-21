"""Unit tests for animgen/scenes_b.py (attention, GPT, generation)."""
import pytest

from animgen import scenes_b
from animgen.scene import preview


@pytest.mark.parametrize("scene", scenes_b.SCENES, ids=lambda s: s.key)
def test_every_step_draws_a_distinct_frame(scene, anim_ctx, anim_opt):
    shots = [preview(scene, anim_ctx, anim_opt, step).tobytes() for step in range(scene.steps)]
    assert len(set(shots)) == scene.steps


@pytest.mark.parametrize("scene", scenes_b.SCENES, ids=lambda s: s.key)
def test_motion_inside_a_step(scene, anim_ctx, anim_opt):
    step = scene.steps - 1
    assert preview(scene, anim_ctx, anim_opt, step, 0.1).tobytes() != preview(scene, anim_ctx, anim_opt, step, 1.0).tobytes()


def test_attention_has_the_seven_stages_of_the_layer():
    assert scenes_b.ATTENTION.steps == 7 and scenes_b.ATTENTION.number == 4
    text = " ".join(scenes_b.ATTENTION.captions)
    for word in ("softmax", "LayerNorm", "feed-forward", "mask"):
        assert word.lower() in text.lower()


def test_generation_gif_belongs_to_sub_task_five():
    assert scenes_b.GENERATION.number == scenes_b.GPT_SCENE.number == 5
    assert scenes_b.GENERATION.key != scenes_b.GPT_SCENE.key


def test_helpers():
    import torch

    assert scenes_b._vmax(torch.tensor([1.0, -3.0, float("-inf")])) == 3.0
    assert scenes_b._vmax(torch.tensor([float("-inf")])) == 1.0
    rows = scenes_b._rows(torch.arange(8.0).reshape(4, 2), 0.5)
    assert rows == [[0.0, 1.0], [2.0, 3.0]] and scenes_b._rows(torch.zeros(3, 2), 0.0) == []
