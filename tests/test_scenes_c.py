"""Unit tests for animgen/scenes_c.py (training pipeline, visualization)."""
import pytest

from animgen import scenes_c
from animgen.canvas import Canvas
from animgen.scene import preview


@pytest.mark.parametrize("scene", scenes_c.SCENES, ids=lambda s: s.key)
def test_every_step_draws_a_distinct_frame(scene, anim_ctx, anim_opt):
    shots = [preview(scene, anim_ctx, anim_opt, step).tobytes() for step in range(scene.steps)]
    assert len(set(shots)) == scene.steps


@pytest.mark.parametrize("step", [0, 3, 4, 5])
def test_training_steps_move_inside_the_step(step, anim_ctx, anim_opt):
    s = scenes_c.TRAINING
    assert preview(s, anim_ctx, anim_opt, step, 0.2).tobytes() != preview(s, anim_ctx, anim_opt, step, 1.0).tobytes()


def test_training_scene_has_six_steps_for_sub_task_six():
    assert (scenes_c.TRAINING.number, scenes_c.TRAINING.steps) == (6, 6)
    assert (scenes_c.VISUALIZATION.number, scenes_c.VISUALIZATION.steps) == (7, 4)


def test_clip_text_shortens_to_fit():
    c = Canvas(800, 450, 1)
    assert scenes_c.clip_text(c, "short", 300) == "short"
    clipped = scenes_c.clip_text(c, "a very long sentence that cannot fit", 80)
    assert clipped.endswith("...") and c.text_width(clipped, 12) <= 80


def test_loss_range_pads_train_and_validation_values():
    lo, hi = scenes_c._loss_range([{"train_loss": 2.0, "val_loss": None}, {"train_loss": 3.0, "val_loss": 3.5}])
    assert lo == pytest.approx(1.9) and hi == pytest.approx(3.6)


def test_vowel_class_colours():
    assert scenes_c._vowel_class("a")[0] == "vowel" and scenes_c._vowel_class("z")[0] == "consonant"
    assert scenes_c._vowel_class(" ")[0].startswith("space")
