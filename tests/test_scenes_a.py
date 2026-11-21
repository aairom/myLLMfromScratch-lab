"""Unit tests for animgen/scenes_a.py (scaffolding, tokenization, embeddings)."""
import pytest

from animgen import scenes_a
from animgen.scene import preview, render_scene


@pytest.mark.parametrize("scene", scenes_a.SCENES, ids=lambda s: s.key)
def test_every_step_draws_a_distinct_frame(scene, anim_ctx, anim_opt):
    shots = [preview(scene, anim_ctx, anim_opt, step).tobytes() for step in range(scene.steps)]
    assert len(set(shots)) == scene.steps  # each step shows something new


@pytest.mark.parametrize("scene", scenes_a.SCENES, ids=lambda s: s.key)
def test_motion_inside_a_step(scene, anim_ctx, anim_opt):
    assert preview(scene, anim_ctx, anim_opt, 1, 0.0).tobytes() != preview(scene, anim_ctx, anim_opt, 1, 1.0).tobytes()


def test_scene_metadata_matches_the_sub_tasks():
    assert [s.number for s in scenes_a.SCENES] == [1, 2, 3]
    assert [s.key for s in scenes_a.SCENES] == ["01_scaffolding", "02_tokenization", "03_embeddings_positional"]
    assert all(len(s.captions) == s.steps >= 3 for s in scenes_a.SCENES)


def test_a_whole_scene_renders(anim_ctx, anim_opt):
    frames = render_scene(scenes_a.TOKENIZATION, anim_ctx, anim_opt)
    assert len(frames) == scenes_a.TOKENIZATION.total_frames()
