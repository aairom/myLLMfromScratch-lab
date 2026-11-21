"""Unit tests for animgen/scene.py."""
import pytest
from PIL import Image

from animgen import scene, theme
from animgen.canvas import Canvas
from animgen.scene import Frame, Scene, phase


def draw_dot(canvas: Canvas, ctx, step: int, t: float) -> None:
    canvas.circle(100 + 100 * step, 200, 10 + 10 * t, fill=theme.TOKEN)


def demo(frames_per_step=3) -> Scene:
    return Scene("99_demo", 9, "Sub-task 9", "Demo", ("one", "two"), draw_dot, frames_per_step)


def test_phase_is_complete_before_partial_now_and_empty_later():
    assert (phase(2, 0.3, 0), phase(2, 0.3, 2), phase(2, 0.3, 3)) == (1.0, 0.3, 0.0)


def test_scene_properties():
    s = demo()
    assert (s.filename, s.steps, s.total_frames()) == ("99_demo.gif", 2, 6)


def test_render_scene_holds_the_last_frame_of_each_step(anim_opt):
    frames = scene.render_scene(demo(), None, anim_opt)
    assert len(frames) == 6 and all(isinstance(f, Frame) for f in frames)
    durations = [f.duration_ms for f in frames]
    assert durations == [anim_opt.frame_ms] * 2 + [anim_opt.frame_ms + anim_opt.hold_ms] + [anim_opt.frame_ms] * 2 + [anim_opt.frame_ms + anim_opt.hold_ms]
    assert frames[0].image.size == (anim_opt.width, anim_opt.height) and frames[0].image.mode == "RGB"
    assert frames[0].image.tobytes() != frames[1].image.tobytes()  # the animation really moves


def test_a_scene_with_one_frame_per_step_still_has_two_tween_frames(anim_opt):
    assert len(scene.render_scene(demo(1), None, anim_opt)) == 4


def test_shared_palette_covers_all_frames():
    imgs = [Image.new("RGB", (40, 40), c) for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    pal = scene.shared_palette(imgs, samples=3)
    assert pal.mode == "P"


def test_save_gif_writes_a_looping_gif_with_durations(tmp_path, anim_opt):
    frames = scene.render_scene(demo(), None, anim_opt)
    path = tmp_path / "sub" / "demo.gif"
    size = scene.save_gif(frames, path)
    assert size == path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.format == "GIF" and im.n_frames > 1 and im.info["loop"] == 0
        assert im.size == (anim_opt.width, anim_opt.height)
        total = 0
        for i in range(im.n_frames):
            im.seek(i)
            total += im.info["duration"]
    assert total == sum(f.duration_ms for f in frames)


def test_save_gif_rejects_an_empty_animation(tmp_path):
    with pytest.raises(ValueError):
        scene.save_gif([], tmp_path / "x.gif")


def test_preview_defaults_to_the_last_step(anim_opt):
    last = scene.preview(demo(), None, anim_opt)
    first = scene.preview(demo(), None, anim_opt, 0, 0.0)
    assert last.size == first.size == (anim_opt.width, anim_opt.height)
    assert last.tobytes() != first.tobytes()
