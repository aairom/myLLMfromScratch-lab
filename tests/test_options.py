"""Unit tests for animgen/options.py."""
from dataclasses import replace
from pathlib import Path

import pytest

import config
from animgen import options
from animgen.options import Options


def make(**kw) -> Options:
    return Options(out_dir=Path("x"), **kw)


def test_defaults_are_valid():
    assert make().problems() == []
    assert make().validated().width == 800


@pytest.mark.parametrize("field,value", [("width", 100), ("height", 5000), ("frame_ms", 5), ("hold_ms", -1),
                                         ("supersample", 4), ("train_steps", 0), ("phrase", "  "), ("attn_tokens", 2)])
def test_every_bad_value_is_reported(field, value):
    problems = replace(make(), **{field: value}).problems()
    assert len(problems) == 1 and field.split("_")[0] in problems[0].lower()


def test_validated_raises_with_all_problems():
    with pytest.raises(ValueError) as err:
        replace(make(), width=1, height=1).validated()
    assert "width" in str(err.value) and "height" in str(err.value)


def test_quick_is_smaller_and_faster():
    q = make(train_steps=300).quick()
    assert (q.width, q.height, q.supersample, q.train_steps) == (640, 360, 1, 30)
    assert make(train_steps=10).quick().train_steps == 10  # never lengthens a short run
    assert q.problems() == []


def test_relative_output_folder_is_taken_from_the_project_root(tmp_path):
    assert options.resolve_out_dir("a/b") == config.PROJECT_ROOT / "a" / "b"
    assert options.resolve_out_dir(str(tmp_path)) == tmp_path


def test_from_env_reads_the_documented_variables(monkeypatch, tmp_path):
    for k, v in {"ANIM_OUTPUT_DIR": str(tmp_path), "ANIM_WIDTH": "640", "ANIM_HEIGHT": "360", "ANIM_FRAME_MS": "80",
                 "ANIM_TRAIN_STEPS": "12", "ANIM_PHRASE": "Hello there", "ANIM_FONT": "", "ANIM_FONT_BOLD": " "}.items():
        monkeypatch.setenv(k, v)
    o = options.from_env()
    assert (o.out_dir, o.width, o.height, o.frame_ms, o.train_steps, o.phrase) == (tmp_path, 640, 360, 80, 12, "Hello there")
    assert o.font is None and o.font_bold is None  # blank means "search the system"


def test_from_env_defaults_and_bad_integer(monkeypatch):
    for k in ("ANIM_OUTPUT_DIR", "ANIM_WIDTH", "ANIM_HEIGHT", "ANIM_FRAME_MS", "ANIM_TRAIN_STEPS", "ANIM_PHRASE"):
        monkeypatch.delenv(k, raising=False)
    o = options.from_env()
    assert o.out_dir == config.PROJECT_ROOT / options.DEFAULT_OUT and o.phrase == options.DEFAULT_PHRASE
    monkeypatch.setenv("ANIM_WIDTH", "wide")
    with pytest.raises(ValueError, match="ANIM_WIDTH"):
        options.from_env()
