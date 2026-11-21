"""Unit tests for animgen/cli.py."""
import io
from dataclasses import replace

import pytest
from PIL import Image

from animgen import cli, registry
from animgen.options import Options


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


def base(tmp_path) -> Options:
    return replace(Options(out_dir=tmp_path).quick(), hold_ms=100)


def test_flags_override_the_environment_defaults(tmp_path):
    args = parse("--width", "700", "--height", "400", "--frame-ms", "50", "--hold-ms", "10", "--steps", "7", "--phrase", "Hi there")
    o = cli.options_from_args(args, base(tmp_path))
    assert (o.width, o.height, o.frame_ms, o.hold_ms, o.train_steps, o.phrase) == (700, 400, 50, 10, 7, "Hi there")


def test_quick_applies_before_explicit_flags(tmp_path):
    o = cli.options_from_args(parse("--quick", "--width", "720"), Options(out_dir=tmp_path))
    assert (o.width, o.height, o.supersample) == (720, 360, 1)


def test_out_folder_is_resolved(tmp_path):
    assert cli.options_from_args(parse("--out", str(tmp_path / "x")), base(tmp_path)).out_dir == tmp_path / "x"


def test_invalid_values_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        cli.options_from_args(parse("--width", "10"), base(tmp_path))


def test_list_prints_every_animation(tmp_path):
    out = io.StringIO()
    assert cli.run(parse("--list"), base(tmp_path), out) == 0
    text = out.getvalue()
    assert all(s.filename in text for s in registry.ALL_SCENES) and not list(tmp_path.iterdir())


def test_run_writes_valid_gifs_and_the_gallery(tmp_path):
    out = io.StringIO()
    code = cli.run(parse("--only", "1", "tokenization"), base(tmp_path), out)
    assert code == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["01_scaffolding.gif", "02_tokenization.gif", "index.html"]
    with Image.open(tmp_path / "02_tokenization.gif") as im:
        assert im.format == "GIF" and im.n_frames > 1 and im.size == (640, 360)
    assert "2/2 animations" in out.getvalue() and "KiB" in out.getvalue()


def test_no_gallery_flag(tmp_path):
    assert cli.run(parse("--only", "1", "--no-gallery"), base(tmp_path), io.StringIO()) == 0
    assert not (tmp_path / "index.html").exists()


def test_unknown_animation_and_bad_options_exit_with_2(tmp_path, capsys):
    assert cli.run(parse("--only", "99"), base(tmp_path), io.StringIO()) == 2
    assert cli.run(parse("--width", "5"), base(tmp_path), io.StringIO()) == 2
    assert "error:" in capsys.readouterr().err


def test_a_failing_scene_is_reported_and_the_rest_still_render(tmp_path, monkeypatch, capsys):
    real = cli.render_scene

    def flaky(scene, ctx, opt):
        if scene.number == 1:
            raise RuntimeError("boom")
        return real(scene, ctx, opt)

    monkeypatch.setattr(cli, "render_scene", flaky)
    code = cli.run(parse("--only", "1", "2"), base(tmp_path), io.StringIO())
    assert code == 1 and (tmp_path / "02_tokenization.gif").is_file() and not (tmp_path / "01_scaffolding.gif").exists()
    assert "FAILED 01_scaffolding.gif" in capsys.readouterr().err


def test_main_parses_argv(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "from_env", lambda: base(tmp_path))
    assert cli.main(["--only", "1", "--no-gallery"]) == 0 and (tmp_path / "01_scaffolding.gif").is_file()
