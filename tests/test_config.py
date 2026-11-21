"""Unit tests for the configuration loader."""
import pytest

import config


def test_defaults(monkeypatch):
    for key in ("APP_PORT", "DEVICE", "DEFAULT_SEED", "INPUT_DIR", "OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    s = config.load_settings()
    assert s.app_port == 8601
    assert s.device == "auto"
    assert s.default_seed == 42
    assert s.input_dir == config.PROJECT_ROOT / "input"
    assert s.output_dir == config.PROJECT_ROOT / "output"


def test_port_5000_rejected(monkeypatch):
    monkeypatch.setenv("APP_PORT", "5000")
    with pytest.raises(ValueError):
        config.load_settings()


def test_invalid_device_rejected(monkeypatch):
    monkeypatch.setenv("DEVICE", "tpu")
    with pytest.raises(ValueError):
        config.load_settings()


def test_absolute_dir_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    assert config.load_settings().output_dir == tmp_path


def test_resolve_device_cpu_and_fallback():
    assert config.resolve_device("cpu").type == "cpu"
    # auto must always yield a usable device type
    assert config.resolve_device("auto").type in ("cpu", "mps", "cuda")
