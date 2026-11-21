"""Unit tests for checkpoint saving and loading."""
import pytest
import torch

from llm.tokenizer import BPETokenizer, CharTokenizer
from llm.transformer import GPT, GPTConfig
from training.checkpoint import (
    CheckpointError,
    list_checkpoints,
    load_checkpoint,
    new_checkpoint_path,
    save_checkpoint,
)
from training.optimizers import build_optimizer
from training.trainer import TrainConfig


def make_model(tok, positional="sinusoidal"):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=tok.vocab_size, d_model=16, n_heads=2, n_layers=1, max_seq_len=16, positional=positional)
    return GPT(cfg)


@pytest.mark.parametrize("positional", ["sinusoidal", "learned"])
def test_roundtrip_gives_identical_logits(tmp_path, positional):
    tok = CharTokenizer.train("hello world")
    model = make_model(tok, positional).eval()
    ids = torch.tensor([tok.encode("hello")])
    path = save_checkpoint(model, tok, name="demo", output_dir=tmp_path)
    loaded = load_checkpoint(path)
    assert torch.equal(loaded["model"](ids), model(ids))
    assert loaded["tokenizer"].itos == tok.itos
    assert not loaded["model"].training
    assert loaded["train_config"] is None and loaded["history"] == []


def test_roundtrip_with_bpe_history_and_optimizer(tmp_path):
    tok = BPETokenizer.train("low lower lowest low low", 5)
    model = make_model(tok)
    opt = build_optimizer("adamw", model.parameters(), 1e-3)
    ids = torch.tensor([tok.encode("low low")])
    _, loss = model(ids, ids)
    loss.backward()
    opt.step()
    history = [{"step": 1, "train_loss": 2.5, "val_loss": None}]
    cfg = TrainConfig(steps=7)
    path = save_checkpoint(model, tok, cfg, history, opt, output_dir=tmp_path)
    loaded = load_checkpoint(path)
    assert loaded["train_config"] == cfg
    assert loaded["history"] == history
    assert loaded["optimizer_state"] is not None
    assert loaded["tokenizer"].encode("lower") == tok.encode("lower")
    fresh = build_optimizer("adamw", loaded["model"].parameters(), 1e-3)
    fresh.load_state_dict(loaded["optimizer_state"])  # resumable


def test_names_are_timestamped_unique_and_sanitised(tmp_path):
    first = new_checkpoint_path("my model/../x", tmp_path)
    assert first.parent == tmp_path and first.suffix == ".pt"
    assert "/" not in first.name and first.name[:8].isdigit() and first.name[8] == "_"
    first.write_bytes(b"x")
    second = new_checkpoint_path("my model/../x", tmp_path)
    assert second != first


def test_new_checkpoint_path_creates_output_dir(tmp_path):
    target = tmp_path / "does" / "not" / "exist"
    new_checkpoint_path("m", target)
    assert target.is_dir()


def test_list_checkpoints_newest_first(tmp_path):
    for name in ("20240101_000000_a.pt", "20250101_000000_b.pt", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_checkpoints(tmp_path)] == ["20250101_000000_b.pt", "20240101_000000_a.pt"]
    assert list_checkpoints(tmp_path / "missing") == []


def test_missing_corrupted_and_incomplete_files(tmp_path):
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope.pt")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a torch file")
    with pytest.raises(CheckpointError):
        load_checkpoint(bad)
    incomplete = tmp_path / "incomplete.pt"
    torch.save({"version": 1}, incomplete)
    with pytest.raises(CheckpointError, match="missing keys"):
        load_checkpoint(incomplete)
    not_dict = tmp_path / "list.pt"
    torch.save([1, 2, 3], not_dict)
    with pytest.raises(CheckpointError):
        load_checkpoint(not_dict)


def test_version_mismatch_and_inconsistent_contents(tmp_path):
    tok = CharTokenizer.train("abc")
    model = make_model(tok)
    path = save_checkpoint(model, tok, output_dir=tmp_path)
    raw = torch.load(path, weights_only=True)

    raw["version"] = 99
    wrong_version = tmp_path / "v.pt"
    torch.save(raw, wrong_version)
    with pytest.raises(CheckpointError, match="version"):
        load_checkpoint(wrong_version)

    raw["version"] = 1
    raw["tokenizer"] = CharTokenizer.train("abcdefgh").to_dict()  # vocab size no longer matches
    mismatch = tmp_path / "m.pt"
    torch.save(raw, mismatch)
    with pytest.raises(CheckpointError, match="vocabulary"):
        load_checkpoint(mismatch)

    raw["tokenizer"] = tok.to_dict()
    raw["model_state"] = {"nonsense": torch.zeros(1)}
    broken = tmp_path / "w.pt"
    torch.save(raw, broken)
    with pytest.raises(CheckpointError, match="inconsistent"):
        load_checkpoint(broken)


def test_explicit_path_is_used(tmp_path):
    tok = CharTokenizer.train("abc")
    target = tmp_path / "sub" / "fixed.pt"
    assert save_checkpoint(make_model(tok), tok, path=target) == target and target.is_file()
