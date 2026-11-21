"""Unit tests for dataset utilities."""
import pytest
import torch

from llm.tokenizer import CharTokenizer
from training import data


def test_bundled_corpus_loads_and_is_nonempty():
    text = data.load_bundled_corpus()
    assert len(text) > 3000 and "robot" in text


def test_find_and_load_input_files_recursive(tmp_path):
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "sub" / "deeper" / "c.TXT").write_text("gamma", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".gitkeep").write_text("")
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00bad")  # invalid UTF-8
    files = [p.name for p in data.find_input_files(tmp_path)]
    assert files == ["a.txt", "bad.txt", "b.md", "c.TXT"] or set(files) == {"a.txt", "bad.txt", "b.md", "c.TXT"}
    texts = data.load_input_texts(tmp_path)
    assert set(texts.values()) == {"alpha", "beta", "gamma"}  # invalid file skipped


def test_missing_input_dir_is_empty(tmp_path):
    assert data.find_input_files(tmp_path / "nope") == []
    assert data.load_input_texts(tmp_path / "nope") == {}


def test_build_corpus_combinations(tmp_path):
    (tmp_path / "x.txt").write_text("custom text", encoding="utf-8")
    both = data.build_corpus(True, True, tmp_path)
    assert "robot" in both and "custom text" in both
    assert data.build_corpus(False, True, tmp_path) == "custom text"
    assert "custom text" not in data.build_corpus(True, False, tmp_path)


def test_build_corpus_without_any_text_raises(tmp_path):
    with pytest.raises(ValueError):
        data.build_corpus(False, True, tmp_path)
    with pytest.raises(ValueError):
        data.build_corpus(False, False, tmp_path)


def test_encode_corpus_dtype():
    tok = CharTokenizer.train("abc")
    ids = data.encode_corpus("abcabc", tok)
    assert ids.dtype == torch.long and ids.shape == (6,)


def test_train_val_split_contiguous():
    d = torch.arange(100)
    tr, va = data.train_val_split(d, 0.2)
    assert len(tr) == 80 and len(va) == 20
    assert torch.equal(torch.cat([tr, va]), d)


def test_train_val_split_validation():
    with pytest.raises(ValueError):
        data.train_val_split(torch.arange(10), 0.0)
    with pytest.raises(ValueError):
        data.train_val_split(torch.arange(10), 1.0)
    with pytest.raises(ValueError):
        data.train_val_split(torch.zeros(2, 2), 0.1)
    assert len(data.train_val_split(torch.arange(5), 0.01)[1]) == 1  # at least one val token


def test_get_batch_shapes_and_shift_by_one():
    d = torch.arange(200)
    x, y = data.get_batch(d, batch_size=8, block_size=16, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 16)
    assert torch.equal(y, x + 1)  # on arange data target = input shifted by one
    assert x.max() < 199 and y.max() <= 199


def test_get_batch_reproducible():
    d = torch.arange(500)
    a = data.get_batch(d, 4, 8, torch.Generator().manual_seed(1))
    b = data.get_batch(d, 4, 8, torch.Generator().manual_seed(1))
    assert torch.equal(a[0], b[0])


def test_get_batch_minimum_length_and_errors():
    d = torch.arange(9)
    x, y = data.get_batch(d, 2, 8)  # exactly block_size + 1 tokens: the only window
    assert torch.equal(x[0], torch.arange(8)) and torch.equal(y[0], torch.arange(1, 9))
    with pytest.raises(ValueError):
        data.get_batch(torch.arange(8), 2, 8)
    with pytest.raises(ValueError):
        data.get_batch(d, 0, 4)
    with pytest.raises(ValueError):
        data.get_batch(d, 2, 0)
