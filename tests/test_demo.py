"""Tests for ui/demo.py: the pure helpers behind every page."""
import pytest
import torch

from llm.tokenizer import build_tokenizer
from llm.transformer import GPT, GPTConfig
from ui import demo

TEXT = "the little robot read the book. the robot read."


@pytest.fixture(scope="module")
def char_tok():
    return build_tokenizer("char", TEXT)


@pytest.fixture(scope="module")
def bpe_tok():
    return build_tokenizer("bpe", TEXT * 3, 15)


@pytest.fixture(scope="module")
def model(char_tok):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=char_tok.vocab_size, d_model=16, n_heads=2, n_layers=2, max_seq_len=16))


# ----- tokenization ---------------------------------------------------------- #
def test_tokenize_text_lengths_match(char_tok):
    tokens, ids = demo.tokenize_text(char_tok, "the robot")
    assert len(tokens) == len(ids) == 9


def test_replay_merges_zero_is_characters():
    assert demo.replay_merges("low", [("l", "o")], 0) == ["l", "o", "w"]


def test_replay_merges_applies_in_order():
    merges = [("l", "o"), ("lo", "w")]
    assert demo.replay_merges("low", merges, 1) == ["lo", "w"]
    assert demo.replay_merges("low", merges, 2) == ["low"]
    assert demo.replay_merges("low", merges, 99) == ["low"]


def test_replay_merges_matches_tokenizer(bpe_tok):
    word = "robot"
    assert demo.replay_merges(word, bpe_tok.merges, len(bpe_tok.merges)) == list(bpe_tok._bpe_word(word))


def test_replay_merges_rejects_negative():
    with pytest.raises(ValueError):
        demo.replay_merges("a", [], -1)


def test_compression_counts_decrease_with_merges():
    counts = demo.compression_counts(TEXT * 3, [0, 5, 20])
    assert counts[0] > counts[5] > counts[20]
    assert list(counts) == [0, 5, 20]


# ----- embeddings / positions ---------------------------------------------- #
def test_clip_ids():
    assert demo.clip_ids([1, 2, 3], 2) == [1, 2]
    with pytest.raises(ValueError):
        demo.clip_ids([1], 0)


def test_embedding_demo_shapes(char_tok):
    out = demo.embedding_demo(char_tok, "the robot", d_model=8, max_len=5)
    assert out["vectors"].shape == (5, 8) and len(out["tokens"]) == 5
    assert out["table"].shape == (char_tok.vocab_size, 8)


def test_embedding_demo_lookup_equals_table_rows(char_tok):
    out = demo.embedding_demo(char_tok, "the", d_model=8)
    assert torch.equal(out["vectors"], out["table"][out["ids"]])


def test_embedding_demo_same_seed_same_table(char_tok):
    a = demo.embedding_demo(char_tok, "ab", 8, seed=3)["table"]
    b = demo.embedding_demo(char_tok, "ab", 8, seed=3)["table"]
    assert torch.equal(a, b)


def test_embedding_demo_scale_multiplies(char_tok):
    plain = demo.embedding_demo(char_tok, "the", 16, seed=1)["vectors"]
    scaled = demo.embedding_demo(char_tok, "the", 16, scale=True, seed=1)["vectors"]
    assert torch.allclose(scaled, plain * 4.0)


def test_embedding_demo_empty_text(char_tok):
    out = demo.embedding_demo(char_tok, "", d_model=8)
    assert out["vectors"].shape == (0, 8)


@pytest.mark.parametrize("kind", ["sinusoidal", "learned"])
def test_positional_demo_shape(kind):
    assert demo.positional_demo(kind, 12, 7).shape == (7, 12)


def test_embed_with_position_rejects_empty(char_tok):
    with pytest.raises(ValueError):
        demo.embed_with_position(char_tok, "", 8, "sinusoidal", 0, 10)


# ----- attention / layernorm / ffn ----------------------------------------- #
def test_attention_demo_shapes_and_causality(char_tok):
    out = demo.attention_demo(char_tok, "the robot", d_model=16, n_heads=4)
    t = len(out["tokens"])
    w = out["details"]["weights"]
    assert w.shape == (4, t, t) and out["output"].shape == (t, 16)
    assert torch.allclose(w.sum(-1), torch.ones(4, t), atol=1e-5)
    assert torch.all(torch.triu(w, diagonal=1) == 0)  # no peeking at the future


def test_attention_demo_bidirectional_sees_future(char_tok):
    w = demo.attention_demo(char_tok, "the robot", 16, 2, causal=False)["details"]["weights"]
    assert torch.triu(w, diagonal=1).sum() > 0


def test_attention_demo_invalid_heads(char_tok):
    with pytest.raises(ValueError):
        demo.attention_demo(char_tok, "the", d_model=10, n_heads=4)


def test_layernorm_demo_normalises(char_tok):
    out = demo.layernorm_demo(char_tok, "the robot", 32)
    normalized = out["stages"]["normalized"]
    assert torch.allclose(normalized.mean(-1), torch.zeros(normalized.size(0)), atol=1e-4)
    assert torch.allclose(normalized.std(-1, unbiased=False), torch.ones(normalized.size(0)), atol=1e-2)
    assert out["before"].mean() > 1.0  # input really was off-centre


def test_ffn_demo_hidden_width(char_tok):
    out = demo.ffn_demo(char_tok, "the", d_model=8, ffn_mult=3, activation="relu")
    assert out["hidden"].shape[-1] == 24 and out["output"].shape[-1] == 8
    assert out["hidden"].min() >= 0  # ReLU


# ----- full model ----------------------------------------------------------- #
def test_model_forward_details(model, char_tok):
    out = demo.model_forward(model, char_tok, "the robot")
    t = len(out["ids"])
    assert out["logits"].shape == (t, char_tok.vocab_size)
    assert len(out["details"]["blocks"]) == 2


def test_model_forward_crops_to_context(model, char_tok):
    out = demo.model_forward(model, char_tok, "the robot read the book " * 4)
    assert len(out["ids"]) == 16 and len(out["tokens"]) == 16


def test_model_forward_restores_training_mode(model, char_tok):
    model.train()
    demo.model_forward(model, char_tok, "the")
    assert model.training


def test_model_forward_rejects_empty(model, char_tok):
    with pytest.raises(ValueError):
        demo.model_forward(model, char_tok, "")


def test_top_next_tokens_sorted_and_normalised(char_tok):
    logits = torch.randn(char_tok.vocab_size)
    rows = demo.top_next_tokens(logits, char_tok, n=5)
    probs = [r["probability"] for r in rows]
    assert probs == sorted(probs, reverse=True) and len(rows) == 5


def test_top_next_tokens_top_k_and_greedy(char_tok):
    logits = torch.randn(char_tok.vocab_size)
    assert len(demo.top_next_tokens(logits, char_tok, top_k=3, n=10)) == 3
    greedy = demo.top_next_tokens(logits, char_tok, temperature=0)
    assert len(greedy) == 1 and greedy[0]["probability"] == 1.0


def test_generate_text_is_reproducible(model, char_tok):
    a = demo.generate_text(model, char_tok, "the ", 10, seed=5)
    b = demo.generate_text(model, char_tok, "the ", 10, seed=5)
    assert a == b and len(a["new_ids"]) == 10 and a["text"].startswith("the ")


def test_generate_text_empty_prompt(model, char_tok):
    out = demo.generate_text(model, char_tok, "", 3, seed=0)
    assert len(out["new_ids"]) == 3 and len(out["prompt_ids"]) == 1


def test_generate_text_greedy_is_deterministic(model, char_tok):
    a = demo.generate_text(model, char_tok, "the", 6, temperature=0)
    b = demo.generate_text(model, char_tok, "the", 6, temperature=0)
    assert a["new_ids"] == b["new_ids"]


def test_step_probabilities(model, char_tok):
    ids = char_tok.encode("the robot")
    rows = demo.step_probabilities(model, char_tok, ids)
    assert len(rows) == len(ids) - 1
    assert all(0.0 <= r["probability"] <= 1.0 and r["rank"] >= 1 for r in rows)


def test_step_probabilities_too_short(model, char_tok):
    assert demo.step_probabilities(model, char_tok, [1]) == []


def test_parameter_summary(model):
    s = demo.parameter_summary(model)
    assert s["total"] == model.num_parameters()
    assert s["megabytes"] == pytest.approx(s["total"] * 4 / 1e6)
