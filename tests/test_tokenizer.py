"""Unit tests for the character and BPE tokenizers."""
import pytest

from llm.tokenizer import (
    UNK_ID,
    BPETokenizer,
    CharTokenizer,
    build_tokenizer,
    pretokenize,
    tokenizer_from_dict,
)

TEXT = "low lower lowest low low newer newer wider"


# ----- character tokenizer ------------------------------------------------- #
def test_char_roundtrip():
    tok = CharTokenizer.train(TEXT)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_char_vocab_is_sorted_with_unk_first():
    tok = CharTokenizer.train("ba ab")
    assert tok.itos == ["<unk>", " ", "a", "b"]
    assert tok.vocab_size == 4


def test_char_unknown_maps_to_unk():
    tok = CharTokenizer.train("abc")
    assert tok.encode("abz") == [tok.stoi["a"], tok.stoi["b"], UNK_ID]
    assert tok.tokens("z") == ["<unk>"]
    assert tok.decode([UNK_ID]) == "\ufffd"


def test_char_empty_input():
    tok = CharTokenizer.train("")
    assert tok.vocab_size == 1
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_char_decode_out_of_range():
    tok = CharTokenizer.train("ab")
    with pytest.raises(ValueError):
        tok.decode([99])
    with pytest.raises(ValueError):
        tok.decode([-1])


def test_char_is_deterministic():
    assert CharTokenizer.train("hello").itos == CharTokenizer.train("olleh").itos


# ----- pretokenizer -------------------------------------------------------- #
def test_pretokenize_is_lossless():
    text = "Hello,  world!\nSecond line... ok?"
    assert "".join(pretokenize(text)) == text


# ----- BPE ------------------------------------------------------------------ #
def test_bpe_roundtrip():
    tok = BPETokenizer.train(TEXT, num_merges=10)
    assert tok.decode(tok.encode(TEXT)) == TEXT


def test_bpe_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train(TEXT, num_merges=1)
    step = tok.trace[0]
    assert step.step == 1
    assert step.count == max(c for _, c in step.top_pairs)
    assert step.new_token == "".join(step.pair)
    assert step.new_token in tok.stoi


def test_bpe_more_merges_shorter_sequences():
    few = BPETokenizer.train(TEXT, num_merges=2)
    many = BPETokenizer.train(TEXT, num_merges=20)
    assert len(many.encode(TEXT)) <= len(few.encode(TEXT))
    assert many.vocab_size > few.vocab_size


def test_bpe_zero_merges_equals_characters():
    tok = BPETokenizer.train(TEXT, num_merges=0)
    assert tok.merges == []
    assert len(tok.encode(TEXT)) == len(TEXT)


def test_bpe_stops_when_no_pairs_left():
    tok = BPETokenizer.train("ab", num_merges=50)
    assert len(tok.merges) == 1  # only "a"+"b" can ever be merged


def test_bpe_merge_order_deterministic():
    a = BPETokenizer.train(TEXT, num_merges=8)
    b = BPETokenizer.train(TEXT, num_merges=8)
    assert a.merges == b.merges and a.itos == b.itos


def test_bpe_unknown_character():
    tok = BPETokenizer.train("abab", num_merges=2)
    ids = tok.encode("abz")
    assert UNK_ID in ids


def test_bpe_negative_merges_rejected():
    with pytest.raises(ValueError):
        BPETokenizer.train("abc", num_merges=-1)


def test_bpe_decode_out_of_range():
    tok = BPETokenizer.train("abab", num_merges=1)
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size])


# ----- serialisation / factory --------------------------------------------- #
@pytest.mark.parametrize("kind", ["char", "bpe"])
def test_serialisation_roundtrip(kind):
    tok = build_tokenizer(kind, TEXT, num_merges=12)
    clone = tokenizer_from_dict(tok.to_dict())
    assert clone.itos == tok.itos
    assert clone.encode(TEXT) == tok.encode(TEXT)


def test_factory_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_tokenizer("word", TEXT)
    with pytest.raises(ValueError):
        tokenizer_from_dict({"type": "word"})
