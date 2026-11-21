"""Tokenizers: turning raw text into integer ids and back.

A language model only understands numbers. A *tokenizer* defines the mapping
between text and a sequence of integer ids drawn from a fixed *vocabulary*.

Two strategies are implemented from scratch:

* :class:`CharTokenizer` - one token per character. Tiny vocabulary, long
  sequences, nothing to learn.
* :class:`BPETokenizer`  - Byte-Pair Encoding. Starts from characters and
  repeatedly merges the most frequent adjacent pair into a new token, so common
  words become single tokens. Shorter sequences, larger vocabulary.

Both share the same small interface (``encode``, ``decode``, ``tokens``,
``vocab_size``, ``to_dict``) so the rest of the project is tokenizer agnostic.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

UNK = "<unk>"  # Id 0 in every vocabulary: stands for "character never seen".
UNK_ID = 0
UNK_REPLACEMENT = "\ufffd"  # What decode() emits for an unknown id.

# GPT-2 style pre-tokenization: a word (with its leading space), a run of
# punctuation, or a run of whitespace. Splitting first stops merges from
# crossing word boundaries and keeps training fast. The pieces always
# concatenate back to the original text, so the process is lossless.
_PRETOKEN_RE = re.compile(r"\s?\w+|\s?[^\w\s]+|\s+")


def pretokenize(text: str) -> list[str]:
    """Split ``text`` into word-like chunks (lossless: ``"".join`` restores it)."""
    return _PRETOKEN_RE.findall(text)


# --------------------------------------------------------------------------- #
# Character-level tokenizer
# --------------------------------------------------------------------------- #
class CharTokenizer:
    """Maps every distinct character of a corpus to an integer id."""

    kind = "char"

    def __init__(self, chars: Iterable[str] = ()):
        # Vocabulary order is deterministic: <unk> first, then sorted characters.
        self.itos: list[str] = [UNK] + sorted(set(chars) - {UNK})
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        """Build the vocabulary from every character that appears in ``text``."""
        return cls(set(text))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokens(self, text: str) -> list[str]:
        """Return the token strings for ``text`` (unknown chars shown as ``<unk>``)."""
        return [c if c in self.stoi else UNK for c in text]

    def encode(self, text: str) -> list[int]:
        """Text -> list of ids. Unknown characters map to ``UNK_ID``."""
        return [self.stoi.get(c, UNK_ID) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        """List of ids -> text. Out-of-range ids raise ``ValueError``."""
        out = []
        for i in ids:
            if not 0 <= i < len(self.itos):
                raise ValueError(f"token id {i} outside vocabulary of size {len(self.itos)}")
            out.append(UNK_REPLACEMENT if i == UNK_ID else self.itos[i])
        return "".join(out)

    def to_dict(self) -> dict:
        return {"type": self.kind, "vocab": self.itos}

    @classmethod
    def from_dict(cls, state: dict) -> "CharTokenizer":
        tok = cls()
        tok.itos = list(state["vocab"])
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        return tok


# --------------------------------------------------------------------------- #
# Byte-Pair Encoding tokenizer
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MergeStep:
    """One BPE training iteration, kept so the UI can replay the algorithm."""

    step: int  # 1-based iteration number
    pair: tuple[str, str]  # the two tokens that were merged
    count: int  # how often the pair occurred in the corpus
    new_token: str  # the token created by the merge
    top_pairs: tuple[tuple[tuple[str, str], int], ...]  # runner-ups, for context


def _merge_word(word: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every adjacent occurrence of ``pair`` inside ``word`` by its concatenation."""
    merged: list[str] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    """Byte-Pair Encoding tokenizer trained on a text corpus.

    Algorithm (training)
        1. Start with every word split into single characters.
        2. Count all adjacent pairs, weighted by word frequency.
        3. Merge the most frequent pair into a new token; record the merge.
        4. Repeat until ``num_merges`` merges were made or no pair is left.

    Algorithm (encoding)
        Split text into words, then apply the recorded merges in the order they
        were learned (earlier merges have priority).
    """

    kind = "bpe"

    def __init__(self) -> None:
        self.itos: list[str] = [UNK]
        self.stoi: dict[str, int] = {UNK: UNK_ID}
        self.merges: list[tuple[str, str]] = []
        self.trace: list[MergeStep] = []  # only populated by train()
        self._ranks: dict[tuple[str, str], int] = {}
        self._cache: dict[str, tuple[str, ...]] = {}

    # ----- training -------------------------------------------------------- #
    @classmethod
    def train(cls, text: str, num_merges: int, top_k: int = 5) -> "BPETokenizer":
        """Learn up to ``num_merges`` merges from ``text``."""
        if num_merges < 0:
            raise ValueError("num_merges must be >= 0")
        tok = cls()
        for ch in sorted(set(text)):
            tok._add_token(ch)

        # Work on unique words with their frequency: much faster than raw text.
        word_freq = Counter(pretokenize(text))
        words = {tuple(w): f for w, f in word_freq.items()}

        for step in range(1, num_merges + 1):
            pair_counts: Counter = Counter()
            for word, freq in words.items():
                for a, b in zip(word, word[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            # Highest count wins; ties broken alphabetically for determinism.
            ranked = sorted(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            best, best_count = ranked[0]
            new_token = best[0] + best[1]

            tok.merges.append(best)
            tok._add_token(new_token)
            tok.trace.append(
                MergeStep(step, best, best_count, new_token, tuple(ranked[:top_k]))
            )
            words = _apply(words, best)

        tok._refresh_ranks()
        return tok

    # ----- vocabulary helpers --------------------------------------------- #
    def _add_token(self, token: str) -> None:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def _refresh_ranks(self) -> None:
        self._ranks = {pair: r for r, pair in enumerate(self.merges)}
        self._cache = {}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # ----- encoding / decoding -------------------------------------------- #
    def _bpe_word(self, word: str) -> tuple[str, ...]:
        """Apply learned merges to a single pre-token (memoised)."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        symbols = list(word)
        while len(symbols) > 1:
            # Find the adjacent pair with the best (lowest) merge rank.
            best_rank, best_idx = None, -1
            for i in range(len(symbols) - 1):
                rank = self._ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_idx = rank, i
            if best_rank is None:
                break
            pair = self.merges[best_rank]
            symbols = list(_merge_word(tuple(symbols), pair))
        result = tuple(symbols)
        self._cache[word] = result
        return result

    def tokens(self, text: str) -> list[str]:
        """Return the token strings for ``text`` (unknown pieces shown as ``<unk>``)."""
        out: list[str] = []
        for word in pretokenize(text):
            for piece in self._bpe_word(word):
                out.append(piece if piece in self.stoi else UNK)
        return out

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, UNK_ID) for t in self.tokens(text)]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            if not 0 <= i < len(self.itos):
                raise ValueError(f"token id {i} outside vocabulary of size {len(self.itos)}")
            out.append(UNK_REPLACEMENT if i == UNK_ID else self.itos[i])
        return "".join(out)

    # ----- persistence ------------------------------------------------------ #
    def to_dict(self) -> dict:
        return {
            "type": self.kind,
            "vocab": self.itos,
            "merges": [list(m) for m in self.merges],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "BPETokenizer":
        tok = cls()
        tok.itos = list(state["vocab"])
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        tok.merges = [tuple(m) for m in state["merges"]]
        tok._refresh_ranks()
        return tok


def _apply(words: dict[tuple[str, ...], int], pair: tuple[str, str]) -> dict[tuple[str, ...], int]:
    """Apply one merge to every word; words that become identical are summed."""
    out: dict[tuple[str, ...], int] = {}
    for word, freq in words.items():
        new_word = _merge_word(word, pair)
        out[new_word] = out.get(new_word, 0) + freq
    return out


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def tokenizer_from_dict(state: dict):
    """Rebuild a tokenizer from the output of ``to_dict`` (used by checkpoints)."""
    kind = state.get("type")
    if kind == CharTokenizer.kind:
        return CharTokenizer.from_dict(state)
    if kind == BPETokenizer.kind:
        return BPETokenizer.from_dict(state)
    raise ValueError(f"unknown tokenizer type: {kind!r}")


def build_tokenizer(kind: str, text: str, num_merges: int = 100):
    """Create and train a tokenizer by name (``"char"`` or ``"bpe"``)."""
    if kind == CharTokenizer.kind:
        return CharTokenizer.train(text)
    if kind == BPETokenizer.kind:
        return BPETokenizer.train(text, num_merges)
    raise ValueError(f"unknown tokenizer kind: {kind!r}")
