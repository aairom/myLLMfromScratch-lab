"""Dataset utilities: from raw text to ``(input, target)`` training batches.

A language model is trained on *shifted* sequences. For a window of ``T + 1``
consecutive token ids ``[t0, t1, ..., tT]``:

    x = [t0, ..., t(T-1)]      (what the model sees)
    y = [t1, ..., tT]          (the next token at every position)

so one window gives ``T`` training examples at once ("predict the next token").
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import config

SAMPLE_CORPUS_PATH = config.PROJECT_ROOT / "data" / "sample_corpus.txt"
TEXT_SUFFIXES = {".txt", ".md"}
IGNORED_NAMES = {".gitkeep", ".DS_Store"}


def load_bundled_corpus(path: Path = SAMPLE_CORPUS_PATH) -> str:
    """Return the small demo corpus shipped with the project."""
    return Path(path).read_text(encoding="utf-8")


def find_input_files(input_dir: Optional[Path] = None) -> list[Path]:
    """List text files under ``input_dir`` recursively (sorted, hidden helpers skipped)."""
    root = Path(input_dir) if input_dir is not None else config.load_settings().input_dir
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.name not in IGNORED_NAMES and p.suffix.lower() in TEXT_SUFFIXES
    )


def load_input_texts(input_dir: Optional[Path] = None) -> dict[str, str]:
    """Read every text file below ``input_dir``; keys are paths relative to it.

    Files that are not valid UTF-8 are skipped rather than aborting the load.
    """
    root = Path(input_dir) if input_dir is not None else config.load_settings().input_dir
    texts: dict[str, str] = {}
    for path in find_input_files(root):
        try:
            texts[str(path.relative_to(root))] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return texts


def build_corpus(
    use_bundled: bool = True,
    use_input: bool = True,
    input_dir: Optional[Path] = None,
) -> str:
    """Concatenate the selected sources, separated by a blank line."""
    parts: list[str] = []
    if use_bundled:
        parts.append(load_bundled_corpus())
    if use_input:
        parts.extend(load_input_texts(input_dir).values())
    corpus = "\n\n".join(p.strip("\n") for p in parts if p.strip())
    if not corpus:
        raise ValueError("no training text available: enable the bundled corpus or add .txt files to input/")
    return corpus


def encode_corpus(text: str, tokenizer) -> torch.Tensor:
    """Tokenize ``text`` into a 1-D ``int64`` tensor of ids."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def train_val_split(data: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a token stream into contiguous train / validation parts.

    The split is contiguous (not shuffled) so validation text is genuinely
    unseen; a shuffled split would leak neighbouring tokens into training.
    """
    if data.dim() != 1:
        raise ValueError("data must be a 1-D tensor of token ids")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n_val = max(1, int(len(data) * val_fraction))
    return data[: len(data) - n_val], data[len(data) - n_val :]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random windows; return ``(x, y)`` of shape ``(B, T)``."""
    if batch_size <= 0 or block_size <= 0:
        raise ValueError("batch_size and block_size must be positive")
    if len(data) < block_size + 1:
        raise ValueError(
            f"data has {len(data)} tokens but needs at least block_size + 1 = {block_size + 1}; "
            "use more text, a smaller context window, or a larger vocabulary (BPE)"
        )
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[s : s + block_size] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + 1 + block_size] for s in starts.tolist()])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y
