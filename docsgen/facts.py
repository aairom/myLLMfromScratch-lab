"""Numbers quoted in the documents, computed from the live code or the stored runs.

Markdown sources write ``{{fact:params}}`` instead of typing a number. At build
time the value is looked up here, so a changed model or corpus changes the
documents (and the tests that guard them) instead of silently leaving stale text.

Two groups of facts exist:

* *live* facts are computed now from the bundled corpus and the default model;
* *recorded* facts come from ``reference_run.json`` (the 300-step run of the
  Training lab defaults) and ``long_runs.json`` (two longer runs).
"""
from __future__ import annotations

import collections
import json
import math
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFERENCE_PATH = HERE / "reference_run.json"
LONG_RUNS_PATH = HERE / "long_runs.json"
VAL_FRACTION = 0.1  # same split as ui.lab.prepare_data


def _n(value: int) -> str:
    return f"{value:,}"


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _live() -> dict[str, str]:
    """Facts that depend on the code and the bundled corpus."""
    from llm.tokenizer import build_tokenizer
    from training.data import build_corpus, encode_corpus, train_val_split
    from ui.state import ModelSettings, build_model

    corpus = build_corpus(use_bundled=True, use_input=False)  # never depend on the user's input/
    char_tok = build_tokenizer("char", corpus)
    model, _ = build_model(ModelSettings(use_input=False))
    cfg = model.cfg
    breakdown = model.parameter_breakdown()
    out: dict[str, str] = {
        "corpus_chars": _n(len(corpus)),
        "char_vocab": str(char_tok.vocab_size),
        "ln_vocab": _f(math.log(char_tok.vocab_size)),
        "params": _n(model.num_parameters()),
        "params_embedding": _n(breakdown["token_embedding"]),
        "params_attention": _n(breakdown["attention"]),
        "params_ffn": _n(breakdown["feed_forward"]),
        "params_layernorm": _n(breakdown["layer_norm"]),
        "params_attn_layer": _n(breakdown["attention"] // cfg.n_layers),
        "params_ffn_layer": _n(breakdown["feed_forward"] // cfg.n_layers),
        "d_model": str(cfg.d_model),
        "n_heads": str(cfg.n_heads),
        "n_layers": str(cfg.n_layers),
        "context": str(cfg.max_seq_len),
        "d_k": str(cfg.d_model // cfg.n_heads),
        "ffn_hidden": str(cfg.ffn_mult * cfg.d_model),
        "res_std": _f(0.02 / math.sqrt(2 * cfg.n_layers), 3),
    }
    for merges in (100, 200):
        tok = build_tokenizer("bpe", corpus, merges)
        n_tokens = len(tok.encode(corpus))
        out[f"bpe{merges}_vocab"] = str(tok.vocab_size)
        out[f"bpe{merges}_tokens"] = _n(n_tokens)
        out[f"bpe{merges}_ratio"] = _f(len(corpus) / n_tokens, 2)
        if merges == 100:
            first = tok.trace[0]
            out["bpe_first_merge"] = f"{first.pair[0]!r} + {first.pair[1]!r} -> {first.new_token!r} ({first.count} times)"

    data = encode_corpus(corpus, char_tok)
    train, val = train_val_split(data, VAL_FRACTION)
    out["train_tokens"] = _n(len(train))
    out["val_tokens"] = _n(len(val))
    counts = collections.Counter(train.tolist())
    total = len(train)
    out["unigram_train"] = _f(-sum(c / total * math.log(c / total) for c in counts.values()))
    out["unigram_val"] = _f(-sum(math.log(counts.get(int(i), 0.5) / total) for i in val) / len(val))
    return out


def _recorded() -> dict[str, str]:
    """Facts taken from the stored reference run and the two longer runs."""
    run = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    hist, cfg = run["history"], run["config"]
    vals = [(h["step"], h["val_loss"]) for h in hist if h.get("val_loss") is not None]
    batch_tokens = cfg["batch_size"] * cfg["block_size"]
    seconds = hist[-1]["elapsed"]
    plateau = [v for s, v in vals if 75 <= s <= 275]
    out = {
        "ref_steps": str(cfg["steps"]),
        "ref_batch": str(cfg["batch_size"]),
        "ref_block": str(cfg["block_size"]),
        "ref_lr": f"{cfg['lr']:g}",
        "ref_lr_final": f"{cfg['lr'] * cfg['min_lr_ratio']:g}",
        "ref_warmup": str(cfg["warmup_steps"]),
        "ref_clip": f"{cfg['grad_clip']:g}",
        "ref_batch_tokens": _n(batch_tokens),
        "ref_total_tokens": _n(batch_tokens * cfg["steps"]),
        "ref_epochs": _f(batch_tokens * cfg["steps"] / 4389, 1),
        "ref_first_train": _f(hist[0]["train_loss"]),
        "ref_last_train": _f(hist[-1]["train_loss"]),
        "ref_first_val": _f(vals[0][1]),
        "ref_last_val": _f(vals[-1][1]),
        "ref_val_plateau_lo": _f(min(plateau)),
        "ref_val_plateau_hi": _f(max(plateau)),
        "ref_ppl_first": _f(math.exp(hist[0]["train_loss"]), 1),
        "ref_ppl_last": _f(math.exp(hist[-1]["train_loss"]), 1),
        "ref_ppl_val": _f(math.exp(vals[-1][1]), 1),
        "ref_gradnorm_first": _f(hist[0]["grad_norm"]),
        "ref_gradnorm_last": _f(hist[-1]["grad_norm"]),
        "ref_seconds": _f(seconds, 1),
        "ref_tps": _n(int(round(batch_tokens * cfg["steps"] / seconds, -3))),
        "ref_device": run["device"],
    }
    longs = json.loads(LONG_RUNS_PATH.read_text(encoding="utf-8"))
    r15, r30 = longs["run_1500"], longs["run_3000"]
    out.update(
        {
            "long1500_train": _f(r15["train_last"]),
            "long1500_val": _f(r15["val_last"]),
            "long1500_val750": _f(r15["val_at"]["750"]),
            "long1500_val1000": _f(r15["val_at"]["1000"]),
            "long3000_train": _f(r30["train_last"]),
            "long3000_val_best": _f(r30["val_best"]),
            "long3000_val_best_step": str(r30["val_best_step"]),
            "long3000_val_last": _f(r30["val_last"]),
        }
    )
    return out


@lru_cache(maxsize=1)
def all_facts() -> dict[str, str]:
    """Every fact as display text (computed once per process)."""
    facts = _live()
    facts.update(_recorded())
    return facts


def get(name: str) -> str:
    """Look up one fact; unknown names raise ``KeyError`` listing the valid ones."""
    facts = all_facts()
    if name not in facts:
        raise KeyError(f"unknown fact {name!r}; known facts: {', '.join(sorted(facts))}")
    return facts[name]
