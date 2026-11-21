"""Everything the scenes draw, computed once from the project's real code.

The context trains the default model for ``train_steps`` steps with the project's
own trainer, so the loss curve, learning-rate schedule and generated text in the
GIFs are the genuine output of ``training.trainer.train``. Properties are lazy:
generating a single GIF only pays for what that GIF needs.
"""
from __future__ import annotations

import dataclasses
from functools import cached_property

import torch

from animgen.options import Options
from llm.attention import causal_mask
from llm.positional import sinusoidal_table
from llm.tokenizer import BPETokenizer, CharTokenizer
from llm.transformer import GPT
from training.data import encode_corpus, get_batch, load_bundled_corpus, train_val_split
from training.optimizers import Adam
from training.trainer import TrainConfig, train
from ui import state as ui_state

BPE_DEMO_MERGES = 12
MAX_TRACE_STEPS = 8


class Context:
    """Lazily computed facts shared by all scenes."""

    def __init__(self, options: Options) -> None:
        self.opt = options
        self.device = torch.device("cpu")  # deterministic and fast enough for this small model

    @cached_property
    def project(self) -> "ProjectFacts":
        return ProjectFacts()

    # ----- data ------------------------------------------------------------- #
    @cached_property
    def corpus(self) -> str:
        return load_bundled_corpus()

    @cached_property
    def phrase(self) -> str:
        """The sentence that travels through every animation (clipped to the context window)."""
        return self.opt.phrase[: self.settings.max_seq_len]

    @cached_property
    def settings(self) -> ui_state.ModelSettings:
        base = dataclasses.replace(ui_state.default_settings(), use_input=False, seed=self.opt.seed)
        return base

    @cached_property
    def char_tok(self) -> CharTokenizer:
        return CharTokenizer.train(self.corpus)

    @cached_property
    def bpe(self) -> BPETokenizer:
        return BPETokenizer.train(self.corpus, BPE_DEMO_MERGES)

    @cached_property
    def phrase_ids(self) -> list[int]:
        return self.char_tok.encode(self.phrase)

    @cached_property
    def attn_text(self) -> str:
        """The first few characters of the phrase, small enough to draw as an attention grid."""
        return self.phrase[: self.opt.attn_tokens]

    def bpe_stage(self, merges: int) -> list[str]:
        """The phrase tokenised with only the first ``merges`` merges (replays the training)."""
        part = BPETokenizer.from_dict({"vocab": self.bpe.itos, "merges": [list(m) for m in self.bpe.merges[:merges]]})
        return part.tokens(self.phrase)

    # ----- the model --------------------------------------------------------- #
    def fresh_model(self) -> GPT:
        """A newly initialised model; the same seed always gives the same weights."""
        torch.manual_seed(self.settings.seed)
        return GPT(self.settings.gpt_config(self.char_tok.vocab_size))

    @cached_property
    def model(self) -> GPT:
        return self.fresh_model()

    @cached_property
    def n_params(self) -> int:
        return self.model.num_parameters()

    @cached_property
    def breakdown(self) -> dict[str, int]:
        return self.model.parameter_breakdown()

    @cached_property
    def pos_table(self) -> torch.Tensor:
        cfg = self.model.cfg
        return sinusoidal_table(cfg.max_seq_len, cfg.d_model)

    @cached_property
    def attn_ids(self) -> torch.Tensor:
        return torch.tensor([self.char_tok.encode(self.attn_text)])

    @cached_property
    def details(self) -> dict:
        """Intermediate tensors of the untrained model on the attention text."""
        with torch.no_grad():
            _, details = self.model(self.attn_ids, return_details=True)
        return details

    @cached_property
    def trained_details(self) -> dict:
        """Intermediate tensors of the trained model on the attention text."""
        model = self.trained  # resolve first: the lazy training run needs gradients
        with torch.no_grad():
            _, details = model(self.attn_ids, return_details=True)
        return details

    @cached_property
    def block0(self) -> dict:
        return self.details["blocks"][0]

    @cached_property
    def mask(self) -> torch.Tensor:
        return causal_mask(len(self.attn_text))

    # ----- training ------------------------------------------------------------- #
    @cached_property
    def train_cfg(self) -> TrainConfig:
        steps = self.opt.train_steps
        return TrainConfig(steps=steps, warmup_steps=max(1, min(20, steps // 5)),
                           eval_interval=max(1, steps // 12), seed=self.opt.seed)

    @cached_property
    def splits(self) -> tuple[torch.Tensor, torch.Tensor]:
        return train_val_split(encode_corpus(self.corpus, self.char_tok))

    @cached_property
    def run(self) -> tuple[GPT, list[dict]]:
        """Train a fresh model with the project's trainer; returns ``(model, history)``."""
        model = self.fresh_model()
        tr, va = self.splits
        history = list(train(model, tr, va, self.train_cfg, self.device))
        return model, history

    @property
    def history(self) -> list[dict]:
        return self.run[1]

    @property
    def trained(self) -> GPT:
        return self.run[0]

    @cached_property
    def batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """One tiny training batch (4 windows of 12 characters) for the data-pipeline step."""
        gen = torch.Generator().manual_seed(self.opt.seed)
        return get_batch(self.splits[0], 4, 12, gen)

    @cached_property
    def unigram_entropy(self) -> float:
        """Entropy (nats) of the character frequencies of the training text: what 'just letter counts' can reach."""
        counts = torch.bincount(self.splits[0], minlength=self.char_tok.vocab_size).double()
        p = counts[counts > 0] / counts.sum()
        return float(-(p * p.log()).sum())

    @cached_property
    def target_id(self) -> int:
        """The character we pretend should follow the phrase (a space)."""
        return self.char_tok.stoi[" "]

    @cached_property
    def adam_demo(self) -> dict:
        """One real Adam step on one real weight, with every intermediate number."""
        model = self.fresh_model()
        gen = torch.Generator().manual_seed(self.opt.seed)
        x, y = get_batch(self.splits[0], self.train_cfg.batch_size, self.train_cfg.block_size, gen)
        _, loss = model(x, y)
        loss.backward()
        param = model.blocks[0].attn.q_proj.weight
        flat = int(param.grad.abs().argmax())
        idx = (flat // param.shape[1], flat % param.shape[1])
        before, grad = float(param[idx].detach()), float(param.grad[idx])
        opt = Adam([param], lr=self.train_cfg.lr)
        opt.step()
        state = opt.state[param]
        beta1, beta2 = opt.defaults["betas"]
        return {
            "loss": loss.item(), "before": before, "grad": grad, "after": float(param[idx].detach()), "lr": self.train_cfg.lr,
            "m": float(state["exp_avg"][idx]), "v": float(state["exp_avg_sq"][idx]),
            "beta1": beta1, "beta2": beta2, "eps": opt.defaults["eps"], "where": f"blocks.0.attn.q_proj.weight{list(idx)}",
        }

    @cached_property
    def untrained_probs(self) -> torch.Tensor:
        """Next-character distribution of the untrained model after the phrase."""
        with torch.no_grad():
            logits = self.model(torch.tensor([self.phrase_ids]))[0, -1]
        return torch.softmax(logits, dim=-1)

    @cached_property
    def trained_logits(self) -> torch.Tensor:
        model = self.trained  # resolve first: the lazy training run needs gradients
        with torch.no_grad():
            return model(torch.tensor([self.phrase_ids]))[0, -1].clone()

    @cached_property
    def generated(self) -> list[int]:
        """Greedy continuation of the phrase by the trained model (ids, phrase excluded)."""
        gen = torch.Generator().manual_seed(self.opt.seed)
        ids = torch.tensor([self.phrase_ids])
        out = self.trained.generate(ids, 24, temperature=0.8, top_k=8, generator=gen)
        return out[0, len(self.phrase_ids):].tolist()


def top_k(probs: torch.Tensor, k: int) -> list[tuple[int, float]]:
    """The ``k`` most likely ``(id, probability)`` pairs."""
    values, index = torch.topk(probs, min(k, probs.numel()))
    return [(int(i), float(v)) for i, v in zip(index, values)]



# --------------------------------------------------------------------------- #
# Facts read from the repository (so the scaffolding / app / compliance GIFs stay true)
# --------------------------------------------------------------------------- #
def _lines(path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


class ProjectFacts:
    """Names and counts read from the files of the repository."""

    def __init__(self, root=None) -> None:
        import config

        self.root = root or config.PROJECT_ROOT

    @cached_property
    def env_keys(self) -> list[tuple[str, str]]:
        """``(KEY, value)`` for every setting of ``.env.example``; long values are shortened."""
        out = []
        for line in _lines(self.root / ".env.example"):
            if line.strip() and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                out.append((key.strip(), value.strip() if len(value) <= 14 else value.strip()[:12] + "..."))
        return out

    @cached_property
    def requirements(self) -> list[str]:
        return [ln.split(">=")[0].strip() for ln in _lines(self.root / "requirements.txt") if ln.strip()]

    @cached_property
    def test_modules(self) -> int:
        return len(list((self.root / "tests").glob("test_*.py")))

    @cached_property
    def gitignore(self) -> list[str]:
        return [ln for ln in _lines(self.root / ".gitignore") if ln.strip() and not ln.startswith("#")]

    @cached_property
    def pages(self) -> list[str]:
        return sorted(p.stem.removeprefix("page_") for p in (self.root / "pages").glob("page_*.py"))

    @cached_property
    def scripts(self) -> list[str]:
        return sorted(p.name for p in (self.root / "scripts").glob("*.sh"))

    @cached_property
    def documents(self) -> list[str]:
        names = ["README.md"] + [f"Docs/{p.name}" for p in sorted((self.root / "Docs").glob("*.md"))]
        return [n for n in names if (self.root / n).is_file()]

    @cached_property
    def state_keys(self) -> list[str]:
        return list(ui_state._STATE_DEFAULTS)
