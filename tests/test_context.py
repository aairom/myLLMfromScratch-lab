"""Unit tests for animgen/context.py."""
import pytest
import torch

from animgen import context
from animgen.context import ProjectFacts, top_k


def test_phrase_is_clipped_to_the_context_window(anim_ctx):
    assert anim_ctx.phrase == anim_ctx.opt.phrase[: anim_ctx.settings.max_seq_len]
    assert anim_ctx.attn_text == anim_ctx.phrase[: anim_ctx.opt.attn_tokens]
    assert len(anim_ctx.phrase_ids) == len(anim_ctx.phrase)


def test_fresh_models_are_identical_for_the_same_seed(anim_ctx):
    a, b = anim_ctx.fresh_model(), anim_ctx.fresh_model()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert anim_ctx.n_params == a.num_parameters() > 0


def test_breakdown_and_positional_table_shapes(anim_ctx):
    cfg = anim_ctx.model.cfg
    parts = {k: v for k, v in anim_ctx.breakdown.items() if k != "total"}
    assert {"token_embedding", "attention", "feed_forward", "layer_norm"} <= set(parts)
    assert sum(parts.values()) == anim_ctx.breakdown["total"] == anim_ctx.n_params
    assert anim_ctx.pos_table.shape == (cfg.max_seq_len, cfg.d_model)


def test_bpe_replay_shrinks_the_token_count(anim_ctx):
    counts = [len(anim_ctx.bpe_stage(k)) for k in (0, 4, context.BPE_DEMO_MERGES)]
    assert counts[0] >= counts[1] >= counts[2] and counts[0] > counts[2]
    assert "".join(anim_ctx.bpe_stage(3)) == anim_ctx.phrase


def test_details_of_untrained_and_trained_model(anim_ctx):
    n = len(anim_ctx.attn_text)
    assert anim_ctx.block0["attention"]["weights"].shape[-2:] == (n, n)
    assert anim_ctx.trained_details["blocks"][0]["attention"]["weights"].shape[-1] == n
    assert anim_ctx.mask.shape == (n, n) and bool(anim_ctx.mask[-1].all()) and not bool(anim_ctx.mask[0, -1])


def test_training_run_is_real_and_loss_falls(anim_ctx):
    hist = anim_ctx.history
    assert len(hist) == anim_ctx.opt.train_steps and hist[-1]["done"]
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert anim_ctx.trained is anim_ctx.run[0]
    assert anim_ctx.train_cfg.steps == anim_ctx.opt.train_steps


def test_batch_targets_are_the_inputs_shifted_by_one(anim_ctx):
    x, y = anim_ctx.batch
    assert x.shape == y.shape == (4, 12) and torch.equal(x[:, 1:], y[:, :-1])


def test_unigram_entropy_lies_between_zero_and_log_vocab(anim_ctx):
    import math

    assert 0 < anim_ctx.unigram_entropy < math.log(anim_ctx.char_tok.vocab_size)
    assert anim_ctx.char_tok.itos[anim_ctx.target_id] == " "


def test_adam_demo_reproduces_the_update_by_hand(anim_ctx):
    d = anim_ctx.adam_demo
    m_hat, v_hat = d["m"] / (1 - d["beta1"]), d["v"] / (1 - d["beta2"])
    by_hand = d["before"] - d["lr"] * m_hat / (v_hat ** 0.5 + d["eps"])
    assert by_hand == pytest.approx(d["after"], abs=1e-6)
    assert d["m"] == pytest.approx((1 - d["beta1"]) * d["grad"]) and d["where"].startswith("blocks.0.attn.q_proj")


def test_probabilities_logits_and_generated_text(anim_ctx):
    assert anim_ctx.untrained_probs.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert anim_ctx.trained_logits.shape == (anim_ctx.char_tok.vocab_size,)
    assert len(anim_ctx.generated) == 24 and all(0 <= i < anim_ctx.char_tok.vocab_size for i in anim_ctx.generated)


def test_top_k_orders_by_probability():
    best = top_k(torch.tensor([0.1, 0.6, 0.3]), 2)
    assert [i for i, _ in best] == [1, 2] and best[0][1] == pytest.approx(0.6)
    assert len(top_k(torch.tensor([0.5, 0.5]), 9)) == 2


def test_project_facts_read_the_repository(anim_ctx):
    f = anim_ctx.project
    assert any(k == "APP_PORT" for k, _ in f.env_keys) and "torch" in f.requirements
    assert f.test_modules > 0 and "home" in f.pages and "start.sh" in f.scripts
    assert "README.md" in f.documents and ".env" in f.gitignore and "model" in f.state_keys


def test_project_facts_of_an_empty_folder_are_empty(tmp_path):
    f = ProjectFacts(tmp_path)
    assert (f.env_keys, f.requirements, f.gitignore, f.documents) == ([], [], [], [])
    assert f.test_modules == 0


def test_long_env_values_are_shortened(tmp_path):
    (tmp_path / ".env.example").write_text("# c\nA=1\nB=" + "x" * 40 + "\n")
    keys = dict(ProjectFacts(tmp_path).env_keys)
    assert keys["A"] == "1" and keys["B"].endswith("...") and len(keys["B"]) < 20
