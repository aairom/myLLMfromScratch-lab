"""Tests for pages/page_inference.py."""
import pytest

from tests.conftest import assert_clean
from ui.state import ModelSettings

SMALL = ModelSettings(d_model=16, n_heads=2, n_layers=1, max_seq_len=16, use_input=False)


def start_training(at, steps=12):
    at.number_input(key="tr_steps").set_value(steps)
    at.number_input(key="tr_block").set_value(8)
    at.number_input(key="tr_redraw").set_value(4)
    at.number_input(key="tr_warm").set_value(2)
    at.run()
    at.button(key="tr_start").click().run()
    return at



def test_inference_warns_when_untrained(run_page):
    at = run_page("inference", SMALL)
    assert not at.exception
    assert any("untrained" in w.value for w in at.warning)
    assert any("Press **Generate**" in i.value for i in at.info)


def test_inference_generates_requested_length(run_page):
    at = run_page("inference", SMALL)
    at.slider(key="inf_max").set_value(20)
    at.run()
    at.button(key="inf_go").click().run()
    assert_clean(at)
    result = at.session_state["inference_result"]["out"]
    assert len(result["new_ids"]) == 20
    assert result["text"].startswith("The little robot")


def test_inference_is_reproducible_with_fixed_seed(run_page):
    at = run_page("inference", SMALL)
    at.button(key="inf_go").click().run()
    first = at.session_state["inference_result"]["out"]["text"]
    at.button(key="inf_go").click().run()
    assert at.session_state["inference_result"]["out"]["text"] == first


def test_inference_greedy_decoding(run_page):
    at = run_page("inference", SMALL)
    at.slider(key="inf_temp").set_value(0.0).run()
    at.button(key="inf_go").click().run()
    assert_clean(at)


def test_inference_empty_prompt(run_page):
    at = run_page("inference", SMALL)
    at.text_area(key="inf_prompt").set_value("").run()
    at.button(key="inf_go").click().run()
    assert_clean(at)


def test_inference_unknown_prompt_characters_warn(run_page):
    at = run_page("inference", SMALL)
    at.text_area(key="inf_prompt").set_value("robot \u2603").run()
    assert any("not in the vocabulary" in w.value for w in at.warning)


def test_inference_after_training_says_trained(run_page):
    at = start_training(run_page("training", SMALL), steps=3)
    inf = run_page("inference", SMALL)
    inf.session_state["model"] = at.session_state["model"]
    inf.session_state["tokenizer"] = at.session_state["tokenizer"]
    inf.session_state["model_source"] = "trained"
    inf.run()
    assert not inf.exception
    assert any("trained in this session" in s.value for s in inf.success)
