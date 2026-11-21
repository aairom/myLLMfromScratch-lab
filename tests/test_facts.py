"""Unit tests for docsgen/facts.py."""
import pytest

from docsgen import facts


def test_known_facts_are_present():
    names = facts.all_facts()
    for key in ("params", "d_model", "n_heads", "n_layers", "context", "ref_steps", "unigram_val"):
        assert key in names, key


def test_get_returns_display_text():
    assert facts.get("params").replace(",", "").isdigit()
    assert facts.get("d_model").isdigit()


def test_unknown_fact_lists_the_known_ones():
    with pytest.raises(KeyError) as info:
        facts.get("no_such_fact")
    message = str(info.value)
    assert "no_such_fact" in message and "params" in message


def test_all_facts_is_cached():
    assert facts.all_facts() is facts.all_facts()


def test_helpers_format_numbers():
    assert facts._n(103360) == "103,360"
    assert facts._f(3.0369, 3) == "3.037"
