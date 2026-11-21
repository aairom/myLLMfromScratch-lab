"""Unit tests for the live-source snippet helper."""
import pytest

from edu import snippets


def test_get_source_of_class_function_and_method():
    cls = snippets.get_source("llm.layers.LayerNorm")
    assert cls.startswith("class LayerNorm") and "def forward" in cls
    fn = snippets.get_source("llm.attention.causal_mask")
    assert fn.startswith("def causal_mask") and "torch.tril" in fn
    method = snippets.get_source("llm.attention.MultiHeadAttention.forward")
    assert method.startswith("def forward")  # dedented


def test_source_keeps_comments_and_docstrings():
    src = snippets.get_source("llm.transformer.GPT.generate")
    assert "# Sample on CPU" in src and '"""' in src


def test_resolve_returns_live_object():
    from llm.layers import LayerNorm

    assert snippets.resolve("llm.layers.LayerNorm") is LayerNorm


def test_docstring_and_line_count():
    assert "normalisation" in snippets.get_docstring("llm.layers.LayerNorm").lower()
    assert snippets.line_count("llm.layers.gelu") >= 3
    assert snippets.get_docstring("llm.layers.relu").startswith("Rectified linear unit")


@pytest.mark.parametrize(
    "bad",
    ["nomodule", "llm.layers.DoesNotExist", "does_not_exist.Thing", "llm.layers.LayerNorm.nope", "llm.layers"],
)
def test_bad_paths_raise_snippet_error(bad):
    with pytest.raises(snippets.SnippetError):
        snippets.get_source(bad)


def test_builtin_without_source_raises():
    with pytest.raises(snippets.SnippetError):
        snippets.get_source("math.sqrt")  # C function: no Python source
