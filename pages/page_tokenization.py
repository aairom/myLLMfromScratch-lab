"""Tokenization: text -> tokens -> integer ids, with a BPE merge replay."""
import streamlit as st

from llm.tokenizer import BPETokenizer
from ui import demo
from ui.components import init_page, int_slider, lesson_tabs, page_header, plot
from ui.state import load_corpus
from viz.tokens import compression_figure, merge_rows, token_rows, tokens_html, vocab_frequency_figure

init_page()
page_header("tokenization")
lesson_tabs("tokenization", extra_sources=("llm.tokenizer.BPETokenizer.train",))

tokenizer = st.session_state["tokenizer"]
settings = st.session_state["settings"]

st.divider()
st.subheader("Try it: encode your own text")
st.caption(f"Using the **{tokenizer.kind}** tokenizer from the sidebar (vocabulary {tokenizer.vocab_size}).")
text = st.text_area("Text", demo.DEFAULT_TEXT, key="tok_text", height=80)
tokens, ids = demo.tokenize_text(tokenizer, text)
if not tokens:
    st.info("Type some text to see its tokens.")
else:
    st.markdown(tokens_html(tokens, ids), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", len(text))
    c2.metric("Tokens", len(tokens))
    c3.metric("Characters per token", f"{len(text) / len(tokens):.2f}")
    if "<unk>" in tokens:
        st.warning("Some characters are not in the vocabulary and became `<unk>` (id 0).")
    with st.expander("Token table"):
        st.dataframe(token_rows(tokens, ids), width="stretch", hide_index=True)
    plot(vocab_frequency_figure(tokens), "tok_freq")

st.divider()
st.subheader("Byte-pair encoding, step by step")
if not isinstance(tokenizer, BPETokenizer):
    st.info("Choose **bpe** as the tokenizer in the sidebar and apply it to replay the merges.")
else:
    st.caption("Each merge fuses the most frequent adjacent pair into one new token.")
    rows = merge_rows(tokenizer)
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True, height=240)
    word = st.text_input("Word to replay", "robot", key="tok_word")
    n_merges = len(tokenizer.merges)
    if word and n_merges:
        upto = int_slider("Merges applied", 0, n_merges, min(n_merges, 10), key="tok_upto")
        pieces = demo.replay_merges(word, tokenizer.merges, upto)
        st.markdown(" + ".join(f"`{p}`" for p in pieces))
        st.caption(f"{len(word)} characters become {len(pieces)} pieces after {upto} merges.")

    st.markdown("**Sequence length versus merges**")
    try:
        corpus = load_corpus(settings)
    except ValueError as exc:
        st.warning(str(exc))
    else:
        counts = demo.compression_counts(corpus, [0, 25, 50, 100, 200, 300])
        plot(compression_figure(len(corpus), counts), "tok_compression")
