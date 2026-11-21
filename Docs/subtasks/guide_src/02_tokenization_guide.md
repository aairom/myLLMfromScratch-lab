# Implementation guide: tokenization

## Role in the project

Tokenization is the entry gate of the model. Everything downstream (embeddings, attention, the loss) sees only integer ids. The tokenizer is also the one component that is **saved with the model**: a checkpoint without its tokenizer cannot be used, because the ids mean nothing without the vocabulary.

```flow Position in the pipeline
 raw text --> [ pretokenize ] --> [ Char | BPE tokenizer ] --> list[int] ids
                                         |                          |
                                         |                          v
                                         |               training.data.encode_corpus
                                         |                          |  torch.long tensor (N,)
                                         v                          v
                                  vocab_size ------------> GPTConfig.vocab_size
                                                                (embedding rows, output logits)
```

## Module map

| File | Contents |
|---|---|
| `llm/tokenizer.py` (279 lines) | `pretokenize`, `CharTokenizer`, `MergeStep`, `BPETokenizer`, `tokenizer_from_dict`, `build_tokenizer` |
| `viz/tokens.py` | token chips, merge table, frequency and compression figures |
| `ui/demo.py` | `tokenize_text`, `replay_merges`, `compression_counts` |
| `pages/page_tokenization.py` | the Streamlit page |
| `tests/test_tokenizer.py` | 19 tests |

## Shared interface

Both tokenizers expose the same members, so no other module needs to know which one is active.

| Member | Type | Meaning |
|---|---|---|
| `kind` | `str` | `"char"` or `"bpe"` (class attribute) |
| `itos` | `list[str]` | id to token string; index 0 is always `<unk>` |
| `stoi` | `dict[str, int]` | token string to id |
| `vocab_size` | `int` (property) | `len(itos)` |
| `tokens(text)` | `list[str]` | token strings; unknown pieces appear as `<unk>` |
| `encode(text)` | `list[int]` | ids; unknown pieces map to `UNK_ID = 0` |
| `decode(ids)` | `str` | text; id 0 becomes U+FFFD; an out-of-range id raises `ValueError` |
| `to_dict()` | `dict` | JSON-like state stored in checkpoints |

## Pre-tokenization

{{source:llm.tokenizer.pretokenize}}

The regular expression `\s?\w+|\s?[^\w\s]+|\s+` produces word chunks that keep their leading space. It is lossless: joining the chunks returns the original text.

## The character tokenizer

{{source:llm.tokenizer.CharTokenizer}}

## The BPE tokenizer

### Training loop

```flow BPE training, one iteration
 words : dict[tuple[str,...], int]     chunk (as characters) -> frequency
        |
        v
 count adjacent pairs, weighted by frequency          pair_counts : Counter
        |
        v
 ranked = sorted(pair_counts, key=(-count, pair))     highest count, ties alphabetical
        |
        v
 best = ranked[0]  ->  new_token = best[0] + best[1]
        |
        +--> merges.append(best)        ordered list, index = rank
        +--> _add_token(new_token)      grows itos and stoi
        +--> trace.append(MergeStep)    kept for the UI
        |
        v
 words = _apply(words, best)            rewrite every chunk, merge equal chunks
```

{{source:llm.tokenizer.BPETokenizer.train}}

### Merge record

{{source:llm.tokenizer.MergeStep}}

### Encoding a chunk

The chunk starts as a list of characters. At each round the adjacent pair with the **lowest rank** is merged, until no pair is in the rank table.

{{source:llm.tokenizer.BPETokenizer._bpe_word}}

Worked example with the bundled corpus and 100 merges (the corpus is {{fact:corpus_chars}} characters; the vocabulary is {{fact:bpe100_vocab}} tokens):

```flow Encoding "The little robot" with 100 merges
text           : The little robot
pre-tokenize   : ["The", " little", " robot"]
"The"          : T h e           -> The               (merge 1: h+e, later merge 26: T+he)
" little"      : ' ' l i t t l e -> ' li' t t le      (' li' and 'le' were learned, 't'+'t' was not)
" robot"       : ' ' r o b o t   -> ' robot'          (a whole word became one token)
tokens         : ['The', ' li', 't', 't', 'le', ' robot']      6 tokens for 16 characters
```

## Data structures

```flow In memory
CharTokenizer
  itos : ['<unk>', '\n', ' ', '!', '"', ',', '.', ':', ... sorted characters ...]       length = vocab_size
  stoi : {'<unk>': 0, '\n': 1, ' ': 2, '!': 3, ...}

BPETokenizer
  itos   : ['<unk>', sorted characters..., merged tokens in merge order]
  stoi   : inverse of itos
  merges : [('h','e'), (' ','t'), (' t','he'), ...]     index = rank
  trace  : [MergeStep(step, pair, count, new_token, top_pairs), ...]   only after train()
  _ranks : {pair: rank}        derived, rebuilt by _refresh_ranks()
  _cache : {chunk: tuple[str,...]}   memoised results, cleared on refresh
```

### Persistence format

```flow tokenizer.to_dict() as stored in a checkpoint
char : {"type": "char", "vocab": [<unk>, ...]}
bpe  : {"type": "bpe",  "vocab": [<unk>, ...], "merges": [["h","e"], [" ","t"], ...]}
```

{{source:llm.tokenizer.tokenizer_from_dict}}

{{source:llm.tokenizer.build_tokenizer}}

!!! warning "The merge trace is not persisted"
    `to_dict()` stores the vocabulary and the merges but **not** `trace`. A tokenizer restored from a checkpoint encodes identically, yet has an empty `trace`, so the merge table on the tokenization page is empty until the tokenizer is retrained.

## Integration points

| Where | How the tokenizer is used |
|---|---|
| `ui.state.make_tokenizer` | trains `build_tokenizer(kind, corpus, merges)` behind an `lru_cache` (8 entries) keyed by kind, text and merge count |
| `ui.state.ModelSettings` | `tokenizer` (`char` or `bpe`) and `bpe_merges` (0 to 2000) |
| `training.data.encode_corpus` | `torch.tensor(tokenizer.encode(text), dtype=torch.long)` |
| `llm.transformer.GPTConfig` | `vocab_size = tokenizer.vocab_size` sizes the embedding and output layers |
| `training.checkpoint` | stores `tokenizer.to_dict()` under the key `tokenizer` and restores it with `tokenizer_from_dict` |
| `ui.demo.generate_text` | decodes sampled ids back to text |

## Configuration and runtime behaviour

- There is no environment variable for tokenization. The choice is a sidebar setting (`ModelSettings.tokenizer`, `bpe_merges`), validated by `ModelSettings.problems()`.
- `MAX_BPE_MERGES = 2000` keeps retraining interactive. The cost grows with the number of merges because every iteration recounts all pairs.
- **Changing the tokenizer rebuilds the model.** The vocabulary size changes the embedding table, so `apply_settings` creates fresh weights and clears the training history.
- Training is deterministic: sorted characters and the `(-count, pair)` tie-break give the same tokenizer for the same text.
- The reference corpus gives {{fact:char_vocab}} character tokens, or {{fact:bpe100_vocab}} and {{fact:bpe200_vocab}} tokens with 100 and 200 merges; compression is {{fact:bpe100_ratio}} and {{fact:bpe200_ratio}} characters per token.

## Tests

The suite checks that pre-tokenization is lossless, that `decode(encode(x)) == x` for known text, that unknown characters map to id 0 and decode to U+FFFD, that BPE training is deterministic and respects the merge budget, that encoding applies merges by rank, and that `to_dict` and `tokenizer_from_dict` round-trip.

## Common pitfalls

- **Comparing losses across tokenizers.** A per-token loss of a BPE model is not comparable to a character model's loss; compare within one tokenizer.
- **Too many merges on a tiny corpus.** Beyond a few hundred merges the vocabulary starts to memorise single words.
- **Unknown characters.** Characters absent from the training text become `<unk>`; paste text with unusual symbols and they will decode as the replacement character.
