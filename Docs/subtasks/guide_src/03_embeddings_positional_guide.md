# Implementation guide: embeddings and positional encoding

## Role in the project

Embeddings and positions form the **input layer** of the GPT. After this stage the model no longer sees integers but a tensor of shape `(batch, time, d_model)` that carries both *what* each token is and *where* it sits.

```flow The input stage inside GPT.forward
 ids  (B, T)  int64
   |
   v   TokenEmbedding        self.weight[ids]                       (B, T, d_model)
 tok
   |
   v   positional module     tok + table[:T]   (sinusoidal buffer or learned parameter)
 with_pos                                                            (B, T, d_model)
   |
   v   Dropout(p)            identity when dropout = 0 (the default)
   x   ---------------------> first TransformerBlock
```

## Module map

| File | Contents |
|---|---|
| `llm/embeddings.py` (78 lines) | `INIT_STRATEGIES`, `init_embedding_`, `TokenEmbedding` |
| `llm/positional.py` (99 lines) | `POSITIONAL_KINDS`, `sinusoidal_table`, `SinusoidalPositionalEncoding`, `LearnedPositionalEmbedding`, `build_positional` |
| `viz/embeddings.py` | `pca_2d`, `cosine_similarity_matrix`, heat maps, scatter, positional curves |
| `ui/demo.py` | `embedding_demo`, `positional_demo`, `embed_with_position` |
| `pages/page_embeddings.py`, `page_positional.py` | the two Streamlit pages |
| `tests/` | `test_embeddings.py` (9), `test_positional.py` (9), `test_viz_embeddings.py` (6) |

## Token embedding

{{source:llm.embeddings.TokenEmbedding}}

Initialisation is a separate function so the position table can reuse it:

{{source:llm.embeddings.init_embedding_}}

### Data structure

```flow TokenEmbedding
weight : nn.Parameter  shape (vocab_size, d_model)   float32
         row i = vector of token i

forward(ids)
  ids  : LongTensor any shape (...)         checked: 0 <= id < vocab_size, else IndexError
  out  : (..., d_model)                     = weight[ids]  (advanced indexing)
  scale: False by default; True multiplies by sqrt(d_model)
```

!!! warning "`scale` is off in the GPT"
    `GPT.__init__` builds `TokenEmbedding(cfg.vocab_size, cfg.d_model, init=cfg.embedding_init)` and never passes `scale=True`. Only the embedding demo page exposes the toggle. The documentation therefore never describes the GPT as scaling embeddings by the square root of the model width.

## Positional encodings

### Sinusoidal table

{{source:llm.positional.sinusoidal_table}}

```flow How the table is built (d_model = 64, max_len = 64)
position   : (L, 1)            0, 1, ..., L-1
even       : (d/2,)            0, 2, 4, ..., d-2
div_term   : (d/2,)            exp(-ln(10000) * even / d)   = 10000^(-2i/d)
angles     : (L, d/2)          position * div_term
table[:, 0::2] = sin(angles)   even columns
table[:, 1::2] = cos(angles)   odd columns        every row has norm sqrt(d/2) = 5.657
```

{{source:llm.positional.SinusoidalPositionalEncoding}}

The table is registered with `register_buffer(..., persistent=False)`. It moves with `.to(device)`, is **not** a parameter (no gradient, no optimizer state) and is **not stored in checkpoints**; it is recomputed from `d_model` and `max_len` when the model is rebuilt.

### Learned table

{{source:llm.positional.LearnedPositionalEmbedding}}

### Factory

{{source:llm.positional.build_positional}}

| Kind | Parameters | In `state_dict` | Notes |
|---|---|---|---|
| `sinusoidal` | 0 | no (non-persistent buffer) | default |
| `learned` | `max_len * d_model` = {{fact:context}} x {{fact:d_model}} for the defaults | yes | counted under the `positional` entry of `parameter_breakdown()` |

## Weight tying

The output head shares the embedding matrix when `tie_weights` is true (the default):

{{file:llm/transformer.py|python|190-198}}

Consequences for the implementation:

- `GPT.parameters()` yields the shared tensor once, so `num_parameters()` counts it once. The default model has **{{fact:params}}** parameters, of which **{{fact:params_embedding}}** are the embedding matrix (the output head contributes 0 in `parameter_breakdown()`).
- `_init_weights` skips `lm_head` when tied, so the user-chosen embedding initialisation is preserved.
- Because the logits are `hidden @ E.T`, every embedding row receives gradient on every step, not only the rows of tokens present in the batch.

## Data flow with shapes

```flow Default model: B = 16, T = 32
ids            (16, 32)
tok_emb        (16, 32, 64)         rows of a (51, 64) table
pos.encoding   (32, 64)             table[:32], broadcast over the batch
with_pos       (16, 32, 64)         sum
drop           (16, 32, 64)         identity at dropout 0
```

## Integration points

| Where | Interaction |
|---|---|
| `llm.transformer.GPT` | owns `tok_emb`, `pos`, `drop`, `lm_head`; sizes come from `GPTConfig` |
| `GPTConfig` | `embedding_init` in `normal`, `uniform`, `xavier`; `positional` in `sinusoidal`, `learned`; `tie_weights` |
| `ui.state.ModelSettings` | the same three fields, validated against `INIT_STRATEGIES` and `POSITIONAL_KINDS` |
| `ui.demo.embedding_demo` | builds a *stand-alone* `TokenEmbedding` with `torch.manual_seed(seed)`, so the page works before any model exists |
| `viz.embeddings` | `pca_2d` uses an SVD of the centred matrix; `embedding_scatter` shows the table, `positional_curves` shows individual dimensions as waves |
| `training.checkpoint` | stores the `model_state`; the sinusoidal table is rebuilt, the learned table is restored |

## Configuration and runtime behaviour

- No environment variables are involved; the choices are sidebar settings.
- A sequence longer than `max_len` raises `ValueError` from `encoding`; `GPT.forward` raises its own error earlier, and `generate` crops the context to `max_seq_len`.
- An id outside `[0, vocab_size)` raises `IndexError` with the valid range; this catches a tokenizer and model that do not belong together.
- Magnitudes at initialisation for the defaults: token rows have mean norm about 0.16, sinusoidal rows exactly 5.657. After the {{fact:ref_steps}}-step reference run the mean token-row norm is about 0.31.
- Changing `d_model`, `max_seq_len`, the tokenizer or the initialisation rebuilds the model (new random weights, empty history).

## Tests

The tests check that the embedding output equals row indexing, that out-of-range ids raise `IndexError`, that every initialisation strategy has the intended standard deviation or bound, that the sinusoidal table matches the closed-form formula and has constant row norm, that odd `d_model` is handled, that the buffer is not persistent, and that the learned table is trainable and limited to `max_len`.

## Common pitfalls

- **Forgetting that the table is not saved.** A sinusoidal model restores correctly only because the table is a pure function of its size.
- **Context mismatch.** A checkpoint trained with `max_seq_len = 64` cannot process longer inputs; `generate` crops to the window.
- **Reading structure into an untrained table.** The PCA plot of random vectors shows noise; structure appears only after training, and after {{fact:ref_steps}} steps it is mostly frequency.
