# Implementation guide: attention, LayerNorm and feed-forward

## Role in the project

This sub-task provides the three building blocks of a transformer block: `MultiHeadAttention` (mixes information between positions), `LayerNorm` (stabilises activations) and `FeedForward` (transforms each position). They are plain `nn.Module` classes with no knowledge of the model that will contain them, which is why each can be demonstrated on its own on the attention, multi-head and norm/FFN pages.

```flow How the pieces are used (one pre-norm block, see sub-task 5)
 x --+--> LayerNorm (ln1) --> MultiHeadAttention --+--> (+) --+--> LayerNorm (ln2) --> FeedForward --+--> (+) --> out
     |                                              |          |                                       |
     +----------------- residual ------------------+          +------------- residual --------------+
```

## Module map

| File | Contents |
|---|---|
| `llm/attention.py` (152 lines) | `causal_mask`, `attention_scores`, `scaled_dot_product_attention`, `MultiHeadAttention` |
| `llm/layers.py` (105 lines) | `LayerNorm`, `gelu`, `relu`, `ACTIVATIONS`, `FeedForward` |
| `viz/attention.py` | heat maps, head grid, Q/K/V maps, attention-received bars, LayerNorm distributions |
| `ui/demo.py` | `attention_demo`, `layernorm_demo`, `ffn_demo` run a stand-alone, untrained layer on user text |
| `pages/` | `page_attention.py`, `page_multihead.py`, `page_norm_ffn.py` |
| `tests/` | `test_attention.py` (16), `test_layers.py` (10), `test_viz_attention.py` (4) |

## Attention functions

The three small functions mirror the three steps of the formula, so each can be tested and displayed separately.

{{source:llm.attention.causal_mask}}

{{source:llm.attention.attention_scores}}

{{source:llm.attention.scaled_dot_product_attention}}

```flow Tensor shapes inside attention (default: B=16, T=32, H=4, d_k=16)
x                      (B, T, d_model)         (16, 32, 64)
q_proj / k_proj / v_proj                       each Linear(64 -> 64)
q, k, v  after _split_heads     (B, H, T, d_k) (16, 4, 32, 16)
scores = q @ k^T / sqrt(d_k)    (B, H, T, T)   (16, 4, 32, 32)      masked with -inf above the diagonal
weights = softmax(scores)       (B, H, T, T)   rows sum to 1, upper triangle exactly 0
head_out = weights @ v          (B, H, T, d_k) (16, 4, 32, 16)
_merge_heads                    (B, T, d_model)(16, 32, 64)
out_proj                        (B, T, d_model)(16, 32, 64)
```

## Multi-head attention module

{{source:llm.attention.MultiHeadAttention}}

### The details dictionary

`forward(x, return_details=True)` returns `(output, details)`. The tensors are detached copies, meant for plotting; they are never used by the loss.

| Key | Shape | Meaning |
|---|---|---|
| `q`, `k`, `v` | `(B, H, T, d_k)` | projected and split activations |
| `scores` | `(B, H, T, T)` | scaled scores **before** softmax, `-inf` where masked |
| `weights` | `(B, H, T, T)` | attention weights after softmax (after `nan_to_num`) |
| `head_output` | `(B, H, T, d_k)` | `weights @ v`, before the heads are merged |

!!! note "Details do not use dropout weights"
    `weights` is the softmax output *before* attention dropout, so the diagnostic view is not distorted by random masking in training mode.

### Parameter count

| Part | Parameters (d = {{fact:d_model}}) |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `out_proj` | 4 x (64 x 64 + 64) = {{fact:params_attn_layer}} |
| All {{fact:n_layers}} layers | {{fact:params_attention}} |

## Layer normalisation

{{source:llm.layers.LayerNorm}}

`stages(x)` returns `mean`, `var`, `normalized` and `output`, so the educational page can show every step of the calculation instead of only the result. `forward` is defined through `stages`, which guarantees that what is shown is what runs.

```flow LayerNorm on one token vector x = [1, 2, 3, 4]
mean       = 2.5
var        = 1.25      (divide by d, not d-1)
normalized = [-1.342, -0.447, 0.447, 1.342]
output     = normalized * weight + bias        weight = 1, bias = 0 at initialisation
```

## Feed-forward network

{{source:llm.layers.FeedForward}}

{{source:llm.layers.gelu}}

| Setting | Value |
|---|---|
| Hidden width | `ffn_mult * d_model` = {{fact:ffn_hidden}} (`ffn_mult` = 4) |
| Parameters per layer | {{fact:params_ffn_layer}} |
| All layers | {{fact:params_ffn}} |
| Activations | `gelu` (default), `relu`; the registry is `ACTIVATIONS` |
| Dropout | on the output only |

`forward(x, return_hidden=True)` also returns the detached hidden activations `(B, T, d_hidden)` so the page can show how many hidden units are active.

## Data structures: what is stored where

```flow Parameters of one block (d = 64)
ln1.weight, ln1.bias                   (64), (64)
attn.q_proj.weight / .bias             (64, 64) / (64)       same for k_proj, v_proj, out_proj
ln2.weight, ln2.bias                   (64), (64)
ffn.fc_in.weight / .bias               (256, 64) / (256)
ffn.fc_out.weight / .bias              (64, 256) / (64)
```

## Integration points

| Where | Interaction |
|---|---|
| `llm.transformer.TransformerBlock` | creates `ln1`, `attn` (`causal=True`), `ln2`, `ffn`; passes `dropout` and `activation` from `GPTConfig` |
| `llm.transformer.GPT._init_weights` | sets `attn.out_proj.weight` and `ffn.fc_out.weight` to std 0.02 / sqrt(2N) = {{fact:res_std}}; all other linears use std 0.02 |
| `ui.state.ModelSettings.problems()` | rejects a `d_model` that is not divisible by `n_heads` with a readable message |
| `viz.attention` | consumes the `details` dictionary; plots use display-safe token labels |
| `edu.lessons` | the `attention`, `multihead`, `layernorm` and `ffn` lessons cite these functions as live snippets |

## Configuration and runtime behaviour

- `d_model % n_heads == 0` is enforced twice: in the sidebar validation and in `MultiHeadAttention.__init__`.
- Input shape is checked: `(B, T, d_model)` with `T > 0`, otherwise `ValueError`.
- The causal mask is created on the input's device for every call; it is a `(T, T)` boolean tensor, cheap relative to the scores.
- Dropout is active only in training mode (`model.train()`); the default is 0.
- The hand-written `LayerNorm` is cross-checked against `torch.nn.LayerNorm` in the tests, within a numeric tolerance.
- Measured on the first 48 characters of the corpus after the {{fact:ref_steps}}-step reference configuration: layer-0 attention entropy fell from 3.25 to about 2.76 nats while layer 1 stayed at about 3.25 (uniform). The pages show these patterns live.

## Tests

The suite verifies: the mask is lower-triangular; rows of the weights sum to 1; future weights are exactly 0; a fully masked row gives zeros, not NaN; scores are scaled by the square root of `d_k`; the output matches `torch.nn.functional.scaled_dot_product_attention`; the multi-head module matches a reference computed from its own details; **changing a later token never changes earlier outputs** (causality) and that non-causal mode does see the future; one head equals full width; the parameter count; every projection receives gradient; invalid configurations raise; dropout acts only in training. For `LayerNorm`: agreement with PyTorch, zero mean and unit variance, finite output for a constant input, gradients. For `FeedForward`: shapes, position-wise independence, parameter counts and argument checks.

## Common pitfalls

- **`d_model` not divisible by heads.** The sidebar shows the message; the model is not rebuilt.
- **Reading weights as explanation.** Attention weights are a diagnostic, not proof of reasoning.
- **Pre-softmax scores contain `-inf`.** Plot code must handle it; `viz.common.to_numpy` converts every non-finite value to NaN, which Plotly leaves blank.
- **Large `T`.** Memory is `B * H * T * T` floats; the interface limits the demo text to the model context.
