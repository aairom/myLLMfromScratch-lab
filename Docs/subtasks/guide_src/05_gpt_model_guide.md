# Implementation guide: the GPT model and generation

## Role in the project

`llm/transformer.py` is the integration point of the model code. It imports the embedding, positional, attention and layer modules and composes them into the one object the rest of the application passes around: `GPT`. The trainer calls `model(x, y)` for a loss, the pages call `model(ids, return_details=True)` for visualisation, and the inference page calls `model.generate(...)`.

```flow Forward pass of GPT.forward (default: B=16, T=32, D=64, V=51)
 ids (B,T) ----> tok_emb ----> pos (+) ----> drop ----> blocks[0] ----> blocks[1] ----> ln_f ----> lm_head ----> logits (B,T,V)
                (B,T,D)       (B,T,D)       (B,T,D)     (B,T,D)        (B,T,D)        (B,T,D)    (B,T,V)            |
                                                                                                                    v
 targets (B,T) --------------------------------------------------------------------------------> cross_entropy -> loss
```

## Module map

| File | Contents |
|---|---|
| `llm/transformer.py` (323 lines) | `GPTConfig`, `cross_entropy`, `TransformerBlock`, `next_token_distribution`, `GPT` |
| `viz/training.py` | `model_stages`, `block_stages`, `dataflow_figure`, `parameter_breakdown_figure` |
| `ui/demo.py` | `model_forward`, `top_next_tokens`, `generate_text`, `step_probabilities`, `parameter_summary` |
| `pages/` | `page_transformer.py`, `page_inference.py` |
| `tests/test_transformer.py` | 28 tests |

## Configuration object

{{source:llm.transformer.GPTConfig}}

| Field | Default | Allowed | Effect |
|---|---|---|---|
| `vocab_size` | (required) | > 0 | rows of the embedding, columns of the logits |
| `d_model` | {{fact:d_model}} | > 0 and divisible by `n_heads` | width of the residual stream |
| `n_heads` | {{fact:n_heads}} | > 0 | heads per attention layer, `d_k = d_model / n_heads` |
| `n_layers` | {{fact:n_layers}} | > 0 | number of blocks |
| `max_seq_len` | {{fact:context}} | > 0 | context window and size of the positional table |
| `dropout` | 0.0 | [0, 1) | attention, output projections, FFN output and the embedding sum |
| `ffn_mult` | 4 | > 0 | hidden width = `ffn_mult * d_model` |
| `activation` | `gelu` | `gelu`, `relu` | FFN non-linearity |
| `positional` | `sinusoidal` | `sinusoidal`, `learned` | position encoding |
| `embedding_init` | `normal` | `normal`, `uniform`, `xavier` | token table initialisation |
| `tie_weights` | `True` | bool | share the embedding with the output head |

`to_dict()` and `from_dict()` make the config JSON-friendly; it is stored in every checkpoint as `model_config`.

## Loss

{{source:llm.transformer.cross_entropy}}

## The block

{{source:llm.transformer.TransformerBlock}}

The `return_details=False` path is the one training uses: two lines, `x = x + attn(ln1(x))` and `x + ffn(ln2(x))`. The `return_details=True` path computes the same values step by step so each intermediate can be returned.

```flow Keys of the block details dictionary
input        x entering the block
ln1          first LayerNorm output
attention    {q, k, v, scores, weights, head_output}      see sub-task 4
attn_out     attention sub-layer output (after out_proj)
after_attn   input + attn_out                              residual stream mid-block
ln2          second LayerNorm output
ffn_hidden   hidden activations (B, T, d_hidden)
ffn_out      FFN output
output       after_attn + ffn_out                          residual stream leaving the block
```

## The model

{{source:llm.transformer.GPT}}

### Return values of `forward`

| Call | Returns |
|---|---|
| `model(ids)` | `logits` `(B, T, V)` |
| `model(ids, targets)` | `(logits, loss)` |
| `model(ids, return_details=True)` | `(logits, details)` |
| `model(ids, targets, return_details=True)` | `(logits, loss, details)` |

`details` has the keys `token_embeddings`, `with_position`, `blocks` (a list of the block dictionaries above), `final_norm` and `logits`. Every tensor in it is detached, so collecting details cannot change training.

Validation raises `ValueError` for a tensor that is not 2-D, for an empty sequence, for a sequence longer than `max_seq_len`, and for `targets` whose shape differs from `ids`.

## Initialisation

{{source:llm.transformer.GPT._init_weights}}

```flow Initial standard deviations
all nn.Linear (except lm_head)            N(0, 0.02^2), biases 0
attn.out_proj.weight, ffn.fc_out.weight   N(0, (0.02 / sqrt(2 * n_layers))^2) = N(0, 0.01^2) for 2 layers
token embedding                           per GPTConfig.embedding_init (normal, std 0.02 by default)
lm_head                                   shares the embedding when tied; N(0, 0.02^2) when not tied
LayerNorm                                 weight 1, bias 0
```

## Introspection

{{source:llm.transformer.GPT.parameter_breakdown}}

```flow Default breakdown (sums to {{fact:params}})
token_embedding   {{fact:params_embedding}}
positional        0
attention         {{fact:params_attention}}
feed_forward      {{fact:params_ffn}}
layer_norm        {{fact:params_layernorm}}
output_head       0     (tied: already counted in token_embedding)
```

A set of parameter ids is used so that a tied matrix is counted exactly once, which is why the components add up to `num_parameters()`.

## Sampling

{{source:llm.transformer.next_token_distribution}}

{{source:llm.transformer.GPT.generate}}

```flow One generation step
ids (B, T_so_far)
   |  crop to last max_seq_len tokens
   v
 forward -> logits[:, -1, :]                 only the last position matters
   |
   v
 next_token_distribution(temperature, top_k, top_p)  -> probs (B, V)
   |
   v
 torch.multinomial(probs.float().cpu(), 1, generator)   CPU draw: a seeded CPU generator works on every device
   |
   v
 ids = cat([ids, next], dim=1)
```

Behaviour worth knowing:

- `generate` switches the model to `eval()` (dropout off) and **restores** the previous mode in a `finally` block, even if sampling fails.
- Passing a `torch.Generator` seeded on the CPU makes generation reproducible; `ui.demo.generate_text` creates one when a seed is given.
- With `temperature = 0` no random draw influences the result: the arg-max is chosen.
- An empty prompt would give a window of length 0, so `ui.demo.generate_text` starts from id 1 when the prompt encodes to nothing.

### Worked sampling example

For logits `[2.0, 1.0, 0.1]`:

| Call | Probabilities |
|---|---|
| default | 0.659, 0.242, 0.099 |
| `temperature=0.5` | 0.864, 0.117, 0.019 |
| `temperature=2.0` | 0.502, 0.304, 0.194 |
| `top_k=2` | 0.731, 0.269, 0.000 |
| `top_p=0.8` | 0.731, 0.269, 0.000 |
| `temperature=0` | 1.000, 0.000, 0.000 |

## Integration points

| Where | Interaction |
|---|---|
| `ui.state.build_model` | `torch.manual_seed(settings.seed)` then `GPT(settings.gpt_config(tokenizer.vocab_size))` |
| `training.trainer.train` | `_, loss = model(x, y)`; `model.cfg.max_seq_len` bounds the training window |
| `training.checkpoint` | stores `model.cfg.to_dict()` and `model.state_dict()`; rebuilds with `GPTConfig.from_dict` |
| `ui.demo.model_forward` | `eval()` + `no_grad`, restores the mode afterwards |
| `viz.training.dataflow_figure` | draws the stages from `GPTConfig` and the shapes only, without running the model |

## Runtime behaviour

- The model runs on the device chosen by `config.resolve_device`: `cuda`, then `mps`, then `cpu`.
- Training mode is the default after construction; the inference pages call `eval()` and restore the previous mode.
- Initial loss on random ids is about 3.95, close to `ln(51)` = {{fact:ln_vocab}}, as expected for a model that has not learned anything.
- The reference run takes {{fact:ref_seconds}} s for {{fact:ref_steps}} steps on `{{fact:ref_device}}`.

## Tests

The suite covers: configuration validation and round-trip; the hand-written cross-entropy against PyTorch, including very large logits and the uniform case `ln(V)`; forward shapes and the near-uniform initial loss; the `return_details` structure; **causality of the whole model**; input validation; weight tying (same tensor object); the parameter breakdown summing to the total; finite gradients for every parameter; a block that behaves as the identity when its sub-layers are zeroed; the model can **overfit a tiny batch** (a strong end-to-end check that gradients and optimisation are wired correctly); every sampling option; reproducible seeded generation; and mode restoration.

## Common pitfalls

- **Wrong vocabulary.** A checkpoint's tokenizer and model belong together; `vocab_size` mismatches surface as an `IndexError` in the embedding.
- **Expecting embedding scaling.** `GPT` does not multiply embeddings by `sqrt(d_model)`.
- **Long prompts.** Anything beyond `max_seq_len` is cropped from the left during generation.
- **Reading text quality from a short run.** After {{fact:ref_steps}} steps the model reproduces letter frequencies, not words.
