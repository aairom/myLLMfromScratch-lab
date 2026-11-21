# Implementation guide: the training pipeline

## Role in the project

The `training/` package turns an untrained `GPT` into a trained one. It owns everything between "a string of text" and "weights that predict the next character": building the corpus, cutting it into windows, running the loop, updating parameters, scheduling the learning rate and persisting the result. The Streamlit pages never implement any of this themselves; they call `ui.lab` helpers that wrap the generator `training.trainer.train`.

```flow One optimisation step (training.trainer.train)
 train_data (N,) ---> get_batch ---> x (B,T), y (B,T)
                                        |
 schedule(i) --> group["lr"]            v
                                  model(x, y)  ----> loss
                                        |
                       optimizer.zero_grad(set_to_none=True)
                                        |
                                  loss.backward()        autograd fills p.grad
                                        |
                       clip_grad_norm(params, grad_clip) -> grad_norm (before clipping)
                                        |
                                  optimizer.step()       hand-written Adam / SGD
                                        |
              every eval_interval steps: estimate_loss(val_data)
                                        |
                                        v
                   yield {step, train_loss, val_loss, lr, grad_norm, tokens_per_sec, elapsed, done}
```

## Module map

| File | Lines | Contents |
|---|---|---|
| `training/data.py` | 113 | `load_bundled_corpus`, `find_input_files`, `load_input_texts`, `build_corpus`, `encode_corpus`, `train_val_split`, `get_batch` |
| `training/optimizers.py` | 172 | `SGD`, `Adam`, `build_optimizer`, `OPTIMIZER_NAMES` |
| `training/schedulers.py` | 68 | `warmup_cosine_lr`, `constant_lr`, `build_schedule`, `SCHEDULE_NAMES` |
| `training/trainer.py` | 183 | `TrainConfig`, `clip_grad_norm`, `estimate_loss`, `train` |
| `training/checkpoint.py` | 125 | `CheckpointError`, `new_checkpoint_path`, `list_checkpoints`, `save_checkpoint`, `load_checkpoint` |
| `ui/lab.py` | | `prepare_data`, `run_training` and the stop and resume plumbing used by the training page |
| `pages/page_training.py` | | the training lab page |
| `tests/` | | `test_trainer.py` 18, `test_optimizers.py` 15, `test_data.py` 11, `test_schedulers.py` 9, `test_checkpoint.py` 9, `test_training.py` 8 |

## Data

{{source:training.data.build_corpus}}

The corpus is the concatenation of the bundled file `data/sample_corpus.txt` and every `.txt` or `.md` file found recursively under `INPUT_DIR` (default `input/`). Files that are not valid UTF-8 are skipped, and `.gitkeep` and `.DS_Store` are ignored. The corpus is then encoded by the tokenizer chosen in sub-task 2, giving a one-dimensional `LongTensor`.

{{source:training.data.train_val_split}}

{{source:training.data.get_batch}}

```flow Shifted windows: one window gives T training examples
 data:   t0  t1  t2  t3  t4  t5  t6 ...
 window starting at s (length T+1):  [ts, ts+1, ..., ts+T]
 x = [ts   ... ts+T-1]     what the model sees
 y = [ts+1 ... ts+T  ]     the next token at every position
```

The split is chronological, not random: the last part of the text is held out. With the bundled corpus there are {{fact:train_tokens}} training and {{fact:val_tokens}} validation tokens. Windows are drawn at random start positions from a seeded `torch.Generator`, so a run is reproducible.

## Optimisers

{{source:training.optimizers.Adam}}

{{source:training.optimizers.SGD}}

{{source:training.optimizers.build_optimizer}}

| Name | Implementation | Notes |
|---|---|---|
| `adam` | `training.optimizers.Adam` | hand-written, L2-style weight decay added to the gradient |
| `adamw` | `training.optimizers.Adam` with decoupled decay | default in `TrainConfig` |
| `sgd` | `training.optimizers.SGD` | optional momentum |
| `torch_adamw`, `torch_sgd` | `torch.optim` | reference implementations used to cross-check the hand-written ones in the tests |

Weight decay is applied to every parameter, including biases, LayerNorm gains and embeddings. This is a deliberate simplification; production code usually excludes them.

## Learning-rate schedules

{{source:training.schedulers.warmup_cosine_lr}}

Only two schedules exist: `warmup_cosine` and `constant`. With the defaults (`lr` 3e-3, 20 warm-up steps, 300 steps, `min_lr_ratio` 0.1) the rate is 0.00015 at step 0, 0.003 at step 19, 0.00165 at step 160 and 0.0003 at step 299.

{{chart:lr}}

## Configuration object

{{source:training.trainer.TrainConfig}}

| Field | Default | Constraint | Effect |
|---|---|---|---|
| `steps` | 300 | > 0 | number of optimisation steps |
| `batch_size` | 16 | > 0 | windows per step |
| `block_size` | 64 | > 0 and at most the model context | tokens per window |
| `optimizer` | `adamw` | one of `OPTIMIZER_NAMES` | update rule |
| `lr` | 3e-3 | >= 0 | peak learning rate |
| `weight_decay` | 0.01 | >= 0 | decay strength |
| `momentum` | 0.9 | >= 0 | SGD only |
| `schedule` | `warmup_cosine` | one of `SCHEDULE_NAMES` | learning-rate curve |
| `warmup_steps` | 20 | >= 0 | linear ramp length |
| `min_lr_ratio` | 0.1 | [0, 1] | final rate as a fraction of `lr` |
| `grad_clip` | 1.0 | >= 0 | maximum global gradient norm; 0 disables clipping |
| `eval_interval` | 25 | > 0 | steps between validation passes |
| `eval_batches` | 10 | > 0 | batches averaged per validation pass |
| `seed` | 42 | any int | seed of the batch generator |

`TrainConfig.__post_init__` raises `ValueError` for any violation. `to_dict()` and `from_dict()` store it in checkpoints.

## Gradient clipping and evaluation

{{source:training.trainer.clip_grad_norm}}

{{source:training.trainer.estimate_loss}}

Evaluation uses a fixed generator, so every call sees the same validation windows and the numbers are comparable across the run. It switches the model to `eval()` and restores the previous mode afterwards.

## The training loop

{{source:training.trainer.train}}

### Metrics dictionary

```flow Schema of each yielded dictionary
step            int     1-based, counted from start_step
train_loss      float   loss of this step's batch
val_loss        float | None   only at step 1, every eval_interval steps and the last step
lr              float   rate applied in this step
grad_norm       float   global norm BEFORE clipping
tokens_per_sec  float   batch_size * block_size / step time
elapsed         float   seconds since the run started
done            bool    True on the last step of the run
```

### Reference run

{{chart:loss}}

{{chart:gradnorm}}

The reference configuration runs {{fact:ref_steps}} steps in {{fact:ref_seconds}} s on `{{fact:ref_device}}`. Training loss falls from about 3.97 to 3.04 and validation loss from 3.94 to 3.09. In this short run the model only learns the character frequencies of the corpus; see the explanation for the plateau and the later drop in longer runs.

## Checkpoints

{{source:training.checkpoint.save_checkpoint}}

{{source:training.checkpoint.load_checkpoint}}

```flow Checkpoint dictionary (torch.save, loaded with weights_only=True)
version          1
model_config     GPTConfig.to_dict()
model_state      model.state_dict()      (the sinusoidal table is not stored)
tokenizer        tokenizer.to_dict()     (a BPE merge trace is not stored)
train_config     TrainConfig.to_dict()
history          list of the metrics dictionaries
optimizer_state  optimizer.state_dict()
```

Files are written to `OUTPUT_DIR` (default `output/`) as `YYYYmmdd_HHMMSS_<name>.pt`; a `_N` suffix is added when two saves fall in the same second, so a file is never overwritten. A missing, corrupted or incompatible file raises `CheckpointError`.

## Integration points

| Where | Interaction |
|---|---|
| `ui.lab.prepare_data` | builds the corpus, encodes it and splits it; trains on all data if validation would be too small |
| `ui.lab.run_training` | iterates `train(...)`, appends each dictionary to `state["history"]`, and polls the stop flag through `should_stop` |
| `pages/page_training.py` | draws the live curves from the history and offers Start, Stop, Resume and checkpoint buttons |
| `llm.transformer.GPT.forward` | supplies the loss through `model(x, y)` |
| `config.load_settings` | provides `input_dir`, `output_dir` and the device |

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `INPUT_DIR` | `training.data` | folder scanned recursively for extra `.txt` and `.md` training text |
| `OUTPUT_DIR` | `training.checkpoint` | folder receiving timestamped checkpoints |
| `DEVICE` | `config.resolve_device` | `auto`, `cpu`, `cuda` or `mps` |

## Runtime behaviour

- A non-finite loss raises `FloatingPointError` with the advice to lower the learning rate. A `block_size` larger than the model context raises `ValueError` before any step runs.
- `should_stop` is polled before every step, so a stop request ends the generator cleanly after the current step.
- The batch generator is seeded with `cfg.seed + start_step`, so the same configuration gives the same losses; the reference run gives identical numbers on CPU and on MPS.
- **Resume caveat 1: the schedule restarts.** `train` indexes the schedule from 0 on every call, and `start_step` only shifts the step label and the seed. After a resume the learning rate jumps back to the warm-up value.
- **Resume caveat 2: the step budget is added.** Resuming runs another `cfg.steps` steps, so a 30-step run plus a resume gives 60 steps in the history.
- **Resume caveat 3: the optimiser is not restored from a file.** `optimizer_state` is saved and returned by `load_checkpoint`, but nothing restores it. A loaded checkpoint sets the session optimiser to `None`, so the Resume button is available within a session but not after loading a file.

## Tests

The suite checks that the hand-written Adam and SGD match `torch.optim` step for step, that the first Adam update is about the learning rate regardless of gradient scale, that clipping rescales to the requested norm and reports the pre-clip value, the schedule values at the boundaries, window shapes and the one-token shift between `x` and `y`, deterministic batches for a given seed, the stop flag, error cases, the checkpoint round trip and the non-overwrite rule, and that a tiny model's loss decreases.

## Common pitfalls

- **Judging a run by the first 300 steps.** The loss plateau near the unigram entropy is expected; longer runs drop sharply after it.
- **Assuming resume continues the schedule.** It does not; see the caveats above.
- **A very high learning rate.** The loop stops with `FloatingPointError` rather than continuing with NaNs.
- **Window longer than the context.** Lower `block_size` or raise the model context.
