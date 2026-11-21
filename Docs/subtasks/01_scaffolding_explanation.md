# Project scaffolding and environment

## Abstract

Before a language model can be trained, the experiment needs a foundation that makes its results *repeatable* and *checkable*. This sub-task builds that foundation: a declared Python environment, configuration that lives outside the code, a deterministic random-number policy, a test harness for numerical code, and repository hygiene that keeps secrets and heavy artefacts out of version control. None of it changes what the model computes, yet every later number in this documentation ({{fact:params}} parameters, a loss falling from {{fact:ref_first_train}} to {{fact:ref_last_train}}) is only meaningful because of it.

## Why scaffolding is a scientific concern

Machine-learning results are unusually fragile. A training run depends on the software versions, the hardware, the random seed, the data split and the hyper-parameters. Henderson et al. (2018) showed that in deep reinforcement learning, differences that look like algorithmic progress can be produced by nothing more than a changed seed or implementation detail. Pineau et al. (2021) turned this into a reproducibility checklist for the field: report the code, the data, the compute environment and the variance between runs.

A teaching project has the same obligation, with one extra reason. A learner who re-runs an experiment and sees a *different* number cannot tell whether they misunderstood the idea or merely had a different seed. The scaffolding removes that ambiguity.

## Concepts

### 1. An isolated, declared environment

All dependencies are listed in `requirements.txt` and installed into a project-local virtual environment (`.venv`) with the `uv` resolver. Isolation matters because numerical libraries change behaviour between versions: kernels are fused differently, default tolerances move, and operators gain or lose determinism guarantees. A declared environment turns "works on my machine" into a statement that can be tested.

### 2. Configuration outside the code

The application reads every tunable value (port, device, default seed, input and output folders) from environment variables, optionally loaded from a `.env` file. This is the *configuration* factor of the twelve-factor methodology (Wiggins, 2011): code is identical across machines, and only the environment differs. It also keeps credentials, user names and URLs out of source files. A `.env.example` documents every variable and is the only variant committed.

### 3. A seed policy

A training run is a function of a random seed in three places: weight initialisation, the choice of training windows, and the choice of tokens when sampling text. The project seeds each place separately and explicitly:

- model construction uses `torch.manual_seed(settings.seed)` on the settings object;
- the trainer draws batches from its own `torch.Generator` seeded with `cfg.seed + start_step`, so a resumed run does not replay the same windows;
- evaluation uses a *fixed* generator, so two evaluations of the same weights return the same number and differences between evaluations reflect the model, not the sample.

With these three seeds, the reference run gives identical losses on CPU and on Apple's MPS device (to the printed precision), which is a stronger property than most small experiments can claim.

### 4. Testing numerical code

Tests for a model cannot compare floating-point numbers for equality, and they cannot rely on a trained model, because training is slow and noisy. The strategy is therefore to test *properties* that must hold for any weights:

| Property | Example check |
|---|---|
| Shape contracts | the output of every layer has the documented shape |
| Mathematical identities | attention weights sum to one over keys; LayerNorm output has zero mean and unit variance |
| Causality | changing a future token never changes an earlier output |
| Agreement with a reference | the hand-written LayerNorm and cross-entropy match PyTorch's within a tolerance |
| Determinism | the same seed gives the same loss |
| Failure modes | invalid configurations raise informative errors |

The application's own suite has **402 unit tests**, and the project rule is that every Python module has a matching `test_<module>.py`.

### 5. Repository hygiene

A research repository must not accumulate secrets, model weights or caches. The `.gitignore` excludes `.env` files, virtual environments, caches, archives, native libraries, and the contents of `input/` and `output/` (the folders themselves are kept with `.gitkeep`). Keeping checkpoints out of version control also keeps the repository small and makes the code, not a binary blob, the source of truth.

## Significance

The scaffolding is what lets the rest of the documentation make *quantitative* claims. When a later page says that the default model has {{fact:params}} parameters or that the reference run reaches a validation loss of {{fact:ref_last_val}}, those numbers are produced by the code in a declared environment, with a recorded seed, and can be regenerated with one command.

## Limitations and honest caveats

- **Determinism is bounded.** Identical seeds give identical results on the tested CPU and MPS devices for this model. Larger models, other back-ends or non-deterministic kernels can still differ in the last digits.
- **Versions are lower bounds.** `requirements.txt` states minimum versions rather than an exact lock file, which favours easy installation over bit-for-bit reproduction years later.
- **Tests verify properties, not quality.** A passing suite shows that the code implements the intended mathematics; it says nothing about how well a trained model writes.

## Where to go next

The environment hosts everything that follows. The first modelling step is [tokenization](02_tokenization_explanation.md), which turns text into the integers the model consumes. The matching [implementation guide](01_scaffolding_implementation_guide.html) shows the files, variables and scripts behind this page.

## References

1. Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018). *Deep Reinforcement Learning That Matters.* AAAI Conference on Artificial Intelligence.
2. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). *Improving Reproducibility in Machine Learning Research.* Journal of Machine Learning Research 22(164).
3. Wiggins, A. (2011). *The Twelve-Factor App.* Configuration: store config in the environment.
