# The training pipeline

## Abstract

Training turns a randomly initialised network into a language model by repeatedly nudging its parameters to lower the next-token loss. This document explains the whole loop: how text becomes windowed batches, how gradients are obtained by backpropagation, how Adam and SGD turn gradients into updates, why the learning rate follows a warm-up and cosine schedule, and why gradients are clipped. It then reads the reference run honestly: in {{fact:ref_steps}} steps the loss falls from {{fact:ref_first_train}} to {{fact:ref_last_train}}, but the validation loss of {{fact:ref_last_val}} shows the model has learned letter frequencies and not yet context.

## Data: from a token stream to batches

### Windows and the shift-by-one target

After tokenization the corpus is one long sequence of ids $d_1, d_2, \dots, d_N$. A training example is a window of $T+1$ consecutive ids. The model sees the first $T$ and is asked to predict, at every position, the id that follows:

$$x = (d_s,\dots,d_{s+T-1}), \qquad y = (d_{s+1},\dots,d_{s+T}).$$

Because the causal mask hides the future, one window supplies $T$ training examples in a single forward pass. A batch stacks $B$ windows whose start positions $s$ are drawn uniformly at random, so one step trains on $B\cdot T$ tokens. The reference run uses $B = {{fact:ref_batch}}$ and $T = {{fact:ref_block}}$, which is {{fact:ref_batch_tokens}} tokens per step.

### Train and validation split

The token stream is cut into a training part and a **validation** part that the optimiser never sees. The cut is *contiguous* (the last 10 %), not shuffled: a shuffled split would put neighbouring characters on both sides of the boundary and let the model "see" its validation text through overlapping context. For the bundled corpus the split is {{fact:train_tokens}} training and {{fact:val_tokens}} validation tokens.

!!! note "How much data is that?"
    The reference run processes {{fact:ref_total_tokens}} tokens in total, which is about {{fact:ref_epochs}} passes over the training split. With so few distinct tokens the model sees every training character many times; this is the root of the overfitting shown below.

## Backpropagation

With parameters $\theta$ and loss $\mathcal{L}(\theta)$, learning needs the gradient $g = \nabla_\theta\mathcal{L}$. The loss is a composition of differentiable operations, so the chain rule gives the gradient layer by layer, from the output backwards. For a composition $\mathcal{L} = f_n\circ\cdots\circ f_1$ with intermediate values $a_k = f_k(a_{k-1})$,

$$\frac{\partial\mathcal{L}}{\partial a_{k-1}} = \left(\frac{\partial a_k}{\partial a_{k-1}}\right)^{\!\top}\frac{\partial\mathcal{L}}{\partial a_k}.$$

Reverse-mode automatic differentiation (Rumelhart, Hinton and Williams, 1986) stores the forward activations and applies this recurrence once, so the cost of the full gradient is a small constant multiple of one forward pass, however many parameters there are. In PyTorch `loss.backward()` performs it and fills `p.grad` for every parameter. Residual connections help here: the identity path adds a term $1$ to each layer's Jacobian, so the gradient reaching early layers does not shrink geometrically with depth.

## Optimisers

An optimiser maps the gradient to an update. All of them below are written by hand in this project, with the same interface as `torch.optim`, and tested step by step against PyTorch's implementations.

### Plain gradient descent and momentum

The simplest rule moves against the gradient, $\theta\leftarrow\theta-\eta\,g$, with learning rate $\eta$. **Momentum** (Polyak, 1964) accumulates a velocity so consistent directions speed up and oscillations cancel:

$$b_t = \mu\,b_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\,b_t.$$

### Adam

Adam (Kingma and Ba, 2015) keeps two running averages per parameter, the first moment (mean) and the second moment (uncentred variance) of the gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^{2}.$$

Both start at zero, so early estimates are biased towards zero; the correction divides by $1-\beta^t$:

$$\hat m_t = \frac{m_t}{1-\beta_1^{t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{t}}, \qquad \theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.$$

The division by $\sqrt{\hat v_t}$ gives every parameter its own effective step size. A useful consequence: at the very first step $\hat m_1 = g_1$ and $\sqrt{\hat v_1} = |g_1|$, so the update has size about $\eta$ **regardless of how large or small the gradient is**. Measured with $\eta = 0.01$, a gradient of 123 and a gradient of 0.001 both moved the parameter by about 0.01. Adam is thus largely insensitive to the gradient's scale, which is one reason it is robust for transformers; the price is that $\eta$ must still be chosen with care.

### Weight decay: L2 versus AdamW

Weight decay shrinks weights to discourage large values. In **Adam with L2 regularisation** the penalty $\lambda\theta$ is added to the gradient, so it passes through the adaptive scaling and decays parameters unevenly. **AdamW** (Loshchilov and Hutter, 2019) *decouples* the decay: weights are first shrunk directly, $\theta\leftarrow(1-\eta\lambda)\theta$, and only then updated by the Adam rule. The reference run uses AdamW with $\lambda = 0.01$.

!!! note "A simplification to be aware of"
    The implementation applies the same weight decay to every parameter, including biases, LayerNorm scales and embeddings. Large-scale training usually excludes these. At this scale the difference is small.

### Choices offered

| Name | Meaning |
|---|---|
| `adam` | hand-written Adam with L2 decay |
| `adamw` | hand-written Adam with decoupled decay (default) |
| `sgd` | hand-written SGD with momentum |
| `torch_adamw`, `torch_sgd` | PyTorch's reference implementations, for comparison |

## Learning-rate schedule

A fixed learning rate is rarely best. At the start the weights are random and Adam's moment estimates are poor, so large steps can derail training; near the end smaller steps let the model settle. The schedule used here (two options exist: `warmup_cosine` and `constant`) has two phases (Loshchilov and Hutter, 2017; Goyal et al., 2017):

$$\eta_t = \begin{cases}\eta_{\max}\,\dfrac{t+1}{t_w} & t < t_w \quad\text{(linear warm-up)}\\[2ex] \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})\left(1+\cos\pi\dfrac{t-t_w}{T_{\text{total}}-t_w}\right) & t\ge t_w \quad\text{(cosine decay)}\end{cases}$$

with $\eta_{\min} = r\,\eta_{\max}$ (here $r = 0.1$). For the reference run, $\eta_{\max} = {{fact:ref_lr}}$, warm-up $t_w = {{fact:ref_warmup}}$ steps, and the rate ends at {{fact:ref_lr_final}}. Computed values: step 0 gives 0.00015, step 19 reaches 0.003, step 160 gives 0.00165 and step 299 gives 0.0003.

{{chart:lr}}

## Gradient clipping

Occasionally a batch produces a very large gradient (a loss spike), and one huge step can undo much progress. **Global-norm clipping** (Pascanu et al., 2013) rescales the whole gradient vector if its L2 norm exceeds a threshold $c$:

$$g\leftarrow g\cdot\min\!\left(1,\frac{c}{\lVert g\rVert_2}\right).$$

The direction is preserved; only the length is capped. The reported gradient norm is the value *before* clipping, which makes it a useful health signal. In the reference run it starts at {{fact:ref_gradnorm_first}}, peaks at about 1.44 and ends at {{fact:ref_gradnorm_last}}, with a clipping threshold of {{fact:ref_clip}}. The norm exceeded the threshold in only 14 of the {{fact:ref_steps}} steps, all of them among the first 14, so clipping acted only at the very start (the peak was at step 3); afterwards gradients were left untouched.

{{chart:gradnorm}}

## Reading the loss curves

{{chart:loss}}

The reference run (seed 42, AdamW, learning rate {{fact:ref_lr}}, batch {{fact:ref_batch}}, window {{fact:ref_block}}) shows three things.

1. **Fast initial drop.** The training loss falls from {{fact:ref_first_train}} to about 3.16 by step 20 and to about 3.06 by step 30. This is the model learning the *unigram distribution*: which characters are common.
2. **A plateau at the unigram floor.** The validation loss sits between {{fact:ref_val_plateau_lo}} and {{fact:ref_val_plateau_hi}} for the rest of the run, ending at {{fact:ref_last_val}}. The entropy of the character frequencies themselves is {{fact:unigram_train}} nats on the training text and {{fact:unigram_val}} on the validation text. A model that has learned *only* letter frequencies cannot do better than this, so after {{fact:ref_steps}} steps the model has **not yet learned to use context**. Its perplexity is {{fact:ref_ppl_val}} on validation, against {{fact:char_vocab}} for a uniform guess.
3. **Cheap to run.** The run takes {{fact:ref_seconds}} s on `{{fact:ref_device}}` (about {{fact:ref_tps}} tokens per second); the same seed gives identical losses on the CPU.

### Training for longer

Longer runs on the same corpus show what happens once the model starts using context:

| Run | Final training loss | Validation loss at step 750 | Validation at step 1,000 | Final validation loss |
|---|---|---|---|---|
| 1,500 steps, window 32 | {{fact:long1500_train}} | {{fact:long1500_val750}} | {{fact:long1500_val1000}} | {{fact:long1500_val}} |
| 3,000 steps, window 64 | {{fact:long3000_train}} | 2.643 | 2.373 | {{fact:long3000_val_last}} (best {{fact:long3000_val_best}} at step {{fact:long3000_val_best_step}}) |

Both longer runs stay on the unigram plateau (validation about 3.10 up to step 500 in the 3,000-step run) and then drop sharply between steps 500 and 750, well below the unigram entropy of {{fact:unigram_val}}. That sudden fall is the point at which the model starts to use context; the plateau is a stage the reference run never leaves within its {{fact:ref_steps}} steps. The 3,000-step run is the classic **overfitting** picture: training loss {{fact:long3000_train}} against a best validation loss of {{fact:long3000_val_best}} reached at step {{fact:long3000_val_best_step}}, then a worse validation loss at the end ({{fact:long3000_val_last}}). The gap between the curves means the model is memorising the {{fact:train_tokens}} training tokens rather than generalising. The usual remedies are more data, dropout, weight decay, a smaller model, and *early stopping* at the best validation step.

!!! warning "In-sample entropies are optimistic"
    Bigram and trigram conditional entropies of the training text (2.18 and 1.31 nats) are measured on the same text they describe, so they flatter a model that memorises. Only the validation loss measures generalisation.

## Evaluation

Every 25 steps by default (plus the first and the last step) the validation loss is estimated by averaging over 10 fixed random batches with dropout switched off. The batch generator is seeded with a constant, so two evaluations of identical weights return the same number and differences reflect the model rather than the sample.

## Checkpoints

A checkpoint stores the model weights, architecture config, tokenizer vocabulary (the ids mean nothing without it), the training config, the loss history and the optimiser state, so a run can be reproduced or inspected later. Files are named with a timestamp and are never overwritten.

## Limitations

1. **Tiny scale.** The conclusions about the plateau and overfitting are specific to this corpus and model size; scaling laws (Kaplan et al., 2020) describe how loss improves with parameters, data and compute at much larger scales.
2. **No data pipeline sophistication.** Windows are sampled with replacement and may overlap; there is no epoch shuffling, packing or mixed precision.
3. **Single seed.** The numbers come from one seed. Run-to-run variation exists and is not quantified here.
4. **Validation doubles as selection set.** There is no separate test split, so picking the best validation step would slightly overstate performance.

## Where to go next

The numbers and curves of this document are drawn by the [visualisation layer](07_visualization_explanation.html). The [implementation guide](06_training_pipeline_implementation_guide.html) shows the loop, the data structures, the checkpoint format and the resume behaviour.

## References

1. Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature 323.
2. Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics 4(5).
3. Kingma, D. P., Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
4. Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR.
5. Loshchilov, I., Hutter, F. (2017). *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR.
6. Goyal, P., et al. (2017). *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677.
7. Pascanu, R., Mikolov, T., Bengio, Y. (2013). *On the difficulty of training recurrent neural networks.* ICML.
8. Kaplan, J., et al. (2020). *Scaling Laws for Neural Language Models.* arXiv:2001.08361.
