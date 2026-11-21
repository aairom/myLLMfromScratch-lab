# The full transformer (GPT) and text generation

## Abstract

This sub-task assembles the parts of the previous three into a **decoder-only transformer**, the architecture family of GPT (Radford et al., 2018, 2019). It explains the residual stream, the pre-norm block, the tied output head and weight initialisation, defines the training loss (cross-entropy), and describes how text is generated one token at a time with temperature, top-$k$ and nucleus sampling. The default model has **{{fact:params}}** parameters.

## The model as a function

A GPT with $N$ layers maps a sequence of token ids $x_1,\dots,x_T$ to a matrix of scores (*logits*) $z_{t}\in\mathbb{R}^{V}$, one row per position, where $z_t$ scores every vocabulary entry as the possible *next* token:

$$P_\theta(x_{t+1}\mid x_{\le t}) = \operatorname{softmax}(z_t).$$

The computation is a fixed pipeline:

$$\begin{aligned} \mathbf{h}^{(0)}_t &= E_{x_t} + \mathbf{p}_t \\ \mathbf{h}^{(\ell)} &= \operatorname{Block}_\ell\!\left(\mathbf{h}^{(\ell-1)}\right), \quad \ell = 1,\dots,N \\ z_t &= \operatorname{LN}_f\!\left(\mathbf{h}^{(N)}_t\right) E^{\top}. \end{aligned}$$

The last line is the tied output head: the logit of token $j$ is the dot product of the final hidden state with the embedding of $j$.

## The residual stream and the pre-norm block

Each block updates a running vector, the **residual stream**, by *adding* corrections to it. With the pre-norm layout used here (GPT-2, Radford et al., 2019):

$$\mathbf{u} = \mathbf{h} + \operatorname{MHA}\!\bigl(\operatorname{LN}_1(\mathbf{h})\bigr), \qquad \mathbf{h}' = \mathbf{u} + \operatorname{FFN}\!\bigl(\operatorname{LN}_2(\mathbf{u})\bigr).$$

The identity path ($\mathbf{h}\to\mathbf{u}\to\mathbf{h}'$) is the crucial feature, introduced as *residual connections* by He et al. (2016) for image networks. It gives gradients a direct route from the loss to every layer, so depth does not make optimisation exponentially harder. Elhage et al. (2021) describe the same structure as a communication channel: attention heads and feed-forward layers *read from* and *write to* a shared stream.

### Measured: blocks write small corrections

At initialisation the stream entering the first block has a mean vector norm of about 5.66, almost entirely the positional signal (see the embedding document). The attention and feed-forward outputs added by each block have mean norms of only about 0.10 to 0.11. Each sub-layer therefore starts as a gentle perturbation of the stream, and the loss at initialisation is close to the uniform-guess value: about 3.95 on a batch of random token ids, against $\ln({{fact:char_vocab}}) = {{fact:ln_vocab}}$.

!!! note "Why initialise residual outputs smaller"
    The last projection in each residual branch (the attention output projection and the second feed-forward matrix) is initialised with standard deviation $0.02/\sqrt{2N}$, which is {{fact:res_std}} for $N = {{fact:n_layers}}$. A stream receives $2N$ additions; if each had variance $\sigma^2$, their sum would have variance $2N\sigma^2$. Scaling by $1/\sqrt{2N}$ keeps the stream's growth independent of depth, as in GPT-2. All other linear layers use standard deviation $0.02$.

## Anatomy of the default model

| Component | Parameters |
|---|---|
| Token embedding (tied with the output head) | {{fact:params_embedding}} |
| Positional encoding (sinusoidal, not trainable) | 0 |
| Attention, {{fact:n_layers}} layers | {{fact:params_attention}} |
| Feed-forward, {{fact:n_layers}} layers | {{fact:params_ffn}} |
| LayerNorm (2 per block + 1 final) | {{fact:params_layernorm}} |
| Output head (tied) | 0 |
| **Total** | **{{fact:params}}** |

Two features are worth noting. First, the feed-forward layers hold about two thirds of all parameters. Second, because the output head is tied, a vocabulary of $V$ entries adds $V\cdot d$ parameters once, not twice. With a larger vocabulary (BPE) the embedding share grows; with a tiny character vocabulary it is small.

## The training objective

The model is trained to maximise the likelihood of the training text, equivalently to minimise the **cross-entropy** between the data distribution and the model's predictions. For $N_{tok}$ prediction targets:

$$\mathcal{L}(\theta) = -\frac{1}{N_{tok}}\sum_{i=1}^{N_{tok}} \log \operatorname{softmax}(z_i)_{y_i}.$$

Computing $\log\operatorname{softmax}$ directly overflows for large logits, so the implementation uses the identity

$$\log\operatorname{softmax}(z)_y = z_y - \log\sum_j e^{z_j} = z_y - \operatorname{logsumexp}(z),$$

with `logsumexp` evaluated in a numerically stable way (it subtracts the maximum internally). The result equals PyTorch's reference cross-entropy to within a small tolerance, which the test suite verifies.

**Reference values.**

- A model that guesses uniformly has loss $\ln V = {{fact:ln_vocab}}$ nats per token.
- A model that knows only the *frequencies* of the characters reaches the unigram entropy, about {{fact:unigram_train}} nats on the training split.
- **Perplexity** is $e^{\mathcal{L}}$, the effective number of equally likely choices the model hesitates between. A loss of {{fact:ref_last_val}} corresponds to a perplexity of {{fact:ref_ppl_val}} for the reference run, compared with {{fact:char_vocab}} for a uniform guess.

Every position predicts its own next token in a single forward pass, so a window of $T$ tokens provides $T$ training examples at once. This is *teacher forcing*: the model always sees the true previous tokens while learning, and the causal mask guarantees that it cannot peek at the answer.

## Generation

At inference the model is run **autoregressively**: predict a distribution for the next token, choose one, append it, and repeat.

1. Crop the sequence to the last $L$ tokens, the context window, because positions beyond it are not defined.
2. Run the model and keep the logits of the *last* position only.
3. Convert them to a probability distribution, with optional adjustments below.
4. Sample a token from that distribution and append it.

### Temperature

Dividing the logits by a temperature $\tau>0$ before the softmax changes how peaked the distribution is:

$$p_j(\tau) = \frac{e^{z_j/\tau}}{\sum_k e^{z_k/\tau}}.$$

For $\tau<1$ the distribution sharpens toward the largest logit; for $\tau>1$ it flattens; $\tau\to0$ is greedy decoding, the arg-max, which is how `temperature = 0` is implemented. Worked example with logits $(2.0, 1.0, 0.1)$:

| Setting | Probabilities |
|---|---|
| $\tau = 1$ | 0.659, 0.242, 0.099 |
| $\tau = 0.5$ | 0.864, 0.117, 0.019 |
| $\tau = 2$ | 0.502, 0.304, 0.194 |
| top-$k$ with $k=2$ | 0.731, 0.269, 0 |
| top-$p$ with $p=0.8$ | 0.731, 0.269, 0 |

### Top-$k$ and nucleus (top-$p$) sampling

Pure sampling occasionally picks a very unlikely token, after which the continuation can derail. **Top-$k$** keeps only the $k$ highest logits (Fan et al., 2018). **Nucleus sampling** (Holtzman et al., 2020) keeps the smallest set of tokens whose cumulative probability reaches $p$, so the number of candidates adapts to how confident the model is. After either filter the remaining probabilities are renormalised. The implementation always retains at least one token.

### Greedy versus sampled text

Greedy decoding is deterministic and tends to repeat itself; sampling is varied but can be incoherent. The choice is a quality-diversity trade-off, not a correctness question, and the interface lets the learner vary all three parameters.

## What a small model can and cannot do

The reference run reaches a validation loss of {{fact:ref_last_val}} after {{fact:ref_steps}} steps, and the unigram entropy of the validation text is {{fact:unigram_val}}. The model has therefore learned **character frequencies**, not yet context, and its samples look like random letters with the right proportions.

Training longer helps: {{fact:long1500_val}} validation loss after 1,500 steps, which is clearly below the unigram level, so the model now uses context. Even then, a corpus of {{fact:corpus_chars}} characters is far too small for fluent text, and long training shows overfitting (see the training explanation). Text sampled after 3,000 steps is word-like but nonsensical, which is what one expects from a model that has memorised local spelling patterns of a tiny corpus.

## Limitations

1. **Scale.** {{fact:params}} parameters is about three orders of magnitude fewer than GPT-2 small (124 million), and a corpus of {{fact:corpus_chars}} characters is many orders of magnitude smaller than the web-scale text such models see. Emergent abilities reported for large models (Wei et al., 2022) are out of reach by design.
2. **No key-value cache.** Generation recomputes the whole window for every new token, costing $O(T^2)$ attention per step. Production systems cache keys and values; this implementation favours clarity.
3. **Context length.** Text longer than the window is cropped from the left, so the model forgets the beginning.
4. **Sampling on CPU.** The random draw is made on the CPU so a seeded CPU generator gives identical results on any device, including Apple's MPS.

## Where to go next

The model exists but has random weights. The next sub-task, [the training pipeline](06_training_pipeline_explanation.html), explains how its parameters are fitted. The [implementation guide](05_gpt_model_implementation_guide.html) shows the forward pass in code and the shapes at every stage.

## References

1. Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training.* OpenAI.
2. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
3. He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Elhage, N., et al. (2021). *A Mathematical Framework for Transformer Circuits.* Transformer Circuits Thread.
5. Fan, A., Lewis, M., Dauphin, Y. (2018). *Hierarchical Neural Story Generation.* ACL.
6. Holtzman, A., Buys, J., Du, L., Forbes, M., Choi, Y. (2020). *The Curious Case of Neural Text Degeneration.* ICLR.
7. Wei, J., et al. (2022). *Emergent Abilities of Large Language Models.* Transactions on Machine Learning Research.
8. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
