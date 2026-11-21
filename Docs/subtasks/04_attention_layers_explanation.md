# Attention, multi-head attention, LayerNorm and the feed-forward network

## Abstract

These four components are the working parts of every transformer block. **Attention** lets each position read information from the others. **Multiple heads** let it read several kinds of information at once. **Layer normalisation** keeps activations at a usable scale as layers are stacked, and the **feed-forward network** transforms each position on its own. This document derives each from first principles, explains why the $1/\sqrt{d_k}$ scaling and the causal mask are needed, and ties the algebra to the parameter counts of the default model ({{fact:params_attention}} parameters in attention, {{fact:params_ffn}} in the feed-forward layers).

## Scaled dot-product attention

### The idea

Attention was introduced for machine translation by Bahdanau, Cho and Bengio (2015) as a soft alignment between source and target words, and became the sole sequence-mixing mechanism in the Transformer (Vaswani et al., 2017). For a sequence of $T$ vectors, every position produces three vectors with learned linear maps:

- a **query** $\mathbf{q}_t$: what this position is looking for;
- a **key** $\mathbf{k}_s$: what position $s$ offers to be found by;
- a **value** $\mathbf{v}_s$: the content position $s$ passes on if selected.

Stacking them as matrices $Q, K, V \in \mathbb{R}^{T \times d_k}$, the output is

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}} + M\right) V.$$

Row $t$ of $QK^\top$ contains the dot products of query $t$ with every key. The softmax turns that row into a probability distribution over positions (the *attention weights*), and the output at $t$ is the corresponding weighted average of the value vectors. $M$ is the mask described below.

### Why divide by $\sqrt{d_k}$

Suppose the components of $\mathbf{q}$ and $\mathbf{k}$ are independent with zero mean and unit variance. Then the dot product $\mathbf{q}\cdot\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$ has

$$\mathbb{E}[\mathbf{q}\cdot\mathbf{k}] = 0, \qquad \operatorname{Var}[\mathbf{q}\cdot\mathbf{k}] = d_k.$$

Its standard deviation grows like $\sqrt{d_k}$. Large scores push the softmax into a saturated regime in which one weight is almost 1 and the others almost 0, and the gradient through the softmax nearly vanishes. Dividing by $\sqrt{d_k}$ restores unit variance. This was checked numerically for $d_k \in \{4, 16, 64, 256\}$: the variance of the raw dot product is close to $d_k$ in every case, and the variance after scaling is about 1.00 in every case.

### The causal mask

A language model predicts token $t+1$ from tokens $1,\dots,t$. If position $t$ could attend to later positions, the answer would be available in the input and the model would learn to copy rather than predict. The mask enforces causality:

$$M_{ts} = \begin{cases} 0 & s \le t \\ -\infty & s > t. \end{cases}$$

Adding $-\infty$ before the softmax gives masked positions exactly zero weight. The attention matrix is therefore lower-triangular, and row $t$ always contains at least the position $t$ itself, so no row is entirely masked in normal use. The implementation still converts a hypothetical fully masked row to zeros instead of NaN.

!!! note "Causality is a testable property"
    Changing any token after position $t$ must leave all outputs at positions $\le t$ unchanged. The test suite checks exactly this, which is stronger than inspecting the mask visually.

### Cost

The score matrix has $T^2$ entries per head, so time and memory grow quadratically with the sequence length. For the context of {{fact:context}} tokens this is negligible; for long documents it is the central bottleneck that motivates sparse and linear attention variants (Child et al., 2019; Katharopoulos et al., 2020).

## Multi-head attention

A single attention distribution can express only one weighting of the past at a time. Multi-head attention runs $h$ attentions in parallel on lower-dimensional projections and combines the results:

$$\operatorname{head}_i = \operatorname{Attention}(XW_i^{Q}, XW_i^{K}, XW_i^{V}), \qquad \operatorname{MHA}(X) = \operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)\,W^{O},$$

with $d_k = d_{\text{model}}/h$. For the default model $d_{\text{model}} = {{fact:d_model}}$ and $h = {{fact:n_heads}}$, so each head has $d_k = {{fact:d_k}}$ dimensions.

**Cost is unchanged.** Splitting the width into heads keeps the total size of the projections the same as one full-width head: four $d\times d$ matrices ($W^Q, W^K, W^V, W^O$) plus biases, that is $4(d^2 + d)$ parameters, which is {{fact:params_attn_layer}} per layer for $d = {{fact:d_model}}$. Heads are therefore a way to *specialise* the same capacity, not to add more.

**What heads learn.** Analyses of trained transformers (Clark et al., 2019; Voita et al., 2019) found heads that attend to the previous token, to the same word elsewhere, to syntactic partners, or to separators; some heads can be removed with little loss. In the very small model used here the picture after {{fact:ref_steps}} training steps is modest. Measured on the first 48 characters of the corpus, averaging over heads and over positions 8 and later, the entropy of the attention distribution is about 3.25 nats at initialisation, which equals the entropy of a perfectly uniform distribution over the visible past. After training, layer 1 is still essentially uniform (3.25 nats), while layer 0 has become sharper (about 2.76 nats, and its largest weight rises from about 0.04 to about 0.16). So only the first layer has begun to favour particular positions; do not expect the clean head specialisation seen in large models.

## Layer normalisation

### Definition

Deep networks train more easily when each layer's inputs stay in a stable range. Ba, Kiros and Hinton (2016) proposed normalising across the *features of one token* rather than across the batch. For a vector $\mathbf{x}\in\mathbb{R}^d$:

$$\mu = \frac1d\sum_{j=1}^{d} x_j, \qquad \sigma^2 = \frac1d\sum_{j=1}^{d}(x_j-\mu)^2, \qquad \hat{x}_j = \frac{x_j-\mu}{\sqrt{\sigma^2+\varepsilon}}, \qquad y_j = \gamma_j\hat{x}_j+\beta_j.$$

The variance is the *biased* estimate (division by $d$), $\varepsilon = 10^{-5}$ prevents division by zero, and $\gamma,\beta\in\mathbb{R}^d$ are learned scale and shift, initialised to 1 and 0. A worked example for $\mathbf{x} = (1, 2, 3, 4)$: $\mu = 2.5$, $\sigma^2 = 1.25$, and the normalised vector is $(-1.342, -0.447, 0.447, 1.342)$.

### Why it fits language models

Unlike batch normalisation, the statistics depend on a single token, so the computation is identical during training and inference, does not depend on batch size, and works for sequences of any length. Each LayerNorm has $2d$ parameters; the default model has {{fact:params_layernorm}} in total (two per block plus one final).

### Pre-norm and post-norm

The original Transformer applied the normalisation *after* each sub-layer (post-norm). GPT-2 and most later models apply it *before* the sub-layer, inside the residual branch (pre-norm). Xiong et al. (2020) showed that pre-norm keeps gradients well-scaled at initialisation and removes the need for a long learning-rate warm-up. This project is **pre-norm**: the block computes $\mathbf{x} + \operatorname{Sublayer}(\operatorname{LN}(\mathbf{x}))$, which the next sub-task assembles.

## The position-wise feed-forward network

Attention is a weighted *average*, which is linear in the values. The non-linearity of a transformer lives in the feed-forward network, applied to each position independently with the same weights:

$$\operatorname{FFN}(\mathbf{x}) = W_2\,\phi(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2, \qquad W_1\in\mathbb{R}^{d_{ff}\times d},\; W_2\in\mathbb{R}^{d\times d_{ff}},\; d_{ff} = 4d.$$

For the default model $d_{ff} = {{fact:ffn_hidden}}$, which gives {{fact:params_ffn_layer}} parameters per layer, about twice the attention parameters of the same layer. Feed-forward layers hold most of a transformer's parameters; Geva et al. (2021) interpret them as key-value memories that store associations learned from the data.

### The activation function

The default activation is GELU (Hendrycks and Gimpel, 2016), which weights an input by the probability that a standard normal variable is below it:

$$\operatorname{GELU}(x) = x\,\Phi(x) \approx 0.5\,x\left(1+\tanh\!\left[\sqrt{2/\pi}\,\left(x+0.044715\,x^{3}\right)\right]\right).$$

The implementation uses the tanh approximation, as in GPT-2. Compared with ReLU, $\max(0,x)$, GELU is smooth and lets small negative inputs pass with a small weight, which gives non-zero gradients where ReLU has none. ReLU remains available as an option.

## What the pieces do together

| Component | Mixes across positions? | Non-linear? | Learnable parameters ($d = {{fact:d_model}}$) |
|---|---|---|---|
| Attention (all heads) | **yes** | only through the softmax | {{fact:params_attn_layer}} per layer |
| LayerNorm | no | yes (normalisation) | 128 per norm |
| Feed-forward | no | **yes** (GELU) | {{fact:params_ffn_layer}} per layer |

The division of labour is the central design idea: attention *moves* information between positions, and the feed-forward network *processes* it at each position.

## Limitations

1. **Quadratic cost** in sequence length, as discussed above.
2. **No built-in order.** Attention is permutation-equivariant; the order comes only from the positional encoding of the previous sub-task.
3. **Interpretation caution.** Attention weights show where information is read from, not necessarily why an output was produced (Jain and Wallace, 2019). They are a diagnostic, not an explanation.
4. **Untrained patterns.** The interface lets learners inspect attention on an untrained layer; those patterns are random-projection artefacts, not learned structure.

## Where to go next

With attention, normalisation and the feed-forward network in place, the next sub-task, [the full GPT model](05_gpt_model_explanation.html), stacks them with residual connections, adds the output head and the loss, and defines how text is generated. The [implementation guide](04_attention_layers_implementation_guide.html) lists the tensors and shapes in detail.

## References

1. Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
3. Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer Normalization.* arXiv:1607.06450.
4. Xiong, R., et al. (2020). *On Layer Normalization in the Transformer Architecture.* ICML.
5. Hendrycks, D., Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs).* arXiv:1606.08415.
6. Clark, K., Khandelwal, U., Levy, O., Manning, C. D. (2019). *What Does BERT Look At? An Analysis of BERT's Attention.* BlackboxNLP.
7. Voita, E., Talbot, D., Moiseev, F., Sennrich, R., Titov, I. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL.
8. Geva, M., Schuster, R., Berant, J., Levy, O. (2021). *Transformer Feed-Forward Layers Are Key-Value Memories.* EMNLP.
9. Jain, S., Wallace, B. C. (2019). *Attention is not Explanation.* NAACL.
10. Child, R., Gray, S., Radford, A., Sutskever, I. (2019). *Generating Long Sequences with Sparse Transformers.* arXiv:1904.10509.
11. Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. (2020). *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.* ICML.
