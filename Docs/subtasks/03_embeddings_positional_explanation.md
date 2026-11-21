# Embeddings and positional encoding

## Abstract

Token ids are arbitrary labels: id 17 is not "closer" to id 18 than to id 40. An **embedding** replaces each id by a learnable vector so that similarity can be expressed as geometry. Self-attention, however, treats its input as an unordered set, so a **positional signal** must be added to tell the model where each token sits. This sub-task implements both from scratch and quantifies a practical subtlety: at initialisation the positional signal is much larger than the token signal.

## Embeddings as a lookup table

Let $E \in \mathbb{R}^{V \times d}$ be a matrix with one row per vocabulary entry, where $V$ is the vocabulary size and $d$ the model width. The embedding of token $i$ is simply row $i$:

$$\mathbf{e}_i = E_{i,:} = \mathbf{o}_i^{\top} E,$$

where $\mathbf{o}_i$ is the one-hot vector of token $i$. Multiplying a one-hot vector by a matrix selects a row, so the layer is an *index operation* and costs no matrix multiplication. Automatic differentiation reflects this: the gradient of the loss with respect to $E$ is non-zero only for the rows that were looked up in the batch.

The idea of learning distributed vector representations of words goes back to Bengio et al. (2003) and was popularised by word2vec (Mikolov et al., 2013). The distributional hypothesis behind it, "a word is characterised by the company it keeps" (Firth, 1957), explains why training moves tokens that occur in similar contexts closer together.

### Size

With the default settings the embedding table has $V \times d = {{fact:char_vocab}} \times {{fact:d_model}}$ parameters, which is **{{fact:params_embedding}}** parameters out of {{fact:params}} in the whole model. With a larger vocabulary, as in BPE, the table grows linearly in $V$.

### Initialisation

Three strategies are offered, all producing small random numbers so that no token starts out special:

| Strategy | Distribution | Intent |
|---|---|---|
| `normal` | $\mathcal{N}(0, \sigma^2)$ with $\sigma = 0.02$ | GPT-style default |
| `uniform` | $U(-a, a)$ with $a = \sigma\sqrt{3}$ | same variance $\sigma^2$, bounded support |
| `xavier` | $U(-b, b)$ with $b = \sqrt{6/(V + d)}$ | Glorot and Bengio (2010): keeps activation variance stable across layers |

For $\sigma = 0.02$ and $d = {{fact:d_model}}$ the expected norm of one row is $\sigma\sqrt{d} = 0.16$. In the reference run, the mean row norm was about 0.16 at initialisation and about 0.31 after training, so the table roughly doubles in length while the tokens organise themselves.

### Weight tying

When `tie_weights` is on, the output layer reuses the embedding matrix: the logit of token $j$ is the dot product between the final hidden state $\mathbf{h}$ and the embedding of $j$,

$$z_j = \mathbf{h}^{\top}\mathbf{e}_j.$$

Press and Wolf (2017) and Inan et al. (2017) showed that tying reduces parameters and often improves language-model quality. It also changes the gradient: with tying, **every** row of $E$ receives gradient from the output layer (all $V$ logits depend on it), whereas an untied input table only updates the rows used in the batch.

### An optional scale factor

The original Transformer multiplies embeddings by $\sqrt{d}$ before adding positions (Vaswani et al., 2017). This implementation offers the same option, `TokenEmbedding(scale=True)`, but it is **off by default**, and the GPT model does not enable it. The next section shows why that matters.

## Why position must be injected

Self-attention computes, for every query, a weighted average of values over all keys. Permuting the input tokens permutes the output in the same way, so the layer is *permutation equivariant*; "dog bites man" and "man bites dog" would produce the same set of vectors. A language model needs order, so a position-dependent vector $\mathbf{p}_t \in \mathbb{R}^d$ is added to the token embedding:

$$\mathbf{x}_t = \mathbf{e}_{x_t} + \mathbf{p}_t.$$

### Sinusoidal encoding

Vaswani et al. (2017) proposed a fixed, parameter-free choice:

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right).$$

Each pair of dimensions is a sine and a cosine of one frequency $\omega_i = 10000^{-2i/d}$. The wavelengths form a geometric progression from $2\pi$ (the fastest pair, distinguishing neighbouring positions) to $10000\cdot 2\pi$ (the slowest, a coarse position signal). For $d = {{fact:d_model}}$ the pairs $i = 0, 16, 31$ have wavelengths of about 6.3, 628 and 47,117 positions.

Two properties motivate the design:

1. **Linear relative shifts.** By the angle-addition identities, $PE_{pos+k}$ is a linear function of $PE_{pos}$:

$$\begin{pmatrix}\sin(\omega(pos+k)) \\ \cos(\omega(pos+k))\end{pmatrix} = \begin{pmatrix}\cos\omega k & \sin\omega k \\ -\sin\omega k & \cos\omega k\end{pmatrix}\begin{pmatrix}\sin(\omega\, pos) \\ \cos(\omega\, pos)\end{pmatrix}.$$

   A network can therefore learn to attend to "the token $k$ steps back" with a fixed linear map.
2. **No length limit by construction.** The formula is defined for any $pos$, although the model is only trained on positions it has seen.

Every row has the same length: each sine/cosine pair contributes $\sin^2 + \cos^2 = 1$, so $\lVert PE_{pos}\rVert = \sqrt{d/2}$, which is 5.657 for $d = {{fact:d_model}}$.

### Learned positional embeddings

GPT-2 (Radford et al., 2019) instead learns a table $P \in \mathbb{R}^{L \times d}$ with one trainable row per position, initialised like the token table. It is more flexible, but it can only represent positions up to $L$ and has no built-in notion of distance. The project offers both so the effect can be compared.

## A scale mismatch worth knowing

Compare the two magnitudes at initialisation, for $d = {{fact:d_model}}$:

| Signal | Norm per vector |
|---|---|
| Token embedding (random, $\sigma = 0.02$) | about 0.16 |
| Sinusoidal position | 5.657 |

The positional signal is roughly **35 times larger** than the token signal when training starts. The $\sqrt{d}$ factor of the original paper addresses exactly this imbalance, with $\sqrt{64} = 8$ raising the token norm to about 1.3. This project keeps the unscaled version because it follows the GPT-2 convention, and because two things compensate: the first LayerNorm (applied before attention in each block) rescales the sum, and the token table grows during training (about 0.31 in the reference run). The likely consequence is that early in training the representation is dominated by *position* until the token part grows; this is an expectation from the norms, not something measured separately here.

!!! note "A teaching opportunity"
    Setting `scale=True` in `TokenEmbedding`, or inspecting the norms on the positional-encoding page, makes the effect visible. It is a good reminder that "small initialisation" and "fixed encoding" are two decisions that interact.

## Geometry of the learned space

After training, nearby points in embedding space tend to be tokens used in similar contexts. Two diagnostics are used in the interface:

- **Cosine similarity**, $\cos\theta = \dfrac{\mathbf{u}\cdot\mathbf{v}}{\lVert\mathbf{u}\rVert\,\lVert\mathbf{v}\rVert}$, which ignores length and measures direction only;
- **Principal-component projection** to two dimensions, which shows the largest directions of variation. A projection of a 64-dimensional space to two dimensions discards most of the structure, so clusters are suggestive, not proof.

With {{fact:ref_steps}} training steps on a small corpus the model has learned only letter frequencies (see the training explanation), so do not expect rich semantic structure; the geometry at this scale mainly separates frequent from rare characters.

## Limitations

1. **Absolute positions only.** Both encodings give each token an absolute index. Later work, such as relative position biases (Shaw et al., 2018) and rotary embeddings (Su et al., 2021), encodes the *distance* between tokens inside attention.
2. **Fixed context.** The learned table cannot go beyond the context length it was built with; here the context is {{fact:context}} tokens.
3. **Embeddings are static per token.** A token has one vector regardless of context; the context-dependent meaning is built by the layers above.

## Where to go next

The vectors now carry token identity and position. The next sub-task, [attention, layer normalisation and the feed-forward network](04_attention_layers_explanation.html), lets each position read from the others. See the [implementation guide](03_embeddings_positional_implementation_guide.html) for the code.

## References

1. Bengio, Y., Ducharme, R., Vincent, P., Jauvin, C. (2003). *A Neural Probabilistic Language Model.* Journal of Machine Learning Research 3.
2. Mikolov, T., Chen, K., Corrado, G., Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781.
3. Firth, J. R. (1957). *A synopsis of linguistic theory 1930-1955.* Studies in Linguistic Analysis.
4. Glorot, X., Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
5. Press, O., Wolf, L. (2017). *Using the Output Embedding to Improve Language Models.* EACL.
6. Inan, H., Khosravi, K., Socher, R. (2017). *Tying Word Vectors and Word Classifiers.* ICLR.
7. Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need.* NeurIPS.
8. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
9. Shaw, P., Uszkoreit, J., Vaswani, A. (2018). *Self-Attention with Relative Position Representations.* NAACL.
10. Su, J., et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864.
