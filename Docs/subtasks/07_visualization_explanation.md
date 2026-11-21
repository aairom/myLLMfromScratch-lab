# Visualization of tensors and training dynamics

## Abstract

A transformer is a pile of tensors, and tensors are hard to read as numbers. This sub-task builds the layer that turns them into pictures: heat maps for matrices, a two-dimensional projection for embeddings, small multiples for heads, and line charts for training. The document explains the scientific ideas behind the figures (principal component analysis, cosine similarity, perceptual encodings of magnitude, smoothing) and the design rules that keep the figures honest: a figure must never invent structure that is not in the data, and it must say plainly when there is nothing to show.

## Why visualise a model at all

Neural networks are opaque by construction: millions of numbers interact non-linearly. Visualisation is one of the oldest tools for **interpretability**. Attention maps were used from the first neural translation systems to check that the model aligned words sensibly (Bahdanau et al., 2015), and projections of word vectors showed that directions in the embedding space encode relations (Mikolov et al., 2013). Two cautions apply, and the application states both next to its figures:

1. A picture is a *view* of the data, not an explanation of the model. An attention weight says where information *could* flow, not that the prediction depends on it (Jain and Wallace, 2019).
2. Every projection discards information. A 2-D scatter of 64-dimensional vectors shows the largest directions of variation and nothing else.

## From a tensor to a visual channel

Cleveland and McGill (1984) ranked how accurately people decode quantitative information from different visual channels. Position along a common scale is decoded best, length next, and colour intensity worst. The consequence for this project is a division of labour:

| Data | Typical shape | Encoding | Why |
|---|---|---|---|
| Loss, learning rate, gradient norm over steps | a sequence | line (position on two aligned scales) | accurate, shows trend and noise |
| Parameters per component | a handful of totals | horizontal bar (length) | easy comparison |
| Attention weights, embeddings, positional table | a matrix | heat map (colour) | the only practical encoding for hundreds of cells, so a colour bar and a fixed range are always provided |
| Embeddings of many tokens | $N \times D$ | 2-D scatter after projection | position again, at the price of a projection |

### Colour scales and the zero point

Two kinds of matrices need two kinds of colour scale.

- **Diverging data** such as embedding components, queries, keys, raw attention scores and normalised activations have a meaningful zero. A diverging palette centred on zero (`RdBu`, `zmid = 0`) makes the sign visible.
- **Probabilities** such as attention weights live in $[0, 1]$. A sequential palette (`Blues`) with the range pinned to $[0, 1]$ makes different heads and different runs comparable. Letting each figure auto-scale would make a nearly uniform attention matrix look as contrasted as a sharp one.

Cosine similarity is pinned to $[-1, 1]$ for the same reason.

## Projecting embeddings: principal component analysis

An embedding table has one row per token and $D$ columns. To draw it we need a map $\mathbb{R}^D \to \mathbb{R}^2$ that keeps as much of the structure as possible. Principal component analysis (Pearson, 1901; Hotelling, 1933) chooses the linear map that preserves the most **variance**.

Let $X \in \mathbb{R}^{N \times D}$ hold the vectors as rows and $\bar{X} = X - \mathbf{1}\boldsymbol{\mu}^{\top}$ the same rows with the mean subtracted. The singular value decomposition

$$\bar{X} = U \Sigma V^{\top}$$

gives the principal directions as the columns of $V$ and the variance along each as $\sigma_i^2 / (N-1)$. The 2-D coordinates are the projections on the first two directions,

$$Z = \bar{X}\,V_{:,1:2}, \qquad \text{explained variance ratio of component } i:\quad \rho_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}.$$

The figure title reports $\rho_1 + \rho_2$, so a reader always knows how much of the data the picture shows.

!!! note "Reading a PCA plot of a freshly initialised model"
    The default token table is drawn from a normal distribution, so no direction is special. Measured on the default model before training, the first two components explain only about 12 % of the variance (6.4 % and 5.9 %). That is what random data looks like: the scatter shows no structure because there is none. Compare it with the sinusoidal positional table, which is highly structured: its first two components explain about 37 % of the variance (23 % and 14 %).

### Why the sign of a component is arbitrary

If $v$ is a principal direction then so is $-v$. A figure may therefore flip between runs without anything having changed. The implementation does not try to fix a sign; it only guarantees that the explained variance and the pairwise distances are the same.

### Degenerate inputs

With a single row, no column, or identical rows, the variance is zero and the decomposition carries no information. The function returns zero coordinates and zero explained variance rather than raising an exception. This is a design choice: a page that is being explored interactively must not crash when the user selects one token.

## Cosine similarity

Euclidean distance between embeddings mixes two things, the **direction** of a vector and its **length**. Cosine similarity isolates the direction:

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert} \in [-1, 1].$$

Dividing every row by its norm gives unit vectors $\hat{\mathbf{a}}_i$, and the whole matrix is one product, $S = \hat{X}\hat{X}^{\top}$, whose diagonal is exactly 1. A tiny constant keeps the division defined for a zero vector.

Two measured baselines help to read the heat maps:

- **Random token vectors.** In 64 dimensions two independent random vectors are nearly orthogonal. For the 51 default tokens the off-diagonal similarities have mean about 0.003 and standard deviation about 0.13. A trained table should deviate from this baseline for tokens that behave alike.
- **Sinusoidal positions.** Neighbouring positions are similar and distant ones less so. For the 64-position table the similarity between positions 0 and 1 is about 0.97, between 0 and 32 about 0.61, and between 0 and 63 about 0.45. The matrix shows a smooth band structure around the diagonal, which is the visible trace of the multi-frequency design of sub-task 3.

## Visualising attention

An attention heat map has the **query** (the position that is looking) on the rows and the **key** (the position being looked at) on the columns. Three conventions matter:

1. **Rows sum to one**, so a row is a probability distribution and can be read as "where does this token get its information".
2. **The causal mask is shown as blank cells**, not as zeros. Before the softmax the masked scores are $-\infty$; the helper maps non-finite values to `NaN`, which the heat map leaves empty. A reader can then see which cells were *excluded* and which were merely *small*.
3. **Repeated tokens need unique labels.** A categorical axis merges equal labels, so two occurrences of the letter `e` would be drawn as one row. Labels are prefixed with the position (`0·e`, `4·e`) and invisible characters are replaced by visible stand-ins (space becomes `␣`).

For several heads, **small multiples** (Tufte, 1983) place the patterns side by side on identical colour scales, which makes the difference between heads the visible variable.

!!! note "What the default model shows"
    At initialisation the attention is nearly uniform: the mean entropy of the weights over the first 48 positions is about 3.25 nats and the largest weight about 0.045. After the 300-step reference run, the first layer has moved to an entropy of about 2.76 nats and a largest weight of about 0.16, while the second layer is still almost uniform. The heat maps therefore show a weak but real change in layer 0 and nothing in layer 1, which agrees with the finding that the short run learns only character statistics.

## Visualising layer normalisation

Two views are used. An **overlaid histogram** of all activations before and after normalisation shows the change of scale. A **per-token bar chart** of the mean and the standard deviation shows the mechanism itself: after normalisation every token vector has mean about 0 and standard deviation about 1 (before applying the learnable gain and bias), whatever its original scale.

## Visualising training

Training produces a **time series** per metric. Four points of method:

### Smoothing

Per-step losses are noisy because every step uses a different random batch. A trailing moving average of window $w$,

$$\bar{\ell}_t = \frac{1}{\min(w, t)} \sum_{i = \max(1, t - w + 1)}^{t} \ell_i,$$

reduces the variance of the estimate by about a factor $w$ for independent noise, at the price of a delay of about $(w-1)/2$ steps. The raw series is kept visible at low opacity so smoothing never hides a spike. The first $w - 1$ points average over fewer samples instead of being dropped.

### Reference line

The loss of a model that guesses uniformly is $\ln V$, here $\ln 51 \approx$ {{fact:ln_vocab}}. Drawing it as a dotted line answers the question every beginner asks, "is this loss good or bad?". The first loss of the reference run, {{fact:ref_first_train}}, sits just above that line, and the training loss of the final step is {{fact:ref_last_train}}.

### Missing values

Validation loss exists only at some steps. The series extraction skips entries that are missing, `None` or non-finite, so the validation curve is drawn through its own points instead of being broken by gaps.

### Logarithmic axes

A log axis is optional. It helps when a loss spans orders of magnitude, but for losses between 2 and 4 a linear axis is more honest, because equal vertical distances then mean equal absolute differences.

{{chart:loss}}

## Visualising the data flow

The data-flow diagram is generated from the *configuration and tensor shapes only*, without running the model. Each box lists a stage, a one-line description and the tensor shape, for example `(1, 64, 64)` for the residual stream and `(1, 64, 51)` for the logits of the default model. Because it is derived from `GPTConfig`, the diagram cannot disagree with the model: change the number of layers or the width and the picture changes with it. A test builds a real model and checks that the shapes in the diagram equal the shapes of the actual forward pass.

The **parameter bar chart** answers a different question: where do the {{fact:params}} parameters live? For the default model, the feed-forward layers hold {{fact:params_ffn}}, attention {{fact:params_attention}}, the token table {{fact:params_embedding}} and the layer norms {{fact:params_layernorm}}. The feed-forward networks hold about two thirds of the parameters, in line with the usual observation for transformers whose hidden width is four times the model width.

## Design principles used throughout

1. **No figure without a purpose and a key.** Every figure answers one question and has a title that states it.
2. **Empty is a state, not an error.** When there is no data (before training starts, no tokens, wrong matrix rank) a blank figure carries a short message. Pages never fail because a plot has nothing to draw.
3. **Pure functions.** Builders take tensors or lists and return a Plotly figure. They import no Streamlit and no model code beyond the configuration dataclass, so they can be tested without a browser.
4. **Honest scales.** Fixed ranges for probabilities and similarities, a zero-centred scale for signed data, and the explained variance in every projection title.
5. **Accessible encodings.** Colour is never the only carrier of meaning: matrices have labelled axes and a colour bar, and tokens that are whitespace are spelled out.

## Limitations

1. **Linear projection only.** PCA preserves global variance, not neighbourhoods. Non-linear methods such as t-SNE (van der Maaten and Hinton, 2008) or UMAP preserve local structure better but have hyper-parameters that change the picture, which is a poor fit for a teaching tool whose figures should be reproducible.
2. **Attention is not explanation.** See Jain and Wallace (2019) and the reply by Wiegreffe and Pinter (2019).
3. **Small data.** With 51 tokens, all point labels fit on a scatter plot. A realistic vocabulary of tens of thousands of tokens would need sampling or interactive filtering.
4. **Plotly in the browser.** Interactivity (hover, zoom) depends on the browser; the explanation pages in this documentation use static SVG instead.

## Where to go next

The figures are placed on the learning pages by the [Streamlit application](09_streamlit_app_explanation.html). The text and formulas that sit beside them come from the [educational content layer](08_educational_content_explanation.html). The [implementation guide](07_visualization_implementation_guide.html) lists every builder, its arguments and its test.

## References

1. Cleveland, W. S., McGill, R. (1984). *Graphical perception: theory, experimentation, and application to the development of graphical methods.* Journal of the American Statistical Association 79(387).
2. Tufte, E. R. (1983). *The Visual Display of Quantitative Information.* Graphics Press.
3. Pearson, K. (1901). *On lines and planes of closest fit to systems of points in space.* Philosophical Magazine 2(11).
4. Hotelling, H. (1933). *Analysis of a complex of statistical variables into principal components.* Journal of Educational Psychology 24.
5. Bahdanau, D., Cho, K., Bengio, Y. (2015). *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR.
6. Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS.
7. Jain, S., Wallace, B. C. (2019). *Attention is not Explanation.* NAACL.
8. Wiegreffe, S., Pinter, Y. (2019). *Attention is not not Explanation.* EMNLP.
9. van der Maaten, L., Hinton, G. (2008). *Visualizing Data using t-SNE.* Journal of Machine Learning Research 9.
