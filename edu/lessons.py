"""Lesson content for every component: intuition, maths, shapes, pitfalls, walkthrough.

Each :class:`Lesson` is plain data (no Streamlit), so it is easy to test and
reuse. ``formulas`` hold LaTeX for ``st.latex``; ``snippets`` hold dotted paths
resolved by :mod:`edu.snippets` so the code shown is the code that runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Formula:
    """A LaTeX expression with a plain-language reading of it."""

    latex: str
    meaning: str


@dataclass(frozen=True)
class Lesson:
    key: str
    title: str
    summary: str  # one line, used on the home page
    intuition: str  # markdown
    steps: tuple[str, ...]  # ordered walkthrough, one sentence each
    formulas: tuple[Formula, ...]
    shapes: tuple[tuple[str, str, str], ...]  # (tensor, shape, meaning)
    pitfalls: tuple[str, ...]
    snippets: tuple[str, ...]  # dotted paths for edu.snippets.get_source
    related: tuple[str, ...] = field(default_factory=tuple)  # keys of related lessons


_LESSONS: tuple[Lesson, ...] = (
    Lesson(
        key="tokenization",
        title="Tokenization",
        summary="Turn text into integer ids from a fixed vocabulary.",
        intuition=(
            "A neural network can only process numbers, so text must first be cut into **tokens** "
            "and each token mapped to an integer id. The set of all tokens is the **vocabulary**.\n\n"
            "* **Character level** - one token per character. The vocabulary is tiny, but sequences are long.\n"
            "* **Byte-Pair Encoding (BPE)** - start from characters and repeatedly merge the most frequent "
            "neighbouring pair into a new token. Frequent words become single tokens, so sequences get shorter "
            "while the vocabulary grows.\n\n"
            "Trade-off: a larger vocabulary shortens sequences (cheaper attention) but gives every token fewer "
            "training examples."
        ),
        steps=(
            "Split the text into pieces (characters, or word-like chunks for BPE).",
            "Count adjacent pairs of symbols across the corpus (BPE only).",
            "Merge the most frequent pair into a new token and record the merge (BPE only).",
            "Repeat until the requested number of merges is reached.",
            "Assign every distinct token an integer id; id 0 is reserved for unknown input.",
            "Encode: apply the merges in order, then look up the ids. Decode: join the token strings.",
        ),
        formulas=(
            Formula(r"\text{compression} = \frac{\#\text{characters}}{\#\text{tokens}}",
                    "How many characters one token covers on average; BPE raises it above 1."),
            Formula(r"\text{pair}^{*} = \arg\max_{(a,b)} \; \mathrm{count}(a, b)",
                    "BPE merges the most frequent adjacent pair at every step."),
        ),
        shapes=(
            ("text", "str", "raw input"),
            ("ids", "(T,)", "T integer token ids"),
        ),
        pitfalls=(
            "Characters not seen while building the vocabulary become `<unk>` and cannot be recovered on decode.",
            "More BPE merges are not always better: rare tokens receive very few training examples.",
            "The tokenizer used for training must be the one used at inference; checkpoints store it.",
        ),
        snippets=(
            "llm.tokenizer.CharTokenizer",
            "llm.tokenizer.BPETokenizer.train",
            "llm.tokenizer.BPETokenizer._bpe_word",
        ),
        related=("embeddings",),
    ),
    Lesson(
        key="embeddings",
        title="Token embeddings",
        summary="Look up a learnable vector for every token id.",
        intuition=(
            "An id such as `17` carries no meaning by itself. An **embedding** is a learnable table with one "
            "row per token; the row is the token's vector. Training moves these vectors so that tokens used "
            "in similar contexts end up pointing in similar directions.\n\n"
            "Mechanically the layer is just *indexing*: `E[ids]`. Only the rows that were looked up receive "
            "gradients."
        ),
        steps=(
            "Create a matrix E of shape (vocab_size, d_model) and initialise it with small random numbers.",
            "For each id in the input, pick the matching row of E.",
            "Optionally multiply by sqrt(d_model) so the embeddings are not drowned by positional signals.",
            "During backpropagation, add gradient only to the rows that were used.",
        ),
        formulas=(
            Formula(r"\mathbf{x}_t = E[\,i_t\,] \in \mathbb{R}^{d_{\text{model}}}",
                    "The vector of token t is row i_t of the embedding matrix."),
            Formula(r"\cos(\mathbf{a},\mathbf{b}) = \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}",
                    "Cosine similarity measures how aligned two embeddings are (1 = same direction)."),
        ),
        shapes=(
            ("E", "(V, D)", "learnable table; V = vocab size, D = d_model"),
            ("ids", "(B, T)", "batch of token ids"),
            ("E[ids]", "(B, T, D)", "one vector per token"),
        ),
        pitfalls=(
            "Initial scale matters: very large values make early training unstable (GPT uses std 0.02).",
            "Embeddings of a freshly initialised model are random; similarities only mean something after training.",
        ),
        snippets=("llm.embeddings.TokenEmbedding", "llm.embeddings.init_embedding_"),
        related=("tokenization", "positional"),
    ),
    Lesson(
        key="positional",
        title="Positional encoding",
        summary="Inject word order, which attention cannot see by itself.",
        intuition=(
            "Self-attention treats its input as a *set*: shuffle the tokens and, apart from the shuffle, the "
            "result is the same. To recover order we add a **position vector** to every token embedding.\n\n"
            "The sinusoidal version uses sine and cosine waves of geometrically spaced wavelengths. Fast waves "
            "separate neighbouring positions, slow waves separate distant ones - like the hands of a clock. "
            "Because a shift by a fixed offset is a linear map of sin/cos pairs, relative positions are easy "
            "to learn. GPT-2 instead learns a position table."
        ),
        steps=(
            "For each position pos and each pair of dimensions (2i, 2i+1) compute an angle pos / 10000^(2i/d).",
            "Put sin(angle) in the even dimension and cos(angle) in the odd one.",
            "Add the resulting (T, d_model) matrix to the token embeddings.",
        ),
        formulas=(
            Formula(r"PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)",
                    "Even dimensions carry a sine wave whose wavelength grows with i."),
            Formula(r"PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)",
                    "Odd dimensions carry the matching cosine."),
            Formula(r"\mathbf{h}_t = \mathbf{x}_t + PE_t",
                    "Position information is simply added to the token embedding."),
        ),
        shapes=(
            ("PE", "(T_max, D)", "fixed table (sinusoidal) or learnable (learned)"),
            ("x + PE[:T]", "(B, T, D)", "broadcast over the batch"),
        ),
        pitfalls=(
            "A learned table cannot represent positions beyond max_seq_len; sinusoidal values exist but the model has not been trained on them.",
            "Without any positional signal the model cannot tell 'dog bites man' from 'man bites dog'.",
        ),
        snippets=("llm.positional.sinusoidal_table", "llm.positional.SinusoidalPositionalEncoding"),
        related=("embeddings", "attention"),
    ),
    Lesson(
        key="attention",
        title="Self-attention",
        summary="Every token takes a weighted average of the other tokens' values.",
        intuition=(
            "Each token produces three vectors: a **query** (what am I looking for?), a **key** (what do I "
            "contain?) and a **value** (what do I hand over if selected?). The dot product of a query with all "
            "keys scores relevance; softmax turns the scores into weights; the output is the weighted sum of "
            "values.\n\n"
            "A **causal mask** sets scores of future positions to minus infinity, so token *t* only reads "
            "tokens up to *t*. That is what lets a decoder-only model train on next-token prediction without "
            "peeking at the answer."
        ),
        steps=(
            "Project the input into queries Q, keys K and values V with three learned matrices.",
            "Compute all pairwise scores Q K^T.",
            "Divide by sqrt(d_k) to keep the scores in a range where softmax has useful gradients.",
            "Mask future positions with minus infinity.",
            "Apply softmax along each row: every row is now a probability distribution.",
            "Multiply the weights by V to obtain the output of every token.",
        ),
        formulas=(
            Formula(r"\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right)V",
                    "M is 0 for allowed positions and minus infinity for masked ones."),
            Formula(r"\mathrm{softmax}(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}}",
                    "Non-negative weights that sum to 1."),
            Formula(r"\mathrm{Var}\!\left(q\cdot k\right) \approx d_k",
                    "Unscaled dot products grow with d_k; dividing by sqrt(d_k) restores unit variance."),
        ),
        shapes=(
            ("Q, K, V", "(B, T, d_k)", "per-head queries, keys and values"),
            ("QK^T", "(B, T, T)", "score of every query against every key"),
            ("weights", "(B, T, T)", "rows sum to 1; upper triangle is 0 when causal"),
            ("output", "(B, T, d_k)", "weights @ V"),
        ),
        pitfalls=(
            "Forgetting the mask in a language model leaks the future and gives deceptively low loss.",
            "Attention cost grows with T squared, which is why context windows are limited.",
            "Attention weights show where the model looks, not necessarily why it answers as it does.",
        ),
        snippets=(
            "llm.attention.causal_mask",
            "llm.attention.attention_scores",
            "llm.attention.scaled_dot_product_attention",
        ),
        related=("multihead", "positional"),
    ),
    Lesson(
        key="multihead",
        title="Multi-head attention",
        summary="Run several attentions in parallel on slices of the vector.",
        intuition=(
            "One attention pattern per layer is limiting: a token may need to look at the previous word *and* "
            "at a matching bracket *and* at the subject of the sentence. **Multi-head attention** splits the "
            "d_model-wide vectors into `h` slices of width d_k = d_model / h and runs an independent attention "
            "on each slice. The head outputs are concatenated and mixed by an output projection.\n\n"
            "Total compute is about the same as one big head, but the model can learn several different "
            "relations at once."
        ),
        steps=(
            "Project x into Q, K, V of width d_model.",
            "Reshape each from (B, T, D) to (B, H, T, d_k): carve the width into H heads.",
            "Run scaled dot-product attention independently in every head.",
            "Reshape back to (B, T, D): concatenate the heads.",
            "Apply the output projection W_O so the heads can exchange information.",
        ),
        formulas=(
            Formula(r"\mathrm{head}_i = \mathrm{Attention}(XW_i^{Q},\,XW_i^{K},\,XW_i^{V})",
                    "Each head has its own projections."),
            Formula(r"\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^{O}",
                    "Concatenate the heads and mix them."),
            Formula(r"d_k = d_{\text{model}} / h",
                    "d_model must be divisible by the number of heads."),
        ),
        shapes=(
            ("x", "(B, T, D)", "input"),
            ("Q, K, V", "(B, H, T, d_k)", "after splitting into heads"),
            ("weights", "(B, H, T, T)", "one attention map per head"),
            ("output", "(B, T, D)", "after merging heads and W_O"),
        ),
        pitfalls=(
            "More heads means narrower heads (d_k shrinks); too many heads can make each one too weak.",
            "Some heads end up redundant; that is normal.",
        ),
        snippets=(
            "llm.attention.MultiHeadAttention._split_heads",
            "llm.attention.MultiHeadAttention._merge_heads",
            "llm.attention.MultiHeadAttention.forward",
        ),
        related=("attention", "layernorm"),
    ),
    Lesson(
        key="layernorm",
        title="Layer normalization",
        summary="Rescale each token's features to zero mean and unit variance.",
        intuition=(
            "Activations can drift to very large or very small values as they pass through many layers, which "
            "makes optimisation hard. **LayerNorm** normalises each token's feature vector on its own: subtract "
            "its mean, divide by its standard deviation, then apply a learnable scale (gamma) and shift (beta) "
            "so the network can undo the normalisation where that helps.\n\n"
            "Unlike BatchNorm it does not depend on other examples in the batch, so it behaves the same in "
            "training and inference and for any batch size."
        ),
        steps=(
            "Compute the mean of the d_model features of one token.",
            "Compute their variance.",
            "Subtract the mean and divide by sqrt(variance + eps).",
            "Multiply by gamma and add beta (both learnable, per feature).",
        ),
        formulas=(
            Formula(r"\mu = \frac{1}{d}\sum_{j=1}^{d} x_j,\qquad \sigma^{2} = \frac{1}{d}\sum_{j=1}^{d}(x_j-\mu)^{2}",
                    "Statistics over the feature axis of one token."),
            Formula(r"\hat{x}_j = \frac{x_j-\mu}{\sqrt{\sigma^{2}+\epsilon}}",
                    "Normalised features: mean 0, variance 1. Epsilon avoids dividing by zero."),
            Formula(r"y_j = \gamma_j\,\hat{x}_j + \beta_j",
                    "Learnable per-feature scale and shift."),
        ),
        shapes=(
            ("x", "(B, T, D)", "input"),
            ("mean, var", "(B, T, 1)", "one statistic per token"),
            ("gamma, beta", "(D,)", "learnable parameters"),
        ),
        pitfalls=(
            "A constant vector has variance 0; epsilon keeps the result finite (it becomes all zeros).",
            "Pre-norm (LN before each sub-layer, used here) trains more stably than the original post-norm.",
        ),
        snippets=("llm.layers.LayerNorm",),
        related=("multihead", "ffn", "residual"),
    ),
    Lesson(
        key="ffn",
        title="Feed-forward network",
        summary="Transform every token independently with a small two-layer MLP.",
        intuition=(
            "Attention moves information *between* tokens; the **feed-forward network** processes each token "
            "*on its own*: expand the vector to 4x the width, apply a non-linearity, project back. This is "
            "where most parameters live and where much factual knowledge is thought to be stored.\n\n"
            "Without the non-linearity two linear layers collapse into one, so GELU (a smooth ReLU) is "
            "essential."
        ),
        steps=(
            "Linear layer: d_model -> 4 * d_model.",
            "Apply the activation (GELU) element-wise.",
            "Linear layer: 4 * d_model -> d_model.",
            "Apply dropout (training only).",
        ),
        formulas=(
            Formula(r"\mathrm{FFN}(x) = W_2\,\phi(W_1 x + b_1) + b_2",
                    "Two linear maps with a non-linearity phi in between."),
            Formula(r"\mathrm{GELU}(x) \approx 0.5\,x\left(1+\tanh\!\Big(\sqrt{2/\pi}\,(x+0.044715\,x^{3})\Big)\right)",
                    "Smooth alternative to ReLU: small negative inputs are damped, not cut."),
        ),
        shapes=(
            ("x", "(B, T, D)", "input"),
            ("hidden", "(B, T, 4D)", "expanded representation"),
            ("output", "(B, T, D)", "same shape as the input"),
        ),
        pitfalls=("The FFN never mixes tokens; if you change one token, only that token's output changes.",),
        snippets=("llm.layers.FeedForward", "llm.layers.gelu"),
        related=("layernorm", "residual"),
    ),
    Lesson(
        key="residual",
        title="Residual connections and the transformer block",
        summary="Each sub-layer adds a correction to a shared 'residual stream'.",
        intuition=(
            "A block does not replace its input; it **adds** to it: `x = x + sublayer(norm(x))`. The original "
            "signal travels unchanged along the 'residual stream', and every layer only writes a small update. "
            "Gradients can flow straight back through the additions, which is what makes stacks of dozens of "
            "layers trainable.\n\n"
            "A decoder block has two sub-layers: attention (communication between tokens) then the feed-forward "
            "network (computation per token)."
        ),
        steps=(
            "Normalise the input (LayerNorm 1).",
            "Run multi-head attention and add the result to the input.",
            "Normalise again (LayerNorm 2).",
            "Run the feed-forward network and add the result.",
        ),
        formulas=(
            Formula(r"x' = x + \mathrm{MHA}(\mathrm{LN}(x))", "Attention sub-layer with residual."),
            Formula(r"y = x' + \mathrm{FFN}(\mathrm{LN}(x'))", "Feed-forward sub-layer with residual."),
            Formula(r"\frac{\partial y}{\partial x} = I + \frac{\partial\, \mathrm{sublayer}}{\partial x}",
                    "The identity term gives gradients a direct path to earlier layers."),
        ),
        shapes=(("x, x', y", "(B, T, D)", "shape never changes inside the residual stream"),),
        pitfalls=("If a sub-layer outputs zeros the block is the identity - a good sanity check when debugging.",),
        snippets=("llm.transformer.TransformerBlock",),
        related=("layernorm", "ffn", "model"),
    ),
    Lesson(
        key="model",
        title="The full GPT model",
        summary="Embeddings, N blocks, a final norm and a vocabulary-sized output layer.",
        intuition=(
            "Putting it together: ids become vectors, positions are added, N transformer blocks refine the "
            "representation, a final LayerNorm tidies it up, and a linear head produces one **logit** per "
            "vocabulary token at every position. Softmax of the logits at position t is the model's belief "
            "about token t+1.\n\n"
            "**Weight tying** reuses the embedding matrix as the output head, saving V x D parameters and "
            "tying 'what a token means' to 'how likely it is to be output'."
        ),
        steps=(
            "Embed the token ids and add positional information.",
            "Pass the result through N transformer blocks.",
            "Apply the final LayerNorm.",
            "Project to vocabulary size to obtain logits.",
            "Compare logits with the shifted targets using cross-entropy (training) or sample (inference).",
        ),
        formulas=(
            Formula(r"P(x_{t+1}\mid x_{\le t}) = \mathrm{softmax}(W_{\text{head}}\, h_t^{(N)})",
                    "The final hidden state at position t predicts the next token."),
            Formula(r"P(x_1,\dots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})",
                    "A language model factorises the probability of a text into next-token predictions."),
        ),
        shapes=(
            ("ids", "(B, T)", "input"),
            ("hidden", "(B, T, D)", "residual stream"),
            ("logits", "(B, T, V)", "one score per vocabulary token"),
        ),
        pitfalls=("Context length is fixed by max_seq_len; longer prompts are cropped to the last tokens.",),
        snippets=("llm.transformer.GPT.forward", "llm.transformer.GPTConfig"),
        related=("residual", "loss"),
    ),
    Lesson(
        key="loss",
        title="Loss: cross-entropy",
        summary="Measure how surprised the model is by the true next token.",
        intuition=(
            "For every position the model outputs a probability for each possible next token. The loss is the "
            "negative log of the probability given to the **correct** token: confident and right -> near 0, "
            "confident and wrong -> large.\n\n"
            "A model guessing uniformly over V tokens scores ln(V), the natural starting point of training. "
            "**Perplexity** = exp(loss) is the effective number of tokens the model is choosing between."
        ),
        steps=(
            "Compute log-softmax of the logits (numerically stable via logsumexp).",
            "Pick the log-probability of the correct token at every position.",
            "Negate and average over all positions.",
        ),
        formulas=(
            Formula(r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta\!\left(y_i \mid x_{<i}\right)",
                    "Average negative log-likelihood of the true tokens."),
            Formula(r"\log \mathrm{softmax}(z)_j = z_j - \log\sum_k e^{z_k}",
                    "Computed with the log-sum-exp trick to avoid overflow."),
            Formula(r"\mathrm{perplexity} = e^{\mathcal{L}}", "Effective branching factor."),
        ),
        shapes=(("logits", "(N, V)", "N = B*T positions"), ("targets", "(N,)", "correct token ids")),
        pitfalls=(
            "Training loss falling while validation loss rises means overfitting.",
            "Loss values of different tokenizers are not directly comparable (different vocabularies).",
        ),
        snippets=("llm.transformer.cross_entropy",),
        related=("model", "optimizer"),
    ),
    Lesson(
        key="optimizer",
        title="Backpropagation and optimizers",
        summary="Use gradients to change the weights: SGD with momentum and Adam.",
        intuition=(
            "**Backpropagation** applies the chain rule through the whole network and returns the gradient of "
            "the loss with respect to every parameter. An **optimizer** turns that gradient into an update.\n\n"
            "* **SGD + momentum** keeps a running velocity so steps accelerate along consistent directions.\n"
            "* **Adam** also tracks the average squared gradient and divides by its root, so each parameter gets "
            "its own step size. It is the default for transformers. **AdamW** applies weight decay directly "
            "to the weights instead of through the gradient.\n\n"
            "Gradient clipping rescales the whole gradient when its norm is too large, protecting against spikes."
        ),
        steps=(
            "Forward pass: compute the loss.",
            "Backward pass: autograd fills p.grad for every parameter.",
            "Clip the global gradient norm.",
            "Optimizer step: update every parameter.",
            "Zero the gradients and repeat with the next batch.",
        ),
        formulas=(
            Formula(r"v_t = \mu\, v_{t-1} + g_t,\qquad \theta_t = \theta_{t-1} - \eta\, v_t",
                    "SGD with momentum mu and learning rate eta."),
            Formula(r"m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t,\qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^{2}",
                    "Adam: moving averages of the gradient and of its square."),
            Formula(r"\hat m_t = \frac{m_t}{1-\beta_1^{t}},\quad \hat v_t = \frac{v_t}{1-\beta_2^{t}},\quad "
                    r"\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}",
                    "Bias correction (the averages start at zero) and the update."),
            Formula(r"g \leftarrow g \cdot \min\!\left(1,\ \frac{c}{\lVert g\rVert_2}\right)",
                    "Global-norm gradient clipping with threshold c."),
        ),
        shapes=(("m, v", "same as parameter", "optimizer state: two extra copies of the model for Adam"),),
        pitfalls=(
            "Adam stores two extra tensors per parameter, so it uses about 3x the memory of the weights alone.",
            "Too large a learning rate makes the loss jump or become NaN; too small makes training crawl.",
        ),
        snippets=("training.optimizers.Adam.step", "training.optimizers.SGD.step", "training.trainer.clip_grad_norm"),
        related=("loss", "schedule"),
    ),
    Lesson(
        key="schedule",
        title="Learning-rate scheduling",
        summary="Warm up, then decay: big steps early, careful steps late.",
        intuition=(
            "A single constant learning rate is a compromise. At the start the weights are random and Adam's "
            "statistics are unreliable, so we **warm up** linearly from almost zero. Afterwards a **cosine "
            "decay** lowers the rate smoothly to a small floor so the model can settle into a good minimum."
        ),
        steps=(
            "For the first warmup steps raise the rate linearly up to the base value.",
            "Then follow half a cosine wave down to min_lr.",
            "Look up the rate before every optimizer step and write it into the optimizer.",
        ),
        formulas=(
            Formula(r"\eta_t = \eta_{\max}\,\frac{t+1}{T_{w}}\quad (t < T_w)",
                    "Linear warm-up over T_w steps."),
            Formula(r"\eta_t = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\!\big(\pi\,\tfrac{t-T_w}{T-T_w}\big)\right)",
                    "Cosine decay from the maximum to the minimum rate."),
        ),
        shapes=(("lr(t)", "scalar", "a pure function of the step number"),),
        pitfalls=("With very short runs a long warm-up wastes most of the training budget.",),
        snippets=("training.schedulers.warmup_cosine_lr",),
        related=("optimizer",),
    ),
    Lesson(
        key="sampling",
        title="Generating text",
        summary="Turn next-token probabilities into text with temperature, top-k and top-p.",
        intuition=(
            "To generate, repeat: feed the text so far, read the probabilities for the next token, pick one, "
            "append it. *How* you pick controls style:\n\n"
            "* **Temperature** rescales logits: below 1 sharper and safer, above 1 flatter and more surprising; "
            "0 means always take the most likely token (greedy).\n"
            "* **Top-k** keeps only the k most likely tokens.\n"
            "* **Top-p (nucleus)** keeps the smallest set of tokens whose probabilities add up to p."
        ),
        steps=(
            "Crop the prompt to the context window and run the model.",
            "Take the logits of the last position.",
            "Apply temperature, then top-k and/or top-p filtering.",
            "Sample one token from the resulting distribution.",
            "Append it and repeat.",
        ),
        formulas=(
            Formula(r"p_j = \frac{e^{z_j/\tau}}{\sum_k e^{z_k/\tau}}",
                    "Softmax with temperature tau."),
            Formula(r"\text{keep smallest } S \ \text{with}\ \sum_{j\in S} p_j \ge p",
                    "Nucleus (top-p) sampling."),
        ),
        shapes=(("logits[:, -1]", "(B, V)", "scores for the next token only"),),
        pitfalls=(
            "A small model trained on little text will repeat itself or produce near-gibberish; that is expected.",
            "Greedy decoding is deterministic and often repetitive.",
        ),
        snippets=("llm.transformer.next_token_distribution", "llm.transformer.GPT.generate"),
        related=("model",),
    ),
)

LESSONS: dict[str, Lesson] = {lesson.key: lesson for lesson in _LESSONS}


def get_lesson(key: str) -> Lesson:
    """Return the lesson with this key (``KeyError`` listing valid keys otherwise)."""
    try:
        return LESSONS[key]
    except KeyError:
        raise KeyError(f"unknown lesson {key!r}; available: {sorted(LESSONS)}") from None


def all_lessons() -> list[Lesson]:
    """Lessons in teaching order."""
    return list(_LESSONS)
