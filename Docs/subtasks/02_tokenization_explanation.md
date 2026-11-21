# Tokenization

## Abstract

A neural network consumes numbers, not text. Tokenization is the first, and often underestimated, modelling decision: it fixes the *alphabet* in which every later sentence is written. This sub-task implements two tokenizers from scratch, a character tokenizer and a byte-pair-encoding (BPE) tokenizer, and measures what the choice costs and buys on the bundled corpus of {{fact:corpus_chars}} characters.

## The problem

A language model defines a probability distribution over sequences. To make that precise we need a finite set $\mathcal{V}$ of symbols, the *vocabulary*, and a deterministic map from text to a sequence of symbols. Writing the sequence as $x = (x_1, \dots, x_T)$ with $x_t \in \mathcal{V}$, an autoregressive model factorises its probability with the chain rule:

$$P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1})$$

The tokenizer chooses what a "symbol" is. The choice trades two quantities against each other:

- the **vocabulary size** $|\mathcal{V}|$, which sets the size of the embedding table and of the output layer;
- the **sequence length** $T$ for a fixed amount of text, which sets how much context the model must carry and, for attention, the quadratic cost $O(T^2)$.

A tokenizer is a point on that trade-off curve, and the compression ratio $r = \text{characters} / \text{tokens}$ measures where.

## Character-level tokenization

The simplest vocabulary is the set of distinct characters in the corpus. For the bundled text it has **{{fact:char_vocab}}** entries: the distinct characters plus a reserved `<unk>` symbol with id 0 that stands for any character not seen during training.

Properties:

| Property | Character tokenizer |
|---|---|
| Vocabulary | {{fact:char_vocab}} symbols |
| Tokens for the corpus | {{fact:corpus_chars}} (one per character) |
| Compression ratio | 1.00 |
| Out-of-vocabulary | only unseen characters, mapped to `<unk>` |
| Needs training | no learning, only a scan of the characters |

Characters never fail on a new word, but they are a very fine grain. The model must learn spelling *and* grammar *and* meaning from the same stream, and the context window of $T$ tokens covers only $T$ characters of text. A uniform guesser over this vocabulary has a loss of $\ln |\mathcal{V}| = \ln {{fact:char_vocab}} = {{fact:ln_vocab}}$ nats per token, which is the reference point against which training progress is judged.

## Byte-pair encoding

Byte-pair encoding was introduced as a data-compression algorithm (Gage, 1994) and adapted to neural machine translation by Sennrich, Haddow and Birch (2016), who used it to build open vocabularies of sub-word units. GPT-2 (Radford et al., 2019) applied it at the byte level. The idea is to let the *data* decide which strings deserve their own symbol.

### Training

1. **Pre-tokenize** the text into word-like chunks. This implementation uses the pattern `\s?\w+|\s?[^\w\s]+|\s+`, a simplified form of the GPT-2 pattern: a word with its leading space, a run of punctuation, or a run of whitespace. The chunks concatenate back to the original text, so the step loses nothing. Because merges never cross chunk boundaries, a space stays attached to the following word.
2. **Initialise** every chunk as a sequence of single characters, and count how often each distinct chunk occurs.
3. **Count pairs.** For every adjacent pair of symbols $(a, b)$ compute its frequency, weighted by the frequency of the chunk it appears in:

$$\operatorname{count}(a, b) = \sum_{w} f(w)\cdot \#\{\text{adjacent occurrences of } (a,b) \text{ in } w\}$$

4. **Merge** the most frequent pair into a new symbol $ab$, add it to the vocabulary and rewrite all chunks. Ties are broken by choosing the lexicographically smallest pair, which makes training deterministic.
5. **Repeat** for a fixed number of merges $M$. The final vocabulary has $|\mathcal{V}| = |\text{characters}| + 1 + M$ entries (the extra one is `<unk>`).

On the bundled corpus the first merge is {{fact:bpe_first_merge}}. The very first merges join frequent letter pairs, then frequent word beginnings, and later whole words: the merge list is a compact record of which strings are frequent in *this* text.

### Encoding

To encode new text, split it into chunks, start each chunk from characters and repeatedly apply the learned merge with the **lowest rank** (the earliest learned) until none applies. Earlier merges have priority, which reproduces the order in which they were learned and makes the result deterministic. Decoding is a lookup followed by concatenation.

### Effect of the number of merges

| Merges $M$ | Vocabulary | Tokens for the corpus | Compression $r$ |
|---|---|---|---|
| 0 (characters) | {{fact:char_vocab}} | {{fact:corpus_chars}} | 1.00 |
| 100 | {{fact:bpe100_vocab}} | {{fact:bpe100_tokens}} | {{fact:bpe100_ratio}} |
| 200 | {{fact:bpe200_vocab}} | {{fact:bpe200_tokens}} | {{fact:bpe200_ratio}} |

Doubling the merges from 100 to 200 adds 100 vocabulary entries but removes far fewer tokens than the first 100 merges did ({{fact:bpe100_tokens}} tokens at 100 merges, {{fact:bpe200_tokens}} at 200). This is the *diminishing return* typical of BPE: early merges capture very frequent strings, later ones capture ever rarer ones.

!!! note "Why compression matters"
    Fewer tokens for the same text means more text fits into a fixed context window and each attention step covers more meaning. The price is a larger embedding table, and rarer tokens that receive fewer training examples each.

## Information-theoretic view

A tokenizer is a reversible code. The loss a model reaches is measured **per token**, so losses of models with different tokenizers are *not directly comparable*: a BPE token carries more information than a character, so its per-token loss is naturally higher. A fair comparison converts to bits per character,

$$\text{bpc} = \frac{\text{loss in nats per token} \times (\text{tokens}/\text{characters})}{\ln 2}.$$

In this project, loss curves are therefore only compared *within* one tokenizer choice.

## Design decisions in this implementation

- **A reserved id 0.** `<unk>` is always id 0, so a model can be created before it has seen the data and the id never depends on the corpus.
- **Deterministic vocabulary.** Characters are sorted, merge ties are broken alphabetically, so the same text always yields the same tokenizer.
- **Lossless pre-tokenization.** `"".join(pretokenize(text)) == text` for every string.
- **A recorded trace.** Each merge is kept as a `MergeStep` (pair, count, new token, runners-up) so the interface can replay the algorithm instead of just showing its result.
- **Memoised encoding.** The result for each distinct chunk is cached, because natural text repeats chunks heavily.

## Limitations

1. **Not byte-level.** A character outside the training text becomes `<unk>`; GPT-2's byte-level BPE avoids this by starting from the 256 byte values. Decoding an `<unk>` id returns U+FFFD, the Unicode replacement character, so the damage is visible rather than silent.
2. **Frequency, not meaning.** BPE merges are driven by counts only, so sub-word boundaries do not always match morphemes.
3. **Tiny corpus.** With {{fact:corpus_chars}} characters, even 200 merges start to memorise individual words rather than discover general sub-words.
4. **Cost.** The training loop recounts every pair at every merge, so its cost grows with the number of merges times the number of distinct words. It is written for clarity, not speed, and the interface therefore caps the number of merges.

## Where to go next

Tokens are still just integers. The next step, [embeddings and positional encoding](03_embeddings_positional_explanation.html), turns each id into a vector the network can compute with. The [implementation guide](02_tokenization_implementation_guide.html) shows the code and data structures.

## References

1. Gage, P. (1994). *A New Algorithm for Data Compression.* The C Users Journal 12(2).
2. Sennrich, R., Haddow, B., Birch, A. (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI technical report.
4. Kudo, T., Richardson, J. (2018). *SentencePiece: A simple and language independent subword tokenizer and detokenizer.* EMNLP (system demonstrations).
