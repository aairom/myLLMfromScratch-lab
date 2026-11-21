# Educational content as data

## Abstract

The application teaches by showing three things side by side: an intuitive explanation, the mathematics, and the code that actually runs. This sub-task is the layer that stores that material. The central design decision is that **a lesson is data, not interface code**: thirteen immutable `Lesson` records hold the text, the formulas, the tensor shapes, the common mistakes and the *names* of the source symbols to display. The code itself is never copied into the lesson; it is read from the running modules at display time. This document explains the learning-science reasoning behind that structure, how the three representations support each other, and what guarantees the design gives (and does not give).

## Learning-science background

Four findings from cognitive psychology shaped the layout of every lesson.

### Cognitive load and worked examples

Working memory can hold only a few new elements at once. Sweller (1988) argued that instruction should keep the *extraneous* load (effort spent on the format rather than the idea) small, so that the learner's capacity is spent on the idea itself. A closely related result is the **worked-example effect**: a novice learns a procedure faster from studying a solved example than from solving the same problem unaided (Sweller and Cooper, 1985). Each lesson therefore contains an ordered list of short *steps*, one sentence each, that walks through the computation like a solved example. The default lessons have between 3 and 6 steps, 57 in total.

### Dual coding and multimedia learning

Paivio (1971) proposed that verbal and visual information are processed in two partly independent channels, so a concept presented in both is remembered better. Mayer (2009) turned this into practical principles: put the picture and the words that describe it close together (*spatial contiguity*), and avoid decoration that does not carry meaning (*coherence*). The lesson tabs (Explain, Math, Code) sit directly above the interactive figure of the same page, and every figure carries a title that states what it shows.

### Three representations of one idea

A concept such as attention is more robust in memory when the learner meets it as prose, as an equation and as a program, and can move between them. The lesson layer is organised around exactly this triple:

| Representation | Field of a lesson | Question it answers |
|---|---|---|
| Intuition | `intuition`, `steps` | what is the idea, and in which order does it happen |
| Mathematics | `formulas`, `shapes` | what is the exact relation, and what dimensions do the tensors have |
| Code | `snippets` | how is it written, so that I can run and change it |

### Anticipating misconceptions

Beginners hold systematic misconceptions, and naming them is more effective than hoping they do not arise. Each lesson has a `pitfalls` list. Examples are "characters not seen while building the vocabulary become `<unk>` and cannot be recovered on decode" for tokenization and "forgetting the mask in a language model leaks the future and gives deceptively low loss" for attention. There are 24 pitfalls in total, and every lesson has at least one.

## The lesson record

A lesson is a frozen dataclass with these fields:

| Field | Type | Role |
|---|---|---|
| `key` | string | stable identifier used by pages, such as `attention` |
| `title`, `summary` | string | page title and the one-line description shown on the home page |
| `intuition` | Markdown string | the explanation in plain language |
| `steps` | tuple of strings | the worked walkthrough |
| `formulas` | tuple of `Formula` | LaTeX expression plus a plain-language reading of it |
| `shapes` | tuple of triples | tensor name, shape and meaning |
| `pitfalls` | tuple of strings | typical mistakes |
| `snippets` | tuple of dotted paths | symbols whose source is displayed |
| `related` | tuple of keys | other lessons to visit next |

The 13 lessons appear in teaching order: tokenization, embeddings, positional encoding, attention, multi-head attention, layer normalisation, feed-forward network, residual connections, the full model, the loss, the optimiser, the learning-rate schedule and sampling. They hold 34 formulas in total, every one paired with a sentence that explains it in words, because an equation without a reading is a common obstacle for newcomers.

### Why immutability

Frozen dataclasses and tuples (rather than lists) make a lesson a *value*: it can be shared between pages and sessions without a defensive copy, and a page cannot change the text that another page shows. It also makes the content easy to test, because the tests iterate over plain data.

## Code that cannot go stale

The classic failure of teaching material is *drift*: the text describes version 1 of the code while the repository holds version 3. Knuth (1984) proposed **literate programming** to keep prose and program together. This project uses a lighter variant of the same idea: the lesson stores only the **dotted path** of a symbol, for example `llm.attention.scaled_dot_product_attention`, and a small resolver finds and prints the live source.

The resolver works in two steps:

1. Import the *longest* prefix of the path that is a module (`llm.attention`, not `llm`).
2. Look up the remaining parts as attributes (`scaled_dot_product_attention`, or `GPT.generate` for a method).

The source is then obtained with Python's `inspect` module and dedented. Because the comments and the docstring are part of the source, **the comments in the code are the line-by-line explanation**: they are written once, next to the code they describe. The 13 lessons reference 26 symbols, which together are 638 lines of source.

!!! note "A guarantee and its limit"
    The design guarantees that the displayed code is the executing code. It does not guarantee that the *prose* in the intuition is still correct after a refactor. That second property is protected by tests that check the structure of the content (every lesson has every section, every path resolves, the LaTeX braces balance, related keys exist) and by a rule for the author: change the lesson text in the same commit as the behaviour it describes.

### Error handling as a teaching feature

If a path stops resolving, the page must not fail: a lesson that cannot show a snippet still has an explanation and a formula. The code tab therefore catches the lookup error, shows a warning that names the missing symbol, and continues with the next snippet. A test asks the Code tab for an extra symbol that does not exist and checks that a warning appears while the page still renders.

## Mathematics for the web

Formulas are stored as LaTeX strings and drawn with Streamlit's KaTeX support. LaTeX is a standard that learners will meet again in papers, so showing it also teaches how to read and write it. A structural test counts opening and closing braces in every formula to catch typing errors before they reach a browser.

## Presentation: three tabs

The Explain tab shows the intuition, the numbered steps and a collapsed list of pitfalls. The Math tab shows each formula with its reading, then a table of tensor shapes. The Code tab shows each snippet in an expander, the first one open. The same three tabs appear on every page that shows a lesson (ten of the thirteen lessons are shown in full; the residual, loss and schedule lessons are summarised on the home page and covered by the model and training pages), so the learner builds one interaction habit and spends no effort on navigation (Sweller's extraneous load again).

!!! note "Progressive disclosure"
    Pitfalls are collapsed and the second and later snippets are folded away. The page starts with the least material that is still complete, and the learner opens more on demand (Nielsen, 2006).

## Worked sampling example: why the lesson contains numbers

Abstract statements about temperature are hard to retain. The sampling lesson therefore pairs the formula

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

with the interactive page, and this documentation gives one concrete case. For the logits $(2, 1, 0.1)$ the probabilities are about $(0.659, 0.242, 0.099)$ at $\tau = 1$, $(0.864, 0.117, 0.019)$ at $\tau = 0.5$ and $(0.502, 0.304, 0.194)$ at $\tau = 2$. Seeing the same three numbers sharpen and flatten teaches the role of $\tau$ faster than the equation alone.

## Evaluation of the design

The content layer was checked in four ways:

1. **Completeness**, by tests: all 13 keys exist in teaching order; every lesson has a title, a summary, a long enough intuition, at least three steps, at least one formula, one shape row, one pitfall and one snippet.
2. **Resolvability**, by tests: every one of the 26 snippet paths returns non-empty source.
3. **Consistency**, by tests: related keys exist and never point to the lesson itself; LaTeX braces balance.
4. **Rendering**, by interface tests: every lesson renders its three tabs without an exception.

There is **no measurement of learning outcomes** in this project. Whether the lessons are effective for a given audience would require a study with real learners (pre and post tests), which is outside the scope.

## Limitations

1. **Prose can drift.** See above: only the code is guaranteed current.
2. **English only.** The text lives in Python tuples; supporting another language would call for a message catalogue.
3. **No prerequisite graph.** `related` is a list of suggestions, not an enforced order.
4. **Static text.** Lessons do not adapt to what the learner has already done in the session.

## Where to go next

The charts and diagrams that accompany each lesson are described in the [visualisation document](07_visualization_explanation.html). The [Streamlit application](09_streamlit_app_explanation.html) shows how the three tabs are placed on each page. The [implementation guide](08_educational_content_implementation_guide.html) lists the record fields, the resolver and the tests.

## References

1. Sweller, J. (1988). *Cognitive load during problem solving: effects on learning.* Cognitive Science 12(2).
2. Sweller, J., Cooper, G. A. (1985). *The use of worked examples as a substitute for problem solving in learning algebra.* Cognition and Instruction 2(1).
3. Paivio, A. (1971). *Imagery and Verbal Processes.* Holt, Rinehart and Winston.
4. Mayer, R. E. (2009). *Multimedia Learning* (2nd ed.). Cambridge University Press.
5. Knuth, D. E. (1984). *Literate Programming.* The Computer Journal 27(2).
6. Nielsen, J. (2006). *Progressive Disclosure.* Nielsen Norman Group.
7. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS (the source of the formulas shown in the attention lessons).
