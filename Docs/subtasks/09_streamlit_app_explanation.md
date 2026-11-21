# The Streamlit application: interaction, state and architecture

## Abstract

The previous sub-tasks produce a model, a training loop, figures and lessons. This sub-task puts them in front of a learner. The scientific questions here belong to human-computer interaction and software architecture rather than to machine learning: how should an interface for *exploring* a model behave, where should state live, how can long computations be shown while they run, and how can a user interface be tested without a person clicking through it? The document explains the execution model of Streamlit, the design of the shared session state, the separation of logic from presentation, the way live training is streamed, and the usability principles behind the ten pages.

## The Streamlit execution model

Most graphical toolkits are **event driven**: a callback runs when a button is clicked and mutates widgets. Streamlit uses a different model, closer to a **spreadsheet** or to **functional reactive programming** (Elliott and Hudak, 1997). Whenever the user changes a widget, the script is executed again **from top to bottom**, and every widget call returns its current value. The page is the output of the script for the present input.

This has three consequences that shaped the code.

1. **Scripts are plain.** A page is a sequence of "read widget, compute, draw" steps with no callback wiring, which is easy to read for a learner.
2. **Nothing survives a rerun unless it is stored.** Local variables are recreated on every run. Anything that must persist, such as the model, the optimiser and the training history, has to be placed in `st.session_state`, a per-browser-session dictionary.
3. **Expensive work must be cached or avoided.** Retraining a tokenizer on every slider move would make the interface unusable. The application therefore memoises tokenizer training by a key of `(kind, corpus text, merges)` and rebuilds the model only when the user presses **Apply**.

!!! note "Sessions are isolated"
    Each browser tab has its own session state, so two learners opening the same server do not share models. The state defaults are deep-copied for every session so that a mutable default such as the history list is never shared.

## State design

### One dictionary, a small set of keys

The state holds only what must outlive a rerun:

| Key | Meaning |
|---|---|
| `settings` | the immutable `ModelSettings` chosen in the sidebar |
| `model`, `tokenizer` | the object every page uses |
| `history` | the list of per-step metric dictionaries |
| `train_cfg` | the training configuration of the last run |
| `optimizer` | the live optimiser, kept so a run can be resumed in the same session |
| `model_source` | `none`, `fresh`, `trained` or `checkpoint`, shown in the sidebar so the learner always knows which weights they are looking at |
| `stop_requested`, `training` | flags for cooperative cancellation and for warning the user |
| `last_error` | the last readable error message |

### Immutable settings, explicit transitions

The sidebar produces a **frozen dataclass**. Changing the model means constructing a new settings object, validating it, and replacing the model, tokenizer, history and optimiser **together** in one `update` call. There is therefore no moment in which a new model exists beside the old training history, which would produce charts that belong to a different network. If validation fails, the old model is kept and the reason is shown. This is the standard way to keep a state machine consistent: *make illegal states unrepresentable* and perform transitions atomically.

The states of `model_source` form a small machine:

| From | Event | To |
|---|---|---|
| none | first page load | fresh |
| fresh, trained, checkpoint | Apply in the sidebar | fresh |
| fresh, checkpoint | a training run produces at least one step | trained |
| any | load a checkpoint | checkpoint |

### Why the logic does not import Streamlit

The state module treats `state` as any mutable mapping. In the application it is `st.session_state`; in tests it is an ordinary `dict`. This is an application of the **dependency inversion principle** (Martin, 2003): high-level logic depends on an abstraction (a mapping), not on the framework. The payoff is that the whole lifecycle can be unit tested in milliseconds without a browser.

## Layers and separation of concerns

The application follows a *presentation, application logic, domain* layering, with the dependencies pointing downwards only:

| Layer | Packages | Knows about Streamlit |
|---|---|---|
| Presentation | `app.py`, `pages/`, `ui/sidebar.py`, `ui/components.py` | yes |
| Application logic | `ui/state.py`, `ui/lab.py`, `ui/demo.py` | no |
| Content and drawing | `edu/`, `viz/` | no |
| Domain | `llm/`, `training/`, `config.py` | no |

A page is intentionally thin: it reads widgets, calls a helper from the logic layer, and passes the result to a figure builder. The helpers in `ui/demo.py` compute what a page needs (for example "run one untrained attention layer on this text") and return plain dictionaries and tensors. This is the *humble object* pattern (Meszaros, 2007): the part that is hard to test, the framework glue, is made as small as possible.

## Navigation and information architecture

Ten pages are registered in one table and presented in the order in which the model is built: Home, tokenization, embeddings, positional encoding, attention, multi-head attention, LayerNorm and feed-forward, the full transformer, the training lab and inference. The order is the order of the data flow, so the navigation itself is a summary of the architecture.

Each learning page has the same skeleton, which follows the *predictable structure* heuristic of Nielsen (1994):

1. a title and a one-line summary;
2. the Explain, Math and Code tabs (sub-task 8);
3. a divider and an interactive section with widgets;
4. figures that respond to the widgets (sub-task 7).

The **sidebar** is shared by all pages and holds the model configuration. This follows the principle of *visibility of system status*: the number of parameters, the vocabulary size, the device and whether the weights are untrained, trained or loaded are always visible.

## Interaction design principles applied

| Principle | How it appears in the application |
|---|---|
| Immediate feedback | changing a slider redraws the figure in the same run |
| Error prevention | a form validates before applying; the training page disables Start when the window is longer than the context |
| Helpful error messages | messages say what is wrong and what to do (for example when `d_model` is not divisible by the head count they explain why each head needs `d_model / heads` dimensions) |
| User control | a Stop button, a Resume button, and nothing that destroys work without saying so (applying settings tells the user that earlier training was discarded) |
| Recognition rather than recall | the settings show their current values and a short help text |
| Graceful empty states | before data exists a consistent message tells the learner what to press |

!!! note "Form versus live widgets"
    The architecture settings are in a **form**: nothing changes until the learner presses a button, because re-initialising a model is expensive and destroys training progress. The widgets on the learning pages are not in a form, because a change there only recomputes a small demonstration.

## Streaming a long computation

Training takes seconds to minutes and the learner should *see* it happen. The training loop is a **generator** that yields one metrics dictionary per step. The page iterates over it, and every few steps clears and redraws the progress area. This is a form of **cooperative multitasking**: the generator produces a value, the consumer decides what to do with it, and control returns.

Three details matter.

1. **Cooperative cancellation.** The Stop button cannot interrupt a Python loop from outside. It sets a flag in the state, and the generator checks the flag before each step. This is the safe approach because the model is never left half-updated.
2. **Redraw cadence.** Redrawing a chart on every step would cost more than the step itself. The page redraws every $N$ steps (default 5) and once at the end.
3. **Unique widget keys.** Streamlit requires a unique key for each chart. Because the same chart is redrawn many times in one script run, the page appends a frame counter to its key. Without it Streamlit raises a duplicate-element error on the second redraw.

### What a resume does and does not do

Resuming reuses the optimiser object that is alive in the session, which keeps Adam's moment estimates. It does not continue the learning-rate schedule: the schedule is indexed from zero again, so the rate returns to the warm-up value. The extra steps are added to the history, so a 30-step run followed by a resume of 30 steps has 60 recorded steps. The optimiser state is written into checkpoints but is not restored from them, so after loading a file the Resume button is unavailable. The interface is honest about this by enabling Resume only when an optimiser exists in the session.

## Testing a user interface

A common objection to interface code is that it can only be tested by hand. Streamlit provides a headless runner, **`AppTest`**, which executes a script with a simulated session and exposes its elements (sliders, buttons, charts, errors, session state). The test suite uses it at three levels:

1. **Pure logic** (settings validation, state transitions, data preparation, the demos) with ordinary unit tests on plain dictionaries.
2. **Page tests**: each of the ten pages is executed, its widgets are driven, and the test asserts that no exception or error banner appears and that the expected state results (for example that training reduces the loss and fills the history).
3. **Whole-application tests**: `app.py` runs with the sidebar, an Apply click rebuilds the model, and invalid settings leave the old model untouched.

Two fixtures protect the user's files: the output and input folders are redirected to a temporary directory for every test, so tests never write into `output/` and never read private files from `input/` into the corpus.

## Reproducibility in an interactive tool

The seed in the settings controls the initial weights, the batch order and the sampling. The same settings therefore give the same loss curve, which lets a learner compare two hyper-parameter choices fairly, and lets this documentation quote measured numbers. On a different device the floating-point sums can differ in the last bits; for the reference run the losses were identical on the CPU and on Apple's MPS backend to the digits reported.

## Limitations

1. **Single user, local.** The application has no authentication and no concurrency control. It binds to `localhost` by default and is meant for one learner.
2. **Training blocks the session.** The run happens inside the script execution; the page cannot be used for something else until it finishes or is stopped.
3. **No persistence of the session.** Closing the tab loses the in-memory model unless a checkpoint was saved.
4. **Scale.** The settings limits (for example 12 layers, 512 wide) keep every page interactive; larger models would call for background workers.

## Where to go next

The scripts that start and stop the application, and the documents and checks around it, are covered in the [last sub-task](10_scripts_docs_compliance_explanation.html). The [implementation guide](09_streamlit_app_implementation_guide.html) lists the page registry, the state keys, the settings fields and the tests.

## References

1. Elliott, C., Hudak, P. (1997). *Functional Reactive Animation.* ICFP.
2. Martin, R. C. (2003). *Agile Software Development: Principles, Patterns, and Practices.* Prentice Hall.
3. Meszaros, G. (2007). *xUnit Test Patterns: Refactoring Test Code.* Addison-Wesley.
4. Nielsen, J. (1994). *10 Usability Heuristics for User Interface Design.* Nielsen Norman Group.
5. Norman, D. A. (2013). *The Design of Everyday Things* (revised ed.). Basic Books.
6. Streamlit documentation: *App execution model* and *Session State*.
