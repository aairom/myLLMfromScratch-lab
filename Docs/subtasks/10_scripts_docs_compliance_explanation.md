# Scripts, documentation and compliance

## Abstract

A teaching project is only useful if someone else can run it, trust it and understand it a year later. This sub-task is about the practices that make that possible: a **single command** to start and stop the application, **configuration kept out of the source**, **documents generated from or checked against the code**, **tests that gate completion**, and an **automated audit** that verifies the project against its own rules. The document explains the reasoning behind each practice with reference to the research on reproducibility and software quality, and states which guarantees are mechanical (checked by a script) and which still depend on a human.

## Reproducibility as a design goal

Computational results are often hard to reproduce. Peng (2011) describes a spectrum from "publication only" to "full replication" in which code, data and environment are all available and runnable. Sandve et al. (2013) condense the practice into ten simple rules, among them: *automate every step, record the versions of the software used, avoid manual manipulation of data, and make scripts and parameters public*. The project applies these in four places.

| Rule | Application |
|---|---|
| Automate | `scripts/start.sh`, `stop.sh`, `cleanup.sh` and `build-docs.sh` each replace a multi-step manual procedure with one command |
| Record the environment | `requirements.txt` lists the libraries; the project-local `.venv` is created with `uv` and refreshed when the file changes |
| Fix the randomness | every random process takes a seed taken from the settings, so the same settings give the same numbers |
| Keep parameters outside the code | a `.env` file, documented by `.env.example`, holds ports, folders and the device |

!!! note "Reproducible is not the same as identical on every machine"
    The reference run gives the same losses on the CPU and on Apple's MPS backend to the digits quoted in this documentation. Floating-point addition is not associative, so on other hardware or library versions the last digits may differ even when the seed is the same.

## Configuration outside the code

The *Twelve-Factor App* methodology (Wiggins, 2011) states that **configuration, which varies between deployments, should be strictly separated from code** and stored in the environment. The project follows it in three ways:

1. **No literal secrets, addresses or ports in the source.** Values come from environment variables read by one module (`config.py`). A default exists for each, so a fresh clone runs without any file.
2. **A documented template.** `.env.example` lists every variable with a comment. The real `.env` is excluded from version control.
3. **Validation at load time.** An invalid port or device raises an error with a clear message instead of failing later in an obscure place. Port 5000 is refused explicitly because macOS uses it for AirDrop.

### A scan instead of a promise

Hard-coded credentials are a common cause of leaks. Meli et al. (2019) found that secrets are leaked at large scale in public repositories and that attackers can discover them quickly. A policy such as "never commit secrets" is only as strong as its enforcement, so the project includes a scanner that looks for quoted literals assigned to names like `password` or `token`, for the shapes of well-known keys, for e-mail addresses in code and, in strict mode, for `http` URLs in source files. A line that has been reviewed can be marked as an exception. A scan finds *patterns*; it cannot prove the absence of every secret.

## Lifecycle scripts

| Script | Behaviour | Property |
|---|---|---|
| `start.sh` | loads `.env` without executing it, refuses port 5000 on macOS, detaches the server, writes a process id file and a log, checks that the process survived the first second, prints the URL | idempotent: a second call reports the running process |
| `stop.sh` | sends a termination signal to the server and its children, waits up to a timeout, forces a kill only if needed, removes the process id file | graceful first, forceful last |
| `cleanup.sh` | removes `node_modules`, `.venv`, `__pycache__` and the runtime folder; `--dry-run` only lists | refuses to touch anything else |
| `build-docs.sh` | rebuilds the HTML documentation from the Markdown sources | fails with a hint when there is no virtual environment |

Two design points deserve a note. **Launching detached** means the learner's terminal is free again and the script can return a URL immediately. **Stopping gracefully** gives the process the chance to release its port and files, which a plain kill would not.

## Documentation as a product

Documentation fails when it is written once and never touched. Four mechanisms reduce that risk.

1. **Mandatory documents.** A `README.md` with diagrams and the licence, a `Quickstart.md` and an `Architecture.md`. The audit checks that they exist, and that other documents live in `Docs/` and not scattered over the root.
2. **Diagrams as text.** The architecture and workflow diagrams are written in Mermaid, so they are versioned and diffed like code.
3. **Code shown from the source.** The application and this documentation show source code read from the live modules, so a displayed listing cannot differ from the code that runs.
4. **Numbers computed, not typed.** In this sub-task documentation the figures quoted in the text, such as the number of parameters or the loss of the reference run, are filled in at build time from the running code and a recorded run. A change in the model changes the document. A handful of measured values without a programmatic source were verified by hand when the document was written.

### The documentation pipeline

The pages for the ten sub-tasks are generated from Markdown by a small, dependency-light builder. Mathematics is written in LaTeX and converted to MathML, which modern browsers render natively without a JavaScript library. Code is highlighted with Pygments. Charts are drawn as inline SVG from the recorded training run. Everything is embedded in the page, so each file works offline when opened from disk.

Because the output is plain HTML, it can be **validated mechanically**. The validator checks that every page has a language attribute, a title, a viewport declaration, exactly one top-level heading with no skipped levels, a skip link and a main landmark; that identifiers are unique and every in-page anchor exists; that no external resource, script or style reference is present; that charts carry a text alternative; that scrollable code blocks can be reached with the keyboard; and that links to the sibling pages point to files that exist. These checks implement a part of the Web Content Accessibility Guidelines (W3C, 2018), for example text alternatives, a logical heading structure and keyboard operability. They do **not** replace an evaluation with assistive technology and real users.

## Tests as a gate

Beck (2002) argued that tests written together with the code turn the fear of changing it into confidence. The project applies two rules:

- **Every source module has a test module.** A script maps each Python file to its test file by name and reports a module without one.
- **Completion requires a green run.** The gate runs the whole suite and also reports untested modules, and only both together give a successful exit code.

The suite has more than 400 tests and mixes three levels, from fast to slow: unit tests of pure functions (the maths is compared with PyTorch references, the optimisers with known update rules), tests of the application logic on plain dictionaries, and headless tests of every Streamlit page. Among them are tests chosen for their *diagnostic power*, such as the causality of the whole model and the ability to overfit a tiny batch, which fail if the wiring is wrong even when each part passes alone.

!!! note "What coverage by name does not tell you"
    The gate checks that a test file with the right name exists. It does not measure the lines or branches that the tests execute. A module could have a test file that tests very little. The review of the tests, and the diagnostic tests above, are what protect against that.

## Version control hygiene

A `.gitignore` is created or repaired by a script and its content is checked against a policy: environment files, virtual environments, caches, compiled binaries, archives, model weights, the contents of the input and output folders (kept in the repository only through placeholder files), and local agent configuration are never committed. Keeping generated and private files out of the history has a practical and a security aspect: a repository stays small and reviewable, and a file that was never committed cannot leak from it.

## The audit

The audit script composes the other checks into one pass: required documents present, a licence section in the README, documents confined to `Docs/`, the ignore policy, the secret scan, executable scripts, and a container file policy when a container file exists. It is **read-only**. Its output ends with an explicit pass or fail so that it can be used in an automated pipeline. It does not run the unit tests, which have their own gate.

Mechanical checks of this kind are valuable because they are cheap to repeat. Their limit is that they check the rules that someone thought of. The policy therefore improves by adding a check whenever a new class of mistake is found.

## Limitations

1. **Rule-based scanners have false negatives.** The secret scan is pattern based.
2. **No continuous integration is configured.** The gates are scripts that a person or a pipeline must run.
3. **Pinned versions are lower bounds.** `requirements.txt` states minimum versions, not an exact lock file, so a fresh install may pick newer libraries.
4. **Accessibility is partly verified.** Automated checks cover structure and alternatives, not colour contrast with every theme or screen-reader behaviour.
5. **English only** documentation and interface.

## Where to go next

The [implementation guide](10_scripts_docs_compliance_implementation_guide.html) shows each script, the documents generator and the audit in detail. To see how the application that these scripts start is organised, read the [Streamlit application document](09_streamlit_app_explanation.html). The [index](index.html) lists all documents.

## References

1. Peng, R. D. (2011). *Reproducible Research in Computational Science.* Science 334(6060).
2. Sandve, G. K., Nekrutenko, A., Taylor, J., Hovig, E. (2013). *Ten Simple Rules for Reproducible Computational Research.* PLoS Computational Biology 9(10).
3. Wiggins, A. (2011). *The Twelve-Factor App.* Section III, Config.
4. Meli, M., McNiece, M. R., Reaves, B. (2019). *How Bad Can It Git? Characterizing Secret Leakage in Public GitHub Repositories.* NDSS.
5. Beck, K. (2002). *Test-Driven Development: By Example.* Addison-Wesley.
6. W3C (2018). *Web Content Accessibility Guidelines (WCAG) 2.1.* W3C Recommendation.
7. Fowler, M. (2012). *TestPyramid.* martinfowler.com.
