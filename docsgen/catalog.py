"""The ten sub-tasks of the project and the file names of their deliverables.

Every sub-task has three deliverables inside ``Docs/subtasks/``:

* ``NN_<slug>_explanation.md``           scientific explanation (hand-written source)
* ``NN_<slug>_explanation.html``         the same explanation as a standalone page (generated)
* ``NN_<slug>_implementation_guide.html`` implementation guide (generated from ``guide_src/``)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SubTask:
    number: int
    slug: str
    title: str
    summary: str

    @property
    def stem(self) -> str:
        return f"{self.number:02d}_{self.slug}"

    @property
    def explanation_md(self) -> str:
        return f"{self.stem}_explanation.md"

    @property
    def explanation_html(self) -> str:
        return f"{self.stem}_explanation.html"

    @property
    def guide_src(self) -> str:
        return f"guide_src/{self.stem}_guide.md"

    @property
    def guide_html(self) -> str:
        return f"{self.stem}_implementation_guide.html"


SUBTASKS: tuple[SubTask, ...] = (
    SubTask(1, "scaffolding", "Project scaffolding and environment",
            "Reproducible environment, configuration through .env, test harness and repository hygiene."),
    SubTask(2, "tokenization", "Tokenization",
            "From text to integer ids: character vocabulary and a hand-written byte-pair encoder."),
    SubTask(3, "embeddings_positional", "Embeddings and positional encoding",
            "Lookup-table embeddings, initialisation, and sinusoidal or learned position signals."),
    SubTask(4, "attention_layers", "Attention, multi-head attention, LayerNorm and feed-forward",
            "Scaled dot-product attention with a causal mask, heads, layer normalisation and the GELU MLP."),
    SubTask(5, "gpt_model", "Full transformer (GPT) and generation",
            "Pre-norm blocks, residual stream, tied output head, cross-entropy loss and sampling."),
    SubTask(6, "training_pipeline", "Training pipeline",
            "Windowed data, backpropagation, hand-written Adam and SGD, learning-rate schedules, checkpoints."),
    SubTask(7, "visualization", "Visualization layer",
            "Plotly figure builders: PCA projections, heat maps, attention maps and live training curves."),
    SubTask(8, "educational_content", "Educational content layer",
            "Lessons as data, LaTeX formulas and code snippets read live from the running source."),
    SubTask(9, "streamlit_app", "Streamlit application and learning pages",
            "Navigation, shared session state, ten pages, live training lab and inference page."),
    SubTask(10, "scripts_docs_compliance", "Scripts, documentation and compliance",
            "One-command lifecycle scripts, project documents, secret scan, test gate and audit."),
)


def get(number: int) -> SubTask:
    """Return the sub-task with the given 1-based number."""
    for sub in SUBTASKS:
        if sub.number == number:
            return sub
    raise KeyError(f"no sub-task {number}")


def neighbours(number: int) -> tuple[Optional[SubTask], Optional[SubTask]]:
    """``(previous, next)`` sub-tasks; ``None`` at either end."""
    sub = get(number)
    index = SUBTASKS.index(sub)
    prev = SUBTASKS[index - 1] if index > 0 else None
    nxt = SUBTASKS[index + 1] if index < len(SUBTASKS) - 1 else None
    return prev, nxt
