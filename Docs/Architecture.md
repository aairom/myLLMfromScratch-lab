# Architecture

## Layers

```mermaid
flowchart TD
    subgraph Input
        I[input/ folder, read recursively]
        D[data/sample_corpus.txt]
        E[.env settings]
    end
    subgraph App
        A[app.py<br/>st.navigation, PAGES registry]
        PG[pages/page_*.py<br/>ten Streamlit pages]
        UI[ui/<br/>state, components, sidebar, lab, demo]
        ED[edu/<br/>lessons, snippets]
        VZ[viz/<br/>tokens, embeddings, attention, training]
        LM[llm/<br/>tokenizer, embeddings, positional,<br/>attention, layers, transformer]
        TR[training/<br/>data, optimizers, schedulers,<br/>trainer, checkpoint]
    end
    subgraph Output
        R[output/ timestamped checkpoints]
    end
    I --> UI
    D --> UI
    E --> A
    A --> PG --> UI
    UI --> ED
    UI --> VZ
    UI --> LM
    UI --> TR
    VZ --> LM
    TR --> LM
    TR --> R
```

Dependency rule: arrows only point downwards. `llm/`, `training/` and `viz/`
never import Streamlit, so they can be tested and read on their own.

## The model (`llm/`)

```mermaid
flowchart TD
    T[text] --> TK[Tokenizer<br/>char or BPE]
    TK --> ID[token ids]
    ID --> EM[Token embedding<br/>lookup table]
    EM --> PE[+ positional encoding<br/>sinusoidal or learned]
    PE --> LN1
    subgraph BLK[Transformer block x N]
        LN1[LayerNorm] --> MHA[Causal multi-head<br/>self-attention]
        MHA --> R1((+))
        R1 --> LN2[LayerNorm]
        LN2 --> FF[Feed-forward<br/>Linear, GELU, Linear]
        FF --> R2((+))
    end
    R2 --> LNF[Final LayerNorm]
    LNF --> HD[LM head<br/>tied with embedding]
    HD --> LG[logits]
    LG --> SM[softmax, sampling]
```

Pre-norm residual blocks are used because they train stably without a long
warm-up. Attention is computed as `softmax(Q K^T / sqrt(d_k) + causal mask) V`
per head, then the heads are concatenated and projected.

## Training loop (`training/`)

```mermaid
flowchart TD
    C[corpus] --> TK[tokenize]
    TK --> SP[train / validation split]
    SP --> BT[random windows<br/>inputs x, targets x shifted by 1]
    BT --> FW[forward pass]
    FW --> LS[cross-entropy loss]
    LS --> BW[backward: autograd gradients]
    BW --> CL[clip global grad norm]
    CL --> OP[optimizer step<br/>SGD / Adam / AdamW]
    SCH[LR schedule<br/>constant or warm-up + cosine] --> OP
    OP --> M[metrics: loss, lr, grad norm, tokens/s]
    M --> LV[live Plotly charts]
    OP --> BT
    OP --> CK[checkpoint in output/]
```

## UI data flow

```mermaid
sequenceDiagram
    participant S as Sidebar form
    participant ST as st.session_state
    participant P as Page
    participant H as ui/demo.py and ui/lab.py
    participant V as viz/
    S->>ST: ModelSettings (validated)
    ST->>ST: ensure_model builds tokenizer + GPT
    P->>ST: read tokenizer, model, device
    P->>H: call pure helper with inputs
    H->>V: build figure from tensors
    V-->>P: Plotly figure
    P-->>P: st.plotly_chart with unique key
```

## Animations (`animgen/`)

```mermaid
flowchart LR
    ENV["ANIM_ variables in .env and CLI flags"] --> OPT[options]
    CTX[context<br/>real tokenizer, model, trainer run] --> SC
    OPT --> CTX
    SC[scenes_a..d<br/>one Scene per sub-task] --> RND[scene.render_scene<br/>tween frames and holds]
    CNV[canvas, parts, theme, fonts<br/>Pillow drawing toolkit] --> SC
    RND --> GIF[scene.save_gif<br/>shared palette]
    GIF --> OUT[Docs/subtasks/animations/*.gif]
    OUT --> GAL[gallery.py index.html]
    REG[registry + cli] --> RND
```

A `Scene` is a caption per step plus one `draw(canvas, ctx, step, t)` function.
The `Context` computes everything lazily from the project's own code, so a GIF
that needs the trained model trains it once and the others stay fast. Like
`llm/` and `training/`, the package never imports Streamlit.

## Design decisions

- **Hand-written components**: `nn.MultiheadAttention`, `nn.TransformerEncoder`
  and `nn.LayerNorm` are used only inside tests as numerical references.
- **Streamlit-free helpers**: page logic lives in `ui/demo.py` and
  `ui/lab.py`, so it is covered by plain unit tests.
- **One test module per source module**, plus `AppTest` runs of every page.
- **Configuration only through `.env`**: host, port, device, seed and folders.
- **Checkpoints** store model config, weights, optimizer state and history;
  the sinusoidal table is a non-persistent buffer and is rebuilt on load.
