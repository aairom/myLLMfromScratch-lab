# Architecture

```mermaid
flowchart TD
    subgraph Input
        I[input/ folder, read recursively]
    end
    subgraph App
        P[Processing]
        UI[Local UI]
    end
    subgraph Output
        R[output/ timestamped results]
    end
    I --> P --> R
    UI --> P
```
