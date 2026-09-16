# vgraph_rag

Hybrid graph-vector RAG over an [OKF](knowledge/) v0.2 knowledge bundle.

`knowledge/` is the source of truth (markdown + YAML frontmatter). `.vgraph/` is a derived, gitignored index.

```mermaid
flowchart LR
  subgraph ingest["Bundle / ingest"]
    KB["Knowledge Bundle<br/>knowledge/"]
    C["Concepts<br/>typed markdown<br/>catalog/log skipped"]
    EMB["Embed<br/>all-MiniLM-L6-v2<br/>1 vector per concept"]
    GR["Graph<br/>link + sibling<br/>undirected"]
    KB --> C --> EMB
    C --> GR
  end

  subgraph idx["Derived index"]
    VG[".vgraph/<br/>graph.json + embeddings.npz"]
  end

  subgraph qry["Query"]
    Q["Query"]
    QE["Same MiniLM"]
    S["Vector top-k seeds"]
    N["1-hop neighbors<br/>score empty"]
    H["Hits"]
    R["retrieve → JSON"]
    PK["pack"]
    LLM["ask → OpenAI-compat LLM"]
    Q --> QE --> S --> N --> H
    H --> R
    H --> PK --> LLM
  end

  EMB --> VG
  GR --> VG
  VG --> S
  VG --> N
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Optional: `OPENAI_BASE_URL`, `OPENAI_MODEL` (default `gpt-4o-mini`).

First ingest downloads a local embedding model (`all-MiniLM-L6-v2` via sentence-transformers / PyTorch).

## Commands

```powershell
python -m vgraph ingest
python -m vgraph retrieve "what is this bundle?"
python -m vgraph ask "what is this bundle?"
```

`--bundle` and `--index` default to `knowledge` and `.vgraph`. `ask` / `retrieve` rebuild the index when `knowledge/` is newer.

Human verification of a concept: use the `verify-concept` Cursor skill (stamps `verified` in frontmatter).
