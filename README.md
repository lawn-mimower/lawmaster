# LawMaster

Retrieval over Indian industrial, manufacturing and environmental law — 43 statutory PDFs,
~1,210 pages, much of it in Hindi, a lot of it tables.

The code lives in [`LawMaster/`](LawMaster/); its [full README](LawMaster/README.md) has
the detail.

## The idea worth stealing

Not "throw the PDFs at a vector store." Legal text isn't one kind of content, and
chunking it uniformly destroys exactly what makes it answerable. A router splits the
corpus four ways and sends each to the store that suits it:

```
PDF corpus ──► Mistral OCR ──► regex router
                                   │
        ┌──────────────┬───────────┴────────┬──────────────┐
        ▼              ▼                    ▼              ▼
   Definitions     Sections              Tables        Amendments
   one chunk       hierarchy-aware      SQLite +       JSON index +
   per term        split at 1500tok     stub chunks    LLM extraction
        │              │                    │              │
        └──────────────┴────────┬───────────┴──────────────┘
                                ▼
                     Agent (Gemini 3 Flash)
              LightRAGTool · SQLTools · AmendmentTool
                                │
                    FastAPI + SSE ──► PDF.js viewer
                                      with citation badges
```

A defined term is a self-contained unit and should be one chunk. A section needs its
place in the hierarchy or you lose what it modifies. A rate table is useless as prose but
answers exactly when queried as SQL. An amendment is a diff against something else and
needs to be resolved, not retrieved.

Answers carry chunk-ID citations back to the source page, which for compliance work is
the difference between useful and unusable.

## Stack

| Role | Model |
|---|---|
| OCR | `mistral-ocr-2512` |
| KG extraction | Qwen3 32B (Groq) |
| Query keywords | Llama 3.3 70B (Groq) |
| Agent | Gemini 3 Flash |
| Embeddings | BGE-large-en-v1.5, local |

## Status — read this before judging the numbers

**14 of 43 documents are indexed** (580 pages, 165 tables). Extraction of the remaining
29 stopped when the Mistral API began returning 500/520s. The pipeline is resume-safe:
re-run `scripts/10_full_pipeline.py --extract-only`, then `--index-only`.

Any benchmark result here is against the partial index.

## Not in this repository

The source corpus, the built vector store, extracted text, the LLM response cache and
session data are all excluded — several hundred MB of it, and not mine to redistribute.
What ships is the pipeline, the agent, the eval harness and the benchmark questions.

You'll need to supply your own corpus and API keys (`.env.example`) to run it.
