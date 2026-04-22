# LawMaster

Legal compliance RAG system for Indian industrial, manufacturing, and environmental law.

## Architecture

```
PDF Corpus (43 docs, 1210 pages)
    |
    v
Mistral OCR 3 (mistral-ocr-2512) --> raw markdown + HTML tables
    |
    v
Regex Router (src/chunk/router.py) --> 4 content types:
    |-- Definitions  --> LightRAG (one chunk per defined term)
    |-- Sections     --> LightRAG (hierarchy-aware, split at 1500 tok)
    |-- Tables       --> SQLite + stub chunks in LightRAG
    |-- Amendments   --> JSON index + LLM structured extraction
    |
    v
LightRAG (KG + vector)          SQLite (tables.db)
    |                                |
    +----------+---------------------+
               |
               v
Agno Agent (Gemini 3 Flash, Google)
    |-- LightRAGTool   (legal text search + chunk-ID citations)
    |-- SQLTools        (table queries)
    |-- AmendmentTool   (amendment lookup)
    |-- ReasoningTools  (think/analyze)
    |
    v
FastAPI + SSE Streaming --> Chat UI (PDF.js document viewer + citation badges)
```

## Models

| Role | Model | Provider |
|------|-------|----------|
| OCR | mistral-ocr-2512 | Mistral |
| KG Extraction | Qwen3 32B | Groq |
| Query Keywords | Llama 3.3 70B | Groq |
| Chatbot Agent | Gemini 3 Flash | Google |
| Embeddings | BGE-large-en-v1.5 | Local |

## Current Index Status

**14 of 43 documents indexed** (580 pages, 165 tables).

Extraction of the remaining 29 documents was interrupted by Mistral API instability (HTTP 500/520 errors from `api.mistral.ai`). The pipeline is resume-safe — re-run `scripts/10_full_pipeline.py --extract-only` when the API stabilizes to extract the remaining docs, then `--index-only` to add them to the index.

### Indexed Documents

**Factories Act & Rules (6 docs, 502 pages):**
- FactoryAct1948.pdf (60p) — core act with definitions, sections, schedules
- FactoryRule_1962_map content.pdf (387p) — detailed factory rules
- Factoryplanapproval_mannual.pdf (16p) — plan approval manual
- List of Industries involving hazardous processes (2p) — First Schedule
- Newntam Vetan Adhiniyam 1948.pdf (36p) — Minimum Wages Act (Hindi)
- challan structure.pdf (1p) — challan format

**Chhattisgarh Industrial Policy (8 docs, 78 pages):**
- Byaj_Anudan_Niyam.pdf (11p) — Interest subsidy rules (Hindi)
- Clinical_Trial_Pratipurti_Niyam.pdf (7p) — Clinical trial reimbursement (Hindi)
- EPF_reimbursement_rule_2024.pdf (7p) — EPF reimbursement (Hindi)
- Electricity_Exemption.pdf (17p) — Electricity exemption rules
- Export_certificate_reimbursement_rule_2024.pdf (7p) — Export cert rules (Hindi)
- Industrial Devt Policy Amend Notification (10p) — Policy amendment
- Jal_Urja_dakshata_Vyay_Pratipurti_Niyam.pdf (8p) — Water/energy efficiency (Hindi)
- Margin_money_subsidy_2024.pdf (11p) — Margin money subsidy (Hindi)

### Not Yet Indexed (29 docs, 630 pages)

Pending Mistral API availability. Includes:
- EIA Notification 2006 (45p), HWM Rules 2016 (68p), CPCB Red/Orange/Green list (109p)
- Industrial Policy 2024-30 Hindi (174p), Policy_Amend_2 (46p)
- Various subsidy/grant notifications

## Setup

```bash
# Environment
conda activate ml-env

# Required .env keys
MISTRAL_API_KEY=...
GROQ_API_KEY=...
GEMINI_API_KEY=...

# Run the server
python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --loop asyncio
```

## Pipeline Scripts

```bash
# Phase 1: Extract PDFs via Mistral OCR 3 (resume-safe)
python scripts/10_full_pipeline.py --extract-only

# Phase 2: Index into LightRAG + SQLite (wipe and rebuild)
python scripts/10_full_pipeline.py --index-only --fresh --max-async 8

# Extract structured amendments
python scripts/13_extract_amendments_llm.py

# Validate with test queries
python scripts/12_validate_index.py
```

## Performance

### Extraction (Mistral OCR 3)

| Document | Pages | Time | Speed |
|----------|-------|------|-------|
| FactoryAct1948.pdf | 60 | 46s | 1.3 pages/s |
| Byaj_Anudan_Niyam.pdf (Hindi) | 11 | 6.2s | 1.8 pages/s |
| Clinical_Trial_Pratipurti_Niyam.pdf | 7 | 165s | 0.04 pages/s (API congestion) |

Speed varies significantly due to Mistral API load. Typical: 1-2 pages/s. During congestion: 10-100x slower or 500/520 errors.

### Indexing (Groq Qwen3 32B + BGE-large)

| Metric | Value |
|--------|-------|
| Throughput | ~20 chunks/min at max_async=8 |
| Per-worker speed | ~142 tok/s (Qwen3 32B on Groq LPU) |
| Entities per chunk | 6.7 avg, 21 max |
| Relations per chunk | 3-5 avg for legal sections, 0 for preamble/stubs |
| Bottleneck | Groq rate limits, not embeddings or graph merge |

**Why indexing speed varies per document:**
- **Chunk density**: 387-page Factory Rules has denser legal text = more tokens per LLM call
- **Entity complexity**: Cross-referenced legal provisions produce 6-21 entities per chunk
- **Rate limits**: 8 async workers share Groq's per-key rate limit
- **Graph merge**: Node/edge deduplication overhead grows with graph size
- **Embeddings**: BGE-large runs locally at ~200 batches/s — never the bottleneck

### Cost (14 docs, 580 pages)

| Phase | Provider | Cost |
|-------|----------|------|
| OCR extraction | Mistral | ~$1.16 |
| KG entity extraction | Groq | ~$1.50 |
| Embeddings | Local | $0 |
| **Total** | | **~$2.66** |

Projected for full 43-doc corpus (1210 pages): ~$5.50.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | SSE-streamed chat with the RAG agent |
| `/api/corpus` | GET | Indexed documents grouped by category |
| `/api/chunks/{chunk_id}` | GET | Single chunk text and metadata |
| `/api/pdf/{source}` | GET | Serve source PDF for the document viewer |

## Citation Pipeline

The agent uses LightRAG's `aquery_data` to retrieve chunks with IDs and metadata. Citations flow as:

1. Agent emits inline `[ref:chunk-id]` references in its response
2. Server resolves chunk IDs → sequential `[1]`, `[2]`, etc. with source/page metadata
3. Frontend renders clickable citation badges
4. Clicking a badge opens the PDF.js viewer at the cited page

## Known Issues

- **Mistral API instability**: Intermittent HTTP 500/520 errors from `api.mistral.ai` during OCR extraction. The pipeline retries 3 times per document with backoff. Re-run with `--extract-only` to resume failed extractions.
- **Mistral tables as HTML links**: Table content appears as `[tbl-0.html](tbl-0.html)` placeholders in markdown. Actual table HTML is captured separately in `_tables.json` files and routed to SQLite.
- **Groq structured outputs**: Qwen3 32B and Llama 3.3 70B don't support `response_format=json_schema`. Query functions use `json_object` mode as a workaround.
- **Citation page resolution**: `[ref:chunk-id]` → page mapping needs end-to-end verification after server restart.
