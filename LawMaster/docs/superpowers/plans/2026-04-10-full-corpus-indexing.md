# Full Corpus Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract, classify, and index all 43 legal PDFs (1210 pages) into LightRAG + SQLite using the Mistral OCR + regex classification pipeline, then validate with the Kimi K2.5 chatbot agent.

**Architecture:** Mistral OCR 3 extracts all PDFs to markdown + HTML tables. Regex-based router classifies content into 4 types (definitions, sections, tables, amendments). LightRAG indexes definitions + sections + table stubs via Groq Qwen3 32B. SQLite stores parsed table data. Agno agent with Kimi K2.5 on Together AI orchestrates queries across all stores.

**Tech Stack:** Mistral OCR 3, LightRAG (lightrag-hku), Groq (Qwen3 32B, Llama 3.3 70B), Together AI (Kimi K2.5), Agno 2.5.2, sentence-transformers (BGE-large), FastAPI, SQLite, BeautifulSoup

---

## File Structure

```
src/
  config.py ........................ Model IDs, API keys, paths (MODIFY)
  extract/
    mistral_extractor.py ........... Mistral OCR 3 extraction (READY)
    docling_extractor.py ........... REMOVE — superseded
  chunk/
    router.py ...................... Regex classification + routing (READY)
  agent/
    agent.py ....................... Agno agent factory — Together AI (READY)
    rag.py ......................... LightRAG singleton — Groq (MODIFY — entity types)
    tools.py ....................... LightRAGTool + AmendmentTool (READY)
    instructions.py ................ System prompt (MODIFY — update for full corpus)
  server.py ........................ FastAPI + SSE (READY)
scripts/
  10_full_pipeline.py .............. Main pipeline orchestration (MODIFY — remove annotations)
  12_validate_index.py ............. NEW — post-indexing validation queries
  13_extract_amendments_llm.py ..... NEW — LLM structured amendment extraction
data/
  tables.db ........................ WIPE + RECREATE
  amendments.json .................. WIPE + RECREATE
rag_storage/ ....................... WIPE + RECREATE
extraction_output/mistral_v2/ ...... Output directory for Mistral extractions
skill/
  SKILL.md ......................... SQL skill (READY)
  references/schema.md ............. Dynamic schema docs (READY)
```

---

### Task 1: Clean up stale files and verify environment

**Files:**
- Remove: `src/extract/docling_extractor.py`
- Verify: `.env` (MISTRAL_API_KEY, GROQ_API_KEY, TOGETHER_API_KEY)

- [ ] **Step 1: Remove stale Docling extractor**

```bash
cd /home/pc/Downloads/LawMaster
rm src/extract/docling_extractor.py
```

- [ ] **Step 2: Verify API keys are set**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import os
for key in ['MISTRAL_API_KEY', 'GROQ_API_KEY', 'TOGETHER_API_KEY']:
    val = os.getenv(key, '')
    print(f'{key}: {\"SET\" if val else \"MISSING\"} ({len(val)} chars)')
"
```

Expected: All 3 keys show SET.

- [ ] **Step 3: Verify packages**

```bash
/home/pc/anaconda3/envs/ml-env/bin/pip list 2>/dev/null | grep -E "mistralai|lightrag|agno|sentence-trans|beautifulsoup"
```

Expected: mistralai, lightrag-hku, agno, sentence-transformers, beautifulsoup4 all present.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove stale docling extractor, verify environment"
```

---

### Task 2: Fix pipeline script — disable annotations, align with regex-only

**Files:**
- Modify: `scripts/10_full_pipeline.py`

The pipeline currently passes `annotate=True` to `extract_pdf()`. Since we decided annotations are unnecessary (regex produces identical routing and all queries passed), disable them.

- [ ] **Step 1: Update extraction call in pipeline**

In `scripts/10_full_pipeline.py`, line ~118, change the `extract_pdf()` call:

```python
# Before:
            result = extract_pdf(
                pdf_path=pdf_info["path"],
                output_dir=out_dir,
                api_key=MISTRAL_API_KEY,
                annotate=True,
            )

# After:
            result = extract_pdf(
                pdf_path=pdf_info["path"],
                output_dir=out_dir,
                api_key=MISTRAL_API_KEY,
            )
```

(`annotate` defaults to `False` already in the extractor signature.)

- [ ] **Step 2: Update output directory to clean path**

In `scripts/10_full_pipeline.py`, change `MISTRAL_OUT` to a cleaner name:

```python
# Before:
MISTRAL_OUT = EXTRACTION_OUTPUT_DIR / "mistral_v2"

# After:
MISTRAL_OUT = EXTRACTION_OUTPUT_DIR / "mistral"
```

- [ ] **Step 3: Verify the pipeline runs dry on one doc**

```bash
mkdir -p /home/pc/Downloads/LawMaster/extraction_output/mistral
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from scripts import *  # just test import
print('Pipeline imports OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/10_full_pipeline.py
git commit -m "fix: disable annotations in pipeline, use regex-only classification"
```

---

### Task 3: Update agent entity types and instructions for full corpus

**Files:**
- Modify: `src/agent/rag.py`
- Modify: `src/agent/instructions.py`

The current entity types and instructions are tailored for the Factories Act only. Expand for the full 43-doc corpus spanning 3 categories.

- [ ] **Step 1: Update entity types in rag.py**

In `src/agent/rag.py`, find the `addon_params` dict in `_build_rag()` and update:

```python
addon_params={
    "entity_types": [
        "Definition", "Section", "Amendment", "Schedule",
        "Act", "Rule", "Authority", "Penalty", "Provision",
        "Industry", "Chemical", "Regulation", "Notification",
        "Subsidy", "Fee", "Classification",
    ],
    "language": "English",
},
```

- [ ] **Step 2: Update system instructions for broader corpus**

In `src/agent/instructions.py`, update the system prompt to reflect the full corpus scope. The current prompt references only "Indian industrial and manufacturing law". Update to:

```python
SYSTEM_INSTRUCTIONS = [
    "You are LawMaster, an expert on Indian industrial, manufacturing, and environmental compliance law.",
    "Your knowledge base covers 43 legal documents across 3 domains:",
    "  1. Factories Act & Rules — workplace safety, hazardous processes, chemical limits, working hours",
    "  2. Pollution Control & Environment — CPCB industry classifications, EIA notifications, HWM rules",
    "  3. Chhattisgarh Industrial Policy — subsidies, grants, incentives, operational rules (Hindi + English)",
    "",
    "TOOL SELECTION:",
    "- search_legal_text: definitions, sections, provisions, procedures, penalties — any narrative legal content",
    "- SQL (run_sql/list_tables): numeric limits, rates, classification tables, fee schedules, subsidy amounts",
    "- lookup_amendments: check if a section has been modified, substituted, or omitted",
    "- think/analyze: break down complex multi-part questions before answering",
    "",
    "RULES:",
    "- Always cite the source document and section in your answer",
    "- If a question spans multiple tools, call them in sequence and synthesize",
    "- For tabular data (exposure limits, industry categories), prefer SQL over RAG",
    "- For Hindi policy documents, answer in English unless the user writes in Hindi",
    "- If the question is outside your indexed corpus, say so clearly — do not hallucinate",
    "- When answering about a specific section, always check amendments to give the current version",
]
```

- [ ] **Step 3: Commit**

```bash
git add src/agent/rag.py src/agent/instructions.py
git commit -m "feat: expand entity types and instructions for full 43-doc corpus"
```

---

### Task 4: Run Phase 1 — Extract all 43 PDFs via Mistral OCR 3

**Files:**
- Run: `scripts/10_full_pipeline.py --extract-only`
- Output: `extraction_output/mistral/` (43 markdown files + tables + metadata)

This is the expensive API call phase (~$2.42 Mistral OCR). Resume-safe — skips already-extracted docs.

- [ ] **Step 1: Wipe old extraction output**

```bash
rm -rf /home/pc/Downloads/LawMaster/extraction_output/mistral/
rm -rf /home/pc/Downloads/LawMaster/extraction_output/mistral_v2/
```

- [ ] **Step 2: Run extraction**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python scripts/10_full_pipeline.py --extract-only 2>&1 | tee extraction_log.txt
```

Expected runtime: 15-30 minutes for 1210 pages.
Expected output: `extraction_output/mistral/extraction_manifest.json` with 43 entries, most status "ok".

- [ ] **Step 3: Verify extraction results**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import json
manifest = json.load(open('extraction_output/mistral/extraction_manifest.json'))
ok = [r for r in manifest if r.get('status') == 'ok']
err = [r for r in manifest if r.get('status') == 'error']
skip = [r for r in manifest if r.get('status') == 'skipped']
pages = sum(r.get('pages', 0) for r in ok)
print(f'Extracted: {len(ok)}, Skipped: {len(skip)}, Errors: {len(err)}, Pages: {pages}')
if err:
    for r in err:
        print(f'  FAIL: {r.get(\"name\")}: {r.get(\"error\", \"?\")[:80]}')
"
```

Expected: 43 extracted, 0 errors, ~1210 pages total.

- [ ] **Step 4: Re-run for any failures**

If any docs failed (transient API errors), re-run with `--resume`:

```bash
/home/pc/anaconda3/envs/ml-env/bin/python scripts/10_full_pipeline.py --extract-only
```

Resume mode skips already-extracted docs and retries failures.

- [ ] **Step 5: Spot-check extraction quality**

```bash
# Check a Hindi doc
head -50 "extraction_output/mistral/industrial policy, ammendments and notifications/Rojgaar_Anudan_Niyam_2024.md"

# Check a large English doc
wc -l "extraction_output/mistral/factories act/FactoryAct1948.md"

# Check table extraction
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import json, glob
tables = glob.glob('extraction_output/mistral/**/*_tables.json', recursive=True)
total = 0
for t in tables:
    data = json.load(open(t))
    total += len(data)
    if data:
        print(f'{t.split(\"/\")[-1]}: {len(data)} tables')
print(f'Total tables extracted: {total}')
"
```

---

### Task 5: Run Phase 2 — Index into LightRAG + SQLite

**Files:**
- Run: `scripts/10_full_pipeline.py --index-only --fresh --max-async 8`
- Output: `rag_storage/` (LightRAG graph), `data/tables.db` (SQLite), `data/raw_amendments.json`

This is the Groq API phase (~$3 for entity extraction). Processes all extracted markdown through the regex router, then indexes.

- [ ] **Step 1: Clear old storage**

```bash
rm -rf /home/pc/Downloads/LawMaster/rag_storage/
rm -f /home/pc/Downloads/LawMaster/data/tables.db
rm -f /home/pc/Downloads/LawMaster/data/raw_amendments.json
```

- [ ] **Step 2: Run indexing**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python scripts/10_full_pipeline.py --index-only --fresh --max-async 8 2>&1 | tee indexing_log.txt
```

Expected runtime: 2-5 hours (4000+ chunks, Groq rate limits).
Monitor progress — each doc prints node/edge counts.

- [ ] **Step 3: Verify indexing results**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import json
results = json.load(open('extraction_output/mistral/index_results.json'))
print(f'Nodes: {results[\"nodes\"]}, Edges: {results[\"edges\"]}')
ok = [d for d in results['docs'] if d.get('status') == 'ok']
err = [d for d in results['docs'] if d.get('status') == 'error']
print(f'Indexed: {len(ok)}, Errors: {len(err)}')
print(f'Total chunks: {sum(d.get(\"chunks\", 0) for d in ok)}')
print(f'Total defs: {sum(d.get(\"defs\", 0) for d in ok)}')
print(f'Total sections: {sum(d.get(\"sections\", 0) for d in ok)}')
print(f'Total tables: {sum(d.get(\"tables\", 0) for d in ok)}')
print(f'Total amendments: {sum(d.get(\"amendments\", 0) for d in ok)}')
"
```

- [ ] **Step 4: Verify SQLite tables**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sqlite3
conn = sqlite3.connect('data/tables.db')
cursor = conn.execute('SELECT table_id, source_document, num_rows, num_cols FROM _table_registry ORDER BY source_document')
rows = cursor.fetchall()
print(f'Tables registered: {len(rows)}')
for r in rows:
    print(f'  {r[0]:40s} | {r[2]:>4} rows | {r[3]:>3} cols | {r[1]}')
conn.close()
"
```

---

### Task 6: Extract structured amendments via LLM

**Files:**
- Create: `scripts/13_extract_amendments_llm.py`
- Output: `data/amendments.json`

The pipeline collects raw amendment text blocks into `data/raw_amendments.json`. This step uses Groq Qwen3 32B to extract structured JSON (target_section, amendment_type, amending_act, effective_date, description).

- [ ] **Step 1: Create amendment extraction script**

```python
#!/usr/bin/env python3
"""Extract structured amendment records from raw amendment blocks via LLM."""

import sys, os, json, asyncio, re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.config import GROQ_API_KEY, EXTRACTION_MODEL, DATA_DIR
from lightrag.llm.openai import openai_complete_if_cache


def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


PROMPT = """Extract all amendments from this legal text. For each amendment, return a JSON object with:
- target_act: The Act being amended
- target_section: The section number being amended (e.g., "2", "41B")
- amendment_type: one of [substituted, inserted, omitted, added, renumbered]
- amending_act: The Act that made the amendment (e.g., "Act 94 of 1976")
- effective_date: When the amendment took effect (if stated), or null
- description: Brief one-sentence description

Return a JSON object with key "amendments" containing an array.
If no amendments are found, return {"amendments": []}.

TEXT:
"""


async def extract_amendments(raw_blocks: list[dict]) -> list[dict]:
    all_amendments = []

    for i, block in enumerate(raw_blocks):
        text = block["text"][:4000]  # Truncate to fit context
        source = block.get("source", "unknown")

        try:
            result = await openai_complete_if_cache(
                model=EXTRACTION_MODEL,
                prompt=PROMPT + text,
                system_prompt="You extract structured amendment data from legal text. Return valid JSON only.",
                history_messages=[],
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
                response_format={"type": "json_object"},
            )
            result = strip_think(result)
            parsed = json.loads(result)
            amendments = parsed.get("amendments", [])
            for a in amendments:
                a["source_document"] = source
            all_amendments.extend(amendments)
            print(f"  [{i+1}/{len(raw_blocks)}] {source}: {len(amendments)} amendments")
        except Exception as e:
            print(f"  [{i+1}/{len(raw_blocks)}] {source}: ERROR — {e}")

    return all_amendments


async def main():
    raw_path = DATA_DIR / "raw_amendments.json"
    if not raw_path.exists():
        print("No raw_amendments.json found. Run the indexing pipeline first.")
        return

    with open(raw_path) as f:
        raw_blocks = json.load(f)

    print(f"Processing {len(raw_blocks)} raw amendment blocks...")
    amendments = await extract_amendments(raw_blocks)

    out_path = DATA_DIR / "amendments.json"
    with open(out_path, "w") as f:
        json.dump(amendments, f, indent=2, ensure_ascii=False)

    print(f"\nExtracted {len(amendments)} structured amendments → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
```

Save to `scripts/13_extract_amendments_llm.py`.

- [ ] **Step 2: Run amendment extraction**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python scripts/13_extract_amendments_llm.py
```

Expected: Structured amendments.json with target_section, amendment_type, etc.

- [ ] **Step 3: Verify amendments**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import json
amendments = json.load(open('data/amendments.json'))
print(f'Total amendments: {len(amendments)}')
types = {}
for a in amendments:
    t = a.get('amendment_type', '?')
    types[t] = types.get(t, 0) + 1
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f'  {t}: {c}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/13_extract_amendments_llm.py data/amendments.json
git commit -m "feat: extract structured amendments via Groq LLM"
```

---

### Task 7: Validate full system with test queries

**Files:**
- Create: `scripts/12_validate_index.py`

Run queries spanning all 3 corpus domains — Factories Act, Pollution Control, Chhattisgarh Industrial Policy — covering all 4 content types.

- [ ] **Step 1: Create validation script**

```python
#!/usr/bin/env python3
"""Validate the full indexed corpus with queries across all domains and content types."""

import sys, os, asyncio, re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.config import RAG_STORAGE_DIR, GROQ_API_KEY, EXTRACTION_MODEL, QUERY_MODEL
from lightrag import QueryParam
from lightrag.llm.openai import openai_complete_if_cache


QUESTIONS = [
    # Factories Act — definitions
    ("What is the definition of 'factory'?", "Definition"),
    ("What does 'hazardous process' mean under the Factories Act?", "Definition"),
    # Factories Act — sections
    ("What safety provisions apply to hazardous processes?", "Section"),
    ("What are the working hour restrictions for adult workers?", "Section"),
    ("What penalties exist for employing children?", "Section"),
    # Factories Act — tables (via RAG stub)
    ("What is the permissible exposure limit for Benzene?", "Table/RAG"),
    # Pollution Control
    ("What industries fall under the Red category in CPCB classification?", "Section/Table"),
    ("What does the EIA Notification 2006 require for environmental clearance?", "Section"),
    ("What are the rules for hazardous waste management?", "Section"),
    # Chhattisgarh Industrial Policy (Hindi docs)
    ("What subsidies are available under the Chhattisgarh Industrial Policy?", "Section"),
    ("What is the process for capital subsidy application?", "Section"),
    # Amendments
    ("Has Section 2 of the Factories Act been amended?", "Amendment"),
]


async def main():
    from src.agent.rag import get_rag

    rag = get_rag()

    async def query_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        if kwargs.pop("keyword_extraction", False):
            kwargs["response_format"] = {"type": "json_object"}
        return await openai_complete_if_cache(
            model=QUERY_MODEL, prompt=prompt,
            system_prompt=system_prompt, history_messages=history_messages or [],
            api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )

    print(f"Validating {len(QUESTIONS)} queries...\n")
    results = []

    for q, expected_type in QUESTIONS:
        print(f"Q [{expected_type}]: {q}")
        try:
            r = await rag.aquery(
                q, param=QueryParam(mode="mix", top_k=5, model_func=query_func)
            )
            answer = str(r)[:300]
            has_content = "[no-context]" not in answer and len(answer) > 50
            status = "PASS" if has_content else "FAIL"
            print(f"  {status}: {answer[:150]}...")
            results.append({"q": q, "type": expected_type, "status": status})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"q": q, "type": expected_type, "status": "ERROR"})
        print()

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"{'='*60}")
    print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed, {errors} errors")
    print(f"{'='*60}")
    for r in results:
        icon = {"PASS": "+", "FAIL": "-", "ERROR": "!"}[r["status"]]
        print(f"  [{icon}] [{r['type']:12s}] {r['q']}")


if __name__ == "__main__":
    asyncio.run(main())
```

Save to `scripts/12_validate_index.py`.

- [ ] **Step 2: Run validation**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python scripts/12_validate_index.py
```

Expected: 10+ out of 12 queries return substantive answers (PASS).

- [ ] **Step 3: Commit**

```bash
git add scripts/12_validate_index.py
git commit -m "feat: add full corpus validation script"
```

---

### Task 8: Test chatbot end-to-end with Kimi K2.5

**Files:**
- Run: `src/server.py`

Validate the full agent loop: user question → Kimi K2.5 → tool selection → LightRAG/SQL/Amendments → streamed response.

- [ ] **Step 1: Start the server**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --loop asyncio
```

- [ ] **Step 2: Test via curl**

```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the definition of factory under the Factories Act?"}'
```

Expected: SSE stream with tool_call events (search_legal_text) followed by content events with the answer.

- [ ] **Step 3: Test via browser**

Open `http://localhost:8000` in browser. Test these questions:
1. "What is the definition of factory?" (RAG)
2. "What is the exposure limit for benzene?" (SQL + RAG)
3. "Has Section 41B been amended?" (Amendment tool)
4. "What subsidies does Chhattisgarh Industrial Policy offer?" (Hindi corpus RAG)
5. "List all Red category industries" (SQL)

Verify: each question triggers the correct tool, returns a substantive answer with citations.

- [ ] **Step 4: Commit final state**

```bash
git add -A
git commit -m "feat: complete full corpus indexing — 43 docs, Mistral OCR, Groq KG, Kimi K2.5 agent"
```

---

## Execution Order and Dependencies

```
Task 1: Clean up + verify env
   │
   ▼
Task 2: Fix pipeline script
   │
   ▼
Task 3: Update entity types + instructions
   │
   ▼
Task 4: Phase 1 — Extract all 43 PDFs (~$2.42, ~20 min)
   │
   ▼
Task 5: Phase 2 — Index into LightRAG + SQLite (~$3, ~3 hrs)
   │
   ├──────────────────┐
   ▼                  ▼
Task 6: Amendments  Task 7: Validate queries
   │                  │
   └──────┬───────────┘
          ▼
Task 8: End-to-end chatbot test
```

Tasks 6 and 7 can run in parallel after Task 5 completes.

## Cost Estimate

| Phase | Provider | Cost |
|-------|----------|------|
| Extraction (1210 pages) | Mistral OCR 3 | ~$2.42 |
| KG Extraction (~4000 chunks) | Groq Qwen3 32B | ~$3.00 |
| Amendment extraction | Groq Qwen3 32B | ~$0.10 |
| Embeddings | Local BGE-large | $0.00 |
| **Total** | | **~$5.50** |
