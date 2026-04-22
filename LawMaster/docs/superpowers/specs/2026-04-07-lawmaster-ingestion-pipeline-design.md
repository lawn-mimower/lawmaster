# LawMaster Ingestion Pipeline - Design & Implementation Plan

## Project Overview

**LawMaster** is a Streamlit-based legal compliance RAG tool for Indian industrial/manufacturing law. This document covers the ingestion pipeline: extracting, structuring, and indexing legal documents into LightRAG (vector + knowledge graph) and SQLite (structured tables).

**PoC scope:** Two documents to validate the full pipeline before scaling to 45 PDFs.
- **English:** `FactoryAct1948.pdf` (722KB, digital, well-structured, all 4 content types, located at `project 2 _ai tool for compliance -20260331T205330Z-1-001/project 2 _ai tool for compliance /factories act/FactoryAct1948.pdf`)
- **Hindi:** `capital_subsidy_Rules.pdf` (2.9MB, Hindi, subsidy rules with tables + definitions, located at `project 2 _ai tool for compliance -20260331T205330Z-1-001/project 2 _ai tool for compliance /industrial policy, ammendments and notifications/capital_subsidy_Rules.pdf`)

## Architecture Decision: Hybrid Extraction (Approach A)

```
English PDF ──> Docling (raw PDF, full AI: Heron layout + TableFormer)
Hindi PDF   ──> Mistral OCR (table_format="html") ──> Docling convert_string(md)
                                                            |
                                                      DoclingDocument
                                                            |
                              +-----------------------------+-----------------------------+
                              |                             |                             |
                     Type Classification              Table Extraction            Amendment Extraction
                     (rules + patterns)               (TableFormer / HTML)        (LLM one-shot)
                              |                             |                             |
                    +----+----+----+                   HTML -> SQL               JSON amendment index
                    |              |                   + stub chunk                    |
              Definitions     Sections                in LightRAG              LightRAG custom KG
              (1 chunk each)  (hierarchy)                                      (edges to parent sections)
                    |              |
                    +--------------+
                           |
                  LightRAG insert_custom_chunks()
                  (with metadata-enriched content)
```

**Why this approach:**
- Docling's Heron layout model + TableFormer for English PDFs (best structure + 97.9% table accuracy)
- Mistral OCR for Hindi PDFs (97.55% Hindi accuracy vs mediocre EasyOCR/Tesseract)
- DoclingDocument as unified intermediate format regardless of source
- Both paths converge, so downstream chunking/indexing logic is shared

## The 4 Content Types

### 1. Definitions

**What they look like:** Section 2 of the Factories Act: `(a) "adult" means a person who has completed his eighteenth year of age;`

**Chunking strategy:** One chunk per definition. Each chunk is self-contained.

**Metadata prefix format (prepended to chunk content before LightRAG insertion):**
```
[Act: Factories Act, 1948] [Chapter: I - Preliminary] [Section: 2] [Type: Definition] [Term: adult]

"adult" means a person who has completed his eighteenth year of age;
```

**Detection method:** Definitions typically appear under a "Definitions" or "Interpretation" heading. Within that section:
- Pattern: `(letter) "term" means ...;` — regex: `\([a-z]+\)\s*"([^"]+)"\s*means`
- Each match is one chunk
- Some definitions have sub-clauses (i), (ii), etc. — keep them together with the parent definition

**LightRAG entity type:** `Definition`

### 2. Sections

**What they look like:** Numbered provisions with headings, sub-sections, clauses.

```
41B. Compulsory disclosure of information by the occupier.—
(1) The occupier of every factory involving a hazardous process shall...
(2) The occupier shall...
```

**Chunking strategy:** One chunk per section (including all sub-sections). If a section exceeds 1500 tokens, split at sub-section boundaries `(1)`, `(2)`, etc., keeping the section heading in each chunk.

**Metadata prefix format:**
```
[Act: Factories Act, 1948] [Chapter: IVA - Provisions Relating to Hazardous Processes] [Section: 41B] [Type: Section]

41B. Compulsory disclosure of information by the occupier.—
(1) The occupier of every factory involving a hazardous process shall...
```

**Detection method:** Docling's `SectionHeaderItem` hierarchy. Sections are identified by:
- Heading level from DoclingDocument tree
- Numeric pattern at start: `\d+[A-Z]?\.\s` (e.g., "41B.")
- Chapter grouping from parent heading

**LightRAG entity type:** `Section`

### 3. Tables

**What they look like:** Schedules, rate tables, classification lists, wage tables, fee structures.

**Storage:** SQLite database at `./data/tables.db`

**Each table gets:**
1. A SQLite table with columns matching the source
2. A metadata row in a `_table_registry` table (source_doc, table_name, description, page_number, column_descriptions)
3. A "stub chunk" inserted into LightRAG so the KG knows the table exists:

```
[Act: Factories Act, 1948] [Schedule: First Schedule] [Type: Table Stub]

The First Schedule contains a list of industries involving hazardous processes
as defined under Section 2(cb). This table has columns: [serial_number,
industry_name, hazardous_process_description]. For specific entries, query
the SQL database table 'factories_act_first_schedule'.
```

**Extraction method:**
- English PDFs: Docling's TableFormer extracts structured `TableItem` objects from DoclingDocument
- Hindi PDFs: Mistral OCR with `table_format="html"` → parse HTML tables with BeautifulSoup → SQL INSERT
- Both paths: an LLM call generates the table description and column descriptions for the stub chunk

**LightRAG entity type:** `Schedule` or `Table`

### 4. Amendments

**What they look like:** Two forms:

*Inline (within Acts):* Footnotes like `[Substituted by Act 94 of 1976, s. 9]`

*Standalone documents:* `Industrial Devt Policy Amend Notification 13.01.2026.pdf` — entire documents that modify other documents.

**Extraction method:** LLM one-shot structured extraction. For each document (or section containing amendment markers), call the LLM with:

```
Prompt: Extract all amendments from the following legal text. For each amendment, provide:
- target_act: The Act being amended
- target_section: The section being amended
- amendment_type: one of [substituted, inserted, omitted, added, renumbered]
- effective_date: When the amendment took effect (if stated)
- amendment_text: The amendment language
- new_text: The substituted/inserted text (if applicable)
- source_document: The document containing this amendment

Return as JSON array.
```

**Storage:** 
- JSON amendment index at `./data/amendments.json`
- Each amendment also inserted into LightRAG via `insert_custom_kg()` with an edge linking the amendment entity to the target section entity

**Query-time behavior:** When a user asks about Section X, the chatbot:
1. Retrieves Section X from LightRAG
2. Checks amendment index for amendments targeting Section X
3. Synthesizes the current effective version

**LightRAG entity type:** `Amendment`

## Technical Configuration

### LightRAG Setup

```python
from lightrag import LightRAG, QueryParam
from lightrag.prompt import PROMPTS

# Custom entity types for legal domain
rag = LightRAG(
    working_dir="./rag_storage",
    addon_params={
        "entity_types": [
            "Definition", "Section", "Amendment", "Schedule",
            "Act", "Rule", "Authority", "Penalty", "Provision"
        ],
        "language": "English",
    },
    chunk_token_size=1500,
    chunk_overlap_token_size=100,
)

# Customize extraction prompt for legal awareness
PROMPTS["entity_extraction_system_prompt"] = """
... (customize to instruct: always extract Act name, Section number,
defined terms, cross-references to other sections/acts as entities;
extract "amends", "refers_to", "defines", "penalizes" as relation types)
"""
```

### Key LightRAG APIs to Use

| API | Purpose |
|-----|---------|
| `insert_custom_chunks(full_text, text_chunks)` | Insert pre-chunked definitions and sections |
| `insert_custom_kg(custom_kg)` | Insert amendment edges linking to parent sections |
| `query(q, param=QueryParam(mode="mix"))` | Query with broadest retrieval (KG + vector) |
| `query_data(q, param)` | Retrieve structured results without LLM generation (for agent use) |
| `QueryParam(only_need_context=True)` | Get context string for custom agent orchestration |

### SQLite Schema

```sql
-- Registry of all extracted tables
CREATE TABLE _table_registry (
    table_id TEXT PRIMARY KEY,
    source_document TEXT NOT NULL,
    table_name TEXT NOT NULL,
    description TEXT,
    page_number INTEGER,
    column_descriptions TEXT,  -- JSON object
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Example: Factories Act First Schedule
CREATE TABLE factories_act_first_schedule (
    serial_number INTEGER,
    industry_name TEXT,
    hazardous_process_description TEXT,
    _source_page INTEGER,
    _source_document TEXT
);
```

### Environment & Dependencies

**Python environment:** `/home/pc/anaconda3/envs/ml-env/bin/python`

**Existing packages:** lightrag-hku 1.4.9.11, mistralai 1.12.3, google-generativeai 0.8.5, sentence-transformers 5.1.0

**New packages needed:**
```
pip install docling          # Document parsing (Heron layout + TableFormer)
pip install beautifulsoup4   # HTML table parsing from Mistral output
pip install marko            # Markdown parsing (Docling dependency, likely auto-installed)
```

**API keys (in .env):**
- `MISTRAL_API_KEY` — for OCR of Hindi documents
- `GEMINI_API_KEY` — for amendment extraction LLM calls (or use Mistral chat)

### File Structure

```
LawMaster/
  .env
  docs/superpowers/specs/        # This design doc
  data/
    tables.db                    # SQLite for extracted tables
    amendments.json              # Amendment cross-reference index
  rag_storage/                   # LightRAG working directory
  src/
    config.py                    # Shared config (paths, API keys, LightRAG init)
    extract/
      __init__.py
      docling_extractor.py       # English PDF -> DoclingDocument
      mistral_extractor.py       # Hindi PDF -> Markdown -> DoclingDocument
      classifier.py              # Classify DoclingDocument elements by content type
    chunk/
      __init__.py
      definition_chunker.py      # Extract + chunk definitions
      section_chunker.py         # Extract + chunk sections with hierarchy
      table_extractor.py         # Extract tables -> SQL + stub chunks
      amendment_extractor.py     # LLM one-shot amendment extraction
    index/
      __init__.py
      lightrag_indexer.py        # Insert chunks + custom KG into LightRAG
      sql_indexer.py             # Insert tables into SQLite
    pipeline.py                  # Orchestrates full pipeline for a document
  scripts/
    process_poc.py               # Run pipeline on the 2 PoC documents
```

## Implementation Plan (Sequential Steps)

### Phase 1: Extraction Layer

**Step 1: Project scaffolding + config**
- Create the file structure above
- Write `config.py`: load `.env`, initialize paths, LightRAG instance with custom entity types
- Install `docling` and verify it loads (model downloads on first run)
- Estimated: straightforward setup

**Step 2: English extraction with Docling**
- Write `docling_extractor.py`: take a PDF path, run Docling's `DocumentConverter.convert()`, return `DoclingDocument`
- Test on `FactoryAct1948.pdf`
- Verify: heading hierarchy is detected, tables are extracted, definitions section is parseable
- Save the DoclingDocument as JSON for inspection

**Step 3: Hindi extraction with Mistral + Docling**
- Write `mistral_extractor.py`: 
  - Call Mistral OCR API with `table_format="html"`, `extract_header=True`, `extract_footer=True`
  - Collect per-page markdown + separate HTML tables
  - Feed markdown into `DocumentConverter.convert_string(md, InputFormat.MD)` → DoclingDocument
  - Store HTML tables separately (they bypass Docling's simple markdown table parser)
- Test on `capital_subsidy_Rules.pdf`
- Verify: Hindi text is accurately extracted, structure is preserved, tables are captured

### Phase 2: Content Type Classification & Chunking

**Step 4: Definition chunker**
- Write `definition_chunker.py`:
  - Traverse DoclingDocument tree, find sections with "Interpretation" / "Definitions" / "paribhasha" headings
  - Within those sections, split on definition patterns: `\([a-z]+\)\s*"([^"]+)"\s*means`
  - For Hindi: equivalent pattern or LLM-assisted extraction
  - Each definition becomes a chunk with metadata prefix
- Test: extract definitions from Factories Act Section 2
- Verify: each definition is a separate chunk, metadata is correct

**Step 5: Section chunker**
- Write `section_chunker.py`:
  - Traverse DoclingDocument tree using `SectionHeaderItem` hierarchy
  - Group content under each section heading
  - Build heading chain metadata (Chapter → Section → Sub-section)
  - If section > 1500 tokens, split at sub-section boundaries
  - Prepend metadata prefix to each chunk
- Test: extract sections from Factories Act Chapters III-V
- Verify: heading chains are correct, chunks are right-sized, no content lost

**Step 6: Table extractor**
- Write `table_extractor.py`:
  - For English (Docling): iterate `DoclingDocument.tables`, extract `TableItem` objects, convert to SQL rows
  - For Hindi (Mistral HTML): parse HTML tables with BeautifulSoup, handle colspan/rowspan, convert to SQL rows
  - Generate table description via LLM call (or simple heuristic from column headers)
  - Create SQLite table + registry entry + LightRAG stub chunk
- Write `sql_indexer.py`: SQLite connection management, table creation, insertion
- Test: extract First Schedule from Factories Act, any tables from capital_subsidy_Rules
- Verify: SQL tables are queryable, stub chunks describe tables accurately

**Step 7: Amendment extractor**
- Write `amendment_extractor.py`:
  - For inline amendments: regex scan for patterns like `[Substituted by...`, `[Ins. by...`, `[Omitted by...`
  - For standalone amendment docs: full-document LLM one-shot extraction
  - Output: JSON array of amendment records
  - Build LightRAG custom KG edges: amendment entity → target section entity
- Test: extract inline amendments from Factories Act
- Verify: amendments correctly reference target sections, JSON index is accurate

### Phase 3: LightRAG Indexing

**Step 8: LightRAG indexer**
- Write `lightrag_indexer.py`:
  - Initialize LightRAG with custom entity types and customized extraction prompt
  - `index_definitions(chunks)`: call `insert_custom_chunks()` for definition chunks
  - `index_sections(chunks)`: call `insert_custom_chunks()` for section chunks
  - `index_table_stubs(stubs)`: call `insert_custom_chunks()` for table descriptor stubs
  - `index_amendments(amendments)`: call `insert_custom_kg()` for amendment edges
- Test: index all extracted content from both PoC documents
- Verify: query LightRAG with test questions, check KG has correct entity types

**Step 9: Pipeline orchestrator**
- Write `pipeline.py`:
  - `process_document(pdf_path, language)`: runs the full pipeline
    1. Extract (Docling or Mistral+Docling based on language)
    2. Classify elements by content type
    3. Chunk each type
    4. Index into LightRAG + SQLite
  - Logging at each stage (document → extraction → N definitions, M sections, K tables, J amendments)
- Write `scripts/process_poc.py`: process both PoC documents, print summary stats

### Phase 4: Validation

**Step 10: End-to-end validation**
- Run pipeline on both PoC documents
- Test queries against LightRAG:
  - `"What is the definition of factory?"` → should return definition chunk
  - `"What safety provisions apply to hazardous processes?"` → should return Section 41A-41H
  - `"What is the First Schedule?"` → should return stub chunk pointing to SQL
- Test SQL queries:
  - `SELECT * FROM factories_act_first_schedule WHERE industry_name LIKE '%chemical%'`
- Check amendment index:
  - Verify amendments reference correct sections
  - Verify KG edges exist between amendment and target section entities
- Document any issues, gaps, or quality problems for iteration

## Edge Cases to Handle During Implementation

1. **Definitions with sub-clauses:** `(cb) "hazardous process" means... (i)... (ii)...` — keep sub-clauses with parent
2. **Sections referencing tables:** "as specified in the First Schedule" — detect and note the cross-reference
3. **Hindi-English mixed content:** Some docs have both — Mistral handles this natively
4. **Empty/garbled OCR output:** Log and skip, don't index garbage
5. **Tables spanning multiple pages:** Mistral returns per-page output — need to merge tables that continue across pages
6. **Footnote amendments vs body text:** Footnotes with `[Substituted by...]` should be parsed as amendments, not as section content

## Success Criteria for PoC

- [ ] Both documents fully processed without errors
- [ ] Definitions: each definition is a separate, correctly-tagged chunk in LightRAG
- [ ] Sections: hierarchy metadata is correct, chunks are 200-1500 tokens
- [ ] Tables: at least 1 table from each document is in SQLite and queryable
- [ ] Amendments: inline amendments from Factories Act are in the JSON index
- [ ] LightRAG queries return relevant results with correct entity types
- [ ] Hindi text is readable and accurate (spot-check 10 random chunks)
