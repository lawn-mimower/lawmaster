#!/usr/bin/env python3
"""Full pipeline: Extract ALL 43 PDFs via Mistral OCR 3 → classify → index into LightRAG + SQLite.

Flow:
  PDF → Mistral OCR 3 (text + HTML tables)
      → Mistral Annotations (content type labels, 8-page batches)
      → Router (definitions / sections / tables / amendments)
      → LightRAG (definitions + sections + table stubs + amendment KG)
      → SQLite (table data)

Usage:
    # Fresh index of everything
    python scripts/10_full_pipeline.py --fresh

    # Resume (skip already-extracted docs)
    python scripts/10_full_pipeline.py --resume

    # Extract only (no indexing)
    python scripts/10_full_pipeline.py --extract-only

    # Index only (from existing extractions)
    python scripts/10_full_pipeline.py --index-only
"""

import sys
import os
import json
import asyncio
import sqlite3
import time
import re
import argparse
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import (
    EXTRACTION_OUTPUT_DIR, RAG_STORAGE_DIR, DATA_DIR,
    MISTRAL_API_KEY, GROQ_API_KEY, EXTRACTION_MODEL, DOCS_BASE,
)

MISTRAL_OUT = EXTRACTION_OUTPUT_DIR / "mistral"

CATEGORIES = {
    "factories act": "Factories Act & Rules",
    "pollution control board": "Pollution Control & Environment",
    "industrial policy, ammendments and notifications": "Chhattisgarh Industrial Policy",
}


def get_all_pdfs() -> list[dict]:
    """Discover all PDFs in corpus."""
    pdfs = []
    for pdf_path in sorted(DOCS_BASE.rglob("*.pdf")):
        rel = pdf_path.relative_to(DOCS_BASE)
        category_dir = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        category = CATEGORIES.get(category_dir, category_dir)
        pdfs.append({
            "path": pdf_path,
            "name": pdf_path.stem,
            "category": category,
            "category_dir": category_dir,
        })
    return pdfs


# ============================================================
# Phase 1: Extraction
# ============================================================

def run_extraction(pdfs: list[dict], resume: bool = True):
    """Extract all PDFs via Mistral OCR 3 + Annotations."""
    from src.extract.mistral_extractor import extract_pdf

    print(f"\n{'='*60}")
    print(f"PHASE 1: EXTRACTION ({len(pdfs)} documents)")
    print(f"{'='*60}\n")

    results = []
    start_all = time.time()

    for i, pdf_info in enumerate(pdfs, 1):
        out_dir = MISTRAL_OUT / pdf_info["category_dir"]
        md_path = out_dir / f"{pdf_info['name']}.md"

        # Resume check
        if resume and md_path.exists() and md_path.stat().st_size > 0:
            print(f"[{i}/{len(pdfs)}] SKIP {pdf_info['path'].name}")
            meta_path = out_dir / f"{pdf_info['name']}_meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            results.append({"status": "skipped", **meta})
            continue

        print(f"[{i}/{len(pdfs)}] {pdf_info['path'].name}")
        print(f"         Category: {pdf_info['category']}")

        try:
            result = extract_pdf(
                pdf_path=pdf_info["path"],
                output_dir=out_dir,
                api_key=MISTRAL_API_KEY,
            )
            results.append({"status": "ok", **result["meta"]})
        except Exception as e:
            print(f"         ERROR: {e}")
            results.append({"status": "error", "name": pdf_info["name"], "error": str(e)})

    elapsed = time.time() - start_all
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\nExtraction: {len(ok)} new, {len(skipped)} skipped, {len(errors)} errors in {elapsed:.0f}s")
    if errors:
        for r in errors:
            print(f"  FAIL: {r.get('name', '?')}: {r.get('error', '?')}")

    # Save manifest
    manifest_path = MISTRAL_OUT / "extraction_manifest.json"
    MISTRAL_OUT.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# Phase 2: Indexing
# ============================================================

async def run_indexing(pdfs: list[dict], max_async: int = 4, fresh: bool = False):
    """Index all extracted content into LightRAG + SQLite."""
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache
    from sentence_transformers import SentenceTransformer
    from src.chunk.router import route_content

    print(f"\n{'='*60}")
    print(f"PHASE 2: INDEXING ({len(pdfs)} documents, max_async={max_async})")
    print(f"{'='*60}\n")

    # --- Fresh start ---
    if fresh:
        import shutil
        if RAG_STORAGE_DIR.exists():
            shutil.rmtree(RAG_STORAGE_DIR)
            print(f"Wiped {RAG_STORAGE_DIR}")
        db_path = DATA_DIR / "tables.db"
        if db_path.exists():
            db_path.unlink()
            print(f"Wiped {db_path}")

    # --- Models ---
    print("Loading BGE-large embeddings...")
    bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)

    async def bge_func(texts):
        return bge_model.encode(texts, normalize_embeddings=True)

    def strip_think(text):
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    GROQ_MAX_RETRIES = 5
    GROQ_RETRY_DELAYS = [2, 5, 10, 30, 60]

    async def groq_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        last_exc = None
        for attempt in range(GROQ_MAX_RETRIES):
            try:
                r = await openai_complete_if_cache(
                    model=EXTRACTION_MODEL, prompt=prompt,
                    system_prompt=system_prompt, history_messages=history_messages or [],
                    api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
                    **kwargs,
                )
                return strip_think(r)
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                transient = any(kw in msg for kw in (
                    "429", "rate_limit", "rate limit", "500", "502", "503",
                    "504", "timeout", "connection", "service unavailable",
                ))
                if not transient:
                    raise
                delay = GROQ_RETRY_DELAYS[min(attempt, len(GROQ_RETRY_DELAYS) - 1)]
                print(f"    [GROQ RETRY] attempt {attempt+1}/{GROQ_MAX_RETRIES}, "
                      f"waiting {delay}s — {e}")
                await asyncio.sleep(delay)
        raise last_exc

    # --- Init LightRAG ---
    RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(RAG_STORAGE_DIR),
        llm_model_func=groq_func,
        llm_model_name=EXTRACTION_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192, func=bge_func,
        ),
        addon_params={
            "entity_types": [
                "Definition", "Section", "Amendment", "Schedule",
                "Act", "Rule", "Authority", "Penalty", "Provision",
                "Industry", "Chemical", "Regulation", "Notification",
            ],
            "language": "English",
        },
        chunk_token_size=1500,
        chunk_overlap_token_size=100,
        entity_extract_max_gleaning=0,
        llm_model_max_async=max_async,
    )
    await rag.initialize_storages()

    graph = rag.chunk_entity_relation_graph._graph
    nodes_start = graph.number_of_nodes()
    edges_start = graph.number_of_edges()
    print(f"Graph start: {nodes_start} nodes, {edges_start} edges")

    # --- Init SQLite ---
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / "tables.db"
    conn = sqlite3.connect(str(db_path))
    _init_table_registry(conn)

    # --- Process each doc ---
    start_all = time.time()
    results = []

    for i, pdf_info in enumerate(pdfs, 1):
        out_dir = MISTRAL_OUT / pdf_info["category_dir"]
        md_path = out_dir / f"{pdf_info['name']}.md"

        if not md_path.exists():
            print(f"[{i}/{len(pdfs)}] SKIP {pdf_info['name']} (not extracted)")
            continue

        print(f"\n[{i}/{len(pdfs)}] {pdf_info['name']}")

        # Load extraction outputs
        markdown = md_path.read_text(encoding="utf-8")

        annotations = []
        annot_path = out_dir / f"{pdf_info['name']}_annotations.json"
        if annot_path.exists():
            with open(annot_path) as f:
                annotations = json.load(f)

        html_tables = []
        tables_path = out_dir / f"{pdf_info['name']}_tables.json"
        if tables_path.exists():
            with open(tables_path) as f:
                html_tables = json.load(f)

        # Route content
        routed = route_content(
            markdown=markdown,
            annotations=annotations,
            html_tables=html_tables,
            source_name=pdf_info["path"].name,
            category=pdf_info["category"],
        )

        n_defs = len(routed["definitions"])
        n_secs = len(routed["sections"])
        n_tbls = len(routed["tables"])
        n_other = len(routed.get("other", []))
        print(f"  Routed: {n_defs} defs, {n_secs} sections, {n_tbls} tables, {n_other} other")

        # --- Insert definitions + sections + other into LightRAG ---
        # Amendments stay inline in section chunks; KG captures them via ER extraction
        chunk_texts = []

        for d in routed["definitions"]:
            chunk_texts.append(f"{d['meta']}\n\n{d['text']}")

        for s in routed["sections"]:
            chunk_texts.append(f"{s['meta']}\n\n{s['text']}")

        for o in routed.get("other", []):
            chunk_texts.append(f"{o['meta']}\n\n{o['text']}")

        # Table stubs — only include if the table will also be inserted into SQL
        for t in routed["tables"]:
            if t.get("rows") and len(t["rows"]) >= 2:
                chunk_texts.append(t["stub_text"])

        if chunk_texts:
            separator = "\n\n===CHUNK_BOUNDARY===\n\n"
            full_text = separator.join(chunk_texts)

            doc_start = time.time()
            try:
                await rag.ainsert(full_text, split_by_character="===CHUNK_BOUNDARY===")
                doc_elapsed = time.time() - doc_start

                nodes_now = graph.number_of_nodes()
                edges_now = graph.number_of_edges()
                print(f"  Indexed {len(chunk_texts)} chunks in {doc_elapsed:.1f}s "
                      f"→ {nodes_now} nodes, {edges_now} edges")

                results.append({
                    "name": pdf_info["name"],
                    "status": "ok",
                    "defs": n_defs, "sections": n_secs,
                    "tables": n_tbls, "other": n_other,
                    "chunks": len(chunk_texts),
                    "elapsed_s": round(doc_elapsed, 1),
                })
            except Exception as e:
                print(f"  INDEX ERROR: {e}")
                results.append({"name": pdf_info["name"], "status": "error", "error": str(e)})
        else:
            print(f"  No chunks to index")
            results.append({"name": pdf_info["name"], "status": "empty"})

        # --- Insert tables into SQLite ---
        for t in routed["tables"]:
            _insert_table_to_sql(conn, t, pdf_info)

    conn.close()

    # --- Summary ---
    elapsed_all = time.time() - start_all
    nodes_end = graph.number_of_nodes()
    edges_end = graph.number_of_edges()

    ok = [r for r in results if r.get("status") == "ok"]

    print(f"\n{'='*60}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*60}")
    print(f"  Documents: {len(ok)}/{len(pdfs)} indexed")
    print(f"  Total chunks: {sum(r.get('chunks', 0) for r in ok)}")
    print(f"  Definitions: {sum(r.get('defs', 0) for r in ok)}")
    print(f"  Sections: {sum(r.get('sections', 0) for r in ok)}")
    print(f"  Tables: {sum(r.get('tables', 0) for r in ok)} → SQLite")
    print(f"  Graph: {nodes_start} → {nodes_end} nodes (+{nodes_end - nodes_start})")
    print(f"  Graph: {edges_start} → {edges_end} edges (+{edges_end - edges_start})")
    print(f"  Time: {elapsed_all:.0f}s ({elapsed_all/60:.1f}min)")

    errors = [r for r in results if r.get("status") == "error"]
    if errors:
        print(f"\n  Errors:")
        for r in errors:
            print(f"    {r['name']}: {r.get('error', '?')}")

    # Save results
    results_path = MISTRAL_OUT / "index_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "nodes": nodes_end, "edges": edges_end,
            "elapsed_s": round(elapsed_all, 1),
            "docs": results,
        }, f, indent=2)


def _init_table_registry(conn):
    """Create table registry if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _table_registry (
            table_id TEXT PRIMARY KEY,
            source_document TEXT NOT NULL,
            table_name TEXT NOT NULL,
            description TEXT,
            page_number INTEGER,
            num_rows INTEGER,
            num_cols INTEGER,
            column_names TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def _insert_table_to_sql(conn, table_data: dict, pdf_info: dict):
    """Insert an extracted table into SQLite."""
    if not table_data.get("rows") or len(table_data["rows"]) < 2:
        return

    headers = table_data.get("headers", [])
    rows = table_data["rows"]
    page = table_data.get("page", 0)

    # Use pre-computed name from router if available, else compute it
    table_name = table_data.get("sql_table_name")
    if not table_name:
        safe_name = re.sub(r'[^a-z0-9_]', '_', pdf_info["name"].lower())
        table_name = f"{safe_name}_p{page}"

    # Clean column names
    col_names = []
    for j, h in enumerate(headers):
        clean = re.sub(r'[^a-z0-9_]', '_', h.lower().strip()) if h.strip() else f"col_{j}"
        clean = clean.strip('_') or f"col_{j}"
        col_names.append(clean)

    if not col_names:
        col_names = [f"col_{j}" for j in range(table_data.get("num_cols", 0))]

    if not col_names:
        return

    # Create table
    col_defs = ", ".join(f'"{c}" TEXT' for c in col_names)
    try:
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.execute(f'CREATE TABLE "{table_name}" ({col_defs}, _source TEXT, _page INTEGER)')

        # Insert data rows (skip header row)
        for row in rows[1:]:
            if all(not cell.strip() for cell in row):
                continue
            # Pad/truncate row to match columns
            padded = (row + [""] * len(col_names))[:len(col_names)]
            placeholders = ", ".join(["?"] * (len(col_names) + 2))
            conn.execute(
                f'INSERT INTO "{table_name}" VALUES ({placeholders})',
                padded + [pdf_info["path"].name, page],
            )

        # Register
        conn.execute(
            "INSERT OR REPLACE INTO _table_registry VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
            (table_name, pdf_info["path"].name, table_name,
             table_data.get("description", ""), page,
             len(rows) - 1, len(col_names),
             json.dumps(col_names)),
        )
        conn.commit()
        print(f"  SQL: {table_name} ({len(rows)-1} rows, {len(col_names)} cols)")

    except Exception as e:
        print(f"  SQL ERROR: {table_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="LawMaster full indexing pipeline")
    parser.add_argument("--fresh", action="store_true", help="Wipe all storage and start fresh")
    parser.add_argument("--resume", action="store_true", help="Skip already-extracted docs")
    parser.add_argument("--extract-only", action="store_true", help="Only run extraction, no indexing")
    parser.add_argument("--index-only", action="store_true", help="Only run indexing from existing extractions")
    parser.add_argument("--max-async", type=int, default=4, help="LLM concurrency for indexing")
    args = parser.parse_args()

    pdfs = get_all_pdfs()
    print(f"Corpus: {len(pdfs)} documents")

    if not args.index_only:
        run_extraction(pdfs, resume=args.resume or not args.fresh)

    if not args.extract_only:
        asyncio.run(run_indexing(pdfs, max_async=args.max_async, fresh=args.fresh))


if __name__ == "__main__":
    main()
