#!/usr/bin/env python3
"""Phase 2: Index ALL extracted markdowns into LightRAG.

Reads markdown files from extraction_output/mistral/, chunks by markdown
headings, enriches with source metadata, and indexes via Groq Qwen3 32B.

Prerequisites:
    - Run 10_extract_all_mistral.py first
    - Clear old RAG storage: rm -rf rag_storage/

Usage:
    python scripts/11_index_all.py
    python scripts/11_index_all.py --resume          # skip already-indexed docs
    python scripts/11_index_all.py --max-async 8     # control concurrency
"""

import sys
import os
import json
import asyncio
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

from src.config import RAG_STORAGE_DIR, GROQ_API_KEY, EXTRACTION_MODEL, EXTRACTION_OUTPUT_DIR

MISTRAL_OUTPUT = EXTRACTION_OUTPUT_DIR / "mistral"

# Category metadata for enrichment
CATEGORIES = {
    "factories act": "Factories Act & Rules",
    "pollution control board": "Pollution Control & Environment",
    "industrial policy, ammendments and notifications": "Chhattisgarh Industrial Policy",
}


def chunk_markdown(text: str, max_chunk_chars: int = 3000, min_chunk_chars: int = 50) -> list[dict]:
    """Split markdown into chunks by headings, preserving heading hierarchy.

    Returns list of dicts with 'text' and 'heading_chain' keys.
    """
    # Split by markdown headings (## or ###, keep # as doc-level)
    # Pattern: split before any heading line
    parts = re.split(r'(?=^#{1,4}\s)', text, flags=re.MULTILINE)

    chunks = []
    current_headings = {}  # level -> heading text

    for part in parts:
        part = part.strip()
        if not part or len(part) < min_chunk_chars:
            # Too small — merge with previous if possible
            if chunks and len(part) > 10:
                chunks[-1]["text"] += "\n\n" + part
            continue

        # Extract heading if this part starts with one
        heading_match = re.match(r'^(#{1,4})\s+(.+?)$', part, re.MULTILINE)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            current_headings[level] = heading_text
            # Clear deeper headings
            for l in list(current_headings.keys()):
                if l > level:
                    del current_headings[l]

        # Build heading chain
        heading_chain = " > ".join(
            current_headings[l] for l in sorted(current_headings.keys())
        )

        # If chunk is too large, split by paragraphs
        if len(part) > max_chunk_chars:
            paragraphs = re.split(r'\n\n+', part)
            sub_chunk = ""
            for para in paragraphs:
                if len(sub_chunk) + len(para) > max_chunk_chars and len(sub_chunk) > min_chunk_chars:
                    chunks.append({"text": sub_chunk.strip(), "heading_chain": heading_chain})
                    sub_chunk = para
                else:
                    sub_chunk += "\n\n" + para if sub_chunk else para
            if sub_chunk.strip() and len(sub_chunk.strip()) > min_chunk_chars:
                chunks.append({"text": sub_chunk.strip(), "heading_chain": heading_chain})
        else:
            chunks.append({"text": part, "heading_chain": heading_chain})

    # Also split by page breaks (---) if no headings produced chunks
    if len(chunks) <= 1 and "---" in text:
        page_parts = text.split("\n\n---\n\n")
        chunks = []
        for i, page in enumerate(page_parts):
            page = page.strip()
            if len(page) > min_chunk_chars:
                chunks.append({"text": page, "heading_chain": f"Page {i+1}"})

    return chunks


def discover_docs() -> list[dict]:
    """Find all extracted markdown files with metadata."""
    docs = []
    for category_dir in sorted(MISTRAL_OUTPUT.iterdir()):
        if not category_dir.is_dir():
            continue
        category = CATEGORIES.get(category_dir.name, category_dir.name)
        for md_path in sorted(category_dir.glob("*.md")):
            meta_path = md_path.with_suffix("").with_name(md_path.stem + "_meta.json")
            meta = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            docs.append({
                "md_path": md_path,
                "name": md_path.stem,
                "category": category,
                "category_dir": category_dir.name,
                "source": meta.get("source", md_path.stem + ".pdf"),
                "pages": meta.get("pages", 0),
            })
    return docs


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Skip docs already in the index")
    parser.add_argument("--max-async", type=int, default=4, help="LLM concurrency (default: 4)")
    parser.add_argument("--fresh", action="store_true", help="Wipe RAG storage and start fresh")
    args = parser.parse_args()

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache
    from sentence_transformers import SentenceTransformer

    # --- Models ---
    print("[1/4] Loading BGE-large embeddings...")
    bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)

    async def bge_func(texts):
        return bge_model.encode(texts, normalize_embeddings=True)

    def strip_think(text):
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    async def groq_extract_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        r = await openai_complete_if_cache(
            model=EXTRACTION_MODEL, prompt=prompt,
            system_prompt=system_prompt, history_messages=history_messages or [],
            api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )
        return strip_think(r)

    # --- Discover docs ---
    print("[2/4] Discovering extracted documents...")
    docs = discover_docs()
    if not docs:
        print(f"  No markdown files found in {MISTRAL_OUTPUT}")
        print(f"  Run 10_extract_all_mistral.py first.")
        return

    print(f"  Found {len(docs)} documents")
    total_pages = sum(d["pages"] for d in docs)
    print(f"  Total pages: {total_pages}")

    # --- Fresh start ---
    if args.fresh:
        import shutil
        if RAG_STORAGE_DIR.exists():
            shutil.rmtree(RAG_STORAGE_DIR)
            print(f"  Wiped {RAG_STORAGE_DIR}")

    # --- Init LightRAG ---
    print(f"[3/4] Initializing LightRAG (max_async={args.max_async})...")
    RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(RAG_STORAGE_DIR),
        llm_model_func=groq_extract_func,
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
        llm_model_max_async=args.max_async,
    )
    await rag.initialize_storages()

    nodes_start = rag.chunk_entity_relation_graph._graph.number_of_nodes()
    edges_start = rag.chunk_entity_relation_graph._graph.number_of_edges()
    print(f"  Graph state: {nodes_start} nodes, {edges_start} edges")

    # --- Index each doc ---
    print(f"\n[4/4] Indexing {len(docs)} documents...\n")
    start_all = time.time()
    results = []

    for i, doc_info in enumerate(docs, 1):
        md_path = doc_info["md_path"]
        name = doc_info["name"]
        category = doc_info["category"]
        source = doc_info["source"]

        print(f"[{i}/{len(docs)}] {source}")
        print(f"       Category: {category}")

        # Read markdown
        text = md_path.read_text(encoding="utf-8")
        if len(text) < 50:
            print(f"       [SKIP] Too short ({len(text)} chars)")
            results.append({"name": name, "status": "skipped", "reason": "too_short"})
            continue

        # Chunk
        chunks = chunk_markdown(text)
        if not chunks:
            print(f"       [SKIP] No chunks produced")
            results.append({"name": name, "status": "skipped", "reason": "no_chunks"})
            continue

        # Enrich with metadata
        chunk_texts = []
        for c in chunks:
            location = c["heading_chain"]
            enriched = f"[Source: {source}] [Category: {category}]"
            if location:
                enriched += f" [Location: {location}]"
            enriched += f"\n\n{c['text']}"
            chunk_texts.append(enriched)

        print(f"       {len(chunk_texts)} chunks, {len(text)} chars")

        # Index
        separator = "\n\n===CHUNK_BOUNDARY===\n\n"
        full_text = separator.join(chunk_texts)

        doc_start = time.time()
        try:
            await rag.ainsert(full_text, split_by_character="===CHUNK_BOUNDARY===")
            doc_elapsed = time.time() - doc_start

            nodes_now = rag.chunk_entity_relation_graph._graph.number_of_nodes()
            edges_now = rag.chunk_entity_relation_graph._graph.number_of_edges()

            print(f"       Done in {doc_elapsed:.1f}s — "
                  f"graph: {nodes_now} nodes, {edges_now} edges")

            results.append({
                "name": name,
                "status": "ok",
                "chunks": len(chunk_texts),
                "chars": len(text),
                "elapsed_s": round(doc_elapsed, 1),
                "nodes": nodes_now,
                "edges": edges_now,
            })
        except Exception as e:
            doc_elapsed = time.time() - doc_start
            print(f"       [ERROR] {e}")
            results.append({
                "name": name,
                "status": "error",
                "error": str(e),
                "elapsed_s": round(doc_elapsed, 1),
            })

    elapsed_all = time.time() - start_all

    # --- Summary ---
    nodes_end = rag.chunk_entity_relation_graph._graph.number_of_nodes()
    edges_end = rag.chunk_entity_relation_graph._graph.number_of_edges()

    ok = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*55}")
    print(f"INDEXING COMPLETE")
    print(f"{'='*55}")
    print(f"  Documents indexed: {len(ok)}/{len(docs)}")
    print(f"  Total chunks: {sum(r.get('chunks', 0) for r in ok)}")
    print(f"  Graph: {nodes_start} → {nodes_end} nodes (+{nodes_end - nodes_start})")
    print(f"  Graph: {edges_start} → {edges_end} edges (+{edges_end - edges_start})")
    print(f"  Time: {elapsed_all:.0f}s ({elapsed_all/60:.1f}min)")

    if errors:
        print(f"\n  Failed docs:")
        for r in errors:
            print(f"    - {r['name']}: {r.get('error', '?')}")

    # Save results
    results_path = MISTRAL_OUTPUT / "index_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "total_docs": len(docs),
            "indexed": len(ok),
            "errors": len(errors),
            "nodes": nodes_end,
            "edges": edges_end,
            "elapsed_s": round(elapsed_all, 1),
            "docs": results,
        }, f, indent=2)
    print(f"\n  Results: {results_path}")


if __name__ == "__main__":
    asyncio.run(main())
