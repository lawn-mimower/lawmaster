#!/usr/bin/env python3
"""Quick test: index just 20 chunks into LightRAG to validate the pipeline.

Uses Qwen3 local embeddings + Gemini Flash LLM (same pattern as lightrag-bench).

Usage:
    rm -rf ~/Downloads/LawMaster/rag_storage_test
    python scripts/03b_index_quick_test.py
"""

import sys
import os
import json
import asyncio
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXTRACTION_OUTPUT_DIR, PROJECT_ROOT, GEMINI_API_KEY

# --- BGE-large Local Embeddings ---
from sentence_transformers import SentenceTransformer

print("Loading BAAI/bge-large-en-v1.5...")
bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

async def bge_embedding_func(texts: list[str]) -> np.ndarray:
    return bge_model.encode(texts, normalize_embeddings=True)


async def main():
    from docling_core.types import DoclingDocument
    from docling_core.transforms.chunker import HierarchicalChunker
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.gemini import gemini_model_complete

    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    # Load DoclingDocument
    doc_json = EXTRACTION_OUTPUT_DIR / "FactoryAct1948_docling.json"
    with open(doc_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc = DoclingDocument.model_validate(data)

    # Chunk
    chunker = HierarchicalChunker(merge_list_items=True)
    all_chunks = list(chunker.chunk(doc))

    # Pick 20 chunks with real legal content (skip TOC/headers)
    selected = [c for c in all_chunks if len(c.text) > 100][:20]
    print(f"Selected {len(selected)} chunks (from {len(all_chunks)} total)")

    for i, c in enumerate(selected):
        headings = " > ".join(c.meta.headings) if c.meta.headings else "none"
        print(f"  {i}: [{headings[:50]}] {c.text[:60]}...")

    # Build enriched texts
    chunk_texts = []
    for c in selected:
        heading_chain = " > ".join(c.meta.headings) if c.meta.headings else ""
        enriched = f"[Act: Factories Act, 1948] [Location: {heading_chain}]\n\n{c.text}"
        chunk_texts.append(enriched)

    full_text = "\n\n---\n\n".join(chunk_texts)

    # Init LightRAG with Qwen3 embeddings + Gemini LLM
    working_dir = PROJECT_ROOT / "rag_storage_test"
    working_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=gemini_model_complete,
        llm_model_name="gemini-2.0-flash",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=bge_embedding_func,
        ),
        addon_params={
            "entity_types": [
                "Definition", "Section", "Amendment", "Schedule",
                "Act", "Rule", "Authority", "Penalty", "Provision",
            ],
            "language": "English",
        },
        chunk_token_size=1500,
        chunk_overlap_token_size=100,
    )
    await rag.initialize_storages()

    # Insert using ainsert (ainsert_custom_chunks has a bug — skips merge_nodes_and_edges)
    # Insert each chunk as a separate document so LightRAG handles full pipeline
    print(f"\nInserting {len(chunk_texts)} chunks...")
    for i, ct in enumerate(chunk_texts):
        print(f"  [{i+1}/{len(chunk_texts)}] {ct[:60]}...")
        await rag.ainsert(ct)
    print("Insert done!")

    # Check graph
    graph = rag.chunk_entity_relation_graph._graph
    print(f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    if graph.number_of_nodes() > 0:
        print("\nSample entities:")
        for node in list(graph.nodes)[:10]:
            ndata = graph.nodes[node]
            print(f"  {node} (type: {ndata.get('entity_type', '?')})")

    # Test query
    print("\n--- Test Query ---")
    result = await rag.aquery(
        "What is the definition of factory?",
        param=QueryParam(mode="mix", top_k=3),
    )
    print(f"Q: What is the definition of factory?")
    print(f"A: {result[:500]}...")


if __name__ == "__main__":
    asyncio.run(main())
