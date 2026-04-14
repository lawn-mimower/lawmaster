#!/usr/bin/env python3
"""Index the full Factories Act into LightRAG.

Uses:
  - BGE-large local embeddings (dim 1024)
  - Gemini 2.0 Flash for entity extraction
  - max_async=16, gleaning=0 for speed

Usage:
    rm -rf ~/Downloads/LawMaster/rag_storage
    python scripts/05_index_full.py
"""

import sys
import os
import json
import asyncio
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import EXTRACTION_OUTPUT_DIR, RAG_STORAGE_DIR, GEMINI_API_KEY

# --- BGE-large Local Embeddings ---
from sentence_transformers import SentenceTransformer
from lightrag.utils import EmbeddingFunc

print("Loading BGE-large...")
bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

async def bge_embedding_func(texts: list[str]) -> np.ndarray:
    return bge_model.encode(texts, normalize_embeddings=True)


async def main():
    from docling_core.types import DoclingDocument
    from docling_core.transforms.chunker import HierarchicalChunker
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_model_complete

    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    # Load DoclingDocument
    doc_json = EXTRACTION_OUTPUT_DIR / "FactoryAct1948_docling.json"
    with open(doc_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    doc = DoclingDocument.model_validate(data)
    print(f"Loaded: {len(doc.texts)} text elements, {len(doc.tables)} tables")

    # Chunk with hierarchy
    chunker = HierarchicalChunker(merge_list_items=True)
    all_chunks = list(chunker.chunk(doc))

    # Filter out tiny chunks (headers, separators)
    content_chunks = [c for c in all_chunks if len(c.text) > 30]
    print(f"Chunks: {len(all_chunks)} total, {len(content_chunks)} with content")

    # Build enriched texts with metadata
    chunk_texts = []
    for c in content_chunks:
        heading_chain = " > ".join(c.meta.headings) if c.meta.headings else ""
        enriched = f"[Act: Factories Act, 1948] [Location: {heading_chain}]\n\n{c.text}"
        chunk_texts.append(enriched)

    # Join all chunks with separator for single ainsert call
    separator = "\n\n===CHUNK_BOUNDARY===\n\n"
    full_text = separator.join(chunk_texts)
    print(f"Total text: {len(full_text)} chars across {len(chunk_texts)} chunks")

    # Init LightRAG
    RAG_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(RAG_STORAGE_DIR),
        llm_model_func=gemini_model_complete,
        llm_model_name="gemini-2.0-flash",
        llm_model_max_async=16,
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
        entity_extract_max_gleaning=0,
    )
    await rag.initialize_storages()

    # Insert with split_by_character to preserve chunk boundaries
    print(f"\nIndexing {len(chunk_texts)} chunks (max_async=16, gleaning=0)...")
    start = time.time()
    await rag.ainsert(full_text, split_by_character="===CHUNK_BOUNDARY===")
    elapsed = time.time() - start

    # Stats
    graph = rag.chunk_entity_relation_graph._graph
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Sample entities:")
    for node in list(graph.nodes)[:15]:
        ndata = graph.nodes[node]
        print(f"  - {node} ({ndata.get('entity_type', '?')})")


if __name__ == "__main__":
    asyncio.run(main())
