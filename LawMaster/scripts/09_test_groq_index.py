#!/usr/bin/env python3
"""Test Groq-based indexing with a small document (Hazardous Industries list, 2 pages).

Usage:
    python scripts/09_test_groq_index.py
"""

import sys
import os
import asyncio
import time
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import EXTRACTION_OUTPUT_DIR, RAG_STORAGE_DIR, GROQ_API_KEY, EXTRACTION_MODEL, QUERY_MODEL

DOCS_BASE = Path("/home/pc/Downloads/LawMaster/project 2 _ai tool for compliance -20260331T205330Z-1-001/project 2 _ai tool for compliance ")
TEST_DOC = DOCS_BASE / "factories act" / "List of Industries involving hazardous processes _ DGFASLI, Mumbai, Ministry of Labour, Government of India_.pdf"


async def main():
    from src.extract.docling_extractor import extract_pdf_with_docling
    from docling_core.types import DoclingDocument
    from docling_core.transforms.chunker import HierarchicalChunker
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache
    from sentence_transformers import SentenceTransformer
    import json, re

    print(f"=== Test: Groq Qwen3 32B extraction ===")
    print(f"Model: {EXTRACTION_MODEL}")
    print(f"Document: {TEST_DOC.name}")

    # Step 1: Extract with Docling
    print(f"\n[1/3] Extracting with Docling...")
    result = extract_pdf_with_docling(TEST_DOC, EXTRACTION_OUTPUT_DIR)
    print(f"  {result['stats']['num_texts']} texts, {result['stats']['num_tables']} tables")

    # Step 2: Chunk
    print(f"\n[2/3] Chunking...")
    with open(result["doc_json_path"], "r") as f:
        doc = DoclingDocument.model_validate(json.load(f))

    chunker = HierarchicalChunker(merge_list_items=True)
    all_chunks = list(chunker.chunk(doc))
    content_chunks = [c for c in all_chunks if len(c.text) > 30]
    print(f"  {len(content_chunks)} chunks (from {len(all_chunks)} total)")

    # Build enriched texts
    chunk_texts = []
    for c in content_chunks:
        heading_chain = " > ".join(c.meta.headings) if c.meta.headings else ""
        enriched = f"[Source: First Schedule, Factories Act 1948] [Location: {heading_chain}]\n\n{c.text}"
        chunk_texts.append(enriched)

    # Step 3: Index with Groq
    print(f"\n[3/3] Indexing with Groq ({EXTRACTION_MODEL})...")

    bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)
    async def bge_func(texts):
        return bge_model.encode(texts, normalize_embeddings=True)

    def strip_think(text):
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    async def groq_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        r = await openai_complete_if_cache(
            model=EXTRACTION_MODEL, prompt=prompt,
            system_prompt=system_prompt, history_messages=history_messages or [],
            api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )
        return strip_think(r)

    async def groq_query_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        """Llama 3.3 70B for queries — patches keyword_extraction to json_object mode."""
        if kwargs.pop("keyword_extraction", False):
            kwargs["response_format"] = {"type": "json_object"}
        return await openai_complete_if_cache(
            model=QUERY_MODEL, prompt=prompt,
            system_prompt=system_prompt, history_messages=history_messages or [],
            api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )

    rag = LightRAG(
        working_dir=str(RAG_STORAGE_DIR),
        llm_model_func=groq_func,
        llm_model_name=EXTRACTION_MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=bge_func),
        addon_params={
            "entity_types": ["Definition", "Section", "Amendment", "Schedule", "Act", "Rule", "Authority", "Penalty", "Provision"],
            "language": "English",
        },
        entity_extract_max_gleaning=0,
        llm_model_max_async=4,
    )
    await rag.initialize_storages()

    nodes_before = rag.chunk_entity_relation_graph._graph.number_of_nodes()
    edges_before = rag.chunk_entity_relation_graph._graph.number_of_edges()
    print(f"  Before: {nodes_before} nodes, {edges_before} edges")

    separator = "\n\n===CHUNK_BOUNDARY===\n\n"
    full_text = separator.join(chunk_texts)

    start = time.time()
    await rag.ainsert(full_text, split_by_character="===CHUNK_BOUNDARY===")
    elapsed = time.time() - start

    nodes_after = rag.chunk_entity_relation_graph._graph.number_of_nodes()
    edges_after = rag.chunk_entity_relation_graph._graph.number_of_edges()

    print(f"\n=== Results ===")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  New nodes: {nodes_after - nodes_before}")
    print(f"  New edges: {edges_after - edges_before}")
    print(f"  Total: {nodes_after} nodes, {edges_after} edges")

    # Quick test query
    print(f"\n=== Test Query ===")
    r = await rag.aquery("What industries involve hazardous processes?", param=QueryParam(mode="mix", top_k=5, model_func=groq_query_func))
    print(f"Q: What industries involve hazardous processes?")
    print(f"A: {r[:400]}...")


if __name__ == "__main__":
    asyncio.run(main())
