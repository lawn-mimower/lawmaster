#!/usr/bin/env python3
"""Test the full Mistral pipeline on ONE document: FactoryAct1948.pdf
Extract → Classify → Route → Index → Query"""

import sys
import os
import json
import asyncio
import sqlite3
import time
import re
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import (
    EXTRACTION_OUTPUT_DIR, RAG_STORAGE_DIR, DATA_DIR,
    MISTRAL_API_KEY, GROQ_API_KEY, EXTRACTION_MODEL, QUERY_MODEL,
    ENGLISH_DOC,
)

TEST_OUT = EXTRACTION_OUTPUT_DIR / "test_mistral"


async def main():
    from src.extract.mistral_extractor import extract_pdf
    from src.chunk.router import route_content
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache
    from sentence_transformers import SentenceTransformer

    # ========== Phase 1: Extract ==========
    print(f"{'='*60}")
    print(f"TEST: {ENGLISH_DOC.name}")
    print(f"{'='*60}\n")

    # Always re-extract (annotations may have failed partially)
    result = extract_pdf(
        pdf_path=ENGLISH_DOC,
        output_dir=TEST_OUT,
        api_key=MISTRAL_API_KEY,
        annotate=True,
    )
    markdown = result["markdown"]
    annotations = result["annotations"]
    html_tables = result["html_tables"]
    meta = result["meta"]

    # ========== Phase 2: Route ==========
    print(f"\n[Route] Classifying content...")
    routed = route_content(
        markdown=markdown,
        annotations=annotations,
        html_tables=html_tables,
        source_name=ENGLISH_DOC.name,
        category="Factories Act & Rules",
    )

    print(f"  Definitions: {len(routed['definitions'])}")
    print(f"  Sections:    {len(routed['sections'])}")
    print(f"  Tables:      {len(routed['tables'])}")
    print(f"  Amendments:  {len(routed['amendments'])}")

    # Show sample definitions
    if routed["definitions"]:
        print(f"\n  Sample definitions:")
        for d in routed["definitions"][:5]:
            term = d.get("term", "?")
            text_preview = d["text"][:80].replace("\n", " ")
            print(f"    [{term}] {text_preview}...")

    # Show sample sections
    if routed["sections"]:
        print(f"\n  Sample sections:")
        for s in routed["sections"][:5]:
            num = s.get("number", "?")
            text_preview = s["text"][:80].replace("\n", " ")
            print(f"    [S.{num}] {text_preview}...")

    # ========== Phase 3: Index ==========
    print(f"\n[Index] Setting up LightRAG...")

    # Clean test RAG storage
    test_rag_dir = RAG_STORAGE_DIR / "test_mistral"
    import shutil
    if test_rag_dir.exists():
        shutil.rmtree(test_rag_dir)
    test_rag_dir.mkdir(parents=True, exist_ok=True)

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
        if kwargs.pop("keyword_extraction", False):
            kwargs["response_format"] = {"type": "json_object"}
        return await openai_complete_if_cache(
            model=QUERY_MODEL, prompt=prompt,
            system_prompt=system_prompt, history_messages=history_messages or [],
            api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1",
            **kwargs,
        )

    rag = LightRAG(
        working_dir=str(test_rag_dir),
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
        llm_model_max_async=8,
    )
    await rag.initialize_storages()

    # Build chunk texts
    chunk_texts = []
    for d in routed["definitions"]:
        chunk_texts.append(f"{d['meta']}\n\n{d['text']}")
    for s in routed["sections"]:
        chunk_texts.append(f"{s['meta']}\n\n{s['text']}")
    for t in routed["tables"]:
        chunk_texts.append(t["stub_text"])

    print(f"  Indexing {len(chunk_texts)} chunks (max_async=8)...")
    start = time.time()
    separator = "\n\n===CHUNK_BOUNDARY===\n\n"
    full_text = separator.join(chunk_texts)
    await rag.ainsert(full_text, split_by_character="===CHUNK_BOUNDARY===")
    elapsed = time.time() - start

    graph = rag.chunk_entity_relation_graph._graph
    print(f"  Done in {elapsed:.1f}s — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # ========== Phase 4: Query ==========
    print(f"\n{'='*60}")
    print(f"TEST QUERIES")
    print(f"{'='*60}\n")

    questions = [
        "What is the definition of 'factory' under the Factories Act?",
        "What safety provisions apply to hazardous processes?",
        "What are the working hour restrictions for adult workers?",
        "What penalties exist for employing children in factories?",
        "What is the permissible exposure limit for Benzene?",
    ]

    for q in questions:
        print(f"Q: {q}")
        try:
            r = await rag.aquery(
                q, param=QueryParam(mode="mix", top_k=5, model_func=groq_query_func)
            )
            print(f"A: {str(r)[:300]}...\n")
        except Exception as e:
            print(f"ERROR: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
