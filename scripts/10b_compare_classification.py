#!/usr/bin/env python3
"""Compare 3 classification approaches on FactoryAct1948:
  A) Regex-only (no annotations)
  B) Kimi K2.5 on Together AI (LLM classification on markdown)
  C) Mistral Annotations (baseline — already ran, load from disk)

All use same OCR output, same indexing, same queries."""

import sys
import os
import json
import asyncio
import time
import re
import shutil
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import (
    EXTRACTION_OUTPUT_DIR, RAG_STORAGE_DIR, GROQ_API_KEY,
    EXTRACTION_MODEL, QUERY_MODEL, TOGETHER_API_KEY, ENGLISH_DOC,
)
from src.chunk.router import route_content

TEST_OUT = EXTRACTION_OUTPUT_DIR / "test_mistral"

QUESTIONS = [
    "What is the definition of 'factory' under the Factories Act?",
    "What safety provisions apply to hazardous processes?",
    "What are the working hour restrictions for adult workers?",
    "What penalties exist for employing children in factories?",
    "What is the permissible exposure limit for Benzene?",
]


async def kimi_classify(markdown: str) -> list[dict]:
    """Classify markdown blocks using Kimi K2.5 on Together AI."""
    import openai

    client = openai.AsyncOpenAI(
        api_key=TOGETHER_API_KEY,
        base_url="https://api.together.xyz/v1",
    )

    prompt = """Classify every distinct content block in this legal document markdown.
For each block, return a JSON object with:
- text_snippet: first 60 characters of the block
- content_type: one of "definition", "section", "table", "amendment", "schedule", "preamble", "footnote", "other"
- section_number: if applicable (e.g. "41B", "2(cb)")
- heading_chain: heading hierarchy (e.g. "Chapter IVA > Section 41B")
- defined_term: for definitions only

Return a JSON object with key "blocks" containing an array of these objects.

DOCUMENT:
""" + markdown[:30000]  # First ~30K chars (fits in context)

    try:
        response = await client.chat.completions.create(
            model="moonshotai/Kimi-K2.5",
            messages=[
                {"role": "system", "content": "You classify legal document content. Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=8000,
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("blocks", [])
    except Exception as e:
        print(f"  Kimi K2.5 classification error: {e}")
        return []


async def build_and_query(label: str, routed: dict, rag_subdir: str):
    """Index routed content and run test queries. Returns answers dict."""
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import openai_complete_if_cache
    from sentence_transformers import SentenceTransformer

    rag_dir = RAG_STORAGE_DIR / rag_subdir
    if rag_dir.exists():
        shutil.rmtree(rag_dir)
    rag_dir.mkdir(parents=True, exist_ok=True)

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
        working_dir=str(rag_dir),
        llm_model_func=groq_func,
        llm_model_name=EXTRACTION_MODEL,
        embedding_func=EmbeddingFunc(embedding_dim=1024, max_token_size=8192, func=bge_func),
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

    # Build chunks
    chunk_texts = []
    for d in routed["definitions"]:
        chunk_texts.append(f"{d['meta']}\n\n{d['text']}")
    for s in routed["sections"]:
        chunk_texts.append(f"{s['meta']}\n\n{s['text']}")
    for t in routed["tables"]:
        chunk_texts.append(t["stub_text"])

    print(f"\n  [{label}] Indexing {len(chunk_texts)} chunks...")
    start = time.time()
    separator = "\n\n===CHUNK_BOUNDARY===\n\n"
    full_text = separator.join(chunk_texts)
    await rag.ainsert(full_text, split_by_character="===CHUNK_BOUNDARY===")
    elapsed = time.time() - start

    graph = rag.chunk_entity_relation_graph._graph
    print(f"  [{label}] {elapsed:.0f}s — {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Query
    answers = {}
    for q in QUESTIONS:
        try:
            r = await rag.aquery(q, param=QueryParam(mode="mix", top_k=5, model_func=groq_query_func))
            answers[q] = str(r)[:300]
        except Exception as e:
            answers[q] = f"ERROR: {e}"

    return answers, graph.number_of_nodes(), graph.number_of_edges()


async def main():
    # Load OCR output (already extracted)
    md_path = TEST_OUT / f"{ENGLISH_DOC.stem}.md"
    if not md_path.exists():
        print(f"ERROR: Run 10a_test_one_doc.py first to extract {ENGLISH_DOC.name}")
        return

    markdown = md_path.read_text(encoding="utf-8")

    html_tables = []
    tables_path = TEST_OUT / f"{ENGLISH_DOC.stem}_tables.json"
    if tables_path.exists():
        with open(tables_path) as f:
            html_tables = json.load(f)

    # Load existing annotations (from previous run)
    annotations = []
    annot_path = TEST_OUT / f"{ENGLISH_DOC.stem}_annotations.json"
    if annot_path.exists():
        with open(annot_path) as f:
            annotations = json.load(f)

    source = ENGLISH_DOC.name
    category = "Factories Act & Rules"

    # ==========================================
    # Test A: Regex-only (no annotations)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"TEST A: REGEX-ONLY CLASSIFICATION")
    print(f"{'='*60}")

    routed_a = route_content(
        markdown=markdown,
        annotations=[],  # No annotations
        html_tables=html_tables,
        source_name=source,
        category=category,
    )
    print(f"  Definitions: {len(routed_a['definitions'])}")
    print(f"  Sections:    {len(routed_a['sections'])}")
    print(f"  Tables:      {len(routed_a['tables'])}")
    print(f"  Amendments:  {len(routed_a['amendments'])}")

    answers_a, nodes_a, edges_a = await build_and_query("A-Regex", routed_a, "test_regex")

    # ==========================================
    # Test B: Kimi K2.5 classification
    # ==========================================
    print(f"\n{'='*60}")
    print(f"TEST B: KIMI K2.5 CLASSIFICATION")
    print(f"{'='*60}")

    print("  Calling Kimi K2.5 for classification...")
    start = time.time()
    kimi_annotations = await kimi_classify(markdown)
    kimi_elapsed = time.time() - start
    print(f"  Got {len(kimi_annotations)} blocks in {kimi_elapsed:.1f}s")

    routed_b = route_content(
        markdown=markdown,
        annotations=kimi_annotations,
        html_tables=html_tables,
        source_name=source,
        category=category,
    )
    print(f"  Definitions: {len(routed_b['definitions'])}")
    print(f"  Sections:    {len(routed_b['sections'])}")
    print(f"  Tables:      {len(routed_b['tables'])}")
    print(f"  Amendments:  {len(routed_b['amendments'])}")

    answers_b, nodes_b, edges_b = await build_and_query("B-Kimi", routed_b, "test_kimi")

    # ==========================================
    # Comparison
    # ==========================================
    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")

    # Load baseline annotations count
    print(f"\n  Routing stats:")
    print(f"  {'':30s} {'A:Regex':>10s} {'B:Kimi':>10s} {'C:Mistral*':>10s}")
    print(f"  {'Definitions':30s} {len(routed_a['definitions']):>10d} {len(routed_b['definitions']):>10d} {'(baseline)':>10s}")
    print(f"  {'Sections':30s} {len(routed_a['sections']):>10d} {len(routed_b['sections']):>10d}")
    print(f"  {'Tables':30s} {len(routed_a['tables']):>10d} {len(routed_b['tables']):>10d}")
    print(f"  {'Amendments':30s} {len(routed_a['amendments']):>10d} {len(routed_b['amendments']):>10d}")

    print(f"\n  KG stats:")
    print(f"  {'':30s} {'A:Regex':>10s} {'B:Kimi':>10s}")
    print(f"  {'Nodes':30s} {nodes_a:>10d} {nodes_b:>10d}")
    print(f"  {'Edges':30s} {edges_a:>10d} {edges_b:>10d}")

    print(f"\n  Query answers (first 200 chars):")
    for q in QUESTIONS:
        print(f"\n  Q: {q}")
        print(f"  A-Regex: {answers_a.get(q, '?')[:200]}...")
        print(f"  B-Kimi:  {answers_b.get(q, '?')[:200]}...")

    # * C:Mistral is the 10a_test_one_doc.py run — already completed separately


if __name__ == "__main__":
    asyncio.run(main())
