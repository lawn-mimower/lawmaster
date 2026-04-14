#!/usr/bin/env python3
"""Index Docling-extracted documents into LightRAG using Gemini backend.

Usage:
    python scripts/03_index_to_lightrag.py

Requires GEMINI_API_KEY in .env

Reads from extraction_output/:
    - FactoryAct1948_docling.json (DoclingDocument)

Indexes into rag_storage/ with:
    - Custom legal entity types
    - HierarchicalChunker for structure-aware chunks
    - Metadata-enriched chunk content
"""

import sys
import os
import json
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXTRACTION_OUTPUT_DIR, RAG_STORAGE_DIR, GEMINI_API_KEY


def load_docling_document(json_path: Path):
    """Load a DoclingDocument from exported JSON."""
    from docling_core.types import DoclingDocument
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return DoclingDocument.model_validate(data)


def chunk_with_hierarchy(doc) -> list[dict]:
    """Use Docling's HierarchicalChunker to create structure-aware chunks."""
    from docling_core.transforms.chunker import HierarchicalChunker

    chunker = HierarchicalChunker(
        merge_list_items=True,
    )

    chunks = []
    for i, chunk in enumerate(chunker.chunk(doc)):
        heading_chain = " > ".join(chunk.meta.headings) if chunk.meta.headings else ""
        doc_items = chunk.meta.doc_items if chunk.meta.doc_items else []

        # Classify chunk type
        chunk_type = "Section"
        text_lower = chunk.text.lower()
        if any(h.lower() in ("interpretation", "definitions", "preliminary")
               for h in (chunk.meta.headings or [])):
            if '"' in chunk.text and "means" in text_lower:
                chunk_type = "Definition"
        if chunk.meta.doc_items:
            for item in chunk.meta.doc_items:
                if hasattr(item, 'label') and 'table' in str(getattr(item, 'label', '')).lower():
                    chunk_type = "Table"
                    break

        # Build metadata-enriched content
        metadata_prefix = f"[Act: Factories Act, 1948] [Type: {chunk_type}]"
        if heading_chain:
            metadata_prefix += f" [Location: {heading_chain}]"

        enriched_text = f"{metadata_prefix}\n\n{chunk.text}"

        chunks.append({
            "index": i,
            "type": chunk_type,
            "headings": chunk.meta.headings or [],
            "heading_chain": heading_chain,
            "text": chunk.text,
            "enriched_text": enriched_text,
            "char_count": len(chunk.text),
        })

    return chunks


async def index_chunks(chunks: list[dict], working_dir: Path):
    """Insert chunks into LightRAG with Gemini backend."""
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_model_complete, gemini_embed

    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    working_dir.mkdir(parents=True, exist_ok=True)

    # gemini_embed is pre-wrapped as EmbeddingFunc with:
    #   dim=1536, max_token=2048, model=gemini-embedding-001
    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=gemini_model_complete,
        llm_model_name="gemini-2.0-flash",
        embedding_func=gemini_embed,
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

    # Prepare chunk texts for insertion
    chunk_texts = [c["enriched_text"] for c in chunks]
    full_text = "\n\n---\n\n".join(chunk_texts)

    print(f"[LightRAG] Inserting {len(chunks)} chunks ({len(full_text)} chars)...")
    print(f"[LightRAG] Chunk types: ", end="")
    type_counts = {}
    for c in chunks:
        type_counts[c["type"]] = type_counts.get(c["type"], 0) + 1
    print(", ".join(f"{t}: {n}" for t, n in type_counts.items()))

    # Use insert_custom_chunks to preserve our chunk boundaries
    await rag.ainsert_custom_chunks(full_text, chunk_texts)

    print(f"[LightRAG] Indexing complete. Storage at: {working_dir}")
    return rag


async def test_queries(rag):
    """Run a few test queries to validate indexing."""
    from lightrag import QueryParam

    test_questions = [
        "What is the definition of factory?",
        "What safety provisions apply to hazardous processes?",
        "What are the penalties for offences under this Act?",
    ]

    print(f"\n{'='*60}")
    print("TEST QUERIES")
    print(f"{'='*60}")

    for q in test_questions:
        print(f"\nQ: {q}")
        print("-" * 40)
        try:
            result = await rag.aquery(
                q,
                param=QueryParam(mode="mix", top_k=3, max_token_for_text_unit=2000),
            )
            # Truncate long responses for display
            display = result[:500] + "..." if len(result) > 500 else result
            print(f"A: {display}")
        except Exception as e:
            print(f"ERROR: {e}")


async def main():
    doc_json = EXTRACTION_OUTPUT_DIR / "FactoryAct1948_docling.json"

    if not doc_json.exists():
        print(f"ERROR: DoclingDocument not found at {doc_json}")
        print("Run scripts/01_extract_english.py first.")
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not found in .env")
        sys.exit(1)

    # Step 1: Load DoclingDocument
    print("=== Loading DoclingDocument ===")
    doc = load_docling_document(doc_json)
    print(f"Loaded: {len(doc.texts)} text elements, {len(doc.tables)} tables")

    # Step 2: Chunk with hierarchy
    print("\n=== Chunking with HierarchicalChunker ===")
    chunks = chunk_with_hierarchy(doc)
    print(f"Created {len(chunks)} chunks")

    # Save chunks for inspection
    chunks_path = EXTRACTION_OUTPUT_DIR / "FactoryAct1948_lightrag_chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunks saved to: {chunks_path}")

    # Show sample
    print("\n--- Sample chunks ---")
    for c in chunks[:5]:
        print(f"  [{c['type']}] {c['heading_chain'][:60]}: {c['text'][:80]}...")

    # Step 3: Index into LightRAG
    print("\n=== Indexing into LightRAG ===")
    rag = await index_chunks(chunks, RAG_STORAGE_DIR)

    # Step 4: Test queries
    await test_queries(rag)

    print("\n=== Done ===")
    print(f"RAG storage: {RAG_STORAGE_DIR}")
    print(f"Chunks JSON: {chunks_path}")


if __name__ == "__main__":
    asyncio.run(main())
