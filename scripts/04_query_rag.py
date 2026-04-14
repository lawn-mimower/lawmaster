#!/usr/bin/env python3
"""Interactive query against the indexed LightRAG storage.

Usage:
    python scripts/04_query_rag.py "your question here"
    python scripts/04_query_rag.py  # interactive mode
"""

import sys
import os
import asyncio
import numpy as np
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import PROJECT_ROOT, GEMINI_API_KEY

# --- BGE-large Local Embeddings ---
from sentence_transformers import SentenceTransformer
from lightrag.utils import EmbeddingFunc

print("Loading BGE-large (offline)...")
bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)

async def bge_embedding_func(texts: list[str]) -> np.ndarray:
    return bge_model.encode(texts, normalize_embeddings=True)


async def main():
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.gemini import gemini_model_complete

    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    working_dir = PROJECT_ROOT / "rag_storage"
    if not working_dir.exists():
        print(f"ERROR: No RAG storage at {working_dir}")
        sys.exit(1)

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
    )
    await rag.initialize_storages()

    # Show graph stats
    graph = rag.chunk_entity_relation_graph._graph
    print(f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    if graph.number_of_nodes() > 0:
        print("Entities:")
        for node in list(graph.nodes)[:15]:
            ndata = graph.nodes[node]
            print(f"  - {node} ({ndata.get('entity_type', '?')})")

    # Query mode
    if len(sys.argv) > 1:
        questions = [" ".join(sys.argv[1:])]
    else:
        print("\n--- Interactive mode (type 'quit' to exit) ---")
        questions = []
        while True:
            q = input("\nQ: ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            if q:
                questions.append(q)

    for q in questions:
        print(f"\nQ: {q}")
        print("-" * 50)
        for mode in ["mix"]:
            result = await rag.aquery(q, param=QueryParam(mode=mode, top_k=5))
            print(f"[{mode}] {result}")


if __name__ == "__main__":
    asyncio.run(main())
