"""LightRAG instance loader. Initialized once, reused across queries.

A single persistent event loop (_rag_loop) runs in a daemon thread.  All
LightRAG async operations — storage init, queries — are dispatched onto it
via run_coroutine_threadsafe().  This guarantees that every asyncio primitive
LightRAG creates (locks, semaphores, storage handles) stays bound to one loop
regardless of which request thread calls run_rag_query().
"""

import os
import asyncio
import threading
import numpy as np
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from sentence_transformers import SentenceTransformer
import nest_asyncio

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_rag_instance = None
_bge_model = None
_rag_loop: asyncio.AbstractEventLoop | None = None
_rag_thread: threading.Thread | None = None


def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        print("[RAG] Loading BGE-large (offline)...")
        _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)
    return _bge_model


async def bge_embedding_func(texts: list[str]) -> np.ndarray:
    return _get_bge_model().encode(texts, normalize_embeddings=True)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 output."""
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


QUERY_MODEL = "llama-3.3-70b-versatile"  # Groq — query reasoning + keyword extraction


async def groq_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """LightRAG-compatible LLM function using Groq via OpenAI-compatible API.

    Disables keyword_extraction structured output because LightRAG passes
    json_schema format which Groq's 70B doesn't support. The 70B follows
    JSON instructions in the prompt without enforcement.
    """
    from src.config import GROQ_API_KEY
    # Groq 70B doesn't support json_schema — force keyword_extraction off
    # so openai_complete_if_cache won't set response_format internally
    kwargs["keyword_extraction"] = False
    result = await openai_complete_if_cache(
        model=QUERY_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        **kwargs,
    )
    return _strip_think_tags(result)


def _ensure_rag_loop():
    """Spin up the persistent RAG event loop (once)."""
    global _rag_loop, _rag_thread
    if _rag_loop is not None:
        return
    _rag_loop = asyncio.new_event_loop()
    nest_asyncio.apply(_rag_loop)
    _rag_thread = threading.Thread(
        target=_rag_loop.run_forever, daemon=True, name="rag-loop"
    )
    _rag_thread.start()


def get_rag_instance() -> LightRAG:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    _ensure_rag_loop()

    from src.config import RAG_STORAGE_DIR

    rag = LightRAG(
        working_dir=str(RAG_STORAGE_DIR),
        llm_model_func=groq_llm_func,
        llm_model_name=QUERY_MODEL,
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

    # Initialize storages on the persistent loop
    future = asyncio.run_coroutine_threadsafe(
        rag.initialize_storages(), _rag_loop
    )
    future.result(timeout=120)
    print(f"[RAG] Loaded. Graph: {rag.chunk_entity_relation_graph._graph.number_of_nodes()} nodes")

    _rag_instance = rag
    return _rag_instance


def run_rag_query(query: str, mode: str = "hybrid") -> str:
    """Run a LightRAG query on the persistent event loop (thread-safe)."""
    rag = get_rag_instance()
    future = asyncio.run_coroutine_threadsafe(
        rag.aquery(query, param=QueryParam(mode=mode)),
        _rag_loop,
    )
    return future.result(timeout=120)


def retrieve_chunks(query: str, mode: str = "hybrid", top_k: int = 10) -> list[dict]:
    """Retrieve relevant chunks with IDs and metadata (no LLM generation).

    Returns list of dicts: {chunk_id, source, page, content_preview, raw_content}
    """
    import re
    rag = get_rag_instance()
    future = asyncio.run_coroutine_threadsafe(
        rag.aquery_data(
            query,
            param=QueryParam(
                mode=mode,
                only_need_context=True,
                include_references=True,
                chunk_top_k=top_k,
            ),
        ),
        _rag_loop,
    )
    result = future.result(timeout=120)

    chunks = result.get("data", {}).get("chunks", [])
    out = []
    for c in chunks:
        raw = c.get("content", "")
        # Extract metadata tags
        source = ""
        page = ""
        m = re.search(r'\[Source: ([^\]]+)\]', raw)
        if m:
            source = m.group(1)
        m = re.search(r'\[Page: (\d+)\]', raw)
        if m:
            page = m.group(1)
        # Strip tags for clean preview
        clean = re.sub(r'\[.*?\]', '', raw).strip()
        out.append({
            "chunk_id": c.get("chunk_id", ""),
            "source": source,
            "page": page,
            "content_preview": clean[:300],
            "raw_content": raw,
        })
    return out
