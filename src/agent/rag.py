"""LightRAG instance loader. Initialized once, reused across queries."""

import os
import numpy as np
from functools import partial
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_rag_instance = None
_bge_model = None


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


def get_rag_instance() -> LightRAG:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    import asyncio

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

    # Initialize with a fresh loop — don't bind to any thread's loop
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(rag.initialize_storages())
    finally:
        loop.close()
    print(f"[RAG] Loaded. Graph: {rag.chunk_entity_relation_graph._graph.number_of_nodes()} nodes")

    _rag_instance = rag
    return _rag_instance
