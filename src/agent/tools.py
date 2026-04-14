"""Custom Agno tools for LawMaster: LightRAG legal text search."""

import time
import queue as queue_mod
from agno.tools import Toolkit
from agno.agent import Agent


class LightRAGSearchTool(Toolkit):
    """Search the legal knowledge graph via LightRAG.

    Uses hybrid mode (KG + vector) for comprehensive retrieval across
    43 indexed legal documents covering Factories Act, Pollution Control,
    and Chhattisgarh Industrial Policy.

    Optionally emits tool_start/tool_result events to an event_queue
    for real-time SSE streaming to the frontend.
    """

    def __init__(self, event_queue: queue_mod.Queue = None):
        super().__init__(
            name="legal_search",
            instructions=[
                "Use search_legal_text to find legal provisions, definitions, sections, and penalties.",
                "Formulate clear, specific queries — e.g. 'definition of factory under Factories Act'.",
                "For structured data (exposure limits, industry classifications, fee tables), use SQL tools instead.",
            ],
        )
        self._event_queue = event_queue
        self.register(self.search_legal_text)

    def _emit(self, event_type: str, data: dict):
        if self._event_queue is not None:
            self._event_queue.put((event_type, data))

    def search_legal_text(self, agent: Agent, query: str) -> str:
        """Search indexed legal documents for provisions, definitions, sections, and penalties.

        Uses knowledge graph + vector search across 43 legal documents including:
        - Factories Act, 1948 and Factory Rules
        - Pollution Control Board regulations (EIA, HWM, CPCB classifications)
        - Chhattisgarh Industrial Policy 2024-30 and notifications

        Args:
            query: Natural language legal question, e.g. 'duties of occupier for hazardous processes'.

        Returns:
            str: Retrieved legal text with source citations.
        """
        self._emit("tool_start", {"name": "search_legal_text", "args_summary": query})
        start = time.time()
        try:
            from src.agent.rag import retrieve_chunks
            chunks = retrieve_chunks(query, top_k=10)
            # Format chunks with IDs so the LLM can cite them inline
            parts = []
            for c in chunks:
                header = f"[ref:{c['chunk_id']}]"
                if c["source"]:
                    header += f" Source: {c['source']}"
                if c["page"]:
                    header += f" | Page: {c['page']}"
                parts.append(f"{header}\n{c['content_preview']}")
            result = "\n\n---\n\n".join(parts) if parts else "No relevant legal text found."
            duration = round(time.time() - start, 1)
            self._emit("tool_result", {"name": "search_legal_text", "duration_s": duration, "success": True})
            return result
        except Exception as e:
            duration = round(time.time() - start, 1)
            self._emit("tool_result", {"name": "search_legal_text", "duration_s": duration, "success": False})
            raise
