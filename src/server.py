"""FastAPI server for LawMaster chatbot with SSE streaming."""

import sys
import os
import json
import asyncio
import re
import queue as queue_mod
import threading
import time as time_mod
from pathlib import Path
from contextlib import asynccontextmanager

# Force standard asyncio loop before anything else
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Load .env early
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent


# --- Helpers ---

def _load_chunks() -> dict:
    """Load LightRAG chunk store (cached after first call)."""
    path = PROJECT_ROOT / "rag_storage" / "kv_store_text_chunks.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _parse_tag(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else None


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="LawMaster", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = PROJECT_ROOT / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# --- Corpus API ---

@app.get("/api/corpus")
async def corpus():
    """Return indexed documents grouped by category."""
    chunks = _load_chunks()
    if not chunks:
        return {"categories": []}

    docs = {}
    for k, v in chunks.items():
        text = v.get("content", v) if isinstance(v, dict) else v
        source = _parse_tag(r'\[Source: ([^\]]+)\]', text)
        if not source:
            continue
        if source not in docs:
            docs[source] = {
                "name": source.replace(".pdf", ""),
                "source": source,
                "category": _parse_tag(r'\[Category: ([^\]]+)\]', text) or "Other",
                "pages": set(),
                "chunks": 0,
            }
        docs[source]["chunks"] += 1
        page = _parse_tag(r'\[Page: (\d+)\]', text)
        if page:
            docs[source]["pages"].add(int(page))

    categories = {}
    for doc in docs.values():
        cat = doc["category"]
        if cat not in categories:
            categories[cat] = {"name": cat, "docs": []}
        categories[cat]["docs"].append({
            "name": doc["name"],
            "source": doc["source"],
            "pages": max(doc["pages"]) if doc["pages"] else doc["chunks"],
            "chunks": doc["chunks"],
        })

    for cat in categories.values():
        cat["docs"].sort(key=lambda d: d["name"])

    return {"categories": sorted(categories.values(), key=lambda c: c["name"])}


# --- Chunk API ---

@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Return a single chunk's text and metadata."""
    chunks = _load_chunks()
    chunk = chunks.get(chunk_id)
    if not chunk:
        return {"error": f"Chunk {chunk_id} not found"}

    text = chunk.get("content", chunk) if isinstance(chunk, dict) else chunk

    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": _parse_tag(r'\[Source: ([^\]]+)\]', text),
        "page": int(_parse_tag(r'\[Page: (\d+)\]', text) or 0),
        "category": _parse_tag(r'\[Category: ([^\]]+)\]', text),
        "location": _parse_tag(r'\[Location: ([^\]]+)\]', text),
        "type": _parse_tag(r'\[Type: ([^\]]+)\]', text),
        "sql_table": _parse_tag(r'\[SQL Table: ([^\]]+)\]', text),
    }


# --- SSE Streaming Chat ---

@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE streaming chat endpoint with real-time tool events."""
    body = await request.json()
    query = body.get("query", "")

    if not query.strip():
        return StreamingResponse(
            iter([f"event: error\ndata: {json.dumps({'type': 'permanent', 'service': 'server', 'code': 400, 'message': 'Empty query'})}\n\n"]),
            media_type="text/event-stream",
        )

    event_queue = queue_mod.Queue()

    def run_agent():
        try:
            from agno.agent import Agent
            from agno.models.together import Together
            from agno.tools.sql import SQLTools
            from src.agent.tools import LightRAGSearchTool
            from src.agent.instructions import SYSTEM_INSTRUCTIONS
            from src.config import TOGETHER_API_KEY

            os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "tables.db"

            # No ReasoningTools — Kimi K2.5 reasons natively via <reasoning> blocks.
            # Adding think/analyze tools causes K2.5 to stop after think() without searching.
            search_tool = LightRAGSearchTool(event_queue=event_queue)
            tools = [search_tool]
            if db_path.exists():
                tools.append(SQLTools(db_url=f"sqlite:///{db_path}"))

            agent = Agent(
                name="LawMaster",
                model=Together(id="moonshotai/kimi-k2.5", max_tokens=16384),
                tools=tools,
                description="You are LawMaster, an expert on Indian industrial and manufacturing law.",
                instructions=SYSTEM_INSTRUCTIONS,
                markdown=True,
                tool_call_limit=10,
            )

            start = time_mod.time()

            # Emit immediate "thinking" event so the frontend knows we're alive
            event_queue.put(("tool_start", {"name": "reasoning", "args_summary": "Thinking..."}))

            # Non-streaming run — Kimi K2.5's stream=True doesn't yield
            # content tokens reliably with tool calls. Tool events still
            # stream in real-time via the LightRAGSearchTool queue wrapper.
            response = agent.run(query)

            # Close the reasoning event
            event_queue.put(("tool_result", {"name": "reasoning", "duration_s": round(time_mod.time() - start, 1), "success": True}))

            # Extract final content
            content = response.content or ""
            if not content and hasattr(response, 'messages'):
                for msg in reversed(response.messages):
                    if hasattr(msg, 'content') and msg.content and msg.role == 'assistant':
                        content = msg.content
                        break

            # Emit content as chunked tokens for frontend streaming effect
            if content:
                # Split into ~word-sized chunks for smooth typing
                words = content.split(' ')
                chunk = ''
                for w in words:
                    chunk += (' ' if chunk else '') + w
                    if len(chunk) > 20:
                        event_queue.put(("content", {"token": chunk}))
                        chunk = ''
                if chunk:
                    event_queue.put(("content", {"token": chunk}))

            duration = round(time_mod.time() - start, 1)
            event_queue.put(("done", {"total_duration_s": duration}))

        except Exception as e:
            error_msg = str(e).lower()
            is_transient = any(kw in error_msg for kw in (
                "503", "429", "500", "timeout", "connection", "rate",
            ))
            service = "together_ai"
            if "groq" in error_msg:
                service = "groq"
            elif "together" not in error_msg and "kimi" not in error_msg:
                service = "server"

            event_queue.put(("error", {
                "type": "transient" if is_transient else "permanent",
                "service": service,
                "code": 503 if "503" in error_msg else 429 if "429" in error_msg else 500,
                "message": str(e)[:200],
            }))

    async def sse_generator():
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        loop = asyncio.get_event_loop()
        tool_count = 0
        while True:
            try:
                event_type, data = await loop.run_in_executor(
                    None, lambda: event_queue.get(timeout=300)
                )
            except queue_mod.Empty:
                yield f"event: error\ndata: {json.dumps({'type': 'transient', 'service': 'server', 'code': 504, 'message': 'Agent timed out after 5 minutes'})}\n\n"
                break

            if event_type == "tool_start":
                tool_count += 1
            if event_type == "done":
                data["tool_count"] = tool_count

            yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

            if event_type in ("done", "error"):
                break

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --- Legacy batch endpoint (backward compat) ---

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")

    if not query.strip():
        return {"error": "Empty query"}

    from src.agent.agent import create_agent
    import concurrent.futures

    agent = create_agent(debug_mode=False)

    def _run():
        return agent.run(query)

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(pool, _run)

    content = response.content or ""
    if not content and hasattr(response, 'messages'):
        for msg in reversed(response.messages):
            if hasattr(msg, 'content') and msg.content and msg.role == 'assistant':
                content = msg.content
                break

    return {"response": content or "No response generated. Please try again."}


@app.get("/health")
async def health():
    return {"status": "ok"}
