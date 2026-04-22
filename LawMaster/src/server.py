"""FastAPI server for LawMaster chatbot with SSE streaming."""

import sys
import os
import json
import asyncio
import re
import uuid
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
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).parent.parent

# PDF document root
DOCS_BASE = PROJECT_ROOT / "project 2 _ai tool for compliance -20260331T205330Z-1-001" / "project 2 _ai tool for compliance "


# --- Helpers ---

def _load_chunks() -> dict:
    """Load LightRAG chunk store (cached after first call)."""
    path = PROJECT_ROOT / "rag_storage" / "kv_store_text_chunks.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


_pdf_index: dict[str, Path] | None = None

def _get_pdf_index() -> dict[str, Path]:
    """Build filename → filepath index for all PDFs (cached)."""
    global _pdf_index
    if _pdf_index is not None:
        return _pdf_index
    _pdf_index = {}
    if DOCS_BASE.exists():
        for pdf in DOCS_BASE.rglob("*.pdf"):
            _pdf_index[pdf.name] = pdf
    return _pdf_index


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


# --- PDF Viewer API ---

@app.get("/api/pdf/{source:path}")
async def serve_pdf(source: str):
    """Serve a PDF document by source filename."""
    index = _get_pdf_index()
    # Try exact match first
    pdf_path = index.get(source)
    if not pdf_path:
        # Try with .pdf extension
        pdf_path = index.get(source + ".pdf")
    if not pdf_path or not pdf_path.exists():
        return {"error": f"PDF not found: {source}"}
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{pdf_path.name}\""},
    )


# --- SSE Streaming Chat ---

@app.post("/chat/stream")
async def chat_stream(request: Request):
    """SSE streaming chat endpoint with real-time tool events."""
    body = await request.json()
    query = body.get("query", "")
    session_id = body.get("session_id") or str(uuid.uuid4())

    if not query.strip():
        return StreamingResponse(
            iter([f"event: error\ndata: {json.dumps({'type': 'permanent', 'service': 'server', 'code': 400, 'message': 'Empty query'})}\n\n"]),
            media_type="text/event-stream",
        )

    event_queue = queue_mod.Queue()

    def run_agent():
        try:
            from agno.agent import Agent
            from agno.models.google import Gemini
            from agno.tools.sql import SQLTools
            from agno.db.sqlite import SqliteDb
            from src.agent.tools import LightRAGSearchTool
            from src.agent.instructions import SYSTEM_INSTRUCTIONS
            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "tables.db"

            # No ReasoningTools — Gemini reasons natively.
            search_tool = LightRAGSearchTool(event_queue=event_queue)
            tools = [search_tool]
            if db_path.exists():
                tools.append(SQLTools(db_url=f"sqlite:///{db_path}"))

            agent = Agent(
                name="LawMaster",
                model=Gemini(id="gemini-3-flash-preview"),
                tools=tools,
                description="You are LawMaster, an expert on Indian industrial and manufacturing law.",
                instructions=SYSTEM_INSTRUCTIONS,
                markdown=True,
                tool_call_limit=20,
                # Session & history
                session_id=session_id,
                db=SqliteDb(
                    session_table="lawmaster_sessions",
                    db_file=str(project_root / "data" / "sessions.db"),
                ),
                add_history_to_context=True,
                num_history_runs=10,
            )

            start = time_mod.time()

            # Emit immediate "thinking" event so the frontend knows we're alive
            event_queue.put(("tool_start", {"name": "reasoning", "args_summary": "Thinking..."}))

            # Non-streaming run — tool events stream in real-time
            # via the LightRAGSearchTool queue wrapper.
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

            if not content:
                content = "I'm sorry, I wasn't able to generate a response. Could you rephrase your question? I can help with Indian industrial law, factory compliance, pollution control regulations, and Chhattisgarh industrial policy."

            # --- Citation extraction: resolve [ref:chunk-xxx] → [N] ---
            citations = []
            chunk_id_to_idx = {}
            chunks_data = _load_chunks()

            def _resolve_ref(m):
                chunk_id = m.group(1)
                # Dedup: same chunk_id → same citation number
                if chunk_id in chunk_id_to_idx:
                    return f"[{chunk_id_to_idx[chunk_id]}]"
                idx = len(citations) + 1
                chunk_id_to_idx[chunk_id] = idx
                # Look up chunk metadata
                chunk = chunks_data.get(chunk_id)
                snippet = ""
                source = ""
                pdf_source = ""
                page = ""
                location = ""
                full_text = ""
                if chunk:
                    raw = chunk.get("content", chunk) if isinstance(chunk, dict) else chunk
                    source = _parse_tag(r'\[Source: ([^\]]+)\]', raw) or ""
                    page = _parse_tag(r'\[Page: (\d+)\]', raw) or ""
                    location = _parse_tag(r'\[Location: ([^\]]+)\]', raw) or ""
                    pdf_source = source
                    clean = re.sub(r'\[.*?\]', '', raw).strip()
                    snippet = clean[:200]
                    full_text = clean
                citations.append({
                    "index": idx,
                    "source": source.replace(".pdf", ""),
                    "page": page,
                    "location": location,
                    "snippet": snippet or chunk_id,
                    "full_text": full_text if full_text else chunk_id,
                    "pdf_source": pdf_source,
                })
                return f"[{idx}]"

            if content:
                # Expand multi-ref brackets: [ref:chunk-aaa, ref:chunk-bbb] → [ref:chunk-aaa][ref:chunk-bbb]
                def _expand_multi_ref(m):
                    ids = re.findall(r'chunk-[a-f0-9]+', m.group(0))
                    return ''.join(f'[ref:{cid}]' for cid in ids)
                content = re.sub(r'\[ref:chunk-[a-f0-9]+(?:,\s*ref:chunk-[a-f0-9]+)+\]', _expand_multi_ref, content)
                # Resolve single refs: [ref:chunk-xxx] → [N]
                content = re.sub(r'\[ref:(chunk-[a-f0-9]+)\]', _resolve_ref, content)
                # Strip any leftover (Source: ...) patterns
                content = re.sub(r'\(Source:\s*(?:[^()]*|\([^()]*\))*\)', '', content)

            # Emit content as chunked tokens for frontend streaming effect
            if content:
                # Split into ~word-sized chunks for smooth typing
                words = content.split(' ')
                chunk = ''
                for w in words:
                    chunk += (' ' if chunk else '') + w
                    if len(chunk) > 20:
                        event_queue.put(("content", {"token": chunk}))
                        time_mod.sleep(0.03)
                        chunk = ''
                if chunk:
                    event_queue.put(("content", {"token": chunk}))

            # Emit citation events for the provenance panel
            for cite in citations:
                event_queue.put(("citation", cite))

            duration = round(time_mod.time() - start, 1)
            event_queue.put(("done", {"total_duration_s": duration, "session_id": session_id}))

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
        start_time = time_mod.time()
        while True:
            try:
                event_type, data = await loop.run_in_executor(
                    None, lambda: event_queue.get(timeout=5)
                )
            except queue_mod.Empty:
                if not thread.is_alive():
                    yield f"event: error\ndata: {json.dumps({'type': 'permanent', 'service': 'server', 'code': 500, 'message': 'Agent thread crashed'})}\n\n"
                    break
                if time_mod.time() - start_time > 300:
                    yield f"event: error\ndata: {json.dumps({'type': 'transient', 'service': 'server', 'code': 504, 'message': 'Agent timed out after 5 minutes'})}\n\n"
                    break
                continue

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
