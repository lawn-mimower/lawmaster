# LawMaster Frontend Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the batch-response chat UI with a three-panel streaming interface featuring a corpus browser, provenance panel with citation highlighting, and real-time tool event display.

**Architecture:** Three-panel layout (Alpine.js + Tailwind CDN, no build step). SSE streaming via tool wrapper side-channel + Agno `stream=True`. Fresh agent per request (no session poisoning). Two new API endpoints (`/api/corpus`, `/api/chunks/{chunk_id}`).

**Tech Stack:** FastAPI, Alpine.js (CDN), Tailwind CSS (CDN), Agno with Together AI (Kimi K2.5), native SSE via fetch+ReadableStream.

**Spec:** `docs/superpowers/specs/2026-04-14-frontend-redesign-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/server.py` | Modify | Add SSE `/chat/stream`, `/api/corpus`, `/api/chunks/{chunk_id}` endpoints |
| `src/agent/tools.py` | Modify | Add event-emitting wrapper to LightRAGSearchTool and hook for SQL tools |
| `static/index.html` | Rewrite | Three-panel SPA with Alpine.js reactivity, Tailwind styling |

Three files. No new modules, no new dependencies to install.

---

### Task 1: SSE Streaming Backend — Tool Event Wrappers

**Files:**
- Modify: `src/agent/tools.py`

The core trick: wrap tool functions to push events to a thread-safe queue as a side effect, so SSE can emit them in real-time while Agno's stream yields content tokens.

- [ ] **Step 1: Add event queue protocol to LightRAGSearchTool**

```python
"""Custom Agno tools for LawMaster: LightRAG legal text search."""

import json
import time
import asyncio
import queue as queue_mod
from lightrag import QueryParam
from agno.tools import Toolkit
from agno.agent import Agent


class LightRAGSearchTool(Toolkit):
    """Search the legal knowledge graph via LightRAG."""

    def __init__(self, event_queue: queue_mod.Queue = None):
        super().__init__(
            name="legal_search",
            instructions=[
                "Use search_legal_text to find legal provisions, definitions, sections, and penalties.",
                "Formulate clear, specific queries — e.g. 'definition of factory under Factories Act'.",
                "For structured data (exposure limits, industry classifications, fee tables), use SQL tools instead.",
            ],
        )
        self._rag = None
        self._event_queue = event_queue
        self.register(self.search_legal_text)

    def _get_rag(self):
        if self._rag is None:
            from src.agent.rag import get_rag_instance
            self._rag = get_rag_instance()
        return self._rag

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
            rag = self._get_rag()
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                rag.aquery(query, param=QueryParam(mode="hybrid"))
            )
            duration = round(time.time() - start, 1)
            self._emit("tool_result", {"name": "search_legal_text", "duration_s": duration, "success": True})
            return result
        except Exception as e:
            duration = round(time.time() - start, 1)
            self._emit("tool_result", {"name": "search_legal_text", "duration_s": duration, "success": False})
            raise
```

- [ ] **Step 2: Verify tool still works without a queue (backward compat)**

Run:
```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import os, sys
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env', override=True)
from src.agent.tools import LightRAGSearchTool
tool = LightRAGSearchTool()  # no queue
from agno.agent import Agent
result = tool.search_legal_text(Agent(model=None), 'definition of factory')
print(f'OK: {len(result)} chars')
"
```

Expected: prints char count, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/agent/tools.py
git commit -m "feat: add event queue to LightRAGSearchTool for SSE streaming"
```

---

### Task 2: SSE Streaming Backend — Server Endpoints

**Files:**
- Modify: `src/server.py`

- [ ] **Step 1: Add `/api/corpus` endpoint**

Add after the existing `/health` endpoint:

```python
@app.get("/api/corpus")
async def corpus():
    """Return indexed documents grouped by category."""
    import re
    chunks_path = Path(__file__).parent.parent / "rag_storage" / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        return {"categories": []}

    with open(chunks_path) as f:
        chunks = json.load(f)

    # Build doc list from chunk metadata
    docs = {}  # source -> {name, pages set, category, chunks}
    for k, v in chunks.items():
        text = v.get("content", v) if isinstance(v, dict) else v
        src_match = re.search(r'\[Source: ([^\]]+)\]', text)
        cat_match = re.search(r'\[Category: ([^\]]+)\]', text)
        page_match = re.search(r'\[Page: (\d+)\]', text)
        if not src_match:
            continue
        source = src_match.group(1)
        if source not in docs:
            docs[source] = {
                "name": source.replace(".pdf", ""),
                "source": source,
                "category": cat_match.group(1) if cat_match else "Other",
                "pages": set(),
                "chunks": 0,
            }
        docs[source]["chunks"] += 1
        if page_match:
            docs[source]["pages"].add(int(page_match.group(1)))

    # Group by category
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

    # Sort
    for cat in categories.values():
        cat["docs"].sort(key=lambda d: d["name"])

    return {"categories": sorted(categories.values(), key=lambda c: c["name"])}
```

- [ ] **Step 2: Add `/api/chunks/{chunk_id}` endpoint**

```python
@app.get("/api/chunks/{chunk_id}")
async def get_chunk(chunk_id: str):
    """Return a single chunk's text and metadata."""
    import re
    chunks_path = Path(__file__).parent.parent / "rag_storage" / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        return {"error": "No chunks indexed"}

    with open(chunks_path) as f:
        chunks = json.load(f)

    chunk = chunks.get(chunk_id)
    if not chunk:
        return {"error": f"Chunk {chunk_id} not found"}

    text = chunk.get("content", chunk) if isinstance(chunk, dict) else chunk

    # Parse metadata from inline tags
    def extract(pattern, text):
        m = re.search(pattern, text)
        return m.group(1) if m else None

    return {
        "chunk_id": chunk_id,
        "text": text,
        "source": extract(r'\[Source: ([^\]]+)\]', text),
        "page": int(extract(r'\[Page: (\d+)\]', text) or 0),
        "category": extract(r'\[Category: ([^\]]+)\]', text),
        "location": extract(r'\[Location: ([^\]]+)\]', text),
        "type": extract(r'\[Type: ([^\]]+)\]', text),
        "sql_table": extract(r'\[SQL Table: ([^\]]+)\]', text),
    }
```

- [ ] **Step 3: Test both endpoints**

Run:
```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --loop asyncio &
sleep 3
curl -s http://localhost:8000/api/corpus | python3 -m json.tool | head -20
curl -s http://localhost:8000/api/chunks/chunk-b1cc7368993f87e152df677502ce0cca | python3 -m json.tool
kill %1
```

Expected: corpus returns 3 categories with docs, chunk returns text with parsed metadata.

- [ ] **Step 4: Add `POST /chat/stream` SSE endpoint**

```python
import queue as queue_mod
import threading
import time as time_mod


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
            from src.agent.agent import create_agent
            from src.agent.tools import LightRAGSearchTool
            from agno.tools.reasoning import ReasoningTools
            from agno.tools.sql import SQLTools
            from agno.models.together import Together
            from src.agent.instructions import SYSTEM_INSTRUCTIONS
            from agno.db.sqlite import SqliteDb

            project_root = Path(__file__).parent.parent
            db_path = project_root / "data" / "tables.db"

            # Build tools with event queue
            search_tool = LightRAGSearchTool(event_queue=event_queue)
            tools = [ReasoningTools(add_instructions=True), search_tool]
            if db_path.exists():
                tools.append(SQLTools(db_url=f"sqlite:///{db_path}"))

            from src.config import TOGETHER_API_KEY
            import os
            os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

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

            # Stream content tokens via Agno stream=True
            full_content = []
            for event in agent.run(query, stream=True):
                if hasattr(event, 'content') and event.content:
                    event_queue.put(("content", {"token": event.content}))
                    full_content.append(event.content)

            duration = round(time_mod.time() - start, 1)

            # Extract citations from the response messages
            content_text = "".join(full_content)
            # Count tool calls from the queue events we already emitted
            event_queue.put(("done", {"total_duration_s": duration}))

        except Exception as e:
            error_msg = str(e).lower()
            is_transient = any(kw in error_msg for kw in ("503", "429", "500", "timeout", "connection", "rate"))
            service = "together_ai" if "together" in error_msg or "kimi" in error_msg else "groq" if "groq" in error_msg else "server"
            event_queue.put(("error", {
                "type": "transient" if is_transient else "permanent",
                "service": service,
                "code": 503 if "503" in error_msg else 429 if "429" in error_msg else 500,
                "message": str(e)[:200],
            }))

    def sse_generator():
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        tool_count = 0
        while True:
            try:
                event_type, data = event_queue.get(timeout=300)
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

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
```

- [ ] **Step 5: Test SSE endpoint**

Run:
```bash
/home/pc/anaconda3/envs/ml-env/bin/python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --loop asyncio &
sleep 3
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the definition of factory?"}' 2>&1 | head -30
kill %1
```

Expected: SSE events appear line by line — tool_start, tool_result, content tokens, done.

- [ ] **Step 6: Commit**

```bash
git add src/server.py
git commit -m "feat: add SSE streaming endpoint with real-time tool events"
```

---

### Task 3: Frontend — Three-Panel Layout Shell

**Files:**
- Rewrite: `static/index.html`

This task creates the layout structure with Alpine.js state management. No streaming yet — just the panels, corpus browser, and chat input.

- [ ] **Step 1: Write the three-panel HTML shell**

Rewrite `static/index.html` with:
- Alpine.js and Tailwind CSS loaded from CDN
- Three-panel layout with CSS flexbox (no react-resizable-panels — keep it simple with CSS `resize`)
- Alpine.js `x-data` for app state: `{ leftOpen: true, rightOpen: false, messages: [], streaming: false, query: '', history: [], historyIndex: -1, corpus: { categories: [] }, citations: [] }`
- Left panel: corpus tree loaded from `/api/corpus` on mount
- Center: message list and input
- Right panel: citation viewer (empty until populated)
- Topbar with toggle buttons
- Light theme colors from the spec

The full HTML file will be ~600 lines. Key Alpine.js components:

```html
<div x-data="lawmaster()" x-init="loadCorpus()" class="h-screen flex flex-col bg-[#f7f8fa] font-sans text-[#1a1d24]">

  <!-- Topbar -->
  <header class="h-13 flex items-center px-5 border-b border-[#e2e5ea] bg-white shadow-sm gap-3">
    <div class="flex items-center gap-2">
      <div class="w-7 h-7 rounded-md bg-gradient-to-br from-teal-600 to-teal-700 flex items-center justify-center text-white text-sm font-bold">L</div>
      <h1 class="text-base font-semibold">LawMaster</h1>
    </div>
    <span class="text-[0.65rem] bg-teal-50 text-teal-700 px-2 py-0.5 rounded-full font-medium" x-text="`${corpusDocCount} docs`"></span>
    <div class="flex-1"></div>
    <button @click="leftOpen = !leftOpen" class="text-xs px-3 py-1 rounded-md border" :class="leftOpen ? 'bg-teal-50 border-teal-200 text-teal-700' : 'border-gray-200 text-gray-500'">Corpus</button>
    <button @click="rightOpen = !rightOpen" class="text-xs px-3 py-1 rounded-md border" :class="rightOpen ? 'bg-teal-50 border-teal-200 text-teal-700' : 'border-gray-200 text-gray-500'">Sources</button>
  </header>

  <!-- Panels -->
  <div class="flex flex-1 overflow-hidden">
    <!-- Left: Corpus -->
    <aside x-show="leftOpen" x-transition class="w-64 border-r border-[#e2e5ea] bg-white flex flex-col overflow-hidden">
      <!-- corpus tree content -->
    </aside>

    <!-- Center: Chat -->
    <main class="flex-1 flex flex-col min-w-0">
      <!-- messages + input -->
    </main>

    <!-- Right: Provenance -->
    <aside x-show="rightOpen" x-transition class="w-[340px] border-l border-[#e2e5ea] bg-white flex flex-col overflow-hidden">
      <!-- citation viewer -->
    </aside>
  </div>
</div>
```

- [ ] **Step 2: Implement the Alpine.js `lawmaster()` data function**

Core state and methods. Place in a `<script>` tag at the bottom:

```javascript
function lawmaster() {
  return {
    // Panel state
    leftOpen: true,
    rightOpen: false,

    // Chat state
    messages: [],
    streaming: false,
    query: '',
    currentStatus: '',
    currentTools: [],
    currentContent: '',

    // Corpus
    corpus: { categories: [] },
    corpusSearch: '',
    get corpusDocCount() {
      return this.corpus.categories.reduce((sum, c) => sum + c.docs.length, 0);
    },
    get filteredCorpus() {
      if (!this.corpusSearch) return this.corpus.categories;
      const q = this.corpusSearch.toLowerCase();
      return this.corpus.categories.map(c => ({
        ...c,
        docs: c.docs.filter(d => d.name.toLowerCase().includes(q)),
      })).filter(c => c.docs.length > 0);
    },

    // Provenance
    citations: [],
    activeCitation: null,

    // History
    history: [],
    historyIndex: -1,

    // Toast
    toast: null,

    async loadCorpus() {
      try {
        const res = await fetch('/api/corpus');
        this.corpus = await res.json();
      } catch (e) {
        console.error('Failed to load corpus:', e);
      }
    },

    async sendMessage() {
      const q = this.query.trim();
      if (!q || this.streaming) return;

      this.history.push(q);
      this.historyIndex = this.history.length;
      this.query = '';
      this.messages.push({ role: 'user', content: q });
      this.streaming = true;
      this.currentStatus = '';
      this.currentTools = [];
      this.currentContent = '';
      this.citations = [];

      // Add placeholder assistant message
      const assistantMsg = { role: 'assistant', content: '', tools: [], done: false };
      this.messages.push(assistantMsg);

      try {
        const response = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: q }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          let eventType = '';
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              eventType = line.slice(7);
            } else if (line.startsWith('data: ') && eventType) {
              const data = JSON.parse(line.slice(6));
              this.handleSSE(eventType, data, assistantMsg);
              eventType = '';
            }
          }
        }
      } catch (e) {
        this.showToast('transient', 'Network', 0, `Connection error: ${e.message}`);
      }

      assistantMsg.done = true;
      this.streaming = false;
      this.currentStatus = '';
    },

    handleSSE(event, data, msg) {
      switch (event) {
        case 'tool_start':
          const displayName = this.toolDisplayName(data.name, data.args_summary);
          this.currentStatus = displayName;
          this.currentTools.push({ name: data.name, display: displayName, duration: null, success: null });
          msg.tools = [...this.currentTools];
          break;
        case 'tool_result':
          const tool = this.currentTools.findLast(t => t.name === data.name && t.duration === null);
          if (tool) {
            tool.duration = data.duration_s;
            tool.success = data.success;
          }
          msg.tools = [...this.currentTools];
          break;
        case 'content':
          this.currentContent += data.token;
          msg.content = this.currentContent;
          this.currentStatus = '';
          break;
        case 'citation':
          this.citations.push(data);
          break;
        case 'done':
          msg.toolCount = data.tool_count;
          msg.totalDuration = data.total_duration_s;
          break;
        case 'error':
          if (data.type === 'transient') {
            this.showToast(data.type, data.service, data.code, data.message);
          } else {
            msg.content = data.message;
            msg.isError = true;
          }
          break;
      }
    },

    toolDisplayName(name, args) {
      const argShort = (args || '').slice(0, 60);
      const map = {
        'search_legal_text': `Searching: ${argShort}...`,
        'run_sql_query': `Querying SQL table...`,
        'think': 'Planning approach...',
        'analyze': 'Analyzing results...',
      };
      return map[name] || `Running ${name}...`;
    },

    showToast(type, service, code, message) {
      this.toast = { type, service, code, message };
      setTimeout(() => { this.toast = null; }, 15000);
    },

    retryLast() {
      this.toast = null;
      if (this.history.length > 0) {
        this.query = this.history[this.history.length - 1];
        // Remove last assistant message (the failed one)
        if (this.messages.length >= 2) {
          this.messages.pop(); // assistant
          this.messages.pop(); // user
        }
        this.sendMessage();
      }
    },

    openCitation(index) {
      this.activeCitation = this.citations.find(c => c.index === index);
      this.rightOpen = true;
      this.$nextTick(() => {
        const el = document.getElementById(`citation-${index}`);
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      });
    },

    handleInputKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
      if (e.key === 'Escape' && this.streaming) {
        this.streaming = false;
        // TODO: abort controller
      }
      if (e.key === 'ArrowUp' && !this.query && this.history.length > 0) {
        e.preventDefault();
        this.historyIndex = Math.max(0, this.historyIndex - 1);
        this.query = this.history[this.historyIndex] || '';
      }
      if (e.key === 'ArrowDown' && !this.query) {
        e.preventDefault();
        this.historyIndex = Math.min(this.history.length, this.historyIndex + 1);
        this.query = this.history[this.historyIndex] || '';
      }
    },

    formatMarkdown(text) {
      if (!text) return '';
      return text
        .replace(/^### (.+)$/gm, '<h4 class="font-semibold text-[#0f1117] mt-3 mb-1">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 class="font-semibold text-[#0f1117] mt-3 mb-1 text-base">$1</h3>')
        .replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-[#0f1117]">$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code class="bg-gray-100 px-1.5 py-0.5 rounded text-sm">$1</code>')
        .replace(/^- (.+)$/gm, '<li class="ml-5 list-disc">$1</li>')
        .replace(/^> (.+)$/gm, '<blockquote class="border-l-3 border-gray-300 pl-3 text-gray-500 my-2">$1</blockquote>')
        .replace(/\n\n/g, '<br><br>')
        // Citation badges: [1], [2] etc
        .replace(/\[(\d+)\]/g, '<button class="inline-flex items-center justify-center bg-teal-50 text-teal-700 text-[0.6rem] font-bold px-1.5 rounded border border-teal-200 cursor-pointer hover:bg-teal-100 align-super mx-0.5" @click="openCitation($1)">$1</button>');
    },
  };
}
```

- [ ] **Step 3: Build the full HTML with all three panels**

Write the complete `static/index.html`. This is the full file content — I'll show the template sections that reference the Alpine.js methods above.

The left panel (corpus browser):
```html
<aside x-show="leftOpen" x-transition.origin.left class="w-64 border-r border-[#e2e5ea] bg-white flex flex-col overflow-hidden shrink-0">
  <div class="px-4 py-3 text-[0.7rem] font-semibold uppercase tracking-wider text-gray-400 border-b border-[#e2e5ea] flex items-center justify-between">
    Indexed Documents <span class="font-normal bg-gray-100 px-2 py-0.5 rounded-full text-[0.62rem]" x-text="corpusDocCount"></span>
  </div>
  <input x-model="corpusSearch" type="text" placeholder="Search documents..."
    class="mx-3 mt-2 mb-1 px-3 py-1.5 bg-[#f7f8fa] border border-[#e2e5ea] rounded-md text-sm outline-none focus:border-teal-400 focus:ring-1 focus:ring-teal-100">
  <div class="flex-1 overflow-y-auto py-1">
    <template x-for="cat in filteredCorpus" :key="cat.name">
      <div class="mb-1">
        <div @click="cat._open = !cat._open" class="flex items-center gap-1.5 px-4 py-2 text-[0.78rem] font-semibold text-gray-700 cursor-pointer hover:bg-gray-50">
          <span class="text-[0.55rem] text-gray-400 transition-transform" :class="cat._open && 'rotate-90'">&#9654;</span>
          <span x-text="cat.name"></span>
          <span class="ml-auto text-[0.6rem] text-gray-400 font-normal" x-text="`${cat.docs.length} docs`"></span>
        </div>
        <div x-show="cat._open" x-transition>
          <template x-for="doc in cat.docs" :key="doc.source">
            <div class="flex items-center gap-2 px-4 py-1.5 pl-8 text-[0.75rem] text-gray-500 cursor-pointer hover:bg-gray-50 hover:text-gray-700 border-l-2 border-transparent">
              <span class="text-[0.65rem]">&#128220;</span>
              <span x-text="doc.name" class="truncate"></span>
              <span class="ml-auto text-[0.6rem] text-gray-300" x-text="`${doc.pages}p`"></span>
            </div>
          </template>
        </div>
      </div>
    </template>
  </div>
</aside>
```

The center chat panel with status bar and messages:
```html
<main class="flex-1 flex flex-col min-w-0 bg-[#f7f8fa]">
  <div class="flex-1 overflow-y-auto px-5 py-6 space-y-5" id="chat-messages">
    <!-- Welcome screen -->
    <div x-show="messages.length === 0" class="max-w-lg mx-auto text-center py-16 text-gray-400">
      <h2 class="text-gray-600 text-xl font-semibold mb-3">Legal Compliance Assistant</h2>
      <p class="mb-6 text-sm leading-relaxed">Ask questions about Indian industrial and manufacturing law.</p>
      <div class="space-y-2">
        <template x-for="q in ['What is the definition of factory under the Factories Act?', 'What are the duties of an occupier regarding hazardous processes?', 'What is the penalty for obstructing an Inspector?', 'What capital subsidy is available for new industries in Chhattisgarh?']">
          <button @click="query = q; sendMessage()" class="block w-full text-left px-4 py-2.5 bg-white border border-[#e2e5ea] rounded-lg text-gray-500 text-sm hover:border-teal-300 hover:text-gray-700 transition" x-text="q"></button>
        </template>
      </div>
    </div>

    <!-- Messages -->
    <template x-for="(msg, i) in messages" :key="i">
      <div class="max-w-[720px] w-full mx-auto flex gap-3">
        <!-- Avatar -->
        <div class="w-7 h-7 rounded-lg flex items-center justify-center text-xs shrink-0 mt-0.5"
          :class="msg.role === 'user' ? 'bg-blue-50 text-blue-500' : 'bg-gradient-to-br from-teal-50 to-emerald-50 text-teal-600'">
          <span x-text="msg.role === 'user' ? '&#128100;' : '&#9878;&#65039;'"></span>
        </div>
        <div class="flex-1 min-w-0">
          <div class="text-[0.65rem] font-semibold uppercase tracking-wide text-gray-400 mb-1" x-text="msg.role === 'user' ? 'You' : 'LawMaster'"></div>

          <!-- Status bar (only for assistant, while streaming this message) -->
          <template x-if="msg.role === 'assistant' && !msg.done && streaming && currentStatus">
            <div class="mb-3 px-3 py-2 bg-teal-50 border border-teal-200 rounded-lg flex items-center gap-2.5 relative overflow-hidden">
              <div class="w-3 h-3 border-2 border-teal-200 border-t-teal-500 rounded-full animate-spin shrink-0"></div>
              <span class="text-sm text-teal-700 font-medium" x-text="currentStatus"></span>
              <div class="absolute bottom-0 left-0 h-0.5 w-1/3 bg-gradient-to-r from-transparent via-teal-400 to-transparent animate-[shimmer_1.8s_ease-in-out_infinite]"></div>
            </div>
          </template>

          <!-- Tool chip (after done) -->
          <template x-if="msg.role === 'assistant' && msg.done && msg.tools && msg.tools.length > 0">
            <div class="mb-2" x-data="{ expanded: false }">
              <button @click="expanded = !expanded"
                class="inline-flex items-center gap-1.5 px-3 py-1 bg-teal-50 border border-teal-200 rounded-full text-[0.72rem] text-teal-700 font-medium hover:bg-teal-100 transition">
                <span class="w-1.5 h-1.5 bg-teal-500 rounded-full"></span>
                <span x-text="`${msg.tools.length} tools used` + (msg.totalDuration ? ` \u00b7 ${msg.totalDuration}s` : '')"></span>
                <span class="text-teal-400" x-text="expanded ? '\u25B4' : '\u25BE'"></span>
              </button>
              <div x-show="expanded" x-transition class="mt-2 p-3 bg-white border border-[#e2e5ea] rounded-lg shadow-sm text-[0.73rem]">
                <template x-for="t in msg.tools" :key="t.display">
                  <div class="flex items-center gap-2 py-1 text-gray-500">
                    <span class="text-teal-500">&#10003;</span>
                    <span class="text-blue-600 font-medium" x-text="t.name"></span>
                    <span x-text="t.display.replace(/^[^:]+:\s*/, '')"></span>
                    <span class="ml-auto text-gray-300 text-[0.68rem]" x-text="t.duration !== null ? t.duration + 's' : '...'"></span>
                  </div>
                </template>
              </div>
            </div>
          </template>

          <!-- Content -->
          <div class="text-[0.88rem] leading-relaxed"
            :class="msg.isError ? 'text-red-600' : 'text-gray-700'"
            x-html="msg.role === 'assistant' ? formatMarkdown(msg.content) : msg.content">
          </div>
        </div>
      </div>
    </template>

    <!-- Streaming indicator when no status and no content yet -->
    <template x-if="streaming && !currentStatus && !currentContent">
      <div class="max-w-[720px] mx-auto text-gray-400 text-sm animate-pulse">Connecting...</div>
    </template>
  </div>

  <!-- Input -->
  <div class="px-5 py-3 border-t border-[#e2e5ea] bg-white">
    <div class="max-w-[720px] mx-auto flex gap-2 items-end">
      <input x-model="query" @keydown="handleInputKeydown($event)"
        type="text" placeholder="Ask about Indian industrial law..."
        class="flex-1 px-4 py-2.5 bg-white border border-[#e2e5ea] rounded-lg text-sm outline-none shadow-sm focus:border-teal-400 focus:ring-1 focus:ring-teal-100"
        :disabled="streaming">
      <button x-show="!streaming" @click="sendMessage()"
        class="px-5 py-2.5 bg-teal-600 text-white rounded-lg text-sm font-medium hover:bg-teal-700 shadow-sm transition">Send</button>
      <button x-show="streaming" @click="streaming = false"
        class="px-4 py-2.5 bg-red-50 text-red-600 border border-red-200 rounded-lg text-sm font-medium hover:bg-red-100 transition">&#9632; Stop</button>
    </div>
  </div>
</main>
```

The right panel (provenance):
```html
<aside x-show="rightOpen" x-transition.origin.right class="w-[340px] border-l border-[#e2e5ea] bg-white flex flex-col overflow-hidden shrink-0">
  <div class="px-4 py-3 text-[0.7rem] font-semibold uppercase tracking-wider text-gray-400 border-b border-[#e2e5ea]">
    Sources & Provenance
  </div>
  <div class="flex-1 overflow-y-auto">
    <template x-if="citations.length === 0">
      <div class="px-4 py-8 text-center text-gray-400 text-sm">
        Click a citation badge <span class="bg-teal-50 text-teal-700 text-[0.6rem] font-bold px-1.5 rounded border border-teal-200">1</span> to view the source.
      </div>
    </template>
    <template x-for="cite in citations" :key="cite.index">
      <div :id="`citation-${cite.index}`" class="p-4 border-b border-[#e2e5ea]"
        :class="activeCitation && activeCitation.index === cite.index ? 'bg-teal-50/50' : ''">
        <div class="text-[0.68rem] font-semibold uppercase tracking-wide text-gray-400 mb-2" x-text="`Source [${cite.index}]`"></div>
        <div class="bg-[#f7f8fa] border border-[#e2e5ea] rounded-lg p-3 shadow-sm">
          <div class="flex items-center gap-1.5 mb-2">
            <span class="text-[0.65rem]">&#128220;</span>
            <span class="text-[0.78rem] font-semibold text-[#0f1117]" x-text="cite.source"></span>
            <span class="ml-auto text-[0.65rem] text-gray-400" x-text="cite.page ? `p. ${cite.page}` : ''"></span>
          </div>
          <div class="text-[0.78rem] text-gray-500 leading-relaxed max-h-28 overflow-y-auto border-l-3 border-teal-400 pl-3"
            x-html="cite.snippet || 'Loading...'">
          </div>
        </div>
      </div>
    </template>
  </div>
</aside>
```

The toast:
```html
<template x-if="toast">
  <div class="fixed top-16 right-5 z-50 bg-white border border-amber-200 rounded-xl p-4 flex items-center gap-3 shadow-lg max-w-sm animate-[slideIn_0.3s_ease]">
    <span class="text-lg">&#9888;&#65039;</span>
    <div class="flex-1">
      <div class="text-sm font-semibold text-amber-600" x-text="`${toast.service} \u2014 ${toast.code}`"></div>
      <div class="text-xs text-gray-500 mt-0.5" x-text="toast.message"></div>
    </div>
    <button @click="retryLast()" class="px-3 py-1 bg-amber-50 border border-amber-200 rounded text-xs text-amber-600 font-medium hover:bg-amber-100">Retry</button>
    <button @click="toast = null" class="text-gray-300 hover:text-gray-500">&times;</button>
  </div>
</template>
```

Combine all sections into one complete file with `<script src="https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js" defer></script>` and `<script src="https://cdn.tailwindcss.com"></script>` in the head.

Add the shimmer keyframe to a `<style>` block:
```css
@keyframes shimmer { from { left: -30%; } to { left: 100%; } }
```

- [ ] **Step 4: Test the static layout**

Run:
```bash
/home/pc/anaconda3/envs/ml-env/bin/python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --loop asyncio
```

Open `http://localhost:8000` in browser. Verify:
- Three panels visible
- Corpus browser loads documents from `/api/corpus`
- Toggle buttons collapse/expand panels
- Welcome screen shows with example queries
- Input field accepts text, Enter sends

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: three-panel chat UI with Alpine.js and Tailwind"
```

---

### Task 4: Frontend — SSE Streaming Integration

**Files:**
- Modify: `static/index.html`

This task connects the Alpine.js frontend to the SSE `/chat/stream` endpoint. The JavaScript from Task 3 already includes the `sendMessage()` and `handleSSE()` methods — this task is about testing the integration and fixing any issues.

- [ ] **Step 1: Start the server and test a full query flow**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -m uvicorn src.server:app --host 0.0.0.0 --port 8000 --loop asyncio
```

Open browser, type "What is the definition of factory under the Factories Act?", press Enter. Verify:
- Status bar appears with "Searching: definition of factory..."
- Status text updates as tools fire
- Content tokens appear incrementally in the message
- When done, status bar collapses to "N tools used · Xs" chip
- Click chip to expand tool trace

- [ ] **Step 2: Test error handling**

Stop the server, try sending a query. Verify:
- Toast appears with connection error
- Retry button works

- [ ] **Step 3: Test keyboard shortcuts**

- `Ctrl+/` toggles left panel
- `Ctrl+.` toggles right panel
- `Up arrow` in empty input recalls last message
- `Escape` during streaming shows stop state

- [ ] **Step 4: Auto-scroll chat to bottom on new messages**

Add to `sendMessage()` after each content token:
```javascript
this.$nextTick(() => {
  const el = document.getElementById('chat-messages');
  el.scrollTop = el.scrollHeight;
});
```

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: connect SSE streaming to chat UI"
```

---

### Task 5: Frontend — Citation Click → Provenance Panel

**Files:**
- Modify: `static/index.html` (minor)
- Modify: `src/server.py` (minor — add citation extraction)

- [ ] **Step 1: Emit citation events from the backend**

After the agent finishes, extract chunk references from the response content and emit citation events. Add to the `run_agent()` function in `src/server.py`, before the `done` event:

```python
# Extract citations from response messages
import re
content_text = "".join(full_content)
# Find [Source: X] [Page: Y] patterns in the LightRAG response
source_pattern = re.findall(r'\[Source: ([^\]]+)\].*?\[Page: (\d+)\].*?\[Location: ([^\]]*)\]', content_text, re.DOTALL)
for i, (source, page, location) in enumerate(source_pattern[:5], 1):  # max 5 citations
    event_queue.put(("citation", {
        "index": i,
        "source": source,
        "page": int(page),
        "location": location,
        "snippet": "",  # populated by frontend via /api/chunks
    }))
```

Note: LightRAG's hybrid response includes source metadata inline. If the model's response doesn't contain structured source tags, the frontend can still render citations from the `[1]`, `[2]` patterns the model generates, with the chunk ID from the LightRAG response.

- [ ] **Step 2: Test citation click flow**

Send a query that triggers citations. Click a `[1]` badge. Verify:
- Right panel opens
- Source card appears with document name, page
- Panel scrolls to the citation

- [ ] **Step 3: Commit**

```bash
git add src/server.py static/index.html
git commit -m "feat: citation click opens provenance panel with source chunk"
```

---

### Task 6: Polish — Colors, Fonts, Welcome Screen

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add Inter font from Google Fonts**

In `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
```

Configure Tailwind:
```html
<script>
  tailwind.config = {
    theme: {
      extend: {
        fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
      },
    },
  }
</script>
```

- [ ] **Step 2: Update welcome screen examples to match corpus**

Replace the example queries with ones that exercise all three document categories:
```javascript
[
  'What is the definition of factory under the Factories Act?',
  'What are the hazardous waste disposal rules under HWM Rules 2016?',
  'What capital subsidy is available for new industries in Chhattisgarh?',
  'What is the permissible exposure limit for benzene?',
]
```

- [ ] **Step 3: Update the "Powered by" text**

Change from "Powered by LightRAG + Gemini Flash" to "Powered by LightRAG + Kimi K2.5".

- [ ] **Step 4: Test full flow end-to-end**

Run the server, open browser, test:
1. Welcome screen with 4 example queries
2. Click an example → streaming works
3. Send a custom query → tool events appear
4. Click citation → provenance panel opens
5. Toggle panels with buttons and keyboard
6. Error toast on connection failure
7. Up arrow recalls last query

- [ ] **Step 5: Commit**

```bash
git add static/index.html
git commit -m "feat: polish fonts, colors, welcome screen"
```

---

## Spec Coverage Check

| Spec Section | Task |
|-------------|------|
| 1. Layout (three-panel) | Task 3 |
| 2. Streaming & SSE | Tasks 1, 2, 4 |
| 3. Status Bar (Concept B) | Task 3 (template), Task 4 (integration) |
| 4. Provenance & Citations | Task 5 |
| 5. Error Handling | Task 2 (backend), Task 3 (toast template) |
| 6. Keyboard & UX | Task 3 (Alpine.js methods) |
| 7. Tech Stack | Task 3 (CDN imports) |
| 8. Backend Changes | Task 2 |
| 9. File Structure | All tasks |
| 10. Out of Scope | N/A (excluded by design) |
