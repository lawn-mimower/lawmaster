# LawMaster Frontend Redesign — Design Spec

**Date**: 2026-04-14
**Status**: Approved
**Audience**: Developer demoing to legal consultants
**Mockups**: `static/mockups/three-panel-light.html`, `static/mockups/thinking-steps-concepts.html`

---

## 1. Layout

Three-panel SPA. Adapted from Chanakya's `react-resizable-panels` architecture but built without React — vanilla JS with Alpine.js for reactivity.

```
┌─────────────┬──────────────────────────────┬───────────────┐
│  Corpus      │         Chat                 │  Provenance   │
│  Browser     │                              │  Panel        │
│  (~260px)    │      (flex: 1)               │  (~340px)     │
│              │                              │               │
│  Collapsible │  Status bar + streaming      │  Collapsible  │
│  Ctrl+/      │  Messages + citations        │  Ctrl+.       │
│              │  Input + stop/send           │  Opens on     │
│              │                              │  citation click│
└─────────────┴──────────────────────────────┴───────────────┘
```

### Left panel: Corpus browser
- Read-only. No upload, no pipeline controls.
- Tree: 3 categories → document list → page count badges.
- Client-side search filter over document names.
- Clicking a doc highlights it (active state).
- Data source: `GET /api/corpus` (reads extraction manifest).

### Center: Chat
- Message list with user/assistant bubbles.
- Status bar (Concept B) between question and answer.
- Streaming tokens append to assistant message in real-time.
- Citation badges `[1]` inline in response text.
- Sources footer below each response listing all citations.
- Input: auto-resize textarea, Enter to send, Shift+Enter newline.
- Stop button replaces Send during streaming.

### Right panel: Provenance
- Hidden by default. Opens on citation click or Ctrl+. toggle.
- Eagerly mounted in DOM (avoids Chanakya's lazy-init citation bug).
- Three sections:
  1. **Source chunks**: extracted text with `<mark>` highlighted query terms. Active citation scrolls into view via `scrollIntoView({ behavior: "smooth" })`.
  2. **Related SQL tables**: table name (monospace), row/col count, column names.
  3. **Document metadata**: source file, category, page count, chunk count.

### Panel mechanics
- Both sidebars collapsible via double-click on resize handle or keyboard shortcut.
- Panel sizes persisted to `localStorage`.
- Responsive: Desktop 3-panel → Tablet chat + one side → Mobile single panel.

---

## 2. Streaming & SSE Contract

Replace `POST /chat` (JSON batch response) with `POST /chat/stream` (Server-Sent Events).

### Event types

| Event | Payload | When |
|-------|---------|------|
| `tool_start` | `{ name, args_summary }` | Agent calls a tool |
| `tool_result` | `{ name, duration_s, success }` | Tool returns |
| `content` | `{ token }` | Answer token generated |
| `citation` | `{ index, chunk_id, source, page, location, snippet }` | After content, before done |
| `done` | `{ tool_count, total_duration_s }` | Stream complete |
| `error` | `{ type, service, code, message }` | On failure |

### Event ordering guarantee

```
tool_start → tool_result    (paired, interleaved with other tool pairs)
content*                    (1+ token chunks, strict order)
citation*                   (0+, all after content)
done                        (exactly 1, last event)
error                       (replaces done on failure)
```

### Frontend consumption

Native `fetch()` + `ReadableStream` reader. No library needed.

```javascript
const response = await fetch('/chat/stream', { method: 'POST', body: JSON.stringify({ query }) });
const reader = response.body.getReader();
const decoder = new TextDecoder();
// Read SSE lines, parse JSON, dispatch to Alpine.js reactive state
```

### Backend implementation

Run agent in a thread, bridge events to SSE via `asyncio.Queue`. Hook into Agno's tool call callbacks to emit `tool_start`/`tool_result`. Stream final response tokens. Catch exceptions and emit typed `error` events with service identification.

---

## 3. Status Bar (Concept B)

Single animated line between the user's question and the assistant's answer.

### States

**In-progress**: Teal-tinted bar with spinner + cycling text + shimmer underline.
- Text updates on each `tool_start` event with natural language: "Searching Factories Act..." not "search_legal_text()".
- Tool name → display name mapping: `search_legal_text` → "Searching {args}", `run_sql_query` → "Querying table {table}", `think` → "Planning approach...", `analyze` → "Analyzing results...".

**Completed**: Collapses to a pill chip: "4 tools used · 6.1s". Click expands trace panel showing each tool with name, description, and duration.

### No-tool queries

If the agent responds without calling any tools (e.g., "Hi, how are you?"), no status bar appears. Content tokens stream directly into the message.

---

## 4. Provenance & Citation Flow

Inspired by Chanakya's citation pipeline, simplified for our stack.

### The interaction

1. SSE `citation` event arrives with `chunk_id`, `source`, `page`, `location`, `snippet`.
2. Frontend renders `[N]` badge inline in the streamed markdown.
3. User clicks `[N]` → right panel opens (if collapsed) → panel scrolls to the matching source card → query terms highlighted with `<mark>`.

### Backend

New endpoint: `GET /api/chunks/{chunk_id}`

```json
{
  "text": "[Source: FactoryAct1948.pdf] [Category: ...] ...\n\nBenzene: TWA 10 ppm...",
  "source": "FactoryAct1948.pdf",
  "page": 55,
  "category": "Factories Act & Rules",
  "location": "Second Schedule",
  "type": "Section"
}
```

Reads directly from LightRAG's `kv_store_text_chunks.json`. No new database.

### Term highlighting

Simple regex: before rendering chunk text in the provenance panel, wrap query terms in `<mark>` tags. Terms extracted from the original user query (split on whitespace, filter stopwords).

### Related tables

When a citation's chunk text contains `[Type: Table Stub]` and `[SQL Table: ...]`, the provenance panel shows the linked SQL table card with name, row count, and columns from `_table_registry`.

---

## 5. Error Handling

### Transient errors (toast)

Triggered by SSE `error` events with `type: "transient"`.

- Top-right toast notification, amber-bordered.
- Shows: service name, HTTP code, human-readable message.
- "Retry" button re-sends the last query.
- Auto-dismiss after 15 seconds.

Service identification mapping:

| Error pattern | Service | Message |
|---------------|---------|---------|
| Together AI 503 | Together AI | "Kimi K2.5 is temporarily down. Usually resolves in 1-2 minutes." |
| Groq 429 | Groq | "Rate limited on Llama 3.3 70B. Retrying automatically..." |
| Groq 500/503 | Groq | "Groq service error. Safe to retry." |
| Connection timeout | Network | "Request timed out. Check your connection." |

### Permanent errors (inline)

Triggered by SSE `error` events with `type: "permanent"`.

- Rendered inside the assistant message bubble, red-tinted.
- No retry button.
- Examples: "This Act is not currently indexed.", "Empty query.", "Query too long."

---

## 6. Keyboard & UX

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | Newline in input |
| `Escape` (while streaming) | Stop generation |
| `Up arrow` (empty input) | Previous message from history |
| `Down arrow` (empty input) | Next message from history |
| `Ctrl+/` | Toggle left panel |
| `Ctrl+.` | Toggle right panel |

Message history stored in `sessionStorage` (per-tab, clears on close). Max 50 entries.

---

## 7. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Reactivity | Alpine.js (CDN, ~15KB) | No build step, declarative, sufficient for panel state + streaming |
| Styling | Tailwind CSS (CDN) | Utility classes, light theme tokens, responsive |
| Font | Inter (Google Fonts) or system-ui | Clean, professional |
| Markdown | Simple regex renderer (existing) | Extend with `<mark>` and `<cite>` support |
| Server | FastAPI + SSE (existing) | Add streaming endpoint |
| Build | None | Single HTML file, zero dependencies to install |

### Color palette (light theme)

| Token | Value | Usage |
|-------|-------|-------|
| Background | `#f7f8fa` | Page background |
| Card | `#ffffff` | Panels, message bubbles |
| Border | `#e2e5ea` | All borders |
| Text | `#1a1d24` | Body text |
| Text muted | `#6b7280` | Secondary text |
| Teal | `#0d9488` | Primary accent — buttons, active states, citations, status bar |
| Teal light | `#ccfbf1` | Citation badge bg, status bar bg |
| Blue | `#2563eb` | Tool names in trace, links |
| Red | `#dc2626` | Errors, stop button |
| Amber | `#d97706` | Toast warnings |

---

## 8. Backend Changes

### New endpoints

| Endpoint | Method | Response | Source |
|----------|--------|----------|--------|
| `POST /chat/stream` | POST | SSE stream | Agent run with event bridging |
| `GET /api/corpus` | GET | `{ categories: [{ name, docs: [{ name, pages, source }] }] }` | Extraction manifest JSON |
| `GET /api/chunks/{chunk_id}` | GET | `{ text, source, page, category, location, type }` | LightRAG kv_store_text_chunks.json |

### Modified

| Endpoint | Change |
|----------|--------|
| `POST /chat` | Keep for backward compat, but frontend uses `/chat/stream` |

### Agent per request

Fresh agent created per request — no session persistence, no history poisoning. The `session_id` from frontend is used only if multi-turn is needed later.

```python
@app.post("/chat/stream")
async def chat_stream(request: Request):
    body = await request.json()
    query = body.get("query", "")
    agent = create_agent(debug_mode=False)
    return StreamingResponse(event_generator(agent, query), media_type="text/event-stream")
```

### Event bridging

Agent runs in a thread. A queue bridges tool callbacks and response tokens to the SSE generator:

```python
import queue, threading

def event_generator(agent, query):
    q = queue.Queue()
    def run():
        # Hook into Agno callbacks, push events to queue
        response = agent.run(query)
        q.put(("done", {...}))
    threading.Thread(target=run).start()
    while True:
        event_type, data = q.get()
        yield f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        if event_type in ("done", "error"):
            break
```

---

## 9. File Structure

```
static/
  index.html              ← single-page app (all HTML + Alpine + Tailwind)
  mockups/                ← design mockups (not served in production)
src/
  server.py               ← FastAPI: SSE endpoint, corpus API, chunk API
  agent/
    agent.py              ← unchanged
    tools.py              ← unchanged
    rag.py                ← unchanged
    instructions.py       ← unchanged (user-modified version)
```

No new files in the agent layer. All frontend in one HTML file. Three new routes in `server.py`.

---

## 10. Out of Scope

- File upload / pipeline controls in the UI
- Multi-turn session persistence (agent is fresh per request)
- Dark theme toggle (light only for now)
- Mobile bottom-nav tabs (desktop-first for demo)
- PDF/DOCX preview in document viewer (markdown text only)
- Concept C floating pills (revisit after B ships)
