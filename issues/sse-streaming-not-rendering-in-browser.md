# SSE streaming works from curl but not in browser frontend

## Summary

The `/chat/stream` SSE endpoint returns correct events (tool_start, tool_result, content tokens, done) verified via curl. The browser frontend receives the HTTP 200 response but shows an infinite spinner — no status bar updates, no content rendering.

## Environment

- Backend: FastAPI `StreamingResponse` with `media_type="text/event-stream"`
- Frontend: Alpine.js 3 (CDN), `fetch` + `ReadableStream` for SSE parsing
- Browser: Chrome/Firefox on Linux
- Agent: Kimi K2.5 on Together AI via Agno (non-streaming `agent.run()`, content chunked post-hoc)

## Reproduction

### Backend works (curl)

```bash
curl -s -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the definition of factory?"}' 
```

Output (correct):
```
event: tool_start
data: {"name": "reasoning", "args_summary": "Thinking..."}

event: tool_start
data: {"name": "search_legal_text", "args_summary": "definition of factory Factories Act 1948 Section 2(m)"}

event: tool_result
data: {"name": "search_legal_text", "duration_s": 11.6, "success": true}

event: tool_result
data: {"name": "reasoning", "duration_s": 18.3, "success": true}

event: content
data: {"token": "## Definition of \"Factory\""}

event: content
data: {"token": "under Indian Law\n\nUnder"}

... (more content tokens)

event: done
data: {"total_duration_s": 18.3, "tool_count": 2}
```

### Frontend fails (browser)

1. Open http://localhost:8000
2. Type "What is the definition of factory?" and press Enter
3. Spinner appears ("Connecting...")
4. Server logs show `POST /chat/stream 200 OK`
5. Spinner never resolves — no status bar, no content, no "done" state
6. Browser console shows no SSE events logged (the `console.log('SSE event:', ...)` line never fires)

## Root cause analysis

### Suspect 1: ReadableStream buffering (most likely)

The frontend parses SSE by splitting on `\n`:
```javascript
const reader = res.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });
  const lines = buf.split('\n');
  buf = lines.pop();
  for (const line of lines) { ... }
}
```

`reader.read()` on a `fetch` response doesn't guarantee line-by-line delivery. The browser may buffer the entire chunked-transfer response and deliver it as one large `value` when the stream closes. This means no events fire until `done=true`, at which point the loop exits before processing them.

**Why curl works but browser doesn't:** curl processes chunks as they arrive on the TCP socket. The browser's fetch API may buffer differently depending on the `Content-Type` and transfer encoding.

**Fix options:**
- A. Use `EventSource` or a polyfill like `@microsoft/fetch-event-source` that handles SSE natively (but EventSource is GET-only, we need POST)
- B. Use `fetch-event-source` which supports POST + proper SSE parsing
- C. Add explicit flush/padding to the SSE generator to force chunk boundaries:
  ```python
  yield f": keepalive\n\n"  # SSE comment, forces a chunk flush
  ```
- D. Switch to a simpler approach: use the existing `POST /chat` endpoint (non-streaming) with a loading state, then add streaming later with proper SSE library

### Suspect 2: Alpine.js reactivity with object mutation

When we do `msg.content = this.currentContent` inside the read loop, Alpine.js may not detect the property change because the `msg` object was pushed to the `messages` array and Alpine tracks array changes, not deep property changes.

**Fix:** Use `this.messages = [...this.messages]` to trigger reactivity after each content update (expensive but reliable), or use Alpine's `$watch` / `Alpine.store`.

### Suspect 3: formatMarkdown binding not re-evaluating

The template uses:
```html
x-html="msg.role === 'assistant' ? formatMarkdown(msg.content) : escapeHtml(msg.content)"
```

If `msg.content` changes but Alpine doesn't re-evaluate `formatMarkdown()`, the DOM stays stale.

**Fix:** Move formatting to computed property or use `x-effect`.

## Event flow diagram

```
Browser sends POST /chat/stream
    │
    ▼
FastAPI spawns agent thread
    │
    ├── Immediately: event_queue.put("tool_start", "reasoning")
    │       → SSE yields "event: tool_start\ndata: {...}\n\n"
    │       → Browser fetch: reader.read() ← does this chunk arrive?
    │
    ├── 10-15s later: LightRAGSearchTool fires
    │       → event_queue.put("tool_start", "search_legal_text")
    │       → SSE yields another event
    │       → Browser: still waiting on first reader.read()?
    │
    ├── Agent completes: content chunks pushed to queue
    │       → SSE yields content events rapidly
    │
    └── event_queue.put("done")
            → SSE yields "event: done\n..."
            → Generator returns, stream closes
            → Browser: reader.read() returns done=true
            → NOW all buffered data arrives at once?
            → But the while loop already exited
```

## Recommended fix

**Option A (quick):** Replace `fetch` + `ReadableStream` with `fetch-event-source` from CDN:
```html
<script src="https://cdn.jsdelivr.net/npm/@microsoft/fetch-event-source@2.0.1/lib/cjs/index.min.js"></script>
```
This library handles POST SSE properly with per-event callbacks.

**Option B (simpler):** Fall back to the batch `/chat` endpoint for now. The three-panel UI, corpus browser, and provenance panel all work. Streaming can be added as an enhancement.

**Option C (investigate):** Add `console.log` inside the `reader.read()` loop to verify whether data arrives chunk-by-chunk or all-at-once. If all-at-once, add padding/flush hints to the SSE generator. If chunk-by-chunk, the issue is in the line parser or Alpine reactivity.

## Files involved

| File | Role |
|------|------|
| `src/server.py:138-230` | SSE endpoint — `sse_generator()` yields events |
| `static/index.html:382-415` | Frontend SSE reader — `fetch` + `ReadableStream` |
| `static/index.html:418-450` | `handleEvent()` — dispatches SSE events to Alpine state |

## Debug aids already in place

- `console.log('SSE event:', evtType, data)` at `index.html:408` — fires if events parse correctly
- Server prints 200 OK with request timing

## Labels

`bug`, `frontend`, `streaming`, `sse`
