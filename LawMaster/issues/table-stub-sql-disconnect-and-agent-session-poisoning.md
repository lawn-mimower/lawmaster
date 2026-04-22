# Table stub / SQL disconnect and agent session poisoning

## Summary

Two issues block reliable end-to-end query answering:

1. **Table stub chunks in LightRAG have no reliable link to their SQL counterparts** -- the agent cannot programmatically go from a retrieved stub to the correct SQL table.
2. **Agent session history accumulation causes empty responses** -- after 2-3 queries the Kimi K2.5 model either stops calling tools or returns empty content.

---

## Issue 1: Table stub / SQL table disconnect

### What happens

The pipeline splits each HTML table into two representations:
- A **stub chunk** in LightRAG (searchable, contains column names and source metadata)
- A **full data table** in SQLite (queryable, contains actual rows)

The stub's job is to act as a bridge: when the KG retrieves a stub, the agent should know which SQL table to query. Currently there is **no explicit link** between them.

### Three specific problems

#### 1a. No foreign key or table name in the stub

The stub text contains only:
```
[Source: FactoryAct1948.pdf] [Category: Factories Act & Rules] [Type: Table Stub] [Page: 55]

Table from FactoryAct1948.pdf, page 55. Columns: Substance, Permissible limits of exposure.
For specific entries, query the SQL database.
```

The corresponding SQL table is `factoryact1948_p54`. The stub doesn't mention this name. The agent has to:
1. Call `list_tables` to get all table names
2. Guess which one matches based on source filename and page number
3. Account for the off-by-one page numbering (see below)

**Fix**: Add `[SQL Table: factoryact1948_p54]` to the stub text during routing.

#### 1b. Off-by-one page numbering

- Stub page: **1-indexed** (router uses `page+1` at `router.py:298,311`)
- SQL table name: **0-indexed** (pipeline uses raw `page` from OCR at `10_full_pipeline.py:392`)

So stub says "Page 55" but the SQL table is `_p54`. The agent has no way to know this.

Current match rate (verified):
```
LightRAG table stubs:       454
SQL tables:                  385
Stubs matchable via page-1:  442  (97%)
Stubs with no SQL match:      12  (orphaned)
Missing SQL (< 2 rows):       69  (filtered out by _insert_table_to_sql)
```

**Fix**: Use consistent page numbering. Either both 0-indexed or both 1-indexed.

#### 1c. 69 stubs point to SQL tables that don't exist

`_insert_table_to_sql` (`10_full_pipeline.py:382`) skips tables with fewer than 2 rows:
```python
if not table_data.get("rows") or len(table_data["rows"]) < 2:
    return
```

But the stub is still created in LightRAG by the router. The agent retrieves the stub, tries to query a table that was never created, and gets an error or no results.

**Fix**: Either also skip stub creation for <2-row tables, or lower the SQL insertion threshold.

### Code locations

| File | Line | What it does |
|------|------|-------------|
| `src/chunk/router.py` | 298,310-313 | Creates stub text with `page+1` |
| `scripts/10_full_pipeline.py` | 382-384 | Skips SQL insert for <2 rows |
| `scripts/10_full_pipeline.py` | 392 | Uses raw `page` (0-indexed) in table name |

---

## Issue 2: Agent session poisoning

### What happens

The agent returns correct, tool-grounded responses on the first 1-2 queries after a fresh start. Subsequent queries degrade to either empty responses or answers without tool calls (pure parametric knowledge).

### Reproduction

```python
from src.agent.agent import create_agent
agent = create_agent()

r1 = agent.run("What is the definition of factory?")
len(r1.content)   # 1847 -- correct, tools called

r2 = agent.run("What are hazardous waste rules?")
len(r2.content)   # 2058 -- correct, tools called

r3 = agent.run("What are the penalties under Factories Act?")
len(r3.content)   # 0 -- empty, despite 9 messages (tools WERE called)
```

Same Q3 on a fresh agent (no history): returns 2423 chars with full orchestration.

### Root cause

Three interacting factors:

**2a. Session history bloat**

Config: `add_history_to_context=True`, `num_history_runs=5`.

Each turn injects ~3000-5000 tokens of prior tool calls and results (think + search_legal_text + analyze + response). After 2-3 queries, the context includes 15,000-25,000 tokens of history. This causes Kimi K2.5 to:
- Decide it already has the information and skip tools
- Exhaust its `max_tokens=16384` budget on `<reasoning>` blocks before producing visible output

**2b. No session isolation between HTTP requests**

```python
# src/server.py
_agent = None  # module-level singleton, shared across ALL requests

def get_agent():
    global _agent
    if _agent is None:
        _agent = create_agent(debug_mode=False)  # no session_id
    return _agent
```

Agno auto-generates one session ID for the singleton. Every request from every user appends to the same session. The `data/sessions.db` already has 29 accumulated sessions.

**2c. Kimi K2.5 thinking tokens consume output budget**

Kimi K2.5 wraps all reasoning in `<reasoning>...</reasoning>` blocks. These count toward the `max_tokens=16384` output limit. With a large input context (system prompt + history + tool results), the model may produce extensive reasoning and then stop before generating any visible content.

Result: `response.content` is empty string (not None), `response.messages[-1].content` is also empty.

### Code locations

| File | Line | What it does |
|------|------|-------------|
| `src/agent/agent.py` | 43-45 | `add_history_to_context=True`, `num_history_runs=5` |
| `src/server.py` | 28-38 | Singleton agent, no per-request session |
| `src/agent/agent.py` | 38 | `max_tokens=16384` (shared with thinking tokens) |

### Proposed fix

```python
# src/server.py -- fresh agent per request
@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query", "")
    if not query.strip():
        return {"error": "Empty query"}

    agent = create_agent(debug_mode=False)
    response = agent.run(query)
    return {"response": response.content or ""}
```

Or if multi-turn is needed: pass a per-user `session_id` from the frontend and reduce `num_history_runs` to 2.

---

## Labels

`bug`, `agent`, `retrieval`, `tables`
