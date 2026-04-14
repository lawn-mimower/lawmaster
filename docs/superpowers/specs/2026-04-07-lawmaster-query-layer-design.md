# LawMaster Query Layer - Agno Agent Design

## Overview

The query layer is an Agno-based agentic chatbot that orchestrates legal compliance queries across multiple data sources: LightRAG (legal text), SQLite (structured tables), and an amendment index. It runs as a Streamlit app.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit Chat UI                     │
│    st.chat_input → agent.run(stream=True)             │
│    Session history, tool call display                  │
└────────────────────────┬─────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    Agno Agent        │
              │    Gemini Flash      │
              │    tool_call_limit=10│
              │    ReAct loop built-in│
              └──────────┬──────────┘
                         │ LLM decides which tool(s) to call
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼─────┐  ┌──────▼──────┐  ┌─────▼──────┐
    │ RAG Tool │  │  SQL Tool   │  │ Amendment  │
    │ LightRAG │  │  SQLite     │  │ Lookup Tool│
    │ .query() │  │  tables.db  │  │ JSON index │
    └──────────┘  └─────────────┘  └────────────┘
```

## File Structure

```
LawMaster/
  src/
    agent/
      __init__.py
      tools.py          # LightRAGTool, AmendmentTool
      agent.py          # Agent factory function
      instructions.py   # System prompt and instructions
    app.py              # Streamlit entry point
```

## Components

### 1. LightRAG Search Tool (`src/agent/tools.py`)

Wraps LightRAG as an Agno Toolkit. The agent calls this for legal text queries: definitions, sections, provisions, penalties.

```python
from agno.tools import Toolkit
from agno.agent import Agent
from lightrag import LightRAG, QueryParam

class LightRAGTool(Toolkit):
    def __init__(self, rag: LightRAG):
        super().__init__(
            name="legal_search",
            instructions=[
                "Use search_legal_text for questions about law definitions, sections, provisions, penalties, and procedures.",
                "Do NOT use this for questions about specific numbers, rates, or tabular data — use SQL for those.",
            ],
        )
        self.rag = rag
        self.register(self.search_legal_text)

    def search_legal_text(self, agent: Agent, query: str) -> str:
        """Search the legal knowledge base for relevant provisions, definitions, and sections.

        Args:
            query: Natural language question about Indian industrial/manufacturing law.

        Returns:
            str: Relevant legal text with source citations.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        result = self.rag.query(query, param=QueryParam(mode="mix", top_k=5))
        return str(result)
```

### 2. SQL Tool

Use Agno's built-in `SQLTools` with SQLite. This handles structured table queries: exposure limits, industry classifications, schedules.

```python
from agno.tools.sql import SQLTools

sql_tools = SQLTools(
    db_url="sqlite:///data/tables.db",
    list_tables=True,
    describe_table=True,
    run_sql_query=True,
)
```

This gives the agent three tools automatically: `list_tables`, `describe_table`, `run_sql_query`.

### 3. Amendment Lookup Tool (`src/agent/tools.py`)

Queries the amendment JSON index. When the agent answers about a section, it can check if amendments modify it.

```python
import json

class AmendmentTool(Toolkit):
    def __init__(self, index_path: str):
        super().__init__(
            name="amendments",
            instructions=[
                "Use lookup_amendments when answering about a specific section to check if it has been amended.",
                "Always surface amendments when the user asks about a section's current status.",
            ],
        )
        self.index_path = index_path
        self.register(self.lookup_amendments)

    def lookup_amendments(self, agent: Agent, section: str, act: str = "Factories Act, 1948") -> str:
        """Look up amendments for a specific section of an Act.

        Args:
            section: Section number (e.g., "41B", "2", "92").
            act: Name of the Act. Defaults to Factories Act, 1948.

        Returns:
            str: JSON array of amendments affecting this section, or "No amendments found."
        """
        with open(self.index_path, "r") as f:
            amendments = json.load(f)

        matches = [
            a for a in amendments
            if a.get("target_section", "").lower() == section.lower()
            and act.lower() in a.get("target_act", "").lower()
        ]

        if not matches:
            return f"No amendments found for Section {section} of {act}."
        return json.dumps(matches, indent=2, ensure_ascii=False)
```

### 4. Agent Factory (`src/agent/agent.py`)

Creates the LawMaster agent with all tools, Gemini backend, and session storage.

```python
import os
from pathlib import Path
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from agno.tools.sql import SQLTools
from agno.storage.sqlite import SqliteStorage

from src.agent.tools import LightRAGTool, AmendmentTool
from src.agent.instructions import SYSTEM_INSTRUCTIONS


def create_lawmaster_agent(
    rag_instance,
    session_id=None,
    debug_mode=False,
) -> Agent:
    """Create a LawMaster compliance agent."""

    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "tables.db"
    amendments_path = project_root / "data" / "amendments.json"
    sessions_db = project_root / "data" / "sessions.db"

    tools = [
        ReasoningTools(add_instructions=True),
        LightRAGTool(rag=rag_instance),
        SQLTools(
            db_url=f"sqlite:///{db_path}",
            list_tables=True,
            describe_table=True,
            run_sql_query=True,
        ),
    ]

    # Add amendment tool if index exists
    if amendments_path.exists():
        tools.append(AmendmentTool(index_path=str(amendments_path)))

    return Agent(
        name="LawMaster",
        model=Gemini(id="gemini-2.0-flash"),
        tools=tools,
        description="You are LawMaster, an expert on Indian industrial and manufacturing law.",
        instructions=SYSTEM_INSTRUCTIONS,
        storage=SqliteStorage(
            table_name="lawmaster_sessions",
            db_file=str(sessions_db),
        ),
        session_id=session_id,
        add_history_to_messages=True,
        num_history_responses=5,
        show_tool_calls=True,
        markdown=True,
        tool_call_limit=10,
        debug_mode=debug_mode,
    )
```

### 5. System Instructions (`src/agent/instructions.py`)

```python
SYSTEM_INSTRUCTIONS = [
    # Identity
    "You are LawMaster, an expert assistant for Indian industrial and manufacturing law compliance.",
    "You help compliance officers, factory owners, and legal consultants understand their obligations.",

    # Reasoning approach
    "Use the think() tool to plan your approach before searching.",
    "Use analyze() to evaluate whether you have enough information to answer.",

    # Tool selection
    "For LEGAL TEXT questions (definitions, sections, provisions, penalties, procedures):",
    "  → Use search_legal_text tool.",
    "For STRUCTURED DATA questions (exposure limits, industry classifications, fee amounts, rate tables):",
    "  → Use list_tables, then describe_table, then run_sql_query.",
    "For AMENDMENT questions (what changed in a section, current effective version):",
    "  → Use lookup_amendments after retrieving the original section.",
    "For questions requiring BOTH text and data:",
    "  → Call search_legal_text first, then run_sql_query to supplement with numbers.",

    # Multi-step reasoning
    "If your first search doesn't fully answer the question, search again with a refined query.",
    "If a legal provision references a Schedule or table, follow up with a SQL query.",
    "If a section has been amended, surface both the original and the amendment.",

    # Clarification
    "If the question is ambiguous (e.g., 'What are my compliance requirements?'), ask:",
    "  - What industry/sector?",
    "  - What scale (micro/small/medium/large)?",
    "  - Which state/jurisdiction?",
    "Do NOT ask for clarification if you can make a reasonable assumption.",

    # Corpus boundaries
    "If the question references an Act not in your knowledge base, clearly state:",
    "  'This Act is not currently indexed in LawMaster. The indexed corpus covers: Factories Act 1948, Chhattisgarh Industrial Policy, and Pollution Control Board regulations.'",

    # Citations
    "ALWAYS cite your sources: Act name, section number, chapter, and page when available.",
    "Format: (Source: Factories Act 1948, Section 41B, Chapter IVA)",

    # Response style
    "Lead with the answer, not the process.",
    "Use bullet points for multi-part provisions.",
    "Include the specific legal text when relevant, not just a summary.",
    "For Hindi documents, provide the key terms in both Hindi and English.",
]
```

### 6. Streamlit App (`src/app.py`)

```python
import streamlit as st
import nest_asyncio
from src.agent.agent import create_lawmaster_agent

nest_asyncio.apply()

st.set_page_config(page_title="LawMaster", page_icon="⚖️", layout="wide")
st.title("⚖️ LawMaster — Legal Compliance Assistant")

def init_agent():
    """Initialize or retrieve the LawMaster agent."""
    if "agent" not in st.session_state or st.session_state["agent"] is None:
        # Initialize LightRAG (loaded once)
        from src.agent.rag import get_rag_instance
        rag = get_rag_instance()
        agent = create_lawmaster_agent(
            rag_instance=rag,
            session_id=st.session_state.get("session_id"),
        )
        st.session_state["agent"] = agent
        st.session_state["session_id"] = agent.session_id
    return st.session_state["agent"]

agent = init_agent()

# Load session history
try:
    agent.load_session()
except Exception:
    pass

# Display chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about Indian industrial law compliance..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        tool_container = st.empty()
        resp_container = st.empty()
        response = ""

        with st.spinner("Researching..."):
            for chunk in agent.run(prompt, stream=True):
                if chunk.tools:
                    with tool_container:
                        for tool in chunk.tools:
                            st.caption(f"🔧 {tool.get('tool_name', 'tool')}: {tool.get('tool_args', {})}")
                if chunk.content:
                    response += chunk.content
                    resp_container.markdown(response)

        st.session_state["messages"].append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.markdown("### Indexed Corpus")
    st.markdown("- Factories Act, 1948 (English)")
    st.markdown("- Capital Subsidy Rules, 2024 (Hindi)")
    st.markdown("- *(more documents coming)*")

    st.markdown("### Tools Available")
    st.markdown("- 📚 Legal text search (LightRAG)")
    st.markdown("- 📊 Table lookup (SQL)")
    st.markdown("- 📝 Amendment checker")

    if st.button("🔄 New Chat"):
        st.session_state["agent"] = None
        st.session_state["messages"] = []
        st.rerun()
```

### 7. RAG Instance Loader (`src/agent/rag.py`)

Initializes LightRAG with the same config used during indexing. Loaded once, reused across queries.

```python
import os
import numpy as np
from pathlib import Path
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.gemini import gemini_model_complete
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_rag_instance = None
_bge_model = None

def _get_bge_model():
    global _bge_model
    if _bge_model is None:
        _bge_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)
    return _bge_model

async def bge_embedding_func(texts: list[str]) -> np.ndarray:
    return _get_bge_model().encode(texts, normalize_embeddings=True)

def get_rag_instance() -> LightRAG:
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    from src.config import RAG_STORAGE_DIR, GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    import asyncio, nest_asyncio
    nest_asyncio.apply()

    rag = LightRAG(
        working_dir=str(RAG_STORAGE_DIR),
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

    loop = asyncio.get_event_loop()
    loop.run_until_complete(rag.initialize_storages())

    _rag_instance = rag
    return _rag_instance
```

## Query Flow Examples

### Example 1: Pure RAG
**User:** "What is the definition of factory?"
1. Agent calls `think()` → "This is a definitional question, search legal text"
2. Agent calls `search_legal_text(query="definition of factory Factories Act")` → returns Section 2(m)
3. Agent responds with definition + citation

### Example 2: Pure SQL
**User:** "What is the permissible exposure limit for benzene?"
1. Agent calls `think()` → "This is a numeric lookup, check SQL tables"
2. Agent calls `list_tables()` → sees `factories_act_second_schedule`
3. Agent calls `run_sql_query("SELECT * FROM factories_act_second_schedule WHERE LOWER(substance) LIKE '%benzene%'")` → returns row
4. Agent responds: "The TWA for benzene is 10 ppm (Source: Factories Act, Second Schedule)"

### Example 3: RAG + SQL
**User:** "What safety measures are required for industries with hazardous processes?"
1. Agent calls `think()` → "Need provisions from the Act + the actual list of hazardous industries"
2. Agent calls `search_legal_text(query="safety measures hazardous processes")` → returns Sections 41A-41H
3. Agent calls `analyze()` → "The provision references the First Schedule. I should get the list."
4. Agent calls `run_sql_query("SELECT industry_name FROM factories_act_first_schedule LIMIT 20")` → returns list
5. Agent synthesizes: provisions + industry list + citations

### Example 4: Amendment-aware
**User:** "What does Section 2(m) currently say?"
1. Agent calls `search_legal_text(query="Section 2(m) Factories Act")` → returns original text
2. Agent calls `lookup_amendments(section="2(m)", act="Factories Act, 1948")` → returns amendments
3. Agent synthesizes: original + amendments into current effective version

### Example 5: Clarification needed
**User:** "What are my compliance requirements?"
1. Agent calls `think()` → "Too vague — need industry, scale, jurisdiction"
2. Agent responds: "To determine your compliance requirements, I need: (1) What industry/sector? (2) What scale? (3) Which state?"
(No tool calls needed — loop exits after LLM responds without calling tools)

### Example 6: Out of corpus
**User:** "What does the Water Act say about effluent discharge?"
1. Agent calls `search_legal_text(query="Water Act effluent discharge")` → no relevant results
2. Agent calls `analyze()` → "The Water Act is not in the indexed corpus"
3. Agent responds: "The Water (Prevention and Control of Pollution) Act, 1974 is not currently indexed in LawMaster..."

## Dependencies

Already installed in ml-env:
- `agno==2.5.2`
- `lightrag-hku==1.4.9.11`
- `sentence-transformers`
- `google-genai`

May need:
- `nest-asyncio` (for running async LightRAG inside Agno's sync tool calls)
- `sqlalchemy` (for Agno's SQLTools — check if installed)

## Running the App

```bash
cd ~/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/streamlit run src/app.py
```
