"""LawMaster agent factory."""

import os
from pathlib import Path
from agno.agent import Agent
from agno.models.together import Together
from agno.tools.sql import SQLTools
from agno.db.sqlite import SqliteDb

from src.agent.tools import LightRAGSearchTool
from src.agent.instructions import SYSTEM_INSTRUCTIONS


def create_agent(session_id=None, debug_mode=False) -> Agent:
    """Create a LawMaster compliance agent with LightRAG retrieval."""

    from src.config import TOGETHER_API_KEY
    os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

    project_root = Path(__file__).parent.parent.parent
    db_path = project_root / "data" / "tables.db"
    sessions_db = project_root / "data" / "sessions.db"

    # No ReasoningTools — Kimi K2.5 reasons natively via <reasoning> blocks
    tools = [
        LightRAGSearchTool(),
    ]

    # Add SQL tools if database exists
    if db_path.exists():
        tools.append(SQLTools(
            db_url=f"sqlite:///{db_path}",
        ))

    return Agent(
        name="LawMaster",
        model=Together(id="moonshotai/kimi-k2.5", max_tokens=16384),
        tools=tools,
        description="You are LawMaster, an expert on Indian industrial and manufacturing law.",
        instructions=SYSTEM_INSTRUCTIONS,
        db=SqliteDb(db_file=str(sessions_db)),
        session_id=session_id,
        add_history_to_context=True,
        num_history_runs=2,
        markdown=True,
        tool_call_limit=10,
        debug_mode=debug_mode,
    )
