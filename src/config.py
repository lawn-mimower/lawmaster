"""Shared configuration for LawMaster pipeline."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTRACTION_OUTPUT_DIR = PROJECT_ROOT / "extraction_output"
RAG_STORAGE_DIR = PROJECT_ROOT / "rag_storage"

DOCS_BASE = PROJECT_ROOT / "project 2 _ai tool for compliance -20260331T205330Z-1-001" / "project 2 _ai tool for compliance "

# PoC documents
ENGLISH_DOC = DOCS_BASE / "factories act" / "FactoryAct1948.pdf"
HINDI_DOC = DOCS_BASE / "industrial policy, ammendments and notifications" / "capital_subsidy_Rules.pdf"

# Load API keys
load_dotenv(PROJECT_ROOT / ".env", override=True)
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Model config
EXTRACTION_MODEL = "llama-3.1-8b-instant"   # Groq - KG entity extraction
CHATBOT_MODEL = "moonshotai/kimi-k2-instruct"  # Groq - agentic chatbot
