# Vectorless RAG Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark PageIndex (vectorless tree-based retrieval) against LightRAG (vector + knowledge graph) on the LawMaster legal corpus, measuring RAGAS metrics, latency, and token usage.

**Architecture:** Two retriever modules behind a shared `RetrieverBase` interface, a benchmark harness that runs both on an auto-generated test set, and RAGAS evaluation with Claude-as-judge. All benchmark code lives in `benchmark/` — no changes to existing `src/`.

**Tech Stack:** PageIndex (self-hosted, GitHub clone), LightRAG (existing), LiteLLM + Groq (Kimi K2 0905), RAGAS 0.4.3, Anthropic Claude (judge), BGE-Large embeddings (existing).

---

## File Map

| File | Responsibility |
|------|---------------|
| `benchmark/__init__.py` | Package marker |
| `benchmark/retrievers/__init__.py` | Package marker |
| `benchmark/retrievers/base.py` | Abstract `RetrieverBase`, `RetrievedChunk`, `AnswerResult` dataclasses |
| `benchmark/retrievers/lightrag_retriever.py` | LightRAG wrapper using Groq/Kimi K2 via `QueryParam.model_func` |
| `benchmark/retrievers/pageindex_retriever.py` | Self-hosted PageIndex wrapper using Groq/Kimi K2 |
| `benchmark/eval/__init__.py` | Package marker |
| `benchmark/eval/generate_testset.py` | Script to auto-generate Q&A test set from source docs |
| `benchmark/eval/judge.py` | RAGAS evaluation with Claude-as-judge |
| `benchmark/run_benchmark.py` | Main harness: runs retrievers, collects metrics, triggers eval |
| `benchmark/testset.json` | Generated test set (version-controlled) |
| `benchmark/results/` | Output directory for raw results, summary, report |
| `benchmark/pageindex_storage/` | PageIndex tree indices (gitignored) |

---

### Task 1: Branch Setup and Dependencies

**Files:**
- Create: `benchmark/__init__.py`
- Create: `benchmark/retrievers/__init__.py`
- Create: `benchmark/eval/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create the feature branch**

```bash
cd /home/pc/Downloads/LawMaster
git checkout -b benchmark/vectorless-rag
```

- [ ] **Step 2: Install dependencies**

```bash
/home/pc/anaconda3/envs/ml-env/bin/pip install ragas anthropic litellm groq
```

PageIndex self-hosted is NOT a pip package. Clone it as a local dependency:

```bash
cd /home/pc/Downloads/LawMaster
git clone https://github.com/VectifyAI/PageIndex.git benchmark/pageindex_lib
```

Install PageIndex's own dependencies (pymupdf, PyPDF2, pyyaml):

```bash
/home/pc/anaconda3/envs/ml-env/bin/pip install pymupdf PyPDF2 pyyaml
```

- [ ] **Step 3: Verify imports work**

```bash
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import litellm; print('litellm', litellm.__version__)
import ragas; print('ragas', ragas.__version__)
import anthropic; print('anthropic', anthropic.__version__)
"
```

Expected: version numbers printed, no errors.

- [ ] **Step 4: Verify PageIndex self-hosted imports**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, 'benchmark/pageindex_lib')
from pageindex import PageIndexClient
print('PageIndex self-hosted imported OK')
"
```

Expected: `PageIndex self-hosted imported OK`

- [ ] **Step 5: Create package directories and .gitignore updates**

Create empty `__init__.py` files:

`benchmark/__init__.py`:
```python
```

`benchmark/retrievers/__init__.py`:
```python
```

`benchmark/eval/__init__.py`:
```python
```

Add to `.gitignore` (create if it doesn't exist):

```
benchmark/results/raw_results.json
benchmark/pageindex_storage/
benchmark/pageindex_lib/
```

- [ ] **Step 6: Add GROQ_API_KEY to .env**

Append to the existing `/home/pc/Downloads/LawMaster/.env`:

```
GROQ_API_KEY=<user's Groq API key>
```

The user must supply this. Print a reminder if it's missing.

- [ ] **Step 7: Commit**

```bash
git add benchmark/__init__.py benchmark/retrievers/__init__.py benchmark/eval/__init__.py .gitignore
git commit -m "Scaffold benchmark directory and install dependencies"
```

---

### Task 2: Retriever Base Interface

**Files:**
- Create: `benchmark/retrievers/base.py`

- [ ] **Step 1: Write the base module**

`benchmark/retrievers/base.py`:
```python
"""Abstract retriever interface and shared data classes for the benchmark."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """A single chunk returned by a retriever."""

    text: str
    source_document: str
    source_section: str
    page_range: str
    relevance_score: float | None = None


@dataclass
class AnswerResult:
    """Full result from a retriever's end-to-end pipeline."""

    answer: str
    retrieved_chunks: list[RetrievedChunk]
    retrieval_latency_ms: float = 0.0
    e2e_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RetrieverBase(ABC):
    """Interface that every benchmark retriever must implement."""

    name: str

    @abstractmethod
    def index(self, documents: list[dict]) -> None:
        """Build or load the retriever's index.

        Args:
            documents: List of dicts with keys 'path' (PDF path) and 'name' (display name).
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the top-k retrieved chunks for a query."""

    @abstractmethod
    def query_end_to_end(self, query: str) -> AnswerResult:
        """Full RAG pipeline: retrieve context, synthesize answer, track metrics."""
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from benchmark.retrievers.base import RetrieverBase, RetrievedChunk, AnswerResult
print('Base interface OK')
print('RetrievedChunk fields:', [f.name for f in RetrievedChunk.__dataclass_fields__.values()])
print('AnswerResult fields:', [f.name for f in AnswerResult.__dataclass_fields__.values()])
"
```

Expected:
```
Base interface OK
RetrievedChunk fields: ['text', 'source_document', 'source_section', 'page_range', 'relevance_score']
AnswerResult fields: ['answer', 'retrieved_chunks', 'retrieval_latency_ms', 'e2e_latency_ms', 'prompt_tokens', 'completion_tokens', 'total_tokens']
```

- [ ] **Step 3: Commit**

```bash
git add benchmark/retrievers/base.py
git commit -m "Add retriever base interface and data classes"
```

---

### Task 3: LightRAG Retriever

**Files:**
- Create: `benchmark/retrievers/lightrag_retriever.py`

**Context needed:**
- Existing LightRAG init: `src/agent/rag.py` (BGE-Large embeddings, Gemini LLM, `rag_storage/`)
- LightRAG supports `QueryParam.model_func` to override the LLM at query time
- The custom `model_func` signature: `async def func(prompt, *, system_prompt=None, history_messages=None, keyword_extraction=False, enable_cot=False, stream=False, **kwargs) -> str`
- Token usage must be captured inside the custom function via a closure

- [ ] **Step 1: Write the LightRAG retriever**

`benchmark/retrievers/lightrag_retriever.py`:
```python
"""LightRAG retriever wrapper for the benchmark.

Uses the existing rag_storage/ index with BGE-Large embeddings.
Swaps the LLM to Kimi K2 via Groq at query time using QueryParam.model_func.
"""

from __future__ import annotations

import os
import re
import time
import asyncio
from pathlib import Path

import litellm
import numpy as np
import nest_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

from benchmark.retrievers.base import AnswerResult, RetrievedChunk, RetrieverBase

nest_asyncio.apply()

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

GROQ_MODEL = "groq/kimi-k2-0905"


class TokenAccumulator:
    """Accumulates token usage across multiple LLM calls within a single query."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage):
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0


class LightRAGRetriever(RetrieverBase):
    """Wraps the existing LightRAG index, using Groq/Kimi K2 for query-time LLM."""

    name = "lightrag"

    def __init__(self, rag_storage_dir: str | Path):
        self.rag_storage_dir = Path(rag_storage_dir)
        self.rag: LightRAG | None = None
        self._bge_model: SentenceTransformer | None = None
        self._token_acc = TokenAccumulator()

    def _get_bge_model(self) -> SentenceTransformer:
        if self._bge_model is None:
            print("[LightRAG] Loading BGE-large (offline)...")
            self._bge_model = SentenceTransformer(
                "BAAI/bge-large-en-v1.5", local_files_only=True
            )
        return self._bge_model

    async def _bge_embedding_func(self, texts: list[str]) -> np.ndarray:
        return self._get_bge_model().encode(texts, normalize_embeddings=True)

    async def _groq_llm_func(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        enable_cot: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> str:
        """LiteLLM-based LLM function for LightRAG query-time override."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response_format = None
        if keyword_extraction:
            response_format = {"type": "json_object"}

        response = await litellm.acompletion(
            model=GROQ_MODEL,
            messages=messages,
            response_format=response_format,
        )

        self._token_acc.add(response.usage)
        return response.choices[0].message.content

    def index(self, documents: list[dict]) -> None:
        """Load the existing LightRAG index (no re-indexing)."""
        from lightrag.llm.gemini import gemini_model_complete

        self.rag = LightRAG(
            working_dir=str(self.rag_storage_dir),
            llm_model_func=gemini_model_complete,
            llm_model_name="gemini-2.0-flash",
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=self._bge_embedding_func,
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
        loop.run_until_complete(self.rag.initialize_storages())
        graph = self.rag.chunk_entity_relation_graph._graph
        print(f"[LightRAG] Loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve chunks via LightRAG mix mode (vector + knowledge graph)."""
        result_text = self.rag.query(
            query,
            param=QueryParam(
                mode="mix",
                top_k=top_k,
                model_func=self._groq_llm_func,
            ),
        )
        return [
            RetrievedChunk(
                text=str(result_text),
                source_document="LightRAG mix retrieval",
                source_section="",
                page_range="",
                relevance_score=None,
            )
        ]

    def query_end_to_end(self, query: str) -> AnswerResult:
        """Full pipeline: retrieve + synthesize with Kimi K2 via Groq."""
        self._token_acc.reset()

        t_start = time.perf_counter()

        result_text = self.rag.query(
            query,
            param=QueryParam(
                mode="mix",
                top_k=5,
                model_func=self._groq_llm_func,
            ),
        )

        e2e_ms = (time.perf_counter() - t_start) * 1000

        chunks = self._parse_lightrag_response(str(result_text))

        return AnswerResult(
            answer=str(result_text),
            retrieved_chunks=chunks,
            retrieval_latency_ms=e2e_ms,
            e2e_latency_ms=e2e_ms,
            prompt_tokens=self._token_acc.prompt_tokens,
            completion_tokens=self._token_acc.completion_tokens,
            total_tokens=self._token_acc.total_tokens,
        )

    def _parse_lightrag_response(self, text: str) -> list[RetrievedChunk]:
        """Extract section references from LightRAG's response text."""
        section_pattern = r"[Ss]ection\s+(\d+[A-Za-z]*(?:\(\d+\))?)"
        sections_found = re.findall(section_pattern, text)
        if not sections_found:
            return [
                RetrievedChunk(
                    text=text,
                    source_document="Factories Act, 1948",
                    source_section="",
                    page_range="",
                )
            ]
        return [
            RetrievedChunk(
                text=text,
                source_document="Factories Act, 1948",
                source_section=f"Section {s}",
                page_range="",
            )
            for s in dict.fromkeys(sections_found)
        ]
```

- [ ] **Step 2: Verify it loads the existing index**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from benchmark.retrievers.lightrag_retriever import LightRAGRetriever
r = LightRAGRetriever('rag_storage')
r.index([])
print('LightRAG retriever loaded successfully')
"
```

Expected: prints graph node/edge counts and `LightRAG retriever loaded successfully`.

- [ ] **Step 3: Smoke-test a query (requires GROQ_API_KEY in .env)**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from benchmark.retrievers.lightrag_retriever import LightRAGRetriever
r = LightRAGRetriever('rag_storage')
r.index([])
result = r.query_end_to_end('What is the definition of factory under the Factories Act?')
print(f'Answer length: {len(result.answer)} chars')
print(f'Tokens: {result.prompt_tokens} prompt, {result.completion_tokens} completion')
print(f'Latency: {result.e2e_latency_ms:.0f} ms')
print(f'Chunks: {len(result.retrieved_chunks)}')
"
```

Expected: non-empty answer, non-zero tokens, latency in the hundreds of ms range.

- [ ] **Step 4: Commit**

```bash
git add benchmark/retrievers/lightrag_retriever.py
git commit -m "Add LightRAG retriever with Groq/Kimi K2 query-time override"
```

---

### Task 4: PageIndex Retriever

**Files:**
- Create: `benchmark/retrievers/pageindex_retriever.py`

**Context needed:**
- Self-hosted PageIndex is cloned to `benchmark/pageindex_lib/`
- Constructor: `PageIndexClient(model="groq/kimi-k2-0905", workspace="./benchmark/pageindex_storage")`
- `client.index(file_path)` returns `doc_id`
- `client.get_document_structure(doc_id)` returns the JSON tree
- `client.get_page_content(doc_id, pages="5-7")` returns page text
- Token tracking via LiteLLM callbacks since PageIndex uses `litellm.acompletion()` internally

- [ ] **Step 1: Write the PageIndex retriever**

`benchmark/retrievers/pageindex_retriever.py`:
```python
"""PageIndex (self-hosted) retriever wrapper for the benchmark.

Uses the VectifyAI/PageIndex GitHub clone with LiteLLM routing to Groq/Kimi K2.
Builds a hierarchical tree index per PDF and uses LLM-driven tree navigation for retrieval.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import litellm

from benchmark.retrievers.base import AnswerResult, RetrievedChunk, RetrieverBase

GROQ_MODEL = "groq/kimi-k2-0905"

# The self-hosted PageIndex is cloned into benchmark/pageindex_lib/
_PAGEINDEX_LIB = Path(__file__).parent.parent / "pageindex_lib"


def _get_pageindex_client():
    """Import and construct the self-hosted PageIndex client."""
    if str(_PAGEINDEX_LIB) not in sys.path:
        sys.path.insert(0, str(_PAGEINDEX_LIB))
    from pageindex import PageIndexClient

    workspace = str(Path(__file__).parent.parent / "pageindex_storage")
    return PageIndexClient(model=GROQ_MODEL, workspace=workspace)


class LiteLLMTokenCallback(litellm.Callback):
    """LiteLLM callback to capture token usage from PageIndex's internal LLM calls."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        usage = getattr(response_obj, "usage", None)
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0


class PageIndexRetriever(RetrieverBase):
    """Wraps the self-hosted PageIndex for vectorless tree-based retrieval."""

    name = "pageindex"

    def __init__(self):
        self.client = None
        self.doc_ids: dict[str, str] = {}  # doc_name -> doc_id
        self._token_cb = LiteLLMTokenCallback()

    def index(self, documents: list[dict]) -> None:
        """Build PageIndex tree indices for each document.

        Args:
            documents: List of dicts with 'path' (PDF path) and 'name' (display name).
        """
        self.client = _get_pageindex_client()

        # Register the token callback
        if self._token_cb not in litellm.callbacks:
            litellm.callbacks.append(self._token_cb)

        for doc in documents:
            pdf_path = doc["path"]
            doc_name = doc["name"]
            print(f"[PageIndex] Indexing {doc_name} from {pdf_path}...")
            doc_id = self.client.index(pdf_path)
            self.doc_ids[doc_name] = doc_id
            print(f"[PageIndex] Indexed {doc_name} -> doc_id={doc_id}")

            # Print tree structure summary
            tree_json = self.client.get_document_structure(doc_id)
            tree = json.loads(tree_json) if isinstance(tree_json, str) else tree_json
            doc_info = self.client.get_document(doc_id)
            print(f"[PageIndex] {doc_name}: {doc_info}")

    def _get_doc_id_for_query(self, query: str) -> str:
        """Pick the most relevant doc_id based on query keywords."""
        query_lower = query.lower()
        for name, doc_id in self.doc_ids.items():
            if "factor" in query_lower and "factor" in name.lower():
                return doc_id
            if "subsid" in query_lower and "subsid" in name.lower():
                return doc_id
            if "capital" in query_lower and "capital" in name.lower():
                return doc_id
        # Default to first document
        return list(self.doc_ids.values())[0]

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Retrieve by navigating the PageIndex tree.

        PageIndex fuses retrieval and reasoning. We use get_document_structure
        to let the LLM navigate, then get_page_content for the selected pages.
        """
        doc_id = self._get_doc_id_for_query(query)
        tree_json = self.client.get_document_structure(doc_id)

        # Ask LLM to select relevant pages from the tree
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document retrieval assistant. Given a document's tree structure "
                    "and a query, identify the most relevant pages. Return ONLY a JSON object "
                    'with key "pages" containing a comma-separated page string like "3,5-7,12". '
                    "No explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Document structure:\n{tree_json}\n\nQuery: {query}",
            },
        ]

        response = litellm.completion(
            model=GROQ_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
        )

        # Parse page selection
        try:
            page_data = json.loads(response.choices[0].message.content)
            pages_str = str(page_data.get("pages", "1"))
        except (json.JSONDecodeError, KeyError):
            pages_str = "1"

        # Fetch selected pages
        page_content = self.client.get_page_content(doc_id, pages=pages_str)

        doc_name = next(
            (n for n, did in self.doc_ids.items() if did == doc_id),
            "Unknown",
        )

        return [
            RetrievedChunk(
                text=str(page_content),
                source_document=doc_name,
                source_section="",
                page_range=pages_str,
            )
        ]

    def query_end_to_end(self, query: str) -> AnswerResult:
        """Full pipeline: tree navigation -> page retrieval -> answer synthesis."""
        self._token_cb.reset()

        t_start = time.perf_counter()

        # Step 1: Retrieve relevant pages
        chunks = self.retrieve(query, top_k=5)

        t_retrieval = (time.perf_counter() - t_start) * 1000

        # Step 2: Synthesize answer from retrieved content
        context = "\n\n".join(c.text for c in chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal expert on Indian industrial and manufacturing law. "
                    "Answer the question based ONLY on the provided context. "
                    "Cite specific section numbers and page references."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]

        response = litellm.completion(model=GROQ_MODEL, messages=messages)
        answer = response.choices[0].message.content

        e2e_ms = (time.perf_counter() - t_start) * 1000

        return AnswerResult(
            answer=answer,
            retrieved_chunks=chunks,
            retrieval_latency_ms=t_retrieval,
            e2e_latency_ms=e2e_ms,
            prompt_tokens=self._token_cb.prompt_tokens,
            completion_tokens=self._token_cb.completion_tokens,
            total_tokens=self._token_cb.total_tokens,
        )
```

- [ ] **Step 2: Verify the module imports cleanly**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from benchmark.retrievers.pageindex_retriever import PageIndexRetriever
print('PageIndexRetriever imported OK')
"
```

Expected: `PageIndexRetriever imported OK`

- [ ] **Step 3: Test index building with one PDF (requires GROQ_API_KEY)**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from benchmark.retrievers.pageindex_retriever import PageIndexRetriever
from src.config import ENGLISH_DOC

r = PageIndexRetriever()
r.index([{'path': str(ENGLISH_DOC), 'name': 'Factories Act, 1948'}])
print(f'Indexed doc_ids: {r.doc_ids}')
"
```

Expected: PageIndex builds a tree index and prints the doc_id. This may take a few minutes as the LLM generates summaries for each tree node.

- [ ] **Step 4: Smoke-test a query**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv()
from benchmark.retrievers.pageindex_retriever import PageIndexRetriever
from src.config import ENGLISH_DOC

r = PageIndexRetriever()
r.index([{'path': str(ENGLISH_DOC), 'name': 'Factories Act, 1948'}])
result = r.query_end_to_end('What is the definition of factory under the Factories Act?')
print(f'Answer length: {len(result.answer)} chars')
print(f'Tokens: {result.prompt_tokens} prompt, {result.completion_tokens} completion')
print(f'Retrieval latency: {result.retrieval_latency_ms:.0f} ms')
print(f'E2E latency: {result.e2e_latency_ms:.0f} ms')
print(f'Pages retrieved: {result.retrieved_chunks[0].page_range}')
"
```

Expected: non-empty answer with section references, non-zero tokens, page range like "3-5".

- [ ] **Step 5: Commit**

```bash
git add benchmark/retrievers/pageindex_retriever.py
git commit -m "Add PageIndex retriever with Groq/Kimi K2 tree-based retrieval"
```

---

### Task 5: Test Set Generation

**Files:**
- Create: `benchmark/eval/generate_testset.py`
- Create: `benchmark/testset.json` (output)

**Context needed:**
- Source docs: `extraction_output/FactoryAct1948_docling.md` and `extraction_output/capital_subsidy_Rules_docling.md`
- This script is run by a Claude Code subagent or manually
- Generates ~28 questions across 6 types with ground-truth answers

- [ ] **Step 1: Write the test set generation script**

`benchmark/eval/generate_testset.py`:
```python
"""Generate a benchmark test set from source legal documents.

Reads the extracted markdown documents and produces Q&A pairs across
6 query types: exact_lookup, definition, penalty, multi_hop, numerical, cross_document.

Usage:
    python benchmark/eval/generate_testset.py

This script uses Groq/Kimi K2 to generate questions with ground-truth answers.
The output is saved to benchmark/testset.json for manual review and version control.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import litellm
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_MODEL = "groq/kimi-k2-0905"

GENERATION_PROMPT = """\
You are a legal domain expert creating a benchmark test set for evaluating RAG systems on Indian legal documents.

You have been given the full text of two legal documents:
1. The Factories Act, 1948 (English)
2. Capital Subsidy Rules (Hindi-origin, translated)

Generate exactly 28 question-answer pairs across these categories:

1. **exact_lookup** (6 questions): Questions asking about a specific section's content.
   Example: "What does Section 7A of the Factories Act state?"

2. **definition** (5 questions): Questions about legal definitions.
   Example: "How is 'hazardous process' defined under the Factories Act?"

3. **penalty** (4 questions): Questions about penalties, fines, or consequences.
   Example: "What is the penalty for obstructing an inspector under the Factories Act?"

4. **multi_hop** (5 questions): Questions requiring connecting information across multiple sections.
   Example: "Which authority oversees factories that use processes listed in the First Schedule, and what powers do they have?"

5. **numerical** (4 questions): Questions about specific numbers, limits, or rates.
   Example: "What is the maximum number of working hours per week allowed under the Factories Act?"

6. **cross_document** (4 questions): Questions that span both documents.
   Example: "How do the capital subsidy provisions interact with factory compliance requirements?"

For EACH question, provide:
- The exact question text
- A comprehensive expected answer (2-5 sentences) citing specific sections
- The source section(s) referenced
- The source document name
- The approximate page range where the answer can be found
- The query type category

Return a JSON array of objects with these exact keys:
{
  "id": "q01",
  "question": "...",
  "expected_answer": "...",
  "source_section": "Section X",
  "source_document": "Factories Act, 1948",
  "source_pages": "3-4",
  "query_type": "definition"
}

IMPORTANT:
- Use ONLY information present in the provided documents
- Expected answers must be factually grounded in the source text
- Include specific section numbers, clause references, and page numbers
- For cross_document questions, reference both documents
- Return ONLY the JSON array, no other text
"""


def load_source_docs() -> str:
    """Load the extracted markdown documents."""
    extraction_dir = PROJECT_ROOT / "extraction_output"

    factories_md = extraction_dir / "FactoryAct1948_docling.md"
    subsidy_md = extraction_dir / "capital_subsidy_Rules_docling.md"

    texts = []
    for path, label in [
        (factories_md, "DOCUMENT 1: Factories Act, 1948"),
        (subsidy_md, "DOCUMENT 2: Capital Subsidy Rules"),
    ]:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            texts.append(f"=== {label} ===\n\n{content}")
        else:
            print(f"WARNING: {path} not found, skipping")

    return "\n\n" + "=" * 80 + "\n\n".join(texts)


def generate_testset() -> dict:
    """Generate the test set using Groq/Kimi K2."""
    source_text = load_source_docs()

    print(f"Source text length: {len(source_text)} chars")
    print("Generating test set with Kimi K2 via Groq...")

    response = litellm.completion(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": source_text},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content

    # Parse - the model might wrap in {"questions": [...]} or return bare array
    parsed = json.loads(raw)
    if isinstance(parsed, dict) and "questions" in parsed:
        questions = parsed["questions"]
    elif isinstance(parsed, list):
        questions = parsed
    else:
        raise ValueError(f"Unexpected response format: {type(parsed)}")

    print(f"Generated {len(questions)} questions")

    # Validate structure
    required_keys = {"id", "question", "expected_answer", "source_section", "source_document", "source_pages", "query_type"}
    for i, q in enumerate(questions):
        missing = required_keys - set(q.keys())
        if missing:
            print(f"WARNING: Question {i} missing keys: {missing}")

    # Build final test set
    testset = {
        "metadata": {
            "generated_by": "generate_testset.py (Kimi K2 via Groq)",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_documents": [
                "FactoryAct1948_docling.md",
                "capital_subsidy_Rules_docling.md",
            ],
            "total_questions": len(questions),
        },
        "questions": questions,
    }

    return testset


def main():
    testset = generate_testset()

    output_path = PROJECT_ROOT / "benchmark" / "testset.json"
    output_path.write_text(json.dumps(testset, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {output_path}")

    # Print summary by type
    from collections import Counter
    types = Counter(q.get("query_type", "unknown") for q in testset["questions"])
    print("\nBreakdown by query type:")
    for qt, count in sorted(types.items()):
        print(f"  {qt}: {count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generation script**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python benchmark/eval/generate_testset.py
```

Expected: generates `benchmark/testset.json` with ~28 questions. Review the output manually to verify answer quality.

- [ ] **Step 3: Manually review and adjust testset.json**

Open `benchmark/testset.json` and verify:
- Questions are clear and answerable from the source docs
- Expected answers cite specific sections
- Source pages are plausible
- All 6 query types are represented
- No hallucinated sections or facts

Fix any issues by hand if needed.

- [ ] **Step 4: Commit**

```bash
git add benchmark/eval/generate_testset.py benchmark/testset.json
git commit -m "Add test set generator and initial 28-question benchmark set"
```

---

### Task 6: RAGAS Evaluation Module

**Files:**
- Create: `benchmark/eval/judge.py`

**Context needed:**
- RAGAS v0.4.3: `evaluate()` deprecated but works, column names: `user_input`, `response`, `retrieved_contexts`, `reference`
- Judge LLM: Claude via `llm_factory("claude-sonnet-4-20250514", provider="anthropic", client=Anthropic())`
- Embeddings needed for AnswerRelevancy and AnswerCorrectness: use existing BGE-Large
- `AnswerRelevancy` and `AnswerCorrectness` need an embedding model. Use `HuggingfaceEmbeddings` from RAGAS or pass embeddings config.

- [ ] **Step 1: Write the judge module**

`benchmark/eval/judge.py`:
```python
"""RAGAS evaluation with Claude-as-judge.

Runs Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall,
and AnswerCorrectness on benchmark results. Uses Anthropic Claude as the
judge LLM and BGE-Large for embedding-based metrics.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import Anthropic
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)
from ragas.llms import llm_factory

# ContextPrecision and ContextRecall may be under different names in v0.4.3
try:
    from ragas.metrics import ContextPrecision, ContextRecall
except ImportError:
    from ragas.metrics import LLMContextPrecisionWithReference as ContextPrecision
    from ragas.metrics import LLMContextRecall as ContextRecall


def build_ragas_dataset(
    raw_results: list[dict],
    retriever_name: str,
) -> EvaluationDataset:
    """Convert benchmark raw results into a RAGAS EvaluationDataset.

    Args:
        raw_results: List of dicts from run_benchmark, each with:
            - question, expected_answer, query_type, source_section
            - results.{retriever_name}.answer
            - results.{retriever_name}.retrieved_chunks[].text
        retriever_name: "lightrag" or "pageindex"

    Returns:
        RAGAS EvaluationDataset ready for evaluation.
    """
    samples = []
    for item in raw_results:
        r = item["results"][retriever_name]
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=r["answer"],
                retrieved_contexts=[c["text"] for c in r["retrieved_chunks"]],
                reference=item["expected_answer"],
            )
        )
    return EvaluationDataset(samples=samples)


def run_ragas_evaluation(
    raw_results: list[dict],
    retriever_name: str,
) -> dict:
    """Run RAGAS evaluation for a single retriever's results.

    Args:
        raw_results: Benchmark raw results.
        retriever_name: "lightrag" or "pageindex".

    Returns:
        Dict with metric names as keys and float scores as values.
    """
    dataset = build_ragas_dataset(raw_results, retriever_name)

    # Create Claude judge
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Required for RAGAS Claude-as-judge evaluation."
        )

    judge_llm = llm_factory(
        "claude-sonnet-4-20250514",
        provider="anthropic",
        client=Anthropic(api_key=anthropic_key),
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        AnswerCorrectness(),
    ]

    print(f"[RAGAS] Evaluating {retriever_name} ({len(dataset)} samples)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
    )

    scores = {k: round(v, 4) for k, v in result.items() if isinstance(v, (int, float))}
    print(f"[RAGAS] {retriever_name} scores: {scores}")

    return scores


def run_ragas_per_type(
    raw_results: list[dict],
    retriever_name: str,
) -> dict[str, dict]:
    """Run RAGAS evaluation broken down by query type.

    Returns:
        Dict mapping query_type -> {metric: score}.
    """
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in raw_results:
        by_type[item["query_type"]].append(item)

    results_by_type = {}
    for query_type, items in by_type.items():
        if len(items) < 2:
            print(f"[RAGAS] Skipping {query_type} for {retriever_name} (only {len(items)} samples)")
            continue
        try:
            scores = run_ragas_evaluation(items, retriever_name)
            results_by_type[query_type] = scores
        except Exception as e:
            print(f"[RAGAS] Error evaluating {query_type} for {retriever_name}: {e}")
            results_by_type[query_type] = {"error": str(e)}

    return results_by_type
```

- [ ] **Step 2: Verify imports**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import sys; sys.path.insert(0, '.')
from benchmark.eval.judge import build_ragas_dataset, run_ragas_evaluation
print('RAGAS judge module imported OK')
"
```

Expected: `RAGAS judge module imported OK`. If RAGAS import errors occur, fix the metric class names based on what's available in the installed version.

- [ ] **Step 3: Commit**

```bash
git add benchmark/eval/judge.py
git commit -m "Add RAGAS evaluation module with Claude-as-judge"
```

---

### Task 7: Benchmark Harness

**Files:**
- Create: `benchmark/run_benchmark.py`

- [ ] **Step 1: Write the benchmark harness**

`benchmark/run_benchmark.py`:
```python
"""Main benchmark harness: runs both retrievers on the test set and evaluates.

Usage:
    python benchmark/run_benchmark.py                  # full benchmark
    python benchmark/run_benchmark.py --skip-eval      # skip RAGAS eval (just collect raw results)
    python benchmark/run_benchmark.py --retriever lightrag  # run only one retriever
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from benchmark.retrievers.base import AnswerResult
from benchmark.retrievers.lightrag_retriever import LightRAGRetriever
from benchmark.retrievers.pageindex_retriever import PageIndexRetriever
from src.config import ENGLISH_DOC, HINDI_DOC, RAG_STORAGE_DIR


RESULTS_DIR = Path(__file__).parent / "results"
TESTSET_PATH = Path(__file__).parent / "testset.json"


def load_testset() -> list[dict]:
    """Load the benchmark test set."""
    data = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))
    return data["questions"]


def init_retrievers(only: str | None = None) -> dict:
    """Initialize and index both retrievers."""
    documents = [
        {"path": str(ENGLISH_DOC), "name": "Factories Act, 1948"},
        {"path": str(HINDI_DOC), "name": "Capital Subsidy Rules"},
    ]

    retrievers = {}

    if only in (None, "lightrag"):
        print("\n=== Initializing LightRAG ===")
        lr = LightRAGRetriever(RAG_STORAGE_DIR)
        lr.index(documents)
        retrievers["lightrag"] = lr

    if only in (None, "pageindex"):
        print("\n=== Initializing PageIndex ===")
        pi = PageIndexRetriever()
        pi.index(documents)
        retrievers["pageindex"] = pi

    return retrievers


def run_single_query(retriever, question: dict) -> dict:
    """Run a single query through a retriever and return structured results."""
    query = question["question"]
    print(f"  [{retriever.name}] {query[:80]}...")

    try:
        result = retriever.query_end_to_end(query)
        return {
            "answer": result.answer,
            "retrieved_chunks": [
                {
                    "text": c.text[:500],  # truncate for storage
                    "source_document": c.source_document,
                    "source_section": c.source_section,
                    "page_range": c.page_range,
                }
                for c in result.retrieved_chunks
            ],
            "retrieval_latency_ms": round(result.retrieval_latency_ms, 1),
            "e2e_latency_ms": round(result.e2e_latency_ms, 1),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
    except Exception as e:
        print(f"  [{retriever.name}] ERROR: {e}")
        return {
            "answer": f"ERROR: {e}",
            "retrieved_chunks": [],
            "retrieval_latency_ms": 0,
            "e2e_latency_ms": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


def compute_custom_metrics(raw_results: list[dict], retriever_name: str) -> dict:
    """Compute aggregated custom metrics (latency, tokens) for a retriever."""
    items = [r["results"][retriever_name] for r in raw_results if retriever_name in r["results"]]
    if not items:
        return {}

    n = len(items)
    return {
        "avg_retrieval_latency_ms": round(sum(i["retrieval_latency_ms"] for i in items) / n, 1),
        "avg_e2e_latency_ms": round(sum(i["e2e_latency_ms"] for i in items) / n, 1),
        "avg_prompt_tokens": round(sum(i["prompt_tokens"] for i in items) / n),
        "avg_completion_tokens": round(sum(i["completion_tokens"] for i in items) / n),
        "avg_total_tokens": round(sum(i["total_tokens"] for i in items) / n),
        "total_prompt_tokens": sum(i["prompt_tokens"] for i in items),
        "total_completion_tokens": sum(i["completion_tokens"] for i in items),
        "total_tokens": sum(i["total_tokens"] for i in items),
    }


def compute_metrics_by_type(raw_results: list[dict], retriever_name: str) -> dict:
    """Compute custom metrics grouped by query_type."""
    by_type: dict[str, list] = defaultdict(list)
    for r in raw_results:
        if retriever_name in r["results"]:
            by_type[r["query_type"]].append(r["results"][retriever_name])

    result = {}
    for qtype, items in by_type.items():
        n = len(items)
        result[qtype] = {
            "count": n,
            "avg_e2e_latency_ms": round(sum(i["e2e_latency_ms"] for i in items) / n, 1),
            "avg_total_tokens": round(sum(i["total_tokens"] for i in items) / n),
        }
    return result


def compute_metrics_by_document(raw_results: list[dict], retriever_name: str) -> dict:
    """Compute custom metrics grouped by source_document."""
    by_doc: dict[str, list] = defaultdict(list)
    for r in raw_results:
        if retriever_name in r["results"]:
            by_doc[r["source_document"]].append(r["results"][retriever_name])

    result = {}
    for doc_name, items in by_doc.items():
        n = len(items)
        result[doc_name] = {
            "count": n,
            "avg_e2e_latency_ms": round(sum(i["e2e_latency_ms"] for i in items) / n, 1),
            "avg_total_tokens": round(sum(i["total_tokens"] for i in items) / n),
        }
    return result


def generate_report(summary: dict, raw_results: list[dict]) -> str:
    """Generate a human-readable comparison report."""
    lines = [
        "# Vectorless RAG Benchmark Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Questions:** {len(raw_results)}",
        "",
        "---",
        "",
        "## Overall Metrics",
        "",
        "| Metric | LightRAG | PageIndex |",
        "|--------|----------|-----------|",
    ]

    # Merge RAGAS + custom metrics
    lr = summary.get("lightrag", {}).get("overall", {})
    pi = summary.get("pageindex", {}).get("overall", {})

    all_keys = sorted(set(list(lr.keys()) + list(pi.keys())))
    for key in all_keys:
        lr_val = lr.get(key, "N/A")
        pi_val = pi.get(key, "N/A")
        if isinstance(lr_val, float):
            lr_val = f"{lr_val:.4f}"
        if isinstance(pi_val, float):
            pi_val = f"{pi_val:.4f}"
        lines.append(f"| {key} | {lr_val} | {pi_val} |")

    lines.extend(["", "---", "", "## By Query Type", ""])

    for retriever_name in ["lightrag", "pageindex"]:
        by_type = summary.get(retriever_name, {}).get("by_query_type", {})
        if not by_type:
            continue
        lines.append(f"### {retriever_name}")
        lines.append("")
        lines.append("| Query Type | Count | Avg E2E Latency (ms) | Avg Tokens |")
        lines.append("|------------|-------|----------------------|------------|")
        for qtype, metrics in sorted(by_type.items()):
            if isinstance(metrics, dict) and "count" in metrics:
                lines.append(
                    f"| {qtype} | {metrics['count']} | "
                    f"{metrics.get('avg_e2e_latency_ms', 'N/A')} | "
                    f"{metrics.get('avg_total_tokens', 'N/A')} |"
                )
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run vectorless RAG benchmark")
    parser.add_argument("--skip-eval", action="store_true", help="Skip RAGAS evaluation")
    parser.add_argument("--retriever", choices=["lightrag", "pageindex"], help="Run only one retriever")
    args = parser.parse_args()

    # Load test set
    questions = load_testset()
    print(f"Loaded {len(questions)} questions from {TESTSET_PATH}")

    # Initialize retrievers
    retrievers = init_retrievers(only=args.retriever)

    # Run benchmark
    print(f"\n{'='*60}")
    print("RUNNING BENCHMARK")
    print(f"{'='*60}\n")

    raw_results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question'][:80]}...")
        entry = {
            "id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "source_section": q["source_section"],
            "source_document": q["source_document"],
            "source_pages": q.get("source_pages", ""),
            "query_type": q["query_type"],
            "results": {},
        }

        for name, retriever in retrievers.items():
            entry["results"][name] = run_single_query(retriever, q)

        raw_results.append(entry)

    # Save raw results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "raw_results.json"
    raw_path.write_text(json.dumps(raw_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRaw results saved to {raw_path}")

    # Build summary
    summary = {}
    for name in retrievers:
        custom = compute_custom_metrics(raw_results, name)
        by_type = compute_metrics_by_type(raw_results, name)
        by_doc = compute_metrics_by_document(raw_results, name)
        summary[name] = {
            "overall": {**custom},
            "by_query_type": by_type,
            "by_document": by_doc,
        }

    # Run RAGAS evaluation
    if not args.skip_eval:
        from benchmark.eval.judge import run_ragas_evaluation, run_ragas_per_type

        for name in retrievers:
            print(f"\n{'='*60}")
            print(f"RAGAS EVALUATION: {name}")
            print(f"{'='*60}\n")

            ragas_scores = run_ragas_evaluation(raw_results, name)
            summary[name]["overall"].update(ragas_scores)

            ragas_by_type = run_ragas_per_type(raw_results, name)
            for qtype, scores in ragas_by_type.items():
                if qtype in summary[name]["by_query_type"]:
                    summary[name]["by_query_type"][qtype].update(scores)
                else:
                    summary[name]["by_query_type"][qtype] = scores
    else:
        print("\nSkipping RAGAS evaluation (--skip-eval)")

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved to {summary_path}")

    # Generate report
    report = generate_report(summary, raw_results)
    report_path = RESULTS_DIR / "comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to {report_path}")

    # Print summary to console
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, data in summary.items():
        print(f"\n{name}:")
        for k, v in data["overall"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run with --skip-eval to verify the harness works**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python benchmark/run_benchmark.py --skip-eval
```

Expected: runs all questions through both retrievers, saves `results/raw_results.json` and `results/summary.json` with latency/token metrics. No RAGAS scores yet.

- [ ] **Step 3: Fix any issues from the dry run**

Inspect errors in the output. Common issues:
- Missing API keys: check `.env` for `GROQ_API_KEY`
- PageIndex indexing failures: may need to adjust PDF paths
- LightRAG query errors: verify `rag_storage/` is intact

- [ ] **Step 4: Run full benchmark with RAGAS evaluation**

Requires `ANTHROPIC_API_KEY` in `.env`:

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python benchmark/run_benchmark.py
```

Expected: full benchmark with RAGAS scores in `results/summary.json` and a comparison report in `results/comparison_report.md`.

- [ ] **Step 5: Commit**

```bash
git add benchmark/run_benchmark.py
git commit -m "Add benchmark harness with RAGAS evaluation and report generation"
```

---

### Task 8: Run Benchmark and Review Results

- [ ] **Step 1: Ensure all API keys are set in .env**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import os
for key in ['GROQ_API_KEY', 'GEMINI_API_KEY', 'ANTHROPIC_API_KEY']:
    val = os.getenv(key)
    status = 'SET' if val else 'MISSING'
    print(f'{key}: {status}')
"
```

Expected: all three keys show `SET`.

- [ ] **Step 2: Run the full benchmark**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python benchmark/run_benchmark.py 2>&1 | tee benchmark/results/benchmark_log.txt
```

This will take several minutes. Expected output flow:
1. LightRAG loads existing index
2. PageIndex builds tree indices for both PDFs
3. 28 questions run through both retrievers
4. RAGAS evaluation runs on all results
5. Summary and report saved

- [ ] **Step 3: Review the comparison report**

```bash
cat /home/pc/Downloads/LawMaster/benchmark/results/comparison_report.md
```

Check that:
- Both retrievers have non-zero scores
- RAGAS metrics are in 0.0-1.0 range
- Latency numbers are reasonable (LightRAG: ~100-1000ms, PageIndex: ~1000-5000ms)
- Token counts are non-zero

- [ ] **Step 4: Review raw results for quality**

```bash
cd /home/pc/Downloads/LawMaster
/home/pc/anaconda3/envs/ml-env/bin/python -c "
import json
raw = json.loads(open('benchmark/results/raw_results.json').read())
# Show first result for each retriever
q = raw[0]
print(f'Question: {q[\"question\"]}')
print(f'Expected: {q[\"expected_answer\"][:200]}...')
for name in ['lightrag', 'pageindex']:
    if name in q['results']:
        r = q['results'][name]
        print(f'\n--- {name} ---')
        print(f'Answer: {r[\"answer\"][:200]}...')
        print(f'Latency: {r[\"e2e_latency_ms\"]}ms, Tokens: {r[\"total_tokens\"]}')
"
```

- [ ] **Step 5: Commit results and report**

```bash
git add benchmark/results/summary.json benchmark/results/comparison_report.md
git commit -m "Add benchmark results: LightRAG vs PageIndex comparison"
```

- [ ] **Step 6: Final commit with all benchmark code**

```bash
git add -A benchmark/
git commit -m "Complete vectorless RAG benchmark: PageIndex vs LightRAG"
```
