#!/usr/bin/env python3
"""Validate the full indexed corpus by running test queries across all domains.

Runs 12 queries spanning Factories Act, Pollution Control, and Chhattisgarh
Industrial Policy, covering definitions, sections, tables, and amendments.
Reports PASS/FAIL for each and prints a summary.

Usage:
    python scripts/12_validate_index.py
"""

import sys
import os
import asyncio
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from lightrag import QueryParam
from src.agent.rag import get_rag_instance, groq_query_func

QUESTIONS = [
    # Factories Act — definitions
    ("What is the definition of 'factory'?", "Definition"),
    ("What does 'hazardous process' mean under the Factories Act?", "Definition"),
    # Factories Act — sections
    ("What safety provisions apply to hazardous processes?", "Section"),
    ("What are the working hour restrictions for adult workers?", "Section"),
    ("What penalties exist for employing children?", "Section"),
    # Factories Act — tables (via RAG stub)
    ("What is the permissible exposure limit for Benzene?", "Table/RAG"),
    # Pollution Control
    ("What industries fall under the Red category in CPCB classification?", "Section/Table"),
    ("What does the EIA Notification 2006 require for environmental clearance?", "Section"),
    ("What are the rules for hazardous waste management?", "Section"),
    # Chhattisgarh Industrial Policy (Hindi docs)
    ("What subsidies are available under the Chhattisgarh Industrial Policy?", "Section"),
    ("What is the process for capital subsidy application?", "Section"),
    # Amendments
    ("Has Section 2 of the Factories Act been amended?", "Amendment"),
]

FAIL_MARKERS = [
    "sorry", "i don't have", "no information", "not found", "no relevant",
    "i cannot", "i could not", "no data", "unable to find", "not available",
    "no context", "[no-context]",
]


def judge_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return False
    text = answer.strip().lower()
    if len(text) < 30:
        return False
    for marker in FAIL_MARKERS:
        if marker in text:
            return False
    return True


async def main():
    print("=" * 70)
    print("  LawMaster -- Full Index Validation")
    print("=" * 70)
    print()

    print("[1/2] Loading RAG instance...")
    t0 = time.time()
    rag = get_rag_instance()
    print(f"      Loaded in {time.time() - t0:.1f}s\n")

    print("[2/2] Running validation queries...\n")
    results = []

    for i, (question, content_type) in enumerate(QUESTIONS, 1):
        print(f"  [{i:2d}/{len(QUESTIONS)}] {content_type:<14s} | {question}")
        try:
            answer = await rag.aquery(
                question,
                param=QueryParam(mode="mix", top_k=5, model_func=groq_query_func),
            )
            passed = judge_answer(answer)
            snippet = (answer or "").strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
        except Exception as e:
            answer = ""
            passed = False
            snippet = f"ERROR: {e}"

        status = "PASS" if passed else "FAIL"
        print(f"         -> {status}: {snippet}")
        print()
        results.append((question, content_type, passed, snippet))

    # Summary
    total = len(results)
    passed = sum(1 for _, _, p, _ in results if p)
    failed = total - passed

    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'#':<4s} {'Type':<14s} {'Result':<6s} Question")
    print("  " + "-" * 66)
    for i, (question, content_type, p, _) in enumerate(results, 1):
        status = "PASS" if p else "FAIL"
        print(f"  {i:<4d} {content_type:<14s} {status:<6s} {question}")
    print("  " + "-" * 66)
    print(f"  TOTAL: {passed}/{total} passed, {failed} failed")
    print()


if __name__ == "__main__":
    asyncio.run(main())
