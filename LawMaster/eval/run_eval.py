#!/usr/bin/env python3
"""Run eval benchmark against the full LawMaster pipeline via /chat/stream SSE.

Usage:
    python eval/run_eval.py --limit 3          # test first 3 untested questions
    python eval/run_eval.py                    # run all untested questions
    python eval/run_eval.py --delay 5          # 5s between queries
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError


SOURCE_FILE = Path(__file__).parent / "legal_qa_benchmark.json"
OUTPUT_FILE = Path(__file__).parent / "legal_qa_benchmark_tested.json"
BASE_URL = "http://localhost:8000"


def query_pipeline(question: str, base_url: str, timeout: int = 300) -> dict:
    """Send a question to /chat/stream and parse all SSE events.

    Returns dict with: generated_answer, num_chunks, num_entities,
    num_relations, num_citations, source_documents_retrieved, contexts.
    """
    payload = json.dumps({"query": question}).encode()
    req = Request(
        f"{base_url}/chat/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    resp = urlopen(req, timeout=timeout)

    generated_answer = ""
    citations = []
    retrieval_stats_list = []
    tool_calls = 0
    event_type = None

    for raw_line in resp:
        line = raw_line.decode("utf-8").rstrip("\n\r")
        if line.startswith("event: "):
            event_type = line[7:]
        elif line.startswith("data: ") and event_type:
            data = json.loads(line[6:])
            if event_type == "tool_start" and data.get("name") == "search_legal_text":
                tool_calls += 1
            elif event_type == "content":
                generated_answer += data.get("token", "")
            elif event_type == "citation":
                citations.append(data)
            elif event_type == "retrieval_stats":
                retrieval_stats_list.append(data)
            elif event_type == "error":
                return {"error": data.get("message", "Unknown error")}
            elif event_type == "done":
                break

    # Aggregate retrieval stats across possible multiple tool calls
    total_chunks = sum(s.get("num_chunks", 0) for s in retrieval_stats_list)
    total_entities = sum(s.get("num_entities", 0) for s in retrieval_stats_list)
    total_relations = sum(s.get("num_relations", 0) for s in retrieval_stats_list)
    all_source_docs = list(set(
        doc for s in retrieval_stats_list for doc in s.get("source_documents", [])
    ))
    all_contexts = [
        ctx for s in retrieval_stats_list for ctx in s.get("contexts", [])
    ]

    return {
        "generated_answer": generated_answer.strip(),
        "num_tool_calls": tool_calls,
        "num_chunks": total_chunks,
        "num_entities": total_entities,
        "num_relations": total_relations,
        "num_citations": len(citations),
        "source_documents_retrieved": all_source_docs,
        "contexts": all_contexts,
    }


def save(benchmark: list[dict]):
    """Write benchmark back to disk."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Run LawMaster eval benchmark")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max questions to process (0 = all)")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between queries (rate limiting)")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Pause after every N questions for review (0 = no pause)")
    parser.add_argument("--base-url", default=BASE_URL,
                        help="Server base URL")
    args = parser.parse_args()

    # Load source benchmark (has source_chunk ground truth)
    with open(SOURCE_FILE, encoding="utf-8") as f:
        source = json.load(f)

    # Load or initialize output (preserves prior results)
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            existing = {e["id"]: e for e in json.load(f)}
    else:
        existing = {}

    # Merge: start from source, overlay any existing results
    benchmark = []
    for entry in source:
        merged = dict(entry)  # source_chunk + ground truth
        if entry["id"] in existing:
            merged.update(existing[entry["id"]])
        benchmark.append(merged)

    # Health check
    try:
        resp = urlopen(f"{args.base_url}/health", timeout=5)
        health = json.loads(resp.read())
        assert health.get("status") == "ok"
    except Exception as e:
        print(f"[ERROR] Server not reachable at {args.base_url}: {e}")
        sys.exit(1)

    print(f"[INFO] Server OK. {len(benchmark)} total questions.")

    # Filter to untested entries
    pending = [q for q in benchmark if "generated_answer" not in q]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"[INFO] {len(pending)} questions to process "
          f"(limit={'all' if args.limit == 0 else args.limit}).\n")

    for i, entry in enumerate(pending):
        qid = entry["id"]
        question = entry["question"]
        print(f"[{i + 1}/{len(pending)}] Q{qid} ({entry['difficulty']}): "
              f"{question[:80]}...")

        try:
            result = query_pipeline(question, args.base_url)
        except Exception as e:
            print(f"  ERROR: {e}")
            entry["generated_answer"] = f"ERROR: {e}"
            entry["eval_timestamp"] = datetime.now(timezone.utc).isoformat()
            save(benchmark)
            time.sleep(args.delay)
            continue

        if "error" in result:
            print(f"  PIPELINE ERROR: {result['error']}")
            entry["generated_answer"] = f"ERROR: {result['error']}"
        else:
            ans_preview = result["generated_answer"][:120].replace("\n", " ")
            print(f"  Answer: {ans_preview}...")
            print(f"  ToolCalls={result['num_tool_calls']}  "
                  f"Chunks={result['num_chunks']}  "
                  f"Entities={result['num_entities']}  "
                  f"Relations={result['num_relations']}  "
                  f"Citations={result['num_citations']}  "
                  f"Sources={len(result['source_documents_retrieved'])}")
            entry.update({
                "generated_answer": result["generated_answer"],
                "num_tool_calls": result["num_tool_calls"],
                "num_chunks": result["num_chunks"],
                "num_entities": result["num_entities"],
                "num_relations": result["num_relations"],
                "num_citations": result["num_citations"],
                "source_documents_retrieved": result["source_documents_retrieved"],
                "contexts": result["contexts"],
            })

        entry["eval_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Save after every question so progress isn't lost
        save(benchmark)

        # Batch pause: print stats every batch_size questions
        completed = i + 1
        if args.batch_size > 0 and completed % args.batch_size == 0 and completed < len(pending):
            tested_so_far = sum(1 for q in benchmark if "generated_answer" in q
                                and not q["generated_answer"].startswith("ERROR"))
            errors_so_far = sum(1 for q in benchmark if "generated_answer" in q
                                and q["generated_answer"].startswith("ERROR"))
            avg_tools = sum(q.get("num_tool_calls", 0) for q in benchmark
                           if "num_tool_calls" in q) / max(tested_so_far, 1)
            avg_chunks = sum(q.get("num_chunks", 0) for q in benchmark
                            if "num_chunks" in q) / max(tested_so_far, 1)
            print(f"\n--- BATCH CHECKPOINT ({completed}/{len(pending)}) ---")
            print(f"  Tested: {tested_so_far}  Errors: {errors_so_far}")
            print(f"  Avg ToolCalls: {avg_tools:.1f}  Avg Chunks: {avg_chunks:.0f}")
            print(f"  Pausing 10s before next batch...\n")
            time.sleep(10)

        if i < len(pending) - 1:
            time.sleep(args.delay)

    tested = sum(1 for q in benchmark if "generated_answer" in q
                 and not q["generated_answer"].startswith("ERROR"))
    print(f"\n[DONE] {tested}/{len(benchmark)} successfully tested. "
          f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
