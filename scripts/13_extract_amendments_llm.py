#!/usr/bin/env python3
"""Extract structured amendments from all extracted document markdowns via LLM.

Reads markdowns directly from extraction_output/mistral/, sends each to
Groq LLM for one-shot amendment extraction, outputs data/amendments.json.

Usage:
    python scripts/13_extract_amendments_llm.py
    python scripts/13_extract_amendments_llm.py --limit 3
    python scripts/13_extract_amendments_llm.py --max-concurrent 4
"""

import sys
import os
import json
import asyncio
import argparse
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import GROQ_API_KEY, EXTRACTION_MODEL, DATA_DIR, EXTRACTION_OUTPUT_DIR
from lightrag.llm.openai import openai_complete_if_cache

MISTRAL_OUT = EXTRACTION_OUTPUT_DIR / "mistral"
VALID_TYPES = {"substituted", "inserted", "omitted", "added", "renumbered"}

PROMPT = """You are analyzing a legal document for amendments — any modification, substitution, insertion, omission, or addition to existing law.

Look for patterns like:
- "Subs. by Act X of YYYY, s. N"
- "Ins. by Act X of YYYY"
- "Omitted by Act X of YYYY"
- "[Substituted by Act X of YYYY]"
- "w.e.f. DD-MM-YYYY" (with effect from)
- Amendment notifications that modify earlier Acts or Rules

For each amendment found, return:
- target_act: The Act being amended (e.g., "Factories Act, 1948")
- target_section: The section number being amended (e.g., "2", "41B", "87")
- amendment_type: one of [substituted, inserted, omitted, added, renumbered]
- amending_act: The Act/notification that made the amendment (e.g., "Act 94 of 1976")
- effective_date: When the amendment took effect (DD-MM-YYYY if stated), or null
- description: Brief one-sentence description of what changed

Return a JSON object: {"amendments": [...]}
If no amendments are found, return: {"amendments": []}

DOCUMENT TEXT:
"""

MAX_CHARS_PER_CALL = 12000  # ~3000 tokens, fits in context with prompt


def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def discover_docs() -> list[dict]:
    """Find all extracted markdown files."""
    docs = []
    for md_path in sorted(MISTRAL_OUT.rglob("*.md")):
        # Skip files with suffixes like _tables.json, _meta.json
        if "_" in md_path.stem and md_path.stem.split("_")[-1] in ("meta", "tables", "ocr"):
            continue
        docs.append({
            "md_path": md_path,
            "name": md_path.stem,
            "source": md_path.stem + ".pdf",
            "category_dir": md_path.parent.name,
        })
    return docs


async def extract_doc(doc: dict, semaphore: asyncio.Semaphore) -> list[dict]:
    """Extract amendments from one document, chunking if needed."""
    text = doc["md_path"].read_text(encoding="utf-8")
    source = doc["source"]
    all_amendments = []

    # Split into chunks if document is too large
    if len(text) <= MAX_CHARS_PER_CALL:
        chunks = [text]
    else:
        # Split by page breaks
        pages = text.split("\n\n---\n\n")
        chunks = []
        current = ""
        for page in pages:
            if len(current) + len(page) > MAX_CHARS_PER_CALL and current:
                chunks.append(current)
                current = page
            else:
                current += "\n\n---\n\n" + page if current else page
        if current:
            chunks.append(current)

    async with semaphore:
        for i, chunk in enumerate(chunks):
            try:
                result = await openai_complete_if_cache(
                    model=EXTRACTION_MODEL,
                    prompt=PROMPT + chunk,
                    system_prompt="You extract structured amendment data from legal text. Return valid JSON only.",
                    history_messages=[],
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                    response_format={"type": "json_object"},
                )
                result = strip_think(result)
                parsed = json.loads(result)
                amendments = parsed.get("amendments", [])

                for a in amendments:
                    atype = a.get("amendment_type", "").lower().strip()
                    if atype not in VALID_TYPES:
                        atype = "substituted"
                    a["amendment_type"] = atype
                    a["source_document"] = source
                all_amendments.extend(amendments)

            except json.JSONDecodeError as e:
                print(f"    JSON parse error for {source} chunk {i+1}: {e}")
            except Exception as e:
                print(f"    Error for {source} chunk {i+1}: {e}")

    return all_amendments


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process only N docs (0=all)")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent LLM calls")
    args = parser.parse_args()

    docs = discover_docs()
    if not docs:
        print(f"No markdowns found in {MISTRAL_OUT}. Run extraction first.")
        return

    if args.limit > 0:
        docs = docs[:args.limit]

    total_chars = sum(d["md_path"].stat().st_size for d in docs)
    total_chunks = sum(1 + d["md_path"].stat().st_size // MAX_CHARS_PER_CALL for d in docs)

    print(f"=== Amendment Extraction ===")
    print(f"  Documents: {len(docs)}")
    print(f"  Total text: {total_chars:,} chars")
    print(f"  Estimated LLM calls: ~{total_chunks}")
    print(f"  Max concurrent: {args.max_concurrent}")
    print()

    start = time.time()
    semaphore = asyncio.Semaphore(args.max_concurrent)
    tasks = [extract_doc(doc, semaphore) for doc in docs]
    results = await asyncio.gather(*tasks)

    all_amendments = []
    for i, amendments in enumerate(results):
        if amendments:
            print(f"  {docs[i]['source']}: {len(amendments)} amendments")
        all_amendments.extend(amendments)

    elapsed = time.time() - start

    # Deduplicate (same section + same amending act = duplicate)
    seen = set()
    unique = []
    for a in all_amendments:
        key = (a.get("target_section", ""), a.get("amending_act", ""), a.get("amendment_type", ""))
        if key not in seen:
            seen.add(key)
            unique.append(a)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "amendments.json"
    with open(out_path, "w") as f:
        json.dump(unique, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"AMENDMENT EXTRACTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Documents processed: {len(docs)}")
    print(f"  Raw amendments found: {len(all_amendments)}")
    print(f"  After dedup: {len(unique)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output: {out_path}")

    types = {}
    for a in unique:
        t = a.get("amendment_type", "?")
        types[t] = types.get(t, 0) + 1
    print(f"\n  By type:")
    for t, c in sorted(types.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")

    sources = {}
    for a in unique:
        s = a.get("source_document", "?")
        sources[s] = sources.get(s, 0) + 1
    print(f"\n  By source (top 10):")
    for s, c in sorted(sources.items(), key=lambda x: -x[1])[:10]:
        print(f"    {s}: {c}")

    with_date = sum(1 for a in unique if a.get("effective_date"))
    with_section = sum(1 for a in unique if a.get("target_section"))
    print(f"\n  Coverage:")
    print(f"    With effective_date: {with_date}/{len(unique)}")
    print(f"    With target_section: {with_section}/{len(unique)}")


if __name__ == "__main__":
    asyncio.run(main())
