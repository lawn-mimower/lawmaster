#!/usr/bin/env python3
"""Proper ER extraction benchmark: temperature=0, 20 chunks, 3 runs, averaged.

Usage:
    python scripts/15_er_benchmark.py
"""

import sys
import os
import asyncio
import time
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from src.config import GROQ_API_KEY, GEMINI_API_KEY, EXTRACTION_OUTPUT_DIR
from lightrag.prompt import PROMPTS

ENTITY_TYPES = [
    "Definition", "Section", "Amendment", "Schedule",
    "Act", "Rule", "Authority", "Penalty", "Provision",
    "Industry", "Chemical", "Regulation", "Notification",
]

examples = "\n".join(PROMPTS["entity_extraction_examples"])
ctx = dict(
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    entity_types=", ".join(ENTITY_TYPES),
    language="English",
)
examples = examples.format(**ctx)

SYSTEM_PROMPT = PROMPTS["entity_extraction_system_prompt"].format(
    **{**ctx, "examples": examples}
)

NUM_RUNS = 3
NUM_CHUNKS = 20


def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def parse_output(text):
    td = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    entities, relations = 0, 0
    entity_names = set()
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(f"entity{td}"):
            parts = line.split(td)
            if len(parts) >= 4:
                entities += 1
                entity_names.add(parts[1].strip().lower())
        elif line.startswith(f"relation{td}"):
            parts = line.split(td)
            if len(parts) >= 5:
                relations += 1
    return entities, relations, entity_names


async def call_model(model_id, chunk_text, api_key, base_url):
    """Call with temperature=0 for determinism."""
    import openai

    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
        **{**ctx, "input_text": chunk_text}
    )

    start = time.time()
    response = await client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=4000,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content or ""
    content = strip_think(content)
    return content, elapsed


async def call_gemini(chunk_text):
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
        **{**ctx, "input_text": chunk_text}
    )

    model = genai.GenerativeModel(
        "gemini-3-flash-preview",
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(temperature=0),
    )

    start = time.time()
    response = model.generate_content(user_prompt)
    elapsed = time.time() - start
    return response.text, elapsed


def load_chunks():
    """Load real chunks from the extracted Factory Act markdown."""
    from src.chunk.router import route_content

    md_path = EXTRACTION_OUTPUT_DIR / "mistral" / "factories act" / "FactoryAct1948.md"
    if not md_path.exists():
        # Fallback to test dir
        md_path = EXTRACTION_OUTPUT_DIR / "test_mistral" / "FactoryAct1948.md"

    markdown = md_path.read_text(encoding="utf-8")
    routed = route_content(
        markdown=markdown,
        annotations=[],
        html_tables=[],
        source_name="FactoryAct1948.pdf",
        category="Factories Act & Rules",
    )

    # Mix of definitions and sections
    chunks = []
    for d in routed["definitions"][:5]:
        chunks.append(f"{d['meta']}\n\n{d['text']}")
    for s in routed["sections"][:15]:
        chunks.append(f"{s['meta']}\n\n{s['text']}")

    return chunks[:NUM_CHUNKS]


async def main():
    GROQ_KEY = os.environ.get("GROQ_API_KEY")

    models = [
        ("Llama 8B (Groq)", "llama-3.1-8b-instant", GROQ_KEY, "https://api.groq.com/openai/v1"),
        ("Gemini 3 Flash", None, None, None),  # special handling
    ]

    print("=" * 80)
    print("ER EXTRACTION BENCHMARK (temperature=0)")
    print("=" * 80)

    chunks = load_chunks()
    print(f"\nChunks: {len(chunks)}")
    print(f"Runs per model: {NUM_RUNS}")
    print(f"Total calls: {len(chunks)} × {NUM_RUNS} × {len(models)} = {len(chunks) * NUM_RUNS * len(models)}")
    avg_chars = sum(len(c) for c in chunks) // len(chunks)
    print(f"Avg chunk size: {avg_chars} chars (~{avg_chars//4} tokens)")

    all_results = {}

    for model_name, model_id, api_key, base_url in models:
        print(f"\n{'─'*80}")
        print(f"MODEL: {model_name}")
        print(f"{'─'*80}")

        run_data = []

        for run in range(NUM_RUNS):
            run_entities = 0
            run_relations = 0
            run_time = 0
            all_entity_names = set()

            for i, chunk in enumerate(chunks):
                try:
                    if model_id is None:  # Gemini
                        result, elapsed = await call_gemini(chunk)
                    else:
                        result, elapsed = await call_model(model_id, chunk, api_key, base_url)

                    ent, rel, names = parse_output(result)
                    run_entities += ent
                    run_relations += rel
                    run_time += elapsed
                    all_entity_names.update(names)

                except Exception as e:
                    print(f"  Run {run+1}, Chunk {i+1}: ERROR — {type(e).__name__}: {str(e)[:60]}")

            run_data.append({
                "entities": run_entities,
                "relations": run_relations,
                "time": round(run_time, 1),
                "unique_entities": len(all_entity_names),
                "avg_time_per_chunk": round(run_time / len(chunks), 2),
            })

            print(f"  Run {run+1}: {run_entities} ent, {run_relations} rel, "
                  f"{run_time:.1f}s total ({run_time/len(chunks):.2f}s/chunk), "
                  f"{len(all_entity_names)} unique entities")

        # Average across runs
        avg_ent = sum(r["entities"] for r in run_data) / NUM_RUNS
        avg_rel = sum(r["relations"] for r in run_data) / NUM_RUNS
        avg_time = sum(r["time"] for r in run_data) / NUM_RUNS
        avg_per_chunk = sum(r["avg_time_per_chunk"] for r in run_data) / NUM_RUNS
        avg_unique = sum(r["unique_entities"] for r in run_data) / NUM_RUNS

        # Variance
        ent_var = max(r["entities"] for r in run_data) - min(r["entities"] for r in run_data)
        rel_var = max(r["relations"] for r in run_data) - min(r["relations"] for r in run_data)

        all_results[model_name] = {
            "avg_entities": round(avg_ent, 1),
            "avg_relations": round(avg_rel, 1),
            "avg_time": round(avg_time, 1),
            "avg_per_chunk": round(avg_per_chunk, 2),
            "avg_unique": round(avg_unique, 1),
            "entity_variance": ent_var,
            "relation_variance": rel_var,
            "runs": run_data,
        }

    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL RESULTS (averaged over 3 runs, temperature=0)")
    print(f"{'='*80}")
    print(f"\n{'Model':<25s} {'Avg Ent':>8s} {'Avg Rel':>8s} {'Ent Var':>8s} {'Rel Var':>8s} {'Avg/chunk':>10s} {'Total':>8s}")
    print("─" * 80)
    for model_name, data in all_results.items():
        print(f"{model_name:<25s} {data['avg_entities']:>8.1f} {data['avg_relations']:>8.1f} "
              f"{'±'+str(data['entity_variance']):>8s} {'±'+str(data['relation_variance']):>8s} "
              f"{str(data['avg_per_chunk'])+'s':>10s} {str(data['avg_time'])+'s':>8s}")

    print(f"\n  Chunks: {len(chunks)}, Runs: {NUM_RUNS}, Temperature: 0")
    print(f"  Lower variance = more deterministic")


if __name__ == "__main__":
    asyncio.run(main())
