#!/usr/bin/env python3
"""Compare ER extraction models: Gemini 3 Flash vs Llama 3.1 8B vs Qwen3 32B (baseline).

Same chunk, same prompt, side-by-side output comparison.

Usage:
    python scripts/14_compare_er_models.py
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

from src.config import GROQ_API_KEY, GEMINI_API_KEY
from lightrag.prompt import PROMPTS

# --- Build the ER extraction prompt (same as LightRAG uses internally) ---

ENTITY_TYPES = [
    "Definition", "Section", "Amendment", "Schedule",
    "Act", "Rule", "Authority", "Penalty", "Provision",
    "Industry", "Chemical", "Regulation", "Notification",
]

examples = "\n".join(PROMPTS["entity_extraction_examples"])
context = dict(
    tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
    completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
    entity_types=", ".join(ENTITY_TYPES),
    language="English",
)
examples = examples.format(**context)

SYSTEM_PROMPT = PROMPTS["entity_extraction_system_prompt"].format(
    **{**context, "examples": examples}
)

# --- Test chunks (from Factory Act) ---

CHUNKS = [
    {
        "name": "Definition chunk (Section 2)",
        "text": """[Source: FactoryAct1948.pdf] [Category: Factories Act & Rules] [Type: Definition] [Term: hazardous process] [Location: CHAPTER I > PRELIMINARY]

(cb) "hazardous process" means any process or activity in relation to an industry specified in the First Schedule where, unless special care is taken, raw materials used therein or the intermediate or finished products, bye-products, wastes or effluents thereof would—
(i) cause material impairment to the health of the persons engaged in or connected therewith, or
(ii) result in the pollution of the general environment:
Provided that the State Government may, by notification in the Official Gazette, amend the First Schedule by way of addition, omission or variation of any industry specified in the said Schedule.""",
    },
    {
        "name": "Section chunk (41B - Hazardous Processes)",
        "text": """[Source: FactoryAct1948.pdf] [Category: Factories Act & Rules] [Type: Section] [Section: 41B] [Location: CHAPTER IVA > PROVISIONS RELATING TO HAZARDOUS PROCESSES]

41B. Compulsory disclosure of information by the occupier.—(1) The occupier of every factory involving a hazardous process shall disclose in the manner prescribed all information regarding dangers, including health hazards and the measures to overcome such hazards arising from the exposure to or handling of the materials or substances in the manufacture, transportation, storage and other processes, to the workers employed in the factory, the Chief Inspector, the local authority within whose jurisdiction the factory is situate and the general public in the neighbourhood.
(2) The occupier shall, at the time of registering the factory involving a hazardous process, lay down a detailed policy with respect to the health and safety of the workers employed therein and intimate such policy to the Chief Inspector and the local authority and, thereafter, at such intervals as may be prescribed, inform the said Inspector and the said authority of any change made in the said policy.""",
    },
    {
        "name": "Amendment-heavy chunk",
        "text": """[Source: FactoryAct1948.pdf] [Category: Factories Act & Rules] [Type: Section] [Section: 92] [Location: CHAPTER X > PENALTIES AND PROCEDURE]

92. General penalty for offences.—Save as is otherwise expressly provided in this Act and subject to the provisions of section 93, if in, or in respect of, any factory there is any contravention of any of the provisions of this Act or of any rules made thereunder or of any order in writing given thereunder, the occupier and manager of the factory shall each be guilty of an offence and punishable with imprisonment for a term which may extend to two years or with fine which may extend to one lakh rupees or with both.
1. Subs. by Act 20 of 1987, s. 36, for "two thousand rupees" (w.e.f. 1-12-1987).
2. Ins. by Act 94 of 1976, s. 45 (w.e.f. 26-10-1976).""",
    },
]


def strip_think(text):
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def parse_entities_relations(text):
    """Parse LightRAG-format entity/relation output."""
    td = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
    entities = []
    relations = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith(f"entity{td}"):
            parts = line.split(td)
            if len(parts) >= 4:
                entities.append({
                    "name": parts[1].strip(),
                    "type": parts[2].strip(),
                    "desc": parts[3].strip()[:80],
                })
        elif line.startswith(f"relation{td}"):
            parts = line.split(td)
            if len(parts) >= 5:
                relations.append({
                    "src": parts[1].strip(),
                    "tgt": parts[2].strip(),
                    "kw": parts[3].strip(),
                    "desc": parts[4].strip()[:80],
                })
    return entities, relations


async def call_openai_compat(model, chunk_text, api_key, base_url, label=""):
    """Call any OpenAI-compatible API."""
    from lightrag.llm.openai import openai_complete_if_cache

    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
        **{**context, "input_text": chunk_text}
    )

    start = time.time()
    result = await openai_complete_if_cache(
        model=model,
        prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        history_messages=[],
        api_key=api_key,
        base_url=base_url,
    )
    elapsed = time.time() - start
    result = strip_think(result)
    return result, elapsed


async def call_gemini(chunk_text):
    """Call Gemini 3 Flash."""
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
        **{**context, "input_text": chunk_text}
    )

    model = genai.GenerativeModel(
        "gemini-3-flash-preview",
        system_instruction=SYSTEM_PROMPT,
    )

    start = time.time()
    response = model.generate_content(user_prompt)
    elapsed = time.time() - start
    return response.text, elapsed


async def main():
    print("=" * 80)
    print("ER EXTRACTION MODEL COMPARISON")
    print("=" * 80)
    print(f"\nSystem prompt: ~{len(SYSTEM_PROMPT)//4} tokens")
    print(f"Entity types: {len(ENTITY_TYPES)}")
    print(f"Test chunks: {len(CHUNKS)}")

    TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")

    models = [
        # (display_name, model_id, provider, api_key, base_url)
        ("Llama 8B (Groq)", "llama-3.1-8b-instant", "openai", GROQ_API_KEY, "https://api.groq.com/openai/v1"),
        ("Gemma 4 31B (Together)", "google/gemma-4-31B-it", "openai", TOGETHER_KEY, "https://api.together.xyz/v1"),
        ("GPT-OSS 20B (Together)", "openai/gpt-oss-20b", "openai", TOGETHER_KEY, "https://api.together.xyz/v1"),
        ("Gemini 3 Flash", "gemini-3-flash-preview", "gemini", None, None),
    ]

    all_results = {}

    for chunk_info in CHUNKS:
        chunk_name = chunk_info["name"]
        chunk_text = chunk_info["text"]
        print(f"\n{'─'*80}")
        print(f"CHUNK: {chunk_name}")
        print(f"  Length: {len(chunk_text)} chars (~{len(chunk_text)//4} tokens)")
        print(f"{'─'*80}")

        for model_name, model_id, provider, api_key, base_url in models:
            try:
                if provider == "openai":
                    result, elapsed = await call_openai_compat(model_id, chunk_text, api_key, base_url)
                else:
                    result, elapsed = await call_gemini(chunk_text)

                entities, relations = parse_entities_relations(result)

                print(f"\n  {model_name}")
                print(f"  Time: {elapsed:.2f}s")
                print(f"  Entities: {len(entities)}, Relations: {len(relations)}")

                if entities:
                    print(f"  Entities:")
                    for e in entities:
                        print(f"    [{e['type']:15s}] {e['name']:30s} — {e['desc']}")

                if relations:
                    print(f"  Relations:")
                    for r in relations:
                        print(f"    {r['src']:20s} → {r['tgt']:20s} ({r['kw']})")

                all_results.setdefault(chunk_name, {})[model_name] = {
                    "entities": len(entities),
                    "relations": len(relations),
                    "time": round(elapsed, 2),
                    "entity_names": [e["name"] for e in entities],
                    "entity_types": [e["type"] for e in entities],
                }

            except Exception as e:
                print(f"\n  {model_name}: ERROR — {e}")
                all_results.setdefault(chunk_name, {})[model_name] = {"error": str(e)}

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"\n{'Chunk':<35s} {'Model':<25s} {'Ent':>4s} {'Rel':>4s} {'Time':>6s}")
    print("─" * 80)
    for chunk_name, results in all_results.items():
        for model_name, data in results.items():
            if "error" in data:
                print(f"{chunk_name[:34]:<35s} {model_name:<25s} {'ERR':>4s} {'':>4s} {'':>6s}")
            else:
                print(f"{chunk_name[:34]:<35s} {model_name:<25s} {data['entities']:>4d} {data['relations']:>4d} {data['time']:>5.2f}s")

    # Entity overlap analysis
    print(f"\n{'='*80}")
    print("ENTITY OVERLAP (which models find the same entities?)")
    print(f"{'='*80}")
    for chunk_name, results in all_results.items():
        print(f"\n  {chunk_name}:")
        entity_sets = {}
        for model_name, data in results.items():
            if "entity_names" in data:
                entity_sets[model_name] = set(n.lower() for n in data["entity_names"])

        if len(entity_sets) >= 2:
            model_names = list(entity_sets.keys())
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    a, b = model_names[i], model_names[j]
                    sa, sb = entity_sets[a], entity_sets[b]
                    overlap = sa & sb
                    only_a = sa - sb
                    only_b = sb - sa
                    print(f"    {a} vs {b}:")
                    print(f"      Shared: {len(overlap)} — {', '.join(sorted(overlap)[:5])}")
                    if only_a:
                        print(f"      Only {a[:10]}: {', '.join(sorted(only_a)[:5])}")
                    if only_b:
                        print(f"      Only {b[:10]}: {', '.join(sorted(only_b)[:5])}")


if __name__ == "__main__":
    asyncio.run(main())
