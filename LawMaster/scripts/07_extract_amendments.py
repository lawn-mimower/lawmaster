#!/usr/bin/env python3
"""Extract amendments from Factories Act using Gemini LLM one-shot extraction.

Feeds the full markdown to Gemini with a structured JSON schema.
The LLM identifies all amendment annotations and returns structured data.

Usage:
    python scripts/07_extract_amendments.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import EXTRACTION_OUTPUT_DIR, DATA_DIR, GEMINI_API_KEY

DATA_DIR.mkdir(parents=True, exist_ok=True)
AMENDMENTS_PATH = DATA_DIR / "amendments.json"

EXTRACTION_PROMPT = """You are a legal document analyst. Extract ALL amendment annotations from the following Indian legal text (Factories Act, 1948).

Amendment annotations look like:
- "Subs. by Act 94 of 1976, s. 2, for sub-clause (ii) (w.e.f. 26-10-1976)"
- "Ins. by Act 20 of 1987, s. 2 (w.e.f. 1-12-1987)"
- "Omitted by s. 2, ibid. (w.e.f. 1-12-1987)"
- "The words '...' omitted by Act 51 of 1970"
- Footnote markers like "1[...text...]" or "2[(...)]" indicating substituted text

For EACH amendment found, extract:
- target_section: the section of the Factories Act being amended (e.g., "2", "41B", "7A"). Determine this from the surrounding context — the amendment footnote typically appears right after or within a section.
- amendment_type: one of "substituted", "inserted", "omitted", "added", "renumbered", "repealed"
- amending_act: the Act that made the change (e.g., "Act 94 of 1976", "Act 20 of 1987", "A.O. 1950")
- amending_section: the section of the amending Act (e.g., "s. 2", "s. 9")
- effective_date: when it took effect, in DD-MM-YYYY format (from "w.e.f." dates). null if not stated.
- description: brief description of what was changed (e.g., "sub-clause (ii) substituted", "words 'except Jammu and Kashmir' omitted")

Return a JSON array. Be thorough — this Act has many footnote-style amendments throughout.

---

LEGAL TEXT:

"""


def extract_amendments_with_llm(markdown: str) -> list[dict]:
    """Use Gemini to extract amendments from the full document."""
    from google import genai

    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Split into chunks if too long (Gemini Flash handles ~1M tokens, should be fine)
    prompt = EXTRACTION_PROMPT + markdown

    print(f"[Gemini] Sending {len(prompt)} chars for amendment extraction...")
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.1,
        },
    )

    raw_text = response.text.strip()
    amendments = json.loads(raw_text)

    # Ensure it's a list
    if isinstance(amendments, dict) and "amendments" in amendments:
        amendments = amendments["amendments"]

    # Add target_act to all
    for a in amendments:
        a["target_act"] = "Factories Act, 1948"

    return amendments


if __name__ == "__main__":
    md_path = EXTRACTION_OUTPUT_DIR / "FactoryAct1948_docling.md"
    if not md_path.exists():
        print(f"ERROR: {md_path} not found.")
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set in .env")
        sys.exit(1)

    markdown = md_path.read_text(encoding="utf-8")
    print(f"Loaded markdown: {len(markdown)} chars")

    amendments = extract_amendments_with_llm(markdown)
    print(f"Extracted {len(amendments)} amendments")

    # Stats
    types = {}
    for a in amendments:
        t = a.get("amendment_type", "unknown")
        types[t] = types.get(t, 0) + 1
    print(f"Types: {types}")

    with_section = sum(1 for a in amendments if a.get("target_section"))
    print(f"With target section: {with_section}/{len(amendments)}")

    # Save
    with open(AMENDMENTS_PATH, "w", encoding="utf-8") as f:
        json.dump(amendments, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {AMENDMENTS_PATH}")

    # Sample
    print(f"\n--- Sample ---")
    for a in amendments[:8]:
        print(f"  [{a.get('amendment_type')}] Section {a.get('target_section', '?')}: "
              f"{a.get('description', '')[:70]} "
              f"(w.e.f. {a.get('effective_date', '?')})")
