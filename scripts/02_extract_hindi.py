#!/usr/bin/env python3
"""Extract capital_subsidy_Rules.pdf (Hindi) using Mistral OCR + Docling structuring.

Usage:
    python scripts/02_extract_hindi.py

Requires MISTRAL_API_KEY in .env

Outputs saved to extraction_output/:
    - capital_subsidy_Rules_mistral_raw.md           (raw Mistral OCR markdown)
    - capital_subsidy_Rules_mistral_ocr_response.json (full OCR API response)
    - capital_subsidy_Rules_mistral_html_tables.json  (HTML tables from Mistral)
    - capital_subsidy_Rules_docling.json              (structured DoclingDocument)
    - capital_subsidy_Rules_docling.md                (structured markdown)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import HINDI_DOC, EXTRACTION_OUTPUT_DIR, MISTRAL_API_KEY
from src.extract.mistral_extractor import extract_pdf_with_mistral

if __name__ == "__main__":
    if not HINDI_DOC.exists():
        print(f"ERROR: PDF not found at {HINDI_DOC}")
        sys.exit(1)

    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY not found in .env")
        sys.exit(1)

    print(f"=== Extracting Hindi document ===")
    print(f"Source: {HINDI_DOC}")
    print(f"Output: {EXTRACTION_OUTPUT_DIR}")
    print()

    result = extract_pdf_with_mistral(HINDI_DOC, EXTRACTION_OUTPUT_DIR, MISTRAL_API_KEY)

    print(f"\n=== Extraction Complete ===")
    print(f"Raw Mistral MD:       {result['raw_markdown_path']}")
    print(f"DoclingDocument JSON: {result['doc_json_path']}")
    print(f"Structured MD:        {result['markdown_path']}")
    print(f"HTML Tables:          {len(result['html_tables'])} extracted")
    print(f"\nStats:")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
