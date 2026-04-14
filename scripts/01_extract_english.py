#!/usr/bin/env python3
"""Extract FactoryAct1948.pdf (English) using Docling's full AI pipeline.

Usage:
    python scripts/01_extract_english.py

Outputs saved to extraction_output/:
    - FactoryAct1948_docling.json   (full DoclingDocument)
    - FactoryAct1948_docling.md     (markdown export)
    - FactoryAct1948_tables.json    (extracted tables)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ENGLISH_DOC, EXTRACTION_OUTPUT_DIR
from src.extract.docling_extractor import extract_pdf_with_docling

if __name__ == "__main__":
    if not ENGLISH_DOC.exists():
        print(f"ERROR: PDF not found at {ENGLISH_DOC}")
        sys.exit(1)

    print(f"=== Extracting English document ===")
    print(f"Source: {ENGLISH_DOC}")
    print(f"Output: {EXTRACTION_OUTPUT_DIR}")
    print()

    result = extract_pdf_with_docling(ENGLISH_DOC, EXTRACTION_OUTPUT_DIR)

    print(f"\n=== Extraction Complete ===")
    print(f"DoclingDocument JSON: {result['doc_json_path']}")
    print(f"Markdown:             {result['markdown_path']}")
    print(f"Tables:               {result['tables_path']}")
    print(f"\nStats:")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
