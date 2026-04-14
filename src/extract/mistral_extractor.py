"""Extract text from Hindi/scanned PDFs using Mistral OCR, then structure via Docling."""

import json
import time
import base64
from pathlib import Path

from mistralai import Mistral
from mistralai.utils.retries import RetryConfig, BackoffStrategy

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat



def extract_pdf_with_mistral(pdf_path: Path, output_dir: Path, api_key: str) -> dict:
    """
    Two-stage extraction:
      1. Mistral OCR → raw markdown + HTML tables
      2. Feed markdown into Docling convert_string → DoclingDocument

    Returns dict with:
        - doc_json: the DoclingDocument as dict
        - markdown: combined markdown string
        - html_tables: list of HTML table strings from Mistral
        - stats: extraction statistics
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem
    client = Mistral(
        api_key=api_key,
        timeout_ms=300_000,
        retry_config=RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=5_000,
                max_interval=60_000,
                max_elapsed_time=300_000,
                exponent=2.0,
            ),
            retry_connection_errors=True,
        ),
    )

    # --- Stage 1: Mistral OCR ---
    print(f"[Mistral] Uploading: {pdf_path.name}")
    start = time.time()

    # Upload file
    with open(pdf_path, "rb") as f:
        uploaded = client.files.upload(
            file={
                "file_name": pdf_path.name,
                "content": f,
            },
            purpose="ocr",
        )

    # Get signed URL
    signed = client.files.get_signed_url(file_id=uploaded.id)

    print(f"[Mistral] Running OCR (table_format=html)...")
    ocr_result = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed.url,
        },
        table_format="html",
        include_image_base64=False,
    )
    ocr_elapsed = time.time() - start
    print(f"[Mistral] OCR done in {ocr_elapsed:.1f}s — {len(ocr_result.pages)} pages")

    # Collect markdown and tables
    page_markdowns = []
    html_tables = []

    for page in ocr_result.pages:
        page_markdowns.append(page.markdown)
        if hasattr(page, "tables") and page.tables:
            for tbl in page.tables:
                html_tables.append({
                    "page_index": page.index,
                    "table_id": tbl.id if hasattr(tbl, "id") else f"tbl-{page.index}",
                    "html": tbl.content if hasattr(tbl, "content") else str(tbl),
                })

    # Save raw Mistral markdown
    combined_md = "\n\n---\n\n".join(page_markdowns)
    raw_md_path = output_dir / f"{stem}_mistral_raw.md"
    raw_md_path.write_text(combined_md, encoding="utf-8")
    print(f"[Mistral] Raw markdown saved: {raw_md_path}")

    # Save HTML tables separately
    if html_tables:
        tables_path = output_dir / f"{stem}_mistral_html_tables.json"
        with open(tables_path, "w", encoding="utf-8") as f:
            json.dump(html_tables, f, indent=2, ensure_ascii=False)
        print(f"[Mistral] {len(html_tables)} HTML tables saved: {tables_path}")

    # Save raw OCR response
    ocr_json_path = output_dir / f"{stem}_mistral_ocr_response.json"
    with open(ocr_json_path, "w", encoding="utf-8") as f:
        # Serialize the OCR result
        ocr_data = {
            "pages": [
                {
                    "index": p.index,
                    "markdown": p.markdown,
                    "tables": [
                        {"id": getattr(t, "id", ""), "content": getattr(t, "content", str(t))}
                        for t in (p.tables or [])
                    ] if hasattr(p, "tables") and p.tables else [],
                }
                for p in ocr_result.pages
            ],
            "model": getattr(ocr_result, "model", "mistral-ocr-latest"),
        }
        json.dump(ocr_data, f, indent=2, ensure_ascii=False)

    # --- Stage 2: Docling structuring ---
    print(f"[Docling] Structuring Mistral markdown...")
    struct_start = time.time()

    converter = DocumentConverter(allowed_formats=[InputFormat.MD])
    result = converter.convert_string(combined_md, format=InputFormat.MD, name=f"{stem}.md")
    doc = result.document
    struct_elapsed = time.time() - struct_start

    # Export DoclingDocument JSON
    doc_dict = doc.export_to_dict()
    json_path = output_dir / f"{stem}_docling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)
    print(f"[Docling] Structured JSON saved: {json_path}")

    # Export final markdown from Docling
    docling_md = doc.export_to_markdown()
    md_path = output_dir / f"{stem}_docling.md"
    md_path.write_text(docling_md, encoding="utf-8")
    print(f"[Docling] Structured markdown saved: {md_path}")

    # Stats
    stats = {
        "source": pdf_path.name,
        "method": "mistral+docling",
        "ocr_elapsed_seconds": round(ocr_elapsed, 2),
        "structuring_elapsed_seconds": round(struct_elapsed, 2),
        "total_elapsed_seconds": round(ocr_elapsed + struct_elapsed, 2),
        "num_pages": len(ocr_result.pages),
        "num_html_tables": len(html_tables),
        "num_docling_texts": len(doc.texts),
        "num_docling_tables": len(doc.tables),
        "raw_markdown_chars": len(combined_md),
        "structured_markdown_chars": len(docling_md),
    }
    print(f"[Done] {stats['num_pages']} pages, {stats['num_html_tables']} HTML tables, "
          f"{stats['num_docling_texts']} text elements")

    return {
        "doc_json_path": str(json_path),
        "markdown_path": str(md_path),
        "raw_markdown_path": str(raw_md_path),
        "html_tables": html_tables,
        "stats": stats,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import HINDI_DOC, EXTRACTION_OUTPUT_DIR, MISTRAL_API_KEY

    result = extract_pdf_with_mistral(HINDI_DOC, EXTRACTION_OUTPUT_DIR, MISTRAL_API_KEY)
    print("\n--- Stats ---")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
