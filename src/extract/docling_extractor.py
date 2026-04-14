"""Extract structured documents from English PDFs using Docling's full AI pipeline."""

import json
import time
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


def extract_pdf_with_docling(pdf_path: Path, output_dir: Path) -> dict:
    """
    Run Docling's full pipeline on a PDF: Heron layout model + TableFormer.

    Returns dict with:
        - doc_json: the full DoclingDocument as dict
        - markdown: the document as markdown string
        - tables: list of extracted tables
        - stats: extraction statistics
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = pdf_path.stem

    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=False,  # English digital PDF — no OCR needed
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"[Docling] Processing: {pdf_path.name}")
    start = time.time()
    result = converter.convert(str(pdf_path))
    elapsed = time.time() - start
    doc = result.document

    # Export markdown
    markdown = doc.export_to_markdown()
    md_path = output_dir / f"{stem}_docling.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(f"[Docling] Markdown saved: {md_path}")

    # Export full document as JSON
    doc_dict = doc.export_to_dict()
    json_path = output_dir / f"{stem}_docling.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2, ensure_ascii=False)
    print(f"[Docling] JSON saved: {json_path}")

    # Extract tables separately
    tables = []
    for i, table_item in enumerate(doc.tables):
        table_data = {
            "index": i,
            "num_rows": table_item.data.num_rows if table_item.data else 0,
            "num_cols": table_item.data.num_cols if table_item.data else 0,
        }
        if table_item.data:
            # Build rows from grid
            grid = table_item.data.grid
            rows = []
            for row in grid:
                rows.append([cell.text for cell in row])
            table_data["rows"] = rows
        tables.append(table_data)

    tables_path = output_dir / f"{stem}_tables.json"
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)
    print(f"[Docling] {len(tables)} tables saved: {tables_path}")

    # Stats
    stats = {
        "source": pdf_path.name,
        "method": "docling",
        "elapsed_seconds": round(elapsed, 2),
        "num_texts": len(doc.texts),
        "num_tables": len(doc.tables),
        "num_pictures": len(doc.pictures),
        "markdown_chars": len(markdown),
    }
    print(f"[Docling] Done in {elapsed:.1f}s — {stats['num_texts']} text elements, {stats['num_tables']} tables")

    return {
        "doc_json_path": str(json_path),
        "markdown_path": str(md_path),
        "tables_path": str(tables_path),
        "tables": tables,
        "stats": stats,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.config import ENGLISH_DOC, EXTRACTION_OUTPUT_DIR

    result = extract_pdf_with_docling(ENGLISH_DOC, EXTRACTION_OUTPUT_DIR)
    print("\n--- Stats ---")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")
