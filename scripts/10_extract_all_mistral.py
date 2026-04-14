#!/usr/bin/env python3
"""Phase 1: Extract ALL 43 PDFs via Mistral OCR → raw markdown.

Saves raw markdown per doc. Skips docs that already have a markdown file
(resume-safe). Run this first, then run 11_index_all.py.

Usage:
    python scripts/10_extract_all_mistral.py
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import EXTRACTION_OUTPUT_DIR, MISTRAL_API_KEY, DOCS_BASE

from mistralai import Mistral
from mistralai.utils.retries import RetryConfig, BackoffStrategy

# --- Retry configuration ---
# SDK handles: 429, 500, 502, 503, 504 via RetryConfig
# Application-level handles: 520 (Cloudflare), timeouts, connection resets
APP_MAX_RETRIES = 3
APP_RETRY_DELAYS = [10, 30, 60]  # seconds between app-level retries

# Status codes the SDK does NOT retry but are transient
TRANSIENT_CODES = {"520", "521", "522", "523", "524"}


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a transient server/network error."""
    msg = str(exc).lower()
    # Cloudflare 52x errors
    for code in TRANSIENT_CODES:
        if code in msg:
            return True
    # Connection / timeout patterns
    if any(kw in msg for kw in ("timed out", "timeout", "connection reset",
                                 "connection aborted", "remotedisconnected",
                                 "service unavailable")):
        return True
    return False


# All 43 PDFs, organized by category for metadata tagging
CATEGORIES = {
    "factories act": "Factories Act & Rules",
    "pollution control board": "Pollution Control & Environment",
    "industrial policy, ammendments and notifications": "Chhattisgarh Industrial Policy",
}

OUTPUT_DIR = EXTRACTION_OUTPUT_DIR / "mistral"


def get_all_pdfs() -> list[dict]:
    """Discover all PDFs in corpus with metadata."""
    pdfs = []
    for pdf_path in sorted(DOCS_BASE.rglob("*.pdf")):
        rel = pdf_path.relative_to(DOCS_BASE)
        category_dir = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        category = CATEGORIES.get(category_dir, category_dir)
        pdfs.append({
            "path": pdf_path,
            "name": pdf_path.stem,
            "category": category,
            "category_dir": category_dir,
        })
    return pdfs


def extract_one(client: Mistral, pdf_info: dict) -> dict:
    """Extract a single PDF via Mistral OCR. Returns stats dict."""
    pdf_path = pdf_info["path"]
    name = pdf_info["name"]
    out_dir = OUTPUT_DIR / pdf_info["category_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"{name}.md"
    meta_path = out_dir / f"{name}_meta.json"

    # Skip if already extracted
    if md_path.exists() and md_path.stat().st_size > 0:
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
        print(f"  [SKIP] {pdf_path.name} (already extracted)")
        return {"status": "skipped", "name": name, **meta}

    print(f"  [OCR] {pdf_path.name}")
    start = time.time()

    try:
        # Upload
        with open(pdf_path, "rb") as f:
            uploaded = client.files.upload(
                file={"file_name": pdf_path.name, "content": f},
                purpose="ocr",
            )

        # Get signed URL
        signed = client.files.get_signed_url(file_id=uploaded.id)

        # OCR
        ocr_result = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed.url},
            table_format="html",
            include_image_base64=False,
        )
        elapsed = time.time() - start

        # Collect markdown per page
        page_markdowns = []
        html_tables = []
        for page in ocr_result.pages:
            page_markdowns.append(page.markdown)
            if hasattr(page, "tables") and page.tables:
                for tbl in page.tables:
                    html_tables.append({
                        "page_index": page.index,
                        "html": getattr(tbl, "content", str(tbl)),
                    })

        # Save raw markdown
        combined_md = "\n\n---\n\n".join(page_markdowns)
        md_path.write_text(combined_md, encoding="utf-8")

        # Save HTML tables if any
        if html_tables:
            tables_path = out_dir / f"{name}_tables.json"
            with open(tables_path, "w", encoding="utf-8") as f:
                json.dump(html_tables, f, indent=2, ensure_ascii=False)

        # Save metadata
        meta = {
            "status": "ok",
            "name": name,
            "source": pdf_path.name,
            "category": pdf_info["category"],
            "pages": len(ocr_result.pages),
            "chars": len(combined_md),
            "tables": len(html_tables),
            "elapsed_s": round(elapsed, 1),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"       {meta['pages']}p, {meta['chars']} chars, "
              f"{meta['tables']} tables, {elapsed:.1f}s")
        return meta

    except Exception as e:
        elapsed = time.time() - start
        transient = _is_transient(e)
        print(f"  [{'TRANSIENT' if transient else 'ERROR'}] {pdf_path.name}: {e}")
        meta = {
            "status": "transient" if transient else "error",
            "name": name,
            "error": str(e),
            "transient": transient,
            "elapsed_s": round(elapsed, 1),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return meta


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = get_all_pdfs()
    total_pages = 0

    print(f"=== Mistral OCR Extraction: {len(pdfs)} documents ===\n")

    client = Mistral(
        api_key=MISTRAL_API_KEY,
        timeout_ms=300_000,  # 5 min per request (upload/OCR can be slow)
        retry_config=RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=5_000,   # 5s first retry
                max_interval=60_000,      # cap at 60s between retries
                max_elapsed_time=300_000, # give up after 5 min of retrying
                exponent=2.0,             # 5s → 10s → 20s → 40s → 60s
            ),
            retry_connection_errors=True,
        ),
    )
    results = []
    start_all = time.time()

    for i, pdf_info in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf_info['category']}")

        # Application-level retry for transient errors the SDK doesn't cover
        result = None
        for attempt in range(1, APP_MAX_RETRIES + 1):
            result = extract_one(client, pdf_info)
            if result["status"] != "transient":
                break
            if attempt < APP_MAX_RETRIES:
                delay = APP_RETRY_DELAYS[attempt - 1]
                print(f"  [RETRY] attempt {attempt}/{APP_MAX_RETRIES}, "
                      f"waiting {delay}s before retry...")
                time.sleep(delay)
                # Remove error meta so extract_one doesn't skip it
                meta_path = (OUTPUT_DIR / pdf_info["category_dir"]
                             / f"{pdf_info['name']}_meta.json")
                meta_path.unlink(missing_ok=True)
            else:
                print(f"  [GIVE UP] {pdf_info['name']} after {APP_MAX_RETRIES} attempts")
                result["status"] = "error"  # promote to hard error

        results.append(result)
        if result.get("pages"):
            total_pages += result["pages"]

    elapsed_all = time.time() - start_all

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n{'='*55}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*55}")
    print(f"  Total docs: {len(pdfs)}")
    print(f"  Extracted: {len(ok)} ({sum(r.get('pages', 0) for r in ok)} pages)")
    print(f"  Skipped: {len(skipped)} (already done)")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed_all:.0f}s ({elapsed_all/60:.1f}min)")

    if errors:
        print(f"\n  Failed docs:")
        for r in errors:
            print(f"    - {r['name']}: {r.get('error', '?')}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Manifest: {manifest_path}")

    # Cost estimate
    cost = total_pages * 0.002  # $2/1000 pages regular
    print(f"\n  Estimated Mistral OCR cost: ~${cost:.2f} ({total_pages} pages @ $2/1000)")


if __name__ == "__main__":
    main()
