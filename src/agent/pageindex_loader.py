"""PageIndex instance loader. Initialized once, reused across queries."""

import sys
from pathlib import Path

_client = None

# Self-hosted PageIndex cloned at project root
_PAGEINDEX_LIB = Path(__file__).parent.parent.parent / "pageindex_lib"


def get_pageindex_client():
    """Return a singleton PageIndexClient with all documents indexed."""
    global _client
    if _client is not None:
        return _client

    if str(_PAGEINDEX_LIB) not in sys.path:
        sys.path.insert(0, str(_PAGEINDEX_LIB))
    from pageindex import PageIndexClient

    from src.config import MISTRAL_API_KEY, ENGLISH_DOC, HINDI_DOC, EXTRACTION_OUTPUT_DIR
    import os
    os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

    workspace = str(Path(__file__).parent.parent.parent / "pageindex_storage")
    client = PageIndexClient(model="mistral/mistral-small-latest", workspace=workspace)

    # Index documents if not already cached in workspace
    docs = [
        {"path": str(ENGLISH_DOC), "name": "Factories Act, 1948",
         "md_path": str(EXTRACTION_OUTPUT_DIR / "FactoryAct1948_docling.md")},
        {"path": str(HINDI_DOC), "name": "Capital Subsidy Rules",
         "md_path": str(EXTRACTION_OUTPUT_DIR / "capital_subsidy_Rules_docling.md")},
    ]

    for doc in docs:
        # Check if already indexed by matching doc_name
        existing = next(
            (did for did, d in client.documents.items()
             if d.get("doc_name") == Path(doc["path"]).name
             or d.get("doc_name") == Path(doc["md_path"]).stem),
            None,
        )
        if existing:
            print(f"[PageIndex] {doc['name']} already indexed (doc_id={existing[:8]}...)")
            continue

        # Try PDF first, fall back to markdown
        for attempt_path, mode in [(doc["path"], "pdf"), (doc["md_path"], "md")]:
            if attempt_path is None or not Path(attempt_path).exists():
                continue
            try:
                print(f"[PageIndex] Indexing {doc['name']} from {Path(attempt_path).name} ({mode})...")
                doc_id = client.index(attempt_path, mode=mode)
                print(f"[PageIndex] Indexed {doc['name']} -> doc_id={doc_id[:8]}...")
                break
            except Exception as e:
                print(f"[PageIndex] Failed ({mode}): {e}")
        else:
            print(f"[PageIndex] WARNING: Could not index {doc['name']}")

    print(f"[PageIndex] Ready. {len(client.documents)} document(s) loaded.")
    _client = client
    return _client
