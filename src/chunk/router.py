"""Route extracted content into typed chunks using Mistral annotations + regex fallbacks.

Takes raw markdown + annotations from mistral_extractor.py and produces
typed, metadata-enriched chunks ready for LightRAG/SQLite insertion.
"""

import re
import json
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup


def route_content(
    markdown: str,
    annotations: list[dict],
    html_tables: list[dict],
    source_name: str,
    category: str,
) -> dict:
    """Route extracted content into 4 content types.

    Returns:
        {
            "definitions": [{"term": str, "text": str, "meta": str}],
            "sections": [{"number": str, "heading": str, "text": str, "meta": str}],
            "tables": [{"page": int, "html": str, "description": str}],
            "amendments": [{"text": str, "meta": str}],
            "other": [{"text": str, "meta": str}],
        }
    """
    # Build annotation lookup: text_snippet → block info
    annot_map = {}
    for block in annotations:
        snippet = block.get("text_snippet", "")[:40]
        if snippet:
            annot_map[snippet] = block

    # Split markdown into blocks by headings and page breaks
    raw_blocks = _split_into_blocks(markdown)

    definitions = []
    sections = []
    amendments = []
    other_chunks = []

    for block in raw_blocks:
        text = block["text"].strip()
        if len(text) < 30:
            continue

        # Try annotation-based classification first
        ctype = _classify_from_annotations(text, annot_map)

        # Fallback to regex classification
        if ctype is None:
            ctype = _classify_by_regex(text, block.get("heading", ""))

        heading_chain = block.get("heading_chain", "")

        if ctype == "definition":
            # Split into individual definitions
            defs = _extract_definitions(text, source_name, category, heading_chain)
            definitions.extend(defs)

        elif ctype in ("section", "amendment"):
            # Split large sections at sub-section boundaries
            sec_chunks = _chunk_section(text, source_name, category, heading_chain)
            sections.extend(sec_chunks)

        else:
            # Preamble, other, footnotes → still index as general content
            sections.append({
                "number": None,
                "heading": block.get("heading", ""),
                "text": text,
                "meta": f"[Source: {source_name}] [Category: {category}] [Type: {ctype or 'Content'}]"
                        + (f" [Location: {heading_chain}]" if heading_chain else ""),
            })

    # Tables from Mistral HTML extraction
    table_chunks = _process_tables(html_tables, source_name, category)

    return {
        "definitions": definitions,
        "sections": sections,
        "tables": table_chunks,
        "amendments": amendments,
        "other": other_chunks,
    }


def _split_into_blocks(markdown: str) -> list[dict]:
    """Split markdown into blocks by headings, preserving hierarchy."""
    parts = re.split(r'(?=^#{1,4}\s)', markdown, flags=re.MULTILINE)
    blocks = []
    current_headings = {}

    for part in parts:
        part = part.strip()
        if not part:
            continue

        heading = ""
        heading_match = re.match(r'^(#{1,4})\s+(.+?)$', part, re.MULTILINE)
        if heading_match:
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            current_headings[level] = heading
            for l in list(current_headings.keys()):
                if l > level:
                    del current_headings[l]

        heading_chain = " > ".join(
            current_headings[l] for l in sorted(current_headings.keys())
        )

        # Also split by page breaks if block is very large
        if len(part) > 5000 and "\n\n---\n\n" in part:
            page_parts = part.split("\n\n---\n\n")
            for i, pp in enumerate(page_parts):
                if pp.strip():
                    blocks.append({
                        "text": pp.strip(),
                        "heading": heading if i == 0 else "",
                        "heading_chain": heading_chain,
                    })
        else:
            blocks.append({
                "text": part,
                "heading": heading,
                "heading_chain": heading_chain,
            })

    # If no headings at all, split by page breaks
    if len(blocks) <= 1:
        page_parts = markdown.split("\n\n---\n\n")
        blocks = []
        for i, page in enumerate(page_parts):
            page = page.strip()
            if page:
                blocks.append({
                    "text": page,
                    "heading": "",
                    "heading_chain": f"Page {i+1}",
                })

    return blocks


def _classify_from_annotations(text: str, annot_map: dict) -> Optional[str]:
    """Try to match text against annotation blocks."""
    text_start = text[:40]
    for snippet, block in annot_map.items():
        if snippet in text_start or text_start in snippet:
            return block.get("content_type", None)
    return None


def _classify_by_regex(text: str, heading: str) -> str:
    """Regex fallback for content type classification."""
    heading_lower = heading.lower()
    text_lower = text[:500].lower()

    # Definitions
    if any(kw in heading_lower for kw in ["definition", "interpretation", "paribhasha", "परिभाष"]):
        return "definition"
    if re.search(r'\([a-z]+\)\s*["\u201c].+?["\u201d]\s*means', text[:1000]):
        return "definition"
    if "से अभिप्रेत है" in text[:1000]:
        return "definition"

    # Amendments
    if re.search(r'\[(Substituted|Inserted|Omitted|Added|Renumbered)\s+by', text):
        # Only if the amendment markers are the primary content
        amendment_markers = len(re.findall(r'\[(Substituted|Ins\.|Omitted|Added)', text))
        total_sentences = max(1, len(re.findall(r'[.।]', text)))
        if amendment_markers / total_sentences > 0.3:
            return "amendment"

    # Schedules
    if any(kw in heading_lower for kw in ["schedule", "अनुसूची", "appendix", "परिशिष्ट"]):
        return "schedule"

    # Default: section
    return "section"


def _extract_definitions(text: str, source: str, category: str, heading_chain: str) -> list[dict]:
    """Split a definitions block into one chunk per defined term."""
    # English pattern: (a) "term" means...
    en_pattern = r'(\([a-z]+\))\s*["\u201c]([^"\u201d]+)["\u201d]\s*means\s*'
    # Hindi pattern: (क) "term" से अभिप्रेत है
    hi_pattern = r'(\([क-ह]+\))\s*["\u201c]([^"\u201d]+)["\u201d]\s*से\s*अभिप्रेत'

    defs = []
    # Try to split by definition boundaries
    matches = list(re.finditer(en_pattern, text)) + list(re.finditer(hi_pattern, text))

    if matches:
        matches.sort(key=lambda m: m.start())
        for i, match in enumerate(matches):
            term = match.group(2).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            def_text = text[start:end].strip()

            defs.append({
                "term": term,
                "text": def_text,
                "meta": (f"[Source: {source}] [Category: {category}] "
                         f"[Type: Definition] [Term: {term}]"
                         + (f" [Location: {heading_chain}]" if heading_chain else "")),
            })
    else:
        # No pattern matches — keep as single block
        defs.append({
            "term": None,
            "text": text,
            "meta": (f"[Source: {source}] [Category: {category}] [Type: Definition]"
                     + (f" [Location: {heading_chain}]" if heading_chain else "")),
        })

    return defs


def _chunk_section(text: str, source: str, category: str, heading_chain: str,
                   max_tokens: int = 1500) -> list[dict]:
    """Chunk a section, splitting at sub-section boundaries if too large."""
    # Rough token estimate: 1 token ≈ 4 chars
    max_chars = max_tokens * 4

    # Extract section number from text
    sec_match = re.match(r'^[\s#]*(\d+[A-Z]?)\.\s', text)
    sec_number = sec_match.group(1) if sec_match else None

    if len(text) <= max_chars:
        return [{
            "number": sec_number,
            "heading": heading_chain.split(" > ")[-1] if heading_chain else "",
            "text": text,
            "meta": (f"[Source: {source}] [Category: {category}] [Type: Section]"
                     + (f" [Section: {sec_number}]" if sec_number else "")
                     + (f" [Location: {heading_chain}]" if heading_chain else "")),
        }]

    # Split at sub-section boundaries: (1), (2), etc.
    parts = re.split(r'(?=\n\(\d+\)\s)', text)
    chunks = []
    current = ""

    for part in parts:
        if len(current) + len(part) > max_chars and len(current) > 200:
            chunks.append(current.strip())
            current = part
        else:
            current += part

    if current.strip():
        chunks.append(current.strip())

    return [{
        "number": sec_number,
        "heading": heading_chain.split(" > ")[-1] if heading_chain else "",
        "text": chunk,
        "meta": (f"[Source: {source}] [Category: {category}] [Type: Section]"
                 + (f" [Section: {sec_number}]" if sec_number else "")
                 + (f" [Location: {heading_chain}]" if heading_chain else "")),
    } for chunk in chunks]


def _process_tables(html_tables: list[dict], source: str, category: str) -> list[dict]:
    """Parse HTML tables into structured data for SQLite + stub chunks."""
    table_chunks = []

    for tbl in html_tables:
        html = tbl.get("html", "")
        page = tbl.get("page_index", 0)

        try:
            soup = BeautifulSoup(html, "html.parser")
            table_el = soup.find("table")
            if not table_el:
                continue

            rows = []
            headers = []
            for i, tr in enumerate(table_el.find_all("tr")):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if i == 0:
                    headers = cells
                rows.append(cells)

            if not rows:
                continue

            # Generate description from headers
            description = f"Table from {source}, page {page+1}"
            if headers:
                description += f". Columns: {', '.join(h for h in headers if h)}"

            # Compute SQL table name (matches _insert_table_to_sql naming)
            safe_name = re.sub(r'[^a-z0-9_]', '_', Path(source).stem.lower())
            sql_table_name = f"{safe_name}_p{page}"

            table_chunks.append({
                "page": page,
                "html": html,
                "headers": headers,
                "rows": rows,
                "num_rows": len(rows),
                "num_cols": len(headers) if headers else (len(rows[0]) if rows else 0),
                "description": description,
                "sql_table_name": sql_table_name,
                "stub_text": (
                    f"[Source: {source}] [Category: {category}] [Type: Table Stub] "
                    f"[Page: {page+1}] [SQL Table: {sql_table_name}]\n\n"
                    f"{description}. "
                    f"To query this table's data, use: SELECT * FROM \"{sql_table_name}\""
                ),
            })
        except Exception as e:
            print(f"  [WARN] Table parse error on page {page}: {e}")

    return table_chunks
