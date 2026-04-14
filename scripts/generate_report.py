#!/usr/bin/env python3
"""Generate LawMaster System Report as PDF."""

import json
import sqlite3
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib import colors

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT = PROJECT_ROOT / "LawMaster_System_Report.pdf"

# Colors
DARK = HexColor("#1a1a2e")
ACCENT = HexColor("#16213e")
GREEN = HexColor("#0f3d3e")
BLUE = HexColor("#2a5a8a")
LIGHT_BG = HexColor("#f0f2f5")
WHITE = HexColor("#ffffff")
TEXT = HexColor("#1a1a1a")
MUTED = HexColor("#6b7280")
SUCCESS = HexColor("#10b981")
WARN = HexColor("#f59e0b")
GRAY_LINE = HexColor("#d1d5db")


def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "DocTitle", parent=styles["Title"],
        fontSize=28, leading=34, textColor=DARK,
        spaceAfter=6, alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontSize=12, leading=16, textColor=MUTED,
        spaceAfter=24,
    ))
    styles.add(ParagraphStyle(
        "SectionHead", parent=styles["Heading1"],
        fontSize=16, leading=22, textColor=DARK,
        spaceBefore=20, spaceAfter=10,
        borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SubHead", parent=styles["Heading2"],
        fontSize=13, leading=18, textColor=ACCENT,
        spaceBefore=14, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=TEXT,
        spaceAfter=8, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        "BodyBold", parent=styles["Normal"],
        fontSize=10, leading=15, textColor=TEXT,
        spaceAfter=4, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Metric", parent=styles["Normal"],
        fontSize=22, leading=26, textColor=DARK,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "MetricLabel", parent=styles["Normal"],
        fontSize=9, leading=12, textColor=MUTED,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "CodeBlock", parent=styles["Normal"],
        fontSize=8.5, leading=12, textColor=HexColor("#374151"),
        fontName="Courier", spaceAfter=4,
        leftIndent=12,
    ))
    styles.add(ParagraphStyle(
        "StatusGreen", parent=styles["Normal"],
        fontSize=10, textColor=SUCCESS, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "StatusYellow", parent=styles["Normal"],
        fontSize=10, textColor=WARN, fontName="Helvetica-Bold",
    ))
    return styles


def make_metric_card(value, label, styles):
    """Create a metric display."""
    return Table(
        [[Paragraph(str(value), styles["Metric"])],
         [Paragraph(label, styles["MetricLabel"])]],
        colWidths=[120],
        rowHeights=[32, 18],
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROUNDEDCORNERS", [6, 6, 6, 6]),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 8),
        ]),
    )


def make_table(headers, rows, col_widths=None):
    """Create a styled data table."""
    data = [headers] + rows
    if col_widths is None:
        col_widths = [None] * len(headers)

    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRAY_LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ])
    return Table(data, colWidths=col_widths, style=style, repeatRows=1)


def hr():
    return HRFlowable(width="100%", thickness=0.5, color=GRAY_LINE, spaceAfter=8, spaceBefore=4)


def build_report():
    styles = get_styles()
    story = []

    # ─── TITLE PAGE ───
    story.append(Spacer(1, 80))
    story.append(Paragraph("LawMaster", styles["DocTitle"]))
    story.append(Paragraph("Legal Compliance RAG System - Technical Report", styles["DocSubtitle"]))
    story.append(hr())
    story.append(Spacer(1, 12))

    # Key metrics row
    metrics = Table(
        [[
            make_metric_card("1,128", "KG Entities", styles),
            make_metric_card("1,548", "KG Relations", styles),
            make_metric_card("969", "Vector Chunks", styles),
            make_metric_card("152", "Amendments", styles),
        ]],
        colWidths=[130, 130, 130, 130],
        style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
    )
    story.append(metrics)
    story.append(Spacer(1, 12))

    metrics2 = Table(
        [[
            make_metric_card("120", "SQL Rows", styles),
            make_metric_card("43", "PDFs in Corpus", styles),
            make_metric_card("2", "Docs Indexed", styles),
            make_metric_card("47/50", "Test Score", styles),
        ]],
        colWidths=[130, 130, 130, 130],
        style=TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]),
    )
    story.append(metrics2)
    story.append(Spacer(1, 16))

    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
        styles["Body"],
    ))
    story.append(Paragraph("Authors: Mihir Mohite, Claude Opus 4.6", styles["Body"]))

    story.append(PageBreak())

    # ─── TABLE OF CONTENTS ───
    story.append(Paragraph("Contents", styles["SectionHead"]))
    story.append(hr())
    toc_items = [
        "1. Executive Summary",
        "2. System Architecture",
        "3. Ingestion Pipeline",
        "4. Data Stores",
        "5. Query Layer (Agno Agent)",
        "6. Evaluation Results",
        "7. Current Status & Next Steps",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles["Body"]))
    story.append(PageBreak())

    # ─── 1. EXECUTIVE SUMMARY ───
    story.append(Paragraph("1. Executive Summary", styles["SectionHead"]))
    story.append(hr())
    story.append(Paragraph(
        "LawMaster is a proof-of-concept legal compliance RAG system designed for Indian industrial "
        "and manufacturing law. It combines a knowledge graph-augmented vector search (LightRAG), "
        "structured SQL tables for regulatory data, and an amendment cross-reference index into a "
        "unified agentic chatbot powered by Agno and Gemini Flash.",
        styles["Body"],
    ))
    story.append(Paragraph(
        "The system was built and validated in a single session. Two documents were processed: "
        "the Factories Act, 1948 (English, via Docling) and the Chhattisgarh Capital Subsidy Rules, 2024 "
        "(Hindi, via Mistral OCR + Docling). The PoC demonstrates accurate multi-source retrieval "
        "across legal text, structured tables, and amendment history, scoring 47/50 on five hard "
        "compliance questions.",
        styles["Body"],
    ))

    story.append(Paragraph("Key Capabilities", styles["SubHead"]))
    capabilities = [
        ["Legal text search", "KG + vector hybrid retrieval via LightRAG (mix mode)"],
        ["Structured data lookup", "SQL queries against regulatory tables (exposure limits, classifications)"],
        ["Amendment tracking", "152 amendments extracted via Gemini LLM, linked to target sections"],
        ["Agentic reasoning", "Agno agent with ReasoningTools (think/analyze) for multi-step queries"],
        ["Multi-source synthesis", "Agent autonomously combines RAG + SQL + amendments in one answer"],
        ["Hindi support", "Mistral OCR extraction validated on Chhattisgarh state documents"],
    ]
    story.append(make_table(
        ["Capability", "Implementation"],
        capabilities,
        col_widths=[150, 340],
    ))
    story.append(PageBreak())

    # ─── 2. SYSTEM ARCHITECTURE ───
    story.append(Paragraph("2. System Architecture", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("High-Level Architecture", styles["SubHead"]))
    arch_text = """The system follows a three-layer architecture: (1) an ingestion pipeline that extracts,
    structures, and indexes legal documents; (2) three data stores (LightRAG for text, SQLite for tables,
    JSON for amendments); and (3) an Agno-based agentic query layer that orchestrates retrieval across
    all sources via a Gemini Flash LLM with function calling."""
    story.append(Paragraph(arch_text, styles["Body"]))

    story.append(Paragraph("Component Stack", styles["SubHead"]))
    stack = [
        ["Layer", "Technology", "Purpose"],
    ]
    stack_data = [
        ["PDF Extraction (EN)", "Docling 2.70.0", "Heron layout model + TableFormer for structure + tables"],
        ["PDF Extraction (HI)", "Mistral OCR 3", "97.55% Hindi accuracy, HTML table extraction"],
        ["Document Structuring", "Docling DoclingDocument", "Unified intermediate format, HierarchicalChunker"],
        ["Vector + KG Store", "LightRAG (lightrag-hku)", "NanoVectorDB + NetworkX knowledge graph"],
        ["Embeddings", "BGE-large-en-v1.5", "1024-dim, local inference, offline mode"],
        ["LLM (Extraction)", "Gemini 2.0 Flash", "Entity/relation extraction for KG construction"],
        ["LLM (Query)", "Gemini 3 Flash Preview", "Agent orchestration, response generation"],
        ["Structured Store", "SQLite", "Regulatory tables (exposure limits, classifications)"],
        ["Amendment Index", "JSON", "152 amendments extracted via Gemini one-shot"],
        ["Agent Framework", "Agno 2.5.2", "ReAct loop, custom tools, Streamlit/FastAPI integration"],
        ["Frontend", "HTML + FastAPI", "SSE streaming, tool call display"],
    ]
    story.append(make_table(
        ["Layer", "Technology", "Purpose"],
        stack_data,
        col_widths=[120, 140, 230],
    ))
    story.append(PageBreak())

    # ─── 3. INGESTION PIPELINE ───
    story.append(Paragraph("3. Ingestion Pipeline", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("Hybrid Extraction Architecture", styles["SubHead"]))
    story.append(Paragraph(
        "The pipeline uses a dual-path extraction strategy: Docling's full AI pipeline (Heron layout "
        "model + TableFormer) for English digital PDFs, and Mistral OCR followed by Docling structuring "
        "for Hindi/scanned documents. Both paths converge on a DoclingDocument, ensuring consistent "
        "downstream processing.",
        styles["Body"],
    ))

    story.append(Paragraph("English Path (Factories Act, 1948)", styles["BodyBold"]))
    eng_stats = [
        ["Metric", "Value"],
        ["Source file", "FactoryAct1948.pdf (722 KB)"],
        ["Extraction time", "10.9 seconds"],
        ["Text elements", "1,304"],
        ["Tables extracted", "6 (Second Schedule - chemical exposure limits)"],
        ["Markdown output", "208,390 characters"],
        ["Chunks indexed", "969 (after filtering <30 char chunks)"],
    ]
    story.append(make_table(["Metric", "Value"], eng_stats[1:], col_widths=[150, 340]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Hindi Path (Capital Subsidy Rules, 2024)", styles["BodyBold"]))
    hindi_stats = [
        ["Source file", "capital_subsidy_Rules.pdf (2.9 MB)"],
        ["Mistral OCR time", "5.5 seconds"],
        ["Pages processed", "11"],
        ["Docling structuring", "0.11 seconds"],
        ["Text elements", "191"],
        ["Tables extracted", "0 (document is rules/text, no tabular data)"],
        ["Markdown output", "26,880 characters"],
    ]
    story.append(make_table(["Metric", "Value"], hindi_stats, col_widths=[150, 340]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("LightRAG Indexing", styles["BodyBold"]))
    story.append(Paragraph(
        "The Factories Act was indexed into LightRAG using Gemini 2.0 Flash for entity/relation "
        "extraction (custom entity types: Definition, Section, Amendment, Schedule, Act, Rule, "
        "Authority, Penalty, Provision) with BGE-large-en-v1.5 for embeddings. Configuration: "
        "llm_model_max_async=16, entity_extract_max_gleaning=0, chunk_token_size=1500.",
        styles["Body"],
    ))
    idx_stats = [
        ["Total indexing time", "228 seconds"],
        ["Knowledge graph nodes", "1,128"],
        ["Knowledge graph edges", "1,548"],
        ["Vector chunks", "969"],
        ["LLM cache entries", "1,071"],
    ]
    story.append(make_table(["Metric", "Value"], idx_stats, col_widths=[150, 340]))
    story.append(PageBreak())

    # ─── 4. DATA STORES ───
    story.append(Paragraph("4. Data Stores", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("4.1 LightRAG (Vector + Knowledge Graph)", styles["SubHead"]))
    story.append(Paragraph(
        "LightRAG combines a NanoVectorDB vector store with a NetworkX knowledge graph. "
        "Queries use 'mix' mode: entity search (low-level keywords) + relation search "
        "(high-level keywords) + direct vector chunk retrieval, merged via round-robin.",
        styles["Body"],
    ))

    story.append(Paragraph("4.2 SQLite Tables", styles["SubHead"]))
    sql_tables = [
        ["factories_act_second_schedule", "112 rows", "Chemical exposure limits (TWA, STEL in ppm and mg/m3)"],
        ["factories_act_second_schedule_silica", "8 rows", "Silica-specific exposure limits"],
        ["_table_registry", "2 entries", "Table metadata and column descriptions"],
    ]
    story.append(make_table(
        ["Table", "Rows", "Description"],
        sql_tables,
        col_widths=[190, 60, 240],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("4.3 Amendment Index", styles["SubHead"]))
    story.append(Paragraph(
        "152 amendment annotations extracted from the Factories Act via a single Gemini 3 Flash "
        "API call. Each amendment includes target section, amendment type, amending act, effective "
        "date, and description.",
        styles["Body"],
    ))
    amend_types = [
        ["substituted", "73", "Text replaced with new version"],
        ["inserted", "59", "New clause/section/explanation added"],
        ["omitted", "7", "Text removed"],
        ["renumbered", "7", "Section/explanation renumbered"],
        ["added", "4", "New provision added"],
        ["repealed", "2", "Section repealed entirely"],
    ]
    story.append(make_table(
        ["Type", "Count", "Description"],
        amend_types,
        col_widths=[100, 50, 340],
    ))
    story.append(Paragraph(
        "149 of 152 amendments (98%) have an effective date extracted. All 152 have a target "
        "section identified.",
        styles["Body"],
    ))
    story.append(PageBreak())

    # ─── 5. QUERY LAYER ───
    story.append(Paragraph("5. Query Layer (Agno Agent)", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph(
        "The query layer uses the Agno framework (v2.5.2) to build a ReAct-style agent with "
        "Gemini 3 Flash Preview as the LLM. The agent has access to four tool groups and "
        "autonomously decides which to call based on the query.",
        styles["Body"],
    ))

    story.append(Paragraph("Agent Tools", styles["SubHead"]))
    tools_data = [
        ["ReasoningTools", "think(), analyze()", "Plan approach, evaluate completeness"],
        ["LightRAGTool", "search_legal_text()", "Search legal provisions, definitions, sections"],
        ["SQLTools", "list_tables(), describe_table(),\nrun_sql_query()", "Query structured regulatory data"],
        ["AmendmentTool", "lookup_amendments()", "Check amendment history for a section"],
    ]
    story.append(make_table(
        ["Toolkit", "Functions", "Purpose"],
        tools_data,
        col_widths=[110, 155, 225],
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Query Flow Patterns", styles["SubHead"]))
    patterns = [
        ["Pure RAG", "Definition/section lookup", "search_legal_text"],
        ["Pure SQL", "Exposure limit lookup", "list_tables + run_sql_query"],
        ["RAG + SQL", "Hazardous substance safety", "search_legal_text + run_sql_query"],
        ["Amendment-aware", "Section change history", "search_legal_text + lookup_amendments"],
        ["Clarification", "Ambiguous compliance question", "No tools (agent asks user)"],
        ["Out of corpus", "Act not in knowledge base", "search_legal_text (empty) + inform user"],
    ]
    story.append(make_table(
        ["Pattern", "Example", "Tools Called"],
        patterns,
        col_widths=[100, 170, 220],
    ))
    story.append(PageBreak())

    # ─── 6. EVALUATION RESULTS ───
    story.append(Paragraph("6. Evaluation Results", styles["SectionHead"]))
    story.append(hr())
    story.append(Paragraph(
        "Five hard compliance questions were tested against ground truth extracted directly from "
        "the source documents. The agent was evaluated on factual accuracy, completeness, correct "
        "tool usage, and source citations.",
        styles["Body"],
    ))

    eval_data = [
        ["Q1: Working hours\n& overtime", "RAG x2 +\nanalyze", "10/10",
         "All values correct (48/week, 9/day, 2x overtime). "
         "Added spread-over and rest intervals as bonus."],
        ["Q2: Carbon Monoxide\nexposure limits", "SQL + RAG x3\n+ analyze", "10/10",
         "SQL returned exact values (50/400 ppm). "
         "Cited S36, S41F, S37. Hybrid RAG+SQL worked."],
        ["Q3: Section 2(m)\namendment history", "RAG x5 +\namend x2 +\nanalyze", "8/10",
         "Got 3 major amendment years correctly. "
         "Missed 2 minor sub-amendments (wording changes)."],
        ["Q4: Child employment\npenalties", "RAG x3 +\namend x2", "10/10",
         "All penalties correct. Added S94 (enhanced repeat penalty) "
         "beyond ground truth."],
        ["Q5: Benzene safety\nprovisions", "SQL + RAG x4\n+ analyze", "9/10",
         "Correct exposure limits from SQL. Good Chapter IVA coverage. "
         "Missing: First Schedule entry 26."],
    ]
    story.append(make_table(
        ["Question", "Tools Used", "Score", "Assessment"],
        eval_data,
        col_widths=[105, 75, 40, 270],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Overall Score: 47/50 (94%)", styles["SectionHead"]))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Key Observations", styles["SubHead"]))
    observations = [
        "The agent naturally combined RAG and SQL tools when answers required both text and data.",
        "ReasoningTools (analyze) allowed the agent to evaluate completeness before answering.",
        "Amendment lookups were triggered proactively when discussing section histories.",
        "The agent refined searches when initial results were insufficient (multi-search pattern).",
        "Two gaps identified: (1) First Schedule cross-references not always followed, "
        "(2) Minor amendment sub-details sometimes missed.",
    ]
    for obs in observations:
        story.append(Paragraph(f"  {obs}", styles["Body"]))

    story.append(PageBreak())

    # ─── 7. STATUS & NEXT STEPS ───
    story.append(Paragraph("7. Current Status & Next Steps", styles["SectionHead"]))
    story.append(hr())

    story.append(Paragraph("Component Status", styles["SubHead"]))
    status_data = [
        ["Docling extraction (English)", "Complete", "1,304 texts, 6 tables"],
        ["Mistral OCR (Hindi)", "Complete", "11 pages, 191 texts"],
        ["LightRAG indexing (English)", "Complete", "1,128 nodes, 1,548 edges"],
        ["LightRAG indexing (Hindi)", "Pending", "Ready to append"],
        ["SQL table extraction", "Complete", "120 rows across 2 tables"],
        ["Amendment extraction", "Complete", "152 amendments via Gemini"],
        ["SQL skill (tailored)", "Complete", "Adapted for legal domain"],
        ["Agno agent + tools", "Complete", "4 tool groups, ReAct loop"],
        ["HTML frontend + FastAPI", "Complete", "SSE streaming, tool display"],
        ["Full corpus (43 PDFs)", "Pending", "Only 2 of 43 docs processed"],
    ]
    story.append(make_table(
        ["Component", "Status", "Details"],
        status_data,
        col_widths=[160, 70, 260],
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Next Steps", styles["SubHead"]))
    next_steps = [
        ("Scale to full corpus",
         "Process remaining 41 PDFs through the hybrid extraction pipeline. "
         "Estimated: ~15 min for Docling, ~2 min for Mistral OCR, ~30 min for LightRAG indexing."),
        ("Hindi document indexing",
         "Append the extracted Hindi document to LightRAG. Requires multilingual "
         "embedding model (BGE-M3) or translation-augmented indexing."),
        ("Table extraction at scale",
         "Extract tables from CPCB classification list, hazardous waste rules, and "
         "industrial policy schedules into SQLite."),
        ("Streamlit production UI",
         "Replace basic HTML frontend with full Streamlit app including session "
         "management, document upload, and export capabilities."),
        ("User context integration",
         "Allow users to specify their industry, scale, and jurisdiction for "
         "personalized compliance assessments (Query Pattern B from design spec)."),
        ("Cross-document graph linking",
         "Build edges between entities across different Acts (e.g., Factories Act "
         "references to Water Act, Air Act) with corpus boundary detection."),
    ]
    for title, desc in next_steps:
        story.append(Paragraph(f"<b>{title}</b>", styles["Body"]))
        story.append(Paragraph(desc, styles["Body"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 24))
    story.append(hr())
    story.append(Paragraph(
        "LawMaster PoC - Built April 7, 2026",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=9, textColor=MUTED, alignment=TA_CENTER),
    ))

    # Build
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=25 * mm,
        rightMargin=25 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title="LawMaster System Report",
        author="Mihir Mohite",
    )
    doc.build(story)
    print(f"Report saved: {OUTPUT}")


if __name__ == "__main__":
    build_report()
