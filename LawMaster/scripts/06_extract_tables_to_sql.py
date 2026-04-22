#!/usr/bin/env python3
"""Extract Docling tables from Factories Act into SQLite.

Tables 0-3 are the Second Schedule (chemical exposure limits) split across pages.
Tables 4-5 are silica-related sub-tables.

Usage:
    python scripts/06_extract_tables_to_sql.py
"""

import sys
import json
import sqlite3
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import EXTRACTION_OUTPUT_DIR, DATA_DIR

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "tables.db"


def clean_substance(name: str) -> str:
    """Remove trailing dots and extra whitespace from substance names."""
    return re.sub(r'[\s.]+$', '', name).strip()


def create_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS _table_registry (
            table_id TEXT PRIMARY KEY,
            source_document TEXT NOT NULL,
            table_name TEXT NOT NULL,
            description TEXT,
            page_number INTEGER,
            column_descriptions TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS factories_act_second_schedule (
            substance TEXT,
            twa_ppm TEXT,
            twa_mg_m3 TEXT,
            stel_ppm TEXT,
            stel_mg_m3 TEXT,
            _source_document TEXT DEFAULT 'Factories Act, 1948',
            _source_table TEXT DEFAULT 'Second Schedule'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS factories_act_second_schedule_silica (
            substance TEXT,
            permissible_limit TEXT,
            unit TEXT,
            _source_document TEXT DEFAULT 'Factories Act, 1948',
            _source_table TEXT DEFAULT 'Second Schedule (Silica)'
        )
    """)
    conn.commit()


def insert_chemical_tables(conn: sqlite3.Connection, tables: list):
    """Tables 0-3: chemical exposure limits (Second Schedule)."""
    rows_inserted = 0
    for table in tables[:4]:
        for row in table["rows"]:
            substance = row[0]
            if not substance or substance.lower() in ("", "substance"):
                continue
            # Skip header rows
            if "ppm" in substance.lower() or "mg/m" in substance.lower():
                continue

            substance = clean_substance(substance)
            twa_ppm = row[1].strip() if len(row) > 1 else ""
            twa_mg = row[2].strip() if len(row) > 2 else ""
            stel_ppm = row[3].strip() if len(row) > 3 else ""
            stel_mg = row[4].strip() if len(row) > 4 else ""

            # Normalize empty markers
            for val in [twa_ppm, twa_mg, stel_ppm, stel_mg]:
                if val in (". .", "..", ""):
                    val = None

            conn.execute(
                "INSERT INTO factories_act_second_schedule (substance, twa_ppm, twa_mg_m3, stel_ppm, stel_mg_m3) VALUES (?, ?, ?, ?, ?)",
                (
                    substance,
                    twa_ppm if twa_ppm not in (". .", "..", "") else None,
                    twa_mg if twa_mg not in (". .", "..", "") else None,
                    stel_ppm if stel_ppm not in (". .", "..", "") else None,
                    stel_mg if stel_mg not in (". .", "..", "") else None,
                ),
            )
            rows_inserted += 1

    conn.commit()
    return rows_inserted


def insert_silica_tables(conn: sqlite3.Connection, tables: list):
    """Tables 4-5: silica exposure limits."""
    rows_inserted = 0
    for table in tables[4:]:
        for row in table["rows"]:
            substance = row[0]
            if not substance or substance.lower() in ("", "substance"):
                continue
            substance = clean_substance(substance)

            # Combine remaining columns as the limit value
            limit_parts = [r.strip() for r in row[1:] if r.strip() and r.strip() not in ("",)]
            limit_val = " ".join(limit_parts) if limit_parts else None

            # Extract unit if present
            unit = None
            if limit_val:
                if "mppcm" in limit_val.lower():
                    unit = "mppcm"
                elif "mg/m" in limit_val.lower():
                    unit = "mg/m3"

            conn.execute(
                "INSERT INTO factories_act_second_schedule_silica (substance, permissible_limit, unit) VALUES (?, ?, ?)",
                (substance, limit_val, unit),
            )
            rows_inserted += 1

    conn.commit()
    return rows_inserted


def register_tables(conn: sqlite3.Connection):
    conn.execute(
        "INSERT OR REPLACE INTO _table_registry VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (
            "factories_act_second_schedule",
            "Factories Act, 1948",
            "Second Schedule - Permissible Levels of Chemical Substances",
            "Permissible exposure limits (TWA and STEL) for chemical substances in workplaces. TWA = Time-Weighted Average (8 hrs). STEL = Short-Term Exposure Limit (15 min).",
            None,
            json.dumps({
                "substance": "Chemical substance name",
                "twa_ppm": "Time-weighted average concentration in ppm (8 hours)",
                "twa_mg_m3": "Time-weighted average concentration in mg/m3 (8 hours)",
                "stel_ppm": "Short-term exposure limit in ppm (15 minutes)",
                "stel_mg_m3": "Short-term exposure limit in mg/m3 (15 minutes)",
            }),
        ),
    )
    conn.execute(
        "INSERT OR REPLACE INTO _table_registry VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)",
        (
            "factories_act_second_schedule_silica",
            "Factories Act, 1948",
            "Second Schedule - Silica Exposure Limits",
            "Permissible exposure limits for silica and related substances (quartz, cristobalite, tridymite, etc.).",
            None,
            json.dumps({
                "substance": "Silica type (quartz, cristobalite, etc.)",
                "permissible_limit": "Permissible concentration limit",
                "unit": "Unit of measurement (mppcm or mg/m3)",
            }),
        ),
    )
    conn.commit()


if __name__ == "__main__":
    tables_path = EXTRACTION_OUTPUT_DIR / "FactoryAct1948_tables.json"
    if not tables_path.exists():
        print(f"ERROR: {tables_path} not found. Run extraction first.")
        sys.exit(1)

    with open(tables_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    print(f"Loaded {len(tables)} tables from Docling extraction")

    # Remove old DB if exists
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    create_schema(conn)

    chem_count = insert_chemical_tables(conn, tables)
    print(f"Inserted {chem_count} chemical exposure limit rows")

    silica_count = insert_silica_tables(conn, tables)
    print(f"Inserted {silica_count} silica exposure limit rows")

    register_tables(conn)
    print(f"Table registry updated")

    # Verify
    print(f"\n--- Verification ---")
    for row in conn.execute("SELECT table_id, table_name FROM _table_registry"):
        print(f"  {row[0]}: {row[1]}")

    print(f"\nSample chemical data:")
    for row in conn.execute("SELECT substance, twa_ppm, twa_mg_m3 FROM factories_act_second_schedule LIMIT 5"):
        print(f"  {row[0]}: TWA {row[1]} ppm / {row[2]} mg/m3")

    print(f"\nBenzene lookup:")
    for row in conn.execute("SELECT * FROM factories_act_second_schedule WHERE LOWER(substance) LIKE '%benzene%'"):
        print(f"  {row}")

    conn.close()
    print(f"\nDatabase saved: {DB_PATH}")
