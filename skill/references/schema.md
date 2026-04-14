# Database Schema

SQLite database at `./data/tables.db`. All tables are read-only extractions from Indian legal documents.

## _table_registry
| Column | Type | Notes |
|--------|------|-------|
| table_id | TEXT, PK | Unique identifier for each extracted table |
| source_document | TEXT, NOT NULL | Legal document the table was extracted from |
| table_name | TEXT, NOT NULL | Human-readable table name |
| description | TEXT | What the table contains |
| page_number | INTEGER | Page in source document |
| column_descriptions | TEXT | JSON object mapping column names to descriptions |
| created_at | TIMESTAMP | When the table was ingested |

**Purpose:** Meta-table that catalogs all extracted tables. Query this first when unsure which table holds the answer.

## factories_act_second_schedule
| Column | Type | Notes |
|--------|------|-------|
| substance | TEXT | Chemical/substance name |
| time_weighted_avg_ppm | TEXT | 8-hour TWA in parts per million |
| time_weighted_avg_mg_m3 | TEXT | 8-hour TWA in mg/m3 |
| short_term_exposure_ppm | TEXT | 15-min STEL in parts per million |
| short_term_exposure_mg_m3 | TEXT | 15-min STEL in mg/m3 |
| _source_page | INTEGER | Page in source document |
| _source_document | TEXT | Source legal document |

**Purpose:** Permissible chemical exposure limits in workplaces. Used to check if a factory's air quality readings comply with legal limits.

## cpcb_industry_classification
| Column | Type | Notes |
|--------|------|-------|
| serial_number | INTEGER | CPCB serial number |
| industry_sector | TEXT | Broad sector (e.g., "Chemical") |
| industry_type | TEXT | Specific industry description |
| category | TEXT | Red, Orange, Green, or White |
| _source_page | INTEGER | Page in source document |
| _source_document | TEXT | Source legal document |

**Purpose:** Central Pollution Control Board classification of industries by pollution potential. Determines environmental clearance requirements.

## factories_act_first_schedule
| Column | Type | Notes |
|--------|------|-------|
| serial_number | INTEGER | Schedule serial number |
| industry_name | TEXT | Industry/sector name |
| hazardous_process_description | TEXT | Description of hazardous process |
| _source_page | INTEGER | Page in source document |
| _source_document | TEXT | Source legal document |

**Purpose:** List of industries involving hazardous processes under the Factories Act. Determines applicability of special safety provisions (Sections 41A-41H).

## Common Column Conventions

All extracted tables include these provenance columns:
- `_source_page` — page number in the original legal document
- `_source_document` — name of the source legal document

These columns enable source citation in every query result.
