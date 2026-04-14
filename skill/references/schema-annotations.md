# Schema Annotations

Human-written notes on column semantics, units, and domain conventions that cannot be inferred from the schema alone.

## Provenance Columns (all tables)

- `_source_page` — page number in the original legal document. Always include in queries to support citation.
- `_source_document` — name of the source Act/Rule/Notification. Always include in queries to support citation.

## Column Semantics

### _table_registry
- `table_id` — snake_case identifier matching the actual SQLite table name. Use this to discover available tables.
- `column_descriptions` — JSON string, not a parsed JSON object. Use `json_extract()` if querying specific column metadata.
- `source_document` — the Act, Rule, or Notification the table was extracted from (e.g., "Factories Act, 1948").

### factories_act_second_schedule
- `substance` — chemical or compound name as written in the schedule. May include variations (e.g., "Benzene (Benzol)"). Search with `LIKE '%benzene%'` for fuzzy matching.
- `time_weighted_avg_ppm` — 8-hour Time-Weighted Average concentration in parts per million. TEXT type because some entries have qualifiers like "C" (ceiling) or "--" (not established).
- `time_weighted_avg_mg_m3` — same limit expressed in milligrams per cubic meter.
- `short_term_exposure_ppm` — 15-minute Short-Term Exposure Limit in ppm. May be NULL or "--" if no STEL is specified for the substance.
- `short_term_exposure_mg_m3` — same STEL in mg/m3.

**Display conventions:** Always show both ppm and mg/m3 values when available. Label as "TWA" and "STEL" respectively. Show "--" or "Not specified" for missing limits.

### cpcb_industry_classification
- `industry_sector` — broad grouping (e.g., "Chemical", "Textile", "Food"). Use for sector-level filtering.
- `industry_type` — specific industry description (e.g., "Dye and dye intermediate manufacturing"). This is the primary search field for industry lookups.
- `category` — one of: `Red`, `Orange`, `Green`, `White`. Determines environmental clearance requirements:
  - **Red**: Highest pollution potential. Requires Environmental Clearance from SEIAA/MoEFCC.
  - **Orange**: Moderate pollution potential. Requires Consent to Operate from SPCB.
  - **Green**: Low pollution potential. Simplified consent process.
  - **White**: Practically non-polluting. Generally exempt from environmental clearance.

**Display conventions:** Always display category as "Red Category", "Orange Category", etc. When listing industries, group or sort by category for readability.

### factories_act_first_schedule
- `industry_name` — industry or sector listed in the First Schedule. Broader than specific factory types.
- `hazardous_process_description` — specific processes within that industry that are classified as hazardous. Multiple processes may exist per industry.

**Legal significance:** Industries listed here trigger mandatory requirements under Sections 41A-41H of the Factories Act: Site Appraisal Committees, compulsory disclosure of hazards, emergency standards, worker medical examinations, and Safety Committee formation.

## Units Reference

| Column Pattern | Unit | Notes |
|----------------|------|-------|
| `*_ppm` | Parts per million | Volume-based concentration in air |
| `*_mg_m3` | mg/m3 | Mass-based concentration in air |
| `*_hours` | Hours | Time durations |
| `*_days` | Days | Time durations |
| `*_amount` or `*_fee` | INR (Indian Rupees) | Monetary values |
| `*_rate` | Percentage | Unless otherwise noted |

## No Sensitive Columns

All data is extracted from publicly available Indian legal documents. There are no sensitive or restricted columns. No access controls needed.
