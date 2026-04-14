# Table Formatting Guidelines

LawMaster SQL outputs are presented as formatted tables, not charts. These guidelines ensure legal data is displayed clearly and citably.

## When to Use `build_table`

- **3+ rows** of results — always present as a table.
- **Comparison queries** — side-by-side values across substances, industries, or categories.
- **List queries** — "list all Red category industries", "show all exposure limits".

## When to Use Plain Text

- **Single value lookups** — "What is the TWA for benzene?" Answer in one sentence.
- **Yes/no questions** — "Is this industry classified as Red?" Answer directly.
- **Counts** — "How many substances are listed?" One sentence.

## Column Header Standards

Use readable, human-friendly column names:

| Raw Column | Display Header |
|------------|---------------|
| substance | Substance |
| time_weighted_avg_ppm | TWA (ppm) |
| time_weighted_avg_mg_m3 | TWA (mg/m3) |
| short_term_exposure_ppm | STEL (ppm) |
| short_term_exposure_mg_m3 | STEL (mg/m3) |
| industry_sector | Industry Sector |
| industry_type | Industry Type |
| category | CPCB Category |
| serial_number | Sr. No. |
| industry_name | Industry |
| hazardous_process_description | Hazardous Process |
| _source_document | Source Document |
| _source_page | Page |

## Formatting Rules

1. **Include source columns** — every table should have "Source Document" and "Page" as the last two columns, or a footnote row citing the source.
2. **Null/empty handling** — display `--` or "Not specified" for NULL or empty values. Never leave blank cells.
3. **Units in headers** — include units in column headers (e.g., "TWA (ppm)") so they don't need to repeat in every cell.
4. **Category coloring** — when describing CPCB categories in text, use the category name clearly: "Red Category", "Orange Category", etc.
5. **Sort order** — default sort by the most natural key:
   - Substances: alphabetical
   - Industries: by category (Red first), then alphabetical
   - Hazardous processes: by serial number
6. **Truncation** — if a description exceeds 80 characters, truncate with "..." in the table and note "full descriptions available on request".
7. **Row limits** — show max 50 rows per table. If more exist, note "Showing 50 of X results. Refine your query for specific entries."

## Citation Footer

For tables with data from a single source, add a footer:

> Source: [Document Name], [Schedule/Table Name], pp. [page range]

For tables combining multiple sources, include source columns per row.
