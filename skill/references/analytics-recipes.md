# Analytics Recipes

When the user asks for a "report", "compliance check", "audit", or any of the trigger phrases below, follow the corresponding multi-query recipe. Run each query as a separate `execute_sql` call, then synthesize findings into a concise narrative with tables where appropriate.

---

## 1. Compliance Profile for an Industry

**Triggers:** "compliance profile for [industry]", "what regulations apply to [industry]", "regulatory overview for [industry]", "tell me about [industry] compliance"

Builds a complete regulatory picture for a given industry type.

**Q1 -- CPCB classification**
```sql
SELECT industry_sector, industry_type, category,
       _source_document, _source_page
FROM cpcb_industry_classification
WHERE LOWER(industry_type) LIKE '%{industry}%'
   OR LOWER(industry_sector) LIKE '%{industry}%'
ORDER BY category, industry_type
LIMIT 20
```

**Q2 -- Hazardous process check**
```sql
SELECT industry_name, hazardous_process_description,
       _source_document, _source_page
FROM factories_act_first_schedule
WHERE LOWER(industry_name) LIKE '%{industry}%'
   OR LOWER(hazardous_process_description) LIKE '%{industry}%'
ORDER BY serial_number
LIMIT 20
```

**Q3 -- Relevant exposure limits (if chemical/manufacturing)**
```sql
SELECT substance,
       COALESCE(time_weighted_avg_ppm, '--') AS twa_ppm,
       COALESCE(time_weighted_avg_mg_m3, '--') AS twa_mg_m3,
       COALESCE(short_term_exposure_ppm, '--') AS stel_ppm,
       _source_document, _source_page
FROM factories_act_second_schedule
WHERE LOWER(substance) LIKE '%{relevant_chemical}%'
ORDER BY substance
LIMIT 50
```

**Synthesis:** Present as a structured compliance profile:
- CPCB category (Red/Orange/Green/White) and what it means for environmental clearance
- Whether the industry involves hazardous processes under First Schedule
- Applicable exposure limits if chemical handling is involved
- Cite source documents and pages for each finding

---

## 2. Exposure Limit Report

**Triggers:** "exposure limits", "chemical limits report", "occupational health limits", "workplace air quality", "permissible limits for [chemicals]"

Comprehensive overview of chemical exposure limits.

**Q1 -- All substances with TWA limits**
```sql
SELECT substance,
       COALESCE(time_weighted_avg_ppm, '--') AS twa_ppm,
       COALESCE(time_weighted_avg_mg_m3, '--') AS twa_mg_m3,
       _source_document, _source_page
FROM factories_act_second_schedule
WHERE time_weighted_avg_ppm IS NOT NULL
  AND time_weighted_avg_ppm != '--'
ORDER BY substance
LIMIT 100
```

**Q2 -- Substances with STEL limits**
```sql
SELECT substance,
       short_term_exposure_ppm AS stel_ppm,
       short_term_exposure_mg_m3 AS stel_mg_m3,
       _source_document, _source_page
FROM factories_act_second_schedule
WHERE short_term_exposure_ppm IS NOT NULL
  AND short_term_exposure_ppm != '--'
ORDER BY substance
LIMIT 100
```

**Q3 -- Total count**
```sql
SELECT COUNT(*) AS total_substances,
       SUM(CASE WHEN short_term_exposure_ppm IS NOT NULL
                 AND short_term_exposure_ppm != '--' THEN 1 ELSE 0 END) AS with_stel
FROM factories_act_second_schedule
LIMIT 1
```

**Synthesis:** Lead with summary stats ("X substances listed, Y have STEL limits"). Present the full table using `build_table`. Note any substances with only mg/m3 values (no ppm equivalent). Cite the Factories Act Second Schedule as the source.

---

## 3. Industry Classification Overview

**Triggers:** "industry classification", "CPCB categories", "pollution categories", "Red/Orange/Green/White", "category breakdown"

Full breakdown of industry classifications.

**Q1 -- Category distribution**
```sql
SELECT category,
       COUNT(*) AS industry_count,
       ROUND(CAST(COUNT(*) AS REAL) / (SELECT COUNT(*) FROM cpcb_industry_classification) * 100, 1) AS percentage
FROM cpcb_industry_classification
GROUP BY category
ORDER BY CASE category
    WHEN 'Red' THEN 1 WHEN 'Orange' THEN 2
    WHEN 'Green' THEN 3 WHEN 'White' THEN 4 END
LIMIT 10
```

**Q2 -- Sector-wise category distribution**
```sql
SELECT industry_sector,
       SUM(CASE WHEN category = 'Red' THEN 1 ELSE 0 END) AS red,
       SUM(CASE WHEN category = 'Orange' THEN 1 ELSE 0 END) AS orange,
       SUM(CASE WHEN category = 'Green' THEN 1 ELSE 0 END) AS green,
       SUM(CASE WHEN category = 'White' THEN 1 ELSE 0 END) AS white,
       COUNT(*) AS total
FROM cpcb_industry_classification
GROUP BY industry_sector
ORDER BY total DESC
LIMIT 30
```

**Q3 -- Most populated sectors**
```sql
SELECT industry_sector, COUNT(*) AS industry_count
FROM cpcb_industry_classification
GROUP BY industry_sector
ORDER BY industry_count DESC
LIMIT 15
```

**Synthesis:** Lead with headline numbers ("X industries classified: Y Red, Z Orange..."). Present sector breakdown using `build_table`. Highlight sectors with the highest proportion of Red category industries. Cite source document.

---

## 4. Hazardous Process Audit

**Triggers:** "hazardous processes", "first schedule", "hazardous industries", "hazardous audit", "which industries are hazardous"

Complete listing and analysis of hazardous processes.

**Q1 -- All hazardous processes**
```sql
SELECT serial_number, industry_name, hazardous_process_description,
       _source_document, _source_page
FROM factories_act_first_schedule
ORDER BY serial_number
LIMIT 100
```

**Q2 -- Count by industry**
```sql
SELECT industry_name,
       COUNT(*) AS process_count,
       GROUP_CONCAT(hazardous_process_description, '; ') AS processes
FROM factories_act_first_schedule
GROUP BY industry_name
ORDER BY process_count DESC
LIMIT 30
```

**Q3 -- Cross-reference with CPCB categories**
```sql
SELECT DISTINCT f.industry_name, f.hazardous_process_description,
       c.category AS cpcb_category
FROM factories_act_first_schedule f
LEFT JOIN cpcb_industry_classification c
    ON LOWER(c.industry_type) LIKE '%' || LOWER(f.industry_name) || '%'
ORDER BY f.serial_number
LIMIT 50
```

**Synthesis:** Present the full hazardous process list using `build_table`. Highlight industries with multiple hazardous processes. Note which hazardous industries also fall under Red category (double regulatory burden). Explain legal implications: Sections 41A-41H apply to all listed industries.

---

## 5. Substance-Specific Compliance Check

**Triggers:** "is [substance] safe at [level]", "check [substance] limit", "compliance check for [substance]", "[substance] permissible limit"

Quick lookup for a specific substance with regulatory context.

**Q1 -- Substance limits**
```sql
SELECT substance,
       time_weighted_avg_ppm AS twa_ppm,
       time_weighted_avg_mg_m3 AS twa_mg_m3,
       short_term_exposure_ppm AS stel_ppm,
       short_term_exposure_mg_m3 AS stel_mg_m3,
       _source_document, _source_page
FROM factories_act_second_schedule
WHERE LOWER(substance) LIKE '%{substance}%'
LIMIT 10
```

**Q2 -- Related substances (same chemical family)**
```sql
SELECT substance,
       COALESCE(time_weighted_avg_ppm, '--') AS twa_ppm,
       COALESCE(time_weighted_avg_mg_m3, '--') AS twa_mg_m3,
       _source_document, _source_page
FROM factories_act_second_schedule
WHERE LOWER(substance) LIKE '%{chemical_family}%'
  AND LOWER(substance) NOT LIKE '%{exact_substance}%'
ORDER BY substance
LIMIT 20
```

**Synthesis:** Lead with the specific limit value and units. If the user provided a measured value, compare it against the legal limit and state whether it is compliant. List related substances for context. Always cite the Factories Act Second Schedule.

---

## 6. Data Coverage Report

**Triggers:** "what tables do you have", "what data is available", "show me the registry", "data coverage", "what can I query"

Overview of all ingested tables and their contents.

**Q1 -- Full registry**
```sql
SELECT table_id, table_name, source_document, description, page_number
FROM _table_registry
ORDER BY source_document, table_name
LIMIT 100
```

**Q2 -- Row counts per table (run dynamically per table_id)**
For each table found in Q1, run:
```sql
SELECT COUNT(*) AS row_count FROM {table_id} LIMIT 1
```

**Synthesis:** Present as a `build_table` with columns: Table Name, Source Document, Description, Row Count. This gives the user a map of what structured data is queryable vs. what they should ask the RAG tool for.

---

## Recipe Behavior Rules

1. **Always run all queries in a recipe** — partial reports are worse than no report.
2. **Synthesize, don't dump** — the narrative summary is the deliverable, not the raw tables.
3. **Cite every finding** — include source document and page for each data point.
4. **Adapt the search term** — replace `{industry}`, `{substance}`, etc. with the user's actual query terms. Use `LOWER()` + `LIKE` for fuzzy matching.
5. **Use readable names** — "Red Category" not "Red", "Time-Weighted Average" not "time_weighted_avg_ppm".
6. **If a query returns no rows**, note it clearly: "No matching entries found in [table]. This [substance/industry] may not be listed in the currently indexed schedules. Try the RAG search tool for narrative legal text."
7. **Use `build_table` for lists** — any result with 3+ rows should be presented as a formatted table.
8. **Cross-reference when relevant** — if an industry query matches both CPCB classification and First Schedule, present both findings together.
9. **Legal context matters** — briefly explain the regulatory significance of findings (e.g., "Red category means Environmental Clearance is required from SEIAA/MoEFCC").
10. **Suggest next steps** — after presenting findings, suggest what the user might want to check next via the RAG tool (e.g., "For the full text of Section 41A requirements, use the RAG search").
