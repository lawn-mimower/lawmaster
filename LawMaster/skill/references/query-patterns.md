# Query Patterns & Best Practices

## Table Discovery

### List all available tables
```sql
SELECT table_id, table_name, source_document, description
FROM _table_registry
ORDER BY table_name
LIMIT 100
```

### Get column info for a specific table
```sql
SELECT table_id, table_name, column_descriptions
FROM _table_registry
WHERE table_id = 'factories_act_second_schedule'
LIMIT 1
```

## Chemical Exposure Limits (Second Schedule)

### Look up a specific substance
```sql
SELECT substance,
       time_weighted_avg_ppm AS twa_ppm,
       time_weighted_avg_mg_m3 AS twa_mg_m3,
       short_term_exposure_ppm AS stel_ppm,
       short_term_exposure_mg_m3 AS stel_mg_m3,
       _source_document, _source_page
FROM factories_act_second_schedule
WHERE LOWER(substance) LIKE '%benzene%'
LIMIT 100
```

### List all substances with exposure limits
```sql
SELECT substance,
       COALESCE(time_weighted_avg_ppm, '--') AS twa_ppm,
       COALESCE(time_weighted_avg_mg_m3, '--') AS twa_mg_m3,
       _source_document, _source_page
FROM factories_act_second_schedule
ORDER BY substance
LIMIT 100
```

### Find substances with ceiling/STEL limits
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

### Count total substances listed
```sql
SELECT COUNT(*) AS substance_count
FROM factories_act_second_schedule
LIMIT 1
```

## Industry Classification (CPCB)

### Look up category for an industry
```sql
SELECT serial_number, industry_sector, industry_type, category,
       _source_document, _source_page
FROM cpcb_industry_classification
WHERE LOWER(industry_type) LIKE '%chemical%'
ORDER BY category, industry_type
LIMIT 100
```

### List all Red category industries
```sql
SELECT serial_number, industry_sector, industry_type,
       _source_document, _source_page
FROM cpcb_industry_classification
WHERE category = 'Red'
ORDER BY industry_sector, industry_type
LIMIT 100
```

### Count industries by category
```sql
SELECT category,
       COUNT(*) AS industry_count
FROM cpcb_industry_classification
GROUP BY category
ORDER BY CASE category
    WHEN 'Red' THEN 1
    WHEN 'Orange' THEN 2
    WHEN 'Green' THEN 3
    WHEN 'White' THEN 4
END
LIMIT 10
```

### List industries in a sector
```sql
SELECT industry_type, category,
       _source_document, _source_page
FROM cpcb_industry_classification
WHERE LOWER(industry_sector) LIKE '%textile%'
ORDER BY category, industry_type
LIMIT 100
```

### Find category for a specific factory type
```sql
SELECT industry_sector, industry_type, category,
       _source_document, _source_page
FROM cpcb_industry_classification
WHERE LOWER(industry_type) LIKE '%dye%'
   OR LOWER(industry_type) LIKE '%dyeing%'
ORDER BY industry_type
LIMIT 100
```

## Hazardous Processes (First Schedule)

### Check if an industry has hazardous processes
```sql
SELECT industry_name, hazardous_process_description,
       _source_document, _source_page
FROM factories_act_first_schedule
WHERE LOWER(industry_name) LIKE '%chemical%'
   OR LOWER(hazardous_process_description) LIKE '%chemical%'
ORDER BY serial_number
LIMIT 100
```

### List all hazardous processes
```sql
SELECT serial_number, industry_name, hazardous_process_description,
       _source_document, _source_page
FROM factories_act_first_schedule
ORDER BY serial_number
LIMIT 100
```

### Count hazardous industries
```sql
SELECT COUNT(DISTINCT industry_name) AS hazardous_industry_count
FROM factories_act_first_schedule
LIMIT 1
```

## Cross-Table Queries

### Check if a chemical factory is both Red category and hazardous
```sql
SELECT c.industry_type, c.category AS cpcb_category,
       f.hazardous_process_description,
       c._source_document AS cpcb_source,
       f._source_document AS first_schedule_source
FROM cpcb_industry_classification c
LEFT JOIN factories_act_first_schedule f
    ON LOWER(c.industry_type) LIKE '%' || LOWER(f.industry_name) || '%'
    OR LOWER(f.industry_name) LIKE '%' || LOWER(c.industry_type) || '%'
WHERE LOWER(c.industry_type) LIKE '%chemical%'
LIMIT 100
```

## SQLite Best Practices

1. **Case-insensitive search** — SQLite has no `ILIKE`. Use `LOWER(column) LIKE '%term%'` instead.
2. **No `DATE_TRUNC` or `INTERVAL`** — use `date()`, `strftime()`, and `datetime()` for date operations.
3. **No `FILTER` clause** — use `CASE WHEN ... THEN 1 ELSE 0 END` inside `SUM()` for conditional aggregation.
4. **No `::` type casting** — use `CAST(x AS REAL)` or `CAST(x AS INTEGER)`.
5. **CTEs work** — `WITH ... AS` is fully supported.
6. **Window functions work** — `ROW_NUMBER()`, `RANK()`, `LEAD()`, `LAG()` are available.
7. **`COALESCE`** — works the same as PostgreSQL. Use for nullable fields.
8. **`GROUP_CONCAT`** — SQLite equivalent of PostgreSQL's `STRING_AGG`. Use for combining values.
9. **`json_extract()`** — use to query JSON stored in TEXT columns (e.g., `column_descriptions` in `_table_registry`).
10. **Always include provenance** — add `_source_document` and `_source_page` to every query for citation support.
