# SQLite Best Practices

LawMaster uses SQLite (not PostgreSQL). This file covers SQLite-specific patterns, syntax differences, and performance considerations.

## Connection Handling

- The database is a single file at `./data/tables.db`.
- Connections are opened read-only (`?mode=ro` or `PRAGMA query_only = ON`).
- No connection pooling needed — SQLite handles concurrent reads natively via WAL mode.
- Close connections promptly after queries to avoid file locks.

## Syntax Differences from PostgreSQL

| PostgreSQL | SQLite Equivalent |
|------------|-------------------|
| `ILIKE '%term%'` | `LOWER(col) LIKE '%term%'` |
| `column::NUMERIC` | `CAST(column AS REAL)` |
| `DATE_TRUNC('month', col)` | `strftime('%Y-%m', col)` |
| `NOW()` | `datetime('now')` |
| `CURRENT_DATE` | `date('now')` |
| `INTERVAL '30 days'` | `datetime('now', '-30 days')` |
| `col - INTERVAL '7 days'` | `datetime(col, '-7 days')` |
| `COUNT(*) FILTER (WHERE ...)` | `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` |
| `STRING_AGG(col, ', ')` | `GROUP_CONCAT(col, ', ')` |
| `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)` | Not natively supported. Use subquery with `LIMIT 1 OFFSET count/2` |
| `DISTINCT ON (col)` | Use `GROUP BY` + subquery or window function with `ROW_NUMBER()` |
| `BOOLEAN` | `INTEGER` (0 = false, 1 = true) |

## Supported Features

- **CTEs** (`WITH ... AS`) — fully supported, including recursive CTEs.
- **Window functions** — `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`, `LEAD()`, `LAG()`, `SUM() OVER()`.
- **JSON functions** — `json_extract()`, `json_each()`, `json_array_length()` for querying JSON stored in TEXT columns.
- **`COALESCE`** — works identically to PostgreSQL.
- **`NULLIF`** — works identically.
- **`ROUND()`** — works identically.
- **Subqueries** — fully supported in `SELECT`, `FROM`, `WHERE`, and `HAVING`.

## Performance Considerations

1. **No query planner hints** — SQLite auto-optimizes. Don't try to force index usage.
2. **LIMIT early** — always apply `LIMIT` to prevent scanning entire tables on large datasets.
3. **Avoid unnecessary JOINs** — most legal table lookups are single-table queries. Only JOIN when cross-referencing between tables (e.g., CPCB + First Schedule).
4. **Text search** — `LOWER(col) LIKE '%term%'` does a full table scan. Acceptable for legal tables (typically < 10K rows) but avoid stacking multiple LIKE conditions unnecessarily.
5. **Aggregate first, then JOIN** — when combining aggregates with registry lookups, aggregate in a CTE first.

## Allowed Operations

- `SELECT` and `WITH` (CTEs) only
- `PRAGMA table_info(table_name)` for column discovery (read-only)

## Blocked Operations

- Any write operation: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`
- `ATTACH DATABASE` — prevents filesystem access
- `load_extension()` — prevents code execution
- Write-mode PRAGMAs: `journal_mode`, `synchronous`, `cache_size`, etc.
