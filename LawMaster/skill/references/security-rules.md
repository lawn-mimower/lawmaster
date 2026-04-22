# Security Rules

## Read-Only Enforcement
1. **Only SELECT queries** — never execute INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or ATTACH.
2. **Always include LIMIT** (max 100 rows).

## SQL Injection Prevention
3. **No multi-statement queries** — semicolons within a query body are rejected. One statement per execution.
4. **SQL comments are stripped** before validation — `--` line comments and `/* */` block comments are removed to prevent bypass.
5. **Queries must start with `SELECT` or `WITH`** (CTEs only).

## SQLite-Specific Restrictions
6. **No ATTACH DATABASE** — prevents attaching external databases or accessing the filesystem.
7. **No PRAGMA writes** — `PRAGMA` is allowed only for read-only introspection (e.g., `PRAGMA table_info(table_name)`). Never allow `PRAGMA journal_mode`, `PRAGMA synchronous`, or any write-mode PRAGMAs.
8. **No `load_extension()`** — loading external extensions is forbidden.

## Query Constraints
9. **Statement timeout** — all queries are capped at 10 seconds.
10. **No system table access** — queries must not reference `sqlite_master`, `sqlite_sequence`, or `sqlite_temp_master` unless using them to discover table structure as a fallback.

## Data Integrity
11. **No sensitive data exists** — all table data is from publicly available Indian legal documents. No access control filtering is needed.
12. **Source provenance required** — every query result presented to the user should include `_source_document` and `_source_page` for traceability.
