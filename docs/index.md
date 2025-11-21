# Moltres Design Notes

## Layers

> Supports Python 3.9+ and assumes SQLAlchemy-compatible drivers.

1. **Expression System** – `moltres.expressions` models columns, literals, math/string ops, and
   aggregations as symbolic trees.
2. **Logical Planner** – `moltres.logical` turns DataFrame actions into plan nodes (Project, Filter,
   Join, Aggregate, Sort, Limit, TableScan).
3. **SQL Compiler** – `moltres.sql.compiler` converts the plan into ANSI SQL with basic dialect
   awareness (SQLite and PostgreSQL quoting, case sensitivity, etc.).
4. **Execution Engine** – `moltres.engine` manages SQLAlchemy connections and materializes results as
   lists of dicts, pandas DataFrames, or polars DataFrames depending on configuration. Supports
   streaming execution via `fetch_stream()` for cursor-based pagination of large result sets.
5. **DDL Layer** – `moltres.sql.ddl` and `moltres.table.schema` provide table creation and schema
   definition utilities that compile to CREATE TABLE and DROP TABLE statements.
6. **Data Loading Layer** – `moltres.dataframe.reader` (now `DataLoader`) provides data source loaders with:
   - File formats: `load.csv()`, `load.json()`, `load.jsonl()`, `load.parquet()`, `load.text()` - all return `Records`
   - Generic format: `load.format(format_name).load(path)` - returns `Records`
   - Schema inference: automatic type detection from data
   - Explicit schemas: `.schema([ColumnDef(...), ...])` for type control
   - Format options: `.option(key, value)` for format-specific settings
   - Streaming: `.stream()` for chunked reading of large files (configurable chunk_size)
   - **Note:** File readers return `Records` (not `DataFrame`). For SQL operations, use `db.table(name).select()`
7. **Write Layer** – `moltres.dataframe.writer` provides DataFrame persistence with:
   - Table writes: `save_as_table()` with schema inference and automatic table creation
   - Existing table inserts: `insertInto()` for appending to pre-existing tables
   - File formats: `csv()`, `json()`, `jsonl()`, `parquet()` with format-specific options
   - Partitioning: `partitionBy()` for directory-based data partitioning
   - Multiple write modes: append, overwrite, error_if_exists
   - Streaming: `.stream()` for chunked writing without materializing entire DataFrame
8. **Mutation Layer** – `moltres.table.mutations` provides eager `insert`, `update`, and `delete`
   helpers that share the same connection stack as queries.

## Workflows

- Create tables programmatically using `db.create_table(name, columns, ...)` with schema definitions
  built from `column()` helpers. Drop tables with `db.drop_table(name)`.
- Load data from files using `db.load.csv(path)`, `db.load.json(path)`, `db.load.parquet(path)`, etc.
  These return `Records` which can be inserted into tables or iterated. Use `.schema([...])` for
  explicit schemas and `.option(key, value)` for format-specific settings.
- For SQL operations on database tables, use `db.table("name").select(...)` to get a DataFrame.
- Use `db.table("name").select(...)` to construct lazy DataFrames.
- Compose joins via `df.join(other_df, on=[("left_col", "right_col")])` and aggregations via
  `df.group_by("country").agg(sum(col("amount")).alias("total"))`.
- Call `collect()` to execute a plan; Moltres compiles SQL at that point. Use `collect(stream=True)` to
  get an iterator of row chunks for large datasets, or enable streaming mode with `.stream()` on readers/writers.
- Write DataFrames to tables using `df.write.save_as_table(name)` with automatic schema inference
  and table creation, or `df.write.insertInto(name)` for existing tables. Control behavior with
  `.mode("append|overwrite|error_if_exists")` and `.schema([ColumnDef(...), ...])` for explicit schemas.
- Write DataFrames to files using `df.write.csv(path)`, `df.write.json(path)`, `df.write.parquet(path)`,
  or the generic `df.write.save(path, format="...")`. Use `.partitionBy("col1", "col2")` for
  directory-based partitioning and `.option(key, value)` for format-specific settings.
- Enable streaming for large file datasets: `db.load.stream().option("chunk_size", 10000).csv("large.csv")`
  returns streaming `Records` that iterate row-by-row. For SQL queries, use `collect(stream=True)` to
  process chunks incrementally. Use `df.write.stream().save_as_table("large_table")` to write without materializing.
- Perform table mutations through the `TableHandle` (`db.table("orders").insert([...])`).

## Testing & Tooling

- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`
- Linting: Ruff + mypy configuration is defined in `pyproject.toml`.
- Optional deps: install with `pip install '.[polars]'` to enable polars fetches.
