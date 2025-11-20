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
   lists of dicts, pandas DataFrames, or polars DataFrames depending on configuration.
5. **Mutation Layer** – `moltres.table.mutations` provides eager `insert`, `update`, and `delete`
   helpers that share the same connection stack as queries.

## Workflows

- Use `db.table("name").select(...)` to construct lazy DataFrames.
- Compose joins via `df.join(other_df, on=[("left_col", "right_col")])` and aggregations via
  `df.group_by("country").agg(sum(col("amount")).alias("total"))`.
- Call `collect()` to execute a plan; Moltres compiles SQL at that point.
- Perform table mutations through the `TableHandle` (`db.table("orders").insert([...])`).

## Testing & Tooling

- Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest`
- Linting: Ruff + mypy configuration is defined in `pyproject.toml`.
- Optional deps: install with `pip install '.[polars]'` to enable polars fetches.
