# Moltres vs SQLFrame Comparison

## Executive Summary

This document compares [Moltres](https://github.com/eddiethedean/moltres) and [SQLFrame](https://github.com/eakmanrq/sqlframe): two Python libraries that provide a DataFrame API with SQL pushdown execution. Both enable running transformation pipelines on database engines without Spark clusters. They differ in API style (drop-in PySpark vs native PySpark-style), supported backends, and feature focus (read/transform vs full CRUD and async).

### Key Findings

- **API style**: SQLFrame implements the PySpark DataFrame API as a drop-in: use `activate(engine="...")` then standard `pyspark.sql` imports; Moltres provides a native PySpark-style API (e.g. `connect()`, `db.table().select()`) and optional Pandas/Polars interfaces, with both camelCase and snake_case method names.
- **Backends**: SQLFrame supports BigQuery, Databricks, DuckDB, Postgres, Snowflake, Spark, GizmoSQL, Redshift (community), and a Standalone session for SQL-only generation; Moltres is SQLAlchemy-based (SQLite, PostgreSQL, MySQL, DuckDB, and any SQLAlchemy-supported database).
- **CRUD**: Moltres provides INSERT (via `Records`), UPDATE, and DELETE with DataFrame-style syntax; SQLFrame focuses on read and transform (no UPDATE/DELETE in its public API).
- **Async**: Moltres has full async/await support (`async_connect()`, `AsyncDatabase`); SQLFrame does not advertise async support in its README or overview docs.
- **SQL visibility**: SQLFrame offers `df.sql(optimize=True)` for human-readable SQL and optional OpenAI-based SQL enhancement; Moltres offers `show_sql()`, `.sql`, `plan_summary()`, `visualize_plan()`, `validate()`, and `performance_hints()`.

### Use Case Recommendations

- **Choose Moltres** when:
  - You use traditional SQL databases (PostgreSQL, MySQL, SQLite) or any SQLAlchemy-backed engine.
  - You need UPDATE/DELETE or bulk INSERT with a DataFrame-like API.
  - You want full async/await for all operations.
  - You need framework integrations (FastAPI, Django, Streamlit, SQLModel) or transaction/batch support.
  - You prefer a single `pip install moltres` with optional extras.

- **Choose SQLFrame** when:
  - You need first-class support for BigQuery, Snowflake, or Databricks.
  - You want to run existing PySpark code with minimal changes (drop-in `activate()` + `pyspark.sql` imports).
  - You want a Standalone session to generate SQL without a database connection.
  - You prefer engine-specific installs (`pip install sqlframe[bigquery]`, etc.) and optional Spark dialect or engine-native dialect configuration.

---

## 1. Initialization and Connection

### SQLFrame

```python
from sqlframe import activate

# Option A: Replace PySpark – activate then use pyspark imports
activate(engine="duckdb")
from pyspark.sql import SparkSession
session = SparkSession.builder.getOrCreate()

# Option B: Engine-native session (no pyspark imports)
from sqlframe.duckdb import DuckDBSession
session = DuckDBSession.builder.getOrCreate()
```

### Moltres

```python
from moltres import connect, async_connect

# Synchronous connection
db = connect("sqlite:///example.db")
db = connect("postgresql://user:pass@localhost/db")

# Async connection
db = async_connect("postgresql+asyncpg://user:pass@localhost/db")
db = async_connect("sqlite+aiosqlite:///example.db")
```

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **Entry point** | `SparkSession` or engine-native session (e.g. `DuckDBSession`) | `Database` from `connect()` or `async_connect()` |
| **Activation** | `activate(engine="...")` for drop-in PySpark | No activation; direct `connect(dsn)` |
| **Configuration** | Per-engine install and optional dialect config | DSN / env vars; SQLAlchemy connection options |
| **Async support** | Not documented in overview | Full async via `async_connect()` and `AsyncDatabase` |

---

## 2. DataFrame API and PySpark Compatibility

### SQLFrame

SQLFrame implements the PySpark DataFrame API so that existing PySpark code can run with minimal changes. After `activate(engine="...")`, you use `pyspark.sql` (SparkSession, functions as F, Window). Method names follow PySpark’s camelCase (e.g. `groupBy`, `orderBy`, `withColumn`).

```python
from sqlframe import activate
activate(engine="bigquery")
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window

session = SparkSession.builder.getOrCreate()
df = (
    session.table('"bigquery-public-data".samples.natality')
    .where(F.col("ever_born") == 1)
    .groupBy("year")
    .agg(F.count("*").alias("num_single_child_families"))
    .withColumn("last_year_num", F.lag(F.col("num_single_child_families"), 1).over(Window.orderBy("year")))
    .limit(5)
)
```

### Moltres

Moltres provides a native PySpark-style API (no pyspark dependency). It supports both camelCase and snake_case (e.g. `group_by` / `groupBy`, `order_by` / `orderBy`). Optional Pandas and Polars interfaces are available for result handling or style preference.

```python
from moltres import col, connect
from moltres.expressions import functions as F

db = connect("sqlite:///example.db")
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[col("orders.customer_id") == col("customers.id")])
    .where(col("active") == True)
    .group_by("country")
    .agg(F.sum(col("amount")).alias("total_amount"))
)
results = df.collect()
```

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **PySpark compatibility** | Drop-in: use `pyspark.sql` after `activate()` | Native API with high PySpark method compatibility; no pyspark dependency |
| **Naming** | PySpark camelCase | Both camelCase and snake_case |
| **Other APIs** | — | Optional Pandas and Polars interfaces |

---

## 3. Supported Backends and Execution

### SQLFrame

- **Engines**: BigQuery, Databricks, DuckDB, Postgres, Snowflake, Spark, GizmoSQL (community), Redshift (community).
- **Standalone**: A session that generates SQL only, with no database connection.
- **Install**: Per-engine extras, e.g. `pip install sqlframe[bigquery]`, `sqlframe[postgres]`, `sqlframe[duckdb]`.

### Moltres

- **Backends**: SQLAlchemy-based; officially documented for SQLite, PostgreSQL, MySQL, DuckDB. Any SQLAlchemy-supported database works (e.g. BigQuery/Snowflake via community drivers, but not first-class in docs).
- **No standalone mode**: Execution is always tied to a `Database` connection (sync or async).
- **Install**: Single `pip install moltres`; optional extras for async drivers, Pandas, Polars, integrations.

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **First-class engines** | BigQuery, Databricks, DuckDB, Postgres, Snowflake, Spark, etc. | SQLite, PostgreSQL, MySQL, DuckDB (SQLAlchemy) |
| **Other backends** | GizmoSQL, Redshift (community) | Any SQLAlchemy-supported DB (e.g. BigQuery/Snowflake via drivers) |
| **SQL-only (no DB)** | Yes (Standalone session) | No |
| **Execution model** | Per-engine; SQL generated for target engine | Single stack: logical plan → SQL → SQLAlchemy execution |

---

## 4. CRUD and Mutations

### SQLFrame

SQLFrame is oriented around reading and transforming data (e.g. `session.table()`, `read.csv`, transformations, `df.write.saveAsTable()`). Its README and overview docs do not describe UPDATE or DELETE operations on existing rows.

### Moltres

Moltres provides full SQL CRUD with a DataFrame-like and database API:

```python
from moltres import col, connect
from moltres.io.records import Records

db = connect("sqlite:///example.db")

# Insert rows
Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
], database=db).insert_into("users")

# Update rows
db.update("users", where=col("active") == 0, set={"active": 1})

# Delete rows
db.delete("users", where=col("email").is_null())
```

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **INSERT** | Via write path (e.g. `saveAsTable`) | `Records.from_list(...).insert_into(table)` |
| **UPDATE** | Not in overview/docs | `db.update(table, where=..., set=...)` |
| **DELETE** | Not in overview/docs | `db.delete(table, where=...)` |

---

## 5. Async Support

### SQLFrame

SQLFrame’s README and stable docs do not describe async/await or an async session API.

### Moltres

Moltres supports async end-to-end: `async_connect()`, `AsyncDatabase`, async table and DataFrame operations (e.g. `await df.collect()`, `await db.table(...)`), and async file readers/writers where applicable.

```python
import asyncio
from moltres import async_connect

async def main():
    async with async_connect("sqlite+aiosqlite:///example.db") as db:
        table_handle = await db.table("users")
        df = table_handle.select()
        results = await df.collect()

asyncio.run(main())
```

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **Async API** | Not documented in overview | Full async: `async_connect()`, `AsyncDatabase`, async collect/table ops |
| **Install** | — | e.g. `pip install moltres[async-postgresql]` or `[async-sqlite]` |

---

## 6. SQL Visibility and Tooling

### SQLFrame

- **Generated SQL**: `df.sql()` returns the compiled SQL; `df.sql(optimize=True)` produces more human-readable SQL.
- **Configuration**: Docs describe generated SQL configuration and optional use of OpenAI to enhance or explain SQL.
- **Dialect**: Default is Spark dialect; input/output dialect can be set to match the engine (see [Input and Output Dialect Configuration](https://sqlframe.readthedocs.io/en/stable/configuration/#input-and-output-dialect)).

### Moltres

- **SQL display**: `df.show_sql()`, `df.sql` property.
- **Plan and validation**: `plan_summary()`, `visualize_plan()`, `validate()`, `performance_hints()`.
- **Dialect**: ANSI plus dialect-specific handling via SQLAlchemy (see `src/moltres/engine/dialects.py`).

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **View SQL** | `df.sql()`, `df.sql(optimize=True)` | `df.show_sql()`, `df.sql` |
| **Readable SQL** | `optimize=True`; optional OpenAI | Plan summary and visualization |
| **Validation / hints** | — | `validate()`, `performance_hints()` |
| **Dialect** | Configurable Spark vs engine-native | SQLAlchemy-based dialect layer |

---

## 7. Installation and Optional Dependencies

### SQLFrame

```bash
# Per-engine (required for that backend)
pip install "sqlframe[bigquery]"
pip install "sqlframe[postgres]"
pip install "sqlframe[duckdb]"
pip install "sqlframe[snowflake]"
pip install "sqlframe[spark]"
# Standalone (no DB)
pip install sqlframe
# Conda
conda install -c conda-forge sqlframe
```

### Moltres

```bash
# Core (SQLite, Postgres, MySQL, DuckDB via SQLAlchemy)
pip install moltres

# Optional
pip install moltres[async-postgresql]   # Async PostgreSQL
pip install moltres[pandas,polars]      # Pandas/Polars result formats
pip install moltres[sqlmodel]            # SQLModel/Pydantic
pip install moltres[streamlit]          # Streamlit
```

### Comparison

| Aspect | SQLFrame | Moltres |
|--------|----------|---------|
| **Default install** | Engine-specific or standalone | Single package; core supports multiple DBs via SQLAlchemy |
| **Extras** | One extra per engine (bigquery, postgres, duckdb, etc.) | Optional async, pandas, polars, sqlmodel, streamlit, etc. |
| **Conda** | conda-forge | Not mentioned in README |

---

## 8. Summary Table

| Feature | SQLFrame | Moltres |
|---------|----------|---------|
| **API style** | Drop-in PySpark (`activate()` + pyspark imports) | Native PySpark-style + optional Pandas/Polars |
| **Entry point** | SparkSession or engine-native session | `connect()` / `async_connect()` → Database |
| **First-class backends** | BigQuery, Databricks, DuckDB, Postgres, Snowflake, Spark, Redshift, GizmoSQL | SQLite, PostgreSQL, MySQL, DuckDB (SQLAlchemy) |
| **Standalone SQL** | Yes | No |
| **INSERT** | Via write path | `Records.insert_into()` |
| **UPDATE / DELETE** | Not in public API | `db.update()`, `db.delete()` |
| **Async** | Not documented | Full async support |
| **SQL display** | `df.sql(optimize=True)` | `show_sql()`, `.sql`, `plan_summary()`, `visualize_plan()` |
| **Validation / hints** | — | `validate()`, `performance_hints()` |
| **Install** | Per-engine or standalone | Single package + optional extras |
| **Dialect** | Spark or engine-native (configurable) | ANSI + SQLAlchemy dialects |

---

## References

- **SQLFrame**: [GitHub](https://github.com/eakmanrq/sqlframe), [Documentation](https://sqlframe.readthedocs.io/en/stable/)
- **Moltres**: [GitHub](https://github.com/eddiethedean/moltres), [Documentation](https://moltres.readthedocs.io/en/latest/)
