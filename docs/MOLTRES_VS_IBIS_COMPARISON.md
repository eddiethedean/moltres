# Moltres vs Ibis Comparison

## Executive Summary

This document compares [Moltres](https://github.com/eddiethedean/moltres) and [Ibis](https://github.com/ibis-project/ibis): two Python libraries that provide a portable DataFrame API with SQL (or backend-native) execution. Ibis is “the portable Python dataframe library” with a single API across many backends; Moltres is a PySpark-style DataFrame layer for SQL with full CRUD and async support. Both compile expressions to SQL (or the backend’s native form) and run on the engine rather than in Python memory.

### Key Findings

- **API style**: Ibis has its own DataFrame API (e.g. `con.table()`, `group_by()`, `agg()`, `order_by()`); Moltres provides a PySpark-style API (`db.table().select()`, `group_by()` / `groupBy()`) and optional Pandas/Polars interfaces.
- **Backends**: Ibis supports about 20 backends (BigQuery, DuckDB, PostgreSQL, Snowflake, PySpark, Polars, DataFusion, Trino, etc.); Moltres is SQLAlchemy-based (SQLite, PostgreSQL, MySQL, DuckDB, and any SQLAlchemy-supported database).
- **CRUD**: Moltres provides INSERT (via `Records`), UPDATE, and DELETE; Ibis focuses on read and transform (no UPDATE/DELETE in its public API).
- **Async**: Moltres has full async/await; Ibis does not emphasize async in its README.
- **SQL mixing**: Ibis allows `t.sql("SELECT ...")` to embed SQL and continue with the DataFrame API; Moltres compiles Python expressions to SQL and offers `show_sql()`, plan summary, and validation.

### Use Case Recommendations

- **Choose Moltres** when:
  - You want a PySpark-style or pandas/polars-style API on SQL databases.
  - You need UPDATE/DELETE or bulk INSERT with a DataFrame-like API.
  - You want full async/await and framework integrations (FastAPI, Django, Streamlit, SQLModel).
  - You use SQLAlchemy-backed databases and prefer a single `pip install moltres` with optional extras.

- **Choose Ibis** when:
  - You need one API across many backends (BigQuery, Snowflake, DuckDB, Polars, PySpark, Trino, etc.).
  - You want to mix raw SQL and DataFrame code (`t.sql("SELECT ...")`) and iterate locally then deploy by changing the backend.
  - You prefer Ibis’s own API and interactive mode (`ibis.options.interactive = True`).
  - You use non-SQL backends (e.g. Polars, DataFusion) or cloud warehouses with first-class Ibis support.

---

## 1. Initialization and Connection

### Ibis

```python
import ibis

# Set default backend (optional)
ibis.set_backend("duckdb")

# Create connection per backend
con = ibis.duckdb.connect()
con = ibis.postgres.connect("postgresql://user:pass@localhost/db")
con = ibis.bigquery.connect(project_id="my-project")
con = ibis.polars.connect()
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

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **Entry point** | Backend-specific connection (e.g. `ibis.duckdb.connect()`) | `Database` from `connect()` or `async_connect()` |
| **Backend switch** | `ibis.set_backend(...)` or different `con` | Different DSN or engine passed to `connect()` |
| **Async support** | Not highlighted in overview | Full async via `async_connect()` and `AsyncDatabase` |

---

## 2. DataFrame API Style

### Ibis

Ibis uses its own API: snake_case methods, keyword aggregations, and a portable expression model. Interactive mode prints table previews.

```python
import ibis
ibis.options.interactive = True
con = ibis.duckdb.connect()
t = con.table("penguins")  # or con.read_csv(...), ibis.examples.penguins.fetch()
g = t.group_by("species", "island").agg(count=t.count()).order_by("count")
# g is displayed automatically in interactive mode
```

### Moltres

Moltres provides a PySpark-style API (camelCase or snake_case) and optional Pandas/Polars interfaces. Execution is explicit (e.g. `.collect()`).

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

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **API lineage** | Ibis-native (portable across backends) | PySpark-style; optional Pandas/Polars |
| **Naming** | snake_case (e.g. `group_by`, `order_by`) | Both camelCase and snake_case |
| **Execution** | Lazy; execute when needed (e.g. `execute()`, or interactive display) | Lazy; `.collect()` or similar to run |
| **Interactive display** | `ibis.options.interactive = True` | No built-in interactive mode; use `.collect()` or pandas/polars result |

---

## 3. Supported Backends and Execution

### Ibis

Ibis supports about 20 backends, including SQL and DataFrame engines:

- **SQL**: BigQuery, ClickHouse, DuckDB, Exasol, MySQL, Oracle, PostgreSQL, RisingWave, SingleStoreDB, SQL Server, SQLite, Snowflake, Trino, Apache Druid, Flink, Impala.
- **Other**: Apache DataFusion, Apache PySpark, Polars, Theseus.

Install per backend, e.g. `pip install 'ibis-framework[duckdb,examples]'`. Same API across backends; expressions compile to SQL or the backend’s native form.

### Moltres

Moltres is SQLAlchemy-based. Documented first-class: SQLite, PostgreSQL, MySQL, DuckDB. Any SQLAlchemy-supported database works (e.g. BigQuery/Snowflake via community drivers). Single `pip install moltres`; optional extras for async drivers, Pandas, Polars, integrations.

### Comparison

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **Scope** | ~20 backends (SQL + DataFusion, Polars, PySpark) | SQLAlchemy-based (any compatible DB) |
| **First-class SQL** | BigQuery, Postgres, Snowflake, DuckDB, Trino, etc. | SQLite, PostgreSQL, MySQL, DuckDB |
| **Non-SQL backends** | Yes (Polars, DataFusion, PySpark) | No (SQL only) |
| **Portability** | “One API, many backends”; switch backend with one line | One stack (SQL); switch DB via DSN or engine |

---

## 4. SQL and Expression Compilation

### Ibis

Ibis compiles expressions to SQL (for SQL backends) or to the backend’s native API. You can inspect SQL and mix raw SQL with the DataFrame API.

```python
# Compile expression to SQL string
ibis.to_sql(g)

# Mix SQL and Python: run SQL, get back an Ibis table expression
a = t.sql("SELECT species, island, count(*) AS count FROM penguins GROUP BY 1, 2")
b = a.order_by("count")
```

### Moltres

Moltres compiles logical plans to SQL via its SQL compiler. No embedded raw SQL strings; you build expressions in Python and can inspect or validate the generated SQL.

```python
df.show_sql()           # Print compiled SQL
sql_string = df.sql      # Property with SQL text
df.plan_summary()       # Query plan summary
df.validate()           # Validate before execution
df.performance_hints()  # Optimization hints
```

### Comparison

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **View SQL** | `ibis.to_sql(expr)` | `df.show_sql()`, `df.sql` |
| **Mix raw SQL** | `t.sql("SELECT ...")` returns expression | No; build with expressions only |
| **Plan / validation** | Backend-dependent | `plan_summary()`, `visualize_plan()`, `validate()`, `performance_hints()` |

---

## 5. CRUD and Mutations

### Ibis

Ibis is oriented around read and transform: tables, filters, joins, aggregations, and writing results (e.g. to tables or files). Its README and overview do not describe UPDATE or DELETE on existing rows.

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

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **INSERT** | Via write path (e.g. to table or file) | `Records.from_list(...).insert_into(table)` |
| **UPDATE** | Not in overview/docs | `db.update(table, where=..., set=...)` |
| **DELETE** | Not in overview/docs | `db.delete(table, where=...)` |

---

## 6. Async Support

### Ibis

Ibis’s README and main docs do not emphasize an async API for connections or execution.

### Moltres

Moltres supports async end-to-end: `async_connect()`, `AsyncDatabase`, and async table/DataFrame operations (e.g. `await df.collect()`, `await db.table(...)`).

### Comparison

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **Async API** | Not documented in overview | Full async: `async_connect()`, `AsyncDatabase`, async collect/table ops |

---

## 7. Installation and Optional Dependencies

### Ibis

```bash
# One backend + examples
pip install 'ibis-framework[duckdb,examples]'

# Other backends (examples from docs)
pip install 'ibis-framework[postgres]'
pip install 'ibis-framework[bigquery]'
pip install 'ibis-framework[snowflake]'
pip install 'ibis-framework[polars]'
# See https://ibis-project.org/install for full list
```

### Moltres

```bash
pip install moltres
pip install moltres[async-postgresql]   # Async PostgreSQL
pip install moltres[pandas,polars]     # Pandas/Polars result formats
pip install moltres[sqlmodel]           # SQLModel/Pydantic
pip install moltres[streamlit]          # Streamlit
```

### Comparison

| Aspect | Ibis | Moltres |
|--------|------|---------|
| **Default install** | Backend-specific extras (e.g. `[duckdb,examples]`) | Single package; core supports multiple DBs via SQLAlchemy |
| **Extras** | Per-backend (duckdb, postgres, bigquery, polars, etc.) | Optional async, pandas, polars, sqlmodel, streamlit, etc. |

---

## 8. Summary Table

| Feature | Ibis | Moltres |
|---------|------|---------|
| **API style** | Ibis-native portable API | PySpark-style + optional Pandas/Polars |
| **Entry point** | Backend connection (e.g. `ibis.duckdb.connect()`) | `connect()` / `async_connect()` → Database |
| **Backends** | ~20 (BigQuery, Snowflake, DuckDB, Polars, PySpark, Trino, etc.) | SQLAlchemy (SQLite, Postgres, MySQL, DuckDB, others) |
| **Non-SQL backends** | Yes (Polars, DataFusion, PySpark) | No |
| **Mix raw SQL** | `t.sql("SELECT ...")` | No; expression-only |
| **INSERT / UPDATE / DELETE** | Write path; no UPDATE/DELETE in overview | Full CRUD: Records, `db.update()`, `db.delete()` |
| **Async** | Not highlighted | Full async support |
| **SQL display** | `ibis.to_sql(expr)` | `show_sql()`, `.sql`, `plan_summary()`, `validate()` |
| **Interactive display** | `ibis.options.interactive = True` | No; use `.collect()` or result format |
| **License** | Apache-2.0 | MIT |

---

## References

- **Ibis**: [GitHub](https://github.com/ibis-project/ibis), [Documentation](https://ibis-project.org)
- **Moltres**: [GitHub](https://github.com/eddiethedean/moltres), [Documentation](https://moltres.readthedocs.io/en/latest/)
