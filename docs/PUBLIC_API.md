# Public API

This page documents the **stable import surface** for Moltres users. It is the
user-facing companion to the maintainer policy in [RELEASE_PROCESS.md](RELEASE_PROCESS.md).

Contract tests in `tests/api/test_public_imports.py` enforce the exports listed here.

## Top-level imports (`moltres`)

```python
from moltres import (
    connect,
    async_connect,
    col,
    lit,
    column,
    MoltresConfig,
    Database,
    AsyncDatabase,
    register_performance_hook,
    unregister_performance_hook,
    __version__,
)
```

### `col` vs `column`

| Symbol | Purpose |
|--------|---------|
| `col("name")` | Column expression for queries (`where`, `select`, etc.) |
| `column("name", "TEXT")` | DDL helper for `db.create_table()` schemas |

### Optional interface wrappers

`PandasDataFrame`, `PolarsDataFrame`, `AsyncPandasDataFrame`, `AsyncPolarsDataFrame`,
and `AsyncDatabase` are loaded lazily. Accessing them without the required extra
raises `ImportError` with an install hint (they are never `None`).

```python
# pip install moltres[pandas]
from moltres import PandasDataFrame
```

## DataFrame (`moltres.dataframe`)

```python
from moltres.dataframe import (
    DataFrame,
    AsyncDataFrame,
    DataLoader,
    DataFrameWriter,
    ReadAccessor,
)
```

Use the PySpark-style API on `DataFrame` returned from `db.table(...).select()` or
`db.load.*` file readers.

### Union semantics (PySpark difference)

| Moltres | PySpark equivalent |
|---------|-------------------|
| `df.union(other)` | `union().distinct()` — **distinct** rows |
| `df.unionAll(other)` | `union()` — **all** rows |

When migrating PySpark `df1.union(df2)`, use **`df1.unionAll(df2)`** in Moltres.

`df.show()` prints rows only by default. Pass `count_total=True` to include a full-table
row count (this runs an extra `COUNT(*)` query).

## Expressions (`moltres.expressions`)

```python
from moltres.expressions import col, lit, Column, sum, avg, when
from moltres.expressions import functions as F
```

## Engine (`moltres.engine`)

Lower-level connection and execution helpers:

```python
from moltres.engine import (
    ConnectionManager,
    QueryExecutor,
    QueryResult,
    DialectSpec,
    get_dialect,
    register_performance_hook,
    unregister_performance_hook,
)
```

## Records and CRUD (`moltres.io.records`)

```python
from moltres.io.records import Records, AsyncRecords, LazyRecords, AsyncLazyRecords
```

Prefer factories and public keyword arguments:

```python
Records.from_list([{"id": 1}], database=db).insert_into("users")
# or
Records(data=[{"id": 1}], database=db).insert_into("users")
```

`Records(_data=...)` and `Records(_database=...)` are **deprecated** (removed in 2.0).

### Database CRUD (sync and async)

Sync:

```python
db.insert("users", rows)
db.update("users", where=col("id") == 1, set={"name": "Alice"})
db.delete("users", where=col("id") == 1)
db.merge("users", rows, on=["id"], when_matched={"name": "Bob"})
```

Async (`AsyncDatabase` — same method signatures):

```python
async def insert_users(db, rows):
    return await db.insert("users", rows)
```

## Reading data: choose the right API

| Goal | API | Returns |
|------|-----|---------|
| Query a SQL table | `db.table("t").select().where(...)` | `DataFrame` |
| Load a file for querying | `db.load.csv("data.csv")` | `DataFrame` |
| Load a file as row dicts | `db.read.records.csv("data.csv")` | `Records` / `LazyRecords` |
| Polars-style file scan | `db.scan_csv("data.csv")` | `PolarsDataFrame` (optional) |

**Canonical paths (1.2+)**

- **Lazy DataFrame from files:** `db.load.csv()` — preferred
- **Eager rows from files:** `db.read.records.csv()`
- **`db.read.csv()` etc.:** deprecated for DataFrame reads; use `db.load.*` instead

## Optional extras

| Extra | Install | Enables |
|-------|---------|---------|
| `pandas` | `pip install moltres[pandas]` | `PandasDataFrame`, pandas result formats |
| `polars` | `pip install moltres[polars]` | `PolarsDataFrame`, polars result formats |
| `parquet` | `pip install moltres[parquet]` | Parquet file I/O via pyarrow |
| `fastapi` | `pip install moltres[fastapi]` | FastAPI integration helpers |
| `duckdb` | `pip install moltres[duckdb]` | DuckDB SQLAlchemy dialect |
| `async` / `async-sqlite` | `pip install moltres[async-sqlite]` | `AsyncDatabase`, async file I/O |
| `sqlmodel` | `pip install moltres[sqlmodel]` | SQLModel / Pydantic model integration |
| `streamlit` | `pip install moltres[streamlit]` | Streamlit components |

## PySpark compatibility

Moltres targets a PySpark-like DataFrame API. Coverage varies by operation and
dialect. See:

- [PySpark migration inconsistencies](PYSPARK_MIGRATION_INCONSISTENCIES.md)
- [Moltres vs PySpark comparison](MOLTRES_VS_PYSPARK_COMPARISON.md)
- [PySpark feature comparison](PYSPARK_FEATURE_COMPARISON.md)

## What is not public API

Modules under `moltres.sql`, `moltres.logical`, `moltres.dataframe.managers`, and
empty `__init__.py` packages are internal implementation details.
