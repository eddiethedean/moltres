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

Optional interface wrappers (`PandasDataFrame`, `PolarsDataFrame`, and their async
variants) are exported when the corresponding extras are installed; otherwise they
are `None`. Install `moltres[pandas]` or `moltres[polars]` before using them.

## DataFrame (`moltres.dataframe`)

```python
from moltres.dataframe import DataFrame, AsyncDataFrame, DataLoader, DataFrameWriter
```

Use the PySpark-style API on `DataFrame` returned from `db.table(...).select()` or
`db.load.*` file readers.

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

Performance hooks are also available from the top-level `moltres` package.

## Records and CRUD (`moltres.io.records`)

```python
from moltres.io.records import Records
```

`Records` holds eager row dicts for inserts and file materialization workflows.

## Reading data: choose the right API

| Goal | API | Returns | When to use |
|------|-----|---------|-------------|
| Query a SQL table | `db.table("t").select().where(...)` | `DataFrame` | Lazy SQL pushdown on existing tables |
| Load a file for querying | `db.load.csv("data.csv")` | `DataFrame` | Lazy scan; file staged via temp table |
| Load a file as row dicts | `db.read.records.csv("data.csv")` | `Records` | Eager rows for `insert_into()` or Python logic |
| Insert / update / delete | `Records(...).insert_into("t")`, `db.update(...)`, `db.delete(...)` | — | CRUD on SQL tables |

**Mental model**

- **`table().select()`** — SQL tables, lazy `DataFrame`, operations compile to SQL.
- **`load.*`** — files → lazy `DataFrame` (query with `.where()`, `.join()`, etc.).
- **`read.records.*`** — files → eager `Records` (row materialization, inserts).
- **`Records`** — in-memory rows for CRUD helpers.

## Optional extras

| Extra | Install | Enables |
|-------|---------|---------|
| `pandas` | `pip install moltres[pandas]` | `PandasDataFrame`, pandas result formats |
| `polars` | `pip install moltres[polars]` | `PolarsDataFrame`, polars result formats |
| `parquet` | `pip install moltres[parquet]` | Parquet file I/O via pyarrow |
| `fastapi` | `pip install moltres[fastapi]` | FastAPI integration helpers |
| `duckdb` | `pip install moltres[duckdb]` | DuckDB SQLAlchemy dialect |
| `async-postgresql` | `pip install moltres[async-postgresql]` | Async PostgreSQL driver |
| `sqlmodel` | `pip install moltres[sqlmodel]` | SQLModel / Pydantic model integration |
| `streamlit` | `pip install moltres[streamlit]` | Streamlit components |

## PySpark compatibility

Moltres targets a PySpark-like DataFrame API. Coverage varies by operation and
dialect. See:

- [PySpark migration inconsistencies](PYSPARK_MIGRATION_INCONSISTENCIES.md)
- [Moltres vs PySpark comparison](MOLTRES_VS_PYSPARK_COMPARISON.md)
- [PySpark feature comparison](PYSPARK_FEATURE_COMPARISON.md)

## What is not public API

Modules under `moltres.sql`, `moltres.logical`, and empty `__init__.py` packages
are internal implementation details. Import from the paths above unless you are
contributing to Moltres itself.
