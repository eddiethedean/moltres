# Moltres

<div align="center">

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://github.com/eddiethedean/moltres)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eddiethedean/moltres/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/moltres/badge/?version=latest)](https://moltres.readthedocs.io/en/latest/?badge=latest)

**The Missing DataFrame Layer for SQL in Python**

**MOLTRES**: **M**odern **O**perations **L**ayer for **T**ransformations, **R**elational **E**xecution, and **S**QL

</div>

---

**Moltres** combines a DataFrame API (like Pandas/Polars), SQL pushdown execution (no data loading into memory), and real SQL CRUD operations (INSERT, UPDATE, DELETE) in one unified interface. See [why Moltres](docs/WHY_MOLTRES.md) and the [comparison guides](https://moltres.readthedocs.io/en/latest/#comparisons) for how it differs from Pandas, Ibis, and PySpark.

Transform millions of rows using familiar DataFrame operations—all executed directly in SQL without materializing data.

## ✨ Key Features

- 🚀 **PySpark-Style DataFrame API** - Primary API; see [PySpark compatibility notes](docs/PYSPARK_MIGRATION_INCONSISTENCIES.md) for details
- 🗄️ **SQL Pushdown Execution** - All operations compile to SQL and run on your database
- ✏️ **Real SQL CRUD** - INSERT, UPDATE, DELETE with DataFrame-style syntax
- 🐼 **Pandas & Polars Interfaces** - Optional pandas/polars-style APIs
- ⚡ **Async Support** - Full async/await support for all operations
- 🔒 **Security First** - Built-in SQL injection prevention
- 🎯 **Framework Integrations** - FastAPI, Django, Streamlit, SQLModel, Pydantic

## 📦 Installation

```bash
pip install moltres

# Optional extras
pip install moltres[async-postgresql]  # Async PostgreSQL
pip install moltres[pandas,polars]     # Pandas/Polars result formats
pip install moltres[sqlmodel]          # SQLModel/Pydantic integration
pip install moltres[streamlit]        # Streamlit integration
pip install moltres[parquet]          # Parquet file I/O (pyarrow)
pip install moltres[fastapi]          # FastAPI integration helpers
pip install moltres[duckdb]            # DuckDB SQLAlchemy dialect
```

### `moltres-core` and pydantable

SQL execution lives in the companion **`moltres-core`** package. You can use
`MoltresPydantableEngine` with [pydantable](https://pypi.org/project/pydantable/) for a
typed, plan-driven API backed by SQL for supported operations. See
[`docs/PYDANTABLE_ENGINE.md`](docs/PYDANTABLE_ENGINE.md). From source, install
`moltres-core` **before** `moltres`:

```bash
pip install -e ./moltres-core
pip install -e .
```

**1.1.0** ships this split on PyPI: `pip install moltres` pulls in **`moltres-core`** automatically. For breaking changes and upgrade notes, see [CHANGELOG.md](CHANGELOG.md).

### Prerequisites

- **Python 3.10+** (see [Runtime support](docs/RUNTIME_SUPPORT.md))
- **SQLAlchemy 2.0+** (installed automatically with `moltres`)
- **Database driver** for your backend (e.g. `psycopg2-binary` for PostgreSQL, `pymysql` for MySQL; SQLite needs no extra driver)
- **Optional extras**: full list in [Public API — Optional extras](docs/PUBLIC_API.md#optional-extras)

## 🚀 Quick Start

**New here?** Follow the [5-minute getting started guide](https://moltres.readthedocs.io/en/latest/guides/getting-started.html) first.

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.table.schema import column

with connect("sqlite:///:memory:") as db:
    db.create_table("orders", [
        column("id", "INTEGER"),
        column("country", "TEXT"),
        column("amount", "REAL"),
    ]).collect()
    Records.from_list([
        {"id": 1, "country": "US", "amount": 100.0},
        {"id": 2, "country": "UK", "amount": 200.0},
    ], database=db).insert_into("orders")

    df = (
        db.table("orders").select()
        .where(col("country") == "US")
        .group_by("country")
        .agg(F.sum(col("amount")).alias("total_amount"))
    )
    print(df.collect())  # [{'country': 'US', 'total_amount': 100.0}]
```

### CRUD Operations

```python
from moltres.io.records import Records

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

### Reading Data: Tables vs Files

| Goal | API | Returns |
|------|-----|---------|
| Query a SQL table lazily | `db.table("orders").select()` | `DataFrame` (SQL pushdown) |
| Load a file as a lazy DataFrame | `db.load.csv("data.csv")` | `DataFrame` (materialized via temp table) |
| Load a file as in-memory rows | `db.read.records.csv("data.csv")` | `Records` (eager, for inserts) |

See [Public API guide](docs/PUBLIC_API.md) for stable import paths.

## 📖 Documentation

- **[Roadmap](ROADMAP.md)** - Future 1.x release phases and competitive priorities
- **[Public API](https://moltres.readthedocs.io/en/latest/PUBLIC_API.html)** - Stable imports and I/O patterns

- **[Getting Started Guide](https://moltres.readthedocs.io/en/latest/guides/getting-started.html)** - Step-by-step introduction (start here)
- **[Examples](https://moltres.readthedocs.io/en/latest/EXAMPLE_SCRIPTS.html)** - Runnable example scripts
- **[User Guides](https://moltres.readthedocs.io/en/latest/#guides-how-to)** - Complete guides for all features
- **[API Reference](https://moltres.readthedocs.io/en/latest/api/dataframe.html)** - Complete API documentation

### Framework Integrations

- **[FastAPI Integration](https://moltres.readthedocs.io/en/latest/EXAMPLE_SCRIPTS.html)** - See `docs/examples/22_fastapi_integration.py`
- **[Django Integration](https://moltres.readthedocs.io/en/latest/guides/django-integration.html)**
- **[Streamlit Integration](https://moltres.readthedocs.io/en/latest/guides/streamlit-integration.html)**
- **[SQLModel & Pydantic](https://moltres.readthedocs.io/en/latest/guides/sqlmodel-integration.html)** - Type-safe models

## 🛠️ Supported Operations

**DataFrame Operations**: `select()`, `where()`, `join()`, `group_by()`, `agg()`, `order_by()`, `limit()`, `distinct()`, `pivot()`, and more

**130+ Functions**: Mathematical, string, date/time, aggregate, window, array, JSON, and utility functions

**SQL Dialects**: SQLite, PostgreSQL, MySQL, and DuckDB are CI-tested; other SQLAlchemy-supported databases are best-effort (see [Runtime support](docs/RUNTIME_SUPPORT.md))

**UX Features**: Enhanced SQL display (`show_sql()`, `sql` property), query plan visualization (`plan_summary()`, `visualize_plan()`), schema discovery (`db.schema()`, `db.tables()`), query validation (`validate()`), performance hints (`performance_hints()`), and interactive help (`help()`, `suggest_next()`)

## 🧪 Development

From a git checkout, install **`moltres-core` before `moltres`** (the monorepo ships two packages):

```bash
pip install -e ./moltres-core
pip install -e ".[dev]"

# Run lint/type/doc-example checks (does NOT run the test suite)
make ci-check

# Run tests (matches CI main matrix)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_asyncio.plugin -p xdist.plugin \
  -m "not postgres and not mysql and not multidb and not tier2_integration and not tier3_integration" \
  -n auto --dist loadgroup
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - see [LICENSE](https://github.com/eddiethedean/moltres/blob/main/LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Python data community**

[⬆ Back to Top](#moltres)

</div>
