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

**Moltres** combines a DataFrame API (like Pandas/Polars), SQL pushdown execution (no data loading into memory), and real SQL CRUD operations (INSERT, UPDATE, DELETE) in one unified interface.

Transform millions of rows using familiar DataFrame operations—all executed directly in SQL without materializing data.

## ✨ Key Features

- 🚀 **PySpark-Style DataFrame API** - Primary API with 98% PySpark compatibility
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

**1.0.0** ships this split on PyPI: `pip install moltres` pulls in **`moltres-core`** automatically. For breaking changes and upgrade notes, see [CHANGELOG.md](CHANGELOG.md).

## 🚀 Quick Start

```python
from moltres import col, connect
from moltres.expressions import functions as F

# Connect to your database
db = connect("sqlite:///example.db")

# DataFrame operations with SQL pushdown (no data loading into memory)
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[col("orders.customer_id") == col("customers.id")])
    .where(col("active") == True)
    .group_by("country")
    .agg(F.sum(col("amount")).alias("total_amount"))
)

# Execute and get results
results = df.collect()  # Returns list of dicts by default
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

## 📖 Documentation

- **[Getting Started Guide](https://moltres.readthedocs.io/en/latest/guides/getting-started.html)** - Step-by-step introduction
- **[Examples Directory](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)** - 29 comprehensive examples
- **[User Guides](https://moltres.readthedocs.io/en/latest/#guides-how-to)** - Complete guides for all features
- **[API Reference](https://moltres.readthedocs.io/en/latest/api/dataframe.html)** - Complete API documentation

### Framework Integrations

- **[FastAPI Integration](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)** - Error handling, dependency injection
- **[Django Integration](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)** - Middleware, template tags, management commands
- **[Streamlit Integration](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)** - Components, caching, query visualization
- **[SQLModel & Pydantic](https://moltres.readthedocs.io/en/latest/guides/sqlmodel-integration.html)** - Type-safe models

## 🛠️ Supported Operations

**DataFrame Operations**: `select()`, `where()`, `join()`, `group_by()`, `agg()`, `order_by()`, `limit()`, `distinct()`, `pivot()`, and more

**130+ Functions**: Mathematical, string, date/time, aggregate, window, array, JSON, and utility functions

**SQL Dialects**: SQLite, PostgreSQL, MySQL, DuckDB, and any SQLAlchemy-supported database

**UX Features**: Enhanced SQL display (`show_sql()`, `sql` property), query plan visualization (`plan_summary()`, `visualize_plan()`), schema discovery (`db.schema()`, `db.tables()`), query validation (`validate()`), performance hints (`performance_hints()`), and interactive help (`help()`, `suggest_next()`)

## 🧪 Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Code quality
ruff check . && ruff format . && mypy src
```

## 🤝 Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](https://moltres.readthedocs.io/en/latest/CONTRIBUTING.html) for guidelines.

## 📄 License

MIT License - see [LICENSE](https://github.com/eddiethedean/moltres/blob/main/LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Python data community**

[⬆ Back to Top](#moltres)

</div>
