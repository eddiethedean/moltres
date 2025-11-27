# Moltres

<div align="center">

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/eddiethedean/moltres)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eddiethedean/moltres/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The Missing DataFrame Layer for SQL in Python**

**MOLTRES**: **M**odern **O**perations **L**ayer for **T**ransformations, **R**elational **E**xecution, and **S**QL

[Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

**Moltres** combines a DataFrame API (like Pandas/Polars), SQL pushdown execution (no data loading into memory), and real SQL CRUD operations (INSERT, UPDATE, DELETE) in one unified interface.

Transform millions of rows using familiar DataFrame operations—all executed directly in SQL without materializing data. Update, insert, and delete with column-aware, type-safe operations.

## ✨ Features

- 🚀 **PySpark-Style DataFrame API** - Primary API with familiar operations (select, filter, join, groupBy, etc.) for seamless migration from PySpark
- 🐼 **Optional Pandas-Style Interface** - Comprehensive Pandas-like API with string accessor, query(), dtypes, shape, pivot, sample, concat, and more
- 🦀 **Optional Polars-Style Interface** - Polars LazyFrame-like API with expression-based operations, set operations, file I/O, CTEs, and more
- 🎯 **98% PySpark API Compatibility** - Near-complete compatibility for seamless migration
- 🗄️ **SQL Pushdown Execution** - All operations compile to SQL and run on your database—no data loading into memory
- ✏️ **Real SQL CRUD** - INSERT, UPDATE, DELETE operations with DataFrame-style syntax
- 📊 **Multiple Formats** - Read/write CSV, JSON, JSONL, Parquet, and more
- 🐼 **Pandas & Polars Integration** - Pass pandas/polars DataFrames directly to moltres operations
- 🌊 **Streaming Support** - Handle datasets larger than memory with chunked processing
- ⚡ **Async Support** - Full async/await support for all operations
- 🔒 **Security First** - Built-in SQL injection prevention and validation
- 🎯 **SQLModel & Pydantic Integration** - Attach models to DataFrames for type safety and validation

## 📦 Installation

```bash
pip install moltres

# Optional: For async support
pip install moltres[async-postgresql]  # PostgreSQL
pip install moltres[async-mysql]       # MySQL
pip install moltres[async-sqlite]     # SQLite

# Optional: For pandas/polars result formats
pip install moltres[pandas,polars]

# Optional: For SQLModel/Pydantic integration
pip install moltres[sqlmodel]  # Includes both SQLModel and Pydantic
```

## 🚀 Quick Start

### Basic DataFrame Operations

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

# Execute and get results (SQL is compiled and executed here)
results = df.collect()  # Returns list of dicts by default
```

### Pandas-Style Interface

```python
df = db.table("users").pandas()

# Pandas-style operations
df[['id', 'name']]  # Select columns
df.query('age > 25 and country == "USA"')  # Query with AND/OR
df['name'].str.upper()  # String accessor
df.groupby('country').agg(age='mean')  # GroupBy
```

📚 **[See the Pandas Interface Guide →](https://github.com/eddiethedean/moltres/blob/main/guides/09-pandas-interface.md)**

### Polars-Style Interface

```python
df = db.table("users").polars()

# Polars-style operations
df.select("id", "name", (col("age") * 2).alias("double_age"))
df.filter((col("age") > 25) & (col("country") == "USA"))
df.group_by("country").agg(F.sum(col("age")))
```

📚 **[See the Polars Interface Guide →](https://github.com/eddiethedean/moltres/blob/main/guides/10-polars-interface.md)**

### SQLModel & Pydantic Integration

Attach SQLModel or Pydantic models to DataFrames for type safety and validation:

```python
from sqlmodel import SQLModel, Field
from pydantic import BaseModel

# SQLModel: Database-backed models
class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int = Field(primary_key=True)
    name: str
    email: str

# Pydantic: Validation-only models
class UserData(BaseModel):
    id: int
    name: str
    email: str

# Use with SQLModel
df = db.table(User).select()
results = df.collect()  # Returns list of User instances

# Use with Pydantic
df = db.table("users").select().with_model(UserData)
results = df.collect()  # Returns list of UserData instances with validation
```

📚 **[See the SQLModel & Pydantic Integration Guide →](https://github.com/eddiethedean/moltres/blob/main/guides/12-sqlmodel-integration.md)**

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

📚 **[See CRUD Operations Guide →](https://github.com/eddiethedean/moltres/blob/main/guides/05-common-patterns.md#data-mutations)**

## 📖 Documentation

### Getting Started
- **[Getting Started Guide](https://github.com/eddiethedean/moltres/blob/main/guides/01-getting-started.md)** - Step-by-step introduction
- **[Examples Directory](https://github.com/eddiethedean/moltres/tree/main/examples)** - 19 comprehensive example files
- **[Examples Guide](https://github.com/eddiethedean/moltres/blob/main/docs/EXAMPLES.md)** - Common patterns and use cases

### Interface Guides
- **[Pandas Interface](https://github.com/eddiethedean/moltres/blob/main/guides/09-pandas-interface.md)** - Complete pandas-style API reference
- **[Polars Interface](https://github.com/eddiethedean/moltres/blob/main/guides/10-polars-interface.md)** - Complete Polars-style API reference
- **[PySpark Migration](https://github.com/eddiethedean/moltres/blob/main/guides/03-migrating-from-pyspark.md)** - Migrating from PySpark

### Core Topics
- **[Reading Data](https://github.com/eddiethedean/moltres/blob/main/guides/01-getting-started.md#reading-data)** - Tables, SQL, files
- **[Writing Data](https://github.com/eddiethedean/moltres/blob/main/guides/01-getting-started.md#writing-data)** - Tables, files, formats
- **[Table Management](https://github.com/eddiethedean/moltres/blob/main/guides/01-getting-started.md#table-management)** - Create, drop, constraints
- **[Schema Inspection](https://github.com/eddiethedean/moltres/blob/main/guides/01-getting-started.md#schema-inspection)** - Reflection and inspection
- **[Streaming](https://github.com/eddiethedean/moltres/blob/main/guides/04-performance-optimization.md#streaming)** - Large dataset handling
- **[Async Operations](https://github.com/eddiethedean/moltres/blob/main/guides/07-advanced-topics.md#async-support)** - Async/await support

### Advanced Topics
- **[Performance Optimization](https://github.com/eddiethedean/moltres/blob/main/guides/04-performance-optimization.md)** - Query optimization and best practices
- **[Error Handling](https://github.com/eddiethedean/moltres/blob/main/guides/06-error-handling.md)** - Exception handling and debugging
- **[Best Practices](https://github.com/eddiethedean/moltres/blob/main/guides/08-best-practices.md)** - Production-ready patterns
- **[Advanced Topics](https://github.com/eddiethedean/moltres/blob/main/guides/07-advanced-topics.md)** - Window functions, CTEs, transactions
- **[SQLModel & Pydantic Integration](https://github.com/eddiethedean/moltres/blob/main/guides/12-sqlmodel-integration.md)** - Type-safe models with SQLModel and Pydantic

### Reference
- **[Why Moltres?](https://github.com/eddiethedean/moltres/blob/main/docs/WHY_MOLTRES.md)** - Understanding the gap Moltres fills
- **[Security Guide](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md)** - Security best practices
- **[Troubleshooting](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[API Reference](https://github.com/eddiethedean/moltres/tree/main/docs/api)** - Complete API documentation

## 📚 Examples

Comprehensive examples demonstrating all Moltres features:

- **[01_connecting.py](https://github.com/eddiethedean/moltres/blob/main/examples/01_connecting.py)** - Database connections (sync and async)
- **[02_dataframe_basics.py](https://github.com/eddiethedean/moltres/blob/main/examples/02_dataframe_basics.py)** - Basic DataFrame operations
- **[03_async_dataframe.py](https://github.com/eddiethedean/moltres/blob/main/examples/03_async_dataframe.py)** - Asynchronous operations
- **[04_joins.py](https://github.com/eddiethedean/moltres/blob/main/examples/04_joins.py)** - Join operations
- **[05_groupby.py](https://github.com/eddiethedean/moltres/blob/main/examples/05_groupby.py)** - GroupBy and aggregation
- **[06_expressions.py](https://github.com/eddiethedean/moltres/blob/main/examples/06_expressions.py)** - Column expressions and functions
- **[07_file_reading.py](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)** - Reading files (CSV, JSON, Parquet)
- **[08_file_writing.py](https://github.com/eddiethedean/moltres/blob/main/examples/08_file_writing.py)** - Writing DataFrames to files
- **[09_table_operations.py](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)** - Table operations and mutations
- **[10_create_dataframe.py](https://github.com/eddiethedean/moltres/blob/main/examples/10_create_dataframe.py)** - Creating DataFrames from Python data
- **[11_window_functions.py](https://github.com/eddiethedean/moltres/blob/main/examples/11_window_functions.py)** - Window functions
- **[12_sql_operations.py](https://github.com/eddiethedean/moltres/blob/main/examples/12_sql_operations.py)** - Raw SQL and SQL operations
- **[13_transactions.py](https://github.com/eddiethedean/moltres/blob/main/examples/13_transactions.py)** - Transaction management
- **[14_reflection.py](https://github.com/eddiethedean/moltres/blob/main/examples/14_reflection.py)** - Schema inspection and reflection
- **[15_pandas_polars_dataframes.py](https://github.com/eddiethedean/moltres/blob/main/examples/15_pandas_polars_dataframes.py)** - Pandas/Polars integration
- **[16_ux_features.py](https://github.com/eddiethedean/moltres/blob/main/examples/16_ux_features.py)** - UX improvements
- **[17_sqlalchemy_models.py](https://github.com/eddiethedean/moltres/blob/main/examples/17_sqlalchemy_models.py)** - SQLAlchemy ORM integration
- **[18_pandas_interface.py](https://github.com/eddiethedean/moltres/blob/main/examples/18_pandas_interface.py)** - Pandas-style interface examples
- **[19_polars_interface.py](https://github.com/eddiethedean/moltres/blob/main/examples/19_polars_interface.py)** - Polars-style interface examples
- **[20_sqlalchemy_integration.py](https://github.com/eddiethedean/moltres/blob/main/examples/20_sqlalchemy_integration.py)** - SQLAlchemy integration patterns
- **[21_sqlmodel_integration.py](https://github.com/eddiethedean/moltres/blob/main/examples/21_sqlmodel_integration.py)** - SQLModel and Pydantic integration

See the [examples directory](https://github.com/eddiethedean/moltres/tree/main/examples) for all example files.

## 🛠️ Supported Operations

### DataFrame Operations
- `select()` / `selectExpr()` - Project columns or SQL expressions
- `where()` / `filter()` - Filter rows
- `join()` - Join with other DataFrames
- `group_by()` / `groupBy()` - Group rows
- `agg()` - Aggregate functions
- `order_by()` / `orderBy()` / `sort()` - Sort rows
- `limit()` - Limit number of rows
- `distinct()` - Remove duplicate rows
- `withColumn()` - Add or rename columns
- `pivot()` - Pivot operations
- `explode()` - Explode array/JSON columns

### Column Expressions
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Boolean**: `&`, `|`, `~`
- **Functions**: 130+ functions including mathematical, string, date/time, aggregate, window, array, JSON, and utility functions
- **Window Functions**: `over()`, `partition_by()`, `order_by()` - Full PySpark compatibility

📚 **[See Expressions Guide →](https://github.com/eddiethedean/moltres/blob/main/examples/06_expressions.py)**

### Supported SQL Dialects
- ✅ **SQLite** - Full support
- ✅ **PostgreSQL** - Full support with dialect-specific optimizations
- ✅ **MySQL** - Full support with dialect-specific optimizations
- ✅ **DuckDB** - Full support with PostgreSQL-compatible optimizations
- ✅ **Other SQLAlchemy-supported databases** - ANSI SQL fallback

## 🧪 Development

### Setup

```bash
# Clone the repository
git clone https://github.com/eddiethedean/moltres.git
cd moltres

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n 9

# Run with coverage
pytest --cov=src/moltres --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking (strict mode enabled)
mypy src
```

### Pre-Commit CI Checks

```bash
# Run all CI checks (linting, type checking, tests)
make ci-check

# Quick linting check only
make ci-check-lint
```

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](https://github.com/eddiethedean/moltres/blob/main/CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Before submitting:**
- Run tests: `pytest`
- Check code quality: `ruff check . && mypy src`
- Update documentation if needed

## 👤 Author

**Odos Matthews**

- GitHub: [@eddiethedean](https://github.com/eddiethedean)
- Email: odosmatthews@gmail.com

## 🙏 Acknowledgments

- Inspired by PySpark's DataFrame API style, but focused on SQL feature support rather than PySpark feature parity
- Built on SQLAlchemy for database connectivity and SQL compilation
- Thanks to all contributors and users

## 📄 License

MIT License - see [LICENSE](https://github.com/eddiethedean/moltres/blob/main/LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Python data community**

[⬆ Back to Top](#moltres)

</div>
