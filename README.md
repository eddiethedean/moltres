# Moltres

<div align="center">

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/eddiethedean/moltres)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eddiethedean/moltres/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The Missing DataFrame Layer for SQL in Python**

**MOLTRES**: **M**odern **O**perations **L**ayer for **T**ransformations, **R**elational **E**xecution, and **S**QL

[Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Why Moltres?](#-why-moltres)

</div>

---

**Moltres** fills a major gap in the Python data ecosystem: it's the **only** library that combines a **DataFrame API** (like Pandas/Polars), **SQL pushdown execution** (no data loading into memory), and **real SQL CRUD operations** (INSERT, UPDATE, DELETE) in one unified interface.

Transform millions of rows using familiar DataFrame operations—all executed directly in SQL without materializing data. Update, insert, and delete with column-aware, type-safe operations. No juggling between Pandas, SQLAlchemy, and raw SQL. Just one library that does it all.

> **🎉 Version 0.11.0 Milestone:** Async PostgreSQL pipelines are now production-ready out of the box. DSN `?options=-csearch_path=...` flags automatically propagate through asyncpg pool connections, and staging tables created by `createDataFrame()` or file readers remain accessible across pooled sessions—no more `UndefinedTableError` surprises. Previous 0.10 improvements such as chunked file reading and near-total PySpark API compatibility are, of course, still included.

## 📑 Table of Contents

- [Features](#-features)
- [What Makes Moltres Unique](#-what-makes-moltres-unique)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Why Moltres?](#-why-moltres)
- [Core Concepts](#-core-concepts)
- [Reading Data](#-reading-data)
- [Writing Data](#-writing-data)
- [Streaming for Large Datasets](#-streaming-for-large-datasets)
- [Table Management](#️-table-management)
- [Data Mutations](#️-data-mutations)
- [Result Formats](#-result-formats)
- [Configuration](#️-configuration)
- [Performance Monitoring](#-performance-monitoring)
- [Security](#-security)
- [Advanced Examples](#-advanced-examples)
- [Supported Operations](#️-supported-operations)
- [Development](#-development)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

## ✨ Features

### Core Capabilities
- 🚀 **DataFrame API** - Familiar operations (select, filter, join, groupBy, etc.) like Pandas/Polars/PySpark
- 🎯 **98% PySpark API Compatibility** - Near-complete compatibility with PySpark's DataFrame API for seamless migration
- 🗄️ **SQL Pushdown Execution** - All operations compile to SQL and run on your database—no data loading into memory
- 📊 **Operates Directly on SQL Tables** - Transform tables without materialization

### SQL & Query Features
- 🔤 **Raw SQL Support** - Execute raw SQL queries with `db.sql()` and get back lazy DataFrames (similar to PySpark's `spark.sql()`)
- 📝 **SQL Expression Selection** - Write SQL expressions directly with `selectExpr()` (similar to PySpark's `selectExpr()`)
- ✏️ **Real SQL CRUD** - INSERT, UPDATE, DELETE operations with DataFrame-style syntax
- 🔄 **Advanced Operations** - Pivot, window functions, semi-joins, anti-joins, and more

### Data I/O
- 📊 **Multiple Formats** - Read/write CSV, JSON, JSONL, Parquet, and more
- 🌊 **Streaming Support** - Handle datasets larger than memory with chunked processing
- 📥 **Flexible Reading** - Read from files, databases, or raw SQL queries

### Developer Experience
- 🔧 **Type Safe** - Full type hints with strict mypy checking and custom type stubs for dependencies
- 🎯 **Zero Dependencies** - Works with just SQLAlchemy (pandas/polars optional)
- ⚡ **Async Support** - Full async/await support for all operations (optional dependency)
- 🔒 **Security First** - Built-in SQL injection prevention and validation
- ⚡ **Performance Monitoring** - Optional hooks for query performance tracking
- 🌍 **Environment Config** - Configure via environment variables for 12-factor apps

## 🔥 What Makes Moltres Unique

Moltres is the **only** Python library that provides:

| Feature | Pandas/Polars | Ibis | SQLAlchemy | SQLModel | **Moltres** |
|--------|----------------|------|-------------|-----------|-------------|
| DataFrame API | ✔ | ✔ | ❌ | ❌ | **✔** |
| SQL Pushdown Execution | ❌ | ✔ | ✔ | ✔ | **✔** |
| **Row-Level INSERT/UPDATE/DELETE** | ❌ | ❌ | ✔ | ✔ | **✔** |
| Lazy query building | ✔ (Polars) | ✔ | ⚠️ | ⚠️ | **✔** |
| Operates directly on SQL tables | ⚠️ limited | ✔ | ✔ | ✔ | **✔** |
| Column-oriented transformations | ✔ | ✔ | ❌ | ❌ | **✔** |

**The combination of DataFrame API + SQL pushdown + CRUD does not exist anywhere else in Python.**

### Key Differentiators

- **Only library with DataFrame API + SQL pushdown + CRUD** - No other Python library offers this combination
- **No data loading into memory for transformations** - All DataFrame operations execute directly in SQL
- **Works with existing SQL infrastructure** - No cluster required, works with SQLite, PostgreSQL, MySQL, and more
- **Type-safe CRUD operations** - Validated, column-aware INSERT, UPDATE, DELETE with DataFrame-style syntax
- **SQL-first design** - Focuses on providing full SQL feature support through a DataFrame API, not replicating every PySpark feature. Features are included only if they map to SQL/SQLAlchemy capabilities and align with SQL pushdown execution.

## 🆕 What's New

### Version 0.11.0

- **Async PostgreSQL Reliability** - DSN `?options=-c...` (e.g., `-csearch_path=my_schema`) are converted into asyncpg `server_settings`, so every pooled connection honors session-level settings without custom hooks.
- **Pooled Async Staging Tables** - Async `createDataFrame()` and file readers now create regular staging tables on PostgreSQL instead of connection-scoped temp tables, ensuring inserts/reads succeed even when operations hop between pooled connections (e.g., pytest-xdist workers).
- **Developer Experience** - Documentation, changelog, and todo tracking updated to reflect the new behavior, making it easier to configure async deployments confidently.

### Version 0.10.0

- **Chunked File Reading** - Files are now read in chunks by default to safely handle files larger than available memory, similar to PySpark's partition-based approach. This prevents out-of-memory errors when processing large datasets. Opt-out available via `stream=False` option.
- **Enhanced Large File Support** - All file reads (`db.read.csv()`, `db.read.json()`, etc.) now use streaming/chunked reading by default, with automatic schema inference from the first chunk and incremental insertion into temporary tables.
- **Improved Memory Safety** - Temporary tables are automatically cleaned up on errors, and empty files are handled gracefully with or without explicit schemas.

### Version 0.9.0

### 🎉 Major Milestone: 98% PySpark API Compatibility!

Moltres now achieves **~98% API compatibility** with PySpark for core DataFrame operations, making migration from PySpark seamless while maintaining SQL-first design principles.

### 🚀 New Features

#### Raw SQL & Expression Support
```python
# Raw SQL queries (PySpark-style)
df = db.sql("SELECT * FROM users WHERE id = :id", id=1)
df = db.sql("SELECT * FROM orders").where(col("amount") > 100)

# SQL expression selection
df.selectExpr("amount * 1.1 as with_tax", "UPPER(name) as name_upper")
df.selectExpr("(amount + tax) * 1.1 as total")
```

#### Enhanced DataFrame API
```python
# Select all columns
df.select("*")  # Explicitly select all
df.select("*", col("new_col"))  # All columns plus new ones

# SQL string predicates
df.filter("age > 18")
df.where("amount >= 100 AND status = 'active'")

# String and dictionary aggregations
df.group_by("category").agg("amount")  # String (defaults to sum)
df.group_by("category").agg({"amount": "sum", "price": "avg"})  # Dict

# Pivot on GroupBy (PySpark-style)
df.group_by("category").pivot("status").agg("amount")  # Auto-infers values

# Explode function
df.select(explode(col("array_col")).alias("value"))

# PySpark-style aliases
df.orderBy("name")  # Alias for order_by()
df.sort("name")  # Alias for order_by()
df.write.saveAsTable("table")  # Alias for save_as_table()

# Improved withColumn
df.withColumn("new_col", col("amount") * 1.1)  # Add or replace
```

**All major DataFrame methods now match PySpark's API!**

### Version 0.8.0

- **PySpark-style `db.read.table()` API** - New `db.read.table()` and `db.read.*` methods that match PySpark's `spark.read.table()` pattern. Read database tables and files with a familiar API: `df = db.read.table("customers")` or `df = db.read.csv("data.csv")`. Returns lazy DataFrame objects for consistency with PySpark.
- **LazyRecords for `db.read.records.*`** - The `db.read.records.*` methods now return lazy `LazyRecords`/`AsyncLazyRecords` objects that materialize on-demand when used. Records automatically materialize when you use Sequence operations (`len()`, `[]`, iteration), call `insert_into()`, or use them with `createDataFrame()`. Explicitly materialize with `.collect()` when needed. This provides better performance by deferring file reads until necessary.
- **Lazy CRUD and DDL Operations** - All DataFrame CRUD and DDL operations are now lazy, requiring an explicit `.collect()` call for execution. This improves composability and aligns with PySpark's lazy evaluation model.
- **Transaction Management** - All operations within a single `.collect()` call are part of a single session that rolls back all changes if any failure occurs.
- **Batch Operation API** - New `db.batch()` context manager to queue multiple lazy operations and execute them together within a single transaction.
- **Type Checking Improvements** - Added pandas-stubs for proper mypy type checking and fixed type compatibility issues.

### Version 0.7.0

- **PostgreSQL and MySQL Testing Infrastructure** - Comprehensive test support for multiple database backends with ephemeral database instances, database fixtures, and extensive test coverage
- **Enhanced Type Safety** - Type overloads for `collect()` methods with improved type inference and better IDE support
- **Improved Code Quality** - Fixed all mypy type checking errors and ruff linting issues, with comprehensive type annotations throughout
- **Database Connection Management** - Added `close()` methods to `Database` and `AsyncDatabase` classes for proper resource cleanup
- **Cross-Database Compatibility** - Fixed PostgreSQL and MySQL-specific issues with JSON extraction, array functions, and async DSN parsing
- **Test Coverage** - 301 passing tests across SQLite, PostgreSQL, and MySQL with async support

### Version 0.6.0

- **Null Handling Convenience Methods** - New `na` property on DataFrame: `df.na.drop()` and `df.na.fill(value)` for convenient null handling
- **Random Sampling** - New `sample(fraction, seed=None)` method for random row sampling with dialect-specific SQL compilation
- **Enhanced Type System** - New data types: `decimal()`, `uuid()`, `json()`, and `interval()` helpers with full SQL support and dialect-specific compilation
- **Interval Arithmetic** - New `date_add()` and `date_sub()` functions for date/time interval operations
- **Join Hints** - New `hints` parameter for `join()` method to provide query optimization hints
- **Complex Join Conditions** - Enhanced `join()` method to support arbitrary Column expressions in join conditions
- **Query Plan Analysis** - New `explain(analyze=False)` method to return query execution plans
- **Pivot Operations** - New `pivot()` method for data reshaping with cross-dialect compatibility

### Version 0.5.0

- **Compressed File Reading** - Automatic detection and support for gzip, bz2, and xz compression in CSV, JSON, JSONL, and text file readers (both sync and async)
- **Array/JSON Functions** - New functions for working with JSON and array data: `json_extract()`, `array()`, `array_length()`, `array_contains()`, `array_position()` with dialect-specific SQL compilation
- **Collect Aggregations** - New aggregation functions `collect_list()` and `collect_set()` for array aggregation (uses `ARRAY_AGG` in PostgreSQL, `group_concat` in SQLite/MySQL)
- **Semi-Join and Anti-Join** - New `semi_join()` and `anti_join()` methods that compile to efficient `EXISTS`/`NOT EXISTS` subqueries
- **MERGE/UPSERT Operations** - New `merge()` method on tables for upsert operations with dialect-specific support (SQLite `ON CONFLICT`, PostgreSQL `MERGE`, MySQL `ON DUPLICATE KEY`)
- **Comprehensive Test Coverage** - All new features include full test coverage with execution tests

<details>
<summary><b>Previous Releases</b></summary>

### Version 0.4.0
- **Strict Type Checking** - Full mypy strict mode compliance with comprehensive type annotations
- **Type Stubs for PyArrow** - Custom type stubs to provide type information for pyarrow library
- **PEP 561 Compliance** - Added `py.typed` marker file
- **Enhanced Type Safety** - Complete type annotations with improved type inference

### Version 0.3.0
- **Separation of File Reads and SQL Operations** - File readers return `Records` instead of `DataFrame`
- **Records Class** - New `Records` and `AsyncRecords` classes for file data
- **Full Async/Await Support** - Complete async API for all operations
- **Async Streaming** - Process large datasets asynchronously

### Version 0.2.0
- **Environment Variable Support** - Configure via environment variables
- **Performance Monitoring Hooks** - Track query execution time
- **Enhanced Security** - Comprehensive SQL injection prevention
- **Modular Architecture** - Refactored file readers

</details>

## 📦 Installation

### Requirements

- Python 3.9+
- SQLAlchemy 2.0+ (for database connectivity)
- A supported SQLAlchemy driver (SQLite, PostgreSQL, MySQL, etc.)

### Install Moltres

```bash
pip install moltres
```

For optional dependencies:

```bash
# For pandas support
pip install moltres[pandas]

# For polars support
pip install moltres[polars]

# For async support (requires async database drivers)
pip install moltres[async]  # Core async support (aiofiles)
pip install moltres[async-postgresql]  # PostgreSQL async (includes async + asyncpg)
pip install moltres[async-mysql]  # MySQL async (includes async + aiomysql)
pip install moltres[async-sqlite]  # SQLite async (includes async + aiosqlite)

# For both pandas and polars
pip install moltres[pandas,polars]
```

## 🚀 Quick Start

### Basic DataFrame Operations

```python
from moltres import col, connect
from moltres.expressions.functions import sum

# Connect to your database
db = connect("sqlite:///example.db")

# DataFrame operations with SQL pushdown (no data loading into memory)
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[("customer_id", "id")])
    .where(col("active") == True)  # noqa: E712
    .group_by("country")
    .agg(sum(col("amount")).alias("total_amount"))
)

# Execute and get results (SQL is compiled and executed here)
results = df.collect()  # Returns list of dicts by default
```

### Raw SQL & SQL Expressions

```python
# Raw SQL queries (PySpark-style)
df = db.sql("SELECT * FROM users WHERE age > 18")
df = db.sql("SELECT * FROM orders WHERE id = :id", id=1).where(col("amount") > 100)

# SQL expression selection
df.selectExpr("amount * 1.1 as with_tax", "UPPER(name) as name_upper")
```

### CRUD Operations

```python
from moltres.io.records import Records

# Insert rows
records = Records(
    _data=[
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": 1},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": 0},
    ],
    _database=db,
)
records.insert_into("customers")  # Executes immediately

# Update rows
df = db.table("customers").select()
df.write.update(
    "customers",
    where=col("active") == 0,
    set={"active": 1, "updated_at": "2024-01-01"}
)  # Executes immediately

# Delete rows
df.write.delete("customers", where=col("email").is_null())  # Executes immediately
```

### Async Support

Moltres also supports async/await for all database operations:

```python
import asyncio
from moltres import async_connect, col

async def main():
    # Connect asynchronously
    db = async_connect("postgresql+asyncpg://user:pass@localhost/db")
    
    # All operations are async
    # For SQL operations, use db.table().select()
    df = db.table("orders").select()
    results = await df.collect()
    
    # Streaming support
    async for chunk in await df.collect(stream=True):
        process_chunk(chunk)
    
    await db.close()

asyncio.run(main())
```

**Note:** Async support requires async database drivers. Install with:
- `pip install moltres[async-postgresql]` for PostgreSQL (includes async + asyncpg)
- `pip install moltres[async-mysql]` for MySQL (includes async + aiomysql)
- `pip install moltres[async-sqlite]` for SQLite (includes async + aiosqlite)
> PostgreSQL async connections now honor DSN `?options=-c...` parameters by translating them into asyncpg `server_settings`. Add `?options=-csearch_path=my_schema` (or other `SET` statements) to keep every pooled connection on the correct schema without custom code.

## 💡 Why Moltres?

### The Gap in Python's Ecosystem

Python has powerful DataFrame tools (Pandas, Polars) and powerful SQL tools (SQLAlchemy, SQLModel), but **no library connects them in a unified, ergonomic way**.

**The problem:** Developers must juggle:
- Pandas or Polars for DataFrame transformations (but data must be loaded into memory)
- SQLAlchemy/ORMs for persistence (but not DataFrame-style)
- Raw SQL for updates/deletes (but not type-safe or composable)

**Moltres fixes this** by providing:
- ✅ **DataFrame API** - Transform data with familiar operations (select, filter, join, groupBy)
- ✅ **SQL Pushdown Execution** - All operations compile to SQL and run on your database—**no data loading into memory**
- ✅ **Real SQL CRUD** - INSERT, UPDATE, DELETE with DataFrame-style syntax
- ✅ **98% PySpark API Compatibility** - Seamless migration from PySpark while maintaining SQL-first design
- ✅ **Works with Existing SQL Infrastructure** - No cluster required, works with SQLite, PostgreSQL, MySQL, and more
- ✅ **Type Safe** - Full type hints for better IDE support and fewer bugs
- ✅ **Production Ready** - Environment variables, connection pooling, monitoring hooks
- ✅ **Secure by Default** - SQL injection prevention built-in

### Who Needs Moltres?

- **Data Engineers** - Avoid loading millions of rows into memory just to update a subset
- **Backend Developers** - Replace many ORM operations with cleaner, column-aware DataFrame syntax
- **Analytics Engineers / dbt Users** - Express SQL models in Python code with DataFrame chaining
- **Product Engineers** - Validated, type-safe CRUD without hand-writing SQL
- **Teams migrating off Spark** - Familiar DataFrame API style for traditional SQL databases—no cluster required. Note: Moltres focuses on SQL features, not PySpark feature parity. Features are included only if they map to SQL capabilities.

## 📖 Core Concepts

> **Design Philosophy:** Moltres provides a DataFrame API that compiles to SQL. We focus on supporting SQL features (standard SQL and common dialect extensions) rather than replicating every PySpark feature. If a feature doesn't map to SQL/SQLAlchemy or doesn't align with SQL pushdown execution, it's not included. However, we maintain **~98% API compatibility** with PySpark for core DataFrame operations to enable seamless migration.

### Key Principles

1. **SQL-First Design** - All operations compile to SQL and execute on the database
2. **No Data Materialization** - Transformations happen in SQL, not in memory
3. **PySpark API Compatibility** - Familiar API for teams migrating from PySpark
4. **Type Safety** - Full type hints and validation throughout
5. **Security by Default** - Built-in SQL injection prevention

### Lazy Evaluation

All DataFrame query operations are lazy—they build a logical plan that only executes when you call `collect()`. DataFrame write operations (insertInto, update, delete) execute eagerly (immediately), matching PySpark's behavior. DDL operations (create_table, drop_table) are lazy and require `.collect()` to execute. The plan is compiled to SQL and executed on your database:

```python
# This doesn't execute any SQL yet
df = db.table("users").select().where(col("age") > 18)

# SQL is compiled and executed here
results = df.collect()
```

### Column Expressions

Moltres supports multiple ways to reference columns, giving you flexibility:

```python
from moltres import col, lit
from moltres.expressions.functions import sum, avg, count, concat

# String names (traditional)
df.select("id", "name", "age")
df.select("*")  # Select all columns
df.select("*", col("new_col"))  # All columns plus new ones
df.where(col("age") > 18)
df.filter("age > 18")  # SQL string predicate (PySpark-compatible)
df.where("age >= 18 AND status = 'active'")  # Complex SQL strings

# Dot notation (PySpark-style)
df.select(df.id, df.name, df.age)
df.where(df.age > 18)
df.order_by(df.name)
df.group_by(df.category).agg(sum(df.amount))

# col() function
df.select(col("id"), col("name"))
df.where(col("age") > 18)

# Mix and match all three methods
df = (
    db.table("sales")
    .select(
        df.product,  # Dot notation
        (col("price") * col("quantity")).alias("revenue"),  # col() function
        concat(df.first_name, lit(" "), df.last_name).alias("full_name"),  # Mixed
    )
    .where(df.date >= "2024-01-01")  # Dot notation in filters
    .group_by("product")  # String name
    .agg(
        sum(df.revenue).alias("total_revenue"),  # Dot notation
        avg(col("price")).alias("avg_price"),  # col() function
        count("*").alias("order_count"),
        "quantity",  # String column name (defaults to sum)
        {"price": "avg", "amount": "sum"},  # Dictionary syntax
    )
)
```

**Note:** Dot notation works by accessing attributes on the DataFrame. If an attribute doesn't exist as a method or property, it's treated as a column name. Existing methods like `select`, `where`, `limit`, and properties like `na`, `write` continue to work as before.

## 📥 Reading Data

### From Database Tables

```python
# Option 1: Use db.table().select() (original API)
df = db.table("customers").select()
df = db.table("customers").select("id", "name", "email")

# Option 2: Use db.read.table() (PySpark-style API)
df = db.read.table("customers")
df = db.read.table("customers").where(col("active") == True).select("id", "name")
```

Both APIs return lazy `DataFrame` objects that can be transformed before execution. The `db.read.table()` API matches PySpark's `spark.read.table()` pattern for consistency.

### SQL Expression Selection

Use `selectExpr()` to write SQL expressions directly, similar to PySpark's `selectExpr()`:

```python
# Basic column selection
df.selectExpr("id", "name", "email")

# With expressions and aliases
df.selectExpr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")

# Complex expressions
df.selectExpr(
    "(amount + tax) * 1.1 as total",
    "CASE WHEN status = 'active' THEN 1 ELSE 0 END as is_active"
)

# Chaining with other operations
df.selectExpr("id", "amount").where(col("amount") > 100)
```

**Key Features:**
- Write SQL expressions directly as strings
- Supports arithmetic, functions, comparisons, and aliases
- Returns lazy `DataFrame` objects that can be chained
- Works with both synchronous and asynchronous DataFrames

### Raw SQL Queries

Execute raw SQL queries and get back a lazy DataFrame, similar to PySpark's `spark.sql()`:

```python
# Basic SQL query
df = db.sql("SELECT * FROM users WHERE age > 18")
results = df.collect()

# Parameterized queries (use :param_name syntax)
df = db.sql("SELECT * FROM users WHERE id = :id AND status = :status", id=1, status="active")
results = df.collect()

# Chain DataFrame operations on SQL results
df = db.sql("SELECT * FROM orders").where(col("amount") > 100).limit(10)
results = df.collect()

# Use with aggregations and joins
df = (
    db.sql("SELECT product, region, SUM(amount) as total FROM sales GROUP BY product, region")
    .where(col("total") > 100)
    .order_by(col("total").desc())
)
results = df.collect()
```

**Key Features:**
- Returns lazy `DataFrame` objects that can be chained with other operations
- Supports parameterized queries using named parameters (`:param_name`)
- SQL dialect is determined by the database connection
- Raw SQL is wrapped in a subquery when chained, enabling full DataFrame API compatibility
- Works with both synchronous and asynchronous databases

### From Files

Moltres supports loading data from various file formats. **File readers (`db.load.*` and `db.read.*`) return lazy `DataFrame` objects** - matching PySpark's API. Files are materialized into temporary tables when `.collect()` is called, enabling SQL pushdown for subsequent operations.
> Async PostgreSQL workloads now use regular staging tables instead of connection-scoped temporary tables, so pooled connections (and pytest-xdist workers) can always access the materialized data before cleanup.

```python
# Option 1: Use db.load.* (original API)
df = db.load.csv("data.csv")
df = db.load.option("delimiter", "|").csv("pipe_delimited.csv")
df = db.load.option("header", False).schema([...]).csv("no_header.csv")

# Option 2: Use db.read.* (PySpark-style API)
df = db.read.csv("data.csv")
df = db.read.json("data.json")  # Array of objects
df = db.read.jsonl("data.jsonl")  # One JSON object per line
df = db.read.parquet("data.parquet")  # Requires pandas and pyarrow
df = db.read.text("log.txt", column_name="line")
df = db.read.textFile("log.txt", column_name="line")  # PySpark-compatible alias
df = db.read.format("csv").option("header", True).load("data.csv")

# Set multiple options at once (PySpark-compatible)
df = db.read.options(header=True, delimiter=",", encoding="UTF-8").csv("data.csv")
df = db.read.options(multiline=True, encoding="UTF-8").json("data.json")

# Both APIs work the same way
df = db.load.csv("data.csv")  # Same as db.read.csv("data.csv")
df = db.read.csv("data.csv")  # Same as db.load.csv("data.csv")

# Transform before materialization
df = db.read.csv("data.csv").where(col("score") > 90).select("name", "score")
rows = df.collect()  # Materializes file and executes SQL operations

# Use db.read.records.* to get LazyRecords (lazy materialization)
lazy_records = db.read.records.csv("data.csv")  # Returns LazyRecords (lazy, not materialized yet)
# LazyRecords automatically materialize when used:
len(lazy_records)  # Auto-materializes when you check length
for row in lazy_records:  # Auto-materializes when you iterate
    process(row)
lazy_records.insert_into("table_name")  # Auto-materializes and executes immediately
rows = lazy_records.rows()  # Auto-materializes when you get rows

# Explicitly materialize with .collect() if needed
records = lazy_records.collect()  # Returns materialized Records object

# For already-materialized data, use dicts() which returns Records directly
records = db.read.records.dicts([{"id": 1, "name": "Alice"}])  # Returns Records (already materialized)
```

### Schema Inference and Explicit Schemas

Schema is automatically inferred from data, but you can provide explicit schemas:

```python
from moltres.table.schema import ColumnDef

schema = [
    ColumnDef(name="id", type_name="INTEGER"),
    ColumnDef(name="name", type_name="TEXT"),
    ColumnDef(name="score", type_name="REAL"),
]

df = db.load.schema(schema).csv("data.csv")
# Or for LazyRecords:
lazy_records = db.read.records.schema(schema).csv("data.csv")
```

**File Format Options:**

**CSV Options:**
- `header` (default: True) - First row contains column names
- `delimiter` or `sep` (default: ",") - Field separator
- `inferSchema` (default: True) - Automatically infer column types
- `encoding` (default: "UTF-8") - File encoding
- `quote` (default: '"') - Quote character
- `escape` (default: "\\") - Escape character
- `nullValue` (default: "") - String representation of null
- `nanValue` (default: "NaN") - String representation of NaN
- `dateFormat` - Date format string for parsing dates
- `timestampFormat` - Timestamp format string for parsing timestamps
- `samplingRatio` (default: 1.0) - Fraction of rows used for schema inference
- `columnNameOfCorruptRecord` - Column name for corrupt records
- `quoteAll` (default: False) - Quote all fields
- `ignoreLeadingWhiteSpace` (default: False) - Ignore leading whitespace
- `ignoreTrailingWhiteSpace` (default: False) - Ignore trailing whitespace
- `comment` - Comment character to skip lines
- `enforceSchema` (default: True) - Enforce schema even if it doesn't match data
- `mode` (default: "PERMISSIVE") - Read mode: "PERMISSIVE", "DROPMALFORMED", or "FAILFAST"
- `compression` - Compression type: "gzip", "bz2", "xz", or None

**JSON Options:**
- `multiline` or `multiLine` (default: False) - If True, reads as JSONL (one object per line)
- `encoding` (default: "UTF-8") - File encoding
- `mode` (default: "PERMISSIVE") - Read mode: "PERMISSIVE", "DROPMALFORMED", or "FAILFAST"
- `columnNameOfCorruptRecord` - Column name for corrupt records
- `dateFormat` - Date format string for parsing dates
- `timestampFormat` - Timestamp format string for parsing timestamps
- `samplingRatio` (default: 1.0) - Fraction of rows used for schema inference
- `lineSep` - Line separator for multiline JSON
- `dropFieldIfAllNull` (default: False) - Drop fields if all values are null
- `compression` - Compression type: "gzip", "bz2", "xz", or None

**Parquet Options:**
- `mergeSchema` (default: False) - Merge schemas from multiple files
- `rebaseDatetimeInRead` (default: True) - Rebase datetime values during read
- `datetimeRebaseMode` (default: "EXCEPTION") - Datetime rebase mode
- `int96RebaseMode` (default: "EXCEPTION") - INT96 rebase mode

**Text Options:**
- `encoding` (default: "UTF-8") - File encoding
- `wholetext` (default: False) - If True, read entire file as single value
- `lineSep` - Line separator (default: newline)
- `compression` - Compression type: "gzip", "bz2", "xz", or None

**Read Modes:**
- `PERMISSIVE` (default) - Sets other fields to null when encountering a corrupted record and puts the malformed string into a field configured by `columnNameOfCorruptRecord`
- `DROPMALFORMED` - Ignores the whole corrupted records
- `FAILFAST` - Throws an exception when it meets corrupted records

**Example with Options:**
```python
# CSV with multiple options
df = db.read.options(
    header=True,
    delimiter="|",
    encoding="UTF-8",
    nullValue="NULL",
    dateFormat="yyyy-MM-dd",
    mode="PERMISSIVE"
).csv("data.csv")

# JSON with options
df = db.read.options(
    multiline=True,
    encoding="UTF-8",
    dropFieldIfAllNull=True,
    mode="DROPMALFORMED"
).json("data.jsonl")

# Text file with wholetext
df = db.read.options(wholetext=True).text("document.txt")
```

> **Important:** 
> - **`db.load.*` and `db.read.*` methods return lazy `DataFrame` objects** - files are materialized into temporary tables when `.collect()` is called, enabling SQL pushdown for subsequent operations. The `db.read.*` API matches PySpark's `spark.read.*` pattern for consistency.
> - **Large File Handling**: By default, files are read in chunks (streaming mode) to safely handle files larger than available memory, similar to PySpark's partition-based approach. This prevents out-of-memory errors when processing large datasets. You can opt-out of chunked reading by setting `stream=False` in options: `db.read.option("stream", False).csv("small_file.csv")`.
> - **`db.read.table()`** - PySpark-style API for reading database tables: `df = db.read.table("customers")`
> - **`db.read.records.*` methods return lazy `LazyRecords`/`AsyncLazyRecords` objects** - Records materialize on-demand when you use Sequence operations (`len()`, indexing, iteration), call `insert_into()`, or use them with `createDataFrame()`. Explicitly materialize with `.collect()` when needed. This provides better performance by deferring file reads until necessary.
>   - LazyRecords automatically materialize when: using `len()`, indexing (`records[0]`), iteration (`for row in records`), calling `insert_into()`, or using with `createDataFrame()`
>   - Explicitly materialize: `records = lazy_records.collect()` to get a materialized `Records` object
>   - For already-materialized data, use `db.read.records.dicts([...])` which returns `Records` directly
> 
> **When to use each:**
> - Use `db.load.*` or `db.read.*` (DataFrames) when you want to transform data before materialization or need SQL pushdown
> - Use `db.read.table()` (PySpark-style) for consistency with PySpark's `spark.read.table()` API
> - Use `db.read.records.*` (LazyRecords) when you want lazy materialization and Records-style operations (insert_into, direct iteration, etc.)

## 📤 Writing Data

### To Database Tables

```python
# Write with automatic schema inference and table creation
df.write.save_as_table("target_table")

# Write modes
df.write.mode("append").save_as_table("target")  # Add to existing (default)
df.write.mode("overwrite").save_as_table("target")  # Replace contents
df.write.mode("ignore").save_as_table("target")  # Skip if the table already exists
df.write.mode("error_if_exists").save_as_table("target")  # Fail if exists

# Insert into existing table (table must exist)
df.write.insertInto("existing_table")

# Update rows in a table (eager execution)
df.write.update("table_name", where=col("id") == 1, set={"name": "Updated"})

# Delete rows from a table (eager execution)
df.write.delete("table_name", where=col("id") == 1)

# With explicit schema
from moltres.table.schema import ColumnDef
schema = [
    ColumnDef(name="id", type_name="INTEGER"),
    ColumnDef(name="name", type_name="TEXT"),
]
df.write.schema(schema).save_as_table("target")
```

### To Files

```python
# Various formats
df.write.csv("output.csv")
df.write.json("output.json")
df.write.jsonl("output.jsonl")
df.write.parquet("output.parquet")  # Requires pandas and pyarrow

# Generic save
df.write.save("output.csv")  # Infers format from extension
df.write.format("csv").options(header=False, delimiter="|").save("custom.csv")
df.write.save("data.txt", format="csv")  # Explicit format
df.write.text("lines.txt")  # Writes the `value` column to disk

# With options
df.write.option("header", True).option("delimiter", "|").csv("output.csv")
df.write.option("compression", "gzip").parquet("output.parquet")

# Bulk options and format builder (PySpark-style)
(
    df.write.format("json")
    .options(mode="ignore", lineSep="\\n")
    .save("data.json")
)

# Partitioned writes
df.write.partitionBy("country", "year").csv("partitioned_data")
```

**File Formats:**
- **CSV**: Standard comma-separated values (options: `header`, `delimiter`)
- **JSON**: Array of objects (options: `indent`; streams automatically when `indent` is 0/`None`)
- **JSONL**: One JSON object per line (streamed by default)
- **Text**: Plain text output sourced from the `value` column (use `.option("column", "col_name")` to change)
- **Parquet**: Columnar format (requires `pandas` and `pyarrow`, options: `compression`)

> ℹ️ **Streaming writes by default:** any sink that requires materializing rows now streams in fixed-size chunks automatically for both sync and async writers. Call `.stream(False)` if you explicitly want to buffer everything in memory first.

> ⚠️ **Bucketing / sorting metadata:** `.bucketBy()` and `.sortBy()` are accepted for PySpark API compatibility, but Moltres will raise `NotImplementedError` until downstream databases understand these layouts.

## 🌊 Streaming for Large Datasets

Moltres supports streaming operations for datasets larger than available memory:

```python
# Enable streaming mode for LazyRecords (using read.records)
lazy_records = db.read.records.stream().option("chunk_size", 10000).csv("large_file.csv")

# Process records (LazyRecords auto-materialize when iterated, streaming processes in chunks)
for row in lazy_records:
    process(row)

# Or materialize all at once
all_rows = lazy_records.rows()  # Auto-materializes all data

# Explicitly materialize to get Records object
records = lazy_records.collect()  # Returns materialized Records

# For DataFrames, use collect(stream=True)
df = db.load.csv("large_file.csv")
for chunk in df.collect(stream=True):  # Returns iterator of row chunks
    process_chunk(chunk)

# Streaming writes (explicit opt-in still supported)
df.write.stream().mode("overwrite").save_as_table("large_table")
df.write.stream(False).csv("output.csv")  # Disable streaming if you really need to buffer

# Streaming SQL queries
df = db.table("large_table").select()
for chunk in df.collect(stream=True):
    process_chunk(chunk)
```

**Streaming Options:**
- `.stream()`: Enable streaming mode manually (writes now stream by default unless you pass `.stream(False)`)
- `.option("chunk_size", N)`: Set chunk size for reads (default: 10000)
- `.option("batch_size", N)`: Set batch size for SQL inserts (default: 10000)
- `collect(stream=True)`: Return iterator of row chunks instead of materializing

**When to Use Streaming:**
- Files or tables larger than available RAM
- Processing data in batches for memory efficiency
- Incremental processing pipelines
- Large data transformations

## 🗄️ Table Management

### Creating Tables

```python
from moltres import column, connect

db = connect("sqlite:///example.db")

# Create a table with schema definition (lazy operation)
customers = db.create_table(
    "customers",
    [
        column("id", "INTEGER", nullable=False, primary_key=True),
        column("name", "TEXT", nullable=False),
        column("email", "TEXT", nullable=True),
        column("active", "INTEGER", default=1),
    ],
).collect()  # Execute the create_table operation

# Insert data using Records
from moltres.io.records import Records
records = Records(
    _data=[
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ],
    _database=db,
)
records.insert_into("customers")  # Executes immediately

# Drop tables (lazy operation)
db.drop_table("customers").collect()  # Execute the drop_table operation
```

The `column()` helper accepts:
- `name`: Column name
- `type_name`: SQL type (e.g., "INTEGER", "TEXT", "REAL", "VARCHAR(255)")
- `nullable`: Whether the column allows NULL (default: True)
- `default`: Default value for the column (optional)
- `primary_key`: Whether this is a primary key column (default: False)

## ✏️ Data Mutations

DataFrame write operations (insert, update, delete) execute **eagerly** (immediately), matching PySpark's behavior. DDL operations (create_table, drop_table) are **lazy** and require `.collect()` to execute:

```python
from moltres import col
from moltres.io.records import Records

# Insert rows using Records (for raw data)
records = Records(
    _data=[
    {"id": 1, "name": "Alice", "active": 1},
    {"id": 2, "name": "Bob", "active": 0},
    ],
    _database=db,
)
records.insert_into("customers")  # Executes immediately

# Or insert from DataFrame
df = db.table("source").select()
df.write.insertInto("customers")  # Executes immediately

# Update rows using DataFrame write API (eager execution)
df = db.table("customers").select()
df.write.update("customers", where=col("id") == 2, set={"active": 1})  # Executes immediately

# Delete rows using DataFrame write API (eager execution)
df.write.delete("customers", where=col("active") == 0)  # Executes immediately
```

**Note:** DDL operations (create_table, drop_table) remain **lazy** and require `.collect()` to execute.

### Transaction Support

All operations within a transaction context share the same database connection and transaction. If any operation fails, all changes are automatically rolled back:

```python
# Execute multiple operations in a single transaction
from moltres.io.records import Records

with db.transaction() as txn:
    # All operations share the same transaction
    records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
    records.insert_into("customers")
    records2 = Records(_data=[{"id": 1, "customer_id": 1, "amount": 100.0}], _database=db)
    records2.insert_into("orders")
    # DataFrame write operations also participate in the transaction
    df = db.table("customers").select()
    df.write.update("customers", where=col("id") == 1, set={"name": "Alice Updated"})
    # If any operation fails, all changes are rolled back
    # If all succeed, transaction commits automatically
```

You can also explicitly control the transaction:

```python
with db.transaction() as txn:
    records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
    records.insert_into("customers")
    txn.commit()  # Explicit commit
    # Or txn.rollback() to rollback
```

**Note:** By default, each write operation executes in its own auto-commit transaction. Use `db.transaction()` to group multiple operations into a single atomic transaction.

### Batch Operations

The batch API allows you to queue multiple lazy operations and execute them together in a single transaction:

```python
# Queue multiple DDL operations and execute them atomically
# Note: DataFrame write operations (insertInto, update, delete) are eager and execute immediately
# Only lazy DDL operations (create_table, drop_table) can be batched
with db.batch():
    # All DDL operations are queued and executed together on exit
    db.create_table("users", [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
    ])
    # After table is created, use Records or DataFrame writes for inserts
    # (These execute immediately, not in the batch)
    # All DDL operations execute together in a single transaction
    # If any operation fails, all changes are rolled back
```

The batch context manager automatically:
- Queues all lazy operations created within the context
- Executes them together in a single transaction when exiting the context
- Rolls back all changes if any operation fails
- Supports both synchronous and asynchronous operations

## 📊 Result Formats

By default, `collect()` returns a list of dictionaries (`fetch_format="records"`), so Moltres works even when pandas/polars are unavailable. You can configure the result format when connecting:

```python
# Default: list of dicts
db = connect("sqlite:///example.db")
results = df.collect()  # List[Dict[str, Any]]

# Pandas DataFrame (requires pandas)
db = connect("sqlite:///example.db", fetch_format="pandas")
results = df.collect()  # pandas.DataFrame

# Polars DataFrame (requires polars)
db = connect("sqlite:///example.db", fetch_format="polars")
results = df.collect()  # polars.DataFrame
```

## ⚙️ Configuration

### Programmatic Configuration

```python
db = connect(
    "postgresql://user:pass@host/dbname",
    echo=True,  # Enable SQL logging
    fetch_format="pandas",
    pool_size=10,
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,  # Connection health checks
)
```

### Environment Variables

Moltres supports configuration via environment variables for easier deployment (12-factor app friendly):

```bash
export MOLTRES_DSN="postgresql://user:pass@host/dbname"
export MOLTRES_POOL_SIZE=10
export MOLTRES_POOL_PRE_PING=true
export MOLTRES_FETCH_FORMAT="pandas"
```

Then in your code:
```python
from moltres import connect

# Uses MOLTRES_DSN from environment
db = connect()
```

**Supported environment variables:**
- `MOLTRES_DSN`: Database connection string
- `MOLTRES_ECHO`: Enable SQL logging (true/false)
- `MOLTRES_FETCH_FORMAT`: "records", "pandas", or "polars"
- `MOLTRES_DIALECT`: Override SQL dialect
- `MOLTRES_POOL_SIZE`: Connection pool size
- `MOLTRES_MAX_OVERFLOW`: Maximum pool overflow
- `MOLTRES_POOL_TIMEOUT`: Pool timeout in seconds
- `MOLTRES_POOL_RECYCLE`: Connection recycle time
- `MOLTRES_POOL_PRE_PING`: Enable connection health checks (true/false)

**Configuration Precedence:** Programmatic arguments > Environment variables > Defaults

## 📈 Performance Monitoring

Moltres provides optional performance monitoring hooks for tracking query execution:

```python
from moltres.engine import register_performance_hook

def log_slow_queries(sql: str, elapsed: float, metadata: dict):
    if elapsed > 1.0:
        print(f"Slow query ({elapsed:.2f}s): {sql[:100]}")
        print(f"  Rows affected: {metadata.get('rowcount', 'N/A')}")

register_performance_hook("query_end", log_slow_queries)

# Now all queries will be monitored
db = connect("sqlite:///example.db")
df.collect()  # Slow queries will be logged

# Unregister when done
from moltres.engine import unregister_performance_hook
unregister_performance_hook("query_end", log_slow_queries)
```

**Available Events:**
- `query_start`: Fired when a query begins execution
- `query_end`: Fired when a query completes (includes elapsed time and metadata)

## 🔒 Security

Moltres includes built-in security features to prevent SQL injection:

- **SQL Identifier Validation** - All table and column names are validated
- **Parameterized Queries** - All user data is passed as parameters, never string concatenation
- **Input Sanitization** - Comprehensive validation of identifiers and inputs

See [`docs/SECURITY.md`](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md) for security best practices and guidelines.

## 📚 Advanced Examples

### Complex Joins and Aggregations

```python
from moltres import col
from moltres.expressions.functions import sum, avg, count

# Multi-table join with aggregations
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[("customer_id", "id")])
    .join(db.table("products").select(), on=[("product_id", "id")])
    .where(col("date") >= "2024-01-01")
    .group_by("country", "category")
    .agg(
        sum(col("amount")).alias("total_revenue"),
        avg(col("amount")).alias("avg_order_value"),
        count("*").alias("order_count"),
    )
    .order_by(col("total_revenue").desc())
    .limit(10)
)

results = df.collect()
```

### Window Functions

```python
# Complex expressions with window functions
df = (
    db.table("sales")
    .select(
        col("product"),
        col("amount"),
        col("date"),
        (col("amount") - avg(col("amount")).over()).alias("deviation_from_avg"),
    )
    .where(col("date") >= "2024-01-01")
)
```

### Complete ETL Pipeline

```python
# Complete ETL pipeline
db = connect("postgresql://user:pass@localhost/warehouse")

# Extract: Load from CSV (returns DataFrame, or use read.records for LazyRecords)
raw_df = db.load.csv("raw_sales.csv")
# Or for LazyRecords (lazy materialization):
raw_records = db.read.records.csv("raw_sales.csv")

# Load raw data into staging table (lazy operation)
db.create_table("staging_sales", [
    column("order_id", "INTEGER"),
    column("product", "TEXT"),
    column("amount", "REAL"),
    column("date", "DATE"),
]).collect()  # Execute the create_table operation
# LazyRecords auto-materialize when insert_into() is called
raw_records.insert_into("staging_sales")  # Auto-materializes and executes immediately

# Transform: Clean and aggregate using SQL operations
cleaned = (
    db.table("staging_sales")
    .select(
        col("order_id"),
        col("product").upper().alias("product"),
        col("amount").cast("REAL"),
        col("date"),
    )
    .where(col("amount") > 0)
    .group_by("product", "date")
    .agg(sum(col("amount")).alias("daily_revenue"))
)

# Load: Write to database
cleaned.write.mode("overwrite").save_as_table("daily_sales_summary")
```

## 🛠️ Supported Operations

### DataFrame Operations (PySpark-Compatible)
- `select()` / `selectExpr()` - Project columns or SQL expressions
- `where()` / `filter()` - Filter rows (supports SQL strings)
- `join()` - Join with other DataFrames
- `group_by()` / `groupBy()` - Group rows
- `agg()` - Aggregate functions (supports strings and dictionaries)
- `order_by()` / `orderBy()` / `sort()` - Sort rows
- `limit()` - Limit number of rows
- `distinct()` - Remove duplicate rows
- `withColumn()` / `withColumnRenamed()` - Add or rename columns
- `pivot()` - Pivot operations (including `groupBy().pivot()`)
- `explode()` - Explode array/JSON columns
- `db.sql()` - Execute raw SQL queries

### DataFrame Write Operations
- `df.write.insertInto("table")` - Insert DataFrame into existing table (eager execution)
- `df.write.update("table", where=..., set={...})` - Update rows in table (eager execution)
- `df.write.delete("table", where=...)` - Delete rows from table (eager execution)
- `df.write.save_as_table("table")` / `saveAsTable()` - Write DataFrame to table (eager execution)

### Column Expressions
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Boolean**: `&`, `|`, `~`
- **Functions**: Comprehensive function library with 130+ functions including:
  - **Mathematical**: `pow()`, `power()`, `sqrt()`, `abs()`, `floor()`, `ceil()`, `round()`, `sin()`, `cos()`, `tan()`, `asin()`, `acos()`, `atan()`, `atan2()`, `log()`, `log10()`, `log2()`, `exp()`, `signum()`, `sign()`, `hypot()`
  - **String**: `concat()`, `upper()`, `lower()`, `substring()`, `trim()`, `ltrim()`, `rtrim()`, `length()`, `lpad()`, `rpad()`, `replace()`, `regexp_extract()`, `regexp_replace()`, `split()`, `initcap()`, `instr()`, `locate()`, `translate()`
  - **Date/Time**: `year()`, `month()`, `day()`, `hour()`, `minute()`, `second()`, `dayofweek()`, `dayofyear()`, `quarter()`, `weekofyear()`, `week()`, `date_format()`, `to_date()`, `to_timestamp()`, `current_date()`, `current_timestamp()`, `datediff()`, `date_add()`, `date_sub()`, `add_months()`, `date_trunc()`, `last_day()`, `months_between()`, `unix_timestamp()`, `from_unixtime()`
  - **Aggregate**: `sum()`, `avg()`, `min()`, `max()`, `count()`, `count_distinct()`, `stddev()`, `variance()`, `corr()`, `covar()`, `collect_list()`, `collect_set()`, `percentile_cont()`, `percentile_disc()`
  - **Window**: `row_number()`, `rank()`, `dense_rank()`, `percent_rank()`, `cume_dist()`, `nth_value()`, `ntile()`, `lag()`, `lead()`, `first_value()`, `last_value()`
  - **Array**: `array()`, `array_length()`, `array_contains()`, `array_position()`, `array_append()`, `array_prepend()`, `array_remove()`, `array_distinct()`, `array_sort()`, `array_max()`, `array_min()`, `array_sum()`
  - **JSON**: `json_extract()`, `json_tuple()`, `from_json()`, `to_json()`, `json_array_length()`
  - **Utility**: `coalesce()`, `greatest()`, `least()`, `when()`, `isnull()`, `isnotnull()`, `isnan()`, `isinf()`, `rand()`, `randn()`, `hash()`, `md5()`, `sha1()`, `sha2()`, `base64()`, `monotonically_increasing_id()`, `crc32()`, `soundex()`
- **Window Functions**: `over()`, `partition_by()`, `order_by()`

### Supported SQL Dialects
- ✅ **SQLite** - Full support
- ✅ **PostgreSQL** - Full support with dialect-specific optimizations
- ✅ **MySQL** - Full support with dialect-specific optimizations
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

## 📖 Documentation

Additional documentation is available in the `docs/` directory:

- **[Why Moltres?](https://github.com/eddiethedean/moltres/blob/main/docs/WHY_MOLTRES.md)** - Understanding the gap Moltres fills and who needs it
- **[Examples](https://github.com/eddiethedean/moltres/blob/main/docs/EXAMPLES.md)** - Common patterns, use cases, and examples for each audience
- **[Security Guide](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md)** - Security best practices and SQL injection prevention
- **[Troubleshooting](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Design Notes](https://github.com/eddiethedean/moltres/blob/main/docs/moltres_plan.md)** - High-level architecture and design decisions
- **[Advocacy Document](https://github.com/eddiethedean/moltres/blob/main/docs/moltres_advocacy.md)** - Detailed positioning and comparison with alternatives

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
