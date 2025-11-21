# Moltres

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> A PySpark-inspired DataFrame API that compiles to SQL and runs on your existing database.

**Moltres** provides a familiar DataFrame API similar to PySpark, but instead of running on a Spark cluster, it compiles your operations into ANSI SQL and executes them directly against your database through SQLAlchemy. Compose column expressions, joins, aggregates, and mutations with the same ergonomics you'd expect from Spark—all while leveraging your existing SQL infrastructure.

## ✨ Features

- 🚀 **PySpark-like API** - Familiar DataFrame operations (select, filter, join, groupBy, etc.)
- 🗄️ **SQL Compilation** - All operations compile to ANSI SQL and run on your database
- 📊 **Multiple Formats** - Read/write CSV, JSON, JSONL, Parquet, and more
- 🌊 **Streaming Support** - Handle datasets larger than memory with chunked processing
- 🔧 **Type Safe** - Full type hints with strict mypy checking and custom type stubs for dependencies
- 🎯 **Zero Dependencies** - Works with just SQLAlchemy (pandas/polars optional)
- 🔒 **Security First** - Built-in SQL injection prevention and validation
- ⚡ **Performance Monitoring** - Optional hooks for query performance tracking
- 🌍 **Environment Config** - Configure via environment variables for 12-factor apps
- ⚡ **Async Support** - Full async/await support for all operations (optional dependency)

## 🆕 What's New in 0.4.0

- **Strict Type Checking** - Full mypy strict mode compliance with comprehensive type annotations across the entire codebase
- **Type Stubs for PyArrow** - Custom type stubs (`stubs/pyarrow/`) to provide type information for pyarrow library
- **PEP 561 Compliance** - Added `py.typed` marker file to signal that the package is fully typed
- **Enhanced Type Safety** - All functions and methods now have complete type annotations with improved type inference
- **Code Quality Improvements** - Removed all unused type ignore comments and fixed type inference issues

## What's New in 0.3.0

- **Separation of File Reads and SQL Operations** - File readers (`db.load.*`) now return `Records` instead of `DataFrame`, making it clear that file data is materialized and not suitable for SQL operations. Use `db.table(name).select()` for SQL queries.
- **Records Class** - New `Records` and `AsyncRecords` classes for file data that support iteration, indexing, and direct use with insert operations
- **Full Async/Await Support** - Complete async API for all database operations, file I/O, and DataFrame operations
- **Async Database Operations** - Use `async_connect()` for async database connections with SQLAlchemy async engines
- **Async File Operations** - Async reading and writing of CSV, JSON, JSONL, Parquet, and text files
- **Optional Async Dependencies** - Install async support with `pip install moltres[async-postgresql]` (or async-mysql, async-sqlite)
- **Async Streaming** - Process large datasets asynchronously with async iterators

## What's New in 0.2.0

- **Environment Variable Support** - Configure Moltres via `MOLTRES_DSN`, `MOLTRES_POOL_SIZE`, etc.
- **Performance Monitoring Hooks** - Track query execution time with `register_performance_hook()`
- **Enhanced Security** - Comprehensive SQL injection prevention and security documentation
- **Modular Architecture** - Refactored file readers into organized, maintainable modules
- **Comprehensive Testing** - 113 test cases covering edge cases, security, and error handling
- **Better Documentation** - Security guide, troubleshooting, and examples documentation

## 📋 Requirements

- Python 3.9+
- SQLAlchemy 2.0+ (for database connectivity)
- A supported SQLAlchemy driver (SQLite, PostgreSQL, MySQL, etc.)

## 📦 Installation

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

```python
from moltres import col, connect
from moltres.expressions.functions import sum

# Connect to your database
db = connect("sqlite:///example.db")

# Compose queries lazily
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[("customer_id", "id")])
    .where(col("customers.active") == True)  # noqa: E712
    .group_by("customers.country")
    .agg(sum(col("orders.amount")).alias("total_amount"))
)

# Execute and get results
results = df.collect()  # Returns list of dicts by default
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

## 💡 Why Moltres?

**Moltres** bridges the gap between PySpark's elegant DataFrame API and traditional SQL databases:

- ✅ **No Spark Cluster Required** - Run on your existing SQL database
- ✅ **Familiar API** - PySpark developers feel at home immediately
- ✅ **Type Safe** - Full type hints for better IDE support and fewer bugs
- ✅ **Production Ready** - Environment variables, connection pooling, monitoring hooks
- ✅ **Secure by Default** - SQL injection prevention built-in
- ✅ **Flexible** - Works with SQLite, PostgreSQL, MySQL, and more

## 📖 Core Concepts

### Lazy Evaluation

All DataFrame operations are lazy—they build a logical plan that only executes when you call `collect()`:

```python
# This doesn't execute any SQL yet
df = db.table("users").select().where(col("age") > 18)

# SQL is compiled and executed here
results = df.collect()
```

### Column Expressions

Build complex expressions using column operations:

```python
from moltres.expressions.functions import sum, avg, count, concat

df = (
    db.table("sales")
    .select(
        col("product"),
        (col("price") * col("quantity")).alias("revenue"),
        concat(col("first_name"), lit(" "), col("last_name")).alias("full_name"),
    )
    .where(col("date") >= "2024-01-01")
    .group_by("product")
    .agg(
        sum(col("revenue")).alias("total_revenue"),
        avg(col("price")).alias("avg_price"),
        count("*").alias("order_count"),
    )
)
```

## 📥 Reading Data

### From Database Tables

```python
# For SQL operations, use db.table().select()
df = db.table("customers").select()
df = db.table("customers").select("id", "name", "email")
```

### From Files

Moltres supports loading data from various file formats. **File readers return `Records`, not `DataFrame`** - this makes it clear that file data is materialized and not suitable for SQL operations.

```python
# CSV files - returns Records
records = db.load.csv("data.csv")
records = db.load.option("delimiter", "|").csv("pipe_delimited.csv")
records = db.load.option("header", False).schema([...]).csv("no_header.csv")

# JSON files - returns Records
records = db.load.json("data.json")  # Array of objects
records = db.load.jsonl("data.jsonl")  # One JSON object per line

# Parquet files (requires pandas and pyarrow) - returns Records
records = db.load.parquet("data.parquet")

# Text files (one line per row) - returns Records
records = db.load.text("log.txt", column_name="line")

# Generic format reader - returns Records
records = db.load.format("csv").option("header", True).load("data.csv")

# Records can be used directly with insert operations
table.insert(records)  # Records implements Sequence protocol
# Or use the convenience method
records.insert_into("table_name")

# Access data
rows = records.rows()  # Get all rows as a list
for row in records:  # Iterate directly
    process(row)
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

records = db.load.schema(schema).csv("data.csv")
```

**File Format Options:**
- **CSV**: `header` (default: True), `delimiter` (default: ","), `inferSchema` (default: True)
- **JSON**: `multiline` (default: False) - if True, reads as JSONL
- **Parquet**: Requires `pandas` and `pyarrow`

> **Important:** File readers (`db.load.*`) return `Records`, not `DataFrame`. Records are materialized data containers that can be:
> - Iterated directly: `for row in records: ...`
> - Converted to a list: `rows = records.rows()`
> - Used with insert operations: `table.insert(records)` or `records.insert_into("table")`
> 
> For SQL operations (select, filter, join, etc.), use `db.table(name).select()` to get a DataFrame.

## 📤 Writing Data

### To Database Tables

```python
# Write with automatic schema inference and table creation
df.write.save_as_table("target_table")

# Write modes
df.write.mode("append").save_as_table("target")  # Add to existing (default)
df.write.mode("overwrite").save_as_table("target")  # Replace contents
df.write.mode("error_if_exists").save_as_table("target")  # Fail if exists

# Insert into existing table (table must exist)
df.write.insertInto("existing_table")

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
df.write.save("data.txt", format="csv")  # Explicit format

# With options
df.write.option("header", True).option("delimiter", "|").csv("output.csv")
df.write.option("compression", "gzip").parquet("output.parquet")

# Partitioned writes
df.write.partitionBy("country", "year").csv("partitioned_data")
```

**File Formats:**
- **CSV**: Standard comma-separated values (options: `header`, `delimiter`)
- **JSON**: Array of objects (options: `indent`)
- **JSONL**: One JSON object per line
- **Parquet**: Columnar format (requires `pandas` and `pyarrow`, options: `compression`)

## 🌊 Streaming for Large Datasets

Moltres supports streaming operations for datasets larger than available memory:

```python
# Enable streaming mode
records = db.load.stream().option("chunk_size", 10000).csv("large_file.csv")

# Process records (streaming Records iterate row-by-row)
for row in records:
    process(row)

# Or materialize all at once
all_rows = records.rows()  # Materializes all data

# Streaming writes
df.write.stream().mode("overwrite").save_as_table("large_table")
df.write.stream().csv("output.csv")

# Streaming SQL queries
df = db.table("large_table").select()
for chunk in df.collect(stream=True):
    process_chunk(chunk)
```

**Streaming Options:**
- `.stream()`: Enable streaming mode (default: False for backward compatibility)
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

# Create a table with schema definition
customers = db.create_table(
    "customers",
    [
        column("id", "INTEGER", nullable=False, primary_key=True),
        column("name", "TEXT", nullable=False),
        column("email", "TEXT", nullable=True),
        column("active", "INTEGER", default=1),
    ],
)

# Insert data
customers.insert([
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
])

# Drop tables
db.drop_table("customers")
```

The `column()` helper accepts:
- `name`: Column name
- `type_name`: SQL type (e.g., "INTEGER", "TEXT", "REAL", "VARCHAR(255)")
- `nullable`: Whether the column allows NULL (default: True)
- `default`: Default value for the column (optional)
- `primary_key`: Whether this is a primary key column (default: False)

## ✏️ Data Mutations

Insert, update, and delete operations run eagerly:

```python
from moltres import col

customers = db.table("customers")

# Insert rows (batch optimized)
customers.insert([
    {"id": 1, "name": "Alice", "active": 1},
    {"id": 2, "name": "Bob", "active": 0},
])

# Update rows
customers.update(where=col("id") == 2, set={"active": 1})

# Delete rows
customers.delete(where=col("active") == 0)
```

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

See [`docs/SECURITY.md`](docs/SECURITY.md) for security best practices and guidelines.

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
    .where(col("orders.date") >= "2024-01-01")
    .group_by("customers.country", "products.category")
    .agg(
        sum(col("orders.amount")).alias("total_revenue"),
        avg(col("orders.amount")).alias("avg_order_value"),
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

# Extract: Load from CSV (returns Records)
raw_records = db.load.csv("raw_sales.csv")

# Load raw data into staging table
db.create_table("staging_sales", [
    column("order_id", "INTEGER"),
    column("product", "TEXT"),
    column("amount", "REAL"),
    column("date", "DATE"),
])
raw_records.insert_into("staging_sales")

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

### DataFrame Operations
- `select()` - Project columns
- `where()` / `filter()` - Filter rows
- `join()` - Join with other DataFrames
- `group_by()` / `groupBy()` - Group rows
- `agg()` - Aggregate functions
- `order_by()` - Sort rows
- `limit()` - Limit number of rows

### Column Expressions
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Boolean**: `&`, `|`, `~`
- **Functions**: `sum()`, `avg()`, `count()`, `concat()`, `coalesce()`, `upper()`, `lower()`, etc.

### Supported SQL Dialects
- ✅ SQLite
- ✅ PostgreSQL
- ✅ MySQL (basic support)
- ✅ Other SQLAlchemy-supported databases (with ANSI SQL fallback)

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

- **[Security Guide](docs/SECURITY.md)** - Security best practices and SQL injection prevention
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Examples](docs/EXAMPLES.md)** - Common patterns and usage examples
- **[Design Notes](docs/moltres_plan.md)** - High-level architecture and design decisions

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Odos Matthews**

- GitHub: [@eddiethedean](https://github.com/eddiethedean)
- Email: odosmatthews@gmail.com

## 🙏 Acknowledgments

- Inspired by PySpark's DataFrame API
- Built on SQLAlchemy for database connectivity
- Thanks to all contributors and users

---

**Made with ❤️ for the Python data community**
