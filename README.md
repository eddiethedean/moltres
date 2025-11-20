# Moltres

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)

> A PySpark-inspired DataFrame API that compiles to SQL and runs on your existing database.

Moltres provides a familiar DataFrame API similar to PySpark, but instead of running on a Spark cluster, it compiles your operations into ANSI SQL and executes them directly against your database through SQLAlchemy. Compose column expressions, joins, aggregates, and mutations with the same ergonomics you'd expect from Spark—all while leveraging your existing SQL infrastructure.

## Features

- 🚀 **PySpark-like API** - Familiar DataFrame operations (select, filter, join, groupBy, etc.)
- 🗄️ **SQL Compilation** - All operations compile to ANSI SQL and run on your database
- 📊 **Multiple Formats** - Read/write CSV, JSON, JSONL, Parquet, and more
- 🌊 **Streaming Support** - Handle datasets larger than memory with chunked processing
- 🔧 **Type Safe** - Full type hints and mypy support
- 🎯 **Zero Dependencies** - Works with just SQLAlchemy (pandas/polars optional)

## Requirements

- Python 3.9+
- SQLAlchemy 2.0+ (for database connectivity)
- A supported SQLAlchemy driver (SQLite, PostgreSQL, MySQL, etc.)

## Installation

```bash
pip install moltres
```

For optional dependencies:

```bash
# For pandas support
pip install moltres[pandas]

# For polars support
pip install moltres[polars]

# For both
pip install moltres[pandas,polars]
```

## Quick Start

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

## Core Concepts

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

## Reading Data

### From Database Tables

```python
# Simple table read
df = db.read.table("customers")

# With column selection
df = db.table("customers").select("id", "name", "email")
```

### From Files

Moltres supports reading from various file formats:

```python
# CSV files
df = db.read.csv("data.csv")
df = db.read.option("delimiter", "|").csv("pipe_delimited.csv")
df = db.read.option("header", False).schema([...]).csv("no_header.csv")

# JSON files
df = db.read.json("data.json")  # Array of objects
df = db.read.jsonl("data.jsonl")  # One JSON object per line

# Parquet files (requires pandas and pyarrow)
df = db.read.parquet("data.parquet")

# Text files (one line per row)
df = db.read.text("log.txt", column_name="line")

# Generic format reader
df = db.read.format("csv").option("header", True).load("data.csv")
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

df = db.read.schema(schema).csv("data.csv")
```

**File Format Options:**
- **CSV**: `header` (default: True), `delimiter` (default: ","), `inferSchema` (default: True)
- **JSON**: `multiline` (default: False) - if True, reads as JSONL
- **Parquet**: Requires `pandas` and `pyarrow`

> **Note:** File-based readers materialize data into memory (not lazy) since files aren't in the SQL database.

## Writing Data

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

## Streaming for Large Datasets

Moltres supports streaming operations for datasets larger than available memory:

```python
# Enable streaming mode
df = db.read.stream().option("chunk_size", 10000).csv("large_file.csv")

# Process chunks one at a time
for chunk in df.collect(stream=True):
    process_chunk(chunk)

# Or materialize all data (backward compatible)
all_rows = df.collect()  # Still works, materializes all chunks

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

## Table Management

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

## Data Mutations

Insert, update, and delete operations run eagerly:

```python
from moltres import col

customers = db.table("customers")

# Insert rows
customers.insert([
    {"id": 1, "name": "Alice", "active": 1},
    {"id": 2, "name": "Bob", "active": 0},
])

# Update rows
customers.update(where=col("id") == 2, set={"active": 1})

# Delete rows
customers.delete(where=col("active") == 0)
```

## Result Formats

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

## Advanced Examples

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

### Window Functions and Subqueries

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

### Data Pipeline Example

```python
# Complete ETL pipeline
db = connect("postgresql://user:pass@localhost/warehouse")

# Extract: Read from CSV
raw_data = db.read.csv("raw_sales.csv")

# Transform: Clean and aggregate
cleaned = (
    raw_data
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

## Supported Operations

### DataFrame Operations
- `select()` - Project columns
- `where()` / `filter()` - Filter rows
- `join()` - Join with other DataFrames
- `group_by()` / `groupBy()` - Group rows
- `agg()` - Aggregate functions
- `order_by()` - Sort rows
- `limit()` - Limit number of rows

### Column Expressions
- Arithmetic: `+`, `-`, `*`, `/`, `%`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean: `&`, `|`, `~`
- Functions: `sum()`, `avg()`, `count()`, `concat()`, `coalesce()`, `upper()`, `lower()`, etc.

### Supported SQL Dialects
- SQLite
- PostgreSQL
- MySQL (basic support)
- Other SQLAlchemy-supported databases (with ANSI SQL fallback)

## Development

### Running Tests

Third-party pytest plugins can interfere with the runtime, so disable auto-loading:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking
mypy src
```

## Documentation

Additional design notes and architecture details are available in the `docs/` directory. The high-level architecture follows the plan in `moltres_plan.md`, covering:

- Expression builder
- Logical planner
- SQL compiler
- Execution engine
- Mutation layer
- Read/Write layers
- Streaming support

## License

MIT

## Author

**Odos Matthews**

- GitHub: [@eddiethedean](https://github.com/eddiethedean)
- Email: odosmatthews@gmail.com

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
