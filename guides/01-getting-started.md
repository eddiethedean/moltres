# Getting Started with Moltres

This guide will help you get up and running with Moltres in minutes.

## Installation

```bash
# Basic installation
pip install moltres

# With async support (choose your database)
pip install moltres[async-postgresql]  # PostgreSQL
pip install moltres[async-mysql]       # MySQL
pip install moltres[async-sqlite]      # SQLite

# With pandas/polars result formats
pip install moltres[pandas,polars]
```

## Your First Connection

Moltres works with any SQLAlchemy-compatible database. Here's how to connect:

**See also:** [Connection examples](https://github.com/eddiethedean/moltres/blob/main/examples/01_connecting.py)

```python
from moltres import connect

# SQLite in-memory (great for learning and testing - no file needed)
db = connect("sqlite:///:memory:")

# SQLite file-based (persistent database)
# db = connect("sqlite:///example.db")

# PostgreSQL (requires PostgreSQL server)
# db = connect("postgresql://user:password@localhost:5432/mydb")

# MySQL (requires MySQL server)
# db = connect("mysql://user:password@localhost:3306/mydb")

# DuckDB (file-based)
# db = connect("duckdb:///path/to/database.db")
```

## Creating Your First Table

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite - no file setup needed
db = connect("sqlite:///:memory:")

# Create a table with explicit schema
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),  # SQLite uses INTEGER for booleans
]).collect()  # .collect() executes the operation

print("Table 'users' created successfully!")
```

## Inserting Data

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Setup database and table
db = connect("sqlite:///:memory:")
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
]).collect()

# Insert from a list of dictionaries
result = Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30, "active": 1},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25, "active": 1},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35, "active": 0},
], database=db).insert_into("users")

print(f"Inserted {result} rows")
```

## Your First Query

**See also:** [DataFrame basics examples](https://github.com/eddiethedean/moltres/blob/main/examples/02_dataframe_basics.py)

```python
from moltres import connect
from moltres import col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
], database=db).insert_into("users")

# Select all columns
df = db.table("users").select()
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', ...}, ...]

# Select specific columns
df = db.table("users").select("id", "name", "email")
results = df.collect()

# Filter rows
df = db.table("users").select().where(col("age") > 25)
results = df.collect()

```

## Understanding Lazy Evaluation

**Key Concept**: Moltres uses lazy evaluation. Operations build a query plan but don't execute until you call `.collect()`.

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
], database=db).insert_into("users")
# This doesn't execute any SQL yet!
df = (
    db.table("users")
    .select("name", "email")
    .where(col("active") == 1)
    .order_by(col("age").desc())
    .limit(10)
)

# SQL is compiled and executed here
results = df.collect()

```

This means you can build complex queries step by step without performance penalty.

## Common Operations

### Filtering

```python
from moltres import connect
from moltres import col
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Single condition
df = db.table("users").select().where(col("age") > 25)

# Multiple conditions (AND)
df = db.table("users").select().where(
    (col("age") > 25) & (col("active") == 1)
)

# OR conditions
df = db.table("users").select().where(
    (col("age") < 18) | (col("age") > 65)
)

# String operations
df = db.table("users").select().where(
    col("email").like("%@example.com")
)

```

### Sorting

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()
# Ascending (default)
df = db.table("users").select().order_by("age")

# Descending
df = db.table("users").select().order_by(col("age").desc())

# Multiple columns
df = db.table("users").select().order_by(
    col("active").desc(),
    col("age").asc()
)

```

### Aggregations

**See also:** [GroupBy and aggregation examples](https://github.com/eddiethedean/moltres/blob/main/examples/05_groupby.py)

```python
from moltres import connect
from moltres import col
from moltres.expressions import functions as F
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Simple aggregation
df = (
    db.table("users")
    .select()
    .group_by("active")
    .agg(F.count("*").alias("count"))
)

# Multiple aggregations
df = (
    db.table("users")
    .select()
    .group_by("active")
    .agg(
        F.count("*").alias("count"),
        F.avg(col("age")).alias("avg_age"),
        F.max(col("age")).alias("max_age")
    )
)

```

### Joins

**See also:** [Join examples](https://github.com/eddiethedean/moltres/blob/main/examples/04_joins.py)

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create a second table for demonstration
db.create_table("orders", [
    column("id", "INTEGER", primary_key=True),
    column("user_id", "INTEGER"),
    column("amount", "REAL"),
]).collect()

# Inner join
df = (
    db.table("users")
    .select()
    .join(
        db.table("orders").select(),
        on=[col("users.id") == col("orders.user_id")]
    )
)

# Left join
df = (
    db.table("users")
    .select()
    .join(
        db.table("orders").select(),
        on=[col("users.id") == col("orders.user_id")],
        how="left"
    )
)

```

## Working with Results

By default, `.collect()` returns a list of dictionaries:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
results = df.collect()
# [{'id': 1, 'name': 'Alice', ...}, {'id': 2, 'name': 'Bob', ...}]

```

You can also get results as pandas or polars DataFrames:

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Pandas DataFrame (requires: pip install moltres[pandas])
results = df.collect(format="pandas")
# Returns: pandas.DataFrame

# Polars DataFrame (requires: pip install moltres[polars])
results = df.collect(format="polars")
# Returns: polars.DataFrame

```

## Updating and Deleting Data

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Update rows
result = db.update(
    "users",
    where=col("active") == 0,
    set={"active": 1}
)
print(f"Updated {result} rows")

# Delete rows
result = db.delete(
    "users",
    where=col("age") < 18
)
print(f"Deleted {result} rows")

```

## Next Steps

- **Learn more**: Check out the [Common Patterns Guide](https://github.com/eddiethedean/moltres/blob/main/guides/05-common-patterns.md)
- **Optimize**: Read the [Performance Guide](https://github.com/eddiethedean/moltres/blob/main/guides/04-performance-optimization.md)
- **Migrate**: If coming from Pandas, see [Migration from Pandas](https://github.com/eddiethedean/moltres/blob/main/guides/02-migrating-from-pandas.md)
- **Examples**: Explore the [examples directory](https://github.com/eddiethedean/moltres/tree/main/examples) in the repository

## Common Pitfalls

1. **Forgetting `.collect()`**: Operations are lazy, so you must call `.collect()` to execute
2. **SQLite booleans**: SQLite uses `INTEGER` (0/1) instead of `BOOLEAN`
3. **Column references**: Use `col("column_name")` for expressions, or strings for simple column names
4. **Connection strings**: Make sure your connection string includes `://` separator

## Getting Help

- **Documentation**: See [docs directory](https://github.com/eddiethedean/moltres/tree/main/docs) for detailed documentation
- **Examples**: Check [examples directory](https://github.com/eddiethedean/moltres/tree/main/examples) for working code samples
- **Troubleshooting**: See [Troubleshooting Guide](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md) for common issues
- **FAQ**: Check [FAQ](https://github.com/eddiethedean/moltres/blob/main/docs/FAQ.md) for frequently asked questions

