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

**See also:** [Connection examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

```python
from moltres import connect

# SQLite in-memory (great for learning and testing - no file needed)
db = connect("sqlite:///:memory:")

# SQLite file-based (persistent database)
# db = connect("sqlite:///example.db")  # Creates a file on disk

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

# Output: Table 'users' created successfully!
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

# Output: Inserted 3 rows
```

## Your First Query

**See also:** [DataFrame basics examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

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
    column("email", "TEXT"),
    column("age", "INTEGER"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
], database=db).insert_into("users")

# Select all columns
df = db.table("users").select()
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', ...}, ...]

# Select specific columns
df = db.table("users").select("id", "name", "email")
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}, {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}]

# Filter rows
df = db.table("users").select().where(col("age") > 25)
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30, ...}, {'id': 3, 'name': 'Charlie', 'age': 35, ...}]

```

## Understanding Lazy Evaluation

**Key Concept**: Moltres uses lazy evaluation. Operations build a query plan but don't execute until you call `.collect()`.

```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30, "active": 1},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25, "active": 1},
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
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30, 'email': 'alice@example.com', 'active': 1}, {'id': 2, 'name': 'Bob', 'age': 25, 'email': 'bob@example.com', 'active': 1}]

```

This means you can build complex queries step by step without performance penalty.

## Common Operations

### Filtering

```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
    column("email", "TEXT"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "active": 1, "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "age": 25, "active": 0, "email": "bob@example.com"},
], database=db).insert_into("users")

# Single condition
df = db.table("users").select().where(col("age") > 25)
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}]

# Multiple conditions (AND)
df = db.table("users").select().where(
    (col("age") > 25) & (col("active") == 1)
)
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30, 'active': 1, 'email': 'alice@example.com'}]

# OR conditions
df = db.table("users").select().where(
    (col("age") < 18) | (col("age") > 65)
)
results = df.collect()
print(results)
# Output: []

# String operations
df = db.table("users").select().where(
    col("email").like("%@example.com")
)
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30, 'active': 1, 'email': 'alice@example.com'}, {'id': 2, 'name': 'Bob', 'age': 25, 'active': 1, 'email': 'bob@example.com'}]

```

### Sorting

```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
], database=db).insert_into("users")

# Ascending (default)
df = db.table("users").select().order_by("age")
results = df.collect()
print(results)
# Output: [{'id': 2, 'name': 'Bob', 'age': 25}, {'id': 1, 'name': 'Alice', 'age': 30}]

# Descending
df = db.table("users").select().order_by(col("age").desc())
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}]

# Multiple columns
df = db.table("users").select().order_by(
    col("age").desc(),
    col("name").asc()
)
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}]

```

### Aggregations

**See also:** [GroupBy and aggregation examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

```python
from moltres import connect, col
from moltres.expressions import functions as F
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "active": 1},
    {"id": 2, "name": "Bob", "age": 25, "active": 1},
], database=db).insert_into("users")

# Simple aggregation
df = (
    db.table("users")
    .select()
    .group_by("active")
    .agg(F.count("*").alias("count"))
)
results = df.collect()
print(results)
# Output: [{'active': 1, 'count': 2}]

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
results = df.collect()
print(results)
# Output: [{'active': 1, 'count': 2, 'avg_age': 27.5, 'max_age': 30}]

```

### Joins

**See also:** [Join examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

```python
from moltres import connect, col
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
# Create sample table and data
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
]).collect()
Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
], database=db).insert_into("users")

df = db.table("users").select()
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com'}, {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'}]

```

You can also get results as pandas or polars DataFrames:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
], database=db).insert_into("users")

df = db.table("users").select()

# Pandas DataFrame (requires: pip install pandas)
try:
    results = df.collect(format="pandas")
    # Returns: pandas.DataFrame
except ImportError:
    print("pandas not installed. Install with: pip install pandas or pip install moltres[pandas]")
# Returns: pandas.DataFrame

# Polars DataFrame (requires: pip install moltres[polars])
results = df.collect(format="polars")
# Returns: polars.DataFrame

```

## Updating and Deleting Data

```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("active", "INTEGER"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "active": 1},
    {"id": 2, "name": "Bob", "active": 0},
], database=db).insert_into("users")

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

## Debugging and UX Features

Moltres provides powerful tools to help you understand and debug your queries:

### Viewing Generated SQL

```python
from moltres import connect, col
from moltres.table.schema import column

db = connect("sqlite:///:memory:")
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]).collect()

df = db.table("users").select().where(col("age") > 25)

# Pretty-print SQL
df.show_sql()

# Get formatted SQL as string
sql = df.sql
print(sql)

# Get SQL preview
preview = df.sql_preview(max_length=100)
print(preview)
```

### Understanding Query Plans

```python
# Get structured plan summary
summary = df.plan_summary()
print(f"Operations: {summary['operations']}")
print(f"Table scans: {summary['table_scans']}")
print(f"Filters: {summary['filters']}")

# Visualize plan as ASCII tree
print(df.visualize_plan())

# Get database execution plan
plan = df.explain()
print(plan)
```

### Schema Discovery

```python
# Get schema for a table
schema = db.schema("users")
for col_def in schema:
    print(f"{col_def.name}: {col_def.type_name}")

# Get all tables with schemas
tables = db.tables()
for table_name, columns in tables.items():
    print(f"{table_name}: {len(columns)} columns")

# Get columns from table handle
handle = db.table("users")
print(handle.columns())  # ['id', 'name', 'age']
```

### Query Validation and Performance Hints

```python
# Validate query for common issues
issues = df.validate()
for issue in issues:
    print(f"[{issue['type']}] {issue['message']}")

# Get performance optimization hints
hints = df.performance_hints()
for hint in hints:
    print(f"- {hint}")
```

### Interactive Help

```python
# Display available operations
df.help()

# Get suggestions for next operations
suggestions = df.suggest_next()
for suggestion in suggestions:
    print(suggestion)
```

**See also:** [Debugging Guide](https://moltres.readthedocs.io/en/latest/DEBUGGING.html) for comprehensive debugging techniques

## Next Steps

- **Learn more**: Check out the [Common Patterns Guide](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)
- **Optimize**: Read the [Performance Guide](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)
- **Debug**: See the [Debugging Guide](https://moltres.readthedocs.io/en/latest/DEBUGGING.html) for query debugging tools
- **Migrate**: If coming from Pandas, see [Migration from Pandas](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pandas.html)
- **Examples**: Explore the [examples overview](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

## Common Pitfalls

1. **Forgetting `.collect()`**: Operations are lazy, so you must call `.collect()` to execute
2. **SQLite booleans**: SQLite uses `INTEGER` (0/1) instead of `BOOLEAN`
3. **Column references**: Use `col("column_name")` for expressions, or strings for simple column names
4. **Connection strings**: Make sure your connection string includes `://` separator

## Getting Help

- **Documentation**: See the [Moltres docs](https://moltres.readthedocs.io/en/latest/) for detailed documentation
- **Examples**: Check the [examples overview](https://moltres.readthedocs.io/en/latest/EXAMPLES.html) for working code samples
- **Troubleshooting**: See the [Troubleshooting Guide](https://moltres.readthedocs.io/en/latest/TROUBLESHOOTING.html) for common issues
- **FAQ**: Check the [FAQ](https://moltres.readthedocs.io/en/latest/FAQ.html) for frequently asked questions

