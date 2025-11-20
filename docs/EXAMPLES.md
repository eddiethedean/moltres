# Common Patterns and Examples

This document provides examples of common patterns when using Moltres.

## Basic Query Patterns

### Simple Select and Filter

```python
from moltres import col, connect

db = connect("sqlite:///example.db")

# Select all columns
df = db.table("users").select()
results = df.collect()

# Select specific columns
df = db.table("users").select("id", "name", "email")
results = df.collect()

# Filter rows
df = db.table("users").select().where(col("active") == True)
results = df.collect()

# Multiple conditions
df = (
    db.table("users")
    .select()
    .where((col("age") >= 18) & (col("active") == True))
)
results = df.collect()
```

### Sorting and Limiting

```python
# Sort by column
df = db.table("users").select().order_by(col("created_at").desc())
results = df.collect()

# Sort by multiple columns
df = (
    db.table("users")
    .select()
    .order_by(col("status"), col("created_at").desc())
)

# Limit results
df = db.table("users").select().limit(10)
results = df.collect()
```

## Aggregations

### Basic Aggregations

```python
from moltres.expressions.functions import sum, avg, count, min, max

# Single aggregation
df = (
    db.table("orders")
    .select()
    .group_by("customer_id")
    .agg(sum(col("amount")).alias("total"))
)

# Multiple aggregations
df = (
    db.table("orders")
    .select()
    .group_by("customer_id")
    .agg(
        sum(col("amount")).alias("total"),
        avg(col("amount")).alias("average"),
        count("*").alias("order_count"),
    )
)
```

### Grouping by Multiple Columns

```python
df = (
    db.table("sales")
    .select()
    .group_by("region", "product_category")
    .agg(
        sum(col("revenue")).alias("total_revenue"),
        count("*").alias("transaction_count"),
    )
)
```

## Joins

### Inner Join

```python
orders_df = db.table("orders").select()
customers_df = db.table("customers").select()

df = orders_df.join(
    customers_df,
    on=[("customer_id", "id")]
).select(
    col("orders.id").alias("order_id"),
    col("customers.name").alias("customer_name"),
    col("orders.amount"),
)
```

### Left Join

```python
df = orders_df.join(
    customers_df,
    on=[("customer_id", "id")],
    how="left"
)
```

### Multiple Join Conditions

```python
df = orders_df.join(
    customers_df,
    on=[
        ("customer_id", "id"),
        ("region", "region"),  # Additional join condition
    ]
)
```

### Custom Join Condition

```python
from moltres import col

df = orders_df.join(
    customers_df,
    condition=(col("orders.customer_id") == col("customers.id")) &
              (col("orders.status") == "active")
)
```

## Complex Queries

### Subqueries (via CTEs - when supported)

```python
# For now, use multiple DataFrames
# First query
top_customers = (
    db.table("orders")
    .select()
    .group_by("customer_id")
    .agg(sum(col("amount")).alias("total"))
    .order_by(col("total").desc())
    .limit(10)
)

# Use results in second query
customer_ids = [row["customer_id"] for row in top_customers.collect()]
df = (
    db.table("customers")
    .select()
    .where(col("id").isin(customer_ids))
)
```

### Window Functions (when supported)

```python
from moltres.expressions.functions import avg

# Running average
df = (
    db.table("sales")
    .select(
        col("date"),
        col("amount"),
        avg(col("amount")).over().alias("running_avg")
    )
)
```

## Data Mutations

### Insert Data

```python
# Single row
table = db.table("users")
table.insert([{"name": "Alice", "email": "alice@example.com"}])

# Multiple rows (batch insert)
table.insert([
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"},
    {"name": "Charlie", "email": "charlie@example.com"},
])
```

### Update Data

```python
from moltres import col

# Update single column
table.update(
    where=col("id") == 1,
    set={"name": "Alice Updated"}
)

# Update multiple columns
table.update(
    where=col("status") == "pending",
    set={
        "status": "processed",
        "processed_at": "2024-01-01 12:00:00"
    }
)
```

### Delete Data

```python
# Delete with condition
table.delete(where=col("status") == "deleted")

# Delete with complex condition
table.delete(
    where=(col("status") == "deleted") & (col("deleted_at") < "2024-01-01")
)
```

## File Operations

### Reading Files

```python
# CSV
df = db.read.csv("data.csv")
df = db.read.option("delimiter", "|").csv("pipe_delimited.csv")
df = db.read.option("header", False).csv("no_header.csv")

# JSON
df = db.read.json("data.json")  # Array of objects
df = db.read.jsonl("data.jsonl")  # One object per line

# Parquet (requires pandas and pyarrow)
df = db.read.parquet("data.parquet")

# Text file
df = db.read.text("log.txt", column_name="line")
```

### Writing Files

```python
df = db.table("users").select()

# CSV
df.write.csv("output.csv")
df.write.option("delimiter", "|").csv("output.csv")

# JSON
df.write.json("output.json")
df.write.jsonl("output.jsonl")

# Parquet
df.write.parquet("output.parquet")
```

## Streaming for Large Datasets

### Streaming Reads

```python
# Read large file in chunks
df = db.read.stream().option("chunk_size", 10000).csv("large_file.csv")

for chunk in df.collect(stream=True):
    process_chunk(chunk)  # Process 10,000 rows at a time
```

### Streaming Writes

```python
# Write large dataset in chunks
df.write.stream().mode("overwrite").save_as_table("large_table")

# Stream to file
df.write.stream().csv("output.csv")
```

### Streaming SQL Queries

```python
# Process query results in chunks
df = db.table("large_table").select()
for chunk in df.collect(stream=True):
    process_chunk(chunk)
```

## Table Management

### Create Table

```python
from moltres import column, connect

db = connect("sqlite:///example.db")

table = db.create_table(
    "users",
    [
        column("id", "INTEGER", nullable=False, primary_key=True),
        column("name", "TEXT", nullable=False),
        column("email", "TEXT", nullable=True),
        column("created_at", "TIMESTAMP", default="CURRENT_TIMESTAMP"),
    ],
)
```

### Drop Table

```python
db.drop_table("users", if_exists=True)
```

## Working with Results

### Default Format (List of Dicts)

```python
db = connect("sqlite:///example.db")
results = df.collect()  # List[Dict[str, Any]]

for row in results:
    print(row["name"], row["email"])
```

### Pandas Format

```python
db = connect("sqlite:///example.db", fetch_format="pandas")
df_result = df.collect()  # pandas.DataFrame

print(df_result.head())
print(df_result.describe())
```

### Polars Format

```python
db = connect("sqlite:///example.db", fetch_format="polars")
df_result = df.collect()  # polars.DataFrame

print(df_result.head())
print(df_result.describe())
```

## Error Handling

### Handling Validation Errors

```python
from moltres.utils.exceptions import ValidationError

try:
    db.table("")  # Empty table name
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Handling Execution Errors

```python
from moltres.utils.exceptions import ExecutionError

try:
    df.collect()
except ExecutionError as e:
    print(f"Query failed: {e}")
```

## Performance Tips

### Use Batch Operations

```python
# ✅ Good: Batch insert
table.insert([row1, row2, ..., row1000])

# ❌ Avoid: Individual inserts in loop
for row in rows:
    table.insert([row])  # Much slower
```

### Use Streaming for Large Data

```python
# ✅ Good: Stream large files
df = db.read.stream().csv("large.csv")
for chunk in df.collect(stream=True):
    process(chunk)

# ❌ Avoid: Loading entire file into memory
df = db.read.csv("large.csv")  # May cause memory issues
all_data = df.collect()  # Loads everything
```

### Optimize Connection Pooling

```python
# For high-concurrency applications
db = connect(
    "postgresql://...",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections
    pool_recycle=3600,    # Recycle after 1 hour
)
```

## Common Patterns

### ETL Pipeline

```python
# Extract
raw_data = db.read.csv("raw_data.csv")

# Transform
cleaned = (
    raw_data
    .select(
        col("id"),
        col("name").upper().alias("name"),
        col("amount").cast("REAL"),
    )
    .where(col("amount") > 0)
    .group_by("category")
    .agg(sum(col("amount")).alias("total"))
)

# Load
cleaned.write.mode("overwrite").save_as_table("summary")
```

### Data Validation

```python
# Check for duplicates
duplicates = (
    db.table("users")
    .select()
    .group_by("email")
    .agg(count("*").alias("count"))
    .where(col("count") > 1)
)

if len(duplicates.collect()) > 0:
    print("Found duplicate emails!")
```

### Data Cleaning

```python
# Remove null values
cleaned = (
    db.table("users")
    .select()
    .where(col("email").is_not_null())
    .where(col("name").is_not_null())
)

# Update null values
db.table("users").update(
    where=col("status").is_null(),
    set={"status": "unknown"}
)
```

