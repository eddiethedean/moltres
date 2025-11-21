# Common Patterns and Examples

This document provides examples of common patterns when using Moltres.

## Showcase: Memory-Efficient Operations

Moltres executes all DataFrame operations directly in SQL—no data loading into memory. This example demonstrates transforming millions of rows without materialization.

### Transform Large Tables Without Loading Data

```python
from moltres import col, connect
from moltres.expressions.functions import sum, avg, count

db = connect("postgresql://user:pass@localhost/warehouse")

# Transform a table with 10M+ rows
# With Pandas: Would require loading all data into memory (could be GBs)
# With Moltres: All operations execute in SQL, zero memory usage for data

df = (
    db.table("sales")  # 10M+ rows
    .select(
        col("product_id"),
        col("region"),
        (col("price") * col("quantity")).alias("revenue"),
        col("date"),
    )
    .where(col("date") >= "2024-01-01")
    .where(col("status") == "completed")
    .group_by("product_id", "region")
    .agg(
        sum(col("revenue")).alias("total_revenue"),
        avg(col("price")).alias("avg_price"),
        count("*").alias("transaction_count"),
    )
    .order_by(col("total_revenue").desc())
    .limit(100)
)

# This compiles to a single SQL query and executes on the database
# No data is loaded into Python memory—everything happens in SQL
results = df.collect()

# Compare with Pandas approach:
# import pandas as pd
# df_pandas = pd.read_sql("SELECT * FROM sales", connection)  # Loads 10M rows into memory!
# df_pandas = df_pandas[df_pandas["date"] >= "2024-01-01"]  # More memory operations
# # ... more in-memory transformations
```

### View the Generated SQL

Enable SQL logging to see exactly what SQL is generated:

```python
# Enable SQL logging to see the generated query
db = connect("postgresql://user:pass@localhost/warehouse", echo=True)

df = (
    db.table("sales")
    .select()
    .where(col("date") >= "2024-01-01")
    .group_by("product_id")
    .agg(sum(col("amount")).alias("total"))
)

# When you call collect(), you'll see the SQL in the console:
# SELECT product_id, SUM(amount) AS total 
# FROM sales 
# WHERE date >= :date_1 
# GROUP BY product_id
results = df.collect()
```

## Showcase: CRUD + DataFrame Workflow

Complete example showing how DataFrame operations chain seamlessly with CRUD operations—all executed in SQL.

### Query → Transform → Update Workflow

```python
from moltres import col, connect
from moltres.expressions.functions import sum, avg

db = connect("postgresql://user:pass@localhost/warehouse")
customers = db.table("customers")
orders = db.table("orders")

# Step 1: Query and analyze using DataFrame operations (executes in SQL)
customer_stats = (
    orders.select()
    .join(customers.select(), on=[("customer_id", "id")])
    .where(col("orders.date") >= "2024-01-01")
    .group_by("customers.id", "customers.name")
    .agg(
        sum(col("orders.amount")).alias("total_spent"),
        avg(col("orders.amount")).alias("avg_order_value"),
    )
    .where(col("total_spent") > 1000)
)

# Step 2: Get results (SQL executed here, no data materialized)
top_customers = customer_stats.collect()

# Step 3: Update based on analysis (executes UPDATE SQL directly)
# Mark high-value customers as VIP
for customer in top_customers:
    customers.update(
        where=col("id") == customer["customers.id"],
        set={"tier": "VIP", "updated_at": "2024-01-15"}
    )

# Or update all at once using a subquery approach
# (This would require a more complex UPDATE with subquery, 
#  but demonstrates the workflow)
```

### Bulk Update Based on DataFrame Query

```python
# Find customers who haven't ordered in 90 days
inactive_customers = (
    customers.select()
    .join(
        orders.select()
        .group_by("customer_id")
        .agg(max(col("date")).alias("last_order_date")),
        on=[("id", "customer_id")],
        how="left"
    )
    .where(
        (col("last_order_date") < "2023-10-15") | 
        col("last_order_date").is_null()
    )
)

# Get customer IDs
inactive_ids = [row["id"] for row in inactive_customers.collect()]

# Update all inactive customers in one operation
customers.update(
    where=col("id").isin(inactive_ids),
    set={"status": "inactive", "inactive_since": "2024-01-15"}
)

# Delete old inactive customers (executes DELETE SQL)
customers.delete(
    where=(col("status") == "inactive") & 
          (col("inactive_since") < "2023-01-01")
)
```

## Async Examples

### Basic Async Query

```python
import asyncio
from moltres import async_connect, col

async def main():
    db = async_connect("sqlite+aiosqlite:///example.db")
    
    # For SQL operations, use db.table().select()
    table_handle = await db.table("users")
    df = table_handle.select()
    results = await df.collect()
    
    print(results)
    await db.close()

asyncio.run(main())
```

### Async File Reading

```python
import asyncio
from moltres import async_connect

async def main():
    db = async_connect("sqlite+aiosqlite:///example.db")
    
    # Load CSV asynchronously (returns AsyncRecords)
    records = await db.load.csv("data.csv")
    rows = await records.rows()  # Get all rows
    
    # Stream large files (returns AsyncRecords)
    records_stream = await db.load.stream().csv("large_file.csv")
    async for row in records_stream:
        process(row)
    
    await db.close()

asyncio.run(main())
```

### Async Mutations

```python
import asyncio
from moltres import async_connect, col

async def main():
    db = async_connect("sqlite+aiosqlite:///example.db")
    
    table = await db.table("users")
    
    # Insert rows
    await table.insert([
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ])
    
    # Update rows
    await table.update(
        where=col("age") < 30,
        set={"status": "young"},
    )
    
    # Delete rows
    await table.delete(where=col("age") > 100)
    
    await db.close()

asyncio.run(main())
```

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

### Loading Files

File readers return `Records`, not `DataFrame`. Records are materialized data that can be inserted into tables or iterated.

```python
# CSV - returns Records
records = db.load.csv("data.csv")
records = db.load.option("delimiter", "|").csv("pipe_delimited.csv")
records = db.load.option("header", False).csv("no_header.csv")

# JSON - returns Records
records = db.load.json("data.json")  # Array of objects
records = db.load.jsonl("data.jsonl")  # One object per line

# Parquet (requires pandas and pyarrow) - returns Records
records = db.load.parquet("data.parquet")

# Text file - returns Records
records = db.load.text("log.txt", column_name="line")

# Use Records with insert operations
table.insert(records)  # Records implements Sequence protocol
# Or use convenience method
records.insert_into("table_name")

# Access data
rows = records.rows()  # Get all rows as a list
for row in records:  # Iterate directly
    process(row)
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
# Load large file in streaming mode (returns Records)
records = db.load.stream().option("chunk_size", 10000).csv("large_file.csv")

# Records iterate row-by-row (streaming happens internally)
for row in records:
    process(row)  # Process one row at a time

# Or materialize all at once
all_rows = records.rows()  # Materializes all data
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
records = db.load.stream().csv("large.csv")
for row in records:
    process(row)  # Processes row-by-row, streaming internally

# ❌ Avoid: Loading entire file into memory at once
records = db.load.csv("large.csv")  # Materializes all data
all_data = records.rows()  # Loads everything into memory
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
# Extract: Load from CSV (returns Records)
raw_records = db.load.csv("raw_data.csv")

# Load raw data into staging table first
db.create_table("staging", [
    column("id", "INTEGER"),
    column("name", "TEXT"),
    column("amount", "REAL"),
    column("category", "TEXT"),
])
raw_records.insert_into("staging")

# Transform: Use SQL operations on the table
cleaned = (
    db.table("staging")
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

## Use Cases by Audience

### For Data Engineers

#### Update Millions of Rows Without Loading into Memory

```python
# Scenario: Update status for millions of orders based on complex conditions
# With Pandas: Would require loading all data, filtering, then updating
# With Moltres: Single UPDATE SQL statement, zero memory usage

orders = db.table("orders")

# Update based on complex DataFrame query logic
# All executed in SQL—no data materialization
orders.update(
    where=(
        (col("status") == "pending") & 
        (col("created_at") < "2024-01-01") &
        (col("payment_confirmed") == True)
    ),
    set={
        "status": "processing",
        "updated_at": "2024-01-15",
        "processed_by": "batch_job"
    }
)

# Verify the update with a DataFrame query (also executes in SQL)
updated_count = (
    orders.select()
    .where(col("status") == "processing")
    .where(col("updated_at") == "2024-01-15")
    .agg(count("*").alias("count"))
    .collect()
)
print(f"Updated {updated_count[0]['count']} orders")
```

#### ETL Pipeline Without Materialization

```python
# Complete ETL pipeline where intermediate results never materialize
# All transformations happen in SQL

# Extract: Load from file (only this step materializes data)
raw_records = db.load.csv("raw_sales.csv")

# Load into staging (batch insert, efficient)
db.create_table("staging_sales", [
    column("order_id", "INTEGER"),
    column("product", "TEXT"),
    column("amount", "REAL"),
    column("date", "DATE"),
    column("region", "TEXT"),
])
raw_records.insert_into("staging_sales")

# Transform: All operations execute in SQL, no intermediate materialization
cleaned = (
    db.table("staging_sales")
    .select(
        col("order_id"),
        col("product").upper().alias("product"),
        col("amount").cast("REAL"),
        col("date"),
        col("region"),
    )
    .where(col("amount") > 0)
    .where(col("date").is_not_null())
)

# Aggregate (still in SQL)
aggregated = (
    cleaned
    .group_by("product", "region", "date")
    .agg(
        sum(col("amount")).alias("daily_revenue"),
        count("*").alias("transaction_count"),
    )
)

# Load: Write to final table (SQL INSERT)
aggregated.write.mode("overwrite").save_as_table("daily_sales_summary")

# Cleanup staging (SQL DELETE)
db.table("staging_sales").delete(where=col("date") < "2024-01-01")
```

### For Backend Developers

#### Replace ORM Operations with DataFrame CRUD

```python
# Instead of ORM-style row-by-row operations, use DataFrame-style bulk operations

users = db.table("users")

# Traditional ORM approach (slow, many queries):
# for user in User.query.filter_by(status='pending'):
#     user.status = 'active'
#     user.updated_at = datetime.now()
#     db.session.commit()

# Moltres approach (single UPDATE SQL statement):
users.update(
    where=col("status") == "pending",
    set={
        "status": "active",
        "updated_at": "2024-01-15 10:00:00"
    }
)

# Bulk insert instead of loop
new_users = [
    {"name": "Alice", "email": "alice@example.com", "role": "user"},
    {"name": "Bob", "email": "bob@example.com", "role": "admin"},
    {"name": "Charlie", "email": "charlie@example.com", "role": "user"},
]
users.insert(new_users)  # Single batch INSERT

# Column-aware updates based on queries
# Find users who need role updates
users_needing_update = (
    users.select()
    .join(db.table("permissions").select(), on=[("id", "user_id")])
    .where(col("permissions.level") == "admin")
    .where(col("users.role") != "admin")
)

# Update all at once
user_ids = [row["id"] for row in users_needing_update.collect()]
users.update(
    where=col("id").isin(user_ids),
    set={"role": "admin", "role_updated_at": "2024-01-15"}
)
```

#### Type-Safe CRUD Without Hand-Writing SQL

```python
# Moltres provides validated, type-safe CRUD operations
# No need to hand-write SQL strings

orders = db.table("orders")

# Insert with validation (table schema is checked)
orders.insert([
    {
        "customer_id": 123,
        "product_id": 456,
        "amount": 99.99,
        "status": "pending"
    }
])

# Update with column expressions (type-safe, validated)
orders.update(
    where=col("status") == "pending",
    set={
        "status": "confirmed",
        "confirmed_at": "2024-01-15 10:00:00"
    }
)

# Delete with complex conditions (compiled to safe SQL)
orders.delete(
    where=(
        (col("status") == "cancelled") & 
        (col("cancelled_at") < "2023-01-01")
    )
)
```

### For Analytics Engineers / dbt Users

#### Express SQL Models in Python with DataFrame Chaining

```python
# Build analytics models using DataFrame chaining
# Similar to dbt, but in Python with full type safety

# Base model: Clean raw data
staging_orders = (
    db.table("raw_orders")
    .select(
        col("order_id"),
        col("customer_id"),
        col("product_id"),
        col("amount").cast("REAL").alias("amount"),
        col("order_date").cast("DATE").alias("order_date"),
    )
    .where(col("amount") > 0)
    .where(col("order_date").is_not_null())
)

# Intermediate model: Customer metrics
customer_metrics = (
    staging_orders
    .group_by("customer_id")
    .agg(
        sum(col("amount")).alias("lifetime_value"),
        count("*").alias("order_count"),
        min(col("order_date")).alias("first_order_date"),
        max(col("order_date")).alias("last_order_date"),
    )
)

# Final model: Customer segments
customer_segments = (
    customer_metrics
    .select(
        col("customer_id"),
        col("lifetime_value"),
        col("order_count"),
        when(col("lifetime_value") > 1000, "high_value")
        .when(col("lifetime_value") > 500, "medium_value")
        .otherwise("low_value")
        .alias("segment"),
    )
)

# Materialize the final model
customer_segments.write.mode("overwrite").save_as_table("customer_segments")

# Use the model in downstream analysis
segment_analysis = (
    db.table("customer_segments")
    .select()
    .group_by("segment")
    .agg(
        count("*").alias("customer_count"),
        avg(col("lifetime_value")).alias("avg_lifetime_value"),
    )
    .order_by(col("avg_lifetime_value").desc())
)
```

#### dbt-like Transformations with Composable Operations

```python
# Build reusable transformation functions
def clean_orders(source_table: str):
    """Clean and standardize order data."""
    return (
        db.table(source_table)
        .select(
            col("order_id"),
            col("customer_id"),
            col("amount").cast("REAL"),
            col("order_date").cast("DATE"),
        )
        .where(col("amount") > 0)
    )

def calculate_metrics(orders_df):
    """Calculate customer metrics."""
    return (
        orders_df
        .group_by("customer_id")
        .agg(
            sum(col("amount")).alias("total_spent"),
            avg(col("amount")).alias("avg_order_value"),
            count("*").alias("order_count"),
        )
    )

# Compose transformations
clean_orders_df = clean_orders("raw_orders")
metrics_df = calculate_metrics(clean_orders_df)

# Save intermediate results (optional)
clean_orders_df.write.mode("overwrite").save_as_table("staging_orders")
metrics_df.write.mode("overwrite").save_as_table("customer_metrics")
```

### For Product Engineers

#### Building Data Features with DataFrame API

```python
# Build data features for ML or analytics using DataFrame operations

# Feature: User engagement score
engagement_features = (
    db.table("users")
    .select()
    .join(
        db.table("events").select()
        .group_by("user_id")
        .agg(
            count("*").alias("event_count"),
            max(col("event_date")).alias("last_active_date"),
        ),
        on=[("id", "user_id")],
        how="left"
    )
    .select(
        col("users.id"),
        col("users.created_at"),
        col("event_count"),
        col("last_active_date"),
        # Calculate days since last active
        (col("last_active_date") - col("users.created_at")).alias("account_age_days"),
    )
)

# Update user table with engagement scores
engagement_data = engagement_features.collect()
for user in engagement_data:
    score = calculate_engagement_score(
        user["event_count"],
        user["account_age_days"]
    )
    db.table("users").update(
        where=col("id") == user["id"],
        set={"engagement_score": score, "score_updated_at": "2024-01-15"}
    )
```

### For Teams Migrating from Spark

#### Spark-like DataFrame API for Traditional SQL Databases

```python
# Moltres provides a Spark-like API but works with existing SQL infrastructure
# No cluster required—works with PostgreSQL, MySQL, SQLite, etc.

# Familiar Spark-style operations
df = (
    db.table("orders")
    .select()  # Like df.select()
    .where(col("status") == "completed")  # Like df.filter()
    .join(
        db.table("customers").select(),
        on=[("customer_id", "id")]  # Like df.join()
    )
    .group_by("customers.country")  # Like df.groupBy()
    .agg(
        sum(col("amount")).alias("total"),  # Like df.agg()
        avg(col("amount")).alias("avg"),
    )
    .order_by(col("total").desc())  # Like df.orderBy()
    .limit(10)  # Like df.limit()
)

# Execute (like df.collect() in Spark)
results = df.collect()

# Write to table (like df.write.saveAsTable() in Spark)
df.write.mode("overwrite").save_as_table("country_summary")

# Key difference: Everything executes in SQL on your existing database
# No need for Spark cluster, YARN, or distributed infrastructure
```

