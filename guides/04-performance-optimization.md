# Performance Optimization Guide

Learn how to write efficient Moltres queries and optimize performance.

## Understanding Moltres Performance

Moltres compiles DataFrame operations into SQL and executes them on your database. Performance depends on:

1. **Database optimizer**: Your database's query planner
2. **Query structure**: How operations are composed
3. **Indexes**: Database indexes on columns used in filters/joins
4. **Connection pooling**: Efficient connection management
5. **Result size**: Amount of data returned

## Best Practices

### 1. Use Indexes

Create indexes on columns used in WHERE clauses, JOINs, and ORDER BY:

```python
from moltres import connect

db = connect("postgresql://user:pass@localhost/mydb")

# Create index on frequently filtered column
db.create_index("idx_user_email", "users", "email").collect()

# Create composite index for multi-column queries
db.create_index(
    "idx_user_country_active",
    "users",
    ["country", "active"]
).collect()
```

**When to index:**
- Columns in WHERE clauses
- JOIN keys
- Columns in ORDER BY
- Columns in GROUP BY (sometimes)

### 2. Filter Early

Apply filters as early as possible in your query chain:

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
# ❌ Bad: Filters large dataset after join
df = (
    db.table("orders")
    .select()
    .join(db.table("users").select(), on=[...])
    .where(col("orders.amount") > 1000)  # Filter after join
)

# ✅ Good: Filter before join
df = (
    db.table("orders")
    .select()
    .where(col("amount") > 1000)  # Filter early
    .join(db.table("users").select(), on=[...])
)

```

### 3. Select Only Needed Columns

Avoid selecting all columns when you only need a few:

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
# ❌ Bad: Selects all columns
df = db.table("users").select()  # Selects all columns
results = df.collect()

# ✅ Good: Select only needed columns
df = db.table("users").select("id", "name", "email")
results = df.collect()

```

### 4. Use LIMIT for Exploration

When exploring data, use LIMIT to avoid loading large result sets:

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
# ✅ Good: Limit results during exploration
df = (
    db.table("users")
    .select()
    .where(col("age") > 25)
    .limit(100)  # Only get first 100 rows
)
results = df.collect()

```

### 5. Optimize Joins

**Use appropriate join types:**

```python
# ✅ Use INNER JOIN when possible (faster than OUTER)
df = df1.join(df2, on=[...], how="inner")

# ✅ Filter before joining (reduces join size)
df1_filtered = df1.where(col("active") == 1)
df = df1_filtered.join(df2, on=[...])
```

**Ensure join keys are indexed:**

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Create indexes on join keys
db.create_index("idx_orders_user_id", "orders", "user_id").collect()
db.create_index("idx_users_id", "users", "id").collect()

```

### 6. Use Aggregations Efficiently

Push aggregations to the database level:

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
# ✅ Good: Aggregation happens in SQL
df = (
    db.table("orders")
    .select()
    .group_by("user_id")
    .agg(F.sum(col("amount")).alias("total"))
    .limit(100)
)
results = df.collect()  # Only 100 aggregated rows returned

# ❌ Bad: Would load all data, then aggregate in Python
# (Moltres doesn't do this, but be aware of the pattern)

```

### 7. Connection Pooling

Configure connection pooling for better performance:

```python
db = connect(
    "postgresql://user:pass@localhost/mydb",
    pool_size=10,        # Number of connections to maintain
    max_overflow=20,     # Additional connections allowed
    pool_timeout=30,     # Timeout for getting connection
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_pre_ping=True   # Verify connections before use
)
```

### 8. Use Streaming for Large Results

For large result sets, use streaming to process in chunks:

```python
# ✅ Good: Stream large results
async def process_large_dataset():
    db = await async_connect("postgresql+asyncpg://...")
    df = db.table("large_table").select()
    
    async for chunk in await df.collect(stream=True):
        # Process chunk (e.g., 1000 rows at a time)
        process_chunk(chunk)
    
    await db.close()
```

### 9. Batch Operations

For INSERT/UPDATE/DELETE, operations are automatically batched, but you can optimize:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# ✅ Good: Single batch insert
records = Records.from_list(large_list, database=db)
result = records.insert_into("users")  # Automatically batched

# ✅ Good: Batch updates
result = db.update(
    "users",
    where=col("status") == "pending",
    set={"status": "active"}
)  # Single SQL statement, updates all matching rows

```

### 10. Avoid N+1 Queries

Don't call `.collect()` in loops:

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
# ❌ Bad: N+1 queries
user_ids = [1, 2, 3, 4, 5]
results = []
for user_id in user_ids:
    df = db.table("orders").select().where(col("user_id") == user_id)
    results.append(df.collect())  # One query per iteration!

# ✅ Good: Single query with IN clause
df = db.table("orders").select().where(col("user_id").isin(user_ids))
results = df.collect()  # Single query

```

## Database-Specific Optimizations

### PostgreSQL

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Use EXPLAIN ANALYZE to understand query plans
plan = df.explain(analyze=True)
print(plan)

# Use PostgreSQL-specific features
# - JSONB for JSON data
# - Array types for arrays
# - Full-text search indexes

```

### MySQL

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Use EXPLAIN to understand query plans
plan = df.explain()
print(plan)

# MySQL-specific optimizations
# - Use InnoDB for transactions
# - Configure buffer pool size
# - Use covering indexes

```

### SQLite

```python
# SQLite is single-threaded, but very fast for small-medium datasets
# Use WAL mode for better concurrency
# db = connect("sqlite:///example.db?mode=rwc")

# Create indexes (SQLite doesn't auto-index foreign keys)
db.create_index("idx_orders_user_id", "orders", "user_id").collect()
```

## Monitoring Performance

### 1. Enable SQL Logging

```python
db = connect(
    "postgresql://user:pass@localhost/mydb",
    echo=True  # Log all SQL statements
)
```

### 2. Use Performance Hooks

```python
from moltres.engine import register_performance_hook

def log_query(sql: str, elapsed: float, metadata: dict):
    if elapsed > 1.0:  # Log slow queries
        print(f"Slow query ({elapsed:.2f}s): {sql[:200]}")

register_performance_hook("query_end", log_query)
```

### 3. Check Query Plans

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Get estimated plan
plan = df.explain()
print(plan)

# Get actual execution stats (PostgreSQL)
plan = df.explain(analyze=True)
print(plan)

```

## Common Performance Pitfalls

### 1. Loading Entire Tables

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
# ❌ Bad: Loads entire table
df = db.table("huge_table").select()
results = df.collect()  # May be millions of rows!

# ✅ Good: Filter first
df = db.table("huge_table").select().where(col("date") >= "2024-01-01")
results = df.collect()

```

### 2. Multiple Collect Calls

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
# ❌ Bad: Multiple queries
df = db.table("users").select()
count = len(df.collect())  # Query 1
filtered = df.where(col("age") > 25).collect()  # Query 2

# ✅ Good: Single query with aggregation
df = db.table("users").select()
count_df = df.select(F.count("*").alias("count"))
count = count_df.collect()[0]["count"]
filtered = df.where(col("age") > 25).collect()

```

### 3. Unnecessary Joins

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
# ❌ Bad: Join when not needed
df = (
    db.table("orders")
    .select()
    .join(db.table("users").select(), on=[...])
    .select("orders.id", "orders.amount")  # Only using orders columns
)

# ✅ Good: No join needed
df = db.table("orders").select("id", "amount")

```

### 4. Over-aggregation

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
# ❌ Bad: Aggregating then filtering
df = (
    db.table("orders")
    .select()
    .group_by("user_id")
    .agg(F.sum(col("amount")).alias("total"))
)
results = [r for r in df.collect() if r["total"] > 1000]  # Filter in Python

# ✅ Good: Filter in SQL using HAVING
df = (
    db.table("orders")
    .select()
    .group_by("user_id")
    .agg(F.sum(col("amount")).alias("total"))
    .having(col("total") > 1000)
)
results = df.collect()

```

## Performance Testing

### Benchmark Your Queries

```python
from moltres import connect
import time
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

def benchmark_query(df):
    start = time.perf_counter()
    results = df.collect()
    elapsed = time.perf_counter() - start
    print(f"Query took {elapsed:.2f}s, returned {len(results)} rows")
    return results

# Test different approaches
df1 = db.table("users").select().where(col("age") > 25)
df2 = db.table("users").select("id", "name").where(col("age") > 25)

results1 = benchmark_query(df1)
results2 = benchmark_query(df2)

```

## Next Steps

- **Patterns**: See [Common Patterns Guide](https://github.com/eddiethedean/moltres/blob/main/guides/05-common-patterns.md) for optimized patterns
- **Best Practices**: Read [Best Practices Guide](https://github.com/eddiethedean/moltres/blob/main/guides/08-best-practices.md)
- **Troubleshooting**: Check [Troubleshooting Guide](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md) for performance issues

