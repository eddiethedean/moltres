# Performance Tuning Guide

This guide covers performance optimization strategies for Moltres queries and best practices for achieving optimal performance.

## Overview

Moltres compiles DataFrame operations to SQL and executes them in your database. Performance depends on:

1. **Database engine** - The underlying SQL database's optimization capabilities
2. **Query structure** - How operations are composed and compiled
3. **Indexes** - Database indexes on frequently queried columns
4. **Connection pooling** - Efficient connection management
5. **Query patterns** - How queries are structured and executed

## SQL Pushdown

Moltres pushes operations down to SQL whenever possible, which means:

- **Filtering** happens in the database: `df.where(col("age") > 18)` → `WHERE age > 18`
- **Aggregations** execute in the database: `df.group_by("category").agg(sum("amount"))` → `GROUP BY category, SUM(amount)`
- **Joins** are compiled to SQL JOINs: `df1.join(df2, on="id")` → `INNER JOIN ... ON ...`
- **Sorting** uses database ORDER BY: `df.order_by(col("name"))` → `ORDER BY name`

This means most operations are executed efficiently in the database, not in Python.

## Performance Best Practices

### 1. Use Indexes

Create indexes on columns used in:
- **WHERE clauses** (filtering)
- **JOIN conditions**
- **ORDER BY** clauses
- **GROUP BY** columns

```python
# Create index for frequently filtered column
db.execute("CREATE INDEX idx_users_age ON users(age)")

# Create index for join column
db.execute("CREATE INDEX idx_orders_customer_id ON orders(customer_id)")

# Composite index for multiple columns
db.execute("CREATE INDEX idx_sales_region_date ON sales(region, date)")
```

### 2. Filter Early

Apply filters as early as possible to reduce data scanned:

```python
# Good: Filter before join
df = (
    db.table("orders")
    .select()
    .where(col("status") == "active")  # Filter early
    .join(customers, on=[("customer_id", "id")])
)

# Less efficient: Join before filter
df = (
    db.table("orders")
    .select()
    .join(customers, on=[("customer_id", "id")])
    .where(col("status") == "active")  # Filter after join
)
```

### 3. Select Only Needed Columns

Avoid `SELECT *` when you only need specific columns:

```python
# Good: Select only needed columns
df = db.table("users").select("id", "name", "email")

# Less efficient: Select all columns
df = db.table("users").select()  # SELECT *
```

### 4. Use LIMIT for Exploratory Queries

When exploring data, use `limit()` to avoid fetching large result sets:

```python
# Good: Limit results during exploration
df = db.table("orders").select().limit(100)

# Avoid: Fetching all rows
df = db.table("orders").select()  # May return millions of rows
```

### 5. Optimize Joins

- Use appropriate join types (INNER vs LEFT)
- Ensure join columns are indexed
- Consider join order for complex queries

```python
# Ensure join columns are indexed
db.execute("CREATE INDEX idx_orders_customer_id ON orders(customer_id)")
db.execute("CREATE INDEX idx_customers_id ON customers(id)")

# Use appropriate join type
df = customers.join(orders, on=[("id", "customer_id")], how="inner")
```

### 6. Use Aggregations Efficiently

Group by indexed columns when possible:

```python
# Create index on grouping column
db.execute("CREATE INDEX idx_orders_category ON orders(category)")

# Group by indexed column
df = (
    db.table("orders")
    .select()
    .group_by("category")
    .agg(sum(col("amount")).alias("total"))
)
```

### 7. Connection Pooling

Configure connection pooling for better performance:

```python
db = connect(
    "postgresql://user:pass@host/dbname",
    pool_size=10,        # Number of connections to maintain
    max_overflow=5,      # Additional connections allowed
    pool_timeout=30,     # Timeout for getting connection
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connections before use
)
```

### 8. Use Streaming for Large Results

For large result sets, use streaming to avoid loading everything into memory:

```python
# Stream results instead of loading all at once
for chunk in df.collect(stream=True):
    process_chunk(chunk)
```

### 9. Batch Operations

Use batch inserts/updates when modifying data:

```python
# Batch insert is more efficient than individual inserts
table.insert_many([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
    # ... many rows
])
```

### 10. Query Timeout

Set appropriate query timeouts to prevent long-running queries:

```python
# Set timeout via configuration
db = connect(
    "postgresql://user:pass@host/dbname",
    query_timeout=30.0  # 30 seconds
)

# Or via environment variable
# MOLTRES_QUERY_TIMEOUT=30.0
```

## Performance Monitoring

Moltres provides performance monitoring hooks:

```python
from moltres.engine import register_performance_hook

def log_slow_queries(sql: str, elapsed: float, metadata: dict):
    if elapsed > 1.0:  # Log queries taking more than 1 second
        print(f"Slow query ({elapsed:.2f}s): {sql[:200]}")
        print(f"  Rows: {metadata.get('rowcount', 'N/A')}")

register_performance_hook("query_end", log_slow_queries)
```

## Database-Specific Optimizations

### PostgreSQL

- Use `EXPLAIN ANALYZE` to understand query plans
- Consider partitioning for large tables
- Use materialized views for complex aggregations
- Enable query statistics: `pg_stat_statements`

```python
# View query plan
df = db.table("orders").select().where(col("amount") > 100)
sql = df.to_sql()
db.execute(f"EXPLAIN ANALYZE {sql}")
```

### SQLite

- Use WAL mode for better concurrency: `PRAGMA journal_mode=WAL`
- Increase cache size: `PRAGMA cache_size = -64000` (64MB)
- Use appropriate page size: `PRAGMA page_size = 4096`

```python
db.execute("PRAGMA journal_mode=WAL")
db.execute("PRAGMA cache_size = -64000")
```

### MySQL

- Use InnoDB engine for better performance
- Configure buffer pool size appropriately
- Use query cache for read-heavy workloads

## Common Performance Issues

### Issue 1: Full Table Scans

**Symptom:** Queries are slow even with small result sets

**Solution:** Add indexes on filtered columns

```python
# Check if index exists
db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")

# Use indexed column in filter
df = db.table("users").select().where(col("email") == "user@example.com")
```

### Issue 2: Large Result Sets

**Symptom:** Memory usage spikes when collecting results

**Solution:** Use streaming or add LIMIT

```python
# Use streaming
for chunk in df.collect(stream=True):
    process_chunk(chunk)

# Or add limit
df = df.limit(1000)
```

### Issue 3: N+1 Query Problem

**Symptom:** Many small queries instead of one large query

**Solution:** Use joins instead of loops

```python
# Bad: N+1 queries
for customer in customers:
    orders = db.table("orders").select().where(col("customer_id") == customer["id"])

# Good: Single query with join
df = customers.join(orders, on=[("id", "customer_id")])
```

### Issue 4: Inefficient Aggregations

**Symptom:** Aggregations are slow

**Solution:** Ensure grouping columns are indexed

```python
# Create index on grouping column
db.execute("CREATE INDEX idx_sales_category ON sales(category)")

# Group by indexed column
df = db.table("sales").select().group_by("category").agg(sum(col("amount")))
```

## Benchmarking

To benchmark Moltres queries:

```python
import time

start = time.perf_counter()
results = df.collect()
elapsed = time.perf_counter() - start

print(f"Query took {elapsed:.3f} seconds")
print(f"Returned {len(results)} rows")
```

## Performance Comparison

Moltres performance characteristics:

- **SQL Pushdown:** Operations execute in the database, not in Python
- **Lazy Evaluation:** Queries are optimized before execution
- **Connection Pooling:** Efficient connection reuse
- **Streaming:** Support for large result sets without memory issues

Compared to:
- **Pandas:** Moltres is faster for large datasets (database handles processing)
- **PySpark:** Moltres has lower overhead (no cluster setup)
- **Raw SQL:** Similar performance (Moltres compiles to SQL)

## Summary

1. **Index frequently queried columns**
2. **Filter early in the query chain**
3. **Select only needed columns**
4. **Use LIMIT for exploration**
5. **Optimize joins with indexes**
6. **Configure connection pooling**
7. **Use streaming for large results**
8. **Monitor slow queries**
9. **Use database-specific optimizations**
10. **Benchmark and profile queries**

For more information, see:
- [Best Practices Guide](./BEST_PRACTICES.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Deployment Guide](./DEPLOYMENT.md)

