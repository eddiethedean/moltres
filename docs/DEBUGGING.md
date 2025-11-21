# Debugging Guide

This guide helps you debug and troubleshoot issues with Moltres queries.

## Viewing Generated SQL

The most important debugging tool is viewing the SQL that Moltres generates:

```python
df = db.table("users").select().where(col("age") > 18)
sql = df.to_sql()
print(sql)
# Output: SELECT * FROM users WHERE age > 18
```

### Why This Helps

- **Verify correctness:** Check if the SQL matches your intent
- **Performance:** See what the database will execute
- **Debug errors:** Understand why a query might fail
- **Optimization:** Identify opportunities for improvement

## Common Error Patterns

### 1. CompilationError

**What it means:** Moltres couldn't compile your operation to SQL.

**Common causes:**
- Unsupported operation for your SQL dialect
- Invalid column references
- Missing required parameters

**How to debug:**
```python
try:
    df = db.table("users").select().where(col("invalid_col") > 18)
    df.collect()
except CompilationError as e:
    print(f"Error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    print(f"Context: {e.context}")
    
    # Check the SQL that was attempted
    try:
        sql = df.to_sql()
        print(f"Generated SQL: {sql}")
    except:
        print("Could not generate SQL")
```

**Solutions:**
- Check column names exist in the table
- Verify the operation is supported for your dialect
- Review the error suggestion

### 2. ExecutionError

**What it means:** The SQL query failed to execute in the database.

**Common causes:**
- Table/column doesn't exist
- SQL syntax error
- Data type mismatch
- Constraint violation

**How to debug:**
```python
try:
    results = df.collect()
except ExecutionError as e:
    print(f"Error: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    
    # Get the SQL that failed
    sql = df.to_sql()
    print(f"Failed SQL: {sql}")
    
    # Try executing directly in database to see detailed error
    try:
        db.execute(sql)
    except Exception as db_error:
        print(f"Database error: {db_error}")
```

**Solutions:**
- Verify table and column names
- Check data types match
- Test SQL directly in database client
- Review database error logs

### 3. QueryTimeoutError

**What it means:** The query exceeded the configured timeout.

**How to debug:**
```python
try:
    results = df.collect()
except QueryTimeoutError as e:
    print(f"Query timed out after {e.context.get('timeout_seconds')} seconds")
    
    # Check the slow query
    sql = df.to_sql()
    print(f"Slow SQL: {sql}")
    
    # Use EXPLAIN to understand the query plan
    explain_sql = f"EXPLAIN ANALYZE {sql}"
    plan = db.execute(explain_sql)
    print(f"Query plan: {plan}")
```

**Solutions:**
- Add indexes on filtered/joined columns
- Filter early to reduce data scanned
- Increase timeout if appropriate
- Break query into smaller chunks

### 4. DatabaseConnectionError

**What it means:** Failed to connect to the database.

**How to debug:**
```python
try:
    db = connect("postgresql://user:pass@host/dbname")
    db.table("users").select().collect()
except DatabaseConnectionError as e:
    print(f"Connection failed: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    
    # Test connection string
    import sqlalchemy
    try:
        engine = sqlalchemy.create_engine("postgresql://user:pass@host/dbname")
        with engine.connect() as conn:
            print("Direct connection works")
    except Exception as conn_error:
        print(f"Direct connection failed: {conn_error}")
```

**Solutions:**
- Verify connection string is correct
- Check database server is running
- Verify network connectivity
- Check credentials and permissions

## Debugging Query Logic

### 1. Step-by-Step Execution

Build queries incrementally and test at each step:

```python
# Step 1: Basic select
df = db.table("users").select()
print("Step 1 SQL:", df.to_sql())
results = df.limit(5).collect()
print("Step 1 results:", results)

# Step 2: Add filter
df = df.where(col("active") == True)
print("Step 2 SQL:", df.to_sql())
results = df.limit(5).collect()
print("Step 2 results:", results)

# Step 3: Add join
df = df.join(orders, on=[("id", "user_id")])
print("Step 3 SQL:", df.to_sql())
results = df.limit(5).collect()
print("Step 3 results:", results)
```

### 2. Verify Intermediate Results

Use `limit()` and `show()` to inspect intermediate results:

```python
# Check what data exists
df = db.table("users").select()
print("Total rows:", len(df.collect()))
df.show(10)  # Show first 10 rows

# Check filtered results
df_filtered = df.where(col("age") > 18)
print("Filtered rows:", len(df_filtered.collect()))
df_filtered.show(10)
```

### 3. Compare Expected vs Actual

```python
# Expected: Users with age > 18
expected_count = 100

# Actual
df = db.table("users").select().where(col("age") > 18)
actual_count = len(df.collect())

print(f"Expected: {expected_count}, Actual: {actual_count}")

if actual_count != expected_count:
    # Debug the filter
    sql = df.to_sql()
    print(f"SQL: {sql}")
    
    # Check data
    all_users = db.table("users").select().collect()
    print(f"Total users: {len(all_users)}")
    print(f"Users with age > 18: {sum(1 for u in all_users if u.get('age', 0) > 18)}")
```

## Debugging Performance Issues

### 1. Use EXPLAIN

Most databases support `EXPLAIN` to see query plans:

```python
df = db.table("orders").select().where(col("status") == "active")
sql = df.to_sql()

# PostgreSQL
explain_sql = f"EXPLAIN ANALYZE {sql}"
plan = db.execute(explain_sql)
print(plan)

# SQLite
explain_sql = f"EXPLAIN QUERY PLAN {sql}"
plan = db.execute(explain_sql)
print(plan)
```

### 2. Monitor Query Execution

Use performance hooks to monitor queries:

```python
from moltres.engine import register_performance_hook

def log_query(sql: str, elapsed: float, metadata: dict):
    if elapsed > 0.1:  # Log queries taking more than 100ms
        print(f"Query ({elapsed:.3f}s): {sql[:200]}")
        print(f"  Rows: {metadata.get('rowcount', 'N/A')}")

register_performance_hook("query_end", log_query)
```

### 3. Profile Query Steps

Time individual operations:

```python
import time

start = time.perf_counter()
df = db.table("users").select()
print(f"Build query: {(time.perf_counter() - start) * 1000:.2f}ms")

start = time.perf_counter()
sql = df.to_sql()
print(f"Compile SQL: {(time.perf_counter() - start) * 1000:.2f}ms")

start = time.perf_counter()
results = df.collect()
print(f"Execute query: {(time.perf_counter() - start) * 1000:.2f}ms")
print(f"Results: {len(results)} rows")
```

## Debugging Data Issues

### 1. Check for NULL Values

```python
from moltres.expressions.functions import isnull, isnotnull

# Count NULL values
null_count = (
    db.table("users")
    .select()
    .where(isnull(col("email")))
    .count()
)
print(f"Users with NULL email: {null_count}")

# Check specific values
df = db.table("users").select().where(col("email").is_null())
df.show(10)
```

### 2. Verify Data Types

```python
# Check column types in database
if db.dialect.name == "postgresql":
    type_query = """
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'users'
    """
    types = db.execute(type_query)
    print("Column types:", types)
```

### 3. Inspect Data Ranges

```python
from moltres.expressions.functions import min, max, avg

# Check data ranges
stats = (
    db.table("orders")
    .select()
    .agg(
        min(col("amount")).alias("min_amount"),
        max(col("amount")).alias("max_amount"),
        avg(col("amount")).alias("avg_amount")
    )
    .collect()
)
print("Order amount stats:", stats)
```

## Debugging Join Issues

### 1. Verify Join Keys

```python
# Check join key values exist
left_ids = set(
    row["id"] 
    for row in db.table("customers").select("id").collect()
)
right_ids = set(
    row["customer_id"] 
    for row in db.table("orders").select("customer_id").collect()
)

print(f"Customers: {len(left_ids)}")
print(f"Orders: {len(right_ids)}")
print(f"Matching: {len(left_ids & right_ids)}")
print(f"Orphaned orders: {len(right_ids - left_ids)}")
```

### 2. Test Join Conditions

```python
# Test join with explicit condition
df = (
    db.table("customers")
    .select()
    .join(
        db.table("orders").select(),
        on=[("id", "customer_id")],
        how="inner"
    )
)

# Check SQL
sql = df.to_sql()
print("Join SQL:", sql)

# Check results
results = df.limit(10).collect()
print("Join results:", results)
```

## Debugging Aggregation Issues

### 1. Verify Grouping Columns

```python
# Check distinct values in grouping column
categories = (
    db.table("orders")
    .select("category")
    .distinct()
    .collect()
)
print("Categories:", [r["category"] for r in categories])

# Compare with aggregation
agg_results = (
    db.table("orders")
    .select()
    .group_by("category")
    .agg(count("*").alias("count"))
    .collect()
)
print("Aggregation results:", agg_results)
```

### 2. Check for NULL in Grouping

```python
# Count NULL values in grouping column
null_count = (
    db.table("orders")
    .select()
    .where(isnull(col("category")))
    .count()
)
print(f"Orders with NULL category: {null_count}")
```

## Tools and Techniques

### 1. Enable SQL Logging

```python
# Enable SQL logging
db = connect(
    "postgresql://user:pass@host/dbname",
    echo=True  # Logs all SQL to console
)
```

### 2. Use Database Tools

- **PostgreSQL:** `psql`, pgAdmin, DBeaver
- **SQLite:** `sqlite3` command-line tool
- **MySQL:** `mysql` command-line tool, MySQL Workbench

Test queries directly in these tools to verify behavior.

### 3. Create Test Queries

```python
# Create a simple test query
def test_query():
    df = db.table("users").select("id", "name").limit(1)
    sql = df.to_sql()
    print(f"Test SQL: {sql}")
    result = df.collect()
    print(f"Test result: {result}")
    return result

# Run test
test_query()
```

## Summary

1. **Use `df.to_sql()`** to see generated SQL
2. **Read error messages** and suggestions carefully
3. **Test incrementally** - build queries step by step
4. **Use `limit()` and `show()`** to inspect results
5. **Use `EXPLAIN`** to understand query plans
6. **Monitor performance** with hooks
7. **Verify data** - check for NULLs, types, ranges
8. **Test in database tools** directly
9. **Enable SQL logging** for detailed debugging
10. **Create test queries** to isolate issues

For more help:
- [Performance Guide](./PERFORMANCE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [FAQ](./FAQ.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

