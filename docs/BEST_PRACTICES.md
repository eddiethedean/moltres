# Best Practices Guide

This guide provides recommendations and patterns for using Moltres effectively in production.

## Code Organization

### 1. Connection Management

**Best Practice:** Create database connections at the application level and reuse them.

```python
# Good: Single connection per application
db = connect("postgresql://user:pass@host/dbname")

# Use the same connection throughout
def get_users():
    return db.table("users").select().collect()

def get_orders():
    return db.table("orders").select().collect()
```

**Avoid:** Creating new connections for each query

```python
# Bad: New connection for each query
def get_users():
    db = connect("postgresql://user:pass@host/dbname")  # Don't do this
    return db.table("users").select().collect()
```

### 2. Query Composition

**Best Practice:** Compose queries step by step for readability

```python
# Good: Clear, readable query composition
df = (
    db.table("orders")
    .select("id", "customer_id", "amount", "status")
    .where(col("status") == "active")
    .where(col("amount") > 100)
    .order_by(col("amount").desc())
    .limit(100)
)
```

**Avoid:** Overly complex one-liners

```python
# Less readable: Everything in one line
df = db.table("orders").select("id", "customer_id", "amount", "status").where(col("status") == "active").where(col("amount") > 100).order_by(col("amount").desc()).limit(100)
```

### 3. Column References

**Best Practice:** Use `col()` consistently for column references

```python
from moltres import col

# Good: Explicit column references
df = (
    db.table("users")
    .select(col("id"), col("name"), col("email"))
    .where(col("active") == True)
)
```

**Avoid:** Mixing string and column references inconsistently

```python
# Less clear: Mixed usage
df = (
    db.table("users")
    .select("id", "name", col("email"))  # Inconsistent
    .where(col("active") == True)
)
```

## Error Handling

### 1. Handle Database Errors

**Best Practice:** Catch and handle specific exceptions

```python
from moltres.utils.exceptions import (
    ExecutionError,
    CompilationError,
    DatabaseConnectionError,
    QueryTimeoutError,
)

try:
    results = df.collect()
except QueryTimeoutError as e:
    print(f"Query timed out: {e}")
    # Handle timeout
except ExecutionError as e:
    print(f"Query failed: {e}")
    print(f"Suggestion: {e.suggestion}")
    # Handle execution error
except DatabaseConnectionError as e:
    print(f"Connection failed: {e}")
    # Handle connection error
```

### 2. Validate Inputs

**Best Practice:** Validate inputs before building queries

```python
def get_user_orders(user_id: int, limit: int = 100):
    if not isinstance(user_id, int) or user_id < 0:
        raise ValueError("user_id must be a positive integer")
    if limit < 0 or limit > 1000:
        raise ValueError("limit must be between 0 and 1000")
    
    return (
        db.table("orders")
        .select()
        .where(col("user_id") == user_id)
        .limit(limit)
        .collect()
    )
```

## Security

### 1. SQL Injection Prevention

**Best Practice:** Use column expressions, not string concatenation

```python
# Good: Safe - uses parameterized queries
user_id = 123
df = db.table("users").select().where(col("id") == user_id)

# Bad: Vulnerable to SQL injection
user_id = "123; DROP TABLE users;"
sql = f"SELECT * FROM users WHERE id = {user_id}"  # DON'T DO THIS
```

### 2. Connection String Security

**Best Practice:** Use environment variables for credentials

```python
import os

# Good: Credentials from environment
dsn = os.environ.get("MOLTRES_DSN")
db = connect(dsn)

# Avoid: Hardcoded credentials
db = connect("postgresql://user:password@host/dbname")  # Don't commit this
```

## Performance

### 1. Use Indexes

**Best Practice:** Create indexes on frequently queried columns

```python
# Create indexes for common queries
db.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
db.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)")
```

### 2. Filter Early

**Best Practice:** Apply filters as early as possible

```python
# Good: Filter before join
df = (
    db.table("orders")
    .select()
    .where(col("status") == "active")  # Filter early
    .join(customers, on=[("customer_id", "id")])
)
```

### 3. Select Only Needed Columns

**Best Practice:** Avoid `SELECT *` when possible

```python
# Good: Select only needed columns
df = db.table("users").select("id", "name", "email")

# Less efficient: Select all columns
df = db.table("users").select()  # SELECT *
```

### 4. Use Streaming for Large Results

**Best Practice:** Stream large result sets

```python
# Good: Stream large results
for chunk in df.collect(stream=True):
    process_chunk(chunk)

# Avoid: Loading everything into memory
results = df.collect()  # May cause memory issues
```

## Testing

### 1. Test Query Generation

**Best Practice:** Verify SQL generation during development

```python
# Check generated SQL
df = db.table("users").select().where(col("age") > 18)
sql = df.to_sql()
print(sql)  # Verify the SQL is correct
```

### 2. Use Test Databases

**Best Practice:** Use separate test databases

```python
# Test database
test_db = connect("sqlite:///test.db")

# Production database
prod_db = connect("postgresql://user:pass@host/prod_db")
```

### 3. Test Error Handling

**Best Practice:** Test error scenarios

```python
import pytest
from moltres.utils.exceptions import ExecutionError

def test_invalid_query():
    df = db.table("nonexistent").select()
    with pytest.raises(ExecutionError):
        df.collect()
```

## Configuration

### 1. Environment Variables

**Best Practice:** Use environment variables for configuration

```bash
# .env file or environment
export MOLTRES_DSN="postgresql://user:pass@host/dbname"
export MOLTRES_POOL_SIZE=10
export MOLTRES_QUERY_TIMEOUT=30.0
```

```python
# Application code
db = connect()  # Uses MOLTRES_DSN from environment
```

### 2. Connection Pooling

**Best Practice:** Configure connection pooling appropriately

```python
db = connect(
    "postgresql://user:pass@host/dbname",
    pool_size=10,        # Match your workload
    max_overflow=5,      # Allow some overflow
    pool_pre_ping=True,  # Verify connections
)
```

## Code Patterns

### 1. Query Builders

**Best Practice:** Create reusable query builders

```python
def build_user_query(active_only: bool = False, min_age: int = None):
    df = db.table("users").select("id", "name", "email")
    
    if active_only:
        df = df.where(col("active") == True)
    
    if min_age is not None:
        df = df.where(col("age") >= min_age)
    
    return df

# Use the builder
active_users = build_user_query(active_only=True, min_age=18).collect()
```

### 2. CTEs for Complex Queries

**Best Practice:** Use CTEs for complex queries

```python
# Good: Use CTEs for readability
active_orders = (
    db.table("orders")
    .select()
    .where(col("status") == "active")
    .cte("active_orders")
)

high_value = (
    active_orders
    .select()
    .where(col("amount") > 1000)
    .cte("high_value")
)

result = high_value.select().collect()
```

### 3. Type Hints

**Best Practice:** Use type hints for better IDE support

```python
from typing import List, Dict
from moltres import DataFrame

def get_active_users() -> List[Dict[str, object]]:
    df: DataFrame = (
        db.table("users")
        .select()
        .where(col("active") == True)
    )
    return df.collect()
```

## Documentation

### 1. Document Complex Queries

**Best Practice:** Add comments for complex logic

```python
# Calculate monthly revenue by category
# Filters active orders, groups by month and category,
# and calculates total revenue
df = (
    db.table("orders")
    .select()
    .where(col("status") == "active")
    .where(col("date") >= "2024-01-01")
    .group_by(
        func.date_trunc("month", col("date")).alias("month"),
        col("category")
    )
    .agg(sum(col("amount")).alias("revenue"))
)
```

### 2. Document Schema Assumptions

**Best Practice:** Document expected schema

```python
def get_user_orders(user_id: int):
    """
    Get orders for a user.
    
    Assumes:
    - orders table has columns: id, user_id, amount, status, date
    - user_id column is indexed for performance
    """
    return (
        db.table("orders")
        .select()
        .where(col("user_id") == user_id)
        .collect()
    )
```

## Common Patterns

### Pattern 1: Pagination

```python
def get_paginated_results(page: int, page_size: int = 20):
    offset = page * page_size
    return (
        db.table("items")
        .select()
        .order_by(col("id"))
        .limit(page_size)
        .offset(offset)  # If supported
        .collect()
    )
```

### Pattern 2: Conditional Filtering

```python
def search_users(name: str = None, email: str = None, active: bool = None):
    df = db.table("users").select()
    
    if name:
        df = df.where(col("name").like(f"%{name}%"))
    if email:
        df = df.where(col("email") == email)
    if active is not None:
        df = df.where(col("active") == active)
    
    return df.collect()
```

### Pattern 3: Aggregations with Conditions

```python
from moltres.expressions.functions import sum, when

# Sum with conditions
df = (
    db.table("orders")
    .select()
    .group_by("category")
    .agg(
        sum(when(col("status") == "active", col("amount"), 0)).alias("active_revenue"),
        sum(col("amount")).alias("total_revenue")
    )
)
```

## Summary

1. **Reuse database connections**
2. **Compose queries clearly**
3. **Use `col()` consistently**
4. **Handle errors appropriately**
5. **Validate inputs**
6. **Prevent SQL injection**
7. **Use environment variables for secrets**
8. **Create indexes for performance**
9. **Filter early in queries**
10. **Select only needed columns**
11. **Use streaming for large results**
12. **Test query generation**
13. **Use type hints**
14. **Document complex queries**
15. **Follow common patterns**

For more information, see:
- [Performance Guide](./PERFORMANCE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Migration Guide](./MIGRATION_SPARK.md)

