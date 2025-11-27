# Error Handling and Debugging Guide

Learn how to handle errors and debug issues in Moltres.

## Common Error Types

### 1. Connection Errors

**Error**: `DatabaseConnectionError` or connection timeout

```python
from moltres import connect
from moltres.utils.exceptions import DatabaseConnectionError

try:
    db = connect("postgresql://user:pass@localhost/mydb")
except DatabaseConnectionError as e:
    print(f"Connection failed: {e}")
    print(f"Suggestion: {e.suggestion}")  # Helpful error message
```

**Common causes:**
- Invalid connection string format
- Database server not running
- Wrong credentials
- Network issues
- Missing async driver (for async connections)

**Solutions:**
```python
# Validate connection string format
# Must include :// separator
db = connect("postgresql://user:pass@localhost/mydb")  # ✅ Correct
db = connect("postgresql:user:pass@localhost/mydb")     # ❌ Wrong

# For async, include driver
db = async_connect("postgresql+asyncpg://...")  # ✅ Correct
db = async_connect("postgresql://...")          # ❌ Wrong (missing +asyncpg)
```

### 2. Compilation Errors

**Error**: `CompilationError` - SQL compilation failed

```python
from moltres import connect
from moltres.utils.exceptions import CompilationError
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

try:
    df = db.table("users").select().where(col("invalid_col") > 25)
    results = df.collect()
except CompilationError as e:
    print(f"Compilation failed: {e}")
    # Check the generated SQL
    print(f"SQL: {df.explain()}")

```

**Common causes:**
- Invalid column names
- Unsupported SQL operations
- Type mismatches

**Solutions:**
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
# Check column names
columns = db.get_columns("users")
print([c.name for c in columns])

# Use correct column references
df = db.table("users").select().where(col("age") > 25)  # ✅ Correct
df = db.table("users").select().where(col("Age") > 25)   # ❌ Case-sensitive

```

### 3. Execution Errors

**Error**: `ExecutionError` - SQL execution failed

```python
from moltres import connect
from moltres.utils.exceptions import ExecutionError
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

try:
    df = db.table("users").select()
    results = df.collect()
except ExecutionError as e:
    print(f"Execution failed: {e}")
    # Check the SQL that failed
    print(f"SQL: {e.sql if hasattr(e, 'sql') else 'N/A'}")

```

**Common causes:**
- Table doesn't exist
- Permission issues
- Constraint violations
- Invalid SQL syntax (database-specific)

**Solutions:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Check if table exists
tables = db.get_table_names()
if "users" not in tables:
    print("Table 'users' does not exist")

# Check table schema
schema = db.get_columns("users")
for col_info in schema:
    print(f"{col_info.name}: {col_info.type_name}")

```

### 4. Validation Errors

**Error**: `ValidationError` - Input validation failed

```python
from moltres import connect
from moltres.utils.exceptions import ValidationError
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

try:
    db.update("users", where=col("id") == 1, set={"invalid_col": "value"})
except ValidationError as e:
    print(f"Validation failed: {e}")

```

**Common causes:**
- Invalid table/column names
- SQL injection attempts (blocked by validation)
- Type mismatches

## Debugging Techniques

### 1. Enable SQL Logging

See the actual SQL being generated and executed:

```python
db = connect(
    "postgresql://user:pass@localhost/mydb",
    echo=True  # Enable SQL logging
)

# Now all SQL will be printed to console
df = db.table("users").select().where(col("age") > 25)
results = df.collect()
# Output: SELECT users.id, users.name, ... FROM users WHERE users.age > 25
```

### 2. Inspect Query Plans

Understand how your query will be executed:

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
# Get estimated execution plan
df = db.table("users").select().where(col("age") > 25)
plan = df.explain()
print(plan)

# Get actual execution stats (PostgreSQL)
plan = df.explain(analyze=True)
print(plan)

```

### 3. Check Generated SQL

See the exact SQL that will be executed:

```python
from moltres import connect
from moltres.sql.compiler import compile_plan
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

df = db.table("users").select().where(col("age") > 25)
sql = compile_plan(df.plan, dialect=db._config.dialect)
print(str(sql.compile(compile_kwargs={"literal_binds": True})))

```

### 4. Validate Before Execution

Check your DataFrame structure before collecting:

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
# Print schema
df = db.table("users").select()
df.printSchema()

# Check logical plan
print(df.plan)

# Get column names
columns = db.get_columns("users")
print([c.name for c in columns])

# Output: root
# Output:  |-- id: INTEGER (nullable = true)
# Output:  |-- name: TEXT (nullable = true)
# Output: TableScan(table='users', alias=None)
# Output: ['id', 'name']
```

### 5. Use Try-Except Blocks

Wrap operations in try-except for better error handling:

```python
from moltres import connect
from moltres.utils.exceptions import (
    DatabaseConnectionError,
    CompilationError,
    ExecutionError,
    ValidationError
)
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

def safe_query(db, table_name, condition):
    try:
        df = db.table(table_name).select().where(condition)
        return df.collect()
    except DatabaseConnectionError as e:
        print(f"Connection error: {e}")
        return None
    except CompilationError as e:
        print(f"Compilation error: {e}")
        return None
    except ExecutionError as e:
        print(f"Execution error: {e}")
        return None
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

```

## Common Issues and Solutions

### Issue 1: "Table does not exist"

**Problem:**
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
df = db.table("users").select()
results = df.collect()  # Error: table "users" does not exist

# Output: [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Bob'}]
```

**Solution:**
```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Check if table exists
tables = db.get_table_names()
if "users" not in tables:
    # Create table first
    db.create_table("users", [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
    ]).collect()

```

### Issue 2: "Column does not exist"

**Problem:**
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
df = db.table("users").select().where(col("age") > 25)
results = df.collect()  # Error: column "age" does not exist

```

**Solution:**
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
# Check actual column names
columns = db.get_columns("users")
print([c.name for c in columns])

# Use correct column name (case-sensitive in some databases)
df = db.table("users").select().where(col("Age") > 25)  # If column is "Age"

```

### Issue 3: "Ambiguous column name"

**Problem:**
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
df = (
    db.table("users").select()
    .join(db.table("orders").select(), on=[col("id") == col("user_id")])
    .select("id")  # Error: ambiguous column "id"
)

```

**Solution:**
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
# Use table-qualified column names
df = (
    db.table("users").select()
    .join(db.table("orders").select(), on=[col("users.id") == col("orders.user_id")])
    .select("users.id")  # Or col("users.id")
)

```

### Issue 4: "Type mismatch"

**Problem:**
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
df = db.table("users").select().where(col("age") == "25")  # age is INTEGER

```

**Solution:**
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
# Use correct type
df = db.table("users").select().where(col("age") == 25)  # Integer, not string

```

### Issue 5: "Boolean value error" (SQLite)

**Problem:**
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
# SQLite uses INTEGER (0/1) instead of BOOLEAN
df = db.table("users").select().where(col("active") == True)  # May not work

```

**Solution:**
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
# Use 1/0 for SQLite
df = db.table("users").select().where(col("active") == 1)  # ✅ Correct for SQLite

# Or use boolean for PostgreSQL/MySQL
df = db.table("users").select().where(col("active") == True)  # ✅ Correct for PostgreSQL

```

### Issue 6: "Connection pool exhausted"

**Problem:**
```python
# Too many concurrent connections
for i in range(1000):
    db = connect("postgresql://...")  # Creating new connections
    results = db.table("users").select().collect()
```

**Solution:**
```python
# Reuse connection
db = connect("postgresql://...")
for i in range(1000):
    results = db.table("users").select().collect()

# Or configure connection pooling
db = connect(
    "postgresql://...",
    pool_size=10,
    max_overflow=20
)
```

## Performance Debugging

### Identify Slow Queries

```python
from moltres.engine import register_performance_hook
import time

def log_slow_queries(sql: str, elapsed: float, metadata: dict):
    if elapsed > 1.0:  # Log queries taking > 1 second
        print(f"Slow query ({elapsed:.2f}s): {sql[:200]}")

register_performance_hook("query_end", log_slow_queries)
```

### Check Query Execution Time

```python
from moltres import connect
import time
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

start = time.perf_counter()
results = df.collect()
elapsed = time.perf_counter() - start
print(f"Query took {elapsed:.2f} seconds")

```

### Analyze Query Plans

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Get execution plan
plan = df.explain(analyze=True)
print(plan)

# Look for:
# - Sequential scans (should use indexes)
# - High cost operations
# - Missing indexes

```

## Best Practices for Error Handling

### 1. Validate Inputs Early

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
def query_users(db, min_age):
    # Validate inputs
    if not isinstance(min_age, int):
        raise ValueError("min_age must be an integer")
    if min_age < 0:
        raise ValueError("min_age must be non-negative")
    
    # Then execute query
    return db.table("users").select().where(col("age") >= min_age).collect()

```

### 2. Use Context Managers

```python
# Ensure connections are closed
with connect("postgresql://...") as db:
    results = db.table("users").select().collect()
# Connection automatically closed
```

### 3. Handle Database-Specific Errors

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
try:
    results = df.collect()
except ExecutionError as e:
    if "does not exist" in str(e):
        # Handle missing table/column
        pass
    elif "permission denied" in str(e):
        # Handle permission issues
        pass
    else:
        # Re-raise unknown errors
        raise

```

### 4. Log Errors Appropriately

```python
from moltres import connect
import logging
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

logger = logging.getLogger(__name__)

try:
    results = df.collect()
except Exception as e:
    logger.error(f"Query failed: {e}", exc_info=True)
    raise

```

## Getting Help

If you encounter errors:

1. **Check the error message**: Moltres provides helpful error messages with suggestions
2. **Enable SQL logging**: Set `echo=True` to see generated SQL
3. **Check query plan**: Use `df.explain()` to understand the query
4. **Validate inputs**: Ensure table/column names are correct
5. **Check documentation**: See `docs/TROUBLESHOOTING.md` for common issues
6. **Review examples**: Check [examples directory](https://github.com/eddiethedean/moltres/tree/main/examples) for working code

## Next Steps

- **Best Practices**: See [Best Practices Guide](https://github.com/eddiethedean/moltres/blob/main/guides/08-best-practices.md)
- **Performance**: Read [Performance Optimization Guide](https://github.com/eddiethedean/moltres/blob/main/guides/04-performance-optimization.md)
- **Troubleshooting**: Check [Troubleshooting Guide](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md) for more solutions

