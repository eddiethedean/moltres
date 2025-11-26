# Troubleshooting Guide

Common issues and solutions when using Moltres.

## Connection Issues

### "Failed to execute query" or Connection Errors

**Problem**: Cannot connect to database or queries fail immediately.

**Solutions**:
1. **Check connection string format**:
```python
# ✅ Correct formats
db = connect("sqlite:///path/to/db.db")
db = connect("postgresql://user:pass@host:5432/dbname")
db = connect("mysql://user:pass@host:3306/dbname")
```

2. **Verify database is accessible**:
```python
import sqlalchemy
engine = sqlalchemy.create_engine("your_connection_string")
with engine.connect() as conn:
    print("Connection successful!")
```

3. **Check network/firewall settings** for remote databases

4. **Verify credentials** are correct

5. **Enable connection pooling** for better reliability:
```python
db = connect(
    "postgresql://...",
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True  # Verify connections before use
)
```

### "Cannot collect a plan without an attached Database"

**Problem**: Trying to execute a DataFrame that isn't bound to a database.

**Solution**: Ensure the DataFrame is created from a database table or has a database attached:

```python
# ✅ Correct
db = connect("sqlite:///example.db")
df = db.table("users").select()
results = df.collect()

# ❌ Incorrect
from moltres.dataframe.dataframe import DataFrame
df = DataFrame(...)  # No database attached
results = df.collect()  # Will fail
```

## Query Issues

### "Unsupported logical plan node" or Compilation Errors

**Problem**: Query cannot be compiled to SQL.

**Solutions**:
1. **Check that all operations are supported**:
   - Basic operations: select, where, join, group_by, order_by, limit
   - Aggregations: sum, avg, count, min, max
   - Window functions: Limited support

2. **Verify column expressions**:
```python
# ✅ Correct
df.where(col("age") > 18)

# ❌ Incorrect
df.where("age > 18")  # Must use Column expressions
```

3. **Check join syntax**:
```python
# ✅ PySpark-style (recommended)
df1.join(df2, on=[col("left_col") == col("right_col")])

# ✅ Tuple syntax (backward compatible)
df1.join(df2, on=[("left_col", "right_col")])

# ✅ Same column name (simplest)
df1.join(df2, on="column")
```

### "Join requires either equality keys or an explicit condition"

**Problem**: Join operation is missing required parameters.

**Solution**: Provide either `on` parameter or `condition`:

```python
# ✅ Option 1: PySpark-style equality join
df1.join(df2, on=[col("id") == col("id")])

# ✅ Option 2: Tuple syntax (backward compatible)
df1.join(df2, on=[("id", "id")])

# ✅ Option 3: Custom condition (for complex joins)
from moltres import col
df1.join(df2, condition=col("df1.id") == col("df2.user_id"))
```

### Empty Results or Unexpected Data

**Problem**: Query returns no results or wrong data.

**Solutions**:
1. **Check filter conditions**:
```python
# Verify the condition
print(df.to_sql())  # See the generated SQL
```

2. **Verify table has data**:
```python
count = len(db.table("users").select().collect())
print(f"Table has {count} rows")
```

3. **Check data types**:
```python
# String comparison
df.where(col("status") == "active")  # Not col("status") == active (missing quotes)
```

## File Reading Issues

### "File not found" Errors

**Problem**: Cannot read CSV/JSON/Parquet files.

**Solutions**:
1. **Use absolute paths**:
```python
from pathlib import Path
file_path = Path("data.csv").resolve()
records = db.load.csv(str(file_path))
```

2. **Check file permissions**

3. **Verify file exists**:
```python
import os
if not os.path.exists("data.csv"):
    print("File not found!")
```

### Schema Inference Issues

**Problem**: Data types are inferred incorrectly.

**Solutions**:
1. **Provide explicit schema**:
```python
from moltres.table.schema import ColumnDef

schema = [
    ColumnDef(name="id", type_name="INTEGER"),
    ColumnDef(name="name", type_name="TEXT"),
    ColumnDef(name="price", type_name="REAL"),
]
records = db.load.schema(schema).csv("data.csv")
```

2. **Disable schema inference**:
```python
records = db.load.option("inferSchema", False).csv("data.csv")
```

## Performance Issues

### Slow Queries

**Problem**: Queries take too long to execute.

**Solutions**:
1. **Use streaming for large datasets**:
```python
records = db.load.stream().csv("large_file.csv")
for row in records:
    process(row)
```

2. **Add indexes** to database tables (at database level)

3. **Use batch inserts** (already implemented automatically):
```python
# Automatically uses batch inserts
table.insert([row1, row2, ..., row1000])
```

4. **Limit results** when possible:
```python
df.limit(100).collect()
```

5. **Check connection pooling**:
```python
db = connect(
    "postgresql://...",
    pool_size=10,
    max_overflow=20
)
```

### Memory Issues

**Problem**: Out of memory errors with large datasets.

**Solutions**:
1. **Use streaming mode**:
```python
records = db.load.stream().option("chunk_size", 10000).csv("large.csv")
for row in records:
    process(row)
```

2. **Process in batches**:
```python
# Process 1000 rows at a time
for i in range(0, total_rows, 1000):
    batch = df.limit(1000).offset(i).collect()
    process_batch(batch)
```

## Type and Format Issues

### "Unknown fetch format" Error

**Problem**: Requested format (pandas/polars) not available.

**Solutions**:
1. **Install required dependencies**:
   ```bash
   pip install moltres[pandas]  # For pandas
   pip install moltres[polars]  # For polars
   ```

2. **Use records format** (default, no dependencies needed):
```python
db = connect("sqlite:///example.db")  # Default: fetch_format="records"
```

### Type Errors with Mypy

**Problem**: Type checker complains about types.

**Solutions**:
1. **Use type hints properly**:
```python
from typing import List, Dict, Any

results: List[Dict[str, Any]] = df.collect()
```

2. **Cast when necessary**:
```python
from typing import cast
pandas_df = cast(pd.DataFrame, df.collect())
```

## Validation Errors

### "SQL identifier cannot be empty" or Invalid Identifier Errors

**Problem**: Table or column name validation fails.

**Solutions**:
1. **Check identifier names**:
```python
# ✅ Valid
db.table("users")
db.table("user_profiles")

# ❌ Invalid
db.table("")  # Empty
db.table("users; DROP")  # Contains invalid characters
```

2. **Validate user input** before using as identifiers:
```python
import re

table_name = get_user_input()
if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
    raise ValueError("Invalid table name")
db.table(table_name)
```

### "Row does not match expected columns"

**Problem**: Inserted rows have inconsistent schemas.

**Solutions**:
1. **Ensure all rows have same columns**:
```python
# ✅ Correct
table.insert([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
])

# ❌ Incorrect
table.insert([
    {"id": 1, "name": "Alice"},
    {"id": 2},  # Missing "name"
])
```

2. **Check column names match table schema**

## Getting Help

If you're still experiencing issues:

1. **Check the generated SQL**:
```python
print(df.to_sql())
```

2. **Enable logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
db = connect("sqlite:///example.db", echo=True)
```

3. **Check GitHub Issues**: https://github.com/eddiethedean/moltres/issues

4. **Create a minimal reproduction**:
   - Small code sample
   - Sample data
   - Expected vs actual behavior
   - Error messages

5. **Check documentation**: See README.md and docs/ directory

