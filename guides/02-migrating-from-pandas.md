# Migrating from Pandas to Moltres

This guide helps you transition from Pandas to Moltres, highlighting key differences and migration patterns.

## Why Migrate?

- **Memory efficiency**: Moltres executes operations in SQL, avoiding data loading into memory
- **Large datasets**: Handle datasets larger than available RAM
- **Database-native**: Work directly with database tables without exporting/importing
- **SQL pushdown**: Leverage database optimizations for better performance

## Key Differences

| Pandas | Moltres |
|--------|---------|
| In-memory operations | SQL pushdown execution |
| `df[df['age'] > 25]` | `df.where(col("age") > 25)` |
| `df.groupby('col').agg(...)` | `df.group_by('col').agg(...)` |
| `df.merge(other, on='key')` | `df.join(other, on=[col("key") == col("other.key")])` |
| Eager execution | Lazy evaluation (call `.collect()`) |

## Migration Patterns

### 1. Reading Data

**Pandas:**
```python
import pandas as pd

# Read from CSV
# df = pd.read_csv("data.csv")  # Requires data.csv file

# Read from SQL
# df = pd.read_sql("SELECT * FROM users", connection)  # Requires database connection
```

**Moltres:**
```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

db = connect("sqlite:///:memory:")

# Create table first
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Read from CSV (returns Records, not DataFrame)
# Note: Requires data.csv file to exist
# records = Records.from_csv("data.csv", database=db)
# records.insert_into("users")

# Insert sample data instead
Records.from_list([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
], database=db).insert_into("users")

# Read from SQL table (returns DataFrame)
df = db.table("users").select()
results = df.collect()  # Returns list of dicts by default

# Or get as pandas DataFrame (requires: pip install pandas)
# results = df.collect(format="pandas")  # Returns pd.DataFrame
```

### 2. Filtering

**Pandas:**
```python
import pandas as pd

# Assume df is already loaded
# df = pd.read_csv("data.csv")

# Single condition
df_filtered = df[df['age'] > 25]

# Multiple conditions
df_filtered = df[(df['age'] > 25) & (df['active'] == True)]

# String operations
df_filtered = df[df['email'].str.contains('@example.com')]
```

**Moltres:**
```python
from moltres import connect
from moltres import col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
    column("email", "TEXT"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "active": 1, "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "age": 25, "active": 0, "email": "bob@example.com"},
], database=db).insert_into("users")

df = db.table("users").select()

# Single condition
df_filtered = df.where(col("age") > 25)

# Multiple conditions
df_filtered = df.where(
    (col("age") > 25) & (col("active") == 1)
)

# String operations
df_filtered = df.where(
    col("email").like("%@example.com")
)

# Execute to get results
results = df_filtered.collect()

```

### 3. Selecting Columns

**Pandas:**
```python
import pandas as pd

# Assume df is already loaded
# df = pd.read_csv("data.csv")

# Select specific columns
df_selected = df[['id', 'name', 'email']]

# Select with new column
df['age_plus_10'] = df['age'] + 10
```

**Moltres:**
```python
from moltres import connect
from moltres import col
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("age", "INTEGER"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
], database=db).insert_into("users")

df = db.table("users").select()

# Select specific columns
df_selected = df.select("id", "name", "email")

# Select with computed column
df_new = df.select(
    "*",  # All existing columns
    (col("age") + 10).alias("age_plus_10")
)

```

### 4. GroupBy and Aggregations

**Pandas:**
```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("sales", [
    column("id", "INTEGER", primary_key=True),
    column("country", "TEXT"),
    column("amount", "REAL"),
]).collect()

Records.from_list([
    {"id": 1, "country": "USA", "amount": 100.0},
    {"id": 2, "country": "UK", "amount": 200.0},
], database=db).insert_into("sales")

df = db.table("sales").select()

# Simple aggregation
result = df.groupby('country').agg({
    'amount': 'sum',
    'id': 'count'
})

# Multiple aggregations
result = df.groupby('country').agg({
    'amount': ['sum', 'mean', 'max'],
    'id': 'count'
})

```

**Moltres:**
```python
from moltres import connect
from moltres import col
from moltres.expressions import functions as F
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("sales", [
    column("id", "INTEGER", primary_key=True),
    column("country", "TEXT"),
    column("amount", "REAL"),
]).collect()

Records.from_list([
    {"id": 1, "country": "USA", "amount": 100.0},
    {"id": 2, "country": "UK", "amount": 200.0},
], database=db).insert_into("sales")

df = db.table("sales").select()

# Simple aggregation
result = (
    df.group_by("country")
    .agg(
        F.sum(col("amount")).alias("total_amount"),
        F.count(col("id")).alias("count")
    )
)

# Multiple aggregations on same column
result = (
    df.group_by("country")
    .agg(
        F.sum(col("amount")).alias("total"),
        F.avg(col("amount")).alias("avg"),
        F.max(col("amount")).alias("max"),
        F.count("*").alias("count")
    )
)

```

### 5. Joins

**Pandas:**
```python
import pandas as pd

# Assume df1 and df2 are already loaded
# df1 = pd.read_csv("data1.csv")
# df2 = pd.read_csv("data2.csv")

# Inner join
result = df1.merge(df2, on='key', how='inner')

# Left join
result = df1.merge(df2, on='key', how='left')

# Right join
result = df1.merge(df2, on='key', how='right')

# Outer join
result = df1.merge(df2, on='key', how='outer')
```

**Moltres:**
```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Setup: Create tables and data
db = connect("sqlite:///:memory:")
db.create_table("table1", [column("key", "INTEGER"), column("value1", "TEXT")]).collect()
db.create_table("table2", [column("key", "INTEGER"), column("value2", "TEXT")]).collect()

df1 = db.table("table1").select()
df2 = db.table("table2").select()

# Inner join (default)
# Note: Use table-qualified column names to avoid ambiguity
result = df1.join(
    df2,
    on=[col("table1.key") == col("table2.key")]
)

# Left join
result = df1.join(
    df2,
    on=[col("table1.key") == col("table2.key")],
    how="left"
)

# Right join
result = df1.join(
    df2,
    on=[col("table1.key") == col("table2.key")],
    how="right"
)

# Outer join
result = df1.join(
    df2,
    on=[col("table1.key") == col("table2.key")],
    how="outer"
)
```

### 6. Sorting

**Pandas:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create table and data first
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "active": 1},
    {"id": 2, "name": "Bob", "age": 25, "active": 0},
], database=db).insert_into("users")

df = db.table("users").select()

# Note: sort_values() is pandas syntax. In Moltres, use order_by()
# This example shows pandas syntax for comparison
try:
    import pandas as pd
    # Convert to pandas first
    df_pandas = df.collect(format="pandas")
    df_sorted = df_pandas.sort_values('age')
    df_sorted = df_pandas.sort_values('age', ascending=False)
    df_sorted = df_pandas.sort_values(['active', 'age'], ascending=[False, True])
except ImportError:
    print("pandas not installed. Install with: pip install pandas or pip install moltres[pandas]")

```

**Moltres:**
```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("active", "INTEGER"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "active": 1},
    {"id": 2, "name": "Bob", "age": 25, "active": 0},
], database=db).insert_into("users")

df = db.table("users").select()

# Ascending
df_sorted = df.order_by("age")

# Descending
df_sorted = df.order_by(col("age").desc())

# Multiple columns
df_sorted = df.order_by(
    col("active").desc(),
    col("age").asc()
)

```

### 7. Adding/Modifying Columns

**Pandas:**
```python
import pandas as pd

# Assume df is already loaded
# df = pd.read_csv("data.csv")

# Add new column
df['new_col'] = df['col1'] + df['col2']

# Modify existing
df['age'] = df['age'] + 1

# Conditional column
df['category'] = df['age'].apply(lambda x: 'adult' if x >= 18 else 'minor')
```

**Moltres:**
```python
from moltres import connect, col
from moltres.expressions import functions as F
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
db.create_table("data", [
    column("id", "INTEGER", primary_key=True),
    column("col1", "INTEGER"),
    column("col2", "INTEGER"),
    column("age", "INTEGER"),
]).collect()

Records.from_list([
    {"id": 1, "col1": 10, "col2": 20, "age": 25},
    {"id": 2, "col1": 15, "col2": 25, "age": 17},
], database=db).insert_into("data")

df = db.table("data").select()

# Add new column (use withColumn or select)
df_new = df.withColumn(
    "new_col",
    col("col1") + col("col2")
)

# Or in select
df_new = df.select(
    "*",
    (col("col1") + col("col2")).alias("new_col")
)

# Conditional column
df_new = df.withColumn(
    "category",
    F.when(col("age") >= 18, "adult").otherwise("minor")
)

```

### 8. Working with Results

**Pandas:**
```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Note: This example shows pandas syntax
try:
    import pandas as pd
    # Results are already DataFrames
    # df = pd.read_csv("data.csv")  # Requires data.csv file
    # result = df.groupby('country').sum()
    # result is a DataFrame
    print("pandas example - requires data.csv file")
except ImportError:
    print("pandas not installed. Install with: pip install pandas or pip install moltres[pandas]")

```

**Moltres:**
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
# Results are lists of dicts by default
df = db.table("users").select()
results = df.collect()
# results is: [{'id': 1, 'name': 'Alice'}, ...]

# Get as pandas DataFrame (requires: pip install pandas)
try:
    results = df.collect(format="pandas")
    # results is: pd.DataFrame
except ImportError:
    print("pandas not installed. Install with: pip install pandas or pip install moltres[pandas]")

# Or convert to pandas after
try:
    import pandas as pd
    results = df.collect()
    df_pandas = pd.DataFrame(results)
except ImportError:
    print("pandas not installed. Install with: pip install pandas or pip install moltres[pandas]")

```

## Using Pandas-Style Interface

Moltres also provides a pandas-style interface for easier migration:

**See also:** [Pandas interface examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html), [Pandas/Polars integration examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html), and the [Pandas Interface Guide](https://moltres.readthedocs.io/en/latest/guides/pandas-interface.html)

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
    column("age", "INTEGER"),
    column("active", "INTEGER"),
    column("country", "TEXT"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "active": 1, "country": "USA"},
    {"id": 2, "name": "Bob", "age": 25, "active": 1, "country": "UK"},
], database=db).insert_into("users")

# Get pandas-style DataFrame (requires: pip install pandas)
try:
    df = db.table("users").pandas()
    
    # Use pandas-style methods
    df_filtered = df.query('age > 25 and active == 1')
    df_grouped = df.groupby('country').agg({'age': 'mean', 'id': 'count'})
    
    # Still need to call collect()
    results = df_filtered.collect()
except ImportError:
    print("pandas not installed. Install with: pip install pandas or pip install moltres[pandas]")

```

## Handling Large Datasets

**Pandas limitation:**
```python
import pandas as pd

# This loads entire dataset into memory
# df = pd.read_csv("huge_file.csv")  # May fail if file > RAM
# df_filtered = df[df['age'] > 25]
```

**Moltres advantage:**
```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Setup database
db = connect("sqlite:///:memory:")
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
], database=db).insert_into("users")

# This executes in SQL - no memory loading
df = db.table("users").select().where(col("age") > 25)
results = df.collect()  # Only results are in memory

# Or use streaming for very large results (requires async)
import asyncio
from moltres import async_connect

async def stream_large_results():
    db = await async_connect("postgresql+asyncpg://user:pass@localhost/db")
    df = db.table("users").select().where(col("age") > 25)
    async for chunk in await df.collect(stream=True):
        process_chunk(chunk)
    await db.close()

# asyncio.run(stream_large_results())
```

## Common Migration Challenges

### Challenge 1: Eager vs Lazy Execution

**Problem**: Forgetting to call `.collect()`

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
# Pandas (eager)
df = pd.read_csv("data.csv")
df_filtered = df[df['age'] > 25]  # Executes immediately

# Moltres (lazy)
df = db.table("users").select()
df_filtered = df.where(col("age") > 25)  # Doesn't execute yet!
results = df_filtered.collect()  # Must call collect()

```

### Challenge 2: Boolean Values

**Problem**: SQLite uses INTEGER (0/1) instead of boolean

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Pandas
df[df['active'] == True]

# Moltres (SQLite)
df.where(col("active") == 1)  # Use 1 instead of True

# Moltres (PostgreSQL/MySQL)
df.where(col("active") == True)  # Can use True

```

### Challenge 3: String Operations

**Pandas:**
```python
import pandas as pd

# Assume df is already loaded
# df = pd.read_csv("data.csv")

df[df['email'].str.contains('@example.com')]
df['name'].str.upper()
```

**Moltres:**
```python
from moltres import connect
from moltres.expressions import functions as F
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

df.where(col("email").like("%@example.com"))
df.select(F.upper(col("name")).alias("name_upper"))

```

## Migration Checklist

- [ ] Replace `pd.read_csv()` with `Records.from_csv()` or `db.table().select()`
- [ ] Replace `df[condition]` with `df.where(condition)`
- [ ] Replace `df.groupby()` with `df.group_by()`
- [ ] Replace `df.merge()` with `df.join()`
- [ ] Add `.collect()` calls where results are needed
- [ ] Update boolean comparisons (use 1/0 for SQLite)
- [ ] Replace pandas string methods with Moltres functions
- [ ] Test with sample data before migrating production code

## Next Steps

- **Performance**: See [Performance Optimization Guide](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)
- **Patterns**: Check [Common Patterns Guide](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)
- **Best Practices**: Read [Best Practices Guide](https://moltres.readthedocs.io/en/latest/guides/best-practices.html)

