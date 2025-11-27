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
df = pd.read_csv("data.csv")

# Read from SQL
df = pd.read_sql("SELECT * FROM users", connection)
```

**Moltres:**
```python
from moltres import connect

db = connect("sqlite:///example.db")

# Read from CSV (returns Records, not DataFrame)
from moltres.io.records import Records
records = Records.from_csv("data.csv", database=db)
records.insert_into("users")

# Read from SQL table (returns DataFrame)
df = db.table("users").select()
results = df.collect()  # Returns list of dicts by default

# Or get as pandas DataFrame
results = df.collect(format="pandas")  # Returns pd.DataFrame
```

### 2. Filtering

**Pandas:**
```python
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
from moltres import col

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
# Ascending
df_sorted = df.sort_values('age')

# Descending
df_sorted = df.sort_values('age', ascending=False)

# Multiple columns
df_sorted = df.sort_values(['active', 'age'], ascending=[False, True])

```

**Moltres:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
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
# Add new column
df['new_col'] = df['col1'] + df['col2']

# Modify existing
df['age'] = df['age'] + 1

# Conditional column
df['category'] = df['age'].apply(lambda x: 'adult' if x >= 18 else 'minor')
```

**Moltres:**
```python
from moltres import connect
from moltres.expressions import functions as F
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

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
# Results are already DataFrames
df = pd.read_csv("data.csv")
result = df.groupby('country').sum()
# result is a DataFrame

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

# Get as pandas DataFrame
results = df.collect(format="pandas")
# results is: pd.DataFrame

# Or convert to pandas after
import pandas as pd
results = df.collect()
df_pandas = pd.DataFrame(results)

```

## Using Pandas-Style Interface

Moltres also provides a pandas-style interface for easier migration:

**See also:** [Pandas interface examples](https://github.com/eddiethedean/moltres/blob/main/examples/18_pandas_interface.py), [Pandas/Polars integration examples](https://github.com/eddiethedean/moltres/blob/main/examples/15_pandas_polars_dataframes.py), and the [Pandas Interface Guide](https://github.com/eddiethedean/moltres/blob/main/guides/09-pandas-interface.md)

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
# Get pandas-style DataFrame
df = db.table("users").pandas()

# Use pandas-style methods
df_filtered = df.query('age > 25 and active == 1')
df_grouped = df.groupby('country').agg({'age': 'mean', 'id': 'count'})
df_merged = df1.merge(df2, left_on='id', right_on='user_id', how='inner')

# Still need to call collect()
results = df_filtered.collect()

```

## Handling Large Datasets

**Pandas limitation:**
```python
# This loads entire dataset into memory
df = pd.read_csv("huge_file.csv")  # May fail if file > RAM
df_filtered = df[df['age'] > 25]
```

**Moltres advantage:**
```python
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

- **Performance**: See [Performance Optimization Guide](https://github.com/eddiethedean/moltres/blob/main/guides/04-performance-optimization.md)
- **Patterns**: Check [Common Patterns Guide](https://github.com/eddiethedean/moltres/blob/main/guides/05-common-patterns.md)
- **Best Practices**: Read [Best Practices Guide](https://github.com/eddiethedean/moltres/blob/main/guides/08-best-practices.md)

