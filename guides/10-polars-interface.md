# Using the Polars-Style Interface in Moltres

This guide covers Moltres's Polars LazyFrame-style interface, which provides a familiar API for users coming from Polars while maintaining SQL pushdown execution.

## Overview

Moltres offers a Polars-style interface via the `.polars()` method that returns a `PolarsDataFrame`. This interface provides:

- Familiar Polars LazyFrame operations (`select()`, `filter()`, `group_by()`, `join()`, etc.)
- Expression-based API for column operations
- Data inspection methods (`schema`, `columns`, `width`, `height`)
- All operations execute in SQL with lazy evaluation

**See also:** [Polars interface examples](https://moltres.readthedocs.io/en/latest/examples/19_polars_interface.html)

## Getting Started

### Creating a PolarsDataFrame

Use the `.polars()` method on any Moltres table:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create a table and insert sample data
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("country", "TEXT"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
    {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
], database=db).insert_into("users")

# Get Polars-style DataFrame
df = db.table("users").polars()
print(f"Columns: {df.columns}")
# Output: Columns: ['id', 'name', 'age', 'country']
```

### Column Access

Access columns using Polars-style indexing:

```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

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

df = db.table("users").polars()

# Single column - returns Column expression
name_col = df['name']
print(f"Type: {type(name_col).__name__}")  # Column

# Multiple columns - returns PolarsDataFrame
df_selected = df[['id', 'name', 'age']]
print(f"Selected columns: {df_selected.columns}")
# Output: Selected columns: ['id', 'name', 'age']

# Boolean indexing
df_filtered = df[df['age'] > 25]
results = df_filtered.collect()
print(f"Found {len(results)} users over 25")
```

## Filtering

The `filter()` method provides Polars-style filtering using Column expressions:

```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

db = connect("sqlite:///:memory:")
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("country", "TEXT"),
]).collect()

Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
    {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
    {"id": 3, "name": "Charlie", "age": 35, "country": "USA"},
], database=db).insert_into("users")

df = db.table("users").polars()

# Simple comparison
results = df.filter(col("age") > 25).collect()
print(f"Found {len(results)} users over 25")

# Multiple conditions with & operator
results = df.filter((col("age") > 25) & (col("country") == "USA")).collect()
print(f"Found {len(results)} users over 25 from USA")

# OR operator
results = df.filter((col("age") > 30) | (col("country") == "UK")).collect()
print(f"Found {len(results)} users")

# Output: Found 2 users over 25
# Output: Found 2 users over 25 from USA
# Output: Found 2 users
```

## Selecting Columns

Use `select()` to choose specific columns or create computed columns:

```python
from moltres import connect, col
from moltres.expressions import functions as F

df = db.table("users").polars()

# Select specific columns
df_selected = df.select("id", "name", "age")
results = df_selected.collect()

# Select with expressions
df_with_expr = df.select(
    "id",
    "name",
    (col("age") * 2).alias("double_age"),
    F.upper(col("name")).alias("name_upper")
)
results = df_with_expr.collect()
```

## Adding and Modifying Columns

Use `with_columns()` to add or modify columns:

```python
from moltres import connect, col

df = db.table("users").polars()

# Add a single column
df_new = df.with_columns((col("age") + 10).alias("age_plus_10"))

# Add multiple columns
df_new = df.with_columns(
    (col("age") + 10).alias("age_plus_10"),
    (col("age") * 2).alias("double_age")
)

# Using tuple syntax
df_new = df.with_columns(("age_plus_10", col("age") + 10))

# with_column() is an alias for single column
df_new = df.with_column((col("age") + 10).alias("age_plus_10"))
```

## GroupBy Operations

Polars-style groupby uses expression-based aggregations:

```python
from moltres import connect, col
from moltres.expressions import functions as F

df = db.table("orders").polars()

# Basic groupby with aggregation
grouped = df.group_by("status")
result = grouped.agg(F.sum(col("amount")).alias("total_amount"))

# Multiple aggregations
result = grouped.agg(
    F.sum(col("amount")).alias("total"),
    F.avg(col("amount")).alias("average"),
    F.count().alias("count")
)

# Convenience methods
result = grouped.sum()    # Sum all numeric columns
result = grouped.mean()   # Mean of all numeric columns
result = grouped.count()  # Count rows per group
result = grouped.min()    # Minimum values
result = grouped.max()    # Maximum values
result = grouped.std()    # Standard deviation
result = grouped.var()    # Variance
result = grouped.n_unique()  # Count distinct values
```

## Joins

Polars-style joins support multiple join types:

```python
from moltres import connect

df1 = db.table("users").polars()
df2 = db.table("orders").polars()

# Inner join with same column name
joined = df1.join(df2, on="id", how="inner")

# Join with different column names
joined = df1.join(df2, left_on="id", right_on="user_id", how="left")

# Multiple join keys
joined = df1.join(df2, on=["id", "status"], how="inner")

# Join types: 'inner', 'left', 'right', 'outer', 'anti', 'semi'
joined = df1.join(df2, on="id", how="anti")  # Rows in left not in right
joined = df1.join(df2, on="id", how="semi")   # Rows in left with matches in right
```

## Sorting

Use `sort()` to order rows:

```python
from moltres import connect

df = db.table("users").polars()

# Sort by single column
df_sorted = df.sort("age")

# Sort descending
df_sorted = df.sort("age", descending=True)

# Sort by multiple columns
df_sorted = df.sort("country", "age", descending=[False, True])
```

## Data Manipulation

### Dropping Columns

```python
df = db.table("users").polars()

# Drop single column
df_dropped = df.drop("age")

# Drop multiple columns
df_dropped = df.drop("age", "country")
```

### Renaming Columns

```python
df = db.table("users").polars()

# Rename columns
df_renamed = df.rename({"name": "full_name", "age": "years"})
```

### Removing Duplicates

```python
df = db.table("users").polars()

# Remove duplicates on all columns
df_unique = df.unique()

# Remove duplicates on specific columns
df_unique = df.unique(subset=["country"])

# distinct() is an alias for unique()
df_unique = df.distinct()
```

### Handling Nulls

```python
df = db.table("users").polars()

# Drop rows with nulls
df_clean = df.drop_nulls()

# Drop nulls in specific columns
df_clean = df.drop_nulls(subset=["name", "age"])

# Fill null values
df_filled = df.fill_null(value=0, subset=["age"])
df_filled = df.fill_null(value="Unknown", subset=["name"])
```

## Limiting and Sampling

```python
df = db.table("users").polars()

# Limit rows
df_limited = df.limit(10)

# Get first n rows
df_head = df.head(5)

# Get last n rows
df_tail = df.tail(5)

# Random sampling
df_sample = df.sample(n=100, seed=42)
df_sample = df.sample(fraction=0.1, seed=42)
```

## Data Inspection

```python
df = db.table("users").polars()

# Get column names
print(df.columns)  # ['id', 'name', 'age', 'country']

# Get number of columns
print(df.width)    # 4

# Get number of rows (requires query execution)
print(df.height)   # 10

# Get schema as (name, dtype) tuples
print(df.schema)   # [('id', 'Int64'), ('name', 'Utf8'), ('age', 'Int64'), ('country', 'Utf8')]

# Lazy() returns self (for API compatibility)
df_lazy = df.lazy()  # Returns self
```

## Collecting Results

The `collect()` method executes the query and returns results:

```python
df = db.table("users").polars()

# Collect as Polars DataFrame (if polars is installed)
results = df.collect()  # Returns pl.DataFrame

# Collect as list of dicts (if polars not installed)
results = df.collect()  # Returns List[Dict[str, Any]]

# Streaming collection
for chunk in df.collect(stream=True):
    process(chunk)  # Each chunk is a Polars DataFrame

# Fetch first n rows without full collection
results = df.fetch(10)  # Returns pl.DataFrame with first 10 rows
```

## Chaining Operations

Polars-style operations can be chained together:

```python
from moltres import connect, col
from moltres.expressions import functions as F

df = db.table("users").polars()

# Chain multiple operations
result = (
    df.filter(col("age") > 25)
    .select("id", "name", "age", "country")
    .sort("age", descending=True)
    .drop("id")
    .limit(10)
)

results = result.collect()
```

## Comparison with Polars LazyFrame

Moltres's Polars interface closely matches Polars LazyFrame API:

| Polars LazyFrame | Moltres PolarsDataFrame | Notes |
|-----------------|------------------------|-------|
| `df.lazy()` | `df.polars()` | Create lazy frame |
| `df.filter()` | `df.filter()` | Filter rows |
| `df.select()` | `df.select()` | Select columns |
| `df.with_columns()` | `df.with_columns()` | Add/modify columns |
| `df.group_by()` | `df.group_by()` | Group operations |
| `df.join()` | `df.join()` | Join DataFrames |
| `df.sort()` | `df.sort()` | Sort rows |
| `df.collect()` | `df.collect()` | Execute query |
| `df.schema` | `df.schema` | Get schema |
| `df.width` | `df.width` | Number of columns |
| `df.height` | `df.height` | Number of rows |

## Key Differences

1. **SQL Execution**: All operations compile to SQL and execute on the database, not in memory
2. **Expression-Based**: Uses Moltres Column expressions instead of Polars expressions
3. **Optional Polars**: Works without polars installed (returns list of dicts from `collect()`)
4. **Database-Bound**: Operations require a database connection

## Reading Files (Polars-Style)

Moltres provides Polars-style file reading methods that return `PolarsDataFrame`:

```python
from moltres import connect
from moltres.table.schema import column

db = connect("sqlite:///:memory:")

# Scan CSV file
df = db.scan_csv("data.csv", header=True)
results = df.collect()

# Scan JSON file
df = db.scan_json("data.json")
results = df.collect()

# Scan JSONL file
df = db.scan_jsonl("data.jsonl")
results = df.collect()

# Scan Parquet file
df = db.scan_parquet("data.parquet")
results = df.collect()

# Scan text file
df = db.scan_text("data.txt", column_name="line")
results = df.collect()

# Scan with explicit schema
schema = [
    column("id", "INTEGER"),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]
df = db.scan_csv("data.csv", schema=schema, header=True)

# Scan with options
df = db.scan_csv("data.csv", header=True, delimiter=";")

# Alternative: Use existing API with .polars()
df = db.read.csv("data.csv").polars()
```

## Writing Files (Polars-Style)

Moltres provides Polars-style file writing methods on `PolarsDataFrame`:

```python
from moltres import connect, col

db = connect("sqlite:///:memory:")
df = db.table("users").polars()

# Write to CSV
df.write_csv("output.csv", header=True)

# Write to JSON
df.write_json("output.json")

# Write to JSONL
df.write_jsonl("output.jsonl")

# Write to Parquet
df.write_parquet("output.parquet")

# Write with options
df.write_csv("output.csv", mode="append", header=True, delimiter=",")

# Write filtered data
df.filter(col("age") > 25).write_json("filtered.json")

# Write with different modes
df.write_csv("output.csv", mode="overwrite")  # Default
df.write_csv("output.csv", mode="append")
df.write_csv("output.csv", mode="error_if_exists")
```

## String Operations (`.str` namespace)

Moltres provides Polars-style string operations via the `.str` accessor:

```python
from moltres import connect, col

db = connect("sqlite:///:memory:")
df = db.table("users").polars()

# String transformations
df.with_columns(df['name'].str.upper().alias('name_upper'))
df.with_columns(df['name'].str.lower().alias('name_lower'))
df.with_columns(df['name'].str.strip().alias('name_trimmed'))

# String matching
df.filter(df['name'].str.contains('Alice'))
df.filter(df['name'].str.starts_with('A'))
df.filter(df['name'].str.ends_with('e'))

# String replacement
df.with_columns(df['name'].str.replace('Alice', 'Alicia').alias('name_replaced'))

# String length
df.with_columns(df['name'].str.len().alias('name_length'))
```

## DateTime Operations (`.dt` namespace)

Moltres provides Polars-style datetime operations via the `.dt` accessor:

```python
from moltres import connect, col

db = connect("sqlite:///:memory:")
df = db.table("events").polars()

# Extract date components
df.with_columns(df['event_date'].dt.year().alias('year'))
df.with_columns(df['event_date'].dt.month().alias('month'))
df.with_columns(df['event_date'].dt.day().alias('day'))
df.with_columns(df['event_date'].dt.hour().alias('hour'))
df.with_columns(df['event_date'].dt.minute().alias('minute'))
df.with_columns(df['event_date'].dt.second().alias('second'))
df.with_columns(df['event_date'].dt.quarter().alias('quarter'))
df.with_columns(df['event_date'].dt.week().alias('week'))
df.with_columns(df['event_date'].dt.day_of_week().alias('day_of_week'))
df.with_columns(df['event_date'].dt.day_of_year().alias('day_of_year'))

# Filter by date components
df.filter(df['event_date'].dt.year() > 2020)
df.filter(df['event_date'].dt.month() == 12)
```

## Window Functions

Moltres supports Polars-style window functions using `.over()`:

```python
from moltres import connect, col
from moltres.expressions import functions as F

db = connect("sqlite:///:memory:")
df = db.table("sales").polars()

# Row number
df.with_columns(F.row_number().over().alias('row_num'))

# Rank with partitioning
df.with_columns(
    F.rank().over(partition_by=col('category')).alias('rank_by_category')
)

# Dense rank with ordering
df.with_columns(
    F.dense_rank().over(
        partition_by=col('category'),
        order_by=col('amount').desc()
    ).alias('dense_rank')
)

# Aggregations over windows
df.with_columns(
    F.sum(col('amount')).over(partition_by=col('category')).alias('category_total')
)
```

## Conditional Expressions (`when().then().otherwise()`)

Moltres supports Polars-style conditional expressions:

```python
from moltres import connect, col
from moltres.expressions import functions as F

db = connect("sqlite:///:memory:")
df = db.table("users").polars()

# Simple when/then/otherwise
df.with_columns(
    F.when(col("age") >= 18, "adult")
    .otherwise("minor")
    .alias("category")
)

# Multiple conditions
df.with_columns(
    F.when(col("age") >= 65, "senior")
    .when(col("age") >= 30, "adult")
    .otherwise("young")
    .alias("age_group")
)
```

## Additional Operations

### Explode and Unnest

```python
# Explode array/JSON columns
df.explode('tags')
df.explode(['tags', 'categories'])

# Unnest struct columns
df.unnest('struct_col')
```

### Pivot

```python
# Pivot DataFrame
df.pivot(
    values='amount',
    index='category',
    columns='status',
    aggregate_function='sum'
)
```

### Slice

```python
# Slice rows
df.slice(10, 5)  # Rows 10-14
df.slice(10)  # All rows from 10 onwards
```

### Utility Methods

```python
# Sample every nth row
df.gather_every(10)  # Every 10th row
df.gather_every(5, offset=2)  # Every 5th row starting from row 2

# Compute quantiles
df.quantile(0.5)  # Median
df.quantile([0.25, 0.5, 0.75])  # Quartiles

# Descriptive statistics
df.describe()  # Count, mean, std, min, max for numeric columns

# Explain query plan
print(df.explain())  # Show SQL query plan

# Add row number
df.with_row_count("row_id")  # Add row number column
df.with_row_count("row_id", offset=10)  # Start from 10
```

## Set Operations

Moltres provides Polars-style set operations for combining DataFrames:

```python
from moltres import connect

db = connect("sqlite:///:memory:")
df1 = db.table("users").polars()
df2 = db.table("users2").polars()

# Concatenate vertically (union all)
df1.concat(df2, how="vertical")
df1.vstack(df2)  # Alias for concat

# Union operations
df1.union(df2)  # Union distinct (default)
df1.union(df2, distinct=False)  # Union all

# Intersection
df1.intersect(df2)  # Common rows only

# Difference
df1.difference(df2)  # Rows in df1 but not in df2

# Cross join (Cartesian product)
df1.cross_join(df2)  # All combinations
```

## SQL Expression Selection

Moltres supports SQL expression selection for convenience:

```python
# Select with SQL expressions
df.select_expr("id", "name", "age * 2 as double_age", "UPPER(name) as name_upper")

# Useful for complex expressions
df.select_expr("amount * 1.1 as with_tax", "CASE WHEN age > 18 THEN 'adult' ELSE 'minor' END as category")
```

## Common Table Expressions (CTEs)

Moltres supports CTEs for complex queries:

```python
from moltres import connect, col

db = connect("sqlite:///:memory:")

# Create a CTE
cte_df = db.table("orders").polars().filter(col("amount") > 100).cte("high_value_orders")

# Query the CTE
result = cte_df.select().collect()

# Recursive CTE
initial = db.table("seed").polars()
recursive = initial.select(...)  # Recursive part that references the CTE
fib_cte = initial.with_recursive("fib", recursive, union_all=True)
```

## Additional Utility Methods

```python
# Rename columns (alias for rename)
df.with_columns_renamed({"old_name": "new_name"})

# Add row number column
df.with_row_count("row_nr")
df.with_row_count("row_id", offset=10)
```

## Best Practices

1. **Use Column Expressions**: Prefer `col("age") > 25` over string-based filters
2. **Chain Operations**: Build complex queries by chaining operations
3. **Lazy Evaluation**: Operations are lazy until `collect()` is called
4. **Type Safety**: Column validation helps catch errors early
5. **SQL Pushdown**: All operations execute in SQL for optimal performance
6. **File I/O**: Use `scan_*` methods for reading and `write_*` methods for writing
7. **String/DateTime Accessors**: Use `.str` and `.dt` for convenient column operations
8. **Window Functions**: Use `.over()` for partitioned aggregations and rankings

## See Also

- [Polars interface examples](https://moltres.readthedocs.io/en/latest/examples/19_polars_interface.html)
- [Pandas interface guide](https://moltres.readthedocs.io/en/latest/guides/pandas-interface.html)
- [Main documentation](https://moltres.readthedocs.io/en/latest/)

