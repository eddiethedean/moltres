# Using the Pandas-Style Interface in Moltres

This guide covers Moltres's pandas-style interface, which provides a familiar API for users coming from pandas while maintaining SQL pushdown execution.

## Overview

Moltres offers a pandas-style interface via the `.pandas()` method that returns a `PandasDataFrame`. This interface provides:

- Familiar pandas operations (`query()`, `groupby()`, `merge()`, etc.)
- String accessor for text operations (`df['col'].str`)
- Data inspection methods (`dtypes`, `shape`, `head()`, `tail()`, etc.)
- All operations execute in SQL with lazy evaluation

**See also:** [Pandas interface examples](https://github.com/eddiethedean/moltres/blob/main/examples/18_pandas_interface.py)

## Getting Started

### Creating a PandasDataFrame

Use the `.pandas()` method on any Moltres DataFrame:

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

# Get pandas-style DataFrame
df = db.table("users").pandas()
print(f"Columns: {df.columns}")
# Output: Columns: ['id', 'name', 'age', 'country']
```

### Column Access

Access columns using pandas-style indexing:

```python
from moltres import connect
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

df = db.table("users").pandas()

# Single column - returns PandasColumn for expressions
name_col = df['name']
print(f"Type: {type(name_col).__name__}")  # PandasColumn

# Multiple columns - returns PandasDataFrame
df_selected = df[['id', 'name', 'age']]
print(f"Selected columns: {df_selected.columns}")
# Output: Selected columns: ['id', 'name', 'age']
```

## Filtering with Query

The `query()` method provides pandas-style filtering with improved syntax:

```python
from moltres import connect
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

df = db.table("users").pandas()

# Simple comparison
results = df.query("age > 25").collect()
print(f"Found {len(results)} users over 25")

# Multiple conditions with AND keyword
results = df.query("age > 25 and country == 'USA'").collect()
print(f"Found {len(results)} users over 25 from USA")

# OR keyword
results = df.query("age > 30 or country == 'UK'").collect()
print(f"Found {len(results)} users")

# Supports both = and == for equality
results = df.query("age == 30").collect()
print(f"Found {len(results)} users age 30")
```

### Query Syntax

- Use comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Use `and`/`or` keywords (in addition to `&`/`|`)
- Support for both `=` and `==` for equality
- Column names must match your schema

**See also:** [Query examples](https://github.com/eddiethedean/moltres/blob/main/examples/18_pandas_interface.py#L122)

## String Accessor

The string accessor (`.str`) provides pandas-style string operations that execute in SQL:

### Basic String Operations

```python
# Convert case
df['name'].str.upper()  # Uppercase
df['name'].str.lower()  # Lowercase

# Trimming
df['name'].str.strip()   # Trim both sides
df['name'].str.lstrip()  # Trim left
df['name'].str.rstrip()  # Trim right
```

### String Matching

```python
# Check if string contains pattern
df[df['name'].str.contains('Ali')]  # Filter rows

# Check prefix/suffix
df[df['name'].str.startswith('A')]  # Names starting with 'A'
df[df['name'].str.endswith('e')]    # Names ending with 'e'

# Case-insensitive search
df[df['name'].str.contains('alice', case=False)]
```

### String Replacement

```python
# Replace substring
df['name'].str.replace('Alice', 'Alicia')
```

### String Length

```python
# Get string length
df['name'].str.len()
```

**Note:** All string operations compile to SQL functions and execute in the database.

## Data Inspection

### Column Types

Get pandas-compatible dtype information:

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
print(df.dtypes)
# Output: {'id': 'int64', 'name': 'object', 'age': 'int64', 'country': 'object'}

```

The `dtypes` property maps SQL types to pandas dtypes:
- Integer types → `int64`
- Float types → `float64`
- Text types → `object`
- Boolean types → `bool`
- Date/Time types → `datetime64[ns]`

### Shape and Size

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Get dimensions (cached after first call)
print(df.shape)  # (rows, columns)

# Check if empty
print(df.empty)  # False

```

**Note:** `shape` and `empty` require query execution and are cached after the first call.

### Head and Tail

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Get first n rows
df.head(5)  # Returns PandasDataFrame with first 5 rows

# Get last n rows (requires sorting - may be expensive)
df.tail(5)  # Returns PandasDataFrame with last 5 rows

```

### Unique Values

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Count unique values in a column
df.nunique('country')  # Returns int

# Count unique values for all columns
df.nunique()  # Returns dict mapping column names to counts

```

### Value Counts

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Count frequency of values in a column
df.value_counts('country')
# Returns list of dicts: [{'country': 'USA', 'count': 2}, ...]

# Normalized counts (as proportions)
df.value_counts('country', normalize=True)

```

## GroupBy Operations

GroupBy provides pandas-style aggregations:

### Basic Aggregations

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Group by column
grouped = df.groupby('country')

# Convenience aggregations
grouped.sum()     # Sum of all numeric columns
grouped.mean()    # Mean of all numeric columns
grouped.min()     # Minimum of all numeric columns
grouped.max()     # Maximum of all numeric columns
grouped.count()   # Count of rows per group
grouped.nunique() # Count distinct for each column

```

### Dictionary Aggregations

```python
# Specify aggregations per column
grouped.agg({
    'age': 'mean',
    'id': 'count'
})
```

### Multiple Grouping Columns

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Group by multiple columns
df.groupby(['country', 'status'])

```

## Merging DataFrames

Pandas-style merge operations:

```python
# Inner join on common column
df1.merge(df2, on='id')

# Left join
df1.merge(df2, on='id', how='left')

# Different column names
df1.merge(df2, left_on='id', right_on='user_id')

# Multiple join keys
df1.merge(df2, on=['id', 'status'])
```

### Join Types

- `inner` - Inner join (default)
- `left` - Left outer join
- `right` - Right outer join
- `outer` / `full` / `full_outer` - Full outer join

## Sorting

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Sort by single column
df.sort_values('age')

# Sort descending
df.sort_values('age', ascending=False)

# Sort by multiple columns
df.sort_values(['country', 'age'], ascending=[True, False])

```

## Dropping Duplicates

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Remove all duplicates
df.drop_duplicates()

# Remove duplicates based on subset of columns
df.drop_duplicates(subset=['country'])

# Keep first or last duplicate
df.drop_duplicates(subset=['country'], keep='first')  # default
df.drop_duplicates(subset=['country'], keep='last')

```

## Assigning New Columns

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Add new column with expression
df.assign(age_plus_10=df['age'] + 10)

# Multiple columns
df.assign(
    age_plus_10=df['age'] + 10,
    is_adult=df['age'] >= 18
)

```

## Renaming Columns

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Rename columns
df.rename(columns={'old_name': 'new_name'})

# Multiple renames
df.rename(columns={
    'id': 'user_id',
    'name': 'user_name'
})

```

## Dropping Columns

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Drop single column
df.drop(columns='age')

# Drop multiple columns
df.drop(columns=['age', 'country'])

```

## Chaining Operations

All operations can be chained for complex pipelines:

```python
result = (
    df[['id', 'name', 'age', 'country']]
    .query("age > 25")
    .sort_values('age')
    .drop(columns=['id'])
)

results = result.collect()
```

## Boolean Indexing

Use boolean conditions for filtering:

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Using loc with boolean condition
df.loc[df['age'] > 25]

# Multiple conditions with & and |
df.loc[(df['age'] > 25) & (df['country'] == 'USA')]
df.loc[(df['age'] < 20) | (df['age'] > 65)]

```

**Note:** Use parentheses around conditions when using `&` and `|`.

## Collecting Results

All operations are lazy - call `.collect()` to execute:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Returns pandas DataFrame (if pandas installed)
results = df.collect()

# Returns list of dicts if pandas not available
results = df.collect()

```

### Streaming Results

For large datasets:

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Stream results in chunks
for chunk in df.collect(stream=True):
    # Process chunk
    process_chunk(chunk)

```

## Error Handling

The pandas interface provides helpful error messages:

```python
# Missing column - provides suggestions
try:
    df['nonexistent_column']
except ValidationError as e:
    print(e.suggestion)  # "Did you mean 'name'?"
```

## Limitations and Differences from Pandas

### Key Differences

1. **Lazy Evaluation**: Operations don't execute until `.collect()` is called
2. **SQL Execution**: All operations compile to SQL and run in the database
3. **No Index**: Moltres doesn't support pandas-style indexes
4. **Limited iloc**: Only boolean indexing is supported for `iloc`
5. **Type System**: Uses SQL types, mapped to pandas dtypes

### Not Yet Supported

- Index-based operations (no index support)
- Complex pandas-style operations that don't map well to SQL
- Some advanced pandas features (check documentation for updates)

## Best Practices

### 1. Use Query for Simple Filtering

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Good - clear and readable
df.query("age > 25 and country == 'USA'")

# Also good - more flexible
df.loc[(df['age'] > 25) & (df['country'] == 'USA')]

```

### 2. Leverage String Accessor

```python
# Good - executed in SQL
df[df['name'].str.contains('Ali')]

# Avoid - would require materialization
# df[df['name'].apply(lambda x: 'Ali' in x)]  # Not supported
```

### 3. Cache Expensive Operations

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Shape is cached after first call
rows, cols = df.shape  # Executes query
rows, cols = df.shape  # Uses cached value

```

### 4. Use Appropriate Join Types

```python
# Choose the right join type for your use case
df1.merge(df2, on='id', how='inner')   # Only matching rows
df1.merge(df2, on='id', how='left')    # All left rows
df1.merge(df2, on='id', how='outer')   # All rows from both
```

### 5. Validate Early

Column validation happens automatically:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Error caught early with helpful message
df.query("nonexistent > 5")  # Raises ValidationError with suggestion

```

## Performance Considerations

1. **Shape/Empty**: Cached after first computation, but requires full query execution
2. **Tail**: Requires sorting entire dataset - can be expensive for large tables
3. **Value Counts**: Executes aggregation query - efficient for large datasets
4. **String Operations**: All executed in SQL - very efficient

## Examples

**See also:** [Complete pandas interface example](https://github.com/eddiethedean/moltres/blob/main/examples/18_pandas_interface.py)

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

db = connect("sqlite:///example.db")

# Setup
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("country", "TEXT"),
]).collect()

from moltres.io.records import Records
Records.from_list([
    {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
    {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
    {"id": 3, "name": "Charlie", "age": 35, "country": "USA"},
], database=db).insert_into("users")

# Use pandas interface
df = db.table("users").pandas()

# Query with AND keyword
results = df.query("age > 25 and country == 'USA'").collect()

# String operations
filtered = df[df['name'].str.contains('Ali')].collect()

# GroupBy aggregations
grouped = df.groupby('country')
summary = grouped.mean().collect()

# Data inspection
print(df.dtypes)  # Column types
print(df.shape)   # Dimensions
print(df.nunique('country'))  # Unique values

```

## Next Steps

- **Migration**: See [Migrating from Pandas Guide](https://github.com/eddiethedean/moltres/blob/main/guides/02-migrating-from-pandas.md) for detailed migration patterns
- **Examples**: Check [Pandas interface examples](https://github.com/eddiethedean/moltres/blob/main/examples/18_pandas_interface.py) for more code samples
- **Performance**: Read [Performance Optimization Guide](https://github.com/eddiethedean/moltres/blob/main/guides/04-performance-optimization.md) for efficiency tips
- **Best Practices**: See [Best Practices Guide](https://github.com/eddiethedean/moltres/blob/main/guides/08-best-practices.md) for coding standards

