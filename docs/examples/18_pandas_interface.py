"""Pandas-style interface for Moltres DataFrames.

This example demonstrates how to use Moltres with a pandas-style API,
providing familiar pandas operations while maintaining SQL pushdown execution.

Required dependencies:
- moltres (required)
- pandas (optional, for pandas DataFrame return types): pip install pandas or pip install moltres[pandas]
"""

from __future__ import annotations

from typing import Any

from moltres import connect
from moltres.table.schema import column

# Check for pandas (optional - example works without it but collect() returns dicts instead of DataFrame)
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    if __name__ == "__main__":
        print("Note: pandas is not installed. Install with: pip install pandas")
        print("Or: pip install moltres[pandas]")
        print(
            "The example will still run, but collect() will return list of dicts instead of pandas DataFrame.\n"
        )


# Helper function to handle both pandas DataFrame and list of dicts
def print_results(results: Any, *fields: str) -> None:
    """Print results handling both pandas DataFrame and list of dicts."""
    try:
        import pandas as pd

        if isinstance(results, pd.DataFrame):
            for _, row in results.iterrows():
                values = [
                    row[field]
                    if isinstance(row[field], str)
                    else int(row[field])
                    if isinstance(row[field], (int, float))
                    else row[field]
                    for field in fields
                ]
                print(f"  {', '.join(str(v) for v in values)}")
        else:
            for row in results:
                values = [row[field] for field in fields]
                print(f"  {', '.join(str(v) for v in values)}")
    except ImportError:
        for row in results:
            values = [row[field] for field in fields]
            print(f"  {', '.join(str(v) for v in values)}")


def print_dict_results(results: Any) -> None:
    """Print results as dicts (for debugging)."""
    try:
        import pandas as pd

        if isinstance(results, pd.DataFrame):
            for _, row in results.iterrows():
                print(f"  {row.to_dict()}")
        else:
            for row in results:
                print(f"  {row}")
    except ImportError:
        for row in results:
            print(f"  {row}")


# Connect to database
db = connect("sqlite:///:memory:")

# Create tables and insert data
db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("age", "INTEGER"),
        column("country", "TEXT"),
    ],
).collect()

db.create_table(
    "orders",
    [
        column("id", "INTEGER", primary_key=True),
        column("user_id", "INTEGER"),
        column("amount", "REAL"),
        column("status", "TEXT"),
    ],
).collect()

from moltres.io.records import Records

Records.from_list(
    [
        {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
        {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
        {"id": 3, "name": "Charlie", "age": 35, "country": "USA"},
        {"id": 4, "name": "David", "age": 28, "country": "UK"},
    ],
    database=db,
).insert_into("users")

Records.from_list(
    [
        {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
        {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
        {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
        {"id": 4, "user_id": 3, "amount": 300.0, "status": "active"},
    ],
    database=db,
).insert_into("orders")

# ============================================================================
# Pandas-style DataFrame operations
# ============================================================================

# Create a PandasDataFrame
df = db.table("users").pandas()

# Column access
print("Column names:", df.columns)
# Output: Column names: ['id', 'name', 'age', 'country']
print()

# Select columns using pandas-style indexing
df_selected = df[["id", "name", "age"]]
print("Selected columns:", df_selected.columns)  # type: ignore[union-attr]
# Output: Selected columns: ['id', 'name', 'age']
print()

# Filter using query() method (pandas-style) - supports AND/OR keywords
df_filtered = df.query("age > 25")
results = df_filtered.collect()
print("Filtered results (age > 25):")
print_results(results, "name", "age", "country")
# Output:
#   Alice, 30, USA
#   Charlie, 35, USA
#   David, 28, UK
print()

# Query with AND/OR keywords (improved syntax)
df_filtered_usa = df.query("age > 25 and country == 'USA'")
results_usa = df_filtered_usa.collect()
print("Filtered results (age > 25 AND country == 'USA' using query):")
print_results(results_usa, "name", "age", "country")
# Output:
#   Alice, 30, USA
#   Charlie, 35, USA
print()

# Alternative: use loc with Column expressions
df_filtered_usa_loc = df.loc[(df["age"] > 25) & (df["country"] == "USA")]  # type: ignore[operator]
results_usa_loc = df_filtered_usa_loc.collect()
print("Filtered results (age > 25 AND country == 'USA' using loc):")
print_results(results_usa_loc, "name", "age", "country")
# Output:
#   Alice, 30, USA
#   Charlie, 35, USA
print()

# Filter using loc accessor
df_filtered2 = df.loc[df["age"] > 28]  # type: ignore[operator]
results2 = df_filtered2.collect()
print("Filtered results (age > 28 using loc):")
print_results(results2, "name", "age")
# Output:
#   Alice, 30
#   Charlie, 35
print()

# GroupBy operations (pandas-style)
print("GroupBy example:")
grouped = df.groupby("country")
result_df = grouped.count()
results = result_df.collect()
print("Count by country:")
print_dict_results(results)
# Output:
#   {'country': 'UK', 'count': 2}
#   {'country': 'USA', 'count': 2}
print()

# GroupBy with aggregation dictionary
df_orders = db.table("orders").pandas()
grouped_orders = df_orders.groupby("status")
agg_result = grouped_orders.agg(amount="sum")
results = agg_result.collect()
print("Sum of amount by status:")
print_dict_results(results)
# Output:
#   {'status': 'active', 'amount_sum': 600.0}
#   {'status': 'completed', 'amount_sum': 150.0}
print()

# Merge operations (pandas-style joins)
print("Merge example:")
df_users = db.table("users").pandas()[["id", "name"]]
df_orders = db.table("orders").pandas()[["user_id", "amount"]]  # type: ignore[assignment]
df_orders_renamed = df_orders.rename(columns={"user_id": "id"})

merged = df_users.merge(df_orders_renamed, on="id", how="inner")  # type: ignore[union-attr]
results = merged.collect()
print("Users merged with orders:")
# Show first 5 rows
try:
    import pandas as pd

    if isinstance(results, pd.DataFrame):
        for _, row in results.head(5).iterrows():
            print(f"  {row.to_dict()}")
    else:
        for row in results[:5]:
            print(f"  {row}")
except ImportError:
    for row in results[:5]:
        print(f"  {row}")
# Output:
#   {'id': 1, 'name': 'Alice', 'amount': 100.0}
#   {'id': 1, 'name': 'Alice', 'amount': 150.0}
#   {'id': 2, 'name': 'Bob', 'amount': 200.0}
#   {'id': 3, 'name': 'Charlie', 'amount': 300.0}
print()

# Sort values
df_sorted = df.sort_values("age", ascending=False)
results = df_sorted.collect()
print("Users sorted by age (descending):")
print_results(results, "name", "age")
# Output:
#   Charlie, 35
#   Alice, 30
#   David, 28
#   Bob, 25
print()

# Assign new columns
df_with_total = df.assign(age_plus_10=df["age"] + 10)  # type: ignore[operator]
results = df_with_total.collect()
print("Users with calculated column (age + 10):")
try:
    import pandas as pd

    if isinstance(results, pd.DataFrame):
        for _, row in results.iterrows():
            print(f"  {row['name']}, age {int(row['age'])}, age_plus_10 {int(row['age_plus_10'])}")
    else:
        for row in results:
            print(f"  {row['name']}, age {row['age']}, age_plus_10 {row['age_plus_10']}")
except ImportError:
    for row in results:
        print(f"  {row['name']}, age {row['age']}, age_plus_10 {row['age_plus_10']}")
# Output:
#   Alice, age 30, age_plus_10 40
#   Bob, age 25, age_plus_10 35
#   Charlie, age 35, age_plus_10 45
#   David, age 28, age_plus_10 38
print()

# Chaining operations
print("Chained operations example:")
result = (
    df[["id", "name", "age", "country"]].query("age > 25").sort_values("age").drop(columns=["id"])  # type: ignore[union-attr]
)
results = result.collect()
print("Filtered, sorted, and columns dropped:")
print_dict_results(results)
# Output (note: drop may require schema info to work in all cases):
#   {'id': 4, 'name': 'David', 'age': 28, 'country': 'UK'}
#   {'id': 1, 'name': 'Alice', 'age': 30, 'country': 'USA'}
#   {'id': 3, 'name': 'Charlie', 'age': 35, 'country': 'USA'}
print()

# ============================================================================
# New Features: String Accessor, Data Inspection, and More
# ============================================================================

# String accessor for text operations (all executed in SQL)
print("String accessor examples:")
name_col = df["name"]
# name_col is a PandasColumn when accessed as single column - type: ignore for examples
# String methods
print(f"  Upper case: {name_col.str.upper()}")  # type: ignore[union-attr]
print(f"  Lower case: {name_col.str.lower()}")  # type: ignore[union-attr]
print(f"  Contains 'Ali': {name_col.str.contains('Ali')}")  # type: ignore[union-attr]
print(f"  Starts with 'A': {name_col.str.startswith('A')}")  # type: ignore[union-attr]
print()

# Use string accessor in filtering - df["name"] returns PandasColumn
df_filtered_names = df[df["name"].str.contains("Ali")]  # type: ignore[union-attr]
results_names = df_filtered_names.collect()  # type: ignore[union-attr]
print("Users with 'Ali' in name:")
print_results(results_names, "name", "age")
# Output:
#   Alice, 30
print()

# Data inspection methods
print("Data inspection:")
print(f"  dtypes: {df.dtypes}")
# Output: {'id': 'int64', 'name': 'object', 'age': 'int64', 'country': 'object'}
print(f"  shape: {df.shape}")
# Output: (4, 4)
print(f"  empty: {df.empty}")
# Output: False
print()

# Convenience methods
print("Head (first 2 rows):")
head_df = df.head(2)
head_results = head_df.collect()
print_results(head_results, "name", "age")
# Output:
#   Alice, 30
#   Bob, 25
print()

# Value counts
print("Value counts for 'country':")
vc_results = df.value_counts("country")
print_dict_results(vc_results)
# Output:
#   {'country': 'USA', 'count': 2}
#   {'country': 'UK', 'count': 2}
print()

# Unique value counts
print("Unique values:")
unique_countries = df.nunique("country")
print(f"  Unique countries: {unique_countries}")
# Output: Unique countries: 2
print()

# Drop duplicates with subset
print("Drop duplicates on 'country':")
df_unique = df.drop_duplicates(subset=["country"])
results_unique = df_unique.collect()
print(f"  Unique rows: {len(results_unique)}")
print_results(results_unique, "name", "country")
print()

# Collect returns pandas DataFrame
try:
    import pandas as pd

    pdf = df.collect()
    print(f"Collected as pandas DataFrame: {type(pdf)}")
    print(f"Shape: {pdf.shape}")
    print(f"Columns: {list(pdf.columns)}")
    print()
    print("First few rows:")
    print(pdf.head())
    # Output:
    # Collected as pandas DataFrame: <class 'pandas.core.frame.DataFrame'>
    # Shape: (4, 4)
    # Columns: ['id', 'name', 'age', 'country']
    #
    # First few rows:
    #    id     name  age country
    # 0   1    Alice   30     USA
    # 1   2      Bob   25      UK
    # 2   3  Charlie   35     USA
    # 3   4    David   28      UK
except ImportError:
    print("pandas not installed - collect() will return list of dicts")
    results = df.collect()
    print(f"Collected as list: {len(results)} rows")

# ============================================================================
# Data Reshaping
# ============================================================================

print("\n" + "=" * 70)
print("Data Reshaping")
print("=" * 70)

# Create a table suitable for pivoting
from moltres.table.schema import column
from moltres.io.records import Records

db.create_table(
    "sales",
    [
        column("category", "TEXT"),
        column("status", "TEXT"),
        column("amount", "REAL"),
    ],
).collect()

Records(
    _data=[
        {"category": "A", "status": "active", "amount": 100.0},
        {"category": "A", "status": "inactive", "amount": 50.0},
        {"category": "B", "status": "active", "amount": 200.0},
    ],
    _database=db,
).insert_into("sales")

df_sales = db.table("sales").pandas()

# Pivot (note: requires pivot_values in underlying implementation)
print("\nPivot example:")
try:
    pivoted = df_sales.pivot(index="category", columns="status", values="amount", aggfunc="sum")
    results = pivoted.collect()
    print(f"  Pivoted: {len(results)} rows")
except Exception as e:
    print(f"  Pivot requires pivot_values: {type(e).__name__}")

# ============================================================================
# Sampling and Limiting
# ============================================================================

print("\n" + "=" * 70)
print("Sampling and Limiting")
print("=" * 70)

df = db.table("users").pandas()

# Sample rows
print("\nSample 2 rows:")
sampled = df.sample(n=2, random_state=42)
results = sampled.collect()
print(f"  Sampled: {len(results)} rows")

# Sample by fraction
print("\nSample 50% of rows:")
sampled_frac = df.sample(frac=0.5, random_state=42)
results_frac = sampled_frac.collect()
print(f"  Sampled: {len(results_frac)} rows")

# Limit
print("\nLimit to 2 rows:")
limited = df.limit(2)
results_limited = limited.collect()
print(f"  Limited: {len(results_limited)} rows")

# ============================================================================
# Concatenation
# ============================================================================

print("\n" + "=" * 70)
print("Concatenation")
print("=" * 70)

df1 = db.table("users").pandas()
df2 = db.table("users").pandas()

# Append
print("\nAppend:")
appended = df1.append(df2)
results = appended.collect()
print(f"  Appended: {len(results)} rows (should be 8)")

# Concatenate vertically
print("\nConcat vertical:")
concatenated = df1.concat(df2, axis=0)
results = concatenated.collect()
print(f"  Concatenated: {len(results)} rows")

# Concatenate horizontally
print("\nConcat horizontal:")
df_users = db.table("users").pandas().select("id", "name")
df_orders = db.table("orders").pandas().select("id", "amount")
concatenated_h = df_users.concat(df_orders, axis=1)
results = concatenated_h.collect()
print(f"  Concatenated: {len(results)} rows (cartesian product)")

# ============================================================================
# Advanced Filtering
# ============================================================================

print("\n" + "=" * 70)
print("Advanced Filtering")
print("=" * 70)

df = db.table("users").pandas()

# isin
print("\nisin filter:")
filtered = df.isin({"age": [30, 35]})
results = filtered.collect()
print(f"  Filtered: {len(results)} rows")
print_results(results, "name", "age")

# between
print("\nbetween filter:")
filtered = df.between(left=25, right=35, inclusive="both")
results = filtered.collect()
print(f"  Filtered: {len(results)} rows")
print_results(results, "name", "age")

# ============================================================================
# SQL Expressions and CTEs
# ============================================================================

print("\n" + "=" * 70)
print("SQL Expressions and CTEs")
print("=" * 70)

df = db.table("users").pandas()

# Select with SQL expressions
print("\nSelect with SQL expressions:")
selected = df.select_expr("id", "name", "age", "age * 2 as double_age")
results = selected.collect()
print(f"  Selected: {len(results)} rows")
if len(results) > 0:
    # Handle both pandas DataFrame and list of dicts
    if PANDAS_AVAILABLE and hasattr(results, "iloc"):
        first_row = results.iloc[0]
        double_age = (
            first_row.get("double_age", "N/A")
            if hasattr(first_row, "get")
            else first_row["double_age"]
        )
    else:
        double_age = results[0].get("double_age", "N/A")
    print(f"  First row double_age: {double_age}")

# CTE
print("\nCommon Table Expression:")
cte_df = df.query("age > 25").cte("adults")
results = cte_df.collect()
print(f"  CTE 'adults': {len(results)} rows")

db.close()
