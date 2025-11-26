"""Pandas-style interface for Moltres DataFrames.

This example demonstrates how to use Moltres with a pandas-style API,
providing familiar pandas operations while maintaining SQL pushdown execution.
"""

from __future__ import annotations

from typing import Any

from moltres import connect
from moltres.table.schema import column


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

# Filter using query() method (pandas-style) - numeric comparisons work well
df_filtered = df.query("age > 25")
results = df_filtered.collect()
print("Filtered results (age > 25):")
print_results(results, "name", "age", "country")
# Output:
#   Alice, 30, USA
#   Charlie, 35, USA
#   David, 28, UK
print()

# For complex filtering with multiple conditions, use Column expressions or chain loc
df_filtered_usa = df.loc[(df["age"] > 25) & (df["country"] == "USA")]  # type: ignore[operator]
results_usa = df_filtered_usa.collect()
print("Filtered results (age > 25 AND country == 'USA' using loc):")
print_results(results_usa, "name", "age", "country")
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
    for row in results[:5]:  # type: ignore[assignment]
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
    for row in results:  # type: ignore[assignment]
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

db.close()
