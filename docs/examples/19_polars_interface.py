"""Polars-style interface for Moltres DataFrames.

This example demonstrates how to use Moltres with a Polars LazyFrame-style API,
providing familiar Polars operations while maintaining SQL pushdown execution.

Required dependencies:
- moltres (required)
- polars (optional, for polars DataFrame return types): pip install polars or pip install moltres[polars]
"""

from __future__ import annotations

from typing import Any

from moltres import connect, col
from moltres.expressions import functions as F
from moltres.table.schema import column

# Check for polars (optional - example works without it but collect() returns dicts instead of DataFrame)
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    if __name__ == "__main__":
        print("Note: polars is not installed. Install with: pip install polars")
        print("Or: pip install moltres[polars]")
        print(
            "The example will still run, but collect() will return list of dicts instead of polars DataFrame.\n"
        )


# Helper function to handle both Polars DataFrame and list of dicts
def print_results(results: Any, *fields: str) -> None:
    """Print results handling both Polars DataFrame and list of dicts."""
    try:
        import polars as pl

        if isinstance(results, pl.DataFrame):
            for row in results.iter_rows(named=True):
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
        import polars as pl

        if isinstance(results, pl.DataFrame):
            for row in results.iter_rows(named=True):
                print(f"  {dict(row)}")
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
# Polars-style DataFrame operations
# ============================================================================

# Create a PolarsDataFrame
df = db.table("users").polars()

# Column access
print("Column names:", df.columns)
# Output: Column names: ['id', 'name', 'age', 'country']
print()

# Select columns using Polars-style indexing
df_selected = df[["id", "name", "age"]]
print("Selected columns:", df_selected.columns)  # type: ignore[union-attr]
# Output: Selected columns: ['id', 'name', 'age']
print()

# Filter using filter() method (Polars-style)
df_filtered = df.filter(col("age") > 25)
results = df_filtered.collect()
print("Filtered results (age > 25):")
print_results(results, "name", "age", "country")
# Output:
#   Alice, 30, USA
#   Charlie, 35, USA
#   David, 28, UK
print()

# Filter with multiple conditions
df_filtered_usa = df.filter((col("age") > 25) & (col("country") == "USA"))
results_usa = df_filtered_usa.collect()
print("Filtered results (age > 25 AND country == 'USA'):")
print_results(results_usa, "name", "age", "country")
# Output:
#   Alice, 30, USA
#   Charlie, 35, USA
print()

# Select specific columns
df_selected = df.select("id", "name", "age")
print("Selected columns using select():")
print(f"  Columns: {df_selected.columns}")
print()

# Add columns using with_columns()
df_with_new_col = df.with_columns((col("age") + 10).alias("age_plus_10"))
results = df_with_new_col.collect()
print("Users with calculated column (age + 10):")
try:
    import polars as pl

    if isinstance(results, pl.DataFrame):
        for row in results.iter_rows(named=True):
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

# GroupBy operations (Polars-style)
print("GroupBy example:")
grouped = df.group_by("country")
result_df = grouped.count()
results = result_df.collect()
print("Count by country:")
print_dict_results(results)
# Output:
#   {'country': 'UK', 'count': 2}
#   {'country': 'USA', 'count': 2}
print()

# GroupBy with aggregation expressions
df_orders = db.table("orders").polars()
grouped_orders = df_orders.group_by("status")
agg_result = grouped_orders.agg(F.sum(col("amount")).alias("total_amount"))
results = agg_result.collect()
print("Sum of amount by status:")
print_dict_results(results)
# Output:
#   {'status': 'active', 'total_amount': 600.0}
#   {'status': 'completed', 'total_amount': 150.0}
print()

# Join operations (Polars-style)
print("Join example:")
df_users = db.table("users").polars()[["id", "name"]]
df_orders = db.table("orders").polars()[["user_id", "amount"]]  # type: ignore[assignment]
df_orders_renamed = df_orders.rename({"user_id": "id"})

joined = df_users.join(df_orders_renamed, on="id", how="inner")  # type: ignore[union-attr]
results = joined.collect()
print("Users joined with orders:")
# Show first 5 rows
try:
    import polars as pl

    if isinstance(results, pl.DataFrame):
        for row in results.head(5).iter_rows(named=True):
            print(f"  {dict(row)}")
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
df_sorted = df.sort("age", descending=True)
results = df_sorted.collect()
print("Users sorted by age (descending):")
print_results(results, "name", "age")
# Output:
#   Charlie, 35
#   Alice, 30
#   David, 28
#   Bob, 25
print()

# Rename columns
df_renamed = df.rename({"name": "full_name"})
print(f"Renamed columns: {df_renamed.columns}")
# Output: Renamed columns: ['id', 'full_name', 'age', 'country']
print()

# Drop columns
df_dropped = df.drop("age")
print(f"Columns after dropping 'age': {df_dropped.columns}")
# Output: Columns after dropping 'age': ['id', 'name', 'country']
print()

# Limit rows
df_limited = df.limit(2)
results = df_limited.collect()
print("First 2 rows:")
print_results(results, "name", "age")
print()

# Head and tail
df_head = df.head(2)
results = df_head.collect()
print("Head (first 2 rows):")
print_results(results, "name", "age")
print()

# Unique/distinct
df_unique = df.unique()
results = df_unique.collect()
print(f"Unique rows: {len(results)}")
print()

# Data inspection
print("Data inspection:")
print(f"  columns: {df.columns}")
print(f"  width: {df.width}")
print(f"  schema: {df.schema}")
print()

# Chaining operations
print("Chained operations example:")
result = df.filter(col("age") > 25).select("id", "name", "age", "country").sort("age").drop("id")
results = result.collect()
print("Filtered, sorted, and columns dropped:")
print_dict_results(results)
print()

# Boolean indexing
df_filtered_bool = df[df["age"] > 28]  # type: ignore[index, operator]
results = df_filtered_bool.collect()  # type: ignore[union-attr]
print("Filtered using boolean indexing (age > 28):")
print_results(results, "name", "age")
# Output:
#   Alice, 30
#   Charlie, 35
print()

# Collect returns Polars DataFrame (if polars is installed)
try:
    import polars as pl

    pdf = df.collect()
    print(f"Collected as Polars DataFrame: {type(pdf)}")
    if isinstance(pdf, pl.DataFrame):
        print(f"Shape: {pdf.shape}")
        print(f"Columns: {list(pdf.columns)}")
        print()
        print("First few rows:")
        print(pdf.head())
        # Output:
        # Collected as Polars DataFrame: <class 'polars.dataframe.frame.DataFrame'>
        # Shape: (4, 4)
        # Columns: ['id', 'name', 'age', 'country']
        #
        # First few rows:
        # shape: (4, 4)
        # ┌─────┬─────────┬─────┬─────────┐
        # │ id  ─ name    ─ age ─ country │
        # │ --- ─ ---     ─ --- ─ ---     │
        # │ i64 ─ str     ─ i64 ─ str     │
        # ╞═════╪═════════╪═════╪═════════╡
        # │ 1   ─ Alice   ─ 30  ─ USA     │
        # │ 2   ─ Bob     ─ 25  ─ UK      │
        # │ 3   ─ Charlie ─ 35  ─ USA     │
        # │ 4   ─ David   ─ 28  ─ UK      │
        # └─────┴─────────┴─────┴─────────┘
except ImportError:
    print("polars not installed - collect() will return list of dicts")
    results = df.collect()
    print(f"Collected as list: {len(results)} rows")

# ============================================================================
# Reading Files (Polars-Style)
# ============================================================================

print("\n" + "=" * 70)
print("Reading Files (Polars-Style)")
print("=" * 70)

import tempfile
import csv
import json
from pathlib import Path

# Create temporary files for demonstration
temp_dir = Path(tempfile.mkdtemp())

# Create CSV file
csv_file = temp_dir / "users.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "name", "age", "country"])
    writer.writerow([1, "Alice", 30, "USA"])
    writer.writerow([2, "Bob", 25, "UK"])

# Scan CSV file
print("\nScanning CSV file:")
df_csv = db.scan_csv(str(csv_file), header=True)
results = df_csv.collect()
print(f"Read {len(results)} rows from CSV")

# Create JSON file
json_file = temp_dir / "users.json"
with open(json_file, "w") as f:
    json.dump(
        [
            {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
            {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
        ],
        f,
    )

# Scan JSON file
print("\nScanning JSON file:")
df_json = db.scan_json(str(json_file))
results = df_json.collect()
print(f"Read {len(results)} rows from JSON")

# Scan with schema
print("\nScanning CSV with explicit schema:")
from moltres.table.schema import column

schema = [
    column("id", "INTEGER"),
    column("name", "TEXT"),
    column("age", "INTEGER"),
    column("country", "TEXT"),
]
df_schema = db.scan_csv(str(csv_file), schema=schema, header=True)
results = df_schema.collect()
print(f"Read {len(results)} rows with schema")

# Alternative: Use existing API with .polars()
print("\nUsing db.read.csv().polars():")
df_alt = db.read.csv(str(csv_file)).polars()
results = df_alt.collect()
print(f"Read {len(results)} rows using alternative method")

# ============================================================================
# Writing Files (Polars-Style)
# ============================================================================

print("\n" + "=" * 70)
print("Writing Files (Polars-Style)")
print("=" * 70)

df = db.table("users").polars()

# Write to CSV
output_csv = temp_dir / "output.csv"
df.write_csv(str(output_csv), header=True)
print(f"\nWritten to CSV: {output_csv}")
print(f"File exists: {output_csv.exists()}")

# Write to JSON
output_json = temp_dir / "output.json"
df.write_json(str(output_json))
print(f"\nWritten to JSON: {output_json}")
print(f"File exists: {output_json.exists()}")

# Write to JSONL
output_jsonl = temp_dir / "output.jsonl"
df.write_jsonl(str(output_jsonl))
print(f"\nWritten to JSONL: {output_jsonl}")
print(f"File exists: {output_jsonl.exists()}")

# Write filtered data
filtered_output = temp_dir / "filtered.json"
df.filter(col("age") > 25).write_json(str(filtered_output))
print(f"\nWritten filtered data (age > 25) to: {filtered_output}")

# Write with options
output_csv_options = temp_dir / "output_options.csv"
df.write_csv(str(output_csv_options), header=True, delimiter=";")
print(f"\nWritten CSV with custom delimiter: {output_csv_options}")

# Read back the written file
print("\nReading back the written CSV:")
df_readback = db.scan_csv(str(output_csv), header=True)
results = df_readback.collect()
print(f"Read back {len(results)} rows")

# Cleanup
import shutil

shutil.rmtree(temp_dir)
print("\nCleaned up temporary files")

# ============================================================================
# Set Operations
# ============================================================================

print("\n" + "=" * 70)
print("Set Operations")
print("=" * 70)

df1 = db.table("users").polars()
df2 = db.table("users").polars()

# Concatenate
concatenated = df1.concat(df2, how="vertical")
results = concatenated.collect()
print(f"\nConcatenated: {len(results)} rows")

# Union
unioned = df1.union(df2, distinct=True)
results = unioned.collect()
print(f"Union distinct: {len(results)} rows")

# Intersect
intersected = df1.intersect(df2)
results = intersected.collect()
print(f"Intersect: {len(results)} rows")

# Cross join
df_users = db.table("users").polars().select("id", "name")
df_orders = db.table("orders").polars().select("id", "amount")
crossed = df_users.cross_join(df_orders)
results = crossed.collect()
print(f"Cross join: {len(results)} rows")

# ============================================================================
# SQL Expression Selection
# ============================================================================

print("\n" + "=" * 70)
print("SQL Expression Selection")
print("=" * 70)

df = db.table("users").polars()

# Select with SQL expressions
selected = df.select_expr("id", "name", "age", "age * 2 as double_age")
results = selected.collect()
print(f"\nSelected with SQL expressions: {len(results)} rows")
if len(results) > 0:
    # Handle both polars DataFrame and list of dicts
    if POLARS_AVAILABLE and hasattr(results, "to_dicts"):
        first_row = results.to_dicts()[0]
        double_age = first_row.get("double_age", "N/A")
    else:
        double_age = results[0].get("double_age", "N/A")
    print(f"Double age: {double_age}")

# ============================================================================
# CTE Support
# ============================================================================

print("\n" + "=" * 70)
print("Common Table Expressions (CTEs)")
print("=" * 70)

# Create a CTE
cte_df = db.table("users").polars().filter(col("age") > 25).cte("adults")
results = cte_df.select().collect()
print(f"\nCTE 'adults': {len(results)} rows")

# ============================================================================
# Additional Utilities
# ============================================================================

print("\n" + "=" * 70)
print("Additional Utilities")
print("=" * 70)

# Add row count
df_with_row = db.table("users").polars().with_row_count("row_id")
results = df_with_row.collect()
print(f"\nWith row count: {len(results)} rows")
if len(results) > 0:
    # Handle both polars DataFrame and list of dicts
    if POLARS_AVAILABLE and hasattr(results, "to_dicts"):
        first_row = results.to_dicts()[0]
        row_id = first_row.get("row_id", "N/A")
    else:
        row_id = results[0].get("row_id", "N/A")
    print(f"First row ID: {row_id}")

# Rename columns
renamed = (
    db.table("users")
    .polars()
    .select("id", "name", "age")
    .with_columns_renamed({"name": "full_name", "age": "years"})
)
results = renamed.collect()
print(f"\nRenamed columns: {len(results)} rows")
if len(results) > 0:
    # Handle both polars DataFrame and list of dicts
    if POLARS_AVAILABLE and hasattr(results, "columns"):
        columns = list(results.columns)
    else:
        columns = list(results[0].keys())
    print(f"Columns: {columns}")

db.close()
