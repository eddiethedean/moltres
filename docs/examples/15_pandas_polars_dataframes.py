"""Example: Using pandas and polars DataFrames with moltres.

This example demonstrates:
- Passing pandas DataFrames directly to moltres operations
- Passing polars DataFrames and LazyFrames to moltres operations
- Lazy conversion and schema preservation
- Using DataFrames with insert_into, createDataFrame, and mutations

Required dependencies:
- moltres (required)
- pandas (optional): pip install pandas or pip install moltres[pandas]
- polars (optional): pip install polars or pip install moltres[polars]
"""

import os
import sys

from moltres import connect, col

# Check for optional dependencies
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

# Exit gracefully if neither pandas nor polars is available
if pd is None and pl is None:
    print("This example requires either pandas or polars to be installed.")
    print("Install with: pip install pandas polars")
    print("Or: pip install moltres[pandas,polars]")
    # Output (if dependencies not installed):
    #   This example requires either pandas or polars to be installed.
    #   Install with: pip install pandas polars
    #   Or: pip install moltres[pandas,polars]
    if __name__ == "__main__":
        sys.exit(1)

# Use in-memory database
db = connect("sqlite:///:memory:")

# Create a table to work with
from moltres.table.schema import column

db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT", nullable=False),
        column("age", "INTEGER"),
        column("email", "TEXT"),
        column("active", "INTEGER"),  # SQLite uses INTEGER for boolean
    ],
).collect()

print("=" * 70)
print("Pandas DataFrame Integration")
print("=" * 70)

if pd is not None:
    # Create a pandas DataFrame
    pandas_df = pd.DataFrame(
        [
            {"id": 1, "name": "Alice", "age": 30, "email": "alice@example.com", "active": 1},
            {"id": 2, "name": "Bob", "age": 25, "email": "bob@example.com", "active": 0},
            {"id": 3, "name": "Charlie", "age": 35, "email": "charlie@example.com", "active": 1},
        ]
    )
    print(f"\n✓ Created pandas DataFrame:\n{pandas_df}")

    # Pass pandas DataFrame directly to Records.from_dataframe()
    from moltres.io.records import Records

    records = Records.from_dataframe(pandas_df, database=db)
    print("\n✓ Created Records from pandas DataFrame (lazy conversion)")

    # Insert into database - schema is automatically inferred!
    result = records.insert_into("users")
    print(f"✓ Inserted {result} rows from pandas DataFrame")

    # Verify the data was inserted
    df = db.table("users").select().where(col("id") <= 3)
    results = df.collect()
    print(f"✓ Verified {len(results)} rows in database")

    # Create DataFrame from pandas DataFrame
    pandas_df2 = pd.DataFrame(
        [{"id": 4, "name": "Diana", "age": 28, "email": "diana@example.com", "active": 1}]
    )
    df = db.createDataFrame(pandas_df2, pk="id")
    df.write.insertInto("users")
    print("✓ Created moltres DataFrame from pandas DataFrame and inserted")

    # Pandas DataFrame with nulls - schema preserves nullability
    pandas_df_with_nulls = pd.DataFrame(
        [
            {"id": 5, "name": "Eve", "age": None, "email": "eve@example.com", "active": 1},
        ]
    )
    records = Records.from_dataframe(pandas_df_with_nulls, database=db)
    records.insert_into("users")
    print("✓ Inserted pandas DataFrame with nulls (schema preserves nullability)")

else:
    print("\n⚠️  Skipping pandas examples (pandas not installed)")

print("\n" + "=" * 70)
print("Polars DataFrame Integration")
print("=" * 70)

if pl is not None:
    # Create a polars DataFrame
    polars_df = pl.DataFrame(
        [
            {"id": 6, "name": "Frank", "age": 32, "email": "frank@example.com", "active": 1},
            {"id": 7, "name": "Grace", "age": 29, "email": "grace@example.com", "active": 0},
        ]
    )
    print(f"\n✓ Created polars DataFrame:\n{polars_df}")

    # Pass polars DataFrame directly to Records.from_dataframe()
    records = Records.from_dataframe(polars_df, database=db)
    print("\n✓ Created Records from polars DataFrame (lazy conversion)")

    # Insert into database
    result = records.insert_into("users")
    print(f"✓ Inserted {result} rows from polars DataFrame")

    # Create DataFrame from polars DataFrame
    polars_df2 = pl.DataFrame(
        [{"id": 8, "name": "Henry", "age": 31, "email": "henry@example.com", "active": 1}]
    )
    df = db.createDataFrame(polars_df2, pk="id")
    df.write.insertInto("users")
    print("✓ Created moltres DataFrame from polars DataFrame and inserted")

    # Polars DataFrame with nulls
    polars_df_with_nulls = pl.DataFrame(
        [
            {"id": 9, "name": "Ivy", "age": None, "email": "ivy@example.com", "active": 1},
        ]
    )
    records = Records.from_dataframe(polars_df_with_nulls, database=db)
    records.insert_into("users")
    print("✓ Inserted polars DataFrame with nulls (schema preserves nullability)")

else:
    print("\n⚠️  Skipping polars examples (polars not installed)")

print("\n" + "=" * 70)
print("Polars LazyFrame Integration")
print("=" * 70)

if pl is not None:
    # Create a temporary CSV file for demonstration
    csv_data = """id,name,age,email,active
10,Jack,27,jack@example.com,1
11,Kate,33,kate@example.com,1
12,Liam,26,liam@example.com,0"""

    csv_file = "temp_users.csv"
    with open(csv_file, "w") as f:
        f.write(csv_data)

    # Create polars LazyFrame (lazy evaluation)
    lazy_df = pl.scan_csv(csv_file)
    print(f"\n✓ Created polars LazyFrame from CSV: {csv_file}")

    # Pass LazyFrame to moltres - conversion happens lazily
    records = Records.from_dataframe(lazy_df, database=db)
    print("✓ Created Records from polars LazyFrame (lazy conversion)")

    # Insert into database - LazyFrame is materialized here
    result = records.insert_into("users")
    print(f"✓ Inserted {result} rows from polars LazyFrame")

    # Clean up temp file
    os.remove(csv_file)
    print(f"✓ Cleaned up temporary file: {csv_file}")

else:
    print("\n⚠️  Skipping LazyFrame examples (polars not installed)")

print("\n" + "=" * 70)
print("Schema Preservation")
print("=" * 70)

if pd is not None:
    # Demonstrate schema preservation
    pandas_df = pd.DataFrame(
        [
            {"id": 13, "name": "Mia", "age": 30, "email": "mia@example.com", "active": 1},
        ]
    )
    records = Records.from_dataframe(pandas_df, database=db)

    # Access schema information
    if records._schema:
        print("\n✓ Schema automatically inferred from pandas DataFrame:")
        for col_def in records._schema:
            print(f"  - {col_def.name}: {col_def.type_name} (nullable={col_def.nullable})")

    records.insert_into("users")
    print("✓ Inserted with preserved schema")

print("\n" + "=" * 70)
print("Lazy Conversion Demonstration")
print("=" * 70)

if pd is not None:
    # Create a large pandas DataFrame (simulated)
    large_pandas_df = pd.DataFrame(
        [
            {"id": 14, "name": "Noah", "age": 28, "email": "noah@example.com", "active": 1},
            {"id": 15, "name": "Olivia", "age": 31, "email": "olivia@example.com", "active": 1},
        ]
    )

    records = Records.from_dataframe(large_pandas_df, database=db)
    print("\n✓ Created Records from pandas DataFrame")
    print("  Note: DataFrame is NOT converted to rows until needed")

    # Conversion happens here (lazy)
    rows = records.rows()
    print(f"✓ Converted to rows on-demand: {len(rows)} rows")

    # DataFrame reference is cleared after conversion
    print("✓ DataFrame reference cleared after conversion (memory efficient)")

    records.insert_into("users")
    print("✓ Inserted rows")

print("\n" + "=" * 70)
print("Query Results")
print("=" * 70)

# Query all users
df = db.table("users").select().order_by(col("id"))
results = df.collect()
print(f"\n✓ Total users in database: {len(results)}")
print("\nFirst 5 users:")
for user in results[:5]:
    print(f"  {user}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Key Takeaways:
1. ✅ Pass pandas/polars DataFrames directly to moltres operations
2. ✅ Schema is automatically inferred (types, nullability)
3. ✅ Lazy conversion - DataFrames converted only when needed
4. ✅ Works with polars LazyFrames (lazy evaluation)
5. ✅ No manual conversion required - seamless integration

Use Cases:
- Load data with pandas/polars, then insert into database
- Process data in-memory, then write to SQL
- Chain pandas/polars operations with SQL operations
- Use polars LazyFrames for efficient data processing
""")
