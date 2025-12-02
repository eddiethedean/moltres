"""Example: Writing DataFrames to files.

This example demonstrates writing DataFrames to various file formats.
"""

from pathlib import Path

from moltres import connect, col

db = connect("sqlite:///:memory:")

# Create table and insert data
from moltres.table.schema import column

db.create_table(
    "products",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("price", "REAL"),
        column("category", "TEXT"),
    ],
).collect()

from moltres.io.records import Records

products_data = [
    {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics"},
    {"id": 2, "name": "Mouse", "price": 29.99, "category": "Electronics"},
    {"id": 3, "name": "Desk", "price": 199.99, "category": "Furniture"},
]

Records(_data=products_data, _database=db).insert_into("products")

df = db.table("products").select()

# Write to CSV
output_dir = Path(__file__).parent.parent / "example_output"
output_dir.mkdir(exist_ok=True)

df.write.csv(str(output_dir / "products.csv"))
print("Written to CSV")
# Output: Written to CSV

# Write to JSON
df.write.json(str(output_dir / "products.json"))
print("Written to JSON")
# Output: Written to JSON

# Write to JSONL
df.write.jsonl(str(output_dir / "products.jsonl"))
print("Written to JSONL")
# Output: Written to JSONL

# Write to Parquet (requires pyarrow)
try:
    df.write.parquet(str(output_dir / "products.parquet"))
    print("Written to Parquet")
    # Output: Written to Parquet
except ImportError:
    print("PyArrow not installed, skipping Parquet example")
    # Output: PyArrow not installed, skipping Parquet example

# Write with options
df.write.option("header", True).option("delimiter", ",").csv(
    str(output_dir / "products_custom.csv")
)
print("Written to CSV with options")
# Output: Written to CSV with options

# Write filtered data
expensive = df.where(col("price") > 100)
expensive.write.json(str(output_dir / "expensive_products.json"))
print("Written filtered data to JSON")
# Output: Written filtered data to JSON

# Write with mode options
df.write.mode("overwrite").csv(str(output_dir / "products.csv"))
print("Overwritten CSV file")
# Output: Overwritten CSV file

db.close()
