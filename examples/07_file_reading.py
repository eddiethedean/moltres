"""Example: Reading files (CSV, JSON, Parquet, Text).

This example demonstrates reading data from various file formats.
"""

from moltres import connect
from pathlib import Path
import json

db = connect("sqlite:///example.db")

# Create sample files
data_dir = Path("example_data")
data_dir.mkdir(exist_ok=True)

# CSV file
csv_file = data_dir / "data.csv"
csv_file.write_text("id,name,age\n1,Alice,30\n2,Bob,25\n3,Charlie,35\n")

# JSON file
json_file = data_dir / "data.json"
json_data = [
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
    {"id": 3, "name": "Charlie", "age": 35},
]
json_file.write_text(json.dumps(json_data))

# JSONL file
jsonl_file = data_dir / "data.jsonl"
with open(jsonl_file, "w") as f:
    for item in json_data:
        f.write(json.dumps(item) + "\n")

# Text file
text_file = data_dir / "data.txt"
text_file.write_text("Line 1\nLine 2\nLine 3\n")

# Read CSV
df = db.load.csv(str(csv_file))
results = df.collect()
print(f"CSV data: {results}")

# Read CSV with options
df = db.load.option("header", True).option("inferSchema", True).csv(str(csv_file))
results = df.collect()
print(f"CSV with options: {results}")

# Read JSON
df = db.load.json(str(json_file))
results = df.collect()
print(f"JSON data: {results}")

# Read JSONL
df = db.load.jsonl(str(jsonl_file))
results = df.collect()
print(f"JSONL data: {results}")

# Read Text
df = db.load.text(str(text_file), column_name="line")
results = df.collect()
print(f"Text data: {results}")

# Read with explicit schema
from moltres.table.schema import ColumnDef

schema = [
    ColumnDef(name="id", type_name="INTEGER"),
    ColumnDef(name="name", type_name="TEXT"),
    ColumnDef(name="age", type_name="INTEGER"),
]
df = db.load.schema(schema).csv(str(csv_file))
results = df.collect()
print(f"CSV with schema: {results}")

# Streaming read (for large files)
for chunk in df.collect(stream=True):
    print(f"Chunk: {chunk}")

db.close()
