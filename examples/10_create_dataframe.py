"""Example: Creating DataFrames from Python data.

This example demonstrates creating DataFrames from lists, dicts, and other sources.
"""

from moltres import connect
import os

# Clean up any existing database file
db_path = "example.db"
if os.path.exists(db_path):
    os.remove(db_path)

db = connect(f"sqlite:///{db_path}")

# Create DataFrame from list of dicts
data = [
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
    {"id": 3, "name": "Charlie", "age": 35},
]

df = db.createDataFrame(data, pk="id")
results = df.collect()
print(f"DataFrame from dicts: {results}")
# Output: DataFrame from dicts: [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 35}]

# Create DataFrame with auto-incrementing primary key
data_no_id = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
]

df = db.createDataFrame(data_no_id, auto_pk="id")
results = df.collect()
print(f"DataFrame with auto_pk: {results}")
# Output: DataFrame with auto_pk: [{'name': 'Alice', 'age': 30, 'id': 1}, {'name': 'Bob', 'age': 25, 'id': 2}]

# Create DataFrame with explicit schema
from moltres.table.schema import ColumnDef

schema = [
    ColumnDef(name="id", type_name="INTEGER", primary_key=True),
    ColumnDef(name="name", type_name="TEXT"),
    ColumnDef(name="age", type_name="INTEGER"),
]

df = db.createDataFrame(data, schema=schema)
results = df.collect()
print(f"DataFrame with schema: {results}")
# Output: DataFrame with schema: [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 35}]

# Create DataFrame from tuples (with schema)
tuple_data = [(1, "Alice", 30), (2, "Bob", 25), (3, "Charlie", 35)]
df = db.createDataFrame(tuple_data, schema=schema)
results = df.collect()
print(f"DataFrame from tuples: {results}")
# Output: DataFrame from tuples: [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 35}]

# Create DataFrame and save to table
df = db.createDataFrame(data, pk="id")
df.write.save_as_table("users")
print("DataFrame saved to table")

# Query the table
table_df = db.table("users").select()
results = table_df.collect()
print(f"Data from table: {results}")
# Output: Data from table: [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 35}]

db.close()
