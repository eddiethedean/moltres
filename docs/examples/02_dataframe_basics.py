"""Example: Basic DataFrame operations.

This example demonstrates basic DataFrame operations like select, filter,
and collect.
"""

from moltres import connect, col

db = connect("sqlite:///:memory:")

# Clean up any existing tables
db.drop_table("users", if_exists=True).collect()

# Create a table and insert data
from moltres.table.schema import column

db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("age", "INTEGER"),
        column("city", "TEXT"),
    ],
).collect()

# Insert data
users_data = [
    {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
    {"id": 2, "name": "Bob", "age": 25, "city": "San Francisco"},
    {"id": 3, "name": "Charlie", "age": 35, "city": "New York"},
]

from moltres.io.records import Records

records = Records(_data=users_data, _database=db)
records.insert_into("users")

# Basic DataFrame operations
df = db.table("users").select()

# Filter
adults = df.where(col("age") >= 30)
results = adults.collect()
print(f"Adults: {results}")
# Output: Adults: [{'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}, {'id': 3, 'name': 'Charlie', 'age': 35, 'city': 'New York'}]

# Select specific columns
names = df.select(col("name"), col("city"))
results = names.collect()
print(f"Names and cities: {results}")
# Output: Names and cities: [{'name': 'Alice', 'city': 'New York'}, {'name': 'Bob', 'city': 'San Francisco'}, {'name': 'Charlie', 'city': 'New York'}]

# Order by
ordered = df.order_by(col("age").desc())
results = ordered.collect()
print(f"Ordered by age: {results}")
# Output: Ordered by age: [{'id': 3, 'name': 'Charlie', 'age': 35, 'city': 'New York'}, {'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}, {'id': 2, 'name': 'Bob', 'age': 25, 'city': 'San Francisco'}]

# Limit
limited = df.limit(2)
results = limited.collect()
print(f"First 2: {results}")
# Output: First 2: [{'id': 1, 'name': 'Alice', 'age': 30, 'city': 'New York'}, {'id': 2, 'name': 'Bob', 'age': 25, 'city': 'San Francisco'}]

db.close()
