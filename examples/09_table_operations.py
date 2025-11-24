"""Example: Table operations (create, drop, mutations).

This example demonstrates table creation, deletion, and data mutations.
"""

from moltres import connect, col

db = connect("sqlite:///example.db")

# Create table
from moltres.table.schema import column

db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("email", "TEXT"),
        column("active", "INTEGER"),  # SQLite uses INTEGER for boolean
    ],
).collect()

# Insert rows
from moltres.io.records import Records

users_data = [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "active": 1},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "active": 1},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": 0},
]

Records(_data=users_data, _database=db).insert_into("users")

# Update rows using mutations API
from moltres.table.mutations import update_rows, delete_rows, merge_rows

table = db.table("users")
update_rows(table, where=col("id") == 1, values={"name": "Alice Updated"})

# Verify update
df = db.table("users").select()
results = df.collect()
print(f"After update: {results}")

# Delete rows
delete_rows(table, where=col("active") == 0)

# Verify delete
results = df.collect()
print(f"After delete: {results}")

# Merge (upsert) rows
merge_data = [
    {"id": 1, "name": "Alice Merged", "email": "alice.new@example.com", "active": 1},
    {"id": 4, "name": "David", "email": "david@example.com", "active": 1},
]

merge_rows(
    table,
    merge_data,
    on=["id"],
    when_matched={"name": "name", "email": "email"},
    when_not_matched={"name": "name", "email": "email", "active": "active"},
)

# Verify merge
results = df.collect()
print(f"After merge: {results}")

# Drop table
db.drop_table("users", if_exists=True).collect()
print("Table dropped")

db.close()
