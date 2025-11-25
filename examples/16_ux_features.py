"""Example: UX Features and Convenience Methods.

This example demonstrates the new UX improvements and convenience methods
added to make Moltres easier to use.
"""

from moltres import col, column, connect
from moltres.io.records import Records

# Connect to database
import os

db_path = "example_ux.db"
if os.path.exists(db_path):
    os.remove(db_path)

db = connect(f"sqlite:///{db_path}")

# ============================================================================
# Records API Improvements
# ============================================================================

print("=" * 70)
print("Records API Improvements")
print("=" * 70)

# Create table (drop if exists to avoid conflicts)
db.drop_table("users", if_exists=True).collect()
db.create_table(
    "users",
    [column("id", "INTEGER", primary_key=True), column("name", "TEXT"), column("email", "TEXT")],
).collect()

# Method 1: Using Records.from_list() (recommended)
records = Records.from_list(
    [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
    ],
    database=db,
)
inserted = records.insert_into("users")
print(f"Inserted {inserted} rows using Records.from_list()")
# Output: Inserted 2 rows using Records.from_list()

# Method 2: Using Records.from_dicts() (convenience for individual rows)
records = Records.from_dicts(
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
    {"id": 4, "name": "Diana", "email": "diana@example.com"},
    database=db,
)
inserted = records.insert_into("users")
print(f"Inserted {inserted} rows using Records.from_dicts()")
# Output: Inserted 2 rows using Records.from_dicts()

# ============================================================================
# Database Convenience Methods
# ============================================================================

print("\n" + "=" * 70)
print("Database Convenience Methods")
print("=" * 70)

# Insert using db.insert()
inserted = db.insert(
    "users",
    [
        {"id": 5, "name": "Eve", "email": "eve@example.com"},
        {"id": 6, "name": "Frank", "email": "frank@example.com"},
    ],
)
print(f"Inserted {inserted} rows using db.insert()")
# Output: Inserted 2 rows using db.insert()

# Update using db.update()
updated = db.update("users", where=col("id") == 1, set={"name": "Alice Updated"})
print(f"Updated {updated} row using db.update()")
# Output: Updated 1 row using db.update()

# Delete using db.delete()
deleted = db.delete("users", where=col("id") == 6)
print(f"Deleted {deleted} row using db.delete()")
# Output: Deleted 1 row using db.delete()

# Merge (upsert) using db.merge()
merged = db.merge(
    "users",
    [{"id": 1, "name": "Alice Merged", "email": "alice@example.com"}],
    on=["id"],
    when_matched={"name": "Alice Merged"},
)
print(f"Merged {merged} row using db.merge()")
# Output: Merged 1 row using db.merge()

# ============================================================================
# Schema Discovery and Inspection
# ============================================================================

print("\n" + "=" * 70)
print("Schema Discovery and Inspection")
print("=" * 70)

# Show tables (formatted output for interactive use)
print("\nTables in database:")
db.show_tables()

# Show schema (formatted output for interactive use)
print("\nSchema for 'users' table:")
db.show_schema("users")

# Get execution plan
print("\nQuery execution plan:")
plan = db.explain("SELECT * FROM users WHERE id = :id", params={"id": 1})
print(plan)

# ============================================================================
# DataFrame Convenience Methods
# ============================================================================

print("\n" + "=" * 70)
print("DataFrame Convenience Methods")
print("=" * 70)

df = db.table("users").select().order_by(col("id"))

# Get first few rows
print("\nFirst 3 rows:")
first_rows = df.head(3)
for row in first_rows:
    print(f"  {row}")

# Get last few rows
print("\nLast 2 rows:")
last_rows = df.tail(2)
for row in last_rows:
    print(f"  {row}")

# Show DataFrame (formatted output)
print("\nFormatted DataFrame output:")
df.show(5)

# Print schema
print("\nDataFrame schema:")
df.printSchema()

# Get query execution plan
print("\nQuery execution plan:")
plan = df.explain()
print(plan)

# ============================================================================
# Error Messages with Suggestions
# ============================================================================

print("\n" + "=" * 70)
print("Error Messages with Suggestions")
print("=" * 70)

# Try querying with a typo in column name
# (This will show a helpful error message with suggestions)
try:
    df = db.table("users").select("nme")  # Typo: "nme" instead of "name"
    df.collect()
except Exception as e:
    print(f"Error with helpful message: {e}")
    # Output: Error with helpful message: (sqlite3.OperationalError) no such column: nme
    # Note: Error suggestions are integrated into ExecutionError when context is provided

# Try querying non-existent table
# (This will show a helpful error message with suggestions)
try:
    db.execute_sql("SELECT * FROM usrs")  # Typo: "usrs" instead of "users"
except Exception as e:
    print(f"Error with helpful message: {e}")
    # Output: Error with helpful message: (sqlite3.OperationalError) no such table: usrs
    # Note: Error suggestions are integrated into ExecutionError when context is provided

# ============================================================================
# Connection String Validation
# ============================================================================

print("\n" + "=" * 70)
print("Connection String Validation")
print("=" * 70)

# Invalid connection string format
try:
    from moltres.utils.exceptions import DatabaseConnectionError

    db_invalid = connect("invalid-connection-string")
except DatabaseConnectionError as e:
    print(f"Validation error: {e}")
    # Output: Validation error: Connection string must include '://' separator. Got: invalid-connection-string...

# Missing async driver (for async connections)
try:
    from moltres import async_connect

    db_async = async_connect("sqlite:///example.db")  # Missing +aiosqlite
except DatabaseConnectionError as e:
    print(f"Async validation error: {e}")
    # Output: Async validation error: Async SQLite connection requires 'sqlite+aiosqlite://' prefix. Got: sqlite:///example.db...

print("\n" + "=" * 70)
print("Example completed!")
print("=" * 70)

db.close()
