"""Example: UX Features and Convenience Methods.

This example demonstrates the new UX improvements and convenience methods
added to make Moltres easier to use, debug, and understand.
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
# Enhanced SQL Display
# ============================================================================

print("=" * 70)
print("Enhanced SQL Display")
print("=" * 70)

# Create tables
db.drop_table("users", if_exists=True).collect()
db.drop_table("orders", if_exists=True).collect()
db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("email", "TEXT"),
        column("age", "INTEGER"),
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

# Insert sample data
Records.from_list(
    [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
    ],
    database=db,
).insert_into("users")

df = db.table("users").select().where(col("age") > 25)

# Method 1: show_sql() - Pretty-print SQL
print("\n1. Using show_sql() to pretty-print SQL:")
df.show_sql()
# Output:
#   SELECT
#      *
# FROM
#    users
# WHERE
#    age > 25

# Method 2: sql property - Get formatted SQL as string
print("\n2. Using sql property:")
sql = df.sql
print(sql[:100] + "...")  # First 100 chars
# Output:   SELECT
#      *
# FROM
#    users
# WHERE
#    age > 25...

# Method 3: to_sql(pretty=True) - Get formatted SQL
print("\n3. Using to_sql(pretty=True):")
formatted_sql = df.to_sql(pretty=True)
print(formatted_sql[:100] + "...")
# Output:   SELECT
#      *
# FROM
#    users
# WHERE
#    age > 25...

# Method 4: sql_preview() - Get SQL preview
print("\n4. Using sql_preview():")
preview = df.sql_preview(max_length=50)
print(preview)
# Output: SELECT *
# FROM users
# WHERE age > 25

# ============================================================================
# Query Plan Visualization
# ============================================================================

print("\n" + "=" * 70)
print("Query Plan Visualization")
print("=" * 70)

# Create a more complex query
df_complex = (
    db.table("users")
    .select(col("id"), col("name"))
    .where(col("age") > 25)
    .order_by(col("name").asc())
    .limit(10)
)

# Method 1: plan_summary() - Get structured plan statistics
print("\n1. Plan summary:")
summary = df_complex.plan_summary()
print(f"  Operations: {summary['operations']}")
print(f"  Table scans: {summary['table_scans']}")
print(f"  Filters: {summary['filters']}")
print(f"  Depth: {summary['depth']}")
print(f"  Total operations: {summary['total_operations']}")
# Output:
#   Operations: ['Limit', 'Sort', 'Filter', 'Project', 'TableScan']
#   Table scans: 1
#   Filters: 1
#   Depth: 4
#   Total operations: 5

# Method 2: visualize_plan() - ASCII tree visualization
print("\n2. Plan visualization:")
print(df_complex.visualize_plan())
# Output:
# └── Limit
#     └── Sort
#         └── Filter [has predicate]
#             └── Project
#                 └── TableScan(users)

# Method 3: explain() - Get database execution plan
print("\n3. Database execution plan:")
plan = df_complex.explain()
print(plan[:200] + "..." if len(plan) > 200 else plan)
# Output (SQLite):
# {'id': 4, 'parent': 0, 'notused': 0, 'detail': 'SCAN users'}
# {'id': 18, 'parent': 0, 'notused': 0, 'detail': 'USE TEMP B-TREE FOR ORDER BY'}

# ============================================================================
# Schema Discovery
# ============================================================================

print("\n" + "=" * 70)
print("Schema Discovery")
print("=" * 70)

# Method 1: db.schema() - Get table schema
print("\n1. Get schema for 'users' table:")
schema = db.schema("users")
for col_def in schema:
    print(f"  {col_def.name}: {col_def.type_name} (nullable={col_def.nullable})")
# Output:
#   id: INTEGER (nullable=True)
#   name: TEXT (nullable=True)
#   email: TEXT (nullable=True)
#   age: INTEGER (nullable=True)

# Method 2: db.tables() - Get all tables with schemas
print("\n2. Get all tables with schemas:")
tables = db.tables()
for table_name, columns in tables.items():
    print(f"  {table_name}: {len(columns)} columns")
# Output:
#   orders: 4 columns
#   users: 4 columns

# Method 3: handle.columns() - Get column names from table handle
print("\n3. Get columns from table handle:")
handle = db.table("users")
columns = handle.columns()
print(f"  Columns: {columns}")
# Output:   Columns: ['id', 'name', 'email', 'age']

# ============================================================================
# Query Validation and Performance Hints
# ============================================================================

print("\n" + "=" * 70)
print("Query Validation and Performance Hints")
print("=" * 70)

# Method 1: validate() - Check for common issues
print("\n1. Validate query:")
issues = df_complex.validate()
for issue in issues:
    print(f"  [{issue['type']}] {issue['message']}")
    if issue.get("suggestion"):
        print(f"    Suggestion: {issue['suggestion']}")
# Output:
#   [info] Query has 1 filter operation(s). Consider adding indexes on filtered columns for better performance.
#     Suggestion: Use db.create_index() to add indexes on frequently filtered columns.

# Method 2: performance_hints() - Get optimization suggestions
print("\n2. Performance hints:")
hints = df_complex.performance_hints()
for hint in hints:
    print(f"  - {hint}")
# Output:
#   - Consider adding indexes on columns used in WHERE clauses for better performance.
#   - Consider using limit() if you only need a subset of results.

# ============================================================================
# Interactive Help
# ============================================================================

print("\n" + "=" * 70)
print("Interactive Help")
print("=" * 70)

# Method 1: help() - Display available operations
print("\n1. Display help (showing first few lines):")
# df_complex.help()  # Uncomment to see full help output
print("  (Use df.help() to see full interactive help)")
# Output: Displays comprehensive help with all available operations

# Method 2: suggest_next() - Get suggestions for next operations
print("\n2. Get suggestions for next operations:")
suggestions = df_complex.suggest_next()
for suggestion in suggestions:
    print(f"  - {suggestion}")
# Output:
#   - Consider selecting specific columns with select() for better performance
#   - Consider adding limit() if you only need a subset of results

# ============================================================================
# Enhanced Error Messages
# ============================================================================

print("\n" + "=" * 70)
print("Enhanced Error Messages")
print("=" * 70)

# Try querying with a typo in column name
print("\n1. Error with column typo:")
try:
    df_error = db.table("users").select("nme")  # Typo: "nme" instead of "name"
    df_error.collect()
except Exception as e:
    error_msg = str(e)
    # Show first part of error (it's long)
    print(f"  Error: {error_msg[:150]}...")
    # Error messages now include helpful suggestions and query context
# Output:
#   Error: SQL execution failed: (sqlite3.OperationalError) no such column: nme
#   [SQL: SELECT nme
#   FROM (SELECT *
#   FROM users) AS anon_1]...

# Try querying non-existent table
print("\n2. Error with table typo:")
try:
    db.execute_sql("SELECT * FROM usrs")  # Typo: "usrs" instead of "users"
except Exception as e:
    error_msg = str(e)
    print(f"  Error: {error_msg[:150]}...")
    # Error messages now include query context and suggestions
# Output:
#   Error: SQL execution failed: (sqlite3.OperationalError) no such table: usrs
#   [SQL: SELECT * FROM usrs]...

# ============================================================================
# Records API Improvements
# ============================================================================

print("\n" + "=" * 70)
print("Records API Improvements")
print("=" * 70)

# Method 1: Using Records.from_list() (recommended)
# Note: Using IDs 10+ to avoid conflicts with earlier inserts
records = Records.from_list(
    [
        {"id": 10, "name": "Alice", "email": "alice@example.com"},
        {"id": 11, "name": "Bob", "email": "bob@example.com"},
    ],
    database=db,
)
inserted = records.insert_into("users")
print(f"Inserted {inserted} rows using Records.from_list()")
# Output: Inserted 2 rows using Records.from_list()

# Method 2: Using Records.from_dicts() (convenience for individual rows)
records = Records.from_dicts(
    {"id": 12, "name": "Charlie", "email": "charlie@example.com"},
    {"id": 13, "name": "Diana", "email": "diana@example.com"},
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
        {"id": 14, "name": "Eve", "email": "eve@example.com", "age": 28},
        {"id": 15, "name": "Frank", "email": "frank@example.com", "age": 32},
    ],
)
print(f"Inserted {inserted} rows using db.insert()")
# Output: Inserted 2 rows using db.insert()

# Update using db.update()
updated = db.update("users", where=col("id") == 1, set={"name": "Alice Updated"})
print(f"Updated {updated} row using db.update()")
# Output: Updated 1 row using db.update()

# Delete using db.delete()
deleted = db.delete("users", where=col("id") == 15)
print(f"Deleted {deleted} row using db.delete()")
# Output: Deleted 1 row using db.delete()
# Note: If id 15 doesn't exist, this will output: Deleted 0 row using db.delete()

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
# Output:
#   {'id': 1, 'name': 'Alice Updated', 'email': 'alice@example.com', 'age': 30}
#   {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'age': 25}
#   {'id': 3, 'name': 'Charlie', 'email': 'charlie@example.com', 'age': 35}

# Get last few rows
print("\nLast 2 rows:")
last_rows = df.tail(2)
for row in last_rows:
    print(f"  {row}")
# Output:
#   {'id': 14, 'name': 'Eve', 'email': 'eve@example.com', 'age': 28}
#   {'id': 13, 'name': 'Diana', 'email': 'diana@example.com', 'age': None}

# Show DataFrame (formatted output)
print("\nFormatted DataFrame output:")
df.show(5)
# Output:
# id | name          | email               | age
# -----------------------------------------------
# 1  | Alice Updated | alice@example.com   | 30
# 2  | Bob           | bob@example.com     | 25
# 3  | Charlie       | charlie@example.com | 35
# 10 | Alice         | alice@example.com   | None
# 11 | Bob           | bob@example.com     | None
# showing top 5 of 9 rows

# Print schema
print("\nDataFrame schema:")
df.printSchema()
# Output:
# root
#  |-- id: INTEGER (nullable = true)
#  |-- name: TEXT (nullable = true)
#  |-- email: TEXT (nullable = true)
#  |-- age: INTEGER (nullable = true)

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
    print(f"Validation error: {str(e)[:150]}...")
    # Output includes helpful suggestions
# Output:
#   Validation error: Connection string must include '://' separator. Got: invalid-connection-string...
#   Suggestion: Connection strings should follow the format: 'dialect://user:pass@host:port/dbname'

print("\n" + "=" * 70)
print("Example completed!")
print("=" * 70)

db.close()
if os.path.exists(db_path):
    os.remove(db_path)
