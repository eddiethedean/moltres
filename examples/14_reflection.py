"""Example: Schema inspection and reflection.

This example demonstrates how to inspect and reflect database schemas
using Moltres's schema inspection and reflection features.
"""

from moltres import connect

db = connect("sqlite:///example.db")

# Clean up any existing tables
db.drop_table("users", if_exists=True).collect()
db.drop_table("orders", if_exists=True).collect()

# Create tables with various column types
from moltres.table.schema import column

db.create_table(
    "users",
    [
        column("id", "INTEGER", nullable=False, primary_key=True),
        column("name", "TEXT", nullable=False),
        column("email", "TEXT", nullable=True),
        column("age", "INTEGER", nullable=True),
    ],
).collect()

db.create_table(
    "orders",
    [
        column("id", "INTEGER", nullable=False, primary_key=True),
        column("user_id", "INTEGER", nullable=False),
        column("amount", "REAL", nullable=False, default=0.0),
        column("status", "TEXT", nullable=False, default="pending"),
    ],
).collect()

# Insert some data
from moltres.io.records import Records

users_data = [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
]

records = Records(_data=users_data, _database=db)
records.insert_into("users")

# Schema Inspection - Get table names
table_names = db.get_table_names()
print(f"Tables in database: {table_names}")
# Output: Tables in database: ['users', 'orders']

# Schema Inspection - Get view names
view_names = db.get_view_names()
print(f"Views in database: {view_names}")
# Output: Views in database: []

# Schema Inspection - Get column information
columns = db.get_columns("users")
print("Columns in 'users' table:")
for col_info in columns:
    print(
        f"  - {col_info.name}: {col_info.type_name} "
        f"(nullable={col_info.nullable}, primary_key={col_info.primary_key})"
    )
# Output:
# Columns in 'users' table:
#   - id: INTEGER (nullable=False, primary_key=True)
#   - name: TEXT (nullable=False, primary_key=False)
#   - email: TEXT (nullable=True, primary_key=False)
#   - age: INTEGER (nullable=True, primary_key=False)

# Table Reflection - Reflect a single table
users_schema = db.reflect_table("users")
print("\nReflected schema for 'users':")
print(f"  Table name: {users_schema.name}")
print(f"  Number of columns: {len(users_schema.columns)}")
for col_def in users_schema.columns:
    print(f"    - {col_def.name}: {col_def.type_name}")
# Output:
# Reflected schema for 'users':
#   Table name: users
#   Number of columns: 4
#     - id: INTEGER
#     - name: TEXT
#     - email: TEXT
#     - age: INTEGER

# Database Reflection - Reflect entire database
all_schemas = db.reflect()
print("\nReflected all tables in database:")
for table_name, schema in all_schemas.items():
    print(f"  {table_name}: {len(schema.columns)} columns")
# Output:
# Reflected all tables in database:
#   users: 4 columns
#   orders: 4 columns

# Using reflected schema to create a new table
# (demonstrating how reflection can be used for schema migration)
reflected_users_schema = db.reflect_table("users")
print("\nUsing reflected schema:")
print(
    f"  Primary key columns: {[col.name for col in reflected_users_schema.columns if col.primary_key]}"
)
print(f"  Nullable columns: {[col.name for col in reflected_users_schema.columns if col.nullable]}")

# Convert ColumnInfo to ColumnDef (useful for schema operations)
from moltres.utils.inspector import get_table_columns

columns_info = get_table_columns(db, "users")
column_defs = [col_info.to_column_def() for col_info in columns_info]
print(f"\nConverted {len(column_defs)} ColumnInfo objects to ColumnDef objects")

db.close()

# Async example
try:
    from moltres import async_connect

    async def async_reflection_example() -> None:
        """Demonstrate async reflection."""
        db = async_connect("sqlite+aiosqlite:///example_async.db")

        # Create a table
        await db.create_table(
            "products",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("price", "REAL"),
            ],
        ).collect()

        # Get table names
        tables = await db.get_table_names()
        print(f"\nAsync - Tables: {tables}")

        # Reflect table
        schema = await db.reflect_table("products")
        print(f"Async - Reflected 'products' table with {len(schema.columns)} columns")

        # Reflect entire database
        all_schemas = await db.reflect()
        print(f"Async - Reflected {len(all_schemas)} tables")

        await db.close()

    # Uncomment to run async example:
    # asyncio.run(async_reflection_example())
except ImportError:
    print("\nAsync dependencies not installed. Install with: pip install moltres[async]")
