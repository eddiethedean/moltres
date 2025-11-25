"""Example: Table operations (create, drop, mutations, constraints, indexes).

This example demonstrates:
- Table creation with constraints (UNIQUE, CHECK, FOREIGN KEY)
- Index creation and management
- Data mutations (insert, update, delete, merge)
"""

from moltres import connect, col
import os

# Clean up any existing database file
db_path = "example.db"
if os.path.exists(db_path):
    os.remove(db_path)

db = connect(f"sqlite:///{db_path}")

# ============================================================================
# Schema Management: Constraints and Indexes
# ============================================================================

from moltres.table.schema import column, unique, check, foreign_key

print("=" * 70)
print("Creating tables with constraints")
print("=" * 70)

# Create table with UNIQUE and CHECK constraints
db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT", nullable=False),
        column("email", "TEXT", nullable=False),
        column("active", "INTEGER"),  # SQLite uses INTEGER for boolean
    ],
    constraints=[
        unique("email", name="uq_user_email"),  # UNIQUE constraint on email
        check("active IN (0, 1)", name="ck_active_boolean"),  # CHECK constraint
    ],
).collect()
print("✓ Created 'users' table with UNIQUE constraint on email and CHECK constraint on active")

# Create table with foreign key constraint and multi-column unique constraint
db.create_table(
    "orders",
    [
        column("id", "INTEGER", primary_key=True),
        column("user_id", "INTEGER", nullable=False),
        column("order_number", "TEXT", nullable=False),
        column("total", "REAL", nullable=False),
        column("status", "TEXT"),
    ],
    constraints=[
        foreign_key("user_id", "users", "id", name="fk_order_user", on_delete="CASCADE"),
        check("total >= 0", name="ck_positive_total"),
        unique(["user_id", "order_number"], name="uq_user_order"),  # Multi-column UNIQUE
    ],
).collect()
print("✓ Created 'orders' table with FOREIGN KEY, CHECK, and multi-column UNIQUE constraints")

# Create indexes for better query performance
print("\n" + "=" * 70)
print("Creating indexes")
print("=" * 70)

db.create_index("idx_user_email", "users", "email").collect()
print("✓ Created index 'idx_user_email' on users(email)")

db.create_index("idx_order_user_status", "orders", ["user_id", "status"]).collect()
print("✓ Created composite index 'idx_order_user_status' on orders(user_id, status)")

db.create_index("idx_order_status", "orders", "status", unique=False).collect()
print("✓ Created index 'idx_order_status' on orders(status)")

# Create unique index
db.create_index("idx_unique_user_email", "users", "email", unique=True).collect()
print("✓ Created unique index 'idx_unique_user_email' on users(email)")

# ============================================================================
# Data Mutations
# ============================================================================

print("\n" + "=" * 70)
print("Inserting data")
print("=" * 70)

from moltres.io.records import Records

users_data = [
    {"id": 1, "name": "Alice", "email": "alice@example.com", "active": 1},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "active": 1},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": 0},
]

Records(_data=users_data, _database=db).insert_into("users")
print("✓ Inserted 3 users")

orders_data = [
    {"id": 1, "user_id": 1, "order_number": "ORD-001", "total": 99.99, "status": "pending"},
    {"id": 2, "user_id": 1, "order_number": "ORD-002", "total": 149.50, "status": "completed"},
    {"id": 3, "user_id": 2, "order_number": "ORD-003", "total": 79.99, "status": "pending"},
]

Records(_data=orders_data, _database=db).insert_into("orders")
print("✓ Inserted 3 orders")

# Query data
print("\n" + "=" * 70)
print("Querying data")
print("=" * 70)

users_df = db.table("users").select()
users = users_df.collect()
print(f"Users: {users}")
# Output: Users: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'active': 1}, {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'active': 1}, {'id': 3, 'name': 'Charlie', 'email': 'charlie@example.com', 'active': 0}]

orders_df = db.table("orders").select()
orders = orders_df.collect()
print(f"Orders: {orders}")
# Output: Orders: [{'id': 1, 'user_id': 1, 'order_number': 'ORD-001', 'total': 99.99, 'status': 'pending'}, {'id': 2, 'user_id': 1, 'order_number': 'ORD-002', 'total': 149.5, 'status': 'completed'}, {'id': 3, 'user_id': 2, 'order_number': 'ORD-003', 'total': 79.99, 'status': 'pending'}]

# Update rows using mutations API
print("\n" + "=" * 70)
print("Updating data")
print("=" * 70)

from moltres.table.mutations import update_rows, delete_rows, merge_rows

table = db.table("users")
update_rows(table, where=col("id") == 1, values={"name": "Alice Updated"})

users = users_df.collect()
print(f"After update: {users}")
# Output: After update: [{'id': 1, 'name': 'Alice Updated', 'email': 'alice@example.com', 'active': 1}, {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'active': 1}, {'id': 3, 'name': 'Charlie', 'email': 'charlie@example.com', 'active': 0}]

# Delete rows
print("\n" + "=" * 70)
print("Deleting data")
print("=" * 70)

delete_rows(table, where=col("active") == 0)

users = users_df.collect()
print(f"After delete (inactive users removed): {users}")
# Output: After delete (inactive users removed): [{'id': 1, 'name': 'Alice Updated', 'email': 'alice@example.com', 'active': 1}, {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'active': 1}]

# Merge (upsert) rows
print("\n" + "=" * 70)
print("Merging (upserting) data")
print("=" * 70)

merge_data = [
    {"id": 1, "name": "Alice Merged", "email": "alice@example.com", "active": 1},
    {"id": 4, "name": "David", "email": "david@example.com", "active": 1},
]

merge_rows(
    table,
    merge_data,
    on=["id"],
    when_matched={"name": "name", "email": "email"},
    when_not_matched={"name": "name", "email": "email", "active": "active"},
)

users = users_df.collect()
print(f"After merge: {users}")
# Output: After merge: [{'id': 1, 'name': 'name', 'email': 'email', 'active': 1}, {'id': 2, 'name': 'Bob', 'email': 'bob@example.com', 'active': 1}, {'id': 4, 'name': 'David', 'email': 'david@example.com', 'active': 1}]

# ============================================================================
# Index Management
# ============================================================================

print("\n" + "=" * 70)
print("Dropping indexes")
print("=" * 70)

db.drop_index("idx_order_status", "orders").collect()
print("✓ Dropped index 'idx_order_status'")

# ============================================================================
# Cleanup
# ============================================================================

print("\n" + "=" * 70)
print("Cleaning up")
print("=" * 70)

db.drop_table("orders", if_exists=True).collect()
print("✓ Dropped 'orders' table")

db.drop_table("users", if_exists=True).collect()
print("✓ Dropped 'users' table")

db.close()
print("\n✓ Example completed successfully!")
