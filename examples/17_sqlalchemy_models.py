"""Example: SQLAlchemy ORM Model Integration.

This example demonstrates how to use SQLAlchemy ORM models with Moltres,
including creating tables from models, querying with models, and model-based joins.
"""

try:
    from sqlalchemy import Column, ForeignKey, Integer, String, Numeric, DateTime
    from sqlalchemy.orm import DeclarativeBase
except ImportError:
    print("SQLAlchemy is required for this example.")
    print("Install with: pip install sqlalchemy")
    exit(1)

from moltres import col, connect
from moltres.io.records import Records


# Define SQLAlchemy models
class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    age = Column(Integer, nullable=True)


class Order(Base):
    """Order model with foreign key."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Numeric(10, 2))
    created_at = Column(DateTime)


# Connect to database
import os

db_path = "example_sqlalchemy.db"
if os.path.exists(db_path):
    os.remove(db_path)
db = connect(f"sqlite:///{db_path}")

# ============================================================================
# Create tables from SQLAlchemy models
# ============================================================================

print("Creating tables from SQLAlchemy models...")
# Output: Creating tables from SQLAlchemy models...

# Create tables directly from model classes
user_table = db.create_table(User).collect()
order_table = db.create_table(Order).collect()

print(f"Created table: {user_table.name}")
# Output: Created table: users
print(f"Created table: {order_table.name}")
# Output: Created table: orders

# ============================================================================
# Insert data
# ============================================================================

print("\nInserting data...")
# Output:
# Output: Inserting data...

# Insert users
Records.from_list(
    [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
    ],
    database=db,
).insert_into("users")

# Insert orders
Records.from_list(
    [
        {"id": 1, "user_id": 1, "amount": 100.50},
        {"id": 2, "user_id": 1, "amount": 250.75},
        {"id": 3, "user_id": 2, "amount": 50.00},
    ],
    database=db,
).insert_into("orders")

print("Data inserted successfully")
# Output: Data inserted successfully

# ============================================================================
# Query using SQLAlchemy models
# ============================================================================

print("\nQuerying using SQLAlchemy models...")
# Output:
# Output: Querying using SQLAlchemy models...

# Get table handle from model
users_df = db.table(User).select()
all_users = users_df.collect()

print(f"\nAll users ({len(all_users)}):")
# Output:
# Output: All users (3):
for user in all_users:
    print(f"  - {user['name']} ({user['email']})")
# Output:   - Alice (alice@example.com)
# Output:   - Bob (bob@example.com)
# Output:   - Charlie (charlie@example.com)

# Filter using model-based table handle
adults_df = db.table(User).select().where(col("age") >= 30)
adults = adults_df.collect()

print(f"\nAdults ({len(adults)}):")
# Output:
# Output: Adults (2):
for adult in adults:
    print(f"  - {adult['name']} (age: {adult['age']})")
# Output:   - Alice (age: 30)
# Output:   - Charlie (age: 35)

# ============================================================================
# Model-based joins
# ============================================================================

print("\nPerforming model-based joins...")
# Output:
# Output: Performing model-based joins...

# Join orders with users using model classes
orders_df = db.table(Order).select()
users_df = db.table(User).select()
joined_df = orders_df.join(users_df, on=[("user_id", "id")])
# After join, select specific columns using col() or column names
joined_df = joined_df.select(col("name"), col("amount"))

results = joined_df.collect()

print(f"\nOrders with user names ({len(results)}):")
# Output:
# Output: Orders with user names (3):
for result in results:
    amount = result["amount"]
    amount_float = float(amount) if amount is not None else 0.0  # type: ignore[arg-type]
    print(f"  - {result['name']}: ${amount_float:.2f}")
# Output:   - Alice: $100.50
# Output:   - Alice: $250.75
# Output:   - Bob: $50.00

# ============================================================================
# Access model class from table handle
# ============================================================================

print("\nAccessing model class from table handle...")
# Output:
# Output: Accessing model class from table handle...

user_handle = db.table(User)
print(f"Table name: {user_handle.name}")
# Output: Table name: users
print(f"Model class: {user_handle.model_class}")
# Output: Model class: <class '__main__.User'>
print(f"Model class name: {user_handle.model_class.__name__ if user_handle.model_class else None}")
# Output: Model class name: User

# ============================================================================
# Backward compatibility
# ============================================================================

print("\nBackward compatibility: traditional API still works...")
# Output:
# Output: Backward compatibility: traditional API still works...

from moltres.table.schema import column

# Traditional API (string + columns) still works
product_table = db.create_table(
    "products",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("price", "REAL"),
    ],
).collect()

print(f"Created table using traditional API: {product_table.name}")
# Output: Created table using traditional API: products
print(f"Model class (should be None): {product_table.model_class}")
# Output: Model class (should be None): None

# Traditional table() with string also works
products_df = db.table("products").select()
print(f"Query using string table name works: {len(products_df.collect())} rows")
# Output: Query using string table name works: 0 rows

# ============================================================================
# Cleanup
# ============================================================================

print("\nExample completed successfully!")
# Output:
# Output: Example completed successfully!
print(f"\nNote: The database file '{db_path}' has been created.")
# Output:
# Output: Note: The database file 'example_sqlalchemy.db' has been created.
print("You can inspect it or delete it when done.")
# Output: You can inspect it or delete it when done.

db.close()
