"""Example: Join operations.

This example demonstrates various join types and join operations.
"""

from moltres import connect

db = connect("sqlite:///:memory:")

# Clean up any existing tables
db.drop_table("customers", if_exists=True).collect()
db.drop_table("orders", if_exists=True).collect()

# Create tables
from moltres.table.schema import column

db.create_table(
    "customers",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("email", "TEXT"),
    ],
).collect()

db.create_table(
    "orders",
    [
        column("id", "INTEGER", primary_key=True),
        column("customer_id", "INTEGER"),
        column("amount", "REAL"),
        column("order_date", "TEXT"),
    ],
).collect()

# Insert data
from moltres.io.records import Records

customers_data = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
]

orders_data = [
    {"id": 1, "customer_id": 1, "amount": 100.0, "order_date": "2024-01-01"},
    {"id": 2, "customer_id": 1, "amount": 200.0, "order_date": "2024-01-15"},
    {"id": 3, "customer_id": 2, "amount": 150.0, "order_date": "2024-02-01"},
]

Records(_data=customers_data, _database=db).insert_into("customers")
Records(_data=orders_data, _database=db).insert_into("orders")

# Inner join
from moltres import col

customers = db.table("customers").select()
orders = db.table("orders").select()

joined = customers.join(orders, on=[col("customers.id") == col("orders.customer_id")], how="inner")
results = joined.collect()
print(f"Inner join: {results}")
# Output: Inner join: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'customer_id': 1, 'amount': 100.0, 'order_date': '2024-01-01'}, {'id': 2, 'name': 'Alice', 'email': 'alice@example.com', 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}, {'id': 3, 'name': 'Bob', 'email': 'bob@example.com', 'customer_id': 2, 'amount': 150.0, 'order_date': '2024-02-01'}]

# Left join
left_joined = customers.join(
    orders, on=[col("customers.id") == col("orders.customer_id")], how="left"
)
results = left_joined.collect()
print(f"Left join: {results}")
# Output: Left join: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'customer_id': 1, 'amount': 100.0, 'order_date': '2024-01-01'}, {'id': 2, 'name': 'Alice', 'email': 'alice@example.com', 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}, {'id': 3, 'name': 'Bob', 'email': 'bob@example.com', 'customer_id': 2, 'amount': 150.0, 'order_date': '2024-02-01'}, {'id': None, 'name': 'Charlie', 'email': 'charlie@example.com', 'customer_id': None, 'amount': None, 'order_date': None}]

# Join with on parameter (using column names)
joined_with_on = customers.join(
    orders, on=[col("customers.id") == col("orders.customer_id")], how="inner"
)
results = joined_with_on.collect()
print(f"Join with on parameter: {results}")
# Output: Join with on parameter: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'customer_id': 1, 'amount': 100.0, 'order_date': '2024-01-01'}, {'id': 2, 'name': 'Alice', 'email': 'alice@example.com', 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}, {'id': 3, 'name': 'Bob', 'email': 'bob@example.com', 'customer_id': 2, 'amount': 150.0, 'order_date': '2024-02-01'}]

# Select specific columns after join (use column names directly)
joined_select = joined.select("name", "amount", "order_date")
results = joined_select.collect()
print(f"Joined with selected columns: {results}")
# Output: Joined with selected columns: [{'name': 'Alice', 'amount': 100.0, 'order_date': '2024-01-01'}, {'name': 'Alice', 'amount': 200.0, 'order_date': '2024-01-15'}, {'name': 'Bob', 'amount': 150.0, 'order_date': '2024-02-01'}]

db.close()
