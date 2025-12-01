"""Example: Raw SQL and SQL operations.

This example demonstrates executing raw SQL queries and using SQL features.
"""

from moltres import connect, col

db = connect("sqlite:///:memory:")

# Create table
from moltres.table.schema import column

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

orders_data = [
    {"id": 1, "customer_id": 1, "amount": 100.0, "order_date": "2024-01-01"},
    {"id": 2, "customer_id": 1, "amount": 200.0, "order_date": "2024-01-15"},
    {"id": 3, "customer_id": 2, "amount": 150.0, "order_date": "2024-02-01"},
]

Records(_data=orders_data, _database=db).insert_into("orders")

# Raw SQL query
df = db.sql("SELECT * FROM orders WHERE amount > :min_amount", min_amount=120.0)
results = df.collect()
print(f"Raw SQL results: {results}")
# Output: Raw SQL results: [{'id': 2, 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}, {'id': 3, 'customer_id': 2, 'amount': 150.0, 'order_date': '2024-02-01'}]

# Raw SQL with chaining
df = db.sql("SELECT * FROM orders").where(col("amount") > 150.0).order_by(col("order_date"))
results = df.collect()
print(f"Raw SQL with chaining: {results}")
# Output: Raw SQL with chaining: [{'id': 2, 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}]

# Common Table Expression (CTE)
df = db.table("orders").select()
cte = df.cte("high_value_orders")
result = cte.select().where(col("amount") > 150.0)
results = result.collect()
print(f"CTE results: {results}")
# Output: CTE results: [{'id': 2, 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}]

# Union
df1 = db.sql("SELECT * FROM orders WHERE customer_id = 1")
df2 = db.sql("SELECT * FROM orders WHERE customer_id = 2")
unioned = df1.union(df2)
results = unioned.collect()
print(f"Union results: {results}")
# Output: Union results: [{'id': 1, 'customer_id': 1, 'amount': 100.0, 'order_date': '2024-01-01'}, {'id': 2, 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}, {'id': 3, 'customer_id': 2, 'amount': 150.0, 'order_date': '2024-02-01'}]

# Union All
union_all = df1.unionAll(df2)
results = union_all.collect()
print(f"Union All results: {results}")
# Output: Union All results: [{'id': 1, 'customer_id': 1, 'amount': 100.0, 'order_date': '2024-01-01'}, {'id': 2, 'customer_id': 1, 'amount': 200.0, 'order_date': '2024-01-15'}, {'id': 3, 'customer_id': 2, 'amount': 150.0, 'order_date': '2024-02-01'}]

# Intersect
df1 = db.sql("SELECT customer_id FROM orders WHERE amount > 100")
df2 = db.sql("SELECT customer_id FROM orders WHERE amount < 200")
intersected = df1.intersect(df2)
results = intersected.collect()
print(f"Intersect results: {results}")
# Output: Intersect results: [{'customer_id': 1}, {'customer_id': 2}]

# Except
except_df = df1.except_(df2)
results = except_df.collect()
print(f"Except results: {results}")
# Output: Except results: []

db.close()
