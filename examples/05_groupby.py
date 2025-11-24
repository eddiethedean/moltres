"""Example: GroupBy operations.

This example demonstrates grouping and aggregation operations.
"""

from moltres import connect, col
from moltres.expressions import functions as F

db = connect("sqlite:///example.db")

# Create table
from moltres.table.schema import column

db.create_table(
    "sales",
    [
        column("id", "INTEGER", primary_key=True),
        column("product", "TEXT"),
        column("category", "TEXT"),
        column("amount", "REAL"),
        column("region", "TEXT"),
    ],
).collect()

# Insert data
from moltres.io.records import Records

sales_data = [
    {"id": 1, "product": "Widget A", "category": "Electronics", "amount": 100.0, "region": "North"},
    {"id": 2, "product": "Widget B", "category": "Electronics", "amount": 150.0, "region": "South"},
    {"id": 3, "product": "Gadget X", "category": "Electronics", "amount": 200.0, "region": "North"},
    {"id": 4, "product": "Tool Y", "category": "Hardware", "amount": 75.0, "region": "South"},
    {"id": 5, "product": "Tool Z", "category": "Hardware", "amount": 125.0, "region": "North"},
]

Records(_data=sales_data, _database=db).insert_into("sales")

# Group by single column
df = db.table("sales").select()
grouped = df.group_by("category")

# Aggregate with Column expressions
result = grouped.agg(F.sum(col("amount")).alias("total"))
results = result.collect()
print(f"Total by category: {results}")
# Output: Total by category: [{'category': 'Electronics', 'total': 450.0}, {'category': 'Hardware', 'total': 200.0}]

# Aggregate with string
result = grouped.agg("amount")
results = result.collect()
print(f"Sum amount by category: {results}")
# Output: Sum amount by category: [{'category': 'Electronics', 'amount': 450.0}, {'category': 'Hardware', 'amount': 200.0}]

# Multiple aggregations
result = grouped.agg(
    {
        "amount": "sum",
        "id": "count",
    }
)
results = result.collect()
print(f"Multiple aggregations: {results}")
# Output: Multiple aggregations: [{'category': 'Electronics', 'amount': 450.0, 'id': 3}, {'category': 'Hardware', 'amount': 200.0, 'id': 2}]

# Group by multiple columns
grouped_multi = df.group_by("category", "region")
result = grouped_multi.agg(F.sum(col("amount")).alias("total"))
results = result.collect()
print(f"Total by category and region: {results}")
# Output: Total by category and region: [{'category': 'Electronics', 'region': 'North', 'total': 300.0}, {'category': 'Electronics', 'region': 'South', 'total': 150.0}, {'category': 'Hardware', 'region': 'North', 'total': 125.0}, {'category': 'Hardware', 'region': 'South', 'total': 75.0}]

# Order by after group by
result = grouped.agg(F.sum(col("amount")).alias("total")).order_by(col("total").desc())
results = result.collect()
print(f"Ordered by total: {results}")
# Output: Ordered by total: [{'category': 'Electronics', 'total': 450.0}, {'category': 'Hardware', 'total': 200.0}]

db.close()
