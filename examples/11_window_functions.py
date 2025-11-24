"""Example: Window functions.

This example demonstrates using window functions for analytical queries.
"""

from moltres import connect, col
from moltres.expressions import functions as F
# Window functions use the .over() method directly on column expressions

db = connect("sqlite:///example.db")

# Create table
from moltres.table.schema import column

db.create_table(
    "sales",
    [
        column("id", "INTEGER", primary_key=True),
        column("product", "TEXT"),
        column("amount", "REAL"),
        column("sale_date", "TEXT"),
        column("region", "TEXT"),
    ],
).collect()

# Insert data
from moltres.io.records import Records

sales_data = [
    {"id": 1, "product": "Widget", "amount": 100.0, "sale_date": "2024-01-01", "region": "North"},
    {"id": 2, "product": "Widget", "amount": 150.0, "sale_date": "2024-01-02", "region": "South"},
    {"id": 3, "product": "Gadget", "amount": 200.0, "sale_date": "2024-01-03", "region": "North"},
    {"id": 4, "product": "Gadget", "amount": 175.0, "sale_date": "2024-01-04", "region": "South"},
    {"id": 5, "product": "Widget", "amount": 120.0, "sale_date": "2024-01-05", "region": "North"},
]

Records(_data=sales_data, _database=db).insert_into("sales")

df = db.table("sales").select()

# Window function: Running total
result = df.select(
    col("product"),
    col("amount"),
    F.sum(col("amount")).over(order_by=col("sale_date")).alias("running_total"),
)
results = result.collect()
print(f"Running total: {results}")

# Window function: Partition by
result = df.select(
    col("product"),
    col("amount"),
    F.sum(col("amount"))
    .over(partition_by=col("product"), order_by=col("sale_date"))
    .alias("product_running_total"),
)
results = result.collect()
print(f"Product running total: {results}")

# Row number (using ascending order)
result = df.select(
    col("product"),
    col("amount"),
    col("region"),
    F.row_number().over(partition_by=col("region"), order_by=col("amount")).alias("rank_in_region"),
)
results = result.collect()
print(f"Rank in region: {results}")

# Rank and dense_rank (using ascending order)
result = df.select(
    col("product"),
    col("amount"),
    F.rank().over(partition_by=col("product"), order_by=col("amount")).alias("rank"),
    F.dense_rank().over(partition_by=col("product"), order_by=col("amount")).alias("dense_rank"),
)
results = result.collect()
print(f"Rank and dense_rank: {results}")

# Window with rows between
result = df.select(
    col("product"),
    col("amount"),
    F.sum(col("amount"))
    .over(
        partition_by=col("product"),
        order_by=col("sale_date"),
        rows_between=(None, 0),  # UNBOUNDED PRECEDING to CURRENT ROW
    )
    .alias("cumulative_sum"),
)
results = result.collect()
print(f"Cumulative sum: {results}")

db.close()
