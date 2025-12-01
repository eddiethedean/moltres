"""Example: Window functions.

This example demonstrates using window functions for analytical queries.
"""

from moltres import connect, col
from moltres.expressions import functions as F
# Window functions use the .over() method directly on column expressions

db = connect("sqlite:///:memory:")

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
# Output: Running total: [{'product': 'Widget', 'amount': 100.0, 'running_total': 100.0}, {'product': 'Widget', 'amount': 150.0, 'running_total': 250.0}, {'product': 'Gadget', 'amount': 200.0, 'running_total': 450.0}, {'product': 'Gadget', 'amount': 175.0, 'running_total': 625.0}, {'product': 'Widget', 'amount': 120.0, 'running_total': 745.0}]

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
# Output: Product running total: [{'product': 'Gadget', 'amount': 200.0, 'product_running_total': 200.0}, {'product': 'Gadget', 'amount': 175.0, 'product_running_total': 375.0}, {'product': 'Widget', 'amount': 100.0, 'product_running_total': 100.0}, {'product': 'Widget', 'amount': 150.0, 'product_running_total': 250.0}, {'product': 'Widget', 'amount': 120.0, 'product_running_total': 370.0}]

# Row number (using ascending order)
result = df.select(
    col("product"),
    col("amount"),
    col("region"),
    F.row_number().over(partition_by=col("region"), order_by=col("amount")).alias("rank_in_region"),
)
results = result.collect()
print(f"Rank in region: {results}")
# Output: Rank in region: [{'product': 'Widget', 'amount': 100.0, 'region': 'North', 'rank_in_region': 1}, {'product': 'Widget', 'amount': 120.0, 'region': 'North', 'rank_in_region': 2}, {'product': 'Gadget', 'amount': 200.0, 'region': 'North', 'rank_in_region': 3}, {'product': 'Widget', 'amount': 150.0, 'region': 'South', 'rank_in_region': 1}, {'product': 'Gadget', 'amount': 175.0, 'region': 'South', 'rank_in_region': 2}]

# Rank and dense_rank (using ascending order)
result = df.select(
    col("product"),
    col("amount"),
    F.rank().over(partition_by=col("product"), order_by=col("amount")).alias("rank"),
    F.dense_rank().over(partition_by=col("product"), order_by=col("amount")).alias("dense_rank"),
)
results = result.collect()
print(f"Rank and dense_rank: {results}")
# Output: Rank and dense_rank: [{'product': 'Gadget', 'amount': 175.0, 'rank': 1, 'dense_rank': 1}, {'product': 'Gadget', 'amount': 200.0, 'rank': 2, 'dense_rank': 2}, {'product': 'Widget', 'amount': 100.0, 'rank': 1, 'dense_rank': 1}, {'product': 'Widget', 'amount': 120.0, 'rank': 2, 'dense_rank': 2}, {'product': 'Widget', 'amount': 150.0, 'rank': 3, 'dense_rank': 3}]

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
# Output: Cumulative sum: [{'product': 'Gadget', 'amount': 200.0, 'cumulative_sum': 200.0}, {'product': 'Gadget', 'amount': 175.0, 'cumulative_sum': 375.0}, {'product': 'Widget', 'amount': 100.0, 'cumulative_sum': 100.0}, {'product': 'Widget', 'amount': 150.0, 'cumulative_sum': 250.0}, {'product': 'Widget', 'amount': 120.0, 'cumulative_sum': 370.0}]

# Window functions in withColumn() (PySpark-compatible, v0.16.0+)
# This is now fully supported, matching PySpark's API
result = df.withColumn(
    "row_num",
    F.row_number().over(partition_by=col("region"), order_by=col("amount")),
)
results = result.collect()
print(f"Window function in withColumn: {results}")
# Output shows all original columns plus row_num

# Multiple window functions with withColumn (using different column names)
result = (
    df.withColumn(
        "row_num", F.row_number().over(partition_by=col("region"), order_by=col("amount"))
    )
    .withColumn("rank", F.rank().over(partition_by=col("region"), order_by=col("amount")))
    .withColumn(
        "dense_rank_val",
        F.dense_rank().over(partition_by=col("region"), order_by=col("amount")),
    )
)
results = result.collect()
print(f"Multiple window functions in withColumn: {results}")
# Output shows all original columns plus row_num, rank, and dense_rank

db.close()
