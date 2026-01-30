"""Example: Show SQL output for Moltres queries.

This demo builds several DataFrame queries and prints the raw SQL
that Moltres generates using to_sql(), show_sql(), and the .sql property.
"""

from moltres import col, column, connect
from moltres.expressions import functions as F
from moltres.io.records import Records

# Use in-memory SQLite so no files to clean up
db = connect("sqlite:///:memory:")

# ---------------------------------------------------------------------------
# Setup: create tables and insert sample data
# ---------------------------------------------------------------------------

db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("country", "TEXT"),
        column("age", "INTEGER"),
    ],
).collect()

db.create_table(
    "orders",
    [
        column("id", "INTEGER", primary_key=True),
        column("user_id", "INTEGER"),
        column("amount", "REAL"),
        column("active", "INTEGER"),  # SQLite uses 0/1 for bool
    ],
).collect()

Records.from_list(
    [
        {"id": 1, "name": "Alice", "country": "US", "age": 30},
        {"id": 2, "name": "Bob", "country": "UK", "age": 25},
        {"id": 3, "name": "Charlie", "country": "US", "age": 35},
    ],
    database=db,
).insert_into("users")

Records.from_list(
    [
        {"id": 1, "user_id": 1, "amount": 100.0, "active": 1},
        {"id": 2, "user_id": 1, "amount": 50.0, "active": 1},
        {"id": 3, "user_id": 2, "amount": 75.0, "active": 0},
    ],
    database=db,
).insert_into("orders")

# ---------------------------------------------------------------------------
# Demo 1: Simple SELECT + WHERE
# ---------------------------------------------------------------------------

print("=" * 70)
print("1. Simple query: select from users where age > 25")
print("=" * 70)

df1 = db.table("users").select().where(col("age") > 25)

print("\n--- to_sql() (compact) ---")
print(df1.to_sql())

print("\n--- to_sql(pretty=True) ---")
print(df1.to_sql(pretty=True))

print("\n--- show_sql() (prints to stdout) ---")
df1.show_sql()

# ---------------------------------------------------------------------------
# Demo 2: JOIN
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("2. Join: users + orders on user_id")
print("=" * 70)

df2 = (
    db.table("orders")
    .select()
    .join(db.table("users").select(), on=[col("orders.user_id") == col("users.id")])
    .where(col("orders.active") == 1)
)

print("\n--- Raw SQL ---")
print(df2.to_sql(pretty=True))

# ---------------------------------------------------------------------------
# Demo 3: GROUP BY + aggregation
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("3. Group by country, sum(amount) with join")
print("=" * 70)

df3 = (
    db.table("orders")
    .select()
    .join(db.table("users").select(), on=[col("orders.user_id") == col("users.id")])
    .where(col("orders.active") == 1)
    .group_by("country")
    .agg(F.sum(col("amount")).alias("total_amount"))
)

print("\n--- Raw SQL ---")
print(df3.to_sql(pretty=True))

# ---------------------------------------------------------------------------
# Demo 4: ORDER BY + LIMIT
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("4. Order by age desc, limit 2")
print("=" * 70)

df4 = db.table("users").select().order_by(col("age").desc()).limit(2)

print("\n--- Raw SQL ---")
print(df4.to_sql(pretty=True))

# ---------------------------------------------------------------------------
# Demo 5: Using .sql property and sql_preview()
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("5. .sql property and sql_preview()")
print("=" * 70)

df5 = db.table("users").select(col("name"), col("country")).where(col("age") >= 25).order_by("name")

print("\n--- df.sql (formatted string) ---")
print(df5.sql)

print("\n--- df.sql_preview(max_length=80) ---")
print(repr(df5.sql_preview(max_length=80)))

db.close()
print("\nDone.")
