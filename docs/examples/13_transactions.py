"""Example: Transactions.

This example demonstrates using transactions for atomic operations.
"""

from typing import cast

from moltres import connect, col

db = connect("sqlite:///example.db")

# Create table
from moltres.table.schema import column

db.create_table(
    "accounts",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("balance", "REAL"),
    ],
).collect()

# Insert initial data
from moltres.io.records import Records

accounts_data = [
    {"id": 1, "name": "Alice", "balance": 1000.0},
    {"id": 2, "name": "Bob", "balance": 500.0},
]

Records(_data=accounts_data, _database=db).insert_into("accounts")

# Transaction: Transfer money
from moltres.table.mutations import update_rows

# Get initial balances
df = db.table("accounts").select()
initial_results = df.collect()
alice_balance: float = cast(float, next(r["balance"] for r in initial_results if r["id"] == 1))
bob_balance: float = cast(float, next(r["balance"] for r in initial_results if r["id"] == 2))

with db.transaction() as txn:
    # Debit from Alice
    table = db.table("accounts")
    update_rows(
        table,
        where=col("id") == 1,
        values={"balance": alice_balance - 100.0},
        transaction=txn.connection,
    )

    # Credit to Bob
    update_rows(
        table,
        where=col("id") == 2,
        values={"balance": bob_balance + 100.0},
        transaction=txn.connection,
    )

    # Transaction commits automatically on exit if no exception

# Verify transfer
results = df.collect()
print(f"After transfer: {results}")
# Output: After transfer: [{'id': 1, 'name': 'Alice', 'balance': 900.0}, {'id': 2, 'name': 'Bob', 'balance': 600.0}]

# Transaction with rollback
try:
    with db.transaction() as txn:
        alice_balance = cast(float, next(r["balance"] for r in results if r["id"] == 1))
        update_rows(
            table,
            where=col("id") == 1,
            values={"balance": alice_balance - 1000.0},
            transaction=txn.connection,
        )
        # This would cause negative balance - rollback
        raise ValueError("Insufficient funds")
except ValueError:
    pass  # Transaction automatically rolls back

# Verify rollback (balance should be unchanged)
results = df.collect()
print(f"After rollback: {results}")
# Output: After rollback: [{'id': 1, 'name': 'Alice', 'balance': 900.0}, {'id': 2, 'name': 'Bob', 'balance': 600.0}]

# Manual commit/rollback
with db.transaction() as txn:
    alice_balance = cast(float, next(r["balance"] for r in results if r["id"] == 1))
    update_rows(
        table,
        where=col("id") == 1,
        values={"balance": alice_balance - 50.0},
        transaction=txn.connection,
    )
    txn.commit()  # Explicit commit

# Verify manual transaction
results = df.collect()
print(f"After manual transaction: {results}")
# Output: After manual transaction: [{'id': 1, 'name': 'Alice', 'balance': 850.0}, {'id': 2, 'name': 'Bob', 'balance': 600.0}]

db.close()
