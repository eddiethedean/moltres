"""Examples demonstrating enhanced transaction control features in Moltres.

This example shows how to use:
- Savepoints and nested transactions
- Isolation levels
- Read-only transactions
- Transaction timeouts
- Transaction state inspection
- Row-level locking (FOR UPDATE, FOR SHARE)
"""

import sys
from pathlib import Path

# Add parent directory to path to import moltres
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from moltres import col, connect
from moltres.io.records import Records
from moltres.table.schema import column


def example_basic_transaction():
    """Example: Basic transaction usage."""
    print("=" * 60)
    print("Example: Basic Transaction")
    print("=" * 60)

    db = connect("sqlite:///:memory:")

    # Create a table
    db.create_table(
        "accounts",
        [
            column("id", "INTEGER", primary_key=True),
            column("balance", "REAL"),
        ],
    ).collect()

    # Insert initial data
    Records(
        _data=[
            {"id": 1, "balance": 1000.0},
            {"id": 2, "balance": 500.0},
        ],
        _database=db,
    ).insert_into("accounts")

    # Transfer money in a transaction
    with db.transaction() as txn:
        # Withdraw from account 1
        from moltres.table.mutations import update_rows

        table = db.table("accounts")
        update_rows(
            table,
            where=col("id") == 1,
            values={"balance": col("balance") - 100.0},
            transaction=txn.connection,
        )

        # Deposit to account 2
        update_rows(
            table,
            where=col("id") == 2,
            values={"balance": col("balance") + 100.0},
            transaction=txn.connection,
        )

        # Transaction commits automatically on exit

    # Verify the transfer
    results = db.table("accounts").select().collect()
    print("Account balances after transfer:")
    for row in results:
        print(f"  Account {row['id']}: ${row['balance']}")

    db.close()
    print()


def example_savepoints():
    """Example: Using savepoints for partial rollback."""
    print("=" * 60)
    print("Example: Savepoints")
    print("=" * 60)

    db = connect("sqlite:///:memory:")

    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("status", "TEXT"),
        ],
    ).collect()

    with db.transaction() as txn:
        # Initial insert
        Records(_data=[{"id": 1, "amount": 100.0, "status": "pending"}], _database=db).insert_into(
            "orders"
        )

        # Create a savepoint
        checkpoint = txn.savepoint("processing_checkpoint")
        print(f"Created savepoint: {checkpoint}")

        # More operations after savepoint
        Records(_data=[{"id": 2, "amount": 200.0, "status": "pending"}], _database=db).insert_into(
            "orders"
        )
        Records(_data=[{"id": 3, "amount": 300.0, "status": "pending"}], _database=db).insert_into(
            "orders"
        )

        print(f"Orders after savepoint: {len(db.table('orders').select().collect())}")

        # Rollback to savepoint
        print("Rolling back to savepoint...")
        txn.rollback_to_savepoint(checkpoint)

        # Verify only first insert remains
        results = db.table("orders").select().collect()
        print(f"Orders after rollback: {len(results)}")
        assert len(results) == 1
        assert results[0]["id"] == 1

    db.close()
    print()


def example_nested_transactions():
    """Example: Nested transactions using savepoints."""
    print("=" * 60)
    print("Example: Nested Transactions")
    print("=" * 60)

    db = connect("sqlite:///:memory:")

    db.create_table(
        "inventory",
        [
            column("product_id", "INTEGER", primary_key=True),
            column("quantity", "INTEGER"),
        ],
    ).collect()

    Records(
        _data=[
            {"product_id": 1, "quantity": 100},
            {"product_id": 2, "quantity": 50},
        ],
        _database=db,
    ).insert_into("inventory")

    with db.transaction() as outer:
        # Outer transaction operation
        from moltres.table.mutations import update_rows

        table = db.table("inventory")
        update_rows(
            table,
            where=col("product_id") == 1,
            values={"quantity": col("quantity") - 10},
            transaction=outer.connection,
        )

        # Nested transaction with savepoint
        try:
            with db.transaction(savepoint=True) as inner:
                # Inner transaction operation
                update_rows(
                    table,
                    where=col("product_id") == 2,
                    values={"quantity": col("quantity") - 20},
                    transaction=inner.connection,
                )

                # Simulate an error in inner transaction
                raise ValueError("Error in inner transaction")

        except ValueError:
            print("Inner transaction failed and rolled back")
            # Outer transaction continues

        # Verify outer transaction persisted
        results = db.table("inventory").select().where(col("product_id") == 1).collect()
        assert results[0]["quantity"] == 90

        # Verify inner transaction was rolled back
        results = db.table("inventory").select().where(col("product_id") == 2).collect()
        assert results[0]["quantity"] == 50  # Unchanged

    print("Outer transaction committed successfully")
    db.close()
    print()


def example_transaction_state_inspection():
    """Example: Inspecting transaction state."""
    print("=" * 60)
    print("Example: Transaction State Inspection")
    print("=" * 60)

    db = connect("sqlite:///:memory:")

    print(f"Initially in transaction: {db.is_in_transaction()}")
    assert db.get_transaction_status() is None

    with db.transaction(readonly=False) as txn:
        print(f"In transaction: {db.is_in_transaction()}")
        status = db.get_transaction_status()
        print(f"Transaction status: {status}")
        assert status is not None
        assert status["readonly"] is False

        # Create a savepoint
        sp = txn.savepoint("test_sp")
        status = db.get_transaction_status()
        print(f"Status after savepoint: {status['savepoints']}")

    print(f"After transaction: {db.is_in_transaction()}")
    db.close()
    print()


def example_row_level_locking():
    """Example: Row-level locking with FOR UPDATE."""
    print("=" * 60)
    print("Example: Row-Level Locking")
    print("=" * 60)

    db = connect("sqlite:///:memory:")

    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("status", "TEXT"),
            column("processed_by", "TEXT"),
        ],
    ).collect()

    Records(
        _data=[
            {"id": 1, "status": "pending", "processed_by": None},
            {"id": 2, "status": "pending", "processed_by": None},
        ],
        _database=db,
    ).insert_into("orders")

    with db.transaction() as txn:
        # Select and lock rows for update
        df = db.table("orders").select().where(col("status") == "pending")
        locked_df = df.select_for_update()

        results = locked_df.collect()
        print(f"Locked {len(results)} rows for update")

        # Process the locked rows
        from moltres.table.mutations import update_rows

        table = db.table("orders")
        for row in results:
            update_rows(
                table,
                where=col("id") == row["id"],
                values={"status": "processing", "processed_by": "worker1"},
                transaction=txn.connection,
            )

    # Verify processing
    results = db.table("orders").select().collect()
    for row in results:
        print(f"Order {row['id']}: {row['status']} by {row['processed_by']}")

    db.close()
    print()


def example_transaction_with_error_handling():
    """Example: Error handling with transactions."""
    print("=" * 60)
    print("Example: Error Handling")
    print("=" * 60)

    db = connect("sqlite:///:memory:")

    db.create_table(
        "payments",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("status", "TEXT"),
        ],
    ).collect()

    Records(_data=[{"id": 1, "amount": 100.0, "status": "pending"}], _database=db).insert_into(
        "payments"
    )

    try:
        with db.transaction() as txn:
            # Critical operation
            from moltres.table.mutations import update_rows

            table = db.table("payments")
            update_rows(
                table,
                where=col("id") == 1,
                values={"status": "processing"},
                transaction=txn.connection,
            )

            # Create checkpoint for non-critical operation
            checkpoint = txn.savepoint("notification_checkpoint")

            # Non-critical operation that might fail
            try:
                # Simulate notification failure
                raise ConnectionError("Failed to send notification")
            except ConnectionError:
                # Rollback only the notification, keep payment processing
                print("Notification failed, rolling back to checkpoint")
                txn.rollback_to_savepoint(checkpoint)

            # Payment processing is preserved
            results = db.table("payments").select().where(col("id") == 1).collect()
            assert results[0]["status"] == "processing"

    except Exception as e:
        print(f"Transaction failed: {e}")

    db.close()
    print()


if __name__ == "__main__":
    example_basic_transaction()
    example_savepoints()
    example_nested_transactions()
    example_transaction_state_inspection()
    example_row_level_locking()
    example_transaction_with_error_handling()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
