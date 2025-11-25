"""Transaction integration tests."""

import pytest

from moltres import col
from moltres.io.records import Records
from moltres.table.schema import column


@pytest.mark.integration
def test_successful_transaction_flow(transaction_db):
    """Test successful transaction: begin → multiple operations → commit → verify."""
    db = transaction_db

    # Begin transaction
    with db.transaction() as txn:
        # Multiple inserts (uses active transaction automatically)
        Records.from_list(
            [{"id": 1, "value": "first"}, {"id": 2, "value": "second"}],
            database=db,
        ).insert_into("test_table")

        # Update within transaction (use connection from transaction)
        from moltres.table.mutations import update_rows

        table = db.table("test_table")
        updated = update_rows(
            table, where=col("id") == 1, values={"value": "updated"}, transaction=txn.connection
        )
        assert updated == 1

        # Transaction commits automatically on exit

    # Verify all changes persisted after commit
    df = db.table("test_table").select()
    results = df.collect()
    assert len(results) == 2
    values = {r["id"]: r["value"] for r in results}
    assert values[1] == "updated"
    assert values[2] == "second"


@pytest.mark.integration
def test_rollback_on_error(transaction_db):
    """Test rollback: begin → operations → error → rollback → verify no changes."""
    db = transaction_db

    # Insert initial data
    Records.from_list([{"id": 1, "value": "initial"}], database=db).insert_into("test_table")

    initial_count = len(db.table("test_table").select().collect())

    # Begin transaction and perform operations
    try:
        with db.transaction():
            # Insert new row
            Records.from_list(
                [{"id": 2, "value": "new"}],
                database=db,
            ).insert_into("test_table")

            # Simulate error (constraint violation)
            Records.from_list(
                [{"id": 1, "value": "duplicate"}],  # Duplicate primary key
                database=db,
            ).insert_into("test_table")

            # Should not reach here
            assert False, "Should have raised an error"
    except Exception:
        # Transaction should rollback automatically
        pass

    # Verify no changes persisted
    final_count = len(db.table("test_table").select().collect())
    assert final_count == initial_count

    # Verify initial data unchanged
    results = db.table("test_table").select().where(col("id") == 1).collect()
    assert len(results) == 1
    assert results[0]["value"] == "initial"


@pytest.mark.integration
def test_rollback_explicit(transaction_db):
    """Test explicit rollback."""
    db = transaction_db

    # Begin transaction and rollback by exiting with exception
    try:
        with db.transaction():
            # Insert data (uses active transaction automatically)
            Records.from_list(
                [{"id": 1, "value": "test"}],
                database=db,
            ).insert_into("test_table")

            # Explicit rollback by raising exception
            raise Exception("Rollback")
    except Exception:
        pass  # Expected for rollback

    # Verify no changes persisted
    results = db.table("test_table").select().collect()
    assert len(results) == 0


@pytest.mark.integration
def test_transaction_with_dataframe_operations(transaction_db):
    """Test transaction with DataFrame write operations."""
    db = transaction_db

    # Create source table

    db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "TEXT"),
        ],
    ).collect()

    Records.from_list(
        [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
        database=db,
    ).insert_into("source")

    # Begin transaction
    with db.transaction():
        # DataFrame write within transaction
        df = db.table("source").select()
        df.write.save_as_table("target")

        # Query within transaction
        target_df = db.table("target").select()
        results = target_df.collect()
        assert len(results) == 2

    # Verify table created after commit
    target_df = db.table("target").select()
    results = target_df.collect()
    assert len(results) == 2


@pytest.mark.integration
def test_transaction_isolation_read_uncommitted(transaction_db):
    """Test transaction isolation - read uncommitted behavior."""
    db = transaction_db

    # Insert initial data
    Records.from_list([{"id": 1, "value": "initial"}], database=db).insert_into("test_table")

    initial_count = len(db.table("test_table").select().collect())

    # Begin transaction 1 and rollback
    try:
        with db.transaction():
            # Insert in transaction 1 (uses active transaction automatically)
            Records.from_list(
                [{"id": 2, "value": "uncommitted"}],
                database=db,
            ).insert_into("test_table")

            # Query within transaction (should see uncommitted changes)
            results = db.table("test_table").select().collect()
            # Should see both initial and new row within transaction
            assert len(results) >= initial_count

            # Rollback transaction 1 by raising exception
            raise Exception("Rollback")
    except Exception:
        pass  # Expected for rollback

    # Verify uncommitted changes not persisted
    results = db.table("test_table").select().collect()
    assert len(results) == 1
    assert results[0]["id"] == 1


@pytest.mark.integration
def test_multiple_transactions_sequential(transaction_db):
    """Test multiple sequential transactions."""
    db = transaction_db

    # Transaction 1
    with db.transaction():
        Records.from_list([{"id": 1, "value": "first"}], database=db).insert_into("test_table")

    # Transaction 2
    with db.transaction():
        Records.from_list([{"id": 2, "value": "second"}], database=db).insert_into("test_table")

    # Verify both committed
    results = db.table("test_table").select().collect()
    assert len(results) == 2


@pytest.mark.integration
def test_transaction_with_update_and_delete(transaction_db):
    """Test transaction with update and delete operations."""
    db = transaction_db

    # Insert initial data
    Records.from_list(
        [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}, {"id": 3, "value": "c"}],
        database=db,
    ).insert_into("test_table")

    # Begin transaction
    with db.transaction() as txn:
        # Update (use connection from transaction)
        from moltres.table.mutations import update_rows, delete_rows

        table = db.table("test_table")

        updated = update_rows(
            table, where=col("id") == 1, values={"value": "updated"}, transaction=txn.connection
        )
        assert updated == 1

        # Delete
        deleted = delete_rows(table, where=col("id") == 3, transaction=txn.connection)
        assert deleted == 1

        # Transaction commits automatically on exit

    # Verify after commit
    results = db.table("test_table").select().collect()
    assert len(results) == 2
    values = {r["id"]: r["value"] for r in results}
    assert values[1] == "updated"
    assert 3 not in values
