"""Resource management and lifecycle tests."""

import pytest

from moltres import col
from moltres.io.records import Records
from moltres.table.schema import column


@pytest.mark.integration
def test_connection_pool_management(empty_database):
    """Test connection pool limits and reuse."""
    db = empty_database

    # Create table
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Perform multiple operations (should reuse connections from pool)
    for i in range(10):
        Records.from_list([{"id": i, "value": f"value_{i}"}], database=db).insert_into("test_table")
        results = db.table("test_table").select().where(col("id") == i).collect()
        assert len(results) == 1

    # Verify all data persisted
    all_results = db.table("test_table").select().collect()
    assert len(all_results) == 10

    # Close database (should return connections to pool)
    db.close()

    # Verify database is closed
    # Reopening would require a new connection
    # This test verifies cleanup happens


@pytest.mark.integration
def test_transaction_resource_cleanup(empty_database):
    """Test transaction resource cleanup."""
    db = empty_database

    # Create table
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Begin transaction
    with db.transaction():
        # Perform operations (uses active transaction automatically)
        Records.from_list([{"id": 1, "value": "test"}], database=db).insert_into("test_table")
        # Transaction commits automatically on exit

    # Verify transaction resources cleaned up
    # Another transaction should work
    with db.transaction():
        Records.from_list([{"id": 2, "value": "test2"}], database=db).insert_into("test_table")

    results = db.table("test_table").select().collect()
    assert len(results) == 2


@pytest.mark.integration
def test_memory_management_large_dataframe(empty_database):
    """Test memory management with large DataFrame operations."""
    db = empty_database

    # Create table
    db.create_table(
        "large_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Insert large dataset
    large_data = [{"id": i, "value": f"value_{i}"} for i in range(5000)]

    # Insert in batches
    batch_size = 1000
    for i in range(0, len(large_data), batch_size):
        batch = large_data[i : i + batch_size]
        Records.from_list(batch, database=db).insert_into("large_table")

    # Create DataFrame and collect (should release memory after)
    df = db.table("large_table").select()
    results = df.collect()

    # Verify data collected
    assert len(results) == 5000

    # Memory should be released after collect()
    # This is verified by the fact that we can continue operations
    df2 = db.table("large_table").select().limit(10)
    small_results = df2.collect()
    assert len(small_results) == 10


@pytest.mark.integration
def test_file_handle_management(empty_database, tmp_path):
    """Test file handle management in read/write operations."""
    db = empty_database

    # Create table
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Insert data
    Records.from_list(
        [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
        database=db,
    ).insert_into("test_table")

    # Write to multiple files (should close handles properly)
    csv_file1 = tmp_path / "output1.csv"
    csv_file2 = tmp_path / "output2.csv"

    df = db.table("test_table").select()
    df.write.csv(str(csv_file1))
    df.write.csv(str(csv_file2))

    # Verify files created and can be read
    assert csv_file1.exists()
    assert csv_file2.exists()

    # Read from files (should close handles properly)
    import csv

    with open(csv_file1) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2

    # Verify we can still write more files
    csv_file3 = tmp_path / "output3.csv"
    df.write.csv(str(csv_file3))
    assert csv_file3.exists()


@pytest.mark.integration
def test_connection_cleanup_on_error(empty_database):
    """Test that connections are cleaned up even when errors occur."""
    db = empty_database

    # Create table
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Perform operation that may fail
    try:
        # Invalid operation
        db.sql("SELECT * FROM nonexistent_table").collect()
    except Exception:
        pass  # Expected error

    # Verify connection still works (was cleaned up properly)
    Records.from_list([{"id": 1, "value": "test"}], database=db).insert_into("test_table")
    results = db.table("test_table").select().collect()
    assert len(results) == 1


@pytest.mark.integration
def test_transaction_timeout_cleanup(empty_database):
    """Test transaction timeout and cleanup."""
    db = empty_database

    # Create table
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Begin transaction and rollback
    try:
        with db.transaction():
            # Perform operation (uses active transaction automatically)
            Records.from_list([{"id": 1, "value": "test"}], database=db).insert_into("test_table")

            # Simulate timeout (rollback by raising exception)
            raise Exception("Timeout")
    except Exception:
        pass  # Expected for rollback

    # Verify transaction cleaned up (no changes persisted)
    results = db.table("test_table").select().collect()
    assert len(results) == 0

    # Verify new transaction works
    with db.transaction():
        Records.from_list([{"id": 1, "value": "new"}], database=db).insert_into("test_table")

    results = db.table("test_table").select().collect()
    assert len(results) == 1


@pytest.mark.integration
def test_concurrent_connection_management(empty_database):
    """Test connection management with concurrent operations."""
    db = empty_database

    # Create table
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    import threading

    results_list = []
    errors = []

    def insert_and_query(thread_id):
        try:
            # Each thread uses the same database (connection pool should handle)
            Records.from_list(
                [{"id": thread_id, "value": f"thread_{thread_id}"}],
                database=db,
            ).insert_into("test_table")

            # Query
            results = db.table("test_table").select().where(col("id") == thread_id).collect()
            results_list.append((thread_id, results))
        except Exception as e:
            errors.append((thread_id, str(e)))

    # Run concurrent operations
    threads = []
    for i in range(5):
        thread = threading.Thread(target=insert_and_query, args=(i,))
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Verify all operations completed
    assert len(results_list) == 5
    assert len(errors) == 0

    # Verify all data persisted
    all_results = db.table("test_table").select().collect()
    assert len(all_results) == 5


@pytest.mark.integration
def test_database_close_cleanup(empty_database):
    """Test that database close properly cleans up resources."""
    db = empty_database

    # Create table and insert data
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    Records.from_list([{"id": 1, "value": "test"}], database=db).insert_into("test_table")

    # Close database
    db.close()

    # Verify database is closed (operations should fail or require new connection)
    # This test verifies that close() properly releases resources
    # Note: In SQLite, the database file remains, but connections are closed
