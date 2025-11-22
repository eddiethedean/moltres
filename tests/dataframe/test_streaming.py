"""Tests for streaming read/write operations."""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from moltres import connect
from moltres.table.schema import ColumnDef


@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    return connect("sqlite:///:memory:")


@pytest.fixture
def large_csv_file():
    """Create a large CSV file for streaming tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "value"])
        writer.writeheader()
        # Write 25000 rows to test chunking
        for i in range(25000):
            writer.writerow({"id": i, "name": f"item_{i}", "value": i * 1.5})
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def large_jsonl_file():
    """Create a large JSONL file for streaming tests."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write 25000 rows
        for i in range(25000):
            json.dump({"id": i, "name": f"item_{i}", "value": i * 1.5}, f)
            f.write("\n")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


def test_stream_csv_read(db, large_csv_file):
    """Test streaming CSV read with chunked processing."""
    records = db.read.records.stream().option("chunk_size", 5000).csv(large_csv_file)

    # Records from streaming reads can be iterated directly
    # For streaming Records, we iterate row by row (not in chunks)
    total_rows = 0
    for row in records:
        assert isinstance(row, dict)
        assert "id" in row
        assert "name" in row
        assert "value" in row
        total_rows += 1
        if total_rows >= 5000:  # Check first chunk
            break

    # Materialize to check total
    all_rows = records.rows()
    assert len(all_rows) == 25000


def test_stream_csv_read_materialize(db, large_csv_file):
    """Test that streaming read can still materialize all data."""
    records = db.read.records.stream().option("chunk_size", 5000).csv(large_csv_file)

    # rows() should materialize all data
    rows = records.rows()
    assert isinstance(rows, list)
    assert len(rows) == 25000
    assert rows[0]["id"] == 0
    assert rows[-1]["id"] == 24999


def test_stream_jsonl_read(db, large_jsonl_file):
    """Test streaming JSONL read with chunked processing."""
    records = db.read.records.stream().option("chunk_size", 5000).jsonl(large_jsonl_file)

    # Iterate over records
    total_rows = 0
    for row in records:
        total_rows += 1
        if total_rows >= 100:  # Just check it works
            break

    # Materialize to check total
    all_rows = records.rows()
    assert len(all_rows) == 25000


def test_stream_write_table(db):
    """Test streaming write to table."""
    # Create a large dataset in memory
    db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert 20000 rows
    rows = [{"id": i, "name": f"item_{i}"} for i in range(20000)]
    db.createDataFrame(rows, pk="id").write.insertInto("source")

    # Read with streaming and insert into target table (using read.records for Records)
    from moltres.io.records import Records

    df = db.load.table("source")
    rows = df.collect()
    records = Records(_data=rows, _database=db)

    # Create target table
    db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert records into target
    records.insert_into("target")

    # Verify all rows were written
    result = db.table("target").select().collect()
    assert len(result) == 20000


def test_stream_write_csv(db, large_csv_file):
    """Test that Records can be read and inserted into a table (file write not supported for Records)."""
    records = db.read.records.stream().option("chunk_size", 5000).csv(large_csv_file)

    # Create a table and insert records
    db.create_table(
        "output",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
            ColumnDef(name="value", type_name="REAL"),
        ],
    ).collect()

    # Insert records into table
    records.insert_into("output")

    # Verify all rows were inserted
    result = db.table("output").select().collect()
    assert len(result) == 25000
    assert result[0]["id"] == 0
    assert result[-1]["id"] == 24999


def test_stream_write_jsonl(db, large_jsonl_file):
    """Test that Records can be read and inserted into a table."""
    records = db.read.records.stream().option("chunk_size", 5000).jsonl(large_jsonl_file)

    # Create a table and insert records
    db.create_table(
        "output",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
            ColumnDef(name="value", type_name="REAL"),
        ],
    ).collect()

    # Insert records into table
    records.insert_into("output")

    # Verify all rows were inserted
    result = db.table("output").select().collect()
    assert len(result) == 25000
    first = result[0]
    assert first["id"] == 0
    last = result[-1]
    assert last["id"] == 24999


def test_stream_sql_query(db):
    """Test streaming SQL query execution."""
    # Create table with data
    db.create_table(
        "test_table",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="value", type_name="TEXT"),
        ],
    ).collect()

    # Insert 15000 rows
    rows = [{"id": i, "value": f"value_{i}"} for i in range(15000)]
    db.createDataFrame(rows, pk="id").write.insertInto("test_table")

    # Query with streaming
    df = db.table("test_table").select()
    chunk_iter = df.collect(stream=True)

    total_rows = 0
    for chunk in chunk_iter:
        assert isinstance(chunk, list)
        total_rows += len(chunk)
        if chunk:
            assert "id" in chunk[0]
            assert "value" in chunk[0]

    assert total_rows == 15000


def test_stream_insert_into(db):
    """Test streaming insertInto with batch inserts."""
    # Create target table
    db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Create source table with data
    db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    rows = [{"id": i, "name": f"item_{i}"} for i in range(10000)]
    db.createDataFrame(rows, pk="id").write.insertInto("source")

    # Stream insert - Records can be inserted directly (using read.records for Records)
    from moltres.io.records import Records

    df = db.load.table("source")
    rows = df.collect()
    records = Records(_data=rows, _database=db)
    records.insert_into("target")

    # Verify
    result = db.table("target").select().collect()
    assert len(result) == 10000


def test_non_streaming_backward_compat(db, large_csv_file):
    """Test that non-streaming mode still works with Records."""
    # Use read.records for Records (backward compatibility)
    records = db.read.records.csv(large_csv_file)
    rows = records.rows()

    assert isinstance(rows, list)
    assert len(rows) == 25000
    # Should not be an iterator
    assert not hasattr(rows, "__next__")


def test_stream_with_explicit_schema(db, large_csv_file):
    """Test streaming read with explicit schema."""
    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
        ColumnDef(name="value", type_name="REAL"),
    ]

    records = db.read.records.stream().schema(schema).option("chunk_size", 5000).csv(large_csv_file)

    # Get all rows to verify types
    all_rows = records.rows()

    # Verify types are correct
    assert isinstance(all_rows[0]["id"], int)
    assert isinstance(all_rows[0]["name"], str)
    assert isinstance(all_rows[0]["value"], float)

    # Verify total
    assert len(all_rows) == 25000
