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
    df = db.read.stream().option("chunk_size", 5000).csv(large_csv_file)

    # collect(stream=True) should return an iterator
    chunk_iter = df.collect(stream=True)
    assert hasattr(chunk_iter, "__iter__")

    # Verify chunks are correct size
    total_rows = 0
    chunk_count = 0
    for chunk in chunk_iter:
        assert isinstance(chunk, list)
        assert len(chunk) <= 5000  # Chunk size limit
        if chunk:
            assert "id" in chunk[0]
            assert "name" in chunk[0]
            assert "value" in chunk[0]
        total_rows += len(chunk)
        chunk_count += 1

    assert total_rows == 25000
    assert chunk_count >= 5  # Should have at least 5 chunks (25000 / 5000)


def test_stream_csv_read_materialize(db, large_csv_file):
    """Test that streaming read can still materialize all data."""
    df = db.read.stream().option("chunk_size", 5000).csv(large_csv_file)

    # collect() without stream should materialize all data
    rows = df.collect()
    assert isinstance(rows, list)
    assert len(rows) == 25000
    assert rows[0]["id"] == 0
    assert rows[-1]["id"] == 24999


def test_stream_jsonl_read(db, large_jsonl_file):
    """Test streaming JSONL read with chunked processing."""
    df = db.read.stream().option("chunk_size", 5000).jsonl(large_jsonl_file)

    chunk_iter = df.collect(stream=True)
    total_rows = 0
    for chunk in chunk_iter:
        total_rows += len(chunk)

    assert total_rows == 25000


def test_stream_write_table(db):
    """Test streaming write to table."""
    # Create a large dataset in memory
    db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    )

    # Insert 20000 rows
    rows = [{"id": i, "name": f"item_{i}"} for i in range(20000)]
    db.table("source").insert(rows)

    # Read with streaming and write with streaming
    df = db.read.stream().option("chunk_size", 5000).table("source")

    df.write.stream().mode("overwrite").save_as_table("target")

    # Verify all rows were written
    result = db.table("target").select().collect()
    assert len(result) == 20000


def test_stream_write_csv(db, large_csv_file):
    """Test streaming write to CSV file."""
    df = db.read.stream().option("chunk_size", 5000).csv(large_csv_file)

    with tempfile.NamedTemporaryFile(mode="r", suffix=".csv", delete=False) as f:
        output_path = f.name

    try:
        df.write.stream().csv(output_path)

        # Verify output file
        with open(output_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 25000
            assert rows[0]["id"] == "0"
            assert rows[-1]["id"] == "24999"
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_stream_write_jsonl(db, large_jsonl_file):
    """Test streaming write to JSONL file."""
    df = db.read.stream().option("chunk_size", 5000).jsonl(large_jsonl_file)

    with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
        output_path = f.name

    try:
        df.write.stream().jsonl(output_path)

        # Verify output file
        with open(output_path, "r") as f:
            lines = f.readlines()
            assert len(lines) == 25000
            first = json.loads(lines[0])
            assert first["id"] == 0
            last = json.loads(lines[-1])
            assert last["id"] == 24999
    finally:
        Path(output_path).unlink(missing_ok=True)


def test_stream_sql_query(db):
    """Test streaming SQL query execution."""
    # Create table with data
    db.create_table(
        "test_table",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="value", type_name="TEXT"),
        ],
    )

    # Insert 15000 rows
    rows = [{"id": i, "value": f"value_{i}"} for i in range(15000)]
    db.table("test_table").insert(rows)

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
    )

    # Create source table with data
    db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    )

    rows = [{"id": i, "name": f"item_{i}"} for i in range(10000)]
    db.table("source").insert(rows)

    # Stream insert
    df = db.read.stream().option("chunk_size", 2000).table("source")
    df.write.stream().option("batch_size", 2000).insertInto("target")

    # Verify
    result = db.table("target").select().collect()
    assert len(result) == 10000


def test_non_streaming_backward_compat(db, large_csv_file):
    """Test that non-streaming mode still works (backward compatibility)."""
    # Without .stream(), should work as before
    df = db.read.csv(large_csv_file)
    rows = df.collect()

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

    df = db.read.stream().schema(schema).option("chunk_size", 5000).csv(large_csv_file)

    chunk_iter = df.collect(stream=True)
    first_chunk = next(chunk_iter)

    # Verify types are correct
    assert isinstance(first_chunk[0]["id"], int)
    assert isinstance(first_chunk[0]["name"], str)
    assert isinstance(first_chunk[0]["value"], float)
