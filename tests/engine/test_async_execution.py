"""Tests for async query execution."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect
from moltres.io.records import AsyncRecords


@pytest.mark.asyncio
async def test_async_fetch(tmp_path):
    """Test async query execution."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create table
    from moltres.table.schema import column

    await db.create_table(
        "users",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()

    # Insert data
    records = AsyncRecords(
        _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db
    )
    await records.insert_into("users")

    # Query data
    # Use db.table().select() for SQL operations
    table_handle = await db.table("users")
    df = table_handle.select()
    results = await df.collect()

    assert len(results) == 2
    assert results[0]["name"] == "Alice"
    assert results[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_batch_insert(tmp_path):
    """Test async batch insert operations."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "items",
        [
            column("id", "INTEGER"),
            column("value", "INTEGER"),
        ],
    ).collect()

    # Insert many rows
    rows = [{"id": i, "value": i * 2} for i in range(100)]
    records = AsyncRecords(_data=rows, _database=db)
    count = await records.insert_into("items")

    assert count == 100

    # Verify
    # Use db.table().select() for SQL operations
    table_handle = await db.table("items")
    df = table_handle.select()
    results = await df.collect()
    assert len(results) == 100

    await db.close()


@pytest.mark.asyncio
async def test_async_streaming(tmp_path):
    """Test async streaming query execution."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "numbers",
        [column("n", "INTEGER")],
    ).collect()

    # Insert many rows
    rows = [{"n": i} for i in range(1000)]
    records = AsyncRecords(_data=rows, _database=db)
    await records.insert_into("numbers")

    # Stream query results
    # Use db.table().select() for SQL operations
    table_handle = await db.table("numbers")
    df = table_handle.select()
    chunk_count = 0
    total_rows = 0
    async for chunk in await df.collect(stream=True):
        chunk_count += 1
        total_rows += len(chunk)

    assert chunk_count > 0
    assert total_rows == 1000

    await db.close()
