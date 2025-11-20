"""Tests for async query execution."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect


@pytest.mark.asyncio
async def test_async_fetch(tmp_path):
    """Test async query execution."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create table
    from moltres.table.schema import column

    table = await db.create_table(
        "users",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    )

    # Insert data
    await table.insert([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Query data
    df = await db.read.table("users")
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

    table = await db.create_table(
        "items",
        [
            column("id", "INTEGER"),
            column("value", "INTEGER"),
        ],
    )

    # Insert many rows
    rows = [{"id": i, "value": i * 2} for i in range(100)]
    count = await table.insert(rows)

    assert count == 100

    # Verify
    df = await db.read.table("items")
    results = await df.collect()
    assert len(results) == 100

    await db.close()


@pytest.mark.asyncio
async def test_async_streaming(tmp_path):
    """Test async streaming query execution."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    table = await db.create_table(
        "numbers",
        [column("n", "INTEGER")],
    )

    # Insert many rows
    rows = [{"n": i} for i in range(1000)]
    await table.insert(rows)

    # Stream query results
    df = await db.read.table("numbers")
    chunk_count = 0
    total_rows = 0
    async for chunk in await df.collect(stream=True):
        chunk_count += 1
        total_rows += len(chunk)

    assert chunk_count > 0
    assert total_rows == 1000

    await db.close()
