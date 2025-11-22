"""Tests for async DataFrame operations."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col


@pytest.mark.asyncio
async def test_async_dataframe_collect(tmp_path):
    """Test async DataFrame collect operation."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
            pk="id",
        )
    ).write.insertInto("users")

    # Query with filters - use db.table().select() for SQL operations
    table_handle = await db.table("users")
    df = table_handle.select()
    filtered = df.where(col("age") > 25)
    results = await filtered.collect()

    assert len(results) == 2
    assert all(r["age"] > 25 for r in results)

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_select(tmp_path):
    """Test async DataFrame column selection."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "data",
        [
            column("id", "INTEGER", primary_key=True),
            column("a", "INTEGER"),
            column("b", "INTEGER"),
            column("c", "INTEGER"),
        ],
    ).collect()

    await (
        await db.createDataFrame(
            [{"a": 1, "b": 2, "c": 3}],
            auto_pk="id",
        )
    ).write.insertInto("data")
    # Use db.table().select() for SQL operations
    table_handle = await db.table("data")
    df = table_handle.select()
    selected = df.select("a", "c")
    results = await selected.collect()

    assert len(results) == 1
    assert "a" in results[0]
    assert "c" in results[0]
    assert "b" not in results[0]

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_limit(tmp_path):
    """Test async DataFrame limit operation."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table("numbers", [column("n", "INTEGER", primary_key=True)]).collect()

    rows = [{"n": i} for i in range(10)]
    await (await db.createDataFrame(rows, pk="n")).write.insertInto("numbers")

    # Use db.table().select() for SQL operations
    table_handle = await db.table("numbers")
    df = table_handle.select()
    limited = df.limit(5)
    results = await limited.collect()

    assert len(results) == 5

    await db.close()
