"""Tests for async table operations."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col


@pytest.mark.asyncio
async def test_async_table_operations(tmp_path):
    """Test async table creation and mutations."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    # Create table
    table = await db.create_table(
        "products",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("price", "REAL"),
        ],
    )

    # Insert rows
    await table.insert(
        [
            {"id": 1, "name": "Widget", "price": 10.0},
            {"id": 2, "name": "Gadget", "price": 20.0},
        ]
    )

    # Update rows
    updated = await table.update(
        where=col("id") == 1,
        set={"price": 15.0},
    )
    assert updated == 1

    # Query
    df = await db.read.table("products")
    results = await df.collect()
    assert len(results) == 2
    assert results[0]["price"] == 15.0

    # Delete rows
    deleted = await table.delete(where=col("id") == 2)
    assert deleted == 1

    # Verify deletion
    results = await df.collect()
    assert len(results) == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_drop_table(tmp_path):
    """Test async table dropping."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    # Create and drop table
    await db.create_table("temp", [column("x", "INTEGER")])
    await db.drop_table("temp", if_exists=True)

    # Should not raise error
    await db.drop_table("temp", if_exists=True)

    await db.close()
