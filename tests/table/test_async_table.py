"""Tests for async table operations."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col
from moltres.io.records import AsyncRecords


@pytest.mark.asyncio
async def test_async_table_operations(tmp_path):
    """Test async table creation and mutations."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    # Create table
    await db.create_table(
        "products",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("price", "REAL"),
        ],
    ).collect()

    # Insert rows using AsyncRecords
    records = AsyncRecords(
        _data=[
            {"id": 1, "name": "Widget", "price": 10.0},
            {"id": 2, "name": "Gadget", "price": 20.0},
        ],
        _database=db,
    )
    await records.insert_into("products")

    # Update rows using DataFrame write API
    table_handle = await db.table("products")
    df = table_handle.select()
    await df.write.update("products", where=col("id") == 1, set={"price": 15.0})

    # Query
    results = await df.collect()
    assert len(results) == 2
    assert results[0]["price"] == 15.0

    # Delete rows using DataFrame write API
    await df.write.delete("products", where=col("id") == 2)

    # Verify deletion
    results = await df.collect()
    assert len(results) == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_database_crud(tmp_path):
    """Test AsyncDatabase insert/update/delete/merge parity with sync Database."""
    db_path = tmp_path / "crud.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "items",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("qty", "INTEGER"),
        ],
    ).collect()

    inserted = await db.insert("items", [{"id": 1, "name": "a", "qty": 1}])
    assert inserted == 1

    updated = await db.update("items", where=col("id") == 1, set={"qty": 2})
    assert updated == 1

    merged = await db.merge(
        "items",
        [{"id": 2, "name": "b", "qty": 3}],
        on=["id"],
        when_matched={"qty": 4},
    )
    assert merged >= 1

    deleted = await db.delete("items", where=col("id") == 2)
    assert deleted == 1

    rows = await (await db.table("items")).select().collect()
    assert len(rows) == 1
    assert rows[0]["qty"] == 2

    await db.close()


@pytest.mark.asyncio
async def test_async_drop_table(tmp_path):
    """Test async table dropping."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    # Create and drop table
    await db.create_table("temp", [column("x", "INTEGER")]).collect()
    await db.drop_table("temp", if_exists=True).collect()

    # Should not raise error
    await db.drop_table("temp", if_exists=True).collect()

    await db.close()
