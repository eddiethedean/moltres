"""Async end-to-end workflow tests."""

from __future__ import annotations

import pytest

try:
    from moltres import async_connect
except ImportError:
    pytest.skip("Async dependencies not installed", allow_module_level=True)

from moltres.expressions import col
from moltres.expressions.functions import sum as sum_
from moltres.io.records import AsyncRecords
from moltres.table.schema import column


@pytest.mark.asyncio
@pytest.mark.integration
async def test_async_etl_pipeline(tmp_path):
    """Insert, transform, and aggregate via async API on a shared SQLite file."""
    db_path = tmp_path / "async_etl.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    try:
        await db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("status", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        records = AsyncRecords(
            _data=[
                {"id": 1, "status": "active", "amount": 100.0},
                {"id": 2, "status": "active", "amount": 200.0},
                {"id": 3, "status": "completed", "amount": 150.0},
            ],
            _database=db,
        )
        await records.insert_into("orders")

        table = await db.table("orders")
        result = (
            await table.select()
            .where(col("status") == "active")
            .group_by("status")
            .agg(sum_(col("amount")).alias("total"))
            .collect()
        )

        assert len(result) == 1
        assert result[0]["status"] == "active"
        assert result[0]["total"] == 300.0
    finally:
        await db.close()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_sync_write_async_read(tmp_path):
    """Data written synchronously is readable through async connection."""
    from moltres import connect

    db_path = tmp_path / "shared.db"
    sync_db = connect(f"sqlite:///{db_path}")
    sync_db.create_table(
        "items",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()
    from moltres.io.records import Records

    Records.from_list([{"id": 1, "value": "alpha"}], database=sync_db).insert_into("items")
    sync_db.close()

    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    try:
        table = await async_db.table("items")
        rows = await table.select().collect()
        assert rows == [{"id": 1, "value": "alpha"}]
    finally:
        await async_db.close()
