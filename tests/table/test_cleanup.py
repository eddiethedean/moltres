import sqlite3

import pytest

from moltres import async_connect, connect
from moltres.table import async_table as async_table_mod
from moltres.table import table as table_mod


def _table_exists(db_path, table_name: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE name = ?", (table_name,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def test_ephemeral_tables_cleaned_on_gc(tmp_path):
    db_path = tmp_path / "sync_cleanup.sqlite"
    db = connect(f"sqlite:///{db_path}")
    df = db.createDataFrame([{"id": 1, "value": "v"}], pk="id")
    ephemeral = next(iter(db._ephemeral_tables))
    # Table exists before cleanup
    assert _table_exists(db_path, ephemeral)

    # Drop references and trigger global cleanup (simulates crash/exit)
    # Note: We need to call cleanup before deleting db, as deletion removes it from weak set
    del df
    table_mod._force_database_cleanup_for_tests()
    del db

    assert not _table_exists(db_path, ephemeral)


def test_create_dataframe_without_primary_key(tmp_path):
    """createDataFrame should succeed even when no pk/auto_pk is provided."""
    db_path = tmp_path / "pkless.sqlite"
    db = connect(f"sqlite:///{db_path}")

    df = db.createDataFrame([{"name": "Alice"}, {"name": "Bob"}])
    rows = df.collect()
    assert [row["name"] for row in rows] == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_async_ephemeral_tables_cleaned_on_gc(tmp_path):
    db_path = tmp_path / "async_cleanup.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    df = await db.createDataFrame([{"id": 1, "value": "v"}], pk="id")
    ephemeral = next(iter(db._ephemeral_tables))
    assert _table_exists(db_path, ephemeral)

    # Remove references and trigger async cleanup hook
    # Note: We need to call cleanup before deleting db, as deletion removes it from weak set
    del df
    async_table_mod._force_async_database_cleanup_for_tests()
    del db

    assert not _table_exists(db_path, ephemeral)


@pytest.mark.asyncio
async def test_async_create_dataframe_without_primary_key(tmp_path):
    """Async createDataFrame should also allow PK-less staging tables."""
    db_path = tmp_path / "async_pkless.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    df = await db.createDataFrame([{"name": "Alice"}])
    rows = await df.collect()
    assert rows[0]["name"] == "Alice"
