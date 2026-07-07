"""Async schema reflection smoke tests.

Full reflection behavior is covered in tests/table/test_reflection.py for sync.
This module keeps representative async coverage only.
"""

from __future__ import annotations

import pytest

try:
    from moltres import async_connect
except ImportError:
    pytest.skip("Async dependencies not installed", allow_module_level=True)

from moltres.table.schema import column


@pytest.mark.asyncio
async def test_async_reflect_table_names_and_columns(tmp_path):
    """Async reflection returns expected table and column metadata."""
    db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'reflect.db'}")
    try:
        await db.create_table(
            "users",
            [
                column("id", "INTEGER", nullable=False, primary_key=True),
                column("name", "TEXT", nullable=False),
            ],
        ).collect()

        tables = await db.get_table_names()
        assert tables == ["users"]

        columns = await db.get_columns("users")
        assert [c.name for c in columns] == ["id", "name"]
        id_col = next(c for c in columns if c.name == "id")
        assert id_col.primary_key is True

        schemas = await db.reflect()
        assert set(schemas) == {"users"}
        assert len(schemas["users"].columns) == 2
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_async_reflect_empty_database(tmp_path):
    """Empty async database reflects to an empty schema map."""
    db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'empty.db'}")
    try:
        assert await db.get_table_names() == []
        assert await db.reflect() == {}
    finally:
        await db.close()
