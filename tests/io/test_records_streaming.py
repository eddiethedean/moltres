"""Streaming insert tests for Records and AsyncRecords."""

from __future__ import annotations

import pytest

from moltres import connect
from moltres.io.records import AsyncRecords, Records


def test_records_insert_into_streams_chunks(monkeypatch):
    """Records.insert_into should process generator chunks without materializing."""
    db = connect("sqlite:///:memory:")

    inserted_chunks: list[int] = []

    def fake_insert_rows(handle, rows, transaction=None):
        inserted_chunks.append(len(rows))
        return len(rows)

    monkeypatch.setattr("moltres.table.mutations.insert_rows", fake_insert_rows)

    chunks = [
        [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        [{"id": 3, "name": "c"}],
    ]

    def chunk_generator():
        for chunk in chunks:
            yield chunk

    records = Records(_generator=chunk_generator, _database=db)
    inserted = records.insert_into("items")

    assert inserted == 3
    assert inserted_chunks == [2, 1]


@pytest.mark.asyncio
async def test_async_records_insert_into_streams_chunks(monkeypatch):
    """AsyncRecords.insert_into should process async generator chunks lazily."""

    class DummyAsyncDB:
        def __init__(self):
            self.connection_manager = type("cm", (), {"active_transaction": None})()

        async def table(self, name):
            return name  # Table handle placeholder

    inserted_chunks: list[int] = []

    async def fake_insert_rows_async(handle, rows, transaction=None):
        inserted_chunks.append(len(rows))
        return len(rows)

    monkeypatch.setattr("moltres.table.async_mutations.insert_rows_async", fake_insert_rows_async)

    async def chunk_generator():
        yield [{"id": 1}]
        yield [{"id": 2}, {"id": 3}]

    records = AsyncRecords(
        _generator=lambda: chunk_generator(),
        _database=DummyAsyncDB(),
    )
    inserted = await records.insert_into("items")

    assert inserted == 3
    assert inserted_chunks == [1, 2]
