"""Tests for streaming insert atomicity and generator handling."""

from __future__ import annotations

from moltres import connect
from moltres.io.records import Records
from moltres.table.schema import column


def test_db_insert_with_generator_inserts_all_chunks() -> None:
    db = connect("sqlite:///:memory:")
    db.create_table("target", [column("id", "INTEGER"), column("value", "TEXT")]).collect()

    chunks = [
        [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}],
        [{"id": 3, "value": "c"}],
    ]

    def gen():
        yield from chunks

    records = Records(_generator=gen, _database=db)
    count = db.insert("target", records)
    assert count == 3
    rows = db.table("target").select().order_by("id").collect()
    assert len(rows) == 3
    assert rows[-1]["id"] == 3
    db.close()
