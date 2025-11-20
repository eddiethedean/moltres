"""Tests for batch insert functionality."""

from __future__ import annotations

from moltres import connect
from moltres.table.schema import column


def test_batch_insert_performance(tmp_path):
    """Test that batch inserts work correctly."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("value", "REAL"),
        ],
    )

    table = db.table("test")

    # Insert multiple rows in one call
    rows = [
        {"id": i, "name": f"Item{i}", "value": float(i * 10)}
        for i in range(1, 101)  # 100 rows
    ]

    result = table.insert(rows)
    assert result == 100

    # Verify all rows were inserted
    df = table.select()
    all_rows = df.collect()
    assert len(all_rows) == 100
    assert all_rows[0]["id"] == 1
    assert all_rows[-1]["id"] == 100


def test_batch_insert_empty_list(tmp_path):
    """Test that inserting empty list returns 0."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    )

    table = db.table("test")
    result = table.insert([])
    assert result == 0


def test_batch_insert_large_dataset(tmp_path):
    """Test batch insert with a larger dataset."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("data", "TEXT"),
        ],
    )

    table = db.table("test")

    # Insert 1000 rows
    rows = [{"id": i, "data": f"data_{i}"} for i in range(1000)]
    result = table.insert(rows)
    assert result == 1000

    # Verify
    df = table.select()
    count = len(df.collect())
    assert count == 1000
