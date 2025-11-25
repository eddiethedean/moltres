"""Tests for Records UX improvements (from_list, from_dicts)."""

from __future__ import annotations


from moltres import connect
from moltres.io.records import Records


def test_records_from_list(tmp_path):
    """Test Records.from_list() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    from moltres.table.schema import column

    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    # Create Records using from_list
    records = Records.from_list([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], database=db)

    # Insert into table
    inserted = records.insert_into("users")
    assert inserted == 2

    # Verify data
    rows = db.table("users").select().collect()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_records_from_dicts(tmp_path):
    """Test Records.from_dicts() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    from moltres.table.schema import column

    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    # Create Records using from_dicts
    records = Records.from_dicts(
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        database=db,
    )

    # Insert into table
    inserted = records.insert_into("users")
    assert inserted == 2

    # Verify data
    rows = db.table("users").select().collect()
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"Alice", "Bob"}


def test_records_from_list_empty():
    """Test Records.from_list() with empty list."""
    records = Records.from_list([])
    assert len(list(records)) == 0


def test_records_from_dicts_single():
    """Test Records.from_dicts() with single dictionary."""
    records = Records.from_dicts({"id": 1, "name": "Alice"})
    assert len(list(records)) == 1
    assert list(records)[0] == {"id": 1, "name": "Alice"}
