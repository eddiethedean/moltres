"""Tests for table validation and error handling."""

from __future__ import annotations

import pytest

from moltres import col, connect
from moltres.utils.exceptions import ValidationError


def test_table_name_validation(tmp_path):
    """Test that invalid table names are rejected."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Empty table name
    with pytest.raises(ValidationError, match="cannot be empty"):
        db.table("")

    # SQL injection attempt
    with pytest.raises(ValidationError, match="invalid characters"):
        db.table("users; DROP TABLE users;")

    # Valid table names should work
    table = db.table("valid_table")
    assert table.name == "valid_table"


def test_create_table_empty_columns(tmp_path):
    """Test that creating a table with no columns raises ValidationError."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import ColumnDef

    with pytest.raises(ValidationError, match="no columns"):
        db.create_table("empty_table", [])


def test_insert_empty_rows(tmp_path):
    """Test that inserting empty rows returns 0."""
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


def test_insert_missing_columns(tmp_path):
    """Test that inserting rows with inconsistent schemas raises ValidationError."""
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

    # Note: SQLite allows inserting rows with fewer columns (missing columns get NULL)
    # So inserting {"id": 1} is actually valid - it will insert id=1, name=NULL
    # The validation only checks that all rows in a batch have the same structure

    # Inconsistent rows in same batch should fail
    with pytest.raises(ValidationError, match="does not match expected columns"):
        table.insert(
            [
                {"id": 1, "name": "Alice"},
                {"id": 2},  # Missing 'name' - inconsistent with first row
            ]
        )

    # Valid insert - all rows have same structure
    result = table.insert([{"id": 1, "name": "Alice"}])
    assert result == 1

    # Valid insert with fewer columns (SQLite allows this)
    result = table.insert([{"id": 2}])
    assert result == 1


def test_update_empty_set(tmp_path):
    """Test that updating with empty set raises ValidationError."""
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
    table.insert([{"id": 1, "name": "Alice"}])

    with pytest.raises(ValidationError, match="at least one value"):
        table.update(where=col("id") == 1, set={})
