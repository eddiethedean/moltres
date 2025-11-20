"""Tests for edge cases and error conditions."""

from __future__ import annotations

import pytest

from moltres import col, connect
from moltres.utils.exceptions import ValidationError


def test_limit_zero(tmp_path):
    """Test that limit(0) returns empty result set."""
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
    table.insert(
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    )

    df = table.select().limit(0)
    results = df.collect()
    assert results == []


def test_limit_negative(tmp_path):
    """Test that negative limit raises ValueError."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
        ],
    )

    table = db.table("test")
    df = table.select()

    with pytest.raises(ValueError, match="must be non-negative"):
        df.limit(-1)


def test_empty_table_query(tmp_path):
    """Test querying an empty table."""
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
    df = table.select()
    results = df.collect()
    assert results == []


def test_null_values(tmp_path):
    """Test handling of NULL values."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("name", "TEXT", nullable=True),
            column("value", "REAL", nullable=True),
        ],
    )

    table = db.table("test")
    table.insert(
        [
            {"id": 1, "name": "Alice", "value": 10.5},
            {"id": 2, "name": None, "value": None},
            {"id": 3, "name": "Bob", "value": None},
        ]
    )

    df = table.select()
    results = df.collect()
    assert len(results) == 3
    assert results[0]["name"] == "Alice"
    assert results[1]["name"] is None
    assert results[2]["value"] is None


def test_collect_without_database():
    """Test that collecting a DataFrame without a database raises RuntimeError."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import TableScan

    plan = TableScan(table="test")
    df = DataFrame(plan=plan, database=None)

    with pytest.raises(RuntimeError, match="Cannot collect a plan without an attached Database"):
        df.collect()


def test_group_by_empty(tmp_path):
    """Test that group_by with no columns raises ValueError."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
        ],
    )

    table = db.table("test")
    df = table.select()

    with pytest.raises(ValueError, match="requires at least one grouping column"):
        df.group_by()


def test_join_different_databases(tmp_path):
    """Test that joining DataFrames from different databases raises ValueError."""
    db1_path = tmp_path / "db1.sqlite"
    db2_path = tmp_path / "db2.sqlite"

    db1 = connect(f"sqlite:///{db1_path}")
    db2 = connect(f"sqlite:///{db2_path}")

    from moltres.table.schema import column

    db1.create_table("table1", [column("id", "INTEGER")])
    db2.create_table("table2", [column("id", "INTEGER")])

    df1 = db1.table("table1").select()
    df2 = db2.table("table2").select()

    with pytest.raises(ValueError, match="different Database instances"):
        df1.join(df2, on=[("id", "id")])


def test_join_without_database():
    """Test that joining DataFrames without databases raises RuntimeError."""
    from moltres.dataframe.dataframe import DataFrame
    from moltres.logical.plan import TableScan

    plan1 = TableScan(table="table1")
    plan2 = TableScan(table="table2")

    df1 = DataFrame(plan=plan1, database=None)
    df2 = DataFrame(plan=plan2, database=None)

    with pytest.raises(RuntimeError, match="must be bound to a Database"):
        df1.join(df2, on=[("id", "id")])
