"""End-to-end regression tests for null-aware SQL comparisons."""

from __future__ import annotations

from moltres import col, connect, lit
from moltres.io.records import Records
from moltres.table.schema import column


def test_null_equality_filters_null_rows(tmp_path):
    """col('x') == None must compile to IS NULL and return null rows only."""
    db_path = tmp_path / "null_eq.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("email", "TEXT")],
    ).collect()
    Records.from_list(
        [
            {"id": 1, "email": "alice@example.com"},
            {"id": 2, "email": None},
            {"id": 3, "email": "bob@example.com"},
        ],
        database=db,
    ).insert_into("users")

    rows = db.table("users").select().where(col("email") == lit(None)).collect()
    assert rows == [{"id": 2, "email": None}]
    db.close()


def test_null_inequality_excludes_null_rows(tmp_path):
    """col('x') != None must compile to IS NOT NULL."""
    db_path = tmp_path / "null_ne.db"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("email", "TEXT")],
    ).collect()
    Records.from_list(
        [
            {"id": 1, "email": "alice@example.com"},
            {"id": 2, "email": None},
        ],
        database=db,
    ).insert_into("users")

    rows = db.table("users").select().where(col("email") != lit(None)).order_by(col("id")).collect()
    assert rows == [{"id": 1, "email": "alice@example.com"}]
    db.close()
