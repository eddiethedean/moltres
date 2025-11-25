"""Tests for synchronous QueryExecutor behavior."""

from __future__ import annotations

from moltres import connect
from moltres.table.schema import column


def _prepare_db(tmp_path):
    db_path = tmp_path / "executor.sqlite"
    db = connect(f"sqlite:///{db_path}")
    db.create_table(
        "items",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()
    return db


def test_query_executor_fetch_reports_rowcount(tmp_path):
    """fetch() should return an accurate rowcount for SELECT queries."""
    db = _prepare_db(tmp_path)
    db.executor.execute(
        "INSERT INTO items (id, name) VALUES (:id, :name)",
        params={"id": 1, "name": "alpha"},
    )
    db.executor.execute(
        "INSERT INTO items (id, name) VALUES (:id, :name)",
        params={"id": 2, "name": "beta"},
    )

    result = db.executor.fetch("SELECT * FROM items")
    assert len(result.rows) == 2
    assert result.rowcount == 2


def test_query_executor_execute_many_batches_params(tmp_path):
    """execute_many() should batch parameter sets in a single call."""
    db = _prepare_db(tmp_path)

    params_list = [
        {"id": 1, "name": "alpha"},
        {"id": 2, "name": "beta"},
        {"id": 3, "name": "gamma"},
    ]
    result = db.executor.execute_many(
        "INSERT INTO items (id, name) VALUES (:id, :name)", params_list
    )

    assert result.rowcount == len(params_list)
    rows = db.executor.fetch("SELECT * FROM items")
    assert rows.rowcount == len(params_list)
