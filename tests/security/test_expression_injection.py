"""Regression tests for expression-level SQL injection fixes."""

from __future__ import annotations

import pytest

from moltres import col, connect
from moltres.expressions import functions as F
from moltres.utils.exceptions import CompilationError, ValidationError


def test_date_add_rejects_malicious_interval() -> None:
    db = connect("sqlite:///:memory:")
    df = db.table("t").select()
    malicious = "1' DAY); DELETE FROM users; --"
    with pytest.raises(CompilationError):
        df.select(F.date_add(col("created_at"), malicious).alias("x")).collect()
    db.close()


def test_join_on_malicious_column_raises() -> None:
    db = connect("sqlite:///:memory:")
    from moltres.table.schema import column

    db.create_table("a", [column("id", "INTEGER")]).collect()
    db.create_table("b", [column("id", "INTEGER")]).collect()
    left = db.table("a").select()
    right = db.table("b").select()
    with pytest.raises(ValidationError):
        left.join(right, on=[('id" OR 1=1 --', "id")]).collect()
    db.close()


def test_writer_table_exists_rejects_malicious_name() -> None:
    db = connect("sqlite:///:memory:")
    from moltres.dataframe.core.dataframe import DataFrame
    from moltres.logical.plan import TableScan

    df = DataFrame(plan=TableScan(table="x"), database=db)
    writer = df.write
    assert writer._table_exists(db, 'users" WHERE 1=1; --') is False
    db.close()
