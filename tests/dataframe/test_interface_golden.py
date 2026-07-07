"""Golden tests comparing Moltres interface collect() output to reference libraries."""

from __future__ import annotations

import pytest

from moltres import connect, col
from moltres.io.records import Records
from moltres.table.schema import column


@pytest.fixture
def orders_db():
    db = connect("sqlite:///:memory:")
    with db.batch():
        db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("user_id", "INTEGER"),
                column("amount", "REAL"),
                column("status", "TEXT"),
            ],
        ).collect()
    Records.from_list(
        [
            {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
            {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
            {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
        ],
        database=db,
    ).insert_into("orders")
    yield db
    db.close()


def _records_to_dicts(rows):
    try:
        import pandas as pd

        if isinstance(rows, pd.DataFrame):
            return rows.to_dict("records")
    except ImportError:
        pass
    try:
        import polars as pl

        if isinstance(rows, pl.DataFrame):
            return rows.to_dicts()
    except ImportError:
        pass
    return rows


def test_pandas_groupby_matches_reference(orders_db):
    pandas = pytest.importorskip("pandas")

    df = orders_db.table("orders").pandas()
    moltres_rows = _records_to_dicts(df.groupby("status").sum().collect())

    ref = (
        pandas.DataFrame(
            [
                {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
                {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
                {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
            ]
        )
        .groupby("status")
        .sum()
        .reset_index()
    )
    ref_by_status = {row["status"]: row["amount"] for row in ref.to_dict("records")}
    moltres_by_status = {row["status"]: row["amount_sum"] for row in moltres_rows}
    assert moltres_by_status == ref_by_status


def test_polars_filter_matches_reference(orders_db):
    polars = pytest.importorskip("polars")

    df = orders_db.table("orders").polars()
    moltres_rows = _records_to_dicts(df.filter(col("amount") > 100).collect())

    ref = polars.DataFrame(
        [
            {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
            {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
        ]
    ).filter(polars.col("amount") > 100)
    assert moltres_rows == ref.to_dicts()


def test_pandas_query_matches_reference(orders_db):
    pandas = pytest.importorskip("pandas")

    df = orders_db.table("orders").pandas()
    moltres_rows = _records_to_dicts(df.query("status == 'active'").collect())

    ref = pandas.DataFrame(
        [
            {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
            {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
            {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
        ]
    ).query("status == 'active'")
    assert moltres_rows == ref.to_dict("records")


def test_polars_groupby_matches_reference(orders_db):
    polars = pytest.importorskip("polars")
    from moltres.expressions import functions as F

    df = orders_db.table("orders").polars()
    moltres_rows = _records_to_dicts(
        df.group_by("status").agg(F.sum(col("amount")).alias("amount")).collect()
    )

    ref = (
        polars.DataFrame(
            [
                {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
                {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
                {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
            ]
        )
        .group_by("status")
        .agg(polars.col("amount").sum().alias("amount"))
    )
    moltres_by_status = {row["status"]: row["amount"] for row in moltres_rows}
    ref_by_status = {row["status"]: row["amount"] for row in ref.to_dicts()}
    assert moltres_by_status == ref_by_status
