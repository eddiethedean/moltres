"""Boundary and missing-coverage behavioral tests for DataFrame contracts."""

from __future__ import annotations

from moltres import col, connect
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.table.schema import column


def test_empty_dataframe_collect_and_groupby(tmp_path):
    db = connect(f"sqlite:///{tmp_path / 'empty.db'}")
    db.create_table(
        "items",
        [column("id", "INTEGER", primary_key=True), column("grp", "TEXT"), column("n", "INTEGER")],
    ).collect()

    rows = db.table("items").select().collect()
    assert rows == []

    grouped = (
        db.table("items").select().group_by("grp").agg(F.sum(col("n")).alias("total")).collect()
    )
    assert grouped == []
    db.close()


def test_groupby_with_null_keys(tmp_path):
    db = connect(f"sqlite:///{tmp_path / 'null_group.db'}")
    db.create_table(
        "items",
        [column("id", "INTEGER", primary_key=True), column("grp", "TEXT"), column("n", "INTEGER")],
    ).collect()
    Records.from_list(
        [
            {"id": 1, "grp": "A", "n": 10},
            {"id": 2, "grp": None, "n": 5},
            {"id": 3, "grp": None, "n": 7},
        ],
        database=db,
    ).insert_into("items")

    rows = db.table("items").select().group_by("grp").agg(F.sum(col("n")).alias("total")).collect()
    by_grp = {r["grp"]: float(r["total"]) for r in rows}
    assert by_grp["A"] == 10.0
    assert by_grp[None] == 12.0
    db.close()


def test_union_requires_same_database(tmp_path):
    db1 = connect(f"sqlite:///{tmp_path / 'a.db'}")
    db2 = connect(f"sqlite:///{tmp_path / 'b.db'}")
    db1.create_table("t", [column("id", "INTEGER", primary_key=True)]).collect()
    db2.create_table("t", [column("id", "INTEGER", primary_key=True)]).collect()

    import pytest

    with pytest.raises(ValueError, match="(?i)different|same"):
        db1.table("t").select().union(db2.table("t").select())

    db1.close()
    db2.close()


def test_show_count_total_flag_does_not_corrupt_results(tmp_path, capsys):
    """show() prints rows; count_total must not change collect() results."""
    db = connect(f"sqlite:///{tmp_path / 'show.db'}")
    db.create_table(
        "t",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    Records.from_list(
        [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        database=db,
    ).insert_into("t")

    df = db.table("t").select().order_by(col("id"))
    before = df.collect()
    df.show(count_total=False)
    df.show(count_total=True)
    after = df.collect()
    assert after == before == [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    captured = capsys.readouterr().out
    assert "a" in captured and "b" in captured
    db.close()
