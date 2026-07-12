"""Behavioral set-operation tests against SQLite (not compiler-string-only)."""

from __future__ import annotations

from moltres import col, connect
from moltres.table.schema import column
from moltres.io.records import Records


def _seed_overlap(db):
    db.create_table(
        "left_t",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()
    db.create_table(
        "right_t",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()
    Records.from_list(
        [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}],
        database=db,
    ).insert_into("left_t")
    Records.from_list(
        [{"id": 3, "value": "B"}, {"id": 4, "value": "C"}],
        database=db,
    ).insert_into("right_t")
    return db.table("left_t").select("value"), db.table("right_t").select("value")


def test_behavioral_union_intersect_except(tmp_path):
    db = connect(f"sqlite:///{tmp_path / 'setops.db'}")
    left, right = _seed_overlap(db)

    union_vals = [r["value"] for r in left.union(right).order_by(col("value")).collect()]
    assert union_vals == ["A", "B", "C"]

    union_all_vals = [r["value"] for r in left.unionAll(right).order_by(col("value")).collect()]
    assert union_all_vals == ["A", "B", "B", "C"]

    intersect_vals = [r["value"] for r in left.intersect(right).order_by(col("value")).collect()]
    assert intersect_vals == ["B"]

    except_vals = [r["value"] for r in left.except_(right).order_by(col("value")).collect()]
    assert except_vals == ["A"]

    db.close()
