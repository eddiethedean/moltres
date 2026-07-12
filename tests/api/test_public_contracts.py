"""Behavioral public-API contracts (independent of implementation details).

These tests encode docs/PUBLIC_API.md and PySpark migration footguns so that
shared incorrect assumptions in production code + older tests would fail here.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from moltres import async_connect, col, column, connect
from moltres.dataframe import DataFrame
from moltres.expressions import functions as F
from moltres.io.records import Records


def test_union_is_distinct_union_all_keeps_duplicates(tmp_path: Path) -> None:
    """PUBLIC_API: union() = DISTINCT; unionAll() = ALL (PySpark migrants use unionAll)."""
    db = connect(f"sqlite:///{tmp_path / 'union_contract.db'}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE t1 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql("INSERT INTO t1 (id, value) VALUES (1, 'A'), (2, 'B')")
        conn.exec_driver_sql("CREATE TABLE t2 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql("INSERT INTO t2 (id, value) VALUES (2, 'B'), (3, 'C')")

    df1 = db.table("t1").select("value")
    df2 = db.table("t2").select("value")

    distinct_rows = df1.union(df2).order_by(col("value")).collect()
    assert [r["value"] for r in distinct_rows] == ["A", "B", "C"]

    all_rows = df1.unionAll(df2).order_by(col("value")).collect()
    assert [r["value"] for r in all_rows] == ["A", "B", "B", "C"]

    # union_all alias must match unionAll semantics
    alias_rows = df1.union_all(df2).order_by(col("value")).collect()
    assert [r["value"] for r in alias_rows] == ["A", "B", "B", "C"]

    db.close()


@pytest.mark.asyncio
async def test_async_union_distinct_vs_union_all(tmp_path: Path) -> None:
    """Async parity for overlapping duplicates — must not mask with .distinct()."""
    pytest.importorskip("aiosqlite")
    db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'async_union.db'}")

    await db.create_table(
        "t1",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()
    await db.create_table(
        "t2",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    await (
        await db.createDataFrame(
            [{"id": 1, "value": "A"}, {"id": 2, "value": "B"}],
            pk="id",
        )
    ).write.insertInto("t1")
    await (
        await db.createDataFrame(
            [{"id": 3, "value": "B"}, {"id": 4, "value": "C"}],
            pk="id",
        )
    ).write.insertInto("t2")

    df1 = (await db.table("t1")).select("value")
    df2 = (await db.table("t2")).select("value")

    distinct_rows = await df1.union(df2).order_by(col("value")).collect()
    assert [r["value"] for r in distinct_rows] == ["A", "B", "C"]

    all_rows = await df1.unionAll(df2).order_by(col("value")).collect()
    assert [r["value"] for r in all_rows] == ["A", "B", "B", "C"]

    await db.close()


def test_col_for_queries_column_for_ddl(tmp_path: Path) -> None:
    """col() builds query expressions; column() defines DDL schemas."""
    db = connect(f"sqlite:///{tmp_path / 'col_column.db'}")

    db.create_table(
        "people",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    inserted = Records.from_list(
        [{"id": 1, "name": "Ada", "age": 36}, {"id": 2, "name": "Bob", "age": 20}],
        database=db,
    ).insert_into("people")
    assert inserted == 2

    rows = db.table("people").select().where(col("age") > 30).collect()
    assert rows == [{"id": 1, "name": "Ada", "age": 36}]

    updated = db.update("people", where=col("name") == "Bob", set={"age": 21})
    assert updated == 1
    assert db.table("people").select().where(col("id") == 2).collect() == [
        {"id": 2, "name": "Bob", "age": 21}
    ]

    deleted = db.delete("people", where=col("age") < 30)
    assert deleted == 1
    assert db.table("people").select().collect() == [{"id": 1, "name": "Ada", "age": 36}]

    db.close()


def test_crud_return_types_and_write_insert_into_none(tmp_path: Path) -> None:
    """insert/update/delete/merge return int; df.write.insertInto returns None."""
    db = connect(f"sqlite:///{tmp_path / 'crud_types.db'}")

    db.create_table(
        "items",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("qty", "INTEGER"),
        ],
    ).collect()
    db.create_table(
        "items_copy",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("qty", "INTEGER"),
        ],
    ).collect()

    n = db.insert("items", [{"id": 1, "name": "widget", "qty": 5}])
    assert isinstance(n, int) and n == 1

    n = db.update("items", where=col("id") == 1, set={"qty": 10})
    assert isinstance(n, int) and n == 1

    n = db.merge(
        "items",
        [{"id": 1, "name": "widget", "qty": 12}, {"id": 2, "name": "gadget", "qty": 3}],
        on=["id"],
        when_matched={"qty": 12},
    )
    assert isinstance(n, int) and n >= 1

    rows = {(r["id"], r["name"], r["qty"]) for r in db.table("items").select().collect()}
    assert (1, "widget", 12) in rows
    assert (2, "gadget", 3) in rows

    write_result = db.table("items").select().write.insertInto("items_copy")
    assert write_result is None
    copy_rows = db.table("items_copy").select().order_by(col("id")).collect()
    assert len(copy_rows) == 2
    assert copy_rows[0]["name"] == "widget"

    n = db.delete("items", where=col("id") == 2)
    assert isinstance(n, int) and n == 1

    db.close()


def test_records_from_list_insert_into_read_back(tmp_path: Path) -> None:
    """Preferred Records path: from_list(..., database=db).insert_into → count + content."""
    db = connect(f"sqlite:///{tmp_path / 'records.db'}")
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("country", "TEXT"),
            column("amount", "REAL"),
        ],
    ).collect()

    n = Records.from_list(
        [
            {"id": 1, "country": "US", "amount": 100.0},
            {"id": 2, "country": "UK", "amount": 200.0},
        ],
        database=db,
    ).insert_into("orders")
    assert n == 2

    rows = db.table("orders").select().order_by(col("id")).collect()
    assert rows == [
        {"id": 1, "country": "US", "amount": 100.0},
        {"id": 2, "country": "UK", "amount": 200.0},
    ]
    db.close()


def test_io_routing_load_vs_read_records_vs_deprecated_read(tmp_path: Path) -> None:
    """load.* → DataFrame; read.records.* → Records; read.csv warns and still works."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("id,name\n1,Alice\n2,Bob\n", encoding="utf-8")
    db = connect(f"sqlite:///{tmp_path / 'io.db'}")

    loaded = db.load.csv(str(csv_path))
    assert isinstance(loaded, DataFrame)
    loaded_rows = loaded.order_by(col("id")).collect()
    assert len(loaded_rows) == 2
    assert loaded_rows[0]["name"] == "Alice"
    assert loaded_rows[1]["name"] == "Bob"
    assert str(loaded_rows[0]["id"]) == "1"
    assert str(loaded_rows[1]["id"]) == "2"

    records = db.read.records.csv(str(csv_path))
    # LazyRecords or Records — materialize if needed
    if hasattr(records, "collect") and not hasattr(records, "rows"):
        material = records.collect()
        row_list = material.rows() if hasattr(material, "rows") else list(material)
    else:
        row_list = records.rows()
    assert len(row_list) == 2
    assert row_list[0]["name"] == "Alice"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        deprecated_df = db.read.csv(str(csv_path))
    assert any(
        issubclass(w.category, DeprecationWarning) and "db.load" in str(w.message) for w in caught
    )
    assert isinstance(deprecated_df, DataFrame)
    assert len(deprecated_df.collect()) == 2

    db.close()


def test_readme_quick_start_pipeline(tmp_path: Path) -> None:
    """README quick-start: create → insert → where → group_by → agg → collect."""
    db = connect(f"sqlite:///{tmp_path / 'quickstart.db'}")

    db.create_table(
        "orders",
        [
            column("id", "INTEGER"),
            column("country", "TEXT"),
            column("amount", "REAL"),
        ],
    ).collect()

    Records.from_list(
        [
            {"id": 1, "country": "US", "amount": 100.0},
            {"id": 2, "country": "UK", "amount": 200.0},
            {"id": 3, "country": "US", "amount": 50.0},
        ],
        database=db,
    ).insert_into("orders")

    df = (
        db.table("orders")
        .select()
        .where(col("country") == "US")
        .group_by("country")
        .agg(F.sum(col("amount")).alias("total_amount"))
    )
    result = df.collect()
    assert len(result) == 1
    assert result[0]["country"] == "US"
    assert float(result[0]["total_amount"]) == 150.0

    db.update("orders", where=col("country") == "US", set={"amount": 150.0})
    us_rows = db.table("orders").select().where(col("country") == "US").collect()
    assert all(float(r["amount"]) == 150.0 for r in us_rows)

    db.delete("orders", where=col("amount") < 100)
    remaining = db.table("orders").select().order_by(col("id")).collect()
    assert all(float(r["amount"]) >= 100 for r in remaining)

    db.close()
