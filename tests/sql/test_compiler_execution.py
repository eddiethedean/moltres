"""Execution tests for compiled logical plans."""

from __future__ import annotations

from moltres import connect
from moltres.io.records import Records
from moltres.table.schema import column


def test_anti_join_execution(tmp_path) -> None:
    """Anti-join returns left rows with no match on the right."""
    db_path = tmp_path / "anti.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "customers",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    db.create_table(
        "orders",
        [column("id", "INTEGER", primary_key=True), column("customer_id", "INTEGER")],
    ).collect()

    Records.from_list(
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ],
        database=db,
    ).insert_into("customers")
    Records.from_list(
        [
            {"id": 10, "customer_id": 1},
            {"id": 11, "customer_id": 2},
        ],
        database=db,
    ).insert_into("orders")

    customers = db.table("customers").select()
    orders = db.table("orders").select()
    result = customers.anti_join(orders, on=[("id", "customer_id")]).collect()

    assert [row["id"] for row in result] == [3]
    assert result[0]["name"] == "Carol"
    db.close()


def test_semi_join_execution(tmp_path) -> None:
    """Semi-join returns left rows that have a match on the right."""
    db_path = tmp_path / "semi.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "customers",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()
    db.create_table(
        "orders",
        [column("id", "INTEGER", primary_key=True), column("customer_id", "INTEGER")],
    ).collect()

    Records.from_list(
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
            {"id": 3, "name": "Carol"},
        ],
        database=db,
    ).insert_into("customers")
    Records.from_list(
        [
            {"id": 10, "customer_id": 1},
            {"id": 11, "customer_id": 2},
        ],
        database=db,
    ).insert_into("orders")

    customers = db.table("customers").select()
    orders = db.table("orders").select()
    result = customers.semi_join(orders, on=[("id", "customer_id")]).collect()

    assert sorted(row["id"] for row in result) == [1, 2]
    db.close()
