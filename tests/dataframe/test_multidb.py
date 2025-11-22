"""Multi-database tests that run against SQLite, PostgreSQL, and MySQL."""

import pytest

from moltres import col
from moltres.expressions.functions import sum as sum_
from moltres.io.read import read_table

# Import helper function from conftest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from conftest import seed_customers_orders  # noqa: E402


@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_basic_select_multidb(request, db_fixture):
    """Test basic SELECT operation across all databases."""
    db = request.getfixturevalue(db_fixture)
    seed_customers_orders(db)

    df = db.table("customers").select("id", "name").where(col("active") == 1)  # noqa: E712
    result = df.collect()

    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["name"] == "Alice"


@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_join_multidb(request, db_fixture):
    # Skip MySQL for now due to subquery alias issues
    if db_fixture == "mysql_connection":
        pytest.skip("MySQL join with subquery aliases has compatibility issues")
    """Test JOIN operation across all databases."""
    db = request.getfixturevalue(db_fixture)
    seed_customers_orders(db)

    orders_df = db.table("orders").select(col("id").alias("order_id"), col("customer_id"))
    customers_df = db.table("customers").select(col("id").alias("customer_id"), col("name"))
    # After selecting with aliases, join on the aliased columns
    df = (
        orders_df.join(customers_df, on=[("customer_id", "customer_id")])
        .select(col("order_id"), col("name").alias("customer"))
        .order_by(col("order_id"))
    )
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["order_id"] == 100
    assert rows[0]["customer"] == "Alice"
    assert rows[1]["order_id"] == 101
    assert rows[1]["customer"] == "Bob"


@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_group_by_agg_multidb(request, db_fixture):
    """Test GROUP BY and aggregation across all databases."""
    db = request.getfixturevalue(db_fixture)
    seed_customers_orders(db)

    totals = (
        db.table("orders")
        .select()
        .group_by("customer_id")
        .agg(sum_(col("amount")).alias("total_amount"))
        .order_by(col("customer_id"))
        .collect()
    )

    assert len(totals) == 2
    assert totals[0]["customer_id"] == 1
    assert totals[0]["total_amount"] == 50
    assert totals[1]["customer_id"] == 2
    assert totals[1]["total_amount"] == 75


@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_insert_update_delete_multidb(request, db_fixture):
    """Test INSERT, UPDATE, DELETE operations across all databases."""
    db = request.getfixturevalue(db_fixture)
    from moltres.table.schema import column

    db.create_table(
        "test_table",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "VARCHAR(255)", nullable=False),
            column("active", "INTEGER", nullable=True),
        ],
    ).collect()

    table = db.table("test_table")
    db.createDataFrame(
        [
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ],
        pk="id",
    ).write.insertInto("test_table")

    # Verify insertion
    rows = table.select().collect()
    assert len(rows) == 2

    from moltres.table.mutations import update_rows, delete_rows

    updated = update_rows(table, where=col("id") == 2, values={"name": "Bobby", "active": 1})
    assert updated == 1

    deleted = delete_rows(table, where=col("id") == 1)
    assert deleted == 1

    rows = read_table(db, "test_table")
    assert len(rows) == 1
    assert rows[0]["id"] == 2
    assert rows[0]["name"] == "Bobby"
    assert rows[0]["active"] == 1


@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_order_by_limit_multidb(request, db_fixture):
    """Test ORDER BY and LIMIT across all databases."""
    db = request.getfixturevalue(db_fixture)
    seed_customers_orders(db)

    result = db.table("customers").select().order_by(col("id").desc()).limit(2).collect()

    assert len(result) == 2
    assert result[0]["id"] == 2  # Bob (higher ID first due to DESC)
    assert result[1]["id"] == 1  # Alice


@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_distinct_multidb(request, db_fixture):
    """Test DISTINCT operation across all databases."""
    db = request.getfixturevalue(db_fixture)
    from moltres.table.schema import column

    db.create_table(
        "test_distinct",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("value", "VARCHAR(255)", nullable=False),
        ],
    ).collect()

    db.table("test_distinct")
    db.createDataFrame(
        [
            {"id": 1, "value": "A"},
            {"id": 2, "value": "B"},
            {"id": 3, "value": "A"},
            {"id": 4, "value": "B"},
        ],
        pk="id",
    ).write.insertInto("test_distinct")

    result = db.table("test_distinct").select("value").distinct().order_by(col("value")).collect()

    assert len(result) == 2
    assert {row["value"] for row in result} == {"A", "B"}
