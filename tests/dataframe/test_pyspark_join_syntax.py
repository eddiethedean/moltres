"""Tests for PySpark-style join syntax with Column expressions."""

from __future__ import annotations

import pytest

from moltres import col, connect
from moltres.table.schema import column


@pytest.fixture
def sample_db(tmp_path):
    """Create a sample database with test tables."""
    db = connect(f"duckdb:///{tmp_path}/test.db")

    with db.batch():
        db.create_table(
            "customers",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "VARCHAR"),
                column("region", "VARCHAR"),
            ],
        ).collect()

        db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("customer_id", "INTEGER"),
                column("amount", "REAL"),
                column("region", "VARCHAR"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "name": "Alice", "region": "US"},
            {"id": 2, "name": "Bob", "region": "UK"},
            {"id": 3, "name": "Charlie", "region": "US"},
        ],
        database=db,
    ).insert_into("customers")

    Records.from_list(
        [
            {"id": 1, "customer_id": 1, "amount": 100.0, "region": "US"},
            {"id": 2, "customer_id": 2, "amount": 200.0, "region": "UK"},
            {"id": 3, "customer_id": 1, "amount": 150.0, "region": "US"},
        ],
        database=db,
    ).insert_into("orders")

    return db


def test_pyspark_style_join_single_condition(sample_db):
    """Test PySpark-style join with single Column expression."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("orders")
        .select()
        .join(
            db.table("customers").select(),
            on=[col("orders.customer_id") == col("customers.id")],
        )
        .collect()
    )

    assert len(result) == 3
    assert all("name" in row for row in result)
    assert all("amount" in row for row in result)


def test_pyspark_style_join_single_column_expression(sample_db):
    """Test PySpark-style join with single Column expression (not in list)."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("orders")
        .select()
        .join(
            db.table("customers").select(),
            on=col("orders.customer_id") == col("customers.id"),
        )
        .collect()
    )

    assert len(result) == 3
    assert all("name" in row for row in result)


def test_pyspark_style_join_multiple_conditions(sample_db):
    """Test PySpark-style join with multiple Column expressions."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("orders")
        .select()
        .join(
            db.table("customers").select(),
            on=[
                col("orders.customer_id") == col("customers.id"),
                col("orders.region") == col("customers.region"),
            ],
        )
        .collect()
    )

    assert len(result) == 3
    assert all("name" in row for row in result)


def test_pyspark_style_join_table_qualified(sample_db):
    """Test PySpark-style join with table-qualified column names."""
    db = sample_db

    result = (
        db.table("orders")
        .select()
        .join(
            db.table("customers").select(),
            on=[col("orders.customer_id") == col("customers.id")],
        )
        .collect()
    )

    assert len(result) == 3
    assert all("name" in row for row in result)


def test_pyspark_style_join_backward_compatible_tuple(sample_db):
    """Test that tuple syntax still works (backward compatibility)."""
    db = sample_db

    result = (
        db.table("orders")
        .select()
        .join(db.table("customers").select(), on=[("customer_id", "id")])
        .collect()
    )

    assert len(result) == 3
    assert all("name" in row for row in result)


def test_pyspark_style_join_same_column_name(sample_db):
    """Test join with same column name using string."""
    db = sample_db

    # Create a table with same column name
    with db.batch():
        db.create_table(
            "orders2",
            [
                column("id", "INTEGER", primary_key=True),
                column("customer_id", "INTEGER"),
                column("amount", "REAL"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "customer_id": 1, "amount": 100.0},
        ],
        database=db,
    ).insert_into("orders2")

    # Join on same column name
    result = (
        db.table("orders").select().join(db.table("orders2").select(), on="customer_id").collect()
    )

    assert len(result) >= 1


def test_pyspark_style_join_left_outer(sample_db):
    """Test PySpark-style left outer join."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("customers")
        .select()
        .join(
            db.table("orders").select(),
            on=[col("customers.id") == col("orders.customer_id")],
            how="left",
        )
        .collect()
    )

    # Should include all customers, even those without orders
    assert len(result) >= 3
    customer_names = {row["name"] for row in result}
    assert "Alice" in customer_names
    assert "Bob" in customer_names
    assert "Charlie" in customer_names


def test_pyspark_style_join_right_outer(sample_db):
    """Test PySpark-style right outer join."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("customers")
        .select()
        .join(
            db.table("orders").select(),
            on=[col("customers.id") == col("orders.customer_id")],
            how="right",
        )
        .collect()
    )

    # Should include all orders
    assert len(result) == 3


def test_pyspark_style_join_inner(sample_db):
    """Test PySpark-style inner join (explicit)."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("orders")
        .select()
        .join(
            db.table("customers").select(),
            on=[col("orders.customer_id") == col("customers.id")],
            how="inner",
        )
        .collect()
    )

    assert len(result) == 3


def test_pyspark_style_join_full_outer(sample_db):
    """Test PySpark-style full outer join."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("customers")
        .select()
        .join(
            db.table("orders").select(),
            on=[col("customers.id") == col("orders.customer_id")],
            how="full",
        )
        .collect()
    )

    # Should include all customers and all orders
    assert len(result) >= 3


def test_pyspark_style_join_with_filter(sample_db):
    """Test PySpark-style join combined with filter."""
    db = sample_db

    # Use table-qualified names to avoid ambiguity
    result = (
        db.table("orders")
        .select()
        .where(col("amount") > 100)
        .join(
            db.table("customers").select(),
            on=[col("orders.customer_id") == col("customers.id")],
        )
        .collect()
    )

    assert len(result) == 2
    assert all(row["amount"] > 100 for row in result)


def test_pyspark_style_join_error_mixed_types():
    """Test that mixing Column and tuple in join raises error."""
    db = connect("duckdb:///:memory:")

    with pytest.raises(
        ValueError, match="All elements in join condition must be Column expressions"
    ):
        (
            db.table("orders")
            .select()
            .join(
                db.table("customers").select(),
                on=[col("orders.customer_id") == col("customers.id"), ("region", "region")],  # type: ignore[list-item]
            )
        )


@pytest.mark.asyncio
async def test_pyspark_style_join_async(tmp_path):
    """Test PySpark-style join with async DataFrame."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed", allow_module_level=True)

    from moltres import async_connect

    db_path = tmp_path / "test_async.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await async_db.create_table(
        "customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "VARCHAR"),
        ],
    ).collect()

    await async_db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("customer_id", "INTEGER"),
            column("amount", "REAL"),
        ],
    ).collect()

    from moltres.io.records import AsyncRecords

    customers_records = AsyncRecords(
        _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
        _database=async_db,
    )
    await customers_records.insert_into("customers")

    orders_records = AsyncRecords(
        _data=[
            {"id": 1, "customer_id": 1, "amount": 100.0},
            {"id": 2, "customer_id": 2, "amount": 200.0},
        ],
        _database=async_db,
    )
    await orders_records.insert_into("orders")

    # Use table-qualified names to avoid ambiguity
    orders_table = await async_db.table("orders")
    customers_table = await async_db.table("customers")
    result = await (
        orders_table.select()
        .join(
            customers_table.select(),
            on=[col("orders.customer_id") == col("customers.id")],
        )
        .collect()
    )

    assert len(result) == 2
    assert all("name" in row for row in result)

    await async_db.close()
