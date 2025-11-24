"""Tests for groupBy().pivot() functionality."""

import pytest
from moltres import connect, col, async_connect
from moltres.expressions.functions import sum


def test_pivot_basic(tmp_path):
    """Test basic pivot functionality."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, status TEXT, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, status, amount) VALUES "
            "('A', 'active', 100.0), "
            "('A', 'inactive', 50.0), "
            "('B', 'active', 200.0), "
            "('B', 'inactive', 150.0)"
        )

    df = (
        db.table("sales")
        .select()
        .group_by("category")
        .pivot("status", values=["active", "inactive"])
        .agg("amount")
    )
    rows = df.collect()

    assert len(rows) == 2
    # Find row for category A
    row_a = next(r for r in rows if r["category"] == "A")
    assert row_a["active"] == 100.0
    assert row_a["inactive"] == 50.0
    # Find row for category B
    row_b = next(r for r in rows if r["category"] == "B")
    assert row_b["active"] == 200.0
    assert row_b["inactive"] == 150.0


def test_pivot_with_column_expression(tmp_path):
    """Test pivot with Column aggregation expression."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, status TEXT, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, status, amount) VALUES "
            "('A', 'active', 100.0), "
            "('A', 'active', 50.0), "
            "('B', 'inactive', 200.0)"
        )

    df = (
        db.table("sales")
        .select()
        .group_by("category")
        .pivot("status", values=["active", "inactive"])
        .agg(sum(col("amount")))
    )
    rows = df.collect()

    assert len(rows) == 2
    row_a = next(r for r in rows if r["category"] == "A")
    assert row_a["active"] == 150.0  # Sum of 100 + 50
    assert row_a["inactive"] is None  # No inactive for A


def test_pivot_multiple_grouping_columns(tmp_path):
    """Test pivot with multiple grouping columns."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (region TEXT, category TEXT, status TEXT, amount REAL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO sales (region, category, status, amount) VALUES "
            "('North', 'A', 'active', 100.0), "
            "('North', 'A', 'inactive', 50.0), "
            "('South', 'A', 'active', 200.0)"
        )

    df = (
        db.table("sales")
        .select()
        .group_by("region", "category")
        .pivot("status", values=["active", "inactive"])
        .agg("amount")
    )
    rows = df.collect()

    assert len(rows) == 2
    row_north = next(r for r in rows if r["region"] == "North" and r["category"] == "A")
    assert row_north["active"] == 100.0
    assert row_north["inactive"] == 50.0


def test_pivot_error_no_aggregation(tmp_path):
    """Test that pivot requires an aggregation."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, status TEXT, amount REAL)")

    pivoted = (
        db.table("sales")
        .select()
        .group_by("category")
        .pivot("status", values=["active", "inactive"])
    )
    with pytest.raises(ValueError, match="agg requires at least one aggregation expression"):
        pivoted.agg()


def test_pivot_error_multiple_aggregations(tmp_path):
    """Test that pivot only supports one aggregation."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (category TEXT, status TEXT, amount REAL, price REAL)"
        )

    pivoted = (
        db.table("sales")
        .select()
        .group_by("category")
        .pivot("status", values=["active", "inactive"])
    )
    with pytest.raises(
        ValueError, match="Pivoted grouped aggregation supports only one aggregation expression"
    ):
        pivoted.agg("amount", "price")


def test_pivot_inferred_values(tmp_path):
    """Test pivot with inferred values (PySpark-style, no explicit values parameter)."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, status TEXT, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, status, amount) VALUES "
            "('A', 'active', 100.0), "
            "('A', 'inactive', 50.0), "
            "('A', 'pending', 25.0), "
            "('B', 'active', 200.0), "
            "('B', 'pending', 75.0)"
        )

    # Test without explicit values - should infer from data
    df = db.table("sales").select().group_by("category").pivot("status").agg("amount")
    rows = df.collect()

    assert len(rows) == 2
    row_a = next(r for r in rows if r["category"] == "A")
    assert row_a["active"] == 100.0
    assert row_a["inactive"] == 50.0
    assert row_a["pending"] == 25.0

    row_b = next(r for r in rows if r["category"] == "B")
    assert row_b["active"] == 200.0
    assert row_b["inactive"] is None  # No inactive for B
    assert row_b["pending"] == 75.0


@pytest.mark.asyncio
async def test_async_pivot_basic(tmp_path):
    """Test basic async pivot functionality."""
    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE sales (category TEXT, status TEXT, amount REAL)")
        await conn.exec_driver_sql(
            "INSERT INTO sales (category, status, amount) VALUES "
            "('A', 'active', 100.0), "
            "('A', 'inactive', 50.0), "
            "('B', 'active', 200.0)"
        )

    table_handle = await db.table("sales")
    pivoted = (
        table_handle.select().group_by("category").pivot("status", values=["active", "inactive"])
    )
    df = await pivoted.agg("amount")
    rows = await df.collect()

    assert len(rows) == 2
    row_a = next(r for r in rows if r["category"] == "A")
    assert row_a["active"] == 100.0
    assert row_a["inactive"] == 50.0
