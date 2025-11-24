"""Tests for select('*') functionality."""

import pytest
from moltres import connect, col, async_connect


def test_select_star_only(tmp_path):
    """Test that select('*') is equivalent to select()."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com'), (2, 'Bob', 'bob@example.com')"
        )

    df1 = db.table("users").select()
    df2 = db.table("users").select("*")

    rows1 = df1.collect()
    rows2 = df2.collect()

    assert rows1 == rows2
    assert len(rows1) == 2
    assert rows1[0]["name"] == "Alice"
    assert rows1[1]["name"] == "Bob"


def test_select_star_with_other_columns(tmp_path):
    """Test that select('*', col('new_col')) works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER, amount REAL)")
        conn.exec_driver_sql("INSERT INTO orders (id, amount) VALUES (1, 100.0), (2, 200.0)")

    df = db.table("orders").select("*", (col("amount") * 1.1).alias("with_tax"))
    rows = df.collect()

    assert len(rows) == 2
    assert "id" in rows[0]
    assert "amount" in rows[0]
    assert "with_tax" in rows[0]
    assert rows[0]["with_tax"] == pytest.approx(110.0)
    assert rows[1]["with_tax"] == pytest.approx(220.0)


def test_select_star_with_string_column(tmp_path):
    """Test that select('*', 'column_name') works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE products (id INTEGER, name TEXT, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO products (id, name, price) VALUES (1, 'Widget', 10.0), (2, 'Gadget', 20.0)"
        )

    df = db.table("products").select("*", "name")
    rows = df.collect()

    assert len(rows) == 2
    assert "id" in rows[0]
    assert "name" in rows[0]
    assert "price" in rows[0]
    # "name" appears twice (once from *, once explicitly)
    assert rows[0]["name"] == "Widget"


def test_select_star_with_expressions(tmp_path):
    """Test that select('*', ...) works with complex expressions."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (id INTEGER, quantity INTEGER, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (id, quantity, price) VALUES (1, 5, 10.0), (2, 3, 20.0)"
        )

    df = db.table("sales").select(
        "*",
        (col("quantity") * col("price")).alias("total"),
        (col("price") * 1.1).alias("price_with_tax"),
    )
    rows = df.collect()

    assert len(rows) == 2
    assert "id" in rows[0]
    assert "quantity" in rows[0]
    assert "price" in rows[0]
    assert "total" in rows[0]
    assert "price_with_tax" in rows[0]
    assert rows[0]["total"] == pytest.approx(50.0)
    assert rows[1]["total"] == pytest.approx(60.0)


def test_select_star_chaining(tmp_path):
    """Test that select('*') can be chained with other operations."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 25), (2, 'Bob', 30), (3, 'Charlie', 25)"
        )

    df = db.table("users").select("*").where(col("age") > 25).order_by(col("name"))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Bob"
    assert rows[0]["age"] == 30


def test_select_star_after_join(tmp_path):
    """Test that select('*') works after joins."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE customers (id INTEGER, name TEXT)")
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount REAL)")
        conn.exec_driver_sql("INSERT INTO customers (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        conn.exec_driver_sql(
            "INSERT INTO orders (id, customer_id, amount) VALUES (1, 1, 100.0), (2, 2, 200.0)"
        )

    customers = db.table("customers").select()
    orders = db.table("orders").select()

    df = customers.join(orders, on=[("id", "customer_id")]).select("*")
    rows = df.collect()

    assert len(rows) == 2
    # Should have all columns from both tables
    assert "id" in rows[0]  # From customers
    assert "name" in rows[0]  # From customers
    assert "amount" in rows[0]  # From orders


def test_select_star_multiple_stars(tmp_path):
    """Test that select('*', '*') handles duplicate stars gracefully."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")

    # Multiple stars should be filtered to one
    df = db.table("users").select("*", "*")
    rows = df.collect()

    assert len(rows) == 1
    assert "id" in rows[0]
    assert "name" in rows[0]


@pytest.mark.asyncio
async def test_async_select_star_only(tmp_path):
    """Test that async select('*') is equivalent to select()."""
    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, email TEXT)")
        await conn.exec_driver_sql(
            "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com'), (2, 'Bob', 'bob@example.com')"
        )

    table_handle = await db.table("users")
    df1 = table_handle.select()
    df2 = table_handle.select("*")

    rows1 = await df1.collect()
    rows2 = await df2.collect()

    assert rows1 == rows2
    assert len(rows1) == 2
    assert rows1[0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_async_select_star_with_other_columns(tmp_path):
    """Test that async select('*', col('new_col')) works."""
    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE orders (id INTEGER, amount REAL)")
        await conn.exec_driver_sql("INSERT INTO orders (id, amount) VALUES (1, 100.0), (2, 200.0)")

    table_handle = await db.table("orders")
    df = table_handle.select("*", (col("amount") * 1.1).alias("with_tax"))
    rows = await df.collect()

    assert len(rows) == 2
    assert "id" in rows[0]
    assert "amount" in rows[0]
    assert "with_tax" in rows[0]
    assert rows[0]["with_tax"] == pytest.approx(110.0)
