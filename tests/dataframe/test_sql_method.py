"""Tests for db.sql() method - raw SQL query execution."""

import pytest
from builtins import sum as builtin_sum

from moltres import col, connect
from moltres.expressions.functions import sum, avg, count


def test_sql_basic_query(tmp_path):
    """Test basic SQL query execution."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)"
        )

    # Basic SQL query
    df = db.sql("SELECT * FROM users")
    results = df.collect()

    assert len(results) == 3
    assert results[0]["name"] == "Alice"
    assert results[1]["name"] == "Bob"
    assert results[2]["name"] == "Charlie"


def test_sql_parameterized_query(tmp_path):
    """Test parameterized SQL query."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, status TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age, status) VALUES (1, 'Alice', 30, 'active'), (2, 'Bob', 25, 'inactive'), (3, 'Charlie', 35, 'active')"
        )

    # Parameterized query
    df = db.sql("SELECT * FROM users WHERE id = :id AND status = :status", id=1, status="active")
    results = df.collect()

    assert len(results) == 1
    assert results[0]["name"] == "Alice"
    assert results[0]["id"] == 1


def test_sql_chaining_operations(tmp_path):
    """Test chaining DataFrame operations on top of SQL results."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL, status TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO orders (id, amount, status) VALUES (1, 100.0, 'pending'), (2, 200.0, 'completed'), (3, 150.0, 'pending')"
        )

    # Chain operations on SQL result
    df = db.sql("SELECT * FROM orders").where(col("amount") > 100).limit(2)
    results = df.collect()

    assert len(results) == 2
    assert all(row["amount"] > 100 for row in results)


def test_sql_with_aggregations(tmp_path):
    """Test SQL query with aggregations and chaining."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (product TEXT, amount REAL, region TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO sales (product, amount, region) VALUES ('A', 100.0, 'US'), ('A', 200.0, 'US'), ('B', 150.0, 'EU')"
        )

    # SQL with aggregation, then chain operations
    df = (
        db.sql("SELECT product, region, SUM(amount) as total FROM sales GROUP BY product, region")
        .where(col("total") > 100)
        .order_by(col("total").desc())
    )
    results = df.collect()

    assert len(results) >= 1
    # Results should be ordered by total descending
    totals = [row["total"] for row in results]
    assert totals == sorted(totals, reverse=True)


def test_sql_with_joins(tmp_path):
    """Test SQL query with joins and chaining."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")
        conn.exec_driver_sql("INSERT INTO orders (id, user_id, amount) VALUES (1, 1, 100.0), (2, 1, 200.0), (3, 2, 150.0)")

    # SQL with join, then chain operations
    df = (
        db.sql("SELECT u.name, o.amount FROM users u JOIN orders o ON u.id = o.user_id")
        .where(col("amount") > 100)
        .select(col("name"), col("amount"))
    )
    results = df.collect()

    assert len(results) >= 1
    assert "name" in results[0]
    assert "amount" in results[0]


def test_sql_empty_result(tmp_path):
    """Test SQL query that returns no results."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")

    # Query that returns no results
    df = db.sql("SELECT * FROM users WHERE id = 999")
    results = df.collect()

    assert len(results) == 0
    assert results == []


def test_sql_streaming(tmp_path):
    """Test SQL query with streaming."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        # Insert multiple rows
        values = ", ".join([f"({i}, 'User{i}')" for i in range(1, 21)])
        conn.exec_driver_sql(f"INSERT INTO users (id, name) VALUES {values}")

    # Stream results
    df = db.sql("SELECT * FROM users")
    chunks = list(df.collect(stream=True))

    assert len(chunks) > 0
    total_rows = builtin_sum(len(chunk) for chunk in chunks)
    assert total_rows == 20


def test_sql_error_handling(tmp_path):
    """Test error handling for invalid SQL."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Invalid SQL should raise an error
    df = db.sql("SELECT * FROM nonexistent_table")
    with pytest.raises(Exception):  # Should raise a database error
        df.collect()


def test_sql_missing_parameters(tmp_path):
    """Test error handling for missing parameters."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")

    # Missing parameter should raise an error
    df = db.sql("SELECT * FROM users WHERE id = :id")
    with pytest.raises(Exception):  # Should raise a parameter error
        df.collect()


@pytest.mark.asyncio
async def test_async_sql_basic_query(tmp_path):
    """Test basic async SQL query execution."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("Async dependencies not installed")

    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        await conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)"
        )

    # Basic SQL query
    df = db.sql("SELECT * FROM users")
    results = await df.collect()

    assert len(results) == 2
    assert results[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_sql_parameterized_query(tmp_path):
    """Test parameterized async SQL query."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("Async dependencies not installed")

    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, status TEXT)")
        await conn.exec_driver_sql(
            "INSERT INTO users (id, name, status) VALUES (1, 'Alice', 'active'), (2, 'Bob', 'inactive')"
        )

    # Parameterized query
    df = db.sql("SELECT * FROM users WHERE id = :id", id=1)
    results = await df.collect()

    assert len(results) == 1
    assert results[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_sql_chaining_operations(tmp_path):
    """Test chaining operations on async SQL results."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("Async dependencies not installed")

    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL)")
        await conn.exec_driver_sql(
            "INSERT INTO orders (id, amount) VALUES (1, 100.0), (2, 200.0), (3, 150.0)"
        )

    # Chain operations
    df = db.sql("SELECT * FROM orders").where(col("amount") > 100).limit(2)
    results = await df.collect()

    assert len(results) == 2
    assert all(row["amount"] > 100 for row in results)

    await db.close()

