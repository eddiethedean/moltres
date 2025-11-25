"""Tests for DataFrame.selectExpr() method."""

import pytest

from moltres import col, connect


def test_select_expr_basic_columns(tmp_path):
    """Test basic column selection with selectExpr."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, email) VALUES (1, 'Alice', 'alice@example.com'), (2, 'Bob', 'bob@example.com')"
        )

    df = db.table("users").select()
    result_df = df.selectExpr("id", "name", "email")
    rows = result_df.collect()

    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["email"] == "alice@example.com"


def test_select_expr_arithmetic(tmp_path):
    """Test arithmetic expressions in selectExpr."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL)")
        conn.exec_driver_sql("INSERT INTO orders (id, amount) VALUES (1, 100.0), (2, 200.0)")

    df = db.table("orders").select()
    result_df = df.selectExpr("id", "amount * 1.1 as with_tax")
    rows = result_df.collect()

    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert abs(rows[0]["with_tax"] - 110.0) < 0.01


def test_select_expr_with_aliases(tmp_path):
    """Test selectExpr with aliases."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")

    df = db.table("users").select()
    result_df = df.selectExpr("id", "name as full_name")
    rows = result_df.collect()

    assert len(rows) == 2
    assert "full_name" in rows[0]
    assert rows[0]["full_name"] == "Alice"


def test_select_expr_functions(tmp_path):
    """Test selectExpr with SQL functions."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")

    df = db.table("users").select()
    result_df = df.selectExpr("id", "UPPER(name) as name_upper")
    rows = result_df.collect()

    assert len(rows) == 2
    assert rows[0]["name_upper"] == "ALICE"
    assert rows[1]["name_upper"] == "BOB"


def test_select_expr_complex_expressions(tmp_path):
    """Test complex expressions in selectExpr."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL, tax REAL)")
        conn.exec_driver_sql(
            "INSERT INTO orders (id, amount, tax) VALUES (1, 100.0, 10.0), (2, 200.0, 20.0)"
        )

    df = db.table("orders").select()
    result_df = df.selectExpr("id", "(amount + tax) * 1.1 as total")
    rows = result_df.collect()

    assert len(rows) == 2
    assert abs(rows[0]["total"] - 121.0) < 0.01


def test_select_expr_chaining(tmp_path):
    """Test chaining selectExpr with other operations."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO orders (id, amount) VALUES (1, 100.0), (2, 200.0), (3, 50.0)"
        )

    df = db.table("orders").select()
    result_df = df.selectExpr("id", "amount").where(col("amount") > 100)
    rows = result_df.collect()

    assert len(rows) == 1
    assert rows[0]["id"] == 2


def test_select_expr_literals(tmp_path):
    """Test selectExpr with literals."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")

    df = db.table("users").select()
    result_df = df.selectExpr("id", "42 as answer", "'hello' as greeting")
    rows = result_df.collect()

    assert len(rows) == 1
    assert rows[0]["answer"] == 42
    assert rows[0]["greeting"] == "hello"


def test_select_expr_comparisons(tmp_path):
    """Test selectExpr with comparison operators."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE orders (id INTEGER PRIMARY KEY, amount REAL)")
        conn.exec_driver_sql("INSERT INTO orders (id, amount) VALUES (1, 100.0), (2, 200.0)")

    df = db.table("orders").select()
    result_df = df.selectExpr("id", "amount > 150 as is_large")
    rows = result_df.collect()

    assert len(rows) == 2
    assert rows[0]["is_large"] == 0  # False
    assert rows[1]["is_large"] == 1  # True


def test_select_expr_error_handling(tmp_path):
    """Test error handling for invalid SQL expressions."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice')")

    df = db.table("users").select()

    # Test invalid syntax - the parser should fail on unexpected tokens
    with pytest.raises((ValueError, Exception)):
        df.selectExpr("invalid syntax !!!")

    # Test unclosed parenthesis
    with pytest.raises(ValueError):
        df.selectExpr("(amount + tax")


@pytest.mark.asyncio
async def test_async_select_expr_basic(tmp_path):
    """Test async selectExpr."""
    from moltres import async_connect

    db_path = tmp_path / "test.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
        await conn.exec_driver_sql("INSERT INTO users (id, name) VALUES (1, 'Alice'), (2, 'Bob')")

    table_handle = await db.table("users")
    df = table_handle.select()
    result_df = df.selectExpr("id", "name", "UPPER(name) as name_upper")
    rows = await result_df.collect()

    assert len(rows) == 2
    assert rows[0]["name_upper"] == "ALICE"
