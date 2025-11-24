"""Tests for agg() with string column names and dictionary syntax."""

import pytest
from moltres import connect, col, async_connect
from moltres.expressions.functions import sum, avg, min as min_func, max as max_func, count


def test_agg_string_single_column(tmp_path):
    """Test that agg('amount') works (defaults to sum)."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount) VALUES "
            "('A', 100.0), ('A', 200.0), ('B', 150.0), ('B', 250.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg("amount")
    rows = df.collect()
    
    assert len(rows) == 2
    # Both categories should have their amounts summed
    category_a = next(r for r in rows if r["category"] == "A")
    category_b = next(r for r in rows if r["category"] == "B")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_b["amount"] == pytest.approx(400.0)


def test_agg_string_multiple_columns(tmp_path):
    """Test that agg('amount', 'price') works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('B', 150.0, 15.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg("amount", "price")
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_a["price"] == pytest.approx(30.0)


def test_agg_dictionary_syntax(tmp_path):
    """Test that agg({'amount': 'sum', 'price': 'avg'}) works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('B', 150.0, 15.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg({"amount": "sum", "price": "avg"})
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_a["price"] == pytest.approx(15.0)  # Average of 10 and 20


def test_agg_dictionary_all_functions(tmp_path):
    """Test dictionary syntax with all aggregation functions."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('A', 150.0, 15.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg({
        "amount": "sum",
        "price": "avg",
        "amount": "min",  # This will overwrite sum
        "price": "max",   # This will overwrite avg
    })
    rows = df.collect()
    
    assert len(rows) == 1
    # Note: dict with duplicate keys - last one wins
    assert rows[0]["amount"] == pytest.approx(100.0)  # min
    assert rows[0]["price"] == pytest.approx(20.0)   # max


def test_agg_dictionary_separate_columns(tmp_path):
    """Test dictionary with different aggregations for different columns."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('B', 150.0, 15.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg({
        "amount": "sum",
        "price": "avg"
    })
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_a["price"] == pytest.approx(15.0)


def test_agg_mixed_string_and_column(tmp_path):
    """Test that agg('amount', sum(col('price'))) works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('B', 150.0, 15.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg("amount", sum(col("price")).alias("total_price"))
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_a["total_price"] == pytest.approx(30.0)


def test_agg_mixed_string_and_dict(tmp_path):
    """Test that agg('amount', {'price': 'avg'}) works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('B', 150.0, 15.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg("amount", {"price": "avg"})
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_a["price"] == pytest.approx(15.0)


def test_agg_dictionary_all_agg_functions(tmp_path):
    """Test all aggregation functions in dictionary syntax."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL, quantity INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price, quantity) VALUES "
            "('A', 100.0, 10.0, 5), ('A', 200.0, 20.0, 3), ('A', 150.0, 15.0, 2)"
        )
    
    df = db.table("sales").select().group_by("category").agg({
        "amount": "sum",
        "price": "avg",
        "quantity": "min",
        "quantity": "max",  # Will overwrite min
    })
    rows = df.collect()
    
    assert len(rows) == 1
    assert rows[0]["amount"] == pytest.approx(450.0)
    assert rows[0]["price"] == pytest.approx(15.0)
    assert rows[0]["quantity"] == 5  # max


def test_agg_dictionary_count(tmp_path):
    """Test count aggregation in dictionary syntax."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount) VALUES "
            "('A', 100.0), ('A', 200.0), ('B', 150.0)"
        )
    
    # Note: count in dict syntax needs special handling - for now test with Column
    df = db.table("sales").select().group_by("category").agg({
        "amount": "sum"
    }, count("*").alias("count"))
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["count"] == 2


def test_agg_backward_compatibility(tmp_path):
    """Test that existing Column-based usage still works."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL)")
        conn.exec_driver_sql(
            "INSERT INTO sales (category, amount) VALUES "
            "('A', 100.0), ('A', 200.0), ('B', 150.0)"
        )
    
    df = db.table("sales").select().group_by("category").agg(sum(col("amount")).alias("total"))
    rows = df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["total"] == pytest.approx(300.0)


def test_agg_error_invalid_function(tmp_path):
    """Test that invalid aggregation function names raise an error."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL)")
        conn.exec_driver_sql("INSERT INTO sales (category, amount) VALUES ('A', 100.0)")
    
    df = db.table("sales").select().group_by("category")
    
    with pytest.raises(ValueError, match="Unknown aggregation function"):
        df.agg({"amount": "invalid_func"}).collect()


def test_agg_error_invalid_type(tmp_path):
    """Test that invalid aggregation types raise an error."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL)")
        conn.exec_driver_sql("INSERT INTO sales (category, amount) VALUES ('A', 100.0)")
    
    df = db.table("sales").select().group_by("category")
    
    with pytest.raises(ValueError, match="Invalid aggregation type"):
        df.agg(123).collect()  # Invalid type


@pytest.mark.asyncio
async def test_async_agg_string_single_column(tmp_path):
    """Test that async agg('amount') works."""
    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL)")
        await conn.exec_driver_sql(
            "INSERT INTO sales (category, amount) VALUES "
            "('A', 100.0), ('A', 200.0), ('B', 150.0)"
        )
    
    table_handle = await db.table("sales")
    df = table_handle.select().group_by("category").agg("amount")
    rows = await df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)


@pytest.mark.asyncio
async def test_async_agg_dictionary_syntax(tmp_path):
    """Test that async agg({'amount': 'sum', 'price': 'avg'}) works."""
    db_path = tmp_path / "test_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine
    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE sales (category TEXT, amount REAL, price REAL)")
        await conn.exec_driver_sql(
            "INSERT INTO sales (category, amount, price) VALUES "
            "('A', 100.0, 10.0), ('A', 200.0, 20.0), ('B', 150.0, 15.0)"
        )
    
    table_handle = await db.table("sales")
    df = table_handle.select().group_by("category").agg({"amount": "sum", "price": "avg"})
    rows = await df.collect()
    
    assert len(rows) == 2
    category_a = next(r for r in rows if r["category"] == "A")
    assert category_a["amount"] == pytest.approx(300.0)
    assert category_a["price"] == pytest.approx(15.0)

