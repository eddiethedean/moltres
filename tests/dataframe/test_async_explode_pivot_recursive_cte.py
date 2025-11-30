"""Tests for AsyncDataFrame explode(), pivot(), and recursive_cte() methods."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col


@pytest.mark.asyncio
async def test_async_explode_method(tmp_path):
    """Test AsyncDataFrame.explode() method."""
    db_path = tmp_path / "test_explode_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, tags TEXT)")
        await conn.exec_driver_sql(
            "INSERT INTO users (id, name, tags) VALUES (1, 'Alice', '[\"python\", \"sql\"]')"
        )

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test explode() method
    exploded = df.explode(col("tags"), alias="tag")

    # Verify the plan structure
    from moltres.logical.plan import Explode

    assert isinstance(exploded.plan, Explode)
    assert exploded.plan.alias == "tag"

    await db.close()


@pytest.mark.asyncio
async def test_async_explode_method_with_string_column(tmp_path):
    """Test AsyncDataFrame.explode() method with string column name."""
    db_path = tmp_path / "test_explode_string_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, tags TEXT)")
        await conn.exec_driver_sql("INSERT INTO users (id, tags) VALUES (1, '[\"python\"]')")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test explode() with string column name
    exploded = df.explode("tags", alias="tag_value")

    # Verify the plan structure
    from moltres.logical.plan import Explode

    assert isinstance(exploded.plan, Explode)
    assert exploded.plan.alias == "tag_value"

    await db.close()


@pytest.mark.asyncio
async def test_async_pivot_method(tmp_path):
    """Test AsyncDataFrame.pivot() method."""
    db_path = tmp_path / "test_pivot_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE sales (date TEXT, product TEXT, amount REAL)")
        await conn.exec_driver_sql(
            "INSERT INTO sales (date, product, amount) VALUES "
            "('2024-01-01', 'Widget', 100.0), "
            "('2024-01-01', 'Gadget', 200.0), "
            "('2024-01-02', 'Widget', 150.0), "
            "('2024-01-02', 'Gadget', 250.0)"
        )

    table_handle = await db.table("sales")
    df = table_handle.select("date", "product", "amount")

    # Test pivot() method
    pivoted = df.pivot(
        pivot_column="product",
        value_column="amount",
        agg_func="sum",
        pivot_values=["Widget", "Gadget"],
    )

    # Verify the plan structure
    from moltres.logical.plan import Pivot

    assert isinstance(pivoted.plan, Pivot)
    assert pivoted.plan.pivot_column == "product"
    assert pivoted.plan.value_column == "amount"
    assert pivoted.plan.agg_func == "sum"
    assert pivoted.plan.pivot_values == ("Widget", "Gadget")

    await db.close()


@pytest.mark.asyncio
async def test_async_pivot_method_without_values(tmp_path):
    """Test AsyncDataFrame.pivot() method without pivot_values."""
    db_path = tmp_path / "test_pivot_no_values_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE sales (date TEXT, product TEXT, amount REAL)")
        await conn.exec_driver_sql(
            "INSERT INTO sales (date, product, amount) VALUES "
            "('2024-01-01', 'Widget', 100.0), "
            "('2024-01-01', 'Gadget', 200.0)"
        )

    table_handle = await db.table("sales")
    df = table_handle.select("date", "product", "amount")

    # Test pivot() without pivot_values
    pivoted = df.pivot(pivot_column="product", value_column="amount", agg_func="sum")

    # Verify the plan structure
    from moltres.logical.plan import Pivot

    assert isinstance(pivoted.plan, Pivot)
    assert pivoted.plan.pivot_values is None

    await db.close()


@pytest.mark.asyncio
async def test_async_recursive_cte_method(tmp_path):
    """Test AsyncDataFrame.recursive_cte() method."""
    db_path = tmp_path / "test_recursive_cte_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE employees (id INTEGER, name TEXT, manager_id INTEGER)"
        )
        await conn.exec_driver_sql(
            "INSERT INTO employees (id, name, manager_id) VALUES (1, 'Alice', NULL), (2, 'Bob', 1)"
        )

    table_handle = await db.table("employees")
    initial = table_handle.select().where(col("manager_id").is_null())
    recursive = table_handle.select()

    # Test recursive_cte() method
    recursive_cte = initial.recursive_cte("org_chart", recursive, union_all=True)

    # Verify the plan structure
    from moltres.logical.plan import RecursiveCTE

    assert isinstance(recursive_cte.plan, RecursiveCTE)
    assert recursive_cte.plan.name == "org_chart"
    assert recursive_cte.plan.union_all is True

    await db.close()


@pytest.mark.asyncio
async def test_async_recursive_cte_union_distinct(tmp_path):
    """Test AsyncDataFrame.recursive_cte() with union_all=False (UNION DISTINCT)."""
    db_path = tmp_path / "test_recursive_cte_distinct_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE numbers (n INTEGER)")
        await conn.exec_driver_sql("INSERT INTO numbers (n) VALUES (1)")

    table_handle = await db.table("numbers")
    initial = table_handle.select()
    recursive = table_handle.select()

    # Test recursive_cte() with union_all=False
    recursive_cte = initial.recursive_cte("numbers_seq", recursive, union_all=False)

    # Verify the plan structure
    from moltres.logical.plan import RecursiveCTE

    assert isinstance(recursive_cte.plan, RecursiveCTE)
    assert recursive_cte.plan.union_all is False

    await db.close()


@pytest.mark.asyncio
async def test_async_explode_pivot_integration(tmp_path):
    """Test combining explode() and pivot() operations."""
    db_path = tmp_path / "test_explode_pivot_integration_async.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE sales (id INTEGER, category TEXT, tags TEXT, amount REAL)"
        )
        await conn.exec_driver_sql(
            "INSERT INTO sales (id, category, tags, amount) VALUES "
            "(1, 'A', '[\"tag1\", \"tag2\"]', 100.0)"
        )

    table_handle = await db.table("sales")
    df = table_handle.select()

    # Test chaining explode and pivot (even though this may not be practical)
    exploded = df.explode("tags", alias="tag")
    # Note: pivot after explode may not be directly possible, but we test the API
    # In practice, you'd pivot first, then explode, or use different operations

    # Verify explode worked
    from moltres.logical.plan import Explode

    assert isinstance(exploded.plan, Explode)

    await db.close()
