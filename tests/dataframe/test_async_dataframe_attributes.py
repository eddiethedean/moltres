"""Tests for AsyncDataFrame attributes: columns, schema, dtypes, printSchema."""

import io
from contextlib import redirect_stdout

import pytest

from moltres import async_connect, col
from moltres.expressions.functions import sum


@pytest.mark.asyncio
async def test_async_columns_property_table_scan(tmp_path):
    """Test .columns property for TableScan in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, email TEXT)"
        )
        await conn.exec_driver_sql(
            "INSERT INTO users (id, name, age, email) VALUES (1, 'Alice', 30, 'alice@example.com')"
        )

    table_handle = await async_db.table("users")
    df = table_handle.select()
    columns = df.columns

    assert isinstance(columns, list)
    assert len(columns) == 4
    assert "id" in columns
    assert "name" in columns
    assert "age" in columns
    assert "email" in columns


@pytest.mark.asyncio
async def test_async_columns_property_project(tmp_path):
    """Test .columns property for Project in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, email TEXT)"
        )

    table_handle = await async_db.table("users")
    df = table_handle.select("id", "name")
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns
    assert "age" not in columns
    assert "email" not in columns


@pytest.mark.asyncio
async def test_async_columns_property_with_alias(tmp_path):
    """Test .columns property with column aliases in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    table_handle = await async_db.table("users")
    df = table_handle.select(col("name").alias("full_name"))
    columns = df.columns

    assert len(columns) == 1
    assert "full_name" in columns
    assert "name" not in columns


@pytest.mark.asyncio
async def test_async_schema_property_table_scan(tmp_path):
    """Test .schema property for TableScan in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
        )

    table_handle = await async_db.table("users")
    df = table_handle.select()
    schema = df.schema

    assert isinstance(schema, list)
    assert len(schema) == 3

    # Check column names
    col_names = [col_info.name for col_info in schema]
    assert "id" in col_names
    assert "name" in col_names
    assert "age" in col_names

    # Check types
    for col_info in schema:
        assert isinstance(col_info.name, str)
        assert isinstance(col_info.type_name, str)
        assert len(col_info.type_name) > 0


@pytest.mark.asyncio
async def test_async_dtypes_property(tmp_path):
    """Test .dtypes property in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    table_handle = await async_db.table("users")
    df = table_handle.select()
    dtypes = df.dtypes

    assert isinstance(dtypes, list)
    assert len(dtypes) == 3

    # Check format: list of tuples
    for dtype in dtypes:
        assert isinstance(dtype, tuple)
        assert len(dtype) == 2
        assert isinstance(dtype[0], str)  # column name
        assert isinstance(dtype[1], str)  # type name

    # Check column names match
    col_names = [dtype[0] for dtype in dtypes]
    assert "id" in col_names
    assert "name" in col_names
    assert "age" in col_names


@pytest.mark.asyncio
async def test_async_print_schema(tmp_path):
    """Test .printSchema() method in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    table_handle = await async_db.table("users")
    df = table_handle.select()

    # Capture stdout
    f = io.StringIO()
    with redirect_stdout(f):
        df.printSchema()

    output = f.getvalue()

    # Check output format
    assert "root" in output
    assert "id:" in output
    assert "name:" in output
    assert "age:" in output
    assert "nullable = true" in output


@pytest.mark.asyncio
async def test_async_columns_property_aggregate(tmp_path):
    """Test .columns property for Aggregate operations in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE orders (id INTEGER, amount REAL, status TEXT)")
        await conn.exec_driver_sql(
            "INSERT INTO orders (id, amount, status) VALUES (1, 100.0, 'pending'), (2, 200.0, 'completed')"
        )

    table_handle = await async_db.table("orders")
    df = table_handle.select().groupBy("status").agg(sum(col("amount")).alias("total"))
    columns = df.columns

    assert len(columns) == 2
    assert "status" in columns
    assert "total" in columns


@pytest.mark.asyncio
async def test_async_columns_property_filter(tmp_path):
    """Test .columns property works through Filter operations in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")

    table_handle = await async_db.table("users")
    df = table_handle.select("id", "name").where(col("age") > 18)
    columns = df.columns

    assert len(columns) == 2
    assert "id" in columns
    assert "name" in columns


@pytest.mark.asyncio
async def test_async_get_table_schema_alias(tmp_path):
    """Test get_table_schema is an alias for get_table_columns in async context."""
    from moltres.utils.inspector import get_table_schema, get_table_columns

    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")

    schema1 = get_table_schema(async_db, "users")
    schema2 = get_table_columns(async_db, "users")

    assert len(schema1) == len(schema2) == 2
    assert schema1[0].name == schema2[0].name
    assert schema1[1].name == schema2[1].name


# Note: Testing inspection errors in async context is complex due to event loop handling
# The error path is tested in sync context (test_get_table_columns_inspection_error)
# and the async engine path is tested in test_get_table_columns_async_engine_no_running_loop


@pytest.mark.asyncio
async def test_async_extract_column_names_star_expansion(tmp_path):
    """Test _extract_column_names handles star (*) column expansion in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")

    table_handle = await async_db.table("users")
    df = table_handle.select("*")
    columns = df.columns

    assert len(columns) == 3
    assert "id" in columns
    assert "name" in columns
    assert "age" in columns


@pytest.mark.asyncio
async def test_async_extract_schema_star_expansion(tmp_path):
    """Test _extract_schema_from_plan handles star (*) column expansion in AsyncDataFrame."""
    db_path = tmp_path / "test.db"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    engine = async_db.connection_manager.engine

    async with engine.begin() as conn:
        await conn.exec_driver_sql("CREATE TABLE users (id INTEGER, name TEXT, age INTEGER)")

    table_handle = await async_db.table("users")
    df = table_handle.select("*")
    schema = df.schema

    assert len(schema) == 3
    col_names = [col_info.name for col_info in schema]
    assert "id" in col_names
    assert "name" in col_names
    assert "age" in col_names

