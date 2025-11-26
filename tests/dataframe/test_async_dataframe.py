"""Tests for async DataFrame operations."""

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col
from moltres.expressions import functions as F


@pytest.mark.asyncio
async def test_async_dataframe_collect(tmp_path):
    """Test async DataFrame collect operation."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
            pk="id",
        )
    ).write.insertInto("users")

    # Query with filters - use db.table().select() for SQL operations
    table_handle = await db.table("users")
    df = table_handle.select()
    filtered = df.where(col("age") > 25)
    results = await filtered.collect()

    assert len(results) == 2
    assert all(r["age"] > 25 for r in results)

    await db.close()


@pytest.mark.asyncio
async def test_async_dot_notation_select(tmp_path):
    """Test dot notation column selection in async select()."""
    db_path = tmp_path / "async_dot_notation_select.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
    ).collect()

    await (
        await db.createDataFrame(
            [{"id": 1, "name": "Alice", "age": 30}, {"id": 2, "name": "Bob", "age": 25}],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test dot notation in select
    result = await df.select(df.id, df.name).collect()

    assert len(result) == 2
    assert "id" in result[0]
    assert "name" in result[0]
    assert "age" not in result[0]
    assert result[0]["id"] == 1
    assert result[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_dot_notation_where(tmp_path):
    """Test dot notation column selection in async where()."""
    db_path = tmp_path / "async_dot_notation_where.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test dot notation in where
    result = await df.where(df.age > 28).order_by(df.id).collect()

    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Charlie"

    await db.close()


@pytest.mark.asyncio
async def test_async_dot_notation_order_by(tmp_path):
    """Test dot notation column selection in async order_by()."""
    db_path = tmp_path / "async_dot_notation_order_by.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test dot notation in order_by
    result = await df.select(df.name).order_by(df.name).collect()

    assert len(result) == 3
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"
    assert result[2]["name"] == "Charlie"

    await db.close()


@pytest.mark.asyncio
async def test_async_order_by_string_parameter(tmp_path):
    """Test async order_by() with string column names (PySpark compatibility)."""
    db_path = tmp_path / "async_order_by_string.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "name": "Charlie", "age": 35},
                {"id": 2, "name": "Alice", "age": 30},
                {"id": 3, "name": "Bob", "age": 25},
            ],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test string column name (PySpark-style)
    result = await df.order_by("name").collect()
    assert len(result) == 3
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"
    assert result[2]["name"] == "Charlie"

    # Test Column object (backward compatibility)
    result2 = await df.order_by(col("name")).collect()
    assert len(result2) == 3
    assert result2[0]["name"] == "Alice"

    # Test mixed string and Column
    result3 = await df.order_by("age", col("name")).collect()
    assert len(result3) == 3
    assert result3[0]["age"] == 25  # Bob
    assert result3[1]["age"] == 30  # Alice
    assert result3[2]["age"] == 35  # Charlie

    # Test PySpark-style aliases with strings
    result4 = await df.orderBy("name").collect()
    assert len(result4) == 3
    assert result4[0]["name"] == "Alice"

    result5 = await df.sort("name").collect()
    assert len(result5) == 3
    assert result5[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_dot_notation_group_by(tmp_path):
    """Test dot notation column selection in async group_by()."""
    db_path = tmp_path / "async_dot_notation_group_by.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "sales",
        [column("id", "INTEGER"), column("category", "TEXT"), column("amount", "REAL")],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 10.0},
                {"id": 2, "category": "A", "amount": 20.0},
                {"id": 3, "category": "B", "amount": 15.0},
            ],
            pk="id",
        )
    ).write.insertInto("sales")

    table_handle = await db.table("sales")
    df = table_handle.select()

    # Test dot notation in group_by
    from moltres.expressions.functions import sum as sum_func

    result = await (
        df.group_by(df.category).agg(sum_func(df.amount).alias("total")).order_by(df.category)
    ).collect()

    assert len(result) == 2
    assert result[0]["category"] == "A"
    assert result[0]["total"] == 30.0
    assert result[1]["category"] == "B"
    assert result[1]["total"] == 15.0

    await db.close()


@pytest.mark.asyncio
async def test_async_dot_notation_methods_still_work(tmp_path):
    """Test that existing methods still work when using dot notation in async."""
    db_path = tmp_path / "async_dot_notation_methods.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
    ).collect()

    await (
        await db.createDataFrame(
            [{"id": 1, "name": "Alice", "age": 30}],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Test that methods still work
    assert hasattr(df, "select")
    assert hasattr(df, "where")
    assert hasattr(df, "limit")
    assert callable(df.select)
    assert callable(df.where)

    # Test that properties still work
    assert hasattr(df, "na")
    assert hasattr(df, "write")
    assert df.na is not None
    assert df.write is not None

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_select(tmp_path):
    """Test async DataFrame column selection."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "data",
        [
            column("id", "INTEGER", primary_key=True),
            column("a", "INTEGER"),
            column("b", "INTEGER"),
            column("c", "INTEGER"),
        ],
    ).collect()

    await (
        await db.createDataFrame(
            [{"a": 1, "b": 2, "c": 3}],
            auto_pk="id",
        )
    ).write.insertInto("data")
    # Use db.table().select() for SQL operations
    table_handle = await db.table("data")
    df = table_handle.select()
    selected = df.select("a", "c")
    results = await selected.collect()

    assert len(results) == 1
    assert "a" in results[0]
    assert "c" in results[0]
    assert "b" not in results[0]

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_limit(tmp_path):
    """Test async DataFrame limit operation."""
    db_path = tmp_path / "test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table("numbers", [column("n", "INTEGER", primary_key=True)]).collect()

    rows = [{"n": i} for i in range(10)]
    await (await db.createDataFrame(rows, pk="n")).write.insertInto("numbers")

    # Use db.table().select() for SQL operations
    table_handle = await db.table("numbers")
    df = table_handle.select()
    limited = df.limit(5)
    results = await limited.collect()

    assert len(results) == 5

    await db.close()


@pytest.mark.asyncio
async def test_async_withColumn_window_function(tmp_path):
    """Test async withColumn() with window functions (PySpark compatibility)."""
    db_path = tmp_path / "async_withcolumn_window.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "sales",
        [
            column("id", "INTEGER"),
            column("category", "TEXT"),
            column("amount", "REAL"),
        ],
    ).collect()

    await (
        await db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100.0},
                {"id": 2, "category": "A", "amount": 200.0},
                {"id": 3, "category": "B", "amount": 150.0},
            ],
            pk="id",
        )
    ).write.insertInto("sales")

    table_handle = await db.table("sales")
    df = table_handle.select()

    # Test basic window function in withColumn
    df_with_row_num = df.withColumn(
        "row_num", F.row_number().over(partition_by=col("category"), order_by=col("amount"))
    )
    result = await df_with_row_num.collect()

    assert len(result) == 3
    assert "row_num" in result[0]
    assert "id" in result[0]
    assert "category" in result[0]
    assert "amount" in result[0]
    # Check row numbers are correct
    row_nums_by_category = {}
    for row in result:
        cat = row["category"]
        if cat not in row_nums_by_category:
            row_nums_by_category[cat] = []
        row_nums_by_category[cat].append(row["row_num"])
    assert sorted(row_nums_by_category["A"]) == [1, 2]
    assert sorted(row_nums_by_category["B"]) == [1]

    await db.close()
