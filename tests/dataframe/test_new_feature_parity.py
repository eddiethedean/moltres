"""Tests for new feature parity across all DataFrame interfaces.

This test file covers:
- Phase 1: Column Access & Selection (__getitem__ for PySpark-style)
- Phase 2: Filtering & Joins (crossJoin, semi_join, anti_join)
- Phase 3: Aggregations & Column Manipulation (nunique, withColumns, dict agg)
- Phase 6: Schema & Inspection (show, take, first, summary, printSchema)
"""

import pytest

from moltres import col, connect
from moltres.expressions import functions as F


def _to_dict_list(results):
    """Convert results (pandas/polars DataFrame or list of dicts) to list of dicts."""
    try:
        import pandas as pd

        if isinstance(results, pd.DataFrame):
            return results.to_dict("records")
    except ImportError:
        pass
    try:
        import polars as pl

        if isinstance(results, pl.DataFrame):
            return results.to_dicts()
    except ImportError:
        pass
    return results


@pytest.fixture
def sample_db():
    """Create a sample database with test data."""
    db = connect("sqlite:///:memory:")

    from moltres.table.schema import column

    with db.batch():
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
                column("country", "TEXT"),
            ],
        ).collect()

        db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("user_id", "INTEGER"),
                column("amount", "REAL"),
                column("status", "TEXT"),
            ],
        ).collect()

        db.create_table(
            "products",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("price", "REAL"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
            {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
            {"id": 3, "name": "Charlie", "age": 35, "country": "USA"},
            {"id": 4, "name": "David", "age": 28, "country": "UK"},
        ],
        database=db,
    ).insert_into("users")

    Records.from_list(
        [
            {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
            {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
            {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
            {"id": 4, "user_id": 3, "amount": 300.0, "status": "active"},
        ],
        database=db,
    ).insert_into("orders")

    Records.from_list(
        [
            {"id": 1, "name": "Product A", "price": 10.0},
            {"id": 2, "name": "Product B", "price": 20.0},
        ],
        database=db,
    ).insert_into("products")

    yield db
    db.close()


# ============================================================================
# Phase 1: Column Access & Selection (__getitem__ for PySpark-style)
# ============================================================================


def test_pyspark_getitem_single_column(sample_db):
    """Test PySpark-style single column access: df['col']."""
    df = sample_db.table("users").select()
    
    # Single column access should return PySparkColumn with accessors
    col_expr = df["age"]
    assert hasattr(col_expr, "_column")
    assert hasattr(col_expr, "str")
    assert hasattr(col_expr, "dt")
    
    # Should work in expressions - use the column in a filter
    result = df.where(df["age"] > 25).collect()
    assert len(result) == 3  # 30, 35, 28
    assert "age" in result[0]


def test_pyspark_getitem_multiple_columns(sample_db):
    """Test PySpark-style multiple column access: df[['col1', 'col2']]."""
    df = sample_db.table("users").select()
    
    # Multiple column access should return DataFrame
    df_selected = df[["id", "name"]]
    from moltres.dataframe.dataframe import DataFrame
    
    assert isinstance(df_selected, DataFrame)
    
    result = df_selected.collect()
    assert len(result) == 4
    assert "id" in result[0]
    assert "name" in result[0]
    assert "age" not in result[0]


def test_pyspark_getitem_boolean_indexing(sample_db):
    """Test PySpark-style boolean indexing: df[df['age'] > 25]."""
    df = sample_db.table("users").select()
    
    # Boolean indexing should filter rows
    df_filtered = df[df["age"] > 25]
    from moltres.dataframe.dataframe import DataFrame
    
    assert isinstance(df_filtered, DataFrame)
    
    result = df_filtered.collect()
    assert len(result) == 3  # 30, 35, 28
    assert all(r["age"] > 25 for r in result)


def test_pyspark_getitem_string_accessor(sample_db):
    """Test PySpark-style string accessor via df['col'].str."""
    df = sample_db.table("users").select()
    
    # String accessor should be available
    col_expr = df["name"]
    assert hasattr(col_expr, "str")
    assert hasattr(col_expr.str, "upper")
    
    # Should work in select
    result = df.select(col_expr.str.upper().alias("name_upper")).collect()
    assert len(result) == 4
    assert "name_upper" in result[0]
    assert result[0]["name_upper"] == "ALICE"


# ============================================================================
# Phase 2: Filtering & Joins
# ============================================================================


def test_pandas_crossjoin(sample_db):
    """Test Pandas-style crossJoin()."""
    df1 = sample_db.table("users").pandas()
    df2 = sample_db.table("products").pandas()
    
    # Cross join should create Cartesian product
    df_cross = df1.crossJoin(df2)
    result = df_cross.collect()
    result_list = _to_dict_list(result)
    
    # 4 users * 2 products = 8 rows
    assert len(result_list) == 8
    assert "id" in result_list[0]  # From users
    assert "name" in result_list[0]  # From users
    assert "price" in result_list[0]  # From products


def test_pandas_cross_join_alias(sample_db):
    """Test Pandas-style cross_join() alias."""
    df1 = sample_db.table("users").pandas()
    df2 = sample_db.table("products").pandas()
    
    # cross_join should be an alias for crossJoin
    df_cross = df1.cross_join(df2)
    result = df_cross.collect()
    
    assert len(result) == 8


def test_polars_semi_join(sample_db):
    """Test Polars-style semi_join()."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("orders").polars()
    
    # Semi join: users who have orders
    df_semi = df1.semi_join(df2, left_on="id", right_on="user_id")
    result = df_semi.collect()
    result_list = _to_dict_list(result)
    
    # Users 1, 2, 3 have orders
    assert len(result_list) == 3
    user_ids = {r["id"] for r in result_list}
    assert user_ids == {1, 2, 3}


def test_polars_anti_join(sample_db):
    """Test Polars-style anti_join()."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("orders").polars()
    
    # Anti join: users who don't have orders
    df_anti = df1.anti_join(df2, left_on="id", right_on="user_id")
    result = df_anti.collect()
    result_list = _to_dict_list(result)
    
    # User 4 has no orders
    assert len(result_list) == 1
    assert result_list[0]["id"] == 4


def test_polars_semi_join_on(sample_db):
    """Test Polars-style semi_join() with 'on' parameter."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("orders").polars()
    
    # Create a temporary column in df2 for matching
    df2_with_user_id = df2.select(col("user_id").alias("id"), col("amount"))
    
    # Semi join with 'on' parameter
    df_semi = df1.semi_join(df2_with_user_id, on="id")
    result = df_semi.collect()
    
    assert len(result) == 3


# ============================================================================
# Phase 3: Aggregations & Column Manipulation
# ============================================================================


def test_pyspark_nunique_single_column(sample_db):
    """Test PySpark-style nunique() for a single column."""
    df = sample_db.table("users").select()
    
    # Count distinct countries
    count = df.nunique("country")
    assert count == 2  # USA and UK


def test_pyspark_nunique_all_columns(sample_db):
    """Test PySpark-style nunique() for all columns."""
    df = sample_db.table("users").select()
    
    # Count distinct for all columns
    counts = df.nunique()
    assert isinstance(counts, dict)
    assert counts["country"] == 2
    assert counts["id"] == 4  # All unique
    assert "name" in counts
    assert "age" in counts


def test_pyspark_withcolumns(sample_db):
    """Test PySpark-style withColumns() to add multiple columns."""
    df = sample_db.table("users").select()
    
    # Add multiple columns at once
    df_new = df.withColumns({
        "age_plus_10": col("age") + 10,
        "age_doubled": col("age") * 2,
    })
    
    result = df_new.collect()
    assert len(result) == 4
    assert "age_plus_10" in result[0]
    assert "age_doubled" in result[0]
    assert result[0]["age_plus_10"] == 40  # 30 + 10
    assert result[0]["age_doubled"] == 60  # 30 * 2


def test_polars_agg_dict_syntax(sample_db):
    """Test Polars-style agg() with dictionary syntax."""
    df = sample_db.table("orders").polars()
    
    # Group by status and aggregate with dict syntax
    result = df.group_by("status").agg({"amount": "sum"})
    result_list = result.collect()
    result_dict_list = _to_dict_list(result_list)
    
    assert len(result_dict_list) == 2  # active and completed
    # Should have aggregated columns
    assert any("amount" in r for r in result_dict_list)


def test_polars_agg_dict_syntax_multiple_cols(sample_db):
    """Test Polars-style agg() with dictionary syntax for multiple columns."""
    df = sample_db.table("orders").polars()
    
    # Create a DataFrame with multiple numeric columns
    df_with_total = df.select(
        col("status"),
        col("amount"),
        (col("amount") * 1.1).alias("amount_with_tax")
    )
    
    # Group by status and aggregate with dict syntax
    result = df_with_total.group_by("status").agg({
        "amount": "sum",
        "amount_with_tax": "avg"
    })
    result_list = result.collect()
    
    assert len(result_list) == 2


# ============================================================================
# Phase 6: Schema & Inspection (Pandas-style)
# ============================================================================


def test_pandas_show(sample_db, capsys):
    """Test Pandas-style show()."""
    df = sample_db.table("users").pandas()
    
    # Should print without error
    df.show(2)
    captured = capsys.readouterr()
    assert "id" in captured.out or "name" in captured.out


def test_pandas_take(sample_db):
    """Test Pandas-style take()."""
    df = sample_db.table("users").pandas()
    
    # Take first 2 rows
    rows = df.take(2)
    assert len(rows) == 2
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    assert "id" in rows[0]


def test_pandas_first(sample_db):
    """Test Pandas-style first()."""
    df = sample_db.table("users").pandas()
    
    # Get first row
    row = df.first()
    assert row is not None
    assert isinstance(row, dict)
    assert "id" in row
    assert row["id"] == 1


def test_pandas_first_empty(sample_db):
    """Test Pandas-style first() on empty DataFrame."""
    df = sample_db.table("users").pandas()
    df_empty = df.query("id > 100")
    
    # First on empty should return None
    row = df_empty.first()
    assert row is None


def test_pandas_summary(sample_db):
    """Test Pandas-style summary()."""
    df = sample_db.table("users").pandas()
    
    # Summary should return a DataFrame
    summary_df = df.summary("count", "mean")
    result = summary_df.collect()
    result_list = _to_dict_list(result)
    
    # Should have summary statistics (may be empty if no numeric columns)
    # Just check it doesn't error
    assert isinstance(result_list, list)


def test_pandas_printSchema(sample_db, capsys):
    """Test Pandas-style printSchema()."""
    df = sample_db.table("users").pandas()
    
    # Should print schema without error
    df.printSchema()
    captured = capsys.readouterr()
    assert "root" in captured.out or "id" in captured.out


# ============================================================================
# Phase 6: Schema & Inspection (Polars-style)
# ============================================================================


def test_polars_show(sample_db, capsys):
    """Test Polars-style show()."""
    df = sample_db.table("users").polars()
    
    # Should print without error
    df.show(2)
    captured = capsys.readouterr()
    assert "id" in captured.out or "name" in captured.out


def test_polars_take(sample_db):
    """Test Polars-style take()."""
    df = sample_db.table("users").polars()
    
    # Take first 2 rows
    rows = df.take(2)
    assert len(rows) == 2
    assert isinstance(rows, list)
    assert isinstance(rows[0], dict)
    assert "id" in rows[0]


def test_polars_first(sample_db):
    """Test Polars-style first()."""
    df = sample_db.table("users").polars()
    
    # Get first row
    row = df.first()
    assert row is not None
    assert isinstance(row, dict)
    assert "id" in row
    assert row["id"] == 1


def test_polars_first_empty(sample_db):
    """Test Polars-style first() on empty DataFrame."""
    df = sample_db.table("users").polars()
    df_empty = df.filter(col("id") > 100)
    
    # First on empty should return None
    row = df_empty.first()
    assert row is None


def test_polars_summary(sample_db):
    """Test Polars-style summary()."""
    df = sample_db.table("users").polars()
    
    # Summary should return a DataFrame
    summary_df = df.summary("count", "mean")
    result = summary_df.collect()
    result_list = _to_dict_list(result)
    
    # Should have summary statistics (may be empty if no numeric columns)
    # Just check it doesn't error
    assert isinstance(result_list, list)


def test_polars_printSchema(sample_db, capsys):
    """Test Polars-style printSchema()."""
    df = sample_db.table("users").polars()
    
    # Should print schema without error
    df.printSchema()
    captured = capsys.readouterr()
    assert "root" in captured.out or "id" in captured.out


# ============================================================================
# Async Tests
# ============================================================================


@pytest.mark.asyncio
async def test_async_pyspark_getitem_single_column(tmp_path):
    """Test async PySpark-style single column access."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")],
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame(
            [{"id": 1, "name": "Alice", "age": 30}, {"id": 2, "name": "Bob", "age": 25}],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    # Single column access
    col_expr = df["age"]
    assert hasattr(col_expr, "_column")
    assert hasattr(col_expr, "str")

    # Use in a filter instead
    result = await df.where(df["age"] > 25).collect()
    assert len(result) == 1  # Only Alice has age > 25

    await db.close()


@pytest.mark.asyncio
async def test_async_pandas_crossjoin(tmp_path):
    """Test async Pandas-style crossJoin()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()
    await db.create_table(
        "products", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    ).write.insertInto("users")
    await (
        await db.createDataFrame([{"id": 1, "name": "Product A"}], pk="id")
    ).write.insertInto("products")

    table1 = await db.table("users")
    table2 = await db.table("products")
    df1 = table1.pandas()
    df2 = table2.pandas()

    df_cross = df1.crossJoin(df2)
    result = await df_cross.collect()

    assert len(result) == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_polars_semi_join(tmp_path):
    """Test async Polars-style semi_join()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect, col

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()
    await db.create_table(
        "orders", [column("id", "INTEGER"), column("user_id", "INTEGER")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    ).write.insertInto("users")
    await (
        await db.createDataFrame([{"id": 1, "user_id": 1}], pk="id")
    ).write.insertInto("orders")

    table1 = await db.table("users")
    table2 = await db.table("orders")
    df1 = table1.polars()
    df2 = table2.polars()

    df_semi = df1.semi_join(df2, left_on="id", right_on="user_id")
    result = await df_semi.collect()

    assert len(result) == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_pyspark_nunique(tmp_path):
    """Test async PySpark-style nunique()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT"), column("country", "TEXT")],
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame(
            [
                {"id": 1, "name": "Alice", "country": "USA"},
                {"id": 2, "name": "Bob", "country": "UK"},
                {"id": 3, "name": "Charlie", "country": "USA"},
            ],
            pk="id",
        )
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    count = await df.nunique("country")
    assert count == 2

    await db.close()


@pytest.mark.asyncio
async def test_async_pyspark_withcolumns(tmp_path):
    """Test async PySpark-style withColumns()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect, col

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("age", "INTEGER")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "age": 30}], pk="id")
    ).write.insertInto("users")

    table_handle = await db.table("users")
    df = table_handle.select()

    df_new = df.withColumns({
        "age_plus_10": col("age") + 10,
    })

    result = await df_new.collect()
    assert len(result) == 1
    assert "age_plus_10" in result[0]
    assert result[0]["age_plus_10"] == 40

    await db.close()


@pytest.mark.asyncio
async def test_async_pandas_show(tmp_path, capsys):
    """Test async Pandas-style show()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.pandas()

    await df.show(1)
    captured = capsys.readouterr()
    assert "id" in captured.out or "name" in captured.out

    await db.close()


@pytest.mark.asyncio
async def test_async_pandas_take(tmp_path):
    """Test async Pandas-style take()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id"
        )
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.pandas()

    rows = await df.take(1)
    assert len(rows) == 1
    assert rows[0]["id"] == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_pandas_first(tmp_path):
    """Test async Pandas-style first()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.pandas()

    row = await df.first()
    assert row is not None
    assert row["id"] == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_pandas_summary(tmp_path):
    """Test async Pandas-style summary()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("age", "INTEGER")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "age": 30}], pk="id")
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.pandas()

    summary_df = await df.summary("count")
    result = await summary_df.collect()
    result_list = _to_dict_list(result)
    # Just check it doesn't error - summary may return empty if no numeric columns
    assert isinstance(result_list, list)

    await db.close()


@pytest.mark.asyncio
async def test_async_polars_show(tmp_path, capsys):
    """Test async Polars-style show()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.polars()

    await df.show(1)
    captured = capsys.readouterr()
    assert "id" in captured.out or "name" in captured.out

    await db.close()


@pytest.mark.asyncio
async def test_async_polars_take(tmp_path):
    """Test async Polars-style take()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id"
        )
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.polars()

    rows = await df.take(1)
    assert len(rows) == 1
    assert rows[0]["id"] == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_polars_first(tmp_path):
    """Test async Polars-style first()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.polars()

    row = await df.first()
    assert row is not None
    assert row["id"] == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_polars_summary(tmp_path):
    """Test async Polars-style summary()."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.table.schema import column

    await db.create_table(
        "users", [column("id", "INTEGER"), column("age", "INTEGER")]
    ).collect()

    from moltres.io.records import Records

    await (
        await db.createDataFrame([{"id": 1, "age": 30}], pk="id")
    ).write.insertInto("users")

    table = await db.table("users")
    df = table.polars()

    summary_df = await df.summary("count")
    result = await summary_df.collect()
    result_list = _to_dict_list(result)
    # Just check it doesn't error - summary may return empty if no numeric columns
    assert isinstance(result_list, list)

    await db.close()

