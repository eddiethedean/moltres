"""Tests for snake_case aliases of camelCase methods."""

from __future__ import annotations

import pytest

from moltres import col, column, connect, async_connect
from moltres.expressions.window import Window
from moltres.io.records import Records, AsyncRecords


@pytest.fixture
def db(tmp_path):
    """Create an in-memory database for testing."""
    return connect("sqlite:///:memory:")


@pytest.fixture
def sample_data(db):
    """Create sample data for testing."""
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("name", "TEXT"),
            column("category", "TEXT"),
        ],
    ).collect()
    records = Records(
        _data=[
            {"id": 1, "amount": 100.0, "name": "Alice", "category": "A"},
            {"id": 2, "amount": 200.0, "name": "Bob", "category": "A"},
            {"id": 3, "amount": 150.0, "name": "Charlie", "category": "B"},
        ],
        _database=db,
    )
    records.insert_into("orders")
    return db


# DataFrame snake_case alias tests


def test_select_expr_alias(sample_data):
    """Test that select_expr() works as alias for selectExpr()."""
    db = sample_data
    df1 = db.table("orders").select().selectExpr("id", "amount * 1.1 as total")
    df2 = db.table("orders").select().select_expr("id", "amount * 1.1 as total")
    results1 = df1.collect()
    results2 = df2.collect()
    assert len(results1) == len(results2) == 3
    assert abs(results1[0]["total"] - results2[0]["total"]) < 0.01
    assert abs(results1[0]["total"] - 110.0) < 0.01


def test_cross_join_alias(sample_data):
    """Test that cross_join() works as alias for crossJoin()."""
    db = sample_data
    df1 = (
        db.table("orders")
        .select("id", "name")
        .withColumnRenamed("id", "id1")
        .withColumnRenamed("name", "name1")
    )
    df2 = (
        db.table("orders")
        .select("id", "name")
        .withColumnRenamed("id", "id2")
        .withColumnRenamed("name", "name2")
    )
    result1 = df1.crossJoin(df2)
    result2 = df1.cross_join(df2)
    assert len(result1.collect()) == len(result2.collect()) == 9


def test_union_all_alias(sample_data):
    """Test that union_all() works as alias for unionAll()."""
    db = sample_data
    df1 = db.table("orders").select().where(col("id") == 1)
    df2 = db.table("orders").select().where(col("id") == 2)
    result1 = df1.unionAll(df2)
    result2 = df1.union_all(df2)
    assert len(result1.collect()) == len(result2.collect()) == 2


def test_drop_duplicates_alias(sample_data):
    """Test that drop_duplicates() works as alias for dropDuplicates()."""
    db = sample_data
    df = db.table("orders").select()
    result1 = df.dropDuplicates()
    result2 = df.drop_duplicates()
    assert len(result1.collect()) == len(result2.collect()) == 3


def test_with_column_alias(sample_data):
    """Test that with_column() works as alias for withColumn()."""
    db = sample_data
    df = db.table("orders").select()
    result1 = df.withColumn("amount_with_tax", col("amount") * 1.1)
    result2 = df.with_column("amount_with_tax", col("amount") * 1.1)
    results1 = result1.collect()
    results2 = result2.collect()
    assert len(results1) == len(results2) == 3
    assert abs(results1[0]["amount_with_tax"] - results2[0]["amount_with_tax"]) < 0.01
    assert abs(results1[0]["amount_with_tax"] - 110.0) < 0.01


def test_with_columns_alias(sample_data):
    """Test that with_columns() works as alias for withColumns()."""
    db = sample_data
    df = db.table("orders").select()
    cols_map = {
        "amount_with_tax": col("amount") * 1.1,
        "amount_doubled": col("amount") * 2,
    }
    result1 = df.withColumns(cols_map)
    result2 = df.with_columns(cols_map)
    results1 = result1.collect()
    results2 = result2.collect()
    assert len(results1) == len(results2) == 3
    assert abs(results1[0]["amount_with_tax"] - results2[0]["amount_with_tax"]) < 0.01
    assert abs(results1[0]["amount_with_tax"] - 110.0) < 0.01
    assert abs(results1[0]["amount_doubled"] - results2[0]["amount_doubled"]) < 0.01
    assert abs(results1[0]["amount_doubled"] - 200.0) < 0.01


def test_with_column_renamed_alias(sample_data):
    """Test that with_column_renamed() works as alias for withColumnRenamed()."""
    db = sample_data
    df = db.table("orders").select()
    result1 = df.withColumnRenamed("name", "user_name")
    result2 = df.with_column_renamed("name", "user_name")
    results1 = result1.collect()
    results2 = result2.collect()
    assert len(results1) == len(results2) == 3
    assert "user_name" in results1[0]
    assert "user_name" in results2[0]
    assert results1[0]["user_name"] == results2[0]["user_name"] == "Alice"


def test_print_schema_alias(sample_data, capsys):
    """Test that print_schema() works as alias for printSchema()."""
    db = sample_data
    df = db.table("orders").select()
    df.printSchema()
    output1 = capsys.readouterr().out
    df.print_schema()
    output2 = capsys.readouterr().out
    # Both should produce similar output
    assert "root" in output1
    assert "root" in output2


# Database create_dataframe alias tests


def test_create_dataframe_alias(db):
    """Test that create_dataframe() works as alias for createDataFrame()."""
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    df1 = db.createDataFrame(data, pk="id")
    df2 = db.create_dataframe(data, pk="id")
    results1 = df1.collect()
    results2 = df2.collect()
    assert len(results1) == len(results2) == 2
    assert results1[0]["name"] == results2[0]["name"] == "Alice"


# Writer snake_case alias tests


def test_partition_by_alias(sample_data):
    """Test that partition_by() works as alias for partitionBy()."""
    db = sample_data
    df = db.table("orders").select()
    writer1 = df.write.partitionBy("category")
    writer2 = df.write.partition_by("category")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


def test_primary_key_alias(sample_data):
    """Test that primary_key() works as alias for primaryKey()."""
    db = sample_data
    df = db.table("orders").select()
    writer1 = df.write.primaryKey("id")
    writer2 = df.write.primary_key("id")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


def test_bucket_by_alias(sample_data):
    """Test that bucket_by() works as alias for bucketBy()."""
    db = sample_data
    df = db.table("orders").select()
    writer1 = df.write.bucketBy(10, "category")
    writer2 = df.write.bucket_by(10, "category")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


def test_sort_by_alias(sample_data):
    """Test that sort_by() works as alias for sortBy()."""
    db = sample_data
    df = db.table("orders").select()
    writer1 = df.write.sortBy("amount")
    writer2 = df.write.sort_by("amount")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


def test_insert_into_alias(sample_data):
    """Test that insert_into() works as alias for insertInto()."""
    db = sample_data
    # Create target table
    db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("name", "TEXT"),
            column("category", "TEXT"),
        ],
    ).collect()
    # Test both methods with different data
    df1 = db.table("orders").select().where(col("id") == 1)
    df1.write.insertInto("target")
    count1 = db.table("target").select().collect()
    df2 = db.table("orders").select().where(col("id") == 2)
    df2.write.insert_into("target")
    count2 = db.table("target").select().collect()
    assert len(count2) == len(count1) + 1


# Reader text_file alias tests


def test_text_file_alias(tmp_path):
    """Test that text_file() works as alias for textFile()."""
    # Create a text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("line1\nline2\nline3")
    db = connect("sqlite:///:memory:")
    df1 = db.read.textFile(str(text_file))
    df2 = db.read.text_file(str(text_file))
    results1 = df1.collect()
    results2 = df2.collect()
    assert len(results1) == len(results2) == 3
    assert results1[0]["value"] == results2[0]["value"] == "line1"


# WindowSpec snake_case alias tests


def test_window_partition_by_alias(sample_data):
    """Test that partition_by() works as alias for partitionBy() on Window."""
    # Test static method on Window class (instance methods conflict with dataclass fields)
    window1 = Window.partitionBy("category")
    window2 = Window.partition_by("category")
    # Both should create WindowSpec instances
    assert type(window1) is type(window2)
    # Access field via __dict__ since methods shadow fields, compare lengths and string representations
    field1 = window1.__dict__["partition_by"]
    field2 = window2.__dict__["partition_by"]
    assert len(field1) == len(field2) == 1
    assert str(field1[0]) == str(field2[0])


def test_window_order_by_alias(sample_data):
    """Test that order_by() works as alias for orderBy() on Window."""
    # Test static method on Window class (instance methods conflict with dataclass fields)
    window1 = Window.orderBy("amount")
    window2 = Window.order_by("amount")
    assert type(window1) is type(window2)
    # Access field via __dict__ since methods shadow fields, compare lengths and string representations
    field1 = window1.__dict__["order_by"]
    field2 = window2.__dict__["order_by"]
    assert len(field1) == len(field2) == 1
    assert str(field1[0]) == str(field2[0])


def test_window_rows_between_alias():
    """Test that rows_between() works as alias for rowsBetween() on Window."""
    # Test static method on Window class (instance methods conflict with dataclass fields)
    window1 = Window.rowsBetween(0, 1)
    window2 = Window.rows_between(0, 1)
    assert type(window1) is type(window2)
    # Access field via __dict__ since methods shadow fields
    assert window1.__dict__["rows_between"] == window2.__dict__["rows_between"] == (0, 1)


def test_window_range_between_alias():
    """Test that range_between() works as alias for rangeBetween() on Window."""
    # Test static method on Window class (instance methods conflict with dataclass fields)
    window1 = Window.rangeBetween(0, 1)
    window2 = Window.range_between(0, 1)
    assert type(window1) is type(window2)
    # Access field via __dict__ since methods shadow fields
    assert window1.__dict__["range_between"] == window2.__dict__["range_between"] == (0, 1)


# Test method chaining with snake_case aliases


def test_snake_case_chaining(sample_data):
    """Test that snake_case aliases work with method chaining."""
    db = sample_data
    df = (
        db.table("orders")
        .select()
        .select_expr("id", "amount * 1.1 as total")
        .with_column("doubled", col("total") * 2)
        .with_column_renamed("doubled", "double_total")
        .where(col("id") == 1)
    )
    results = df.collect()
    assert len(results) == 1
    assert "double_total" in results[0]
    assert abs(results[0]["double_total"] - 220.0) < 0.01


# AsyncDataFrame snake_case alias tests


@pytest.fixture
def async_db(tmp_path):
    """Create an async in-memory database for testing."""
    return async_connect("sqlite+aiosqlite:///:memory:")


@pytest.fixture
async def async_sample_data(async_db):
    """Create sample data for async testing."""
    await async_db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("name", "TEXT"),
            column("category", "TEXT"),
        ],
    ).collect()
    records = AsyncRecords(
        _data=[
            {"id": 1, "amount": 100.0, "name": "Alice", "category": "A"},
            {"id": 2, "amount": 200.0, "name": "Bob", "category": "A"},
            {"id": 3, "amount": 150.0, "name": "Charlie", "category": "B"},
        ],
        _database=async_db,
    )
    await records.insert_into("orders")
    return async_db


@pytest.mark.asyncio
async def test_async_select_expr_alias(async_sample_data):
    """Test that select_expr() works as alias for selectExpr() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df1 = table_handle.select().selectExpr("id", "amount * 1.1 as total")
    df2 = table_handle.select().select_expr("id", "amount * 1.1 as total")
    results1 = await df1.collect()
    results2 = await df2.collect()
    assert len(results1) == len(results2) == 3
    assert abs(results1[0]["total"] - results2[0]["total"]) < 0.01
    assert abs(results1[0]["total"] - 110.0) < 0.01


@pytest.mark.asyncio
async def test_async_cross_join_alias(async_sample_data):
    """Test that cross_join() works as alias for crossJoin() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df1 = (
        table_handle.select("id", "name")
        .withColumnRenamed("id", "id1")
        .withColumnRenamed("name", "name1")
    )
    df2 = (
        table_handle.select("id", "name")
        .withColumnRenamed("id", "id2")
        .withColumnRenamed("name", "name2")
    )
    result1 = df1.crossJoin(df2)
    result2 = df1.cross_join(df2)
    assert len(await result1.collect()) == len(await result2.collect()) == 9


@pytest.mark.asyncio
async def test_async_union_all_alias(async_sample_data):
    """Test that union_all() works as alias for unionAll() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df1 = table_handle.select().where(col("id") == 1)
    df2 = table_handle.select().where(col("id") == 2)
    result1 = df1.unionAll(df2)
    result2 = df1.union_all(df2)
    assert len(await result1.collect()) == len(await result2.collect()) == 2


@pytest.mark.asyncio
async def test_async_drop_duplicates_alias(async_sample_data):
    """Test that drop_duplicates() works as alias for dropDuplicates() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    result1 = df.dropDuplicates()
    result2 = df.drop_duplicates()
    assert len(await result1.collect()) == len(await result2.collect()) == 3


@pytest.mark.asyncio
async def test_async_with_column_alias(async_sample_data):
    """Test that with_column() works as alias for withColumn() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    result1 = df.withColumn("amount_with_tax", col("amount") * 1.1)
    result2 = df.with_column("amount_with_tax", col("amount") * 1.1)
    results1 = await result1.collect()
    results2 = await result2.collect()
    assert len(results1) == len(results2) == 3
    assert abs(results1[0]["amount_with_tax"] - results2[0]["amount_with_tax"]) < 0.01
    assert abs(results1[0]["amount_with_tax"] - 110.0) < 0.01


@pytest.mark.asyncio
async def test_async_with_columns_alias(async_sample_data):
    """Test that with_columns() works as alias for withColumns() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    cols_map = {
        "amount_with_tax": col("amount") * 1.1,
        "amount_doubled": col("amount") * 2,
    }
    result1 = df.withColumns(cols_map)
    result2 = df.with_columns(cols_map)
    results1 = await result1.collect()
    results2 = await result2.collect()
    assert len(results1) == len(results2) == 3
    assert abs(results1[0]["amount_with_tax"] - results2[0]["amount_with_tax"]) < 0.01
    assert abs(results1[0]["amount_with_tax"] - 110.0) < 0.01
    assert abs(results1[0]["amount_doubled"] - results2[0]["amount_doubled"]) < 0.01
    assert abs(results1[0]["amount_doubled"] - 200.0) < 0.01


@pytest.mark.asyncio
async def test_async_with_column_renamed_alias(async_sample_data):
    """Test that with_column_renamed() works as alias for withColumnRenamed() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    result1 = df.withColumnRenamed("name", "user_name")
    result2 = df.with_column_renamed("name", "user_name")
    results1 = await result1.collect()
    results2 = await result2.collect()
    assert len(results1) == len(results2) == 3
    assert "user_name" in results1[0]
    assert "user_name" in results2[0]
    assert results1[0]["user_name"] == results2[0]["user_name"] == "Alice"


@pytest.mark.asyncio
async def test_async_print_schema_alias(async_sample_data, capsys):
    """Test that print_schema() works as alias for printSchema() on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    df.printSchema()
    output1 = capsys.readouterr().out
    df.print_schema()
    output2 = capsys.readouterr().out
    # Both should produce similar output
    assert "root" in output1
    assert "root" in output2


# AsyncDataFrameWriter snake_case alias tests


@pytest.mark.asyncio
async def test_async_partition_by_alias(async_sample_data):
    """Test that partition_by() works as alias for partitionBy() on AsyncDataFrameWriter."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    writer1 = df.write.partitionBy("category")
    writer2 = df.write.partition_by("category")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


@pytest.mark.asyncio
async def test_async_primary_key_alias(async_sample_data):
    """Test that primary_key() works as alias for primaryKey() on AsyncDataFrameWriter."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    writer1 = df.write.primaryKey("id")
    writer2 = df.write.primary_key("id")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


@pytest.mark.asyncio
async def test_async_bucket_by_alias(async_sample_data):
    """Test that bucket_by() works as alias for bucketBy() on AsyncDataFrameWriter."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    writer1 = df.write.bucketBy(10, "category")
    writer2 = df.write.bucket_by(10, "category")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


@pytest.mark.asyncio
async def test_async_sort_by_alias(async_sample_data):
    """Test that sort_by() works as alias for sortBy() on AsyncDataFrameWriter."""
    db = async_sample_data
    table_handle = await db.table("orders")
    df = table_handle.select()
    writer1 = df.write.sortBy("amount")
    writer2 = df.write.sort_by("amount")
    # Both should return the same writer type
    assert type(writer1) is type(writer2)


@pytest.mark.asyncio
async def test_async_insert_into_alias(async_sample_data):
    """Test that insert_into() works as alias for insertInto() on AsyncDataFrameWriter."""
    db = async_sample_data
    # Create target table
    await db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("name", "TEXT"),
            column("category", "TEXT"),
        ],
    ).collect()
    table_handle = await db.table("orders")
    # Test both methods with different data
    df1 = table_handle.select().where(col("id") == 1)
    await df1.write.insertInto("target")
    target_handle = await db.table("target")
    count1 = await target_handle.select().collect()
    df2 = table_handle.select().where(col("id") == 2)
    await df2.write.insert_into("target")
    count2 = await target_handle.select().collect()
    assert len(count2) == len(count1) + 1


@pytest.mark.asyncio
async def test_async_create_dataframe_alias(async_db):
    """Test that create_dataframe() works as alias for createDataFrame() on AsyncDatabase."""
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    df1 = await async_db.createDataFrame(data, pk="id")
    df2 = await async_db.create_dataframe(data, pk="id")
    results1 = await df1.collect()
    results2 = await df2.collect()
    assert len(results1) == len(results2) == 2
    assert results1[0]["name"] == results2[0]["name"] == "Alice"


@pytest.mark.asyncio
async def test_async_text_file_alias(tmp_path):
    """Test that text_file() works as alias for textFile() on AsyncDataLoader."""
    # Create a text file
    text_file = tmp_path / "test_async.txt"
    text_file.write_text("line1\nline2\nline3")
    db = async_connect("sqlite+aiosqlite:///:memory:")
    df1 = await db.read.textFile(str(text_file))
    df2 = await db.read.text_file(str(text_file))
    results1 = await df1.collect()
    results2 = await df2.collect()
    assert len(results1) == len(results2) == 3
    assert results1[0]["value"] == results2[0]["value"] == "line1"
    await db.close()


@pytest.mark.asyncio
async def test_async_snake_case_chaining(async_sample_data):
    """Test that snake_case aliases work with method chaining on AsyncDataFrame."""
    db = async_sample_data
    table_handle = await db.table("orders")
    # Chain multiple snake_case methods together - use selectExpr to create computed column
    df = (
        table_handle.select()
        .select_expr("id", "amount * 1.1 as total", "amount * 2.2 as double_total")
        .where(col("id") == 1)
    )
    results = await df.collect()
    assert len(results) == 1
    assert "double_total" in results[0]
    assert abs(results[0]["double_total"] - 220.0) < 0.01
