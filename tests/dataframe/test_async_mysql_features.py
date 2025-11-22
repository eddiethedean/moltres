"""Async tests for MySQL-specific features."""

import pytest

try:
    import aiomysql  # noqa: F401
except ImportError:
    pytest.skip("aiomysql not installed", allow_module_level=True)

from moltres import col
from moltres.expressions.functions import (
    array_contains,
    array_length,
    array_position,
    collect_list,
    collect_set,
    json_extract,
)


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_json_type(mysql_async_connection):
    """Test JSON type support in MySQL (async)."""
    import json as json_module
    from moltres.table.schema import json

    db = mysql_async_connection
    from moltres.table.schema import column

    await db.create_table(
        "test_json",
        [
            column("id", "INTEGER", primary_key=True),
            json("data"),
        ],
    ).collect()

    table = await db.table("test_json")
    # MySQL requires JSON to be serialized as string
    await (
        await db.createDataFrame(
            [{"id": 1, "data": json_module.dumps({"key": "value", "number": 42})}],
            pk="id",
        )
    ).write.insertInto("test_json")
    result = await table.select().collect()
    assert len(result) == 1
    # MySQL returns JSON as string, parse it
    data = result[0]["data"]
    if isinstance(data, str):
        data = json_module.loads(data)
    assert data["key"] == "value"


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_uuid_type(mysql_async_connection):
    """Test UUID type support in MySQL (async) (CHAR(36))."""
    from moltres.table.schema import uuid

    db = mysql_async_connection
    await db.create_table(
        "test_uuid",
        [
            uuid("id"),
        ],
    ).collect()

    import uuid as uuid_module
    from moltres.table.schema import uuid as uuid_col_def

    test_uuid = uuid_module.uuid4()
    table = await db.table("test_uuid")
    await (
        await db.createDataFrame(
            [{"id": str(test_uuid)}],
            schema=[uuid_col_def("id", primary_key=True)],
        )
    ).write.insertInto("test_uuid")
    result = await table.select().collect()
    assert len(result) == 1
    assert result[0]["id"] == str(test_uuid)


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_group_concat_collect_list(mysql_async_connection):
    """Test collect_list() uses GROUP_CONCAT in MySQL (async)."""
    from moltres.table.schema import column

    db = mysql_async_connection
    await db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    ).collect()

    await db.table("test_array")
    await (
        await db.createDataFrame(
            [
                {"id": 1, "value": "a"},
                {"id": 2, "value": "b"},
                {"id": 3, "value": "a"},
            ],
            pk="id",
        )
    ).write.insertInto("test_array")

    table_handle = await db.table("test_array")
    result = await (
        table_handle.select()
        .group_by("id")
        .agg(collect_list(col("value")).alias("values"))
        .collect()
    )

    assert len(result) == 3
    # MySQL returns comma-separated string
    assert result[0]["values"] is not None


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_group_concat_collect_set(mysql_async_connection):
    """Test collect_set() uses GROUP_CONCAT(DISTINCT) in MySQL (async)."""
    from moltres.table.schema import column

    db = mysql_async_connection
    await db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    ).collect()

    await db.table("test_array")
    await (
        await db.createDataFrame(
            [
                {"id": 1, "value": "a"},
                {"id": 2, "value": "b"},
                {"id": 3, "value": "a"},
            ],
            pk="id",
        )
    ).write.insertInto("test_array")

    table_handle = await db.table("test_array")
    result = await (
        table_handle.select()
        .group_by("id")
        .agg(collect_set(col("value")).alias("values"))
        .collect()
    )

    assert len(result) == 3


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_json_extract_mysql(mysql_async_connection):
    """Test json_extract() with MySQL JSON (async)."""
    import json as json_module
    from moltres.table.schema import json

    db = mysql_async_connection
    from moltres.table.schema import column

    await db.create_table(
        "test_json",
        [
            column("id", "INTEGER", primary_key=True),
            json("data"),
        ],
    ).collect()

    await db.table("test_json")
    # MySQL requires JSON to be serialized as string
    await (
        await db.createDataFrame(
            [{"id": 1, "data": json_module.dumps({"key": "value", "nested": {"deep": 42}})}],
            pk="id",
        )
    ).write.insertInto("test_json")

    table_handle = await db.table("test_json")
    result = await table_handle.select(
        json_extract(col("data"), "$.key").alias("extracted")
    ).collect()

    assert len(result) == 1
    # MySQL may return JSON with quotes
    extracted = result[0]["extracted"]
    if isinstance(extracted, str) and extracted.startswith('"'):
        extracted = json_module.loads(extracted)
    assert extracted == "value"


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_array_functions_mysql(mysql_async_connection):
    """Test array functions in MySQL (async) (using JSON arrays)."""
    from moltres.table.schema import column
    import json as json_module
    from moltres import col

    db = mysql_async_connection
    await db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("arr", "JSON"),
        ],
    ).collect()

    table = await db.table("test_array")
    await (
        await db.createDataFrame(
            [{"id": 1, "arr": json_module.dumps([1, 2, 3])}],
            pk="id",
        )
    ).write.insertInto("test_array")

    result = await table.select(
        array_length(col("arr")).alias("len"),
        array_contains(col("arr"), 2).alias("contains"),
        array_position(col("arr"), 2).alias("position"),
    ).collect()
    assert len(result) == 1
    assert result[0]["len"] == 3
    # MySQL returns 1 for true, 0 for false (not boolean)
    assert result[0]["contains"] == 1 or result[0]["contains"] is True
    # array_position is not fully implemented for MySQL (returns NULL)
    # MySQL's JSON_SEARCH returns a path string like "$[0]", not an index
    assert result[0]["position"] is None


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_date_add_mysql(mysql_async_connection):
    """Test date_add() function in MySQL (async)."""
    from moltres import col
    from moltres.expressions.functions import date_add
    from moltres.table.schema import column

    db = mysql_async_connection
    await db.create_table(
        "test_dates",
        [
            column("id", "INTEGER", primary_key=True),
            column("date_col", "DATE"),
        ],
    ).collect()

    table = await db.table("test_dates")
    await (
        await db.createDataFrame(
            [{"id": 1, "date_col": "2024-01-01"}],
            pk="id",
        )
    ).write.insertInto("test_dates")

    result = await table.select(date_add(col("date_col"), "1 DAY").alias("next_day")).collect()
    assert len(result) == 1
    assert result[0]["next_day"] is not None


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_date_sub_mysql(mysql_async_connection):
    """Test date_sub() function in MySQL (async)."""
    from moltres import col
    from moltres.expressions.functions import date_sub
    from moltres.table.schema import column

    db = mysql_async_connection
    await db.create_table(
        "test_dates",
        [
            column("id", "INTEGER", primary_key=True),
            column("date_col", "DATE"),
        ],
    ).collect()

    table = await db.table("test_dates")
    await (
        await db.createDataFrame(
            [{"id": 1, "date_col": "2024-01-02"}],
            pk="id",
        )
    ).write.insertInto("test_dates")

    result = await table.select(date_sub(col("date_col"), "1 DAY").alias("prev_day")).collect()
    assert len(result) == 1
    assert result[0]["prev_day"] is not None


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_lateral_join_mysql(mysql_async_connection):
    """Test LATERAL join in MySQL 8.0+ (async)."""
    from moltres.table.schema import column

    db = mysql_async_connection
    await db.create_table(
        "test_lateral",
        [
            column("id", "INTEGER", primary_key=True),
            column("data", "JSON"),
        ],
    ).collect()

    await db.table("test_lateral")
    # MySQL requires JSON arrays to be serialized as string
    import json as json_module

    await (
        await db.createDataFrame(
            [{"id": 1, "data": json_module.dumps([1, 2, 3])}],
            pk="id",
        )
    ).write.insertInto("test_lateral")
    # LATERAL join with JSON_TABLE (MySQL 8.0+)
    # Note: explode() is not yet fully implemented
    pytest.skip("explode() is not yet fully implemented for MySQL")


@pytest.mark.asyncio
@pytest.mark.mysql
async def test_async_on_duplicate_key_update(mysql_async_connection):
    """Test INSERT ... ON DUPLICATE KEY UPDATE in MySQL (async)."""
    from moltres.table.schema import column

    db = mysql_async_connection
    await db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    ).collect()

    await db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    ).collect()

    source_table = await db.table("source")
    await (
        await db.createDataFrame(
            [{"id": 1, "value": "new"}],
            pk="id",
        )
    ).write.insertInto("source")

    target_table = await db.table("target")
    await (
        await db.createDataFrame(
            [{"id": 1, "value": "old"}],
            pk="id",
        )
    ).write.insertInto("target")

    # Merge using the correct API: rows (list of dicts), not DataFrame
    from moltres.table.async_mutations import merge_rows_async

    source_rows = await source_table.select().collect()
    result = await merge_rows_async(
        target_table,
        source_rows,
        on=["id"],
        when_matched={"value": "new"},  # Update value when matched
    )

    assert result >= 1

    final = await target_table.select().collect()
    assert len(final) == 1  # Merge updated existing row, didn't insert new one
    assert final[0]["value"] == "new"
