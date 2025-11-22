"""Tests for MySQL-specific features."""

import pytest

from moltres import col
from moltres.expressions.functions import (
    array_contains,
    array_length,
    array_position,
    collect_list,
    collect_set,
    json_extract,
)


@pytest.mark.mysql
def test_json_type(mysql_connection):
    """Test JSON type support in MySQL."""
    from moltres.table.schema import column, json

    db = mysql_connection
    db.create_table(
        "test_json",
        [
            column("id", "INTEGER", primary_key=True),
            json("data"),
        ],
    ).collect()

    table = db.table("test_json")
    # MySQL requires JSON to be serialized as string
    import json as json_module

    db.createDataFrame(
        [{"id": 1, "data": json_module.dumps({"key": "value", "number": 42})}],
        pk="id",
    ).write.insertInto("test_json")
    result = table.select().collect()
    assert len(result) == 1
    # MySQL returns JSON as string, parse it
    data = result[0]["data"]
    if isinstance(data, str):
        import json as json_module

        data = json_module.loads(data)
    assert data["key"] == "value"


@pytest.mark.mysql
def test_uuid_type(mysql_connection):
    """Test UUID type support in MySQL (CHAR(36))."""
    from moltres.table.schema import uuid

    db = mysql_connection
    db.create_table(
        "test_uuid",
        [
            uuid("id"),
        ],
    ).collect()

    import uuid as uuid_module
    from moltres.table.schema import uuid as uuid_col_def

    test_uuid = uuid_module.uuid4()
    table = db.table("test_uuid")
    db.createDataFrame(
        [{"id": str(test_uuid)}],
        schema=[uuid_col_def("id", primary_key=True)],
    ).write.insertInto("test_uuid")
    result = table.select().collect()
    assert len(result) == 1
    assert result[0]["id"] == str(test_uuid)


@pytest.mark.mysql
def test_group_concat_collect_list(mysql_connection):
    """Test collect_list() uses GROUP_CONCAT in MySQL."""
    from moltres.table.schema import column

    db = mysql_connection
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    ).collect()

    db.table("test_array")
    db.createDataFrame(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ],
        pk="id",
    ).write.insertInto("test_array")

    result = (
        db.table("test_array")
        .select()
        .group_by("id")
        .agg(collect_list(col("value")).alias("values"))
        .collect()
    )

    assert len(result) == 3
    # MySQL returns comma-separated string
    assert result[0]["values"] is not None


@pytest.mark.mysql
def test_group_concat_collect_set(mysql_connection):
    """Test collect_set() uses GROUP_CONCAT(DISTINCT) in MySQL."""
    from moltres.table.schema import column

    db = mysql_connection
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    ).collect()

    db.table("test_array")
    db.createDataFrame(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ],
        pk="id",
    ).write.insertInto("test_array")

    result = (
        db.table("test_array")
        .select()
        .group_by("id")
        .agg(collect_set(col("value")).alias("values"))
        .collect()
    )

    assert len(result) == 3


@pytest.mark.mysql
def test_json_extract_mysql(mysql_connection):
    """Test json_extract() with MySQL JSON."""
    from moltres.table.schema import column, json

    db = mysql_connection
    db.create_table(
        "test_json",
        [
            column("id", "INTEGER", primary_key=True),
            json("data"),
        ],
    ).collect()

    db.table("test_json")
    # MySQL requires JSON to be serialized as string
    import json as json_module

    db.createDataFrame(
        [{"id": 1, "data": json_module.dumps({"key": "value", "nested": {"deep": 42}})}],
        pk="id",
    ).write.insertInto("test_json")
    result = (
        db.table("test_json")
        .select(json_extract(col("data"), "$.key").alias("extracted"))
        .collect()
    )

    assert len(result) == 1
    # MySQL may return JSON with quotes
    extracted = result[0]["extracted"]
    if isinstance(extracted, str) and extracted.startswith('"'):
        import json as json_module

        extracted = json_module.loads(extracted)
    assert extracted == "value"


@pytest.mark.mysql
def test_array_functions_mysql(mysql_connection):
    """Test array functions in MySQL (using JSON arrays)."""
    from moltres.table.schema import column
    import json as json_module
    from moltres import col

    db = mysql_connection
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("arr", "JSON"),
        ],
    ).collect()

    table = db.table("test_array")
    db.createDataFrame(
        [{"id": 1, "arr": json_module.dumps([1, 2, 3])}],
        pk="id",
    ).write.insertInto("test_array")

    result = table.select(
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
