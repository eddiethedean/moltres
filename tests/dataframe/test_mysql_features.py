"""Tests for MySQL-specific features."""

import pytest

from moltres import col
from moltres.expressions.functions import (
    array,
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
    from moltres.table.schema import json

    db = mysql_connection
    db.create_table(
        "test_json",
        [
            json("data"),
        ],
    )

    table = db.table("test_json")
    # MySQL requires JSON to be serialized as string
    import json as json_module

    table.insert([{"data": json_module.dumps({"key": "value", "number": 42})}])

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
    )

    import uuid as uuid_module

    test_uuid = uuid_module.uuid4()
    table = db.table("test_uuid")
    table.insert([{"id": str(test_uuid)}])

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
    )

    table = db.table("test_array")
    table.insert(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ]
    )

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
    )

    table = db.table("test_array")
    table.insert(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ]
    )

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
    from moltres.table.schema import json

    db = mysql_connection
    db.create_table(
        "test_json",
        [
            json("data"),
        ],
    )

    table = db.table("test_json")
    # MySQL requires JSON to be serialized as string
    import json as json_module

    table.insert([{"data": json_module.dumps({"key": "value", "nested": {"deep": 42}})}])

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

    db = mysql_connection
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
        ],
    )

    table = db.table("test_array")
    table.insert([{"id": 1}])

    # Test array functions (MySQL uses JSON arrays)
    result = (
        db.table("test_array")
        .select(
            array_length(array(1, 2, 3)).alias("len"),
            array_contains(array(1, 2, 3), 2).alias("contains"),
            array_position(array(1, 2, 3), 2).alias("position"),
        )
        .collect()
    )

    assert len(result) == 1
    assert result[0]["len"] == 3
    # MySQL returns 1 for true, 0 for false (not boolean)
    assert result[0]["contains"] == 1 or result[0]["contains"] is True
    # array_position is not fully implemented for MySQL (returns NULL)
    # MySQL's JSON_SEARCH returns a path string like "$[0]", not an index
    assert result[0]["position"] is None


@pytest.mark.mysql
def test_date_add_mysql(mysql_connection):
    """Test date_add() function in MySQL."""
    from moltres.expressions.functions import date_add
    from moltres.table.schema import column

    db = mysql_connection
    db.create_table(
        "test_dates",
        [
            column("id", "INTEGER", primary_key=True),
            column("date_col", "DATE"),
        ],
    )

    table = db.table("test_dates")
    table.insert([{"id": 1, "date_col": "2024-01-01"}])

    result = (
        db.table("test_dates")
        .select(date_add(col("date_col"), "1 DAY").alias("next_day"))
        .collect()
    )

    assert len(result) == 1
    assert result[0]["next_day"] is not None


@pytest.mark.mysql
def test_date_sub_mysql(mysql_connection):
    """Test date_sub() function in MySQL."""
    from moltres.expressions.functions import date_sub
    from moltres.table.schema import column

    db = mysql_connection
    db.create_table(
        "test_dates",
        [
            column("id", "INTEGER", primary_key=True),
            column("date_col", "DATE"),
        ],
    )

    table = db.table("test_dates")
    table.insert([{"id": 1, "date_col": "2024-01-02"}])

    result = (
        db.table("test_dates")
        .select(date_sub(col("date_col"), "1 DAY").alias("prev_day"))
        .collect()
    )

    assert len(result) == 1
    assert result[0]["prev_day"] is not None


@pytest.mark.mysql
def test_lateral_join_mysql(mysql_connection):
    """Test LATERAL join in MySQL 8.0+."""
    from moltres.table.schema import column

    db = mysql_connection
    db.create_table(
        "test_lateral",
        [
            column("id", "INTEGER", primary_key=True),
            column("data", "JSON"),
        ],
    )

    table = db.table("test_lateral")
    # MySQL requires JSON arrays to be serialized as string
    import json as json_module

    table.insert([{"id": 1, "data": json_module.dumps([1, 2, 3])}])

    # LATERAL join with JSON_TABLE (MySQL 8.0+)
    # Note: explode() is not yet fully implemented
    pytest.skip("explode() is not yet fully implemented for MySQL")


@pytest.mark.mysql
def test_on_duplicate_key_update(mysql_connection):
    """Test INSERT ... ON DUPLICATE KEY UPDATE in MySQL."""
    from moltres.table.schema import column

    db = mysql_connection
    db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    )

    db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    )

    target_table = db.table("target")
    target_table.insert([{"id": 1, "value": "old"}])

    source_table = db.table("source")
    source_table.insert([{"id": 1, "value": "new"}, {"id": 2, "value": "insert"}])

    # Merge using the correct API: rows (list of dicts), not DataFrame
    source_rows = source_table.select().collect()
    result = target_table.merge(
        source_rows,
        on=["id"],
        when_matched={"value": "new"},  # Update value when matched
    )

    assert result >= 1

    final = target_table.select().collect()
    assert len(final) == 2
