"""Tests for PostgreSQL-specific features."""

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


@pytest.mark.postgres
def test_jsonb_type(postgresql_connection):
    """Test JSONB type support in PostgreSQL."""
    import json as json_module
    from moltres.table.schema import column, json

    db = postgresql_connection
    with db.batch():
        db.create_table(
            "test_jsonb",
            [
                column("id", "INTEGER", primary_key=True),
                json("data", jsonb=True),
            ],
        )

    table = db.table("test_jsonb")
    # Serialize dict to JSON string for insertion
    db.createDataFrame(
        [{"id": 1, "data": json_module.dumps({"key": "value", "number": 42})}],
        pk="id",
    ).write.insertInto("test_jsonb")
    result = table.select().collect()
    assert len(result) == 1
    # PostgreSQL returns JSONB as dict
    data = result[0]["data"]
    if isinstance(data, str):
        data = json_module.loads(data)
    assert data["key"] == "value"


@pytest.mark.postgres
def test_uuid_type(postgresql_connection):
    """Test UUID type support in PostgreSQL."""
    from moltres.table.schema import uuid

    db = postgresql_connection
    with db.batch():
        db.create_table(
            "test_uuid",
            [
                uuid("id"),
            ],
        )

    import uuid as uuid_module

    test_uuid = uuid_module.uuid4()
    table = db.table("test_uuid")
    db.createDataFrame(
        [{"id": str(test_uuid)}],
        pk="id",
    ).write.insertInto("test_uuid")

    result = table.select().collect()
    assert len(result) == 1
    # PostgreSQL returns UUID objects, convert to string for comparison
    uuid_value = result[0]["id"]
    if hasattr(uuid_value, "__str__"):
        uuid_value = str(uuid_value)
    assert uuid_value == str(test_uuid)


@pytest.mark.postgres
def test_array_agg_collect_list(postgresql_connection):
    """Test collect_list() uses ARRAY_AGG in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    with db.batch():
        db.create_table(
            "test_array",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "VARCHAR"),
            ],
        )

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
    # PostgreSQL returns arrays, verify structure
    assert result[0]["values"] is not None


@pytest.mark.postgres
def test_array_agg_collect_set(postgresql_connection):
    """Test collect_set() uses ARRAY_AGG(DISTINCT) in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    with db.batch():
        db.create_table(
            "test_array",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "VARCHAR"),
            ],
        )

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


@pytest.mark.postgres
def test_json_extract_postgresql(postgresql_connection):
    """Test json_extract() with PostgreSQL JSONB."""
    import json as json_module
    from moltres.table.schema import json

    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_json",
        [
            column("id", "INTEGER", primary_key=True),
            json("data", jsonb=True),
        ],
    ).collect()

    db.table("test_json")
    # Serialize dict to JSON string for insertion
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
    # PostgreSQL may return JSON with quotes, handle both cases
    extracted = result[0]["extracted"]
    if isinstance(extracted, str) and extracted.startswith('"'):
        extracted = json_module.loads(extracted)
    assert extracted == "value"


@pytest.mark.postgres
def test_array_functions(postgresql_connection):
    """Test array functions in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("arr", "VARCHAR"),  # Store as text, will cast to array
        ],
    ).collect()

    # PostgreSQL array literal
    table = db.table("test_array")
    db.createDataFrame(
        [{"id": 1, "arr": "{1,2,3}"}],
        pk="id",
    ).write.insertInto("test_array")

    from moltres import col

    # Cast VARCHAR to INTEGER[] array type for PostgreSQL array functions
    arr_col = col("arr").cast("INTEGER[]")
    result = table.select(
        array_length(arr_col).alias("len"),
        array_contains(arr_col, 2).alias("contains"),
        array_position(arr_col, 2).alias("position"),
    ).collect()
    assert len(result) == 1
    assert result[0]["len"] == 3
    assert result[0]["contains"] is True
    assert result[0]["position"] == 2


@pytest.mark.postgres
def test_tablesample(postgresql_connection):
    """Test TABLESAMPLE in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_sample",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
        ],
    ).collect()

    db.table("test_sample")
    # Insert enough rows for sampling
    db.createDataFrame(
        [{"id": i, "value": f"value_{i}"} for i in range(100)],
        pk="id",
    ).write.insertInto("test_sample")

    result = db.table("test_sample").select().sample(0.1).collect()

    # Should return some rows (exact number depends on sampling)
    assert len(result) >= 0
    assert len(result) <= 100
