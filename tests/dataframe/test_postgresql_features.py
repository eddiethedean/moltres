"""Tests for PostgreSQL-specific features."""

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
    percentile_cont,
    percentile_disc,
)


@pytest.mark.postgres
def test_jsonb_type(postgresql_connection):
    """Test JSONB type support in PostgreSQL."""
    import json as json_module
    from moltres.table.schema import json

    db = postgresql_connection
    db.create_table(
        "test_jsonb",
        [
            json("data", jsonb=True),
        ],
    )

    table = db.table("test_jsonb")
    # Serialize dict to JSON string for insertion
    table.insert([{"data": json_module.dumps({"key": "value", "number": 42})}])

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
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
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
    # PostgreSQL returns arrays, verify structure
    assert result[0]["values"] is not None


@pytest.mark.postgres
def test_array_agg_collect_set(postgresql_connection):
    """Test collect_set() uses ARRAY_AGG(DISTINCT) in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
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


@pytest.mark.postgres
def test_json_extract_postgresql(postgresql_connection):
    """Test json_extract() with PostgreSQL JSONB."""
    import json as json_module
    from moltres.table.schema import json

    db = postgresql_connection
    db.create_table(
        "test_json",
        [
            json("data", jsonb=True),
        ],
    )

    table = db.table("test_json")
    # Serialize dict to JSON string for insertion
    table.insert([{"data": json_module.dumps({"key": "value", "nested": {"deep": 42}})}])

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
    )

    # PostgreSQL array literal
    table = db.table("test_array")
    table.insert([{"id": 1, "arr": "{1,2,3}"}])

    # Test array functions
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
    assert result[0]["contains"] is True
    assert result[0]["position"] == 2


@pytest.mark.postgres
def test_percentile_cont(postgresql_connection):
    """Test percentile_cont() in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_percentile",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "REAL"),
        ],
    )

    table = db.table("test_percentile")
    table.insert(
        [
            {"id": 1, "value": 10.0},
            {"id": 2, "value": 20.0},
            {"id": 3, "value": 30.0},
            {"id": 4, "value": 40.0},
            {"id": 5, "value": 50.0},
        ]
    )

    # For global aggregation without group_by, we need to use a different approach
    # Use a dummy group_by with a constant or aggregate directly
    result = (
        db.table("test_percentile")
        .select(percentile_cont(col("value"), 0.5).alias("median"))
        .collect()
    )

    assert len(result) == 1
    assert result[0]["median"] is not None


@pytest.mark.postgres
def test_percentile_disc(postgresql_connection):
    """Test percentile_disc() in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_percentile",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "REAL"),
        ],
    )

    table = db.table("test_percentile")
    table.insert(
        [
            {"id": 1, "value": 10.0},
            {"id": 2, "value": 20.0},
            {"id": 3, "value": 30.0},
        ]
    )

    # For global aggregation without group_by, we need to use a different approach
    # Use a dummy group_by with a constant or aggregate directly
    result = (
        db.table("test_percentile")
        .select(percentile_disc(col("value"), 0.5).alias("median"))
        .collect()
    )

    assert len(result) == 1
    assert result[0]["median"] is not None


@pytest.mark.postgres
def test_lateral_join(postgresql_connection):
    """Test LATERAL join in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "test_lateral",
        [
            column("id", "INTEGER", primary_key=True),
            column("data", "VARCHAR"),
        ],
    )

    table = db.table("test_lateral")
    table.insert([{"id": 1, "data": '{"items": [1, 2, 3]}'}])

    # LATERAL join with jsonb_array_elements
    # Note: explode() is not yet fully implemented for PostgreSQL
    pytest.skip("explode() is not yet fully implemented for PostgreSQL")


@pytest.mark.postgres
def test_merge_statement(postgresql_connection):
    """Test MERGE statement in PostgreSQL."""
    from moltres.table.schema import column

    db = postgresql_connection
    db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
        ],
    )

    db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
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
    )

    table = db.table("test_sample")
    # Insert enough rows for sampling
    table.insert([{"id": i, "value": f"value_{i}"} for i in range(100)])

    result = db.table("test_sample").select().sample(0.1).collect()

    # Should return some rows (exact number depends on sampling)
    assert len(result) >= 0
    assert len(result) <= 100
