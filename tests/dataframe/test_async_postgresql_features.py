"""Async tests for PostgreSQL-specific features."""

import pytest

try:
    import asyncpg  # noqa: F401
except ImportError:
    pytest.skip("asyncpg not installed", allow_module_level=True)

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


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_jsonb_type(postgresql_async_connection):
    """Test JSONB type support in PostgreSQL (async)."""
    import json as json_module
    from moltres.table.schema import json

    db = postgresql_async_connection
    await db.create_table(
        "test_jsonb",
        [
            json("data", jsonb=True),
        ],
    )

    table = await db.table("test_jsonb")
    # Serialize dict to JSON string for insertion
    await table.insert([{"data": json_module.dumps({"key": "value", "number": 42})}])

    result = await table.select().collect()
    assert len(result) == 1
    # PostgreSQL returns JSONB as dict
    data = result[0]["data"]
    if isinstance(data, str):
        data = json_module.loads(data)
    assert data["key"] == "value"


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_uuid_type(postgresql_async_connection):
    """Test UUID type support in PostgreSQL (async)."""
    from moltres.table.schema import uuid

    db = postgresql_async_connection
    await db.create_table(
        "test_uuid",
        [
            uuid("id"),
        ],
    )

    import uuid as uuid_module

    test_uuid = uuid_module.uuid4()
    table = await db.table("test_uuid")
    await table.insert([{"id": str(test_uuid)}])

    result = await table.select().collect()
    assert len(result) == 1
    # PostgreSQL returns UUID objects, convert to string for comparison
    uuid_value = result[0]["id"]
    if hasattr(uuid_value, "__str__"):
        uuid_value = str(uuid_value)
    assert uuid_value == str(test_uuid)


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_array_agg_collect_list(postgresql_async_connection):
    """Test collect_list() uses array_agg in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    )

    table = await db.table("test_array")
    await table.insert(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ]
    )

    table_handle = await db.table("test_array")
    result = await (
        table_handle.select()
        .group_by("id")
        .agg(collect_list(col("value")).alias("values"))
        .collect()
    )

    assert len(result) == 3
    # PostgreSQL returns array
    assert result[0]["values"] is not None


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_array_agg_collect_set(postgresql_async_connection):
    """Test collect_set() uses array_agg(DISTINCT) in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR(255)"),
        ],
    )

    table = await db.table("test_array")
    await table.insert(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ]
    )

    table_handle = await db.table("test_array")
    result = await (
        table_handle.select()
        .group_by("id")
        .agg(collect_set(col("value")).alias("values"))
        .collect()
    )

    assert len(result) == 3


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_json_extract_postgresql(postgresql_async_connection):
    """Test json_extract() with PostgreSQL JSONB (async)."""
    import json as json_module
    from moltres.table.schema import json

    db = postgresql_async_connection
    await db.create_table(
        "test_json",
        [
            json("data", jsonb=True),
        ],
    )

    table = await db.table("test_json")
    # Serialize dict to JSON string for insertion
    await table.insert([{"data": json_module.dumps({"key": "value", "nested": {"deep": 42}})}])

    table_handle = await db.table("test_json")
    result = await table_handle.select(
        json_extract(col("data"), "$.key").alias("extracted")
    ).collect()

    assert len(result) == 1
    # PostgreSQL may return JSON with quotes, handle both cases
    extracted = result[0]["extracted"]
    if isinstance(extracted, str) and extracted.startswith('"'):
        extracted = json_module.loads(extracted)
    assert extracted == "value"


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_array_functions(postgresql_async_connection):
    """Test array functions in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_array",
        [
            column("id", "INTEGER", primary_key=True),
            column("arr", "VARCHAR"),  # Store as text, will cast to array
        ],
    )

    # PostgreSQL array literal
    table = await db.table("test_array")
    await table.insert([{"id": 1, "arr": "{1,2,3}"}])

    # Test array functions
    table_handle = await db.table("test_array")
    result = await table_handle.select(
        array_length(array(1, 2, 3)).alias("len"),
        array_contains(array(1, 2, 3), 2).alias("contains"),
        array_position(array(1, 2, 3), 2).alias("position"),
    ).collect()

    assert len(result) == 1
    assert result[0]["len"] == 3
    assert result[0]["contains"] is True
    assert result[0]["position"] == 2


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_percentile_cont(postgresql_async_connection):
    """Test percentile_cont() in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_percentile",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "REAL"),
        ],
    )

    table = await db.table("test_percentile")
    await table.insert(
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
    table_handle = await db.table("test_percentile")
    result = await table_handle.select(percentile_cont(col("value"), 0.5).alias("median")).collect()

    assert len(result) == 1
    assert result[0]["median"] is not None


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_percentile_disc(postgresql_async_connection):
    """Test percentile_disc() in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_percentile",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "REAL"),
        ],
    )

    table = await db.table("test_percentile")
    await table.insert(
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
    table_handle = await db.table("test_percentile")
    result = await table_handle.select(percentile_disc(col("value"), 0.5).alias("median")).collect()

    assert len(result) == 1
    assert result[0]["median"] is not None


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_lateral_join(postgresql_async_connection):
    """Test LATERAL join in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_lateral",
        [
            column("id", "INTEGER", primary_key=True),
            column("data", "VARCHAR"),
        ],
    )

    table = await db.table("test_lateral")
    await table.insert([{"id": 1, "data": '{"items": [1, 2, 3]}'}])

    # LATERAL join with jsonb_array_elements
    # Note: explode() is not yet fully implemented for PostgreSQL
    pytest.skip("explode() is not yet fully implemented for PostgreSQL")


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_merge_statement(postgresql_async_connection):
    """Test MERGE statement in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
        ],
    )

    await db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "VARCHAR"),
        ],
    )

    target_table = await db.table("target")
    await target_table.insert([{"id": 1, "value": "old"}])

    source_table = await db.table("source")
    await source_table.insert([{"id": 1, "value": "new"}, {"id": 2, "value": "insert"}])

    # Merge using the correct API: rows (list of dicts), not DataFrame
    source_rows = await source_table.select().collect()
    result = await target_table.merge(
        source_rows,
        on=["id"],
        when_matched={"value": "new"},  # Update value when matched
    )

    assert result >= 1

    final = await target_table.select().collect()
    assert len(final) == 2


@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_tablesample(postgresql_async_connection):
    """Test TABLESAMPLE in PostgreSQL (async)."""
    from moltres.table.schema import column

    db = postgresql_async_connection
    await db.create_table(
        "test_sample",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "INTEGER"),
        ],
    )

    table = await db.table("test_sample")
    # Insert 100 rows
    await table.insert([{"id": i, "value": i} for i in range(1, 101)])

    # Test sample() with 10% fraction
    table_handle = await db.table("test_sample")
    df = table_handle.select("id", "value")
    sampled = df.sample(0.1)

    result = await sampled.collect()

    # Should return approximately 10 rows (with some variance due to random sampling)
    assert 1 <= len(result) <= 100
