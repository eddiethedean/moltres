"""Tests for async data loading operations."""

import json

import pytest

from moltres import async_connect, column
from moltres.io.records import AsyncLazyRecords, AsyncRecords
from moltres.table.schema import ColumnDef


@pytest.mark.asyncio
async def test_async_load_csv(tmp_path):
    """Test async CSV loading returns AsyncDataFrame."""
    db_path = tmp_path / "async_load_csv.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")

    # Load CSV - returns AsyncDataFrame
    df = await db.load.csv(str(csv_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_load_json(tmp_path):
    """Test async JSON loading returns AsyncDataFrame."""
    db_path = tmp_path / "async_load_json.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], f)

    # Load JSON - returns AsyncDataFrame
    df = await db.load.json(str(json_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_records_iteration(tmp_path):
    """Test that AsyncRecords can be async iterated."""
    db_path = tmp_path / "async_records_iter.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Use read.records for AsyncLazyRecords (returns synchronously)
    records = db.read.records.csv(str(csv_path))

    # Async iteration
    rows = []
    async for row in records:
        rows.append(row)

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_records_insert_into(tmp_path):
    """Test that AsyncRecords can be inserted into a table."""
    db_path = tmp_path / "async_records_insert.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Use read.records for AsyncLazyRecords (returns synchronously)
    records = db.read.records.csv(str(csv_path))

    # Create table
    await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert records
    count = await records.insert_into("target")
    assert count == 2

    # Verify
    table_handle = await db.table("target")
    df = table_handle.select()
    result = await df.collect()
    assert len(result) == 2
    assert result[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_records_direct_insert(tmp_path):
    """Test that AsyncRecords can be passed directly to table.insert()."""
    db_path = tmp_path / "async_records_direct_insert.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], f)

    # Use read.records for AsyncLazyRecords (returns synchronously)
    records = db.read.records.json(str(json_path))

    # Create table
    table = await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert records directly
    # Note: AsyncRecords needs to be materialized first for insert
    rows = await records.rows()
    await (await db.createDataFrame(rows, pk="id")).write.insertInto("target")

    # Verify
    df = table.select()
    result = await df.collect()
    assert len(result) == 2

    await db.close()


@pytest.mark.asyncio
async def test_async_records_schema(tmp_path):
    """Test that AsyncRecords provide schema information."""
    db_path = tmp_path / "async_records_schema.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Use read.records for AsyncLazyRecords (returns synchronously)
    records = db.read.records.csv(str(csv_path))

    # Check schema - materialize first to get inferred schema
    materialized = await records.collect()
    schema = materialized.schema
    assert schema is not None
    assert len(schema) == 2
    assert schema[0].name == "id"
    assert schema[1].name == "name"

    await db.close()


@pytest.mark.asyncio
async def test_async_records_streaming(tmp_path):
    """Test AsyncRecords in streaming mode."""
    db_path = tmp_path / "async_records_streaming.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create large CSV file
    csv_path = tmp_path / "large.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        for i in range(50):
            f.write(f"{i},user_{i}\n")

    # Load in streaming mode using read.records (returns AsyncLazyRecords synchronously)
    records = db.read.records.stream().option("chunk_size", 10).csv(str(csv_path))

    # Async iterate
    count = 0
    async for row in records:
        assert "id" in row
        assert "name" in row
        count += 1
        if count >= 10:
            break

    # Materialize all
    all_rows = await records.rows()
    assert len(all_rows) == 50

    await db.close()


@pytest.mark.asyncio
async def test_async_records_from_table(tmp_path):
    """Test AsyncRecords from db.load.table()."""
    db_path = tmp_path / "async_records_table.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create and populate table
    await db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()
    await (
        await db.createDataFrame(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            pk="id",
        )
    ).write.insertInto("source")
    # Load table as AsyncRecords (backward compatibility)
    # Note: db.load.table() now returns AsyncDataFrame, use read.records for Records
    from moltres.io.records import AsyncRecords

    df = await db.load.table("source")
    rows = await df.collect()
    records = AsyncRecords(_data=rows, _database=db)

    # Should work like any AsyncRecords
    rows = await records.rows()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Can be inserted into another table
    await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()
    count = await records.insert_into("target")
    assert count == 2

    await db.close()


@pytest.mark.asyncio
async def test_async_records_with_explicit_schema(tmp_path):
    """Test AsyncRecords with explicit schema."""
    db_path = tmp_path / "async_records_schema_explicit.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
    ]

    records = db.read.records.schema(schema).csv(str(csv_path))
    rows = await records.rows()

    assert len(rows) == 1
    assert isinstance(rows[0]["id"], int)
    assert rows[0]["id"] == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_records_empty_data(tmp_path):
    """Test AsyncRecords with empty data."""
    db_path = tmp_path / "async_records_empty.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
    ]

    records = db.read.records.schema(schema).csv(str(csv_path))
    rows = await records.rows()

    assert len(rows) == 0
    assert rows == []

    await db.close()


@pytest.mark.asyncio
async def test_async_records_insert_into_with_table_handle(tmp_path):
    """Test AsyncRecords.insert_into() with AsyncTableHandle."""
    db_path = tmp_path / "async_records_insert_handle.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Use read.records for AsyncLazyRecords (returns synchronously)
    records = db.read.records.csv(str(csv_path))

    # Create table
    table = await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert using AsyncTableHandle
    count = await records.insert_into(table)
    assert count == 1

    # Verify
    df = table.select()
    result = await df.collect()
    assert len(result) == 1
    assert result[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_csv_gzip(tmp_path):
    """Test reading gzip-compressed CSV file asynchronously."""
    import gzip

    db_path = tmp_path / "async_read_csv_gzip.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create gzip-compressed CSV file
    csv_path = tmp_path / "data.csv.gz"
    with gzip.open(csv_path, "wt", encoding="utf-8") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")

    # Read compressed CSV (using read.records for AsyncLazyRecords)
    records = db.read.records.csv(str(csv_path))
    rows = await records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5

    await db.close()


@pytest.mark.asyncio
async def test_async_read_json_gzip(tmp_path):
    """Test reading gzip-compressed JSON file asynchronously."""
    import gzip

    db_path = tmp_path / "async_read_json_gzip.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create gzip-compressed JSON file
    json_path = tmp_path / "data.json.gz"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    with gzip.open(json_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    # Read compressed JSON (using read.records for AsyncLazyRecords)
    records = db.read.records.json(str(json_path))
    rows = await records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_text_gzip(tmp_path):
    """Test reading gzip-compressed text file asynchronously."""
    import gzip

    db_path = tmp_path / "async_read_text_gzip.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create gzip-compressed text file
    text_path = tmp_path / "data.txt.gz"
    with gzip.open(text_path, "wt", encoding="utf-8") as f:
        f.write("line 1\n")
        f.write("line 2\n")

    # Read compressed text (using read.records for AsyncLazyRecords)
    records = db.read.records.text(str(text_path))
    rows = await records.rows()

    assert len(rows) == 2
    assert rows[0]["value"] == "line 1"
    assert rows[1]["value"] == "line 2"

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_read_jsonl(tmp_path):
    """Test reading JSONL file as AsyncDataFrame."""
    db_path = tmp_path / "async_df_jsonl.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')
        f.write('{"id": 2, "name": "Bob"}\n')

    df = await db.load.jsonl(str(jsonl_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_read_text(tmp_path):
    """Test reading text file as AsyncDataFrame."""
    db_path = tmp_path / "async_df_text.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        f.write("line 1\n")
        f.write("line 2\n")

    df = await db.load.text(str(text_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["value"] == "line 1"

    await db.close()


@pytest.mark.asyncio
async def test_async_dataframe_transformations(tmp_path):
    """Test AsyncDataFrame transformations before materialization."""
    db_path = tmp_path / "async_df_transforms.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")
        f.write("3,Charlie,92.0\n")

    from moltres import col

    df = await db.load.csv(str(csv_path))
    filtered_df = df.where(col("score") > 90)
    rows = await filtered_df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_records_jsonl(tmp_path):
    """Test async read.records.jsonl() convenience method."""
    db_path = tmp_path / "async_read_records_jsonl.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')

    records = db.read.records.jsonl(str(jsonl_path))
    rows = await records.rows()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_records_text(tmp_path):
    """Test async read.records.text() convenience method."""
    db_path = tmp_path / "async_read_records_text.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        f.write("line 1\n")

    records = db.read.records.text(str(text_path))
    rows = await records.rows()

    assert len(rows) == 1
    assert rows[0]["value"] == "line 1"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_records_dicts(tmp_path):
    """Test async read.records.dicts() convenience method."""
    db_path = tmp_path / "async_read_records_dicts.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres.io.records import AsyncRecords

    data = [{"id": 1, "name": "Alice"}]
    # dicts() returns AsyncRecords directly (not lazy) since data is already materialized
    records = db.read.records.dicts(data)

    rows = await records.rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert isinstance(records, AsyncRecords)

    await db.close()


@pytest.mark.asyncio
async def test_async_lazy_records_explicit_collect(tmp_path):
    """Test AsyncLazyRecords explicit .collect() materialization."""
    db_path = tmp_path / "async_lazy_records_collect.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Create AsyncLazyRecords
    lazy_records = db.read.records.csv(str(csv_path))
    assert isinstance(lazy_records, AsyncLazyRecords)

    # Explicitly materialize with .collect()
    records = await lazy_records.collect()
    assert isinstance(records, AsyncRecords)
    rows = await records.rows()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_lazy_records_auto_materialize_iteration(tmp_path):
    """Test AsyncLazyRecords auto-materialization for async iteration."""
    db_path = tmp_path / "async_lazy_records_iter.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # Async iteration should auto-materialize
    rows = []
    async for row in lazy_records:
        rows.append(row)
        if len(rows) >= 2:
            break

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_lazy_records_auto_materialize_insert_into(tmp_path):
    """Test AsyncLazyRecords auto-materialization for insert_into()."""
    db_path = tmp_path / "async_lazy_records_insert.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create table
    await db.create_table(
        "target",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # insert_into() should auto-materialize
    count = await lazy_records.insert_into("target")
    assert count == 2

    # Verify data was inserted
    table_handle = await db.table("target")
    df = table_handle.select()
    rows = await df.collect()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_lazy_records_auto_materialize_create_dataframe(tmp_path):
    """Test AsyncLazyRecords auto-materialization when used in createDataFrame."""
    db_path = tmp_path / "async_lazy_records_create_df.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # createDataFrame should auto-materialize AsyncLazyRecords
    df = await db.createDataFrame(lazy_records, pk="id")
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_lazy_records_backward_compatibility(tmp_path):
    """Test that AsyncLazyRecords maintains backward compatibility with AsyncRecords interface."""
    db_path = tmp_path / "async_lazy_records_compat.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # All AsyncRecords methods should work (auto-materialize)
    rows = await lazy_records.rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    # Async iteration should work
    async for row in lazy_records:
        assert row["name"] == "Alice"
        break

    await db.close()


@pytest.mark.asyncio
async def test_async_read_table_pyspark_style(tmp_path):
    """Test PySpark-style await db.read.table() returning an AsyncDataFrame."""
    db_path = tmp_path / "async_read_table_pyspark.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    from moltres import column, col

    # Create and populate table
    await db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    from moltres.io.records import AsyncRecords

    records = AsyncRecords(
        _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db
    )
    await records.insert_into("users")

    # Read using PySpark-style await db.read.table() - returns AsyncDataFrame
    df = await db.read.table("users")
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Verify it's an AsyncDataFrame (can be transformed)
    filtered_df = df.where(col("id") == 1)
    filtered_rows = await filtered_df.collect()
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_csv_pyspark_style(tmp_path):
    """Test PySpark-style await db.read.csv() returning an AsyncDataFrame."""
    db_path = tmp_path / "async_read_csv_pyspark.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read using PySpark-style await db.read.csv() - returns AsyncDataFrame
    df = await db.read.csv(str(csv_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Verify it's an AsyncDataFrame (can be transformed)
    from moltres import col

    filtered_df = df.where(col("id") == 1)
    filtered_rows = await filtered_df.collect()
    assert len(filtered_rows) == 1

    await db.close()


@pytest.mark.asyncio
async def test_async_read_json_pyspark_style(tmp_path):
    """Test PySpark-style await db.read.json() returning an AsyncDataFrame."""
    db_path = tmp_path / "async_read_json_pyspark.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Read using PySpark-style await db.read.json() - returns AsyncDataFrame
    df = await db.read.json(str(json_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_readers_backward_compatibility(tmp_path):
    """Test that db.read.records.* returns AsyncLazyRecords synchronously (backward compatible interface)."""
    db_path = tmp_path / "async_read_backward_compat.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Verify db.read.records.* returns AsyncLazyRecords synchronously
    records = db.read.records.csv(str(csv_path))
    assert isinstance(records, AsyncLazyRecords)
    rows = await records.rows()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    # Verify both APIs work side by side
    df = await db.read.csv(str(csv_path))
    df_rows = await df.collect()

    assert len(df_rows) == 1
    assert df_rows[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_builder_methods_pyspark_style(tmp_path):
    """Test builder methods (schema, option) with await db.read.* API."""
    db_path = tmp_path / "async_read_builder_pyspark.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file with pipe delimiter
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name\n")
        f.write("1|Alice\n")

    # Use builder methods with await db.read API
    df = await db.read.option("delimiter", "|").csv(str(csv_path))
    rows = await df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_format_pyspark_style(tmp_path):
    """Test PySpark-style await db.read.format().load() API."""
    db_path = tmp_path / "async_read_format_pyspark.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Use format().load() API
    format_reader = await db.read.format("csv")
    df = await format_reader.load(str(csv_path))
    rows = await df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    await db.close()
