"""Tests for async data loading operations."""

import json

import pytest

from moltres import async_connect
from moltres.table.schema import ColumnDef


@pytest.mark.asyncio
async def test_async_load_csv(tmp_path):
    """Test async CSV loading returns AsyncRecords."""
    db_path = tmp_path / "async_load_csv.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")

    # Load CSV - returns AsyncRecords
    records = await db.load.csv(str(csv_path))
    rows = await records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_load_json(tmp_path):
    """Test async JSON loading returns AsyncRecords."""
    db_path = tmp_path / "async_load_json.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], f)

    # Load JSON - returns AsyncRecords
    records = await db.load.json(str(json_path))
    rows = await records.rows()

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

    records = await db.load.csv(str(csv_path))

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

    records = await db.load.csv(str(csv_path))

    # Create table
    await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    )

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

    records = await db.load.json(str(json_path))

    # Create table
    table = await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    )

    # Insert records directly
    # Note: AsyncRecords needs to be materialized first for insert
    rows = await records.rows()
    count = await table.insert(rows)
    assert count == 2

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

    records = await db.load.csv(str(csv_path))

    # Check schema
    schema = records.schema
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

    # Load in streaming mode
    records = await db.load.stream().option("chunk_size", 10).csv(str(csv_path))

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
    table = await db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    )
    await table.insert([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Load table as AsyncRecords
    records = await db.load.table("source")

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
    )
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

    records = await db.load.schema(schema).csv(str(csv_path))
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

    records = await db.load.schema(schema).csv(str(csv_path))
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

    records = await db.load.csv(str(csv_path))

    # Create table
    table = await db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    )

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

    # Read compressed CSV
    records = await db.load.csv(str(csv_path))
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

    # Read compressed JSON
    records = await db.load.json(str(json_path))
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

    # Read compressed text
    records = await db.load.text(str(text_path))
    rows = await records.rows()

    assert len(rows) == 2
    assert rows[0]["value"] == "line 1"
    assert rows[1]["value"] == "line 2"

    await db.close()
