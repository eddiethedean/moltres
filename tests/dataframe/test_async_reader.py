"""Tests for async data loading operations."""

import json
import os

import pytest


if os.environ.get("MOLTRES_SKIP_PANDAS_TESTS") == "1":
    pytest.skip(
        "Skipping pandas-dependent tests (MOLTRES_SKIP_PANDAS_TESTS=1)",
        allow_module_level=True,
    )

if os.environ.get("MOLTRES_SKIP_PANDAS_TESTS"):
    pytest.skip(
        "Skipping pandas-dependent tests (MOLTRES_SKIP_PANDAS_TESTS=1)",
        allow_module_level=True,
    )

from moltres import async_connect
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
async def test_async_read_parquet(tmp_path):
    """Test async Parquet reading."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        pytest.skip("pyarrow and pandas required for Parquet tests")

    db_path = tmp_path / "async_read_parquet.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create Parquet file
    parquet_path = tmp_path / "data.parquet"
    data = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
    table = pa.Table.from_pandas(data)
    pq.write_table(table, str(parquet_path))

    # Read Parquet
    df = await db.load.parquet(str(parquet_path))
    rows = await df.collect()

    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[1]["id"] == 2
    assert rows[1]["name"] == "Bob"

    await db.close()


@pytest.mark.asyncio
async def test_async_read_parquet_file_not_found(tmp_path):
    """Test FileNotFoundError for async Parquet reading."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow required for Parquet tests")

    db_path = tmp_path / "async_read_parquet_error.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    with pytest.raises(FileNotFoundError, match="Parquet file not found"):
        df = await db.load.parquet(str(tmp_path / "nonexistent.parquet"))
        await df.collect()  # Trigger actual reading

    await db.close()


@pytest.mark.asyncio
async def test_async_read_parquet_empty_file(tmp_path):
    """Test async Parquet reading with empty file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        pytest.skip("pyarrow and pandas required for Parquet tests")

    db_path = tmp_path / "async_read_parquet_empty.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create empty Parquet file
    parquet_path = tmp_path / "empty.parquet"
    data = pd.DataFrame()
    table = pa.Table.from_pandas(data)
    pq.write_table(table, str(parquet_path))

    # Read empty Parquet with schema

    schema = [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")]
    df = await db.load.schema(schema).parquet(str(parquet_path))
    rows = await df.collect()

    assert len(rows) == 0

    await db.close()


@pytest.mark.asyncio
async def test_async_read_parquet_with_schema(tmp_path):
    """Test async Parquet reading with explicit schema."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        pytest.skip("pyarrow and pandas required for Parquet tests")

    db_path = tmp_path / "async_read_parquet_schema.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create Parquet file
    parquet_path = tmp_path / "data.parquet"
    data = pd.DataFrame([{"id": 1, "value": 10.5}])
    table = pa.Table.from_pandas(data)
    pq.write_table(table, str(parquet_path))

    # Read with explicit schema

    schema = [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="value", type_name="REAL")]
    df = await db.load.schema(schema).parquet(str(parquet_path))
    rows = await df.collect()

    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["value"] == 10.5

    await db.close()


@pytest.mark.asyncio
async def test_async_read_parquet_stream(tmp_path):
    """Test async Parquet streaming reading."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        pytest.skip("pyarrow and pandas required for Parquet tests")

    db_path = tmp_path / "async_read_parquet_stream.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create Parquet file with multiple row groups
    parquet_path = tmp_path / "data.parquet"
    data = pd.DataFrame([{"id": i, "value": i * 10} for i in range(100)])
    table = pa.Table.from_pandas(data)
    pq.write_table(table, str(parquet_path), row_group_size=25)  # 4 row groups

    # Read Parquet in streaming mode
    records = db.read.records.parquet(str(parquet_path))
    rows = await records.rows()

    assert len(rows) == 100
    assert rows[0]["id"] == 0
    assert rows[99]["id"] == 99

    await db.close()


@pytest.mark.asyncio
async def test_async_read_parquet_stream_empty(tmp_path):
    """Test async Parquet streaming with empty file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        pytest.skip("pyarrow and pandas required for Parquet tests")

    db_path = tmp_path / "async_read_parquet_stream_empty.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create empty Parquet file
    parquet_path = tmp_path / "empty.parquet"
    data = pd.DataFrame()
    table = pa.Table.from_pandas(data)
    pq.write_table(table, str(parquet_path))

    # Read empty Parquet with schema

    schema = [ColumnDef(name="id", type_name="INTEGER")]
    records = db.read.records.schema(schema).parquet(str(parquet_path))
    rows = await records.rows()

    assert len(rows) == 0

    await db.close()


@pytest.mark.asyncio
async def test_async_read_parquet_merge_schema(tmp_path):
    """Test async Parquet reading with mergeSchema option."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pandas as pd
    except ImportError:
        pytest.skip("pyarrow and pandas required for Parquet tests")

    db_path = tmp_path / "async_read_parquet_merge.sqlite"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create Parquet file
    parquet_path = tmp_path / "data.parquet"
    data = pd.DataFrame([{"id": 1, "name": "Alice"}])
    table = pa.Table.from_pandas(data)
    pq.write_table(table, str(parquet_path))

    # Read with mergeSchema option (no-op for single file, but tests the code path)
    df = await db.load.option("mergeSchema", True).parquet(str(parquet_path))
    rows = await df.collect()

    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"

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
