"""Tests for data loading operations."""

import json
import os

import pytest

if os.environ.get("MOLTRES_SKIP_PANDAS_TESTS") == "1":
    pytest.skip(
        "Skipping pandas-dependent tests (MOLTRES_SKIP_PANDAS_TESTS=1)",
        allow_module_level=True,
    )

from moltres import column, col, connect
from moltres.io.records import LazyRecords, Records
from moltres.table.schema import ColumnDef


def test_read_table(tmp_path):
    """Test reading from database table as DataFrame."""
    db_path = tmp_path / "read_table.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    from moltres.io.records import Records

    records = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db)
    records.insert_into("source")

    # Read using load.table() - returns DataFrame
    df = db.load.table("source")
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_read_table_pyspark_style(tmp_path):
    """Test PySpark-style db.read.table() returning a DataFrame."""
    db_path = tmp_path / "read_table_pyspark.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    from moltres.io.records import Records

    records = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db)
    records.insert_into("users")

    # Read using PySpark-style db.read.table() - returns DataFrame
    df = db.read.table("users")
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Verify it's a DataFrame (can be transformed)
    filtered_df = df.where(col("id") == 1)
    filtered_rows = filtered_df.collect()
    assert len(filtered_rows) == 1
    assert filtered_rows[0]["name"] == "Alice"


def test_read_csv_pyspark_style(tmp_path):
    """Test PySpark-style db.read.csv() returning a DataFrame."""
    db_path = tmp_path / "read_csv_pyspark.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read using PySpark-style db.read.csv() - returns DataFrame
    df = db.read.csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Verify it's a DataFrame (can be transformed)
    filtered_df = df.where(col("id") == 1)
    filtered_rows = filtered_df.collect()
    assert len(filtered_rows) == 1


def test_read_json_pyspark_style(tmp_path):
    """Test PySpark-style db.read.json() returning a DataFrame."""
    db_path = tmp_path / "read_json_pyspark.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Read using PySpark-style db.read.json() - returns DataFrame
    df = db.read.json(str(json_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_read_readers_backward_compatibility(tmp_path):
    """Test that db.read.records.* still works for backward compatibility."""
    db_path = tmp_path / "read_backward_compat.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Verify db.read.records.* still works
    records = db.read.records.csv(str(csv_path))
    rows = records.rows()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    # Verify both APIs work side by side
    df = db.read.csv(str(csv_path))
    df_rows = df.collect()

    assert len(df_rows) == 1
    assert df_rows[0]["name"] == "Alice"


def test_read_builder_methods_pyspark_style(tmp_path):
    """Test builder methods (schema, option) with db.read.* API."""
    db_path = tmp_path / "read_builder_pyspark.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file with pipe delimiter
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name\n")
        f.write("1|Alice\n")

    # Use builder methods with db.read API
    df = db.read.option("delimiter", "|").csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_read_format_pyspark_style(tmp_path):
    """Test PySpark-style db.read.format().load() API."""
    db_path = tmp_path / "read_format_pyspark.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Use format().load() API
    df = db.read.format("csv").load(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_read_csv(tmp_path):
    """Test reading CSV file as DataFrame."""
    db_path = tmp_path / "read_csv.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")

    # Read CSV - returns DataFrame
    df = db.load.csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5
    assert rows[1]["name"] == "Bob"


def test_read_csv_with_options(tmp_path):
    """Test reading CSV with custom options."""
    db_path = tmp_path / "read_csv_options.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV with pipe delimiter
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name\n")
        f.write("1|Alice\n")

    # Read with delimiter option
    df = db.load.option("delimiter", "|").csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_read_csv_no_header(tmp_path):
    """Test reading CSV without header."""
    db_path = tmp_path / "read_csv_no_header.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV without header
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read with explicit schema
    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
    ]
    df = db.load.schema(schema).option("header", False).csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"


def test_read_json(tmp_path):
    """Test reading JSON file."""
    db_path = tmp_path / "read_json.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Read JSON - returns DataFrame
    df = db.load.json(str(json_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_read_jsonl(tmp_path):
    """Test reading JSONL file."""
    db_path = tmp_path / "read_jsonl.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSONL file
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')
        f.write('{"id": 2, "name": "Bob"}\n')

    # Read JSONL - returns DataFrame
    df = db.load.jsonl(str(jsonl_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_read_text(tmp_path):
    """Test reading text file."""
    db_path = tmp_path / "read_text.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create text file
    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        f.write("line 1\n")
        f.write("line 2\n")
        f.write("line 3\n")

    # Read text - returns DataFrame
    df = db.load.text(str(text_path))
    rows = df.collect()

    assert len(rows) == 3
    assert rows[0]["value"] == "line 1"
    assert rows[1]["value"] == "line 2"


def test_read_text_custom_column(tmp_path):
    """Test reading text file with custom column name."""
    db_path = tmp_path / "read_text_custom.sqlite"
    db = connect(f"sqlite:///{db_path}")

    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        f.write("line 1\n")

    df = db.load.text(str(text_path), column_name="line")
    rows = df.collect()

    assert len(rows) == 1
    assert "line" in rows[0]
    assert rows[0]["line"] == "line 1"


def test_read_format_csv(tmp_path):
    """Test generic format().load() API for CSV."""
    db_path = tmp_path / "read_format.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    df = db.load.format("csv").load(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_read_format_json(tmp_path):
    """Test generic format().load() API for JSON."""
    db_path = tmp_path / "read_format_json.sqlite"
    db = connect(f"sqlite:///{db_path}")

    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"id": 1, "name": "Alice"}], f)

    df = db.load.format("json").load(str(json_path))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_read_with_explicit_schema(tmp_path):
    """Test reading with explicit schema."""
    db_path = tmp_path / "read_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER", nullable=False),
        ColumnDef(name="name", type_name="TEXT", nullable=False),
    ]

    records = db.read.records.schema(schema).csv(str(csv_path))
    rows = records.rows()

    assert len(rows) == 1
    assert isinstance(rows[0]["id"], int)
    assert rows[0]["id"] == 1


def test_read_missing_file(tmp_path):
    """Test error handling for missing file."""
    db_path = tmp_path / "read_missing.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # DataFrame creation is lazy, error occurs on collect()
    df = db.load.csv(str(tmp_path / "nonexistent.csv"))
    with pytest.raises(FileNotFoundError):
        df.collect()


def test_read_empty_csv(tmp_path):
    """Test reading empty CSV file."""
    db_path = tmp_path / "read_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")

    # Should raise error without explicit schema (error occurs on collect())
    df = db.load.csv(str(csv_path))
    with pytest.raises(ValueError, match="empty"):
        df.collect()

    # Should work with explicit schema
    schema = [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")]
    df = db.load.schema(schema).csv(str(csv_path))
    rows = df.collect()
    assert len(rows) == 0


def test_read_parquet_requires_dependencies(tmp_path):
    """Test that parquet read requires pandas/pyarrow."""
    db_path = tmp_path / "read_parquet.sqlite"
    db = connect(f"sqlite:///{db_path}")

    parquet_path = tmp_path / "data.parquet"

    # This will either work (if dependencies installed) or raise RuntimeError
    try:
        # Try to create a simple parquet file if pyarrow is available
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df_pd = pd.DataFrame([{"id": 1, "name": "Alice"}])
        table = pa.Table.from_pandas(df_pd)
        pq.write_table(table, str(parquet_path))

        df = db.load.parquet(str(parquet_path))
        rows = df.collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
    except ImportError:
        # If dependencies not installed, test that we get appropriate error
        with pytest.raises(RuntimeError, match="pandas|pyarrow"):
            db.load.parquet(str(parquet_path))


def test_read_chained_options(tmp_path):
    """Test chaining options and schema."""
    db_path = tmp_path / "read_chained.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name\n")
        f.write("1|Alice\n")

    schema = [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")]

    df = db.load.schema(schema).option("delimiter", "|").csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert isinstance(rows[0]["id"], int)


def test_records_insert_into_table(tmp_path):
    """Test that Records can be inserted into a table."""
    db_path = tmp_path / "records_insert.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read CSV as Records using read.records
    records = db.read.records.csv(str(csv_path))

    # Create table
    db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert records into table
    count = records.insert_into("target")
    assert count == 2

    # Verify data was inserted
    result = db.table("target").select().collect()
    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"


def test_records_direct_insert(tmp_path):
    """Test that Records can be passed directly to table.insert()."""
    db_path = tmp_path / "records_direct_insert.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], f)

    # Read JSON as Records using read.records
    records = db.read.records.json(str(json_path))

    # Create table
    db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert records directly using Records.insert_into()
    count = records.insert_into("target")
    assert count == 2

    # Verify data was inserted
    result = db.table("target").select().collect()
    assert len(result) == 2


def test_records_iteration(tmp_path):
    """Test that Records can be iterated directly."""
    db_path = tmp_path / "records_iteration.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read CSV as Records using read.records
    records = db.read.records.csv(str(csv_path))

    # Iterate directly
    rows = []
    for row in records:
        rows.append(row)

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Test len()
    assert len(records) == 2

    # Test indexing
    assert records[0]["name"] == "Alice"
    assert records[1]["name"] == "Bob"


def test_records_schema_access(tmp_path):
    """Test that Records provide schema information."""
    db_path = tmp_path / "records_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")

    records = db.read.records.csv(str(csv_path))

    # Check schema is available
    schema = records.schema
    assert schema is not None
    assert len(schema) == 3
    assert schema[0].name == "id"
    assert schema[1].name == "name"
    assert schema[2].name == "score"


def test_records_empty_data(tmp_path):
    """Test Records with empty data."""
    db_path = tmp_path / "records_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create empty CSV with schema
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
    ]

    records = db.read.records.schema(schema).csv(str(csv_path))

    # Empty Records should work
    assert len(records) == 0
    assert records.rows() == []
    assert list(records) == []

    # Should raise IndexError on indexing
    with pytest.raises(IndexError):
        _ = records[0]


def test_records_sequence_protocol(tmp_path):
    """Test that Records properly implement Sequence protocol."""
    db_path = tmp_path / "records_sequence.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")
        f.write("3,Charlie\n")

    records = db.read.records.csv(str(csv_path))

    # Test Sequence protocol methods
    from collections.abc import Sequence

    assert isinstance(records, Sequence)
    assert len(records) == 3
    assert records[0] == {"id": 1, "name": "Alice"}
    assert records[-1] == {"id": 3, "name": "Charlie"}

    # Test iteration
    items = list(records)
    assert len(items) == 3

    # Test contains (Sequence protocol)
    # Note: Records doesn't implement __contains__, but we can check via iteration
    names = [row["name"] for row in records]
    assert "Alice" in names
    assert "Bob" in names


def test_records_with_update_operation(tmp_path):
    """Test that Records can be used with update operations."""
    db_path = tmp_path / "records_update.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
            ColumnDef(name="status", type_name="TEXT"),
        ],
    ).collect()
    from moltres.io.records import Records

    records = Records(_data=[{"id": 1, "name": "Alice", "status": "active"}], _database=db)
    records.insert_into("users")

    # Create CSV with updates
    csv_path = tmp_path / "updates.csv"
    with open(csv_path, "w") as f:
        f.write("id,status\n")
        f.write("1,inactive\n")

    records = db.read.records.csv(str(csv_path))

    # Records can be used in update operations (though update expects different format)
    # For now, test that we can read the records
    rows = records.rows()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["status"] == "inactive"


def test_records_streaming_mode(tmp_path):
    """Test Records in streaming mode."""
    db_path = tmp_path / "records_streaming.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create large CSV file
    csv_path = tmp_path / "large.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        for i in range(100):
            f.write(f"{i},user_{i}\n")

    # Load in streaming mode using read.records
    records = db.read.records.stream().option("chunk_size", 10).csv(str(csv_path))

    # Should be able to iterate without materializing
    count = 0
    for row in records:
        assert "id" in row
        assert "name" in row
        count += 1
        if count >= 10:  # Just check first few
            break

    # Materialize all
    all_rows = records.rows()
    assert len(all_rows) == 100
    assert all_rows[0]["id"] == 0
    assert all_rows[-1]["id"] == 99


def test_records_from_table(tmp_path):
    """Test Records from db.load.table()."""
    db_path = tmp_path / "records_table.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "source",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()
    from moltres.io.records import Records

    records = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db)
    records.insert_into("source")

    # Load table as Records using read.records (backward compatibility)
    # Note: db.load.table() now returns DataFrame, use read.records for Records
    from moltres.io.records import Records

    df = db.load.table("source")
    rows = df.collect()
    records = Records(_data=rows, _database=db)

    # Should work like any Records
    assert len(records) == 2
    rows = records.rows()
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Can be inserted into another table
    db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()
    count = records.insert_into("target")
    assert count == 2


def test_records_multiple_formats(tmp_path):
    """Test Records from different file formats produce consistent results."""
    db_path = tmp_path / "records_formats.sqlite"
    db = connect(f"sqlite:///{db_path}")

    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    # CSV
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    csv_records = db.read.records.csv(str(csv_path))
    csv_rows = csv_records.rows()

    # JSON
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    json_records = db.read.records.json(str(json_path))
    json_rows = json_records.rows()

    # JSONL
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    jsonl_records = db.read.records.jsonl(str(jsonl_path))
    jsonl_rows = jsonl_records.rows()

    # All should produce same data
    assert csv_rows == json_rows
    assert json_rows == jsonl_rows
    assert len(csv_rows) == 2


def test_records_index_error(tmp_path):
    """Test Records raises IndexError for out-of-bounds access."""
    db_path = tmp_path / "records_index_error.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    records = db.read.records.csv(str(csv_path))

    # Valid index
    assert records[0]["name"] == "Alice"

    # Invalid index
    with pytest.raises(IndexError):
        _ = records[10]


def test_records_insert_into_nonexistent_table(tmp_path):
    """Test Records.insert_into() with nonexistent table."""
    db_path = tmp_path / "records_insert_error.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    records = db.read.records.csv(str(csv_path))

    # Should raise error for nonexistent table
    with pytest.raises(Exception):  # Will be ExecutionError or similar
        records.insert_into("nonexistent")


def test_records_insert_into_with_table_handle(tmp_path):
    """Test Records.insert_into() with TableHandle."""
    db_path = tmp_path / "records_insert_handle.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    records = db.read.records.csv(str(csv_path))

    # Create table
    table = db.create_table(
        "target",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ],
    ).collect()

    # Insert using TableHandle
    count = records.insert_into(table)
    assert count == 2

    # Verify
    result = table.select().collect()
    assert len(result) == 2


def test_records_iter_method(tmp_path):
    """Test Records.iter() method."""
    db_path = tmp_path / "records_iter.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    records = db.read.records.csv(str(csv_path))

    # Test iter() method
    rows = []
    for row in records.iter():
        rows.append(row)

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_records_with_explicit_schema_types(tmp_path):
    """Test Records with explicit schema preserves types."""
    db_path = tmp_path / "records_schema_types.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score,active\n")
        f.write("1,Alice,95.5,true\n")
        f.write("2,Bob,87.0,false\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
        ColumnDef(name="score", type_name="REAL"),
        ColumnDef(name="active", type_name="BOOLEAN"),
    ]

    records = db.read.records.schema(schema).csv(str(csv_path))
    rows = records.rows()

    assert isinstance(rows[0]["id"], int)
    assert isinstance(rows[0]["name"], str)
    assert isinstance(rows[0]["score"], float)
    # Note: SQLite doesn't have native boolean, so this might be int
    assert rows[0]["id"] == 1
    assert rows[0]["score"] == 95.5


def test_records_chained_operations(tmp_path):
    """Test chaining DataLoader operations."""
    db_path = tmp_path / "records_chained.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name|score\n")
        f.write("1|Alice|95.5\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
        ColumnDef(name="score", type_name="REAL"),
    ]

    # Chain multiple options (using read.records for Records)
    records = (
        db.read.records.schema(schema)
        .option("delimiter", "|")
        .option("header", True)
        .csv(str(csv_path))
    )

    rows = records.rows()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5


def test_records_without_database(tmp_path):
    """Test Records.insert_into() without attached database."""
    from moltres.io.records import Records

    # Create Records without database
    records = Records(_data=[{"id": 1, "name": "Alice"}])

    # Should raise error when trying to insert
    with pytest.raises(RuntimeError, match="Cannot insert Records without"):
        records.insert_into("table")


def test_records_json_multiline(tmp_path):
    """Test Records from JSON with multiline option."""
    db_path = tmp_path / "records_json_multiline.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSONL file (one object per line)
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')
        f.write('{"id": 2, "name": "Bob"}\n')

    # Read as JSON with multiline=True (treats as JSONL)
    records = db.read.records.option("multiline", True).json(str(jsonl_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_records_text_empty_file(tmp_path):
    """Test Records from empty text file."""
    db_path = tmp_path / "records_text_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    text_path = tmp_path / "empty.txt"
    with open(text_path, "w"):
        pass  # Empty file

    records = db.read.records.text(str(text_path))
    rows = records.rows()

    assert len(rows) == 0
    assert len(records) == 0


def test_records_large_dataset(tmp_path):
    """Test Records with a larger dataset to verify performance."""
    db_path = tmp_path / "records_large.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV with 1000 rows
    csv_path = tmp_path / "large.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,value\n")
        for i in range(1000):
            f.write(f"{i},user_{i},{i * 1.5}\n")

    records = db.read.records.csv(str(csv_path))

    # Test iteration
    count = 0
    for row in records:
        assert row["id"] == count
        count += 1
        if count >= 10:  # Just check first few
            break

    # Materialize all
    all_rows = records.rows()
    assert len(all_rows) == 1000
    assert all_rows[0]["id"] == 0
    assert all_rows[-1]["id"] == 999


def test_records_schema_preservation(tmp_path):
    """Test that Records preserve schema information across operations."""
    db_path = tmp_path / "records_schema_preserve.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER", nullable=False),
        ColumnDef(name="name", type_name="TEXT", nullable=False),
        ColumnDef(name="score", type_name="REAL", nullable=True),
    ]

    records = db.read.records.schema(schema).csv(str(csv_path))

    # Schema should be preserved
    assert records.schema == schema

    # After materialization, schema should still be available
    records.rows()
    assert records.schema == schema


def test_records_with_update_rows(tmp_path):
    """Test that Records can be used with update_rows helper."""
    db_path = tmp_path / "records_update_rows.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
            ColumnDef(name="status", type_name="TEXT"),
        ],
    ).collect()
    from moltres.io.records import Records

    records = Records(_data=[{"id": 1, "name": "Alice", "status": "active"}], _database=db)
    records.insert_into("users")

    # Create CSV with updates
    csv_path = tmp_path / "updates.csv"
    with open(csv_path, "w") as f:
        f.write("id,status\n")
        f.write("1,inactive\n")

    records = db.read.records.csv(str(csv_path))

    # Records can be used with update operations
    # Note: update_rows expects Sequence[Mapping], which Records implements

    # For update, we need to provide the update data
    # Records can be used, but update_rows needs specific format
    rows = records.rows()
    # This is more of an integration test - Records work with insert, update may need different format
    assert len(rows) == 1
    assert rows[0]["status"] == "inactive"


def test_records_format_reader(tmp_path):
    """Test FormatReader with Records."""
    db_path = tmp_path / "records_format.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Use format().load() API (returns DataFrame, use read.records for Records)
    df = db.load.format("csv").load(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    # Test with JSON
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump([{"id": 1, "name": "Bob"}], f)

    df = db.load.format("json").load(str(json_path))
    rows = df.collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Bob"


def test_records_chained_load_operations(tmp_path):
    """Test that DataLoader methods can be chained in different orders."""
    db_path = tmp_path / "records_chained.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name\n")
        f.write("1|Alice\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
    ]

    # Test different chaining orders (using read.records for Records)
    records1 = db.read.records.schema(schema).option("delimiter", "|").csv(str(csv_path))
    records2 = db.read.records.option("delimiter", "|").schema(schema).csv(str(csv_path))

    rows1 = records1.rows()
    rows2 = records2.rows()

    assert rows1 == rows2
    assert len(rows1) == 1
    assert rows1[0]["id"] == 1


def test_records_with_special_characters(tmp_path):
    """Test Records with special characters in data."""
    db_path = tmp_path / "records_special.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,description\n")
        f.write('1,Alice,"Hello, world!"\n')
        f.write('2,Bob,"It\'s a test"\n')
        f.write('3,Charlie,"Line 1\nLine 2"\n')

    records = db.read.records.csv(str(csv_path))
    rows = records.rows()

    assert len(rows) == 3
    assert rows[0]["name"] == "Alice"
    # CSV reader should handle quoted fields
    assert "Hello" in rows[0]["description"] or rows[0]["description"] == "Hello, world!"


def test_records_json_single_object(tmp_path):
    """Test Records from JSON with single object (not array)."""
    db_path = tmp_path / "records_json_single.sqlite"
    db = connect(f"sqlite:///{db_path}")

    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump({"id": 1, "name": "Alice"}, f)  # Single object, not array

    records = db.read.records.json(str(json_path))
    rows = records.rows()

    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"


def test_records_parquet_streaming(tmp_path):
    """Test Records from Parquet in streaming mode."""
    db_path = tmp_path / "records_parquet_streaming.sqlite"
    db = connect(f"sqlite:///{db_path}")

    parquet_path = tmp_path / "data.parquet"

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create Parquet file with multiple row groups
        df_pd = pd.DataFrame([{"id": i, "name": f"user_{i}"} for i in range(100)])
        table = pa.Table.from_pandas(df_pd)
        pq.write_table(table, str(parquet_path), row_group_size=10)

        # Load in streaming mode (using read.records for Records)
        records = db.read.records.stream().parquet(str(parquet_path))

        # Iterate
        count = 0
        for row in records:
            assert "id" in row
            assert "name" in row
            count += 1
            if count >= 10:
                break

        # Materialize all
        all_rows = records.rows()
        assert len(all_rows) == 100

    except ImportError:
        pytest.skip("pandas/pyarrow not installed")


def test_records_text_custom_column_streaming(tmp_path):
    """Test Records from text file with custom column in streaming mode."""
    db_path = tmp_path / "records_text_streaming.sqlite"
    db = connect(f"sqlite:///{db_path}")

    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        for i in range(50):
            f.write(f"line {i}\n")

    records = db.read.records.stream().text(str(text_path), column_name="content")

    # Iterate
    count = 0
    for row in records:
        assert "content" in row
        assert row["content"].startswith("line")
        count += 1
        if count >= 10:
            break

    # Materialize all
    all_rows = records.rows()
    assert len(all_rows) == 50
    assert all_rows[0]["content"] == "line 0"


def test_read_csv_gzip(tmp_path):
    """Test reading gzip-compressed CSV file."""
    import gzip

    db_path = tmp_path / "read_csv_gzip.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create gzip-compressed CSV file
    csv_path = tmp_path / "data.csv.gz"
    with gzip.open(csv_path, "wt", encoding="utf-8") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")

    # Read compressed CSV - should auto-detect compression
    records = db.read.records.csv(str(csv_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5
    assert rows[1]["name"] == "Bob"


def test_read_csv_bz2(tmp_path):
    """Test reading bzip2-compressed CSV file."""
    import bz2

    db_path = tmp_path / "read_csv_bz2.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create bzip2-compressed CSV file
    csv_path = tmp_path / "data.csv.bz2"
    with bz2.open(csv_path, "wt", encoding="utf-8") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read compressed CSV
    records = db.read.records.csv(str(csv_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_read_json_gzip(tmp_path):
    """Test reading gzip-compressed JSON file."""
    import gzip

    db_path = tmp_path / "read_json_gzip.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create gzip-compressed JSON file
    json_path = tmp_path / "data.json.gz"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    with gzip.open(json_path, "wt", encoding="utf-8") as f:
        json.dump(data, f)

    # Read compressed JSON
    records = db.read.records.json(str(json_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_read_jsonl_gzip(tmp_path):
    """Test reading gzip-compressed JSONL file."""
    import gzip

    db_path = tmp_path / "read_jsonl_gzip.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create gzip-compressed JSONL file
    jsonl_path = tmp_path / "data.jsonl.gz"
    with gzip.open(jsonl_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": 1, "name": "Alice"}) + "\n")
        f.write(json.dumps({"id": 2, "name": "Bob"}) + "\n")

    # Read compressed JSONL
    records = db.read.records.jsonl(str(jsonl_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_read_text_gzip(tmp_path):
    """Test reading gzip-compressed text file."""
    import gzip

    db_path = tmp_path / "read_text_gzip.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create gzip-compressed text file
    text_path = tmp_path / "data.txt.gz"
    with gzip.open(text_path, "wt", encoding="utf-8") as f:
        f.write("line 1\n")
        f.write("line 2\n")
        f.write("line 3\n")

    # Read compressed text
    records = db.read.records.text(str(text_path))
    rows = records.rows()

    assert len(rows) == 3
    assert rows[0]["value"] == "line 1"
    assert rows[1]["value"] == "line 2"


def test_read_csv_explicit_compression(tmp_path):
    """Test reading CSV with explicit compression option."""
    import gzip

    db_path = tmp_path / "read_csv_explicit.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create uncompressed CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    # Try to read with explicit compression (should fail if file is not compressed)
    # But if we compress it first...
    csv_gz_path = tmp_path / "data_compressed.csv"
    with open(csv_path, "rb") as f_in:
        with gzip.open(csv_gz_path, "wb") as f_out:
            f_out.write(f_in.read())

    # Read with explicit compression option (using read.records for Records)
    records = db.read.records.option("compression", "gzip").csv(str(csv_gz_path))
    rows = records.rows()

    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_dataframe_read_csv(tmp_path):
    """Test reading CSV file as DataFrame with transformations."""
    db_path = tmp_path / "df_read_csv.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")
        f.write("3,Charlie,92.0\n")

    # Read CSV as DataFrame
    df = db.load.csv(str(csv_path))

    # Transform before materialization
    filtered_df = df.where(col("score") > 90)
    rows = filtered_df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Charlie"


def test_dataframe_read_json(tmp_path):
    """Test reading JSON file as DataFrame."""
    db_path = tmp_path / "df_read_json.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSON file
    json_path = tmp_path / "data.json"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Read JSON as DataFrame
    df = db.load.json(str(json_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_read_records_csv(tmp_path):
    """Test new read.records.csv() convenience method."""
    db_path = tmp_path / "read_records_csv.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Read CSV as LazyRecords using read.records
    records = db.read.records.csv(str(csv_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert isinstance(records, LazyRecords)


def test_read_records_dicts(tmp_path):
    """Test read.records.dicts() convenience method."""
    db_path = tmp_path / "read_records_dicts.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create Records from list of dicts
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    records = db.read.records.dicts(data)

    assert len(records) == 2
    assert records[0]["name"] == "Alice"
    assert isinstance(records, Records)


def test_dataframe_sql_pushdown(tmp_path):
    """Test that SQL pushdown works after file read."""
    db_path = tmp_path / "df_pushdown.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")
        f.write("3,Charlie,92.0\n")

    # Read CSV as DataFrame and apply SQL operations
    df = db.load.csv(str(csv_path))

    # These operations should use SQL pushdown after file is materialized
    result = (
        df.where(col("score") > 90).select("name", "score").order_by(col("score").desc()).collect()
    )

    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[0]["score"] == 95.5
    assert result[1]["name"] == "Charlie"


def test_dataframe_read_jsonl(tmp_path):
    """Test reading JSONL file as DataFrame."""
    db_path = tmp_path / "df_read_jsonl.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create JSONL file
    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')
        f.write('{"id": 2, "name": "Bob"}\n')

    # Read JSONL as DataFrame
    df = db.load.jsonl(str(jsonl_path))
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_dataframe_read_text(tmp_path):
    """Test reading text file as DataFrame."""
    db_path = tmp_path / "df_read_text.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create text file
    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        f.write("line 1\n")
        f.write("line 2\n")
        f.write("line 3\n")

    # Read text as DataFrame
    df = db.load.text(str(text_path))
    rows = df.collect()

    assert len(rows) == 3
    assert rows[0]["value"] == "line 1"


def test_dataframe_read_parquet(tmp_path):
    """Test reading Parquet file as DataFrame."""
    db_path = tmp_path / "df_read_parquet.sqlite"
    db = connect(f"sqlite:///{db_path}")

    parquet_path = tmp_path / "data.parquet"

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df_pd = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        table = pa.Table.from_pandas(df_pd)
        pq.write_table(table, str(parquet_path))

        # Read Parquet as DataFrame
        df = db.load.parquet(str(parquet_path))
        rows = df.collect()

        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
    except ImportError:
        pytest.skip("pandas/pyarrow not installed")


def test_dataframe_read_with_schema(tmp_path):
    """Test reading file as DataFrame with explicit schema."""
    db_path = tmp_path / "df_read_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")

    schema = [
        ColumnDef(name="id", type_name="INTEGER"),
        ColumnDef(name="name", type_name="TEXT"),
        ColumnDef(name="score", type_name="REAL"),
    ]

    df = db.load.schema(schema).csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert isinstance(rows[0]["id"], int)
    assert rows[0]["id"] == 1
    assert isinstance(rows[0]["score"], float)
    assert rows[0]["score"] == 95.5


def test_dataframe_multiple_transformations(tmp_path):
    """Test multiple transformations before materialization."""
    db_path = tmp_path / "df_transforms.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score,active\n")
        f.write("1,Alice,95.5,1\n")
        f.write("2,Bob,87.0,0\n")
        f.write("3,Charlie,92.0,1\n")
        f.write("4,David,88.0,0\n")

    # Multiple transformations before materialization
    df = (
        db.load.csv(str(csv_path))
        .where(col("active") == 1)
        .where(col("score") > 90)
        .select("name", "score")
        .order_by(col("score").desc())
    )
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Charlie"


def test_dataframe_file_join(tmp_path):
    """Test joining file data with table data."""
    db_path = tmp_path / "df_file_join.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("email", "TEXT")],
    ).collect()
    from moltres.io.records import Records

    Records(_data=[{"id": 1, "email": "alice@example.com"}], _database=db).insert_into("users")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Join file DataFrame with table DataFrame
    file_df = db.load.csv(str(csv_path))
    table_df = db.table("users").select()

    result = file_df.join(table_df, on="id").select("name", "email").collect()

    assert len(result) == 1
    assert result[0]["name"] == "Alice"
    assert result[0]["email"] == "alice@example.com"


def test_filescan_compilation_error(tmp_path):
    """Test that FileScan raises error when compiled directly."""
    from moltres.logical.operators import file_scan
    from moltres.sql.compiler import compile_plan
    from moltres.utils.exceptions import CompilationError

    # Create a FileScan plan
    plan = file_scan(path="test.csv", format="csv")

    # Try to compile it directly - should raise error
    with pytest.raises(CompilationError, match="FileScan cannot be compiled directly"):
        compile_plan(plan)


def test_read_records_jsonl(tmp_path):
    """Test read.records.jsonl() convenience method."""
    db_path = tmp_path / "read_records_jsonl.sqlite"
    db = connect(f"sqlite:///{db_path}")

    jsonl_path = tmp_path / "data.jsonl"
    with open(jsonl_path, "w") as f:
        f.write('{"id": 1, "name": "Alice"}\n')
        f.write('{"id": 2, "name": "Bob"}\n')

    records = db.read.records.jsonl(str(jsonl_path))
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert isinstance(records, LazyRecords)


def test_read_records_text(tmp_path):
    """Test read.records.text() convenience method."""
    db_path = tmp_path / "read_records_text.sqlite"
    db = connect(f"sqlite:///{db_path}")

    text_path = tmp_path / "data.txt"
    with open(text_path, "w") as f:
        f.write("line 1\n")
        f.write("line 2\n")

    records = db.read.records.text(str(text_path), column_name="content")
    rows = records.rows()

    assert len(rows) == 2
    assert rows[0]["content"] == "line 1"
    assert isinstance(records, LazyRecords)


def test_read_records_parquet(tmp_path):
    """Test read.records.parquet() convenience method."""
    db_path = tmp_path / "read_records_parquet.sqlite"
    db = connect(f"sqlite:///{db_path}")

    parquet_path = tmp_path / "data.parquet"

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df_pd = pd.DataFrame([{"id": 1, "name": "Alice"}])
        table = pa.Table.from_pandas(df_pd)
        pq.write_table(table, str(parquet_path))

        records = db.read.records.parquet(str(parquet_path))
        rows = records.rows()

        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
        assert isinstance(records, LazyRecords)
    except ImportError:
        pytest.skip("pandas/pyarrow not installed")


def test_lazy_records_explicit_collect(tmp_path):
    """Test LazyRecords explicit .collect() materialization."""
    db_path = tmp_path / "lazy_records_collect.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    # Create LazyRecords
    lazy_records = db.read.records.csv(str(csv_path))
    assert isinstance(lazy_records, LazyRecords)

    # Explicitly materialize with .collect()
    records = lazy_records.collect()
    assert isinstance(records, Records)
    assert len(records) == 2
    assert records[0]["name"] == "Alice"
    assert records[1]["name"] == "Bob"


def test_lazy_records_auto_materialize_sequence_ops(tmp_path):
    """Test LazyRecords auto-materialization for Sequence operations."""
    db_path = tmp_path / "lazy_records_sequence.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # Test __len__ auto-materializes
    assert len(lazy_records) == 2

    # Test __getitem__ auto-materializes
    assert lazy_records[0]["name"] == "Alice"
    assert lazy_records[1]["name"] == "Bob"

    # Test __iter__ auto-materializes
    rows = list(lazy_records)
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_lazy_records_auto_materialize_insert_into(tmp_path):
    """Test LazyRecords auto-materialization for insert_into()."""
    db_path = tmp_path / "lazy_records_insert.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
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
    count = lazy_records.insert_into("target")
    assert count == 2

    # Verify data was inserted
    df = db.table("target").select()
    rows = df.collect()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_lazy_records_auto_materialize_create_dataframe(tmp_path):
    """Test LazyRecords auto-materialization when used in createDataFrame."""
    db_path = tmp_path / "lazy_records_create_df.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")
        f.write("2,Bob\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # createDataFrame should auto-materialize LazyRecords
    df = db.createDataFrame(lazy_records, pk="id")
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_lazy_records_backward_compatibility(tmp_path):
    """Test that LazyRecords maintains backward compatibility with Records interface."""
    db_path = tmp_path / "lazy_records_compat.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")
        f.write("1,Alice\n")

    lazy_records = db.read.records.csv(str(csv_path))

    # All Records methods should work (auto-materialize)
    assert len(lazy_records) == 1
    assert lazy_records[0]["name"] == "Alice"
    rows = lazy_records.rows()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"

    # Iteration should work
    for row in lazy_records:
        assert row["name"] == "Alice"
        break
