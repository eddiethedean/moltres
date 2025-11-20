"""Tests for DataFrame read operations."""

import json

import pytest

from moltres import column, connect
from moltres.table.schema import ColumnDef


def test_read_table(tmp_path):
    """Test reading from database table."""
    db_path = tmp_path / "read_table.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    )
    source.insert([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Read using read.table()
    df = db.read.table("source")
    rows = df.collect()

    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"


def test_read_csv(tmp_path):
    """Test reading CSV file."""
    db_path = tmp_path / "read_csv.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create CSV file
    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id,name,score\n")
        f.write("1,Alice,95.5\n")
        f.write("2,Bob,87.0\n")

    # Read CSV
    df = db.read.csv(str(csv_path))
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
    df = db.read.option("delimiter", "|").csv(str(csv_path))
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
    df = db.read.schema(schema).option("header", False).csv(str(csv_path))
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

    # Read JSON
    df = db.read.json(str(json_path))
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

    # Read JSONL
    df = db.read.jsonl(str(jsonl_path))
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

    # Read text
    df = db.read.text(str(text_path))
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

    df = db.read.text(str(text_path), column_name="line")
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

    df = db.read.format("csv").load(str(csv_path))
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

    df = db.read.format("json").load(str(json_path))
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

    df = db.read.schema(schema).csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert isinstance(rows[0]["id"], int)
    assert rows[0]["id"] == 1


def test_read_missing_file(tmp_path):
    """Test error handling for missing file."""
    db_path = tmp_path / "read_missing.sqlite"
    db = connect(f"sqlite:///{db_path}")

    with pytest.raises(FileNotFoundError):
        db.read.csv(str(tmp_path / "nonexistent.csv"))


def test_read_empty_csv(tmp_path):
    """Test reading empty CSV file."""
    db_path = tmp_path / "read_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w") as f:
        f.write("id,name\n")

    # Should raise error without explicit schema
    with pytest.raises(ValueError, match="empty"):
        db.read.csv(str(csv_path))

    # Should work with explicit schema
    schema = [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")]
    df = db.read.schema(schema).csv(str(csv_path))
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

        df = db.read.parquet(str(parquet_path))
        rows = df.collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"
    except ImportError:
        # If dependencies not installed, test that we get appropriate error
        with pytest.raises(RuntimeError, match="pandas|pyarrow"):
            db.read.parquet(str(parquet_path))


def test_read_chained_options(tmp_path):
    """Test chaining options and schema."""
    db_path = tmp_path / "read_chained.sqlite"
    db = connect(f"sqlite:///{db_path}")

    csv_path = tmp_path / "data.csv"
    with open(csv_path, "w") as f:
        f.write("id|name\n")
        f.write("1|Alice\n")

    schema = [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")]

    df = db.read.schema(schema).option("delimiter", "|").csv(str(csv_path))
    rows = df.collect()

    assert len(rows) == 1
    assert isinstance(rows[0]["id"], int)
