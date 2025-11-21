"""Tests for extended DataFrame write methods (insertInto, file formats, etc.)."""

import json

import pytest

from moltres import column, connect


def test_insert_into_existing_table(tmp_path):
    """Test insertInto() with existing table."""
    db_path = tmp_path / "insert_into.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table first
    target = db.create_table(
        "target",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    )
    target.insert([{"id": 1, "name": "Alice"}])

    # Create source DataFrame
    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    )
    source.insert([{"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}])

    # Insert into existing table
    df = db.table("source").select()
    df.write.insertInto("target")

    # Verify data was inserted
    rows = target.select().collect()
    assert len(rows) == 3
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"
    assert rows[2]["name"] == "Charlie"


def test_insert_into_nonexistent_table(tmp_path):
    """Test insertInto() raises error for non-existent table."""
    db_path = tmp_path / "insert_into_error.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER")])
    source.insert([{"id": 1}])

    df = db.table("source").select()
    with pytest.raises(ValueError, match="does not exist"):
        df.write.insertInto("nonexistent")


def test_save_csv(tmp_path):
    """Test saving DataFrame as CSV."""
    db_path = tmp_path / "save_csv.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT"), column("score", "REAL")],
    )
    source.insert(
        [
            {"id": 1, "name": "Alice", "score": 95.5},
            {"id": 2, "name": "Bob", "score": 87.0},
        ]
    )

    df = db.table("source").select()
    csv_path = tmp_path / "output.csv"
    df.write.csv(str(csv_path))

    # Verify CSV file
    assert csv_path.exists()
    with open(csv_path) as f:
        lines = f.readlines()
        assert len(lines) == 3  # header + 2 data rows
        assert "id,name,score" in lines[0]
        assert "1,Alice,95.5" in lines[1]


def test_save_csv_with_options(tmp_path):
    """Test saving CSV with custom options."""
    db_path = tmp_path / "save_csv_options.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER"), column("name", "TEXT")])
    source.insert([{"id": 1, "name": "Alice"}])

    df = db.table("source").select()
    csv_path = tmp_path / "output.csv"
    df.write.option("header", False).option("delimiter", "|").csv(str(csv_path))

    # Verify CSV file
    with open(csv_path) as f:
        content = f.read()
        assert "|" in content
        assert "id" not in content  # No header


def test_save_json(tmp_path):
    """Test saving DataFrame as JSON."""
    db_path = tmp_path / "save_json.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER"), column("name", "TEXT")])
    source.insert([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    df = db.table("source").select()
    json_path = tmp_path / "output.json"
    df.write.json(str(json_path))

    # Verify JSON file
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[1]["name"] == "Bob"


def test_save_jsonl(tmp_path):
    """Test saving DataFrame as JSONL."""
    db_path = tmp_path / "save_jsonl.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER"), column("name", "TEXT")])
    source.insert([{"id": 1, "name": "Alice"}])

    df = db.table("source").select()
    jsonl_path = tmp_path / "output.jsonl"
    df.write.jsonl(str(jsonl_path))

    # Verify JSONL file
    assert jsonl_path.exists()
    with open(jsonl_path) as f:
        lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["name"] == "Alice"


def test_save_with_format_inference(tmp_path):
    """Test save() method with format inference from extension."""
    db_path = tmp_path / "save_format.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER")])
    source.insert([{"id": 1}])

    df = db.table("source").select()

    # Test CSV inference
    csv_path = tmp_path / "data.csv"
    df.write.save(str(csv_path))
    assert csv_path.exists()

    # Test JSON inference
    json_path = tmp_path / "data.json"
    df.write.save(str(json_path))
    assert json_path.exists()


def test_save_with_explicit_format(tmp_path):
    """Test save() method with explicit format."""
    db_path = tmp_path / "save_explicit.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER")])
    source.insert([{"id": 1}])

    df = db.table("source").select()
    output_path = tmp_path / "data.txt"
    df.write.save(str(output_path), format="csv")
    assert output_path.exists()


def test_save_partitioned_csv(tmp_path):
    """Test saving partitioned CSV files."""
    db_path = tmp_path / "partitioned.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("country", "TEXT"), column("value", "INTEGER")],
    )
    source.insert(
        [
            {"id": 1, "country": "US", "value": 100},
            {"id": 2, "country": "US", "value": 200},
            {"id": 3, "country": "UK", "value": 150},
        ]
    )

    df = db.table("source").select()
    output_path = tmp_path / "partitioned"
    df.write.partitionBy("country").csv(str(output_path))

    # Verify partitioned structure
    us_dir = tmp_path / "partitioned" / "country=US"
    uk_dir = tmp_path / "partitioned" / "country=UK"
    assert us_dir.exists()
    assert uk_dir.exists()
    assert (us_dir / "data.csv").exists()
    assert (uk_dir / "data.csv").exists()

    # Verify data in partitions
    with open(us_dir / "data.csv") as f:
        lines = f.readlines()
        assert len(lines) == 3  # header + 2 rows
    with open(uk_dir / "data.csv") as f:
        lines = f.readlines()
        assert len(lines) == 2  # header + 1 row


def test_save_partitioned_json(tmp_path):
    """Test saving partitioned JSON files."""
    db_path = tmp_path / "partitioned_json.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER"), column("category", "TEXT")])
    source.insert(
        [
            {"id": 1, "category": "A"},
            {"id": 2, "category": "B"},
        ]
    )

    df = db.table("source").select()
    output_path = tmp_path / "partitioned_json"
    df.write.partitionBy("category").json(str(output_path))

    # Verify partitioned structure
    a_dir = tmp_path / "partitioned_json" / "category=A"
    b_dir = tmp_path / "partitioned_json" / "category=B"
    assert (a_dir / "data.json").exists()
    assert (b_dir / "data.json").exists()


def test_save_parquet_requires_dependencies(tmp_path):
    """Test that parquet save requires pandas/pyarrow."""
    db_path = tmp_path / "parquet_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table("source", [column("id", "INTEGER")])
    source.insert([{"id": 1}])

    df = db.table("source").select()
    parquet_path = tmp_path / "output.parquet"

    # This will either work (if dependencies installed) or raise RuntimeError
    try:
        df.write.parquet(str(parquet_path))
        # If we get here, dependencies are installed - verify file exists
        assert parquet_path.exists()
    except RuntimeError as e:
        # Expected if dependencies not installed
        assert "pandas" in str(e) or "pyarrow" in str(e)


def test_insert_into_alias(tmp_path):
    """Test insert_into() alias method."""
    db_path = tmp_path / "insert_into_alias.sqlite"
    db = connect(f"sqlite:///{db_path}")

    target = db.create_table("target", [column("id", "INTEGER")])
    source = db.create_table("source", [column("id", "INTEGER")])
    source.insert([{"id": 1}])

    df = db.table("source").select()
    df.write.insert_into("target")

    rows = target.select().collect()
    assert len(rows) == 1
