"""Tests for DataFrame write operations."""

from moltres import col, column, connect
from moltres.table.schema import ColumnDef


def test_write_append_mode(tmp_path):
    """Test writing DataFrame in append mode."""
    db_path = tmp_path / "write_append.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source table
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    )
    source.insert(
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    )

    # Write to new table
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Verify data was written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Append more data
    source.insert([{"id": 3, "name": "Charlie"}])
    df2 = db.table("source").select().where(col("id") == 3)
    df2.write.mode("append").save_as_table("target")

    # Verify append worked
    rows = target.select().collect()
    assert len(rows) == 3


def test_write_overwrite_mode(tmp_path):
    """Test writing DataFrame in overwrite mode."""
    db_path = tmp_path / "write_overwrite.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate initial table
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "INTEGER"),
        ],
    )
    source.insert([{"id": 1, "value": 100}])

    # Write initial data
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Verify initial write
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["value"] == 100

    # Overwrite with new data
    source.insert([{"id": 2, "value": 200}])
    df2 = db.table("source").select()
    df2.write.mode("overwrite").save_as_table("target")

    # Verify overwrite (should only have new data)
    rows = target.select().collect()
    assert len(rows) == 2
    assert rows[0]["value"] == 100
    assert rows[1]["value"] == 200


def test_write_error_if_exists_mode(tmp_path):
    """Test error_if_exists mode raises error when table exists."""
    db_path = tmp_path / "write_error.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table first
    source = db.create_table(
        "source",
        [column("id", "INTEGER")],
    )
    source.insert([{"id": 1}])

    # Write once (creates table)
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Try to write again with error_if_exists
    import pytest

    with pytest.raises(ValueError, match="already exists"):
        df.write.mode("error_if_exists").save_as_table("target")


def test_write_with_explicit_schema(tmp_path):
    """Test writing with explicit schema."""
    db_path = tmp_path / "write_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source with different types
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("score", "REAL"),
        ],
    )
    source.insert([{"id": 1, "name": "Alice", "score": 95.5}])

    # Write with explicit schema
    explicit_schema = [
        ColumnDef(name="id", type_name="INTEGER", nullable=False),
        ColumnDef(name="name", type_name="TEXT", nullable=False),
        ColumnDef(name="score", type_name="REAL", nullable=True),
    ]

    df = db.table("source").select()
    df.write.schema(explicit_schema).save_as_table("target")

    # Verify table was created with correct schema
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5


def test_write_empty_dataframe(tmp_path):
    """Test writing empty DataFrame creates table but inserts nothing."""
    db_path = tmp_path / "write_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source table
    db.create_table(
        "source",
        [column("id", "INTEGER")],
    )

    # Create empty DataFrame
    df = db.table("source").select().where(col("id") == 999)

    # Write empty DataFrame with explicit schema (required for empty DataFrames)
    explicit_schema = [ColumnDef(name="id", type_name="INTEGER", nullable=True)]
    df.write.schema(explicit_schema).save_as_table("target")

    # Verify table exists but is empty
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 0


def test_write_with_transformed_columns(tmp_path):
    """Test writing DataFrame with transformed/aliased columns."""
    db_path = tmp_path / "write_transformed.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("first_name", "TEXT"),
            column("last_name", "TEXT"),
        ],
    )
    source.insert([{"id": 1, "first_name": "John", "last_name": "Doe"}])

    # Create DataFrame with transformed columns
    df = db.table("source").select(
        col("id"),
        (col("first_name") + " " + col("last_name")).alias("full_name"),
    )

    # Write transformed DataFrame
    df.write.save_as_table("target")

    # Verify data
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert "id" in rows[0]
    assert "full_name" in rows[0]


def test_write_chained_api(tmp_path):
    """Test chained write API similar to PySpark."""
    db_path = tmp_path / "write_chained.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER")],
    )
    source.insert([{"id": 1}])

    df = db.table("source").select()
    df.write.mode("append").option("test", "value").save_as_table("target")

    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
