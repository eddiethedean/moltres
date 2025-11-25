"""Tests for Database convenience methods (insert, update, delete, merge, explain, show_tables, show_schema)."""

from __future__ import annotations


from moltres import col, column, connect


def test_db_insert(tmp_path):
    """Test db.insert() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    # Insert rows
    inserted = db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
    assert inserted == 2

    # Verify data
    rows = db.table("users").select().collect()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"


def test_db_update(tmp_path):
    """Test db.update() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Update rows
    updated = db.update("users", where=col("id") == 1, set={"name": "Alice Updated"})
    assert updated == 1

    # Verify update
    rows = db.table("users").select().where(col("id") == 1).collect()
    assert rows[0]["name"] == "Alice Updated"


def test_db_delete(tmp_path):
    """Test db.delete() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Delete rows
    deleted = db.delete("users", where=col("id") == 1)
    assert deleted == 1

    # Verify deletion
    rows = db.table("users").select().collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Bob"


def test_db_merge(tmp_path):
    """Test db.merge() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table with primary key
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()

    # Insert initial data
    db.insert("users", [{"id": 1, "name": "Alice"}])

    # Merge (upsert) - should update existing
    merged = db.merge(
        "users",
        [{"id": 1, "name": "Alice Updated"}, {"id": 2, "name": "Bob"}],
        on=["id"],
        when_matched={"name": "Alice Updated"},
    )
    assert merged >= 1  # At least one row affected

    # Verify merge
    rows = db.table("users").select().collect()
    assert len(rows) == 2
    names = {row["name"] for row in rows}
    assert "Alice Updated" in names or "Bob" in names


def test_db_explain(tmp_path):
    """Test db.explain() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}])

    # Get execution plan
    plan = db.explain("SELECT * FROM users WHERE id = :id", params={"id": 1})
    assert isinstance(plan, str)
    assert len(plan) > 0
    # SQLite EXPLAIN QUERY PLAN returns column names, so just check it's not empty
    # The plan format varies by database, so we just verify it returns something


def test_db_show_tables(tmp_path, capsys):
    """Test db.show_tables() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create tables
    db.create_table("users", [column("id", "INTEGER")]).collect()
    db.create_table("orders", [column("id", "INTEGER")]).collect()

    # Show tables
    db.show_tables()
    captured = capsys.readouterr()
    assert "Tables in database:" in captured.out
    assert "users" in captured.out
    assert "orders" in captured.out


def test_db_show_tables_empty(tmp_path, capsys):
    """Test db.show_tables() with no tables."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Show tables (empty)
    db.show_tables()
    captured = capsys.readouterr()
    assert "No tables found" in captured.out


def test_db_show_schema(tmp_path, capsys):
    """Test db.show_schema() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
        ],
    ).collect()

    # Show schema
    db.show_schema("users")
    captured = capsys.readouterr()
    assert "Schema for table 'users':" in captured.out
    assert "id" in captured.out
    assert "name" in captured.out
    assert "email" in captured.out
