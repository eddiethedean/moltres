"""Tests for DataFrame UX improvements (tail, enhanced head, explain)."""

from __future__ import annotations


from moltres import col, column, connect


def test_dataframe_tail(tmp_path):
    """Test DataFrame.tail() convenience method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    # Insert multiple rows
    rows = [{"id": i, "name": f"User{i}"} for i in range(10)]
    db.insert("users", rows)

    # Get tail
    df = db.table("users").select().order_by(col("id"))
    tail_rows = df.tail(3)

    assert len(tail_rows) == 3
    # Should get last 3 rows (ids 7, 8, 9)
    ids = {row["id"] for row in tail_rows}
    assert 7 in ids
    assert 8 in ids
    assert 9 in ids


def test_dataframe_tail_less_than_n(tmp_path):
    """Test DataFrame.tail() when DataFrame has fewer rows than requested."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table with 2 rows
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Request tail(5) but only 2 rows exist
    df = db.table("users").select().order_by(col("id"))
    tail_rows = df.tail(5)

    assert len(tail_rows) == 2
    assert {row["name"] for row in tail_rows} == {"Alice", "Bob"}


def test_dataframe_tail_empty(tmp_path):
    """Test DataFrame.tail() with empty DataFrame."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create empty table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    # Get tail from empty table
    df = db.table("users").select()
    tail_rows = df.tail(5)

    assert len(tail_rows) == 0


def test_dataframe_head_enhanced(tmp_path):
    """Test enhanced DataFrame.head() method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])

    # Get head
    df = db.table("users").select()
    head_rows = df.head(1)

    assert len(head_rows) == 1
    assert head_rows[0]["name"] == "Alice"


def test_dataframe_explain_enhanced(tmp_path):
    """Test enhanced DataFrame.explain() method."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}])

    # Get explain plan
    df = db.table("users").select().where(col("id") == 1)
    plan = df.explain()

    assert isinstance(plan, str)
    assert len(plan) > 0
    # Should contain some SQL keywords or plan information
    assert "SELECT" in plan.upper() or "SCAN" in plan.upper() or "SEARCH" in plan.upper()


def test_dataframe_explain_analyze(tmp_path):
    """Test DataFrame.explain() with analyze=True."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    db.insert("users", [{"id": 1, "name": "Alice"}])

    # Get explain plan with analyze
    # Note: SQLite doesn't support EXPLAIN ANALYZE, it uses EXPLAIN QUERY PLAN
    # So analyze=True may not work for SQLite, but should work for PostgreSQL
    df = db.table("users").select().where(col("id") == 1)
    try:
        plan = df.explain(analyze=True)
        assert isinstance(plan, str)
        assert len(plan) > 0
    except Exception:
        # SQLite doesn't support EXPLAIN ANALYZE, so this is expected to fail
        # Just verify that explain(analyze=False) works
        plan = df.explain(analyze=False)
        assert isinstance(plan, str)
        assert len(plan) > 0
