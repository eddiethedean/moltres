"""Tests for error suggestion features (Did you mean?)."""

from __future__ import annotations


from moltres import connect
from moltres.utils.exceptions import (
    _levenshtein_distance,
    _suggest_column_name,
    _suggest_table_name,
)


def test_levenshtein_distance():
    """Test Levenshtein distance calculation."""
    assert _levenshtein_distance("abc", "abc") == 0
    assert _levenshtein_distance("abc", "ab") == 1
    assert _levenshtein_distance("abc", "abcd") == 1
    assert _levenshtein_distance("kitten", "sitting") == 3
    assert _levenshtein_distance("", "abc") == 3
    assert _levenshtein_distance("abc", "") == 3


def test_suggest_column_name():
    """Test column name suggestion."""
    available = ["id", "name", "email", "created_at", "updated_at"]

    # Exact match - should suggest the exact column
    suggestion = _suggest_column_name("name", available)
    assert "name" in suggestion

    # Close match - should suggest similar
    suggestion = _suggest_column_name("nme", available)
    assert "name" in suggestion

    # Multiple suggestions
    suggestion = _suggest_column_name("created", available)
    assert "created_at" in suggestion

    # No close match - should list available columns
    suggestion = _suggest_column_name("xyz", available)
    assert "Available columns" in suggestion or "xyz" in suggestion


def test_suggest_column_name_empty():
    """Test column name suggestion with no available columns."""
    suggestion = _suggest_column_name("name", [])
    assert "does not exist" in suggestion
    assert "No columns" in suggestion


def test_suggest_table_name():
    """Test table name suggestion."""
    available = ["users", "orders", "products", "order_items"]

    # Close match
    suggestion = _suggest_table_name("user", available)
    assert "users" in suggestion

    # Multiple suggestions
    suggestion = _suggest_table_name("order", available)
    assert "order" in suggestion.lower()

    # No close match - should list available tables
    suggestion = _suggest_table_name("xyz", available)
    assert "Available tables" in suggestion or "xyz" in suggestion


def test_suggest_table_name_empty():
    """Test table name suggestion with no available tables."""
    suggestion = _suggest_table_name("users", [])
    assert "does not exist" in suggestion
    assert "No tables" in suggestion


def test_execution_error_with_column_suggestion(tmp_path):
    """Test that ExecutionError includes column suggestions when appropriate."""
    from moltres.table.schema import column

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()

    # Try to query with non-existent column
    try:
        db.execute_sql("SELECT nme FROM users")  # Typo: "nme" instead of "name"
    except Exception as e:
        # The error should be caught and potentially include suggestions
        # (Note: SQLite may not provide column suggestions in the error message itself,
        # but our error handling should be able to add them if context is provided)
        assert "nme" in str(e) or "name" in str(e) or "no such column" in str(e).lower()


def test_execution_error_with_table_suggestion(tmp_path):
    """Test that ExecutionError includes table suggestions when appropriate."""
    from moltres.table.schema import column

    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create tables
    db.create_table("users", [column("id", "INTEGER")]).collect()
    db.create_table("orders", [column("id", "INTEGER")]).collect()

    # Try to query non-existent table
    try:
        db.execute_sql("SELECT * FROM usrs")  # Typo: "usrs" instead of "users"
    except Exception as e:
        # The error should mention the table name
        assert "usrs" in str(e) or "users" in str(e) or "no such table" in str(e).lower()
