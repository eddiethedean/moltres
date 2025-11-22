"""Security tests for SQL injection prevention."""

from __future__ import annotations

import pytest

from moltres import connect
from moltres.utils.exceptions import ValidationError


def test_sql_injection_table_name(tmp_path):
    """Test that SQL injection attempts in table names are prevented."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Attempt SQL injection via table name
    malicious_names = [
        "users; DROP TABLE users;--",
        "users' OR '1'='1",
        'users"; DROP TABLE users;--',
        "users; DELETE FROM users;",
    ]

    for name in malicious_names:
        with pytest.raises(ValidationError, match="invalid characters"):
            db.table(name)


def test_sql_injection_column_name(tmp_path):
    """Test that SQL injection attempts in column names are prevented."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()

    table = db.table("test")
    table.insert([{"id": 1, "name": "Alice"}]).collect()

    # Attempt SQL injection via column name - validation happens during SQL compilation
    malicious_columns = [
        "name; DROP TABLE test;--",
        "name' OR '1'='1",
    ]

    for col_name in malicious_columns:
        # Validation happens when SQL is compiled, not when select() is called
        df = table.select(col_name)
        with pytest.raises(ValidationError, match="Invalid column name|invalid characters"):
            df.to_sql()  # This triggers SQL compilation and validation


def test_sql_injection_parameterized_queries(tmp_path):
    """Test that parameterized queries prevent injection in values."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()

    table = db.table("test")

    # These should be safe because values are parameterized
    # The injection attempt should be stored as a literal value, not executed
    malicious_values = [
        "'; DROP TABLE test;--",
        "' OR '1'='1",
        "'; DELETE FROM test;--",
    ]

    for idx, value in enumerate(malicious_values):
        # This should insert the value as-is, not execute it
        table.insert([{"id": idx + 1, "name": value}]).collect()

    # Verify the values were stored as literals (not executed)
    df = table.select()
    results = df.collect()
    assert len(results) == 3
    # Verify all rows exist and have the malicious strings stored as literal values
    names = [r["name"] for r in results]
    assert any("DROP" in str(name) for name in names)
    assert any("OR" in str(name) for name in names)


def test_identifier_validation_comprehensive():
    """Comprehensive test of identifier validation."""
    from moltres.sql.builders import quote_identifier

    # Valid identifiers
    valid = [
        "table_name",
        "table123",
        "_private",
        "schema.table",
        "schema.table.column",
        "CamelCase",
        "table_name_123",
    ]

    for identifier in valid:
        # Should not raise
        result = quote_identifier(identifier)
        assert result.startswith('"') and result.endswith('"')

    # Invalid identifiers
    invalid = [
        "",  # Empty
        "   ",  # Whitespace only
        "table; DROP",  # Semicolon
        "table'",  # Single quote
        'table"',  # Double quote
        "table\\",  # Backslash
    ]

    for identifier in invalid:
        with pytest.raises(ValidationError):
            quote_identifier(identifier)

    # Note: Some characters like newlines, tabs, and SQL comments in identifiers
    # might be handled differently by different databases, so we focus on
    # the most critical injection patterns (semicolons, quotes, backslashes)
