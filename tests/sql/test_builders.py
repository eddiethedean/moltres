"""Tests for SQL builder utilities."""

from __future__ import annotations

import pytest

from moltres.sql.builders import comma_separated, format_literal, quote_identifier
from moltres.utils.exceptions import ValidationError


def test_comma_separated():
    """Test comma_separated helper."""
    assert comma_separated(["a", "b", "c"]) == "a, b, c"
    assert comma_separated(["x"]) == "x"
    assert comma_separated([]) == ""


def test_format_literal():
    """Test format_literal helper."""
    assert format_literal(None) == "NULL"
    assert format_literal(True) == "TRUE"
    assert format_literal(False) == "FALSE"
    assert format_literal(42) == "42"
    assert format_literal(3.14) == "3.14"
    assert format_literal("hello") == "'hello'"
    assert format_literal("it's") == "'it''s'"  # SQL escaping


def test_format_literal_unsupported():
    """Test format_literal with unsupported types."""
    with pytest.raises(TypeError, match="Unsupported literal type"):
        format_literal([])


def test_quote_identifier_basic():
    """Test basic identifier quoting."""
    assert quote_identifier("table") == '"table"'
    assert quote_identifier("table", quote_char="`") == "`table`"
    assert quote_identifier("schema.table") == '"schema"."table"'


def test_quote_identifier_empty():
    """Test that empty identifiers raise ValidationError."""
    with pytest.raises(ValidationError, match="cannot be empty"):
        quote_identifier("")

    with pytest.raises(ValidationError, match="cannot be empty"):
        quote_identifier("   ")


def test_quote_identifier_sql_injection():
    """Test that SQL injection patterns are rejected."""
    # Semicolon injection
    with pytest.raises(ValidationError, match="invalid characters"):
        quote_identifier("table; DROP TABLE users;")

    # Quote injection
    with pytest.raises(ValidationError, match="invalid characters"):
        quote_identifier("table'")

    # Double quote injection
    with pytest.raises(ValidationError, match="invalid characters"):
        quote_identifier('table"')

    # Backslash injection
    with pytest.raises(ValidationError, match="invalid characters"):
        quote_identifier("table\\")


def test_quote_identifier_empty_part():
    """Test that qualified names with empty parts are rejected."""
    with pytest.raises(ValidationError, match="empty part"):
        quote_identifier("schema..table")

    with pytest.raises(ValidationError, match="empty part"):
        quote_identifier(".table")

    with pytest.raises(ValidationError, match="empty part"):
        quote_identifier("schema.")


def test_quote_identifier_valid():
    """Test that valid identifiers are accepted."""
    assert quote_identifier("table_name") == '"table_name"'
    assert quote_identifier("table123") == '"table123"'
    assert quote_identifier("_private") == '"_private"'
    assert quote_identifier("schema.table") == '"schema"."table"'
    assert quote_identifier("schema.table.column") == '"schema"."table"."column"'
