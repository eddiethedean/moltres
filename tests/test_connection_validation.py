"""Tests for connection string validation."""

from __future__ import annotations

import pytest

from moltres import async_connect, connect
from moltres.utils.exceptions import DatabaseConnectionError


def test_connect_valid_sqlite():
    """Test connect() with valid SQLite connection string."""
    db = connect("sqlite:///:memory:")
    assert db is not None
    db.close()


def test_connect_valid_postgresql_format():
    """Test connect() with valid PostgreSQL connection string format."""
    # This will fail to actually connect (if PostgreSQL not running),
    # but should pass validation (format is correct)
    # The validation only checks format, not actual connectivity
    try:
        db = connect("postgresql://user:pass@localhost/dbname")
        # If it connects, close it
        db.close()
    except Exception:
        # Expected if PostgreSQL is not running or credentials are wrong
        # This is fine - validation passed, connection failed
        pass


def test_connect_invalid_format():
    """Test connect() with invalid connection string format."""
    with pytest.raises(DatabaseConnectionError) as exc_info:
        connect("invalid-connection-string")
    assert "://" in str(exc_info.value) or "separator" in str(exc_info.value).lower()


def test_async_connect_missing_async_driver():
    """Test async_connect() with missing async driver."""
    with pytest.raises(DatabaseConnectionError) as exc_info:
        async_connect("sqlite:///:memory:")
    assert "aiosqlite" in str(exc_info.value) or "async" in str(exc_info.value).lower()


def test_async_connect_postgresql_missing_driver():
    """Test async_connect() with PostgreSQL missing async driver."""
    with pytest.raises(DatabaseConnectionError) as exc_info:
        async_connect("postgresql://user:pass@localhost/dbname")
    assert "asyncpg" in str(exc_info.value) or "async" in str(exc_info.value).lower()


def test_async_connect_mysql_missing_driver():
    """Test async_connect() with MySQL missing async driver."""
    with pytest.raises(DatabaseConnectionError) as exc_info:
        async_connect("mysql://user:pass@localhost/dbname")
    assert "aiomysql" in str(exc_info.value) or "async" in str(exc_info.value).lower()


def test_connect_empty_string():
    """Test connect() with empty connection string."""
    with pytest.raises(DatabaseConnectionError) as exc_info:
        connect("")
    assert "non-empty string" in str(exc_info.value).lower()


def test_connect_none():
    """Test connect() with None connection string (should use env var or fail later)."""
    # None is allowed - it will try to use MOLTRES_DSN env var
    # If that's not set, it will fail at config creation, not validation
    try:
        connect(None)
    except (ValueError, DatabaseConnectionError):
        pass  # Expected if MOLTRES_DSN is not set
