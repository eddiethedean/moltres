"""Tests for exception hierarchy."""

from __future__ import annotations

from moltres.utils.exceptions import (
    CompilationError,
    DatabaseConnectionError,
    ExecutionError,
    MoltresError,
    SchemaError,
    UnsupportedOperationError,
    ValidationError,
)


def test_exception_hierarchy():
    """Test that all exceptions inherit from MoltresError."""
    assert issubclass(CompilationError, MoltresError)
    assert issubclass(ExecutionError, MoltresError)
    assert issubclass(ValidationError, MoltresError)
    assert issubclass(SchemaError, MoltresError)
    assert issubclass(DatabaseConnectionError, MoltresError)
    assert issubclass(UnsupportedOperationError, MoltresError)


def test_exception_instantiation():
    """Test that exceptions can be instantiated with messages."""
    msg = "Test error message"

    # All exceptions may have auto-generated suggestions, so we check that the message is included
    assert msg in str(CompilationError(msg))
    assert msg in str(ExecutionError(msg))
    assert msg in str(ValidationError(msg))
    assert msg in str(SchemaError(msg))
    assert msg in str(DatabaseConnectionError(msg))
    assert msg in str(UnsupportedOperationError(msg))

    # Test that exceptions can be created without suggestions
    assert msg in str(CompilationError(msg, suggestion=None))
    assert msg in str(ExecutionError(msg, suggestion=None))
    assert msg in str(ValidationError(msg, suggestion=None))


def test_exception_chaining():
    """Test that exceptions can be chained with cause."""
    cause = ValueError("Original error")
    try:
        raise ExecutionError("Wrapped error") from cause
    except ExecutionError as error:
        assert error.__cause__ is cause
        assert str(error) == "Wrapped error"
