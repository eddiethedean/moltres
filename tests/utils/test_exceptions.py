"""Tests for exception hierarchy."""

from __future__ import annotations

import pytest

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

    assert str(CompilationError(msg)) == msg
    assert str(ExecutionError(msg)) == msg
    assert str(ValidationError(msg)) == msg
    assert str(SchemaError(msg)) == msg
    assert str(DatabaseConnectionError(msg)) == msg
    assert str(UnsupportedOperationError(msg)) == msg


def test_exception_chaining():
    """Test that exceptions can be chained with cause."""
    cause = ValueError("Original error")
    try:
        raise ExecutionError("Wrapped error") from cause
    except ExecutionError as error:
        assert error.__cause__ is cause
        assert str(error) == "Wrapped error"
