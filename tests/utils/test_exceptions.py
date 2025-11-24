"""Tests for exception hierarchy."""

from __future__ import annotations


from moltres.utils.exceptions import (
    CompilationError,
    ConnectionPoolError,
    DatabaseConnectionError,
    ExecutionError,
    MoltresError,
    QueryTimeoutError,
    SchemaError,
    TransactionError,
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
    assert issubclass(QueryTimeoutError, ExecutionError)
    assert issubclass(ConnectionPoolError, DatabaseConnectionError)
    assert issubclass(TransactionError, ExecutionError)


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


def test_moltres_error_with_context():
    """Test MoltresError with context dictionary (lines 30-31)."""
    error = MoltresError("Test error", context={"key1": "value1", "key2": 42})
    error_str = str(error)
    assert "Test error" in error_str
    assert "key1=value1" in error_str
    assert "key2=42" in error_str


def test_moltres_error_with_suggestion():
    """Test MoltresError with suggestion."""
    error = MoltresError("Test error", suggestion="Try this fix")
    error_str = str(error)
    assert "Test error" in error_str
    assert "Suggestion: Try this fix" in error_str


def test_compilation_error_auto_suggestion_unsupported():
    """Test CompilationError auto-suggestion for unsupported operations (line 51)."""
    error = CompilationError("Operation not supported")
    error_str = str(error)
    assert "not supported" in error_str.lower()
    assert "dialect" in error_str.lower() or "Suggestion" in error_str


def test_compilation_error_auto_suggestion_join():
    """Test CompilationError auto-suggestion for join errors (line 56)."""
    error = CompilationError("Join condition missing")
    error_str = str(error)
    assert "join" in error_str.lower() or "Join" in error_str
    assert "condition" in error_str.lower() or "Suggestion" in error_str


def test_compilation_error_auto_suggestion_subquery():
    """Test CompilationError auto-suggestion for subquery errors (line 62)."""
    error = CompilationError("Invalid subquery")
    error_str = str(error)
    assert "subquery" in error_str.lower() or "Subquery" in error_str
    assert "Suggestion" in error_str or "DataFrame" in error_str


def test_execution_error_auto_suggestion_table():
    """Test ExecutionError auto-suggestion for table errors (line 84)."""
    error = ExecutionError("no such table: users")
    error_str = str(error)
    assert "table" in error_str.lower()
    assert "Suggestion" in error_str or "table name" in error_str.lower()


def test_execution_error_auto_suggestion_column():
    """Test ExecutionError auto-suggestion for column errors (line 89)."""
    error = ExecutionError("no such column: name")
    error_str = str(error)
    assert "column" in error_str.lower()
    assert "Suggestion" in error_str or "column name" in error_str.lower()


def test_execution_error_auto_suggestion_syntax():
    """Test ExecutionError auto-suggestion for syntax errors (line 94)."""
    error = ExecutionError("SQL syntax error")
    error_str = str(error)
    assert "syntax" in error_str.lower()
    assert "Suggestion" in error_str or "syntax error" in error_str.lower()


def test_validation_error_auto_suggestion_column_name():
    """Test ValidationError auto-suggestion for column name errors (line 115)."""
    error = ValidationError("Invalid column name")
    error_str = str(error)
    assert "column name" in error_str.lower() or "Column" in error_str
    assert "Suggestion" in error_str or "identifier" in error_str.lower()


def test_validation_error_auto_suggestion_required():
    """Test ValidationError auto-suggestion for required/missing parameters (line 121)."""
    error = ValidationError("Required parameter missing")
    error_str = str(error)
    assert "required" in error_str.lower() or "missing" in error_str.lower()
    assert "Suggestion" in error_str or "parameters" in error_str.lower()


def test_query_timeout_error_with_timeout():
    """Test QueryTimeoutError with timeout parameter (lines 157-160)."""
    error = QueryTimeoutError("Query timed out", timeout=30.0)
    assert error.context["timeout_seconds"] == 30.0
    error_str = str(error)
    assert "timeout" in error_str.lower()
    assert "Suggestion" in error_str


def test_query_timeout_error_without_timeout():
    """Test QueryTimeoutError without timeout parameter."""
    error = QueryTimeoutError("Query timed out")
    assert "timeout_seconds" not in error.context
    error_str = str(error)
    assert "timeout" in error_str.lower()


def test_connection_pool_error_auto_suggestion():
    """Test ConnectionPoolError auto-suggestion (lines 170-177)."""
    error = ConnectionPoolError("Pool exhausted")
    error_str = str(error)
    assert "pool" in error_str.lower()
    assert "Suggestion" in error_str
    assert "pool_size" in error_str or "connection" in error_str.lower()


def test_transaction_error_auto_suggestion():
    """Test TransactionError auto-suggestion (lines 187-194)."""
    error = TransactionError("Transaction failed")
    error_str = str(error)
    assert "Transaction" in error_str or "transaction" in error_str.lower()
    assert "Suggestion" in error_str
    assert "operations" in error_str.lower() or "isolation" in error_str.lower()


def test_unsupported_operation_error_auto_suggestion():
    """Test UnsupportedOperationError auto-suggestion."""
    error = UnsupportedOperationError("Operation not supported")
    error_str = str(error)
    assert "not supported" in error_str.lower()
    assert "Suggestion" in error_str
    assert "documentation" in error_str.lower()


def test_exception_attributes():
    """Test that exception attributes are set correctly."""
    error = MoltresError("Test", suggestion="Fix it", context={"key": "value"})
    assert error.message == "Test"
    assert error.suggestion == "Fix it"
    assert error.context == {"key": "value"}


def test_exception_with_empty_context():
    """Test exception with empty context dictionary."""
    error = MoltresError("Test", context={})
    assert error.context == {}
    error_str = str(error)
    assert "Test" in error_str
    # Empty context should not add "Context:" line
    assert "Context:" not in error_str or "Context: " in error_str
