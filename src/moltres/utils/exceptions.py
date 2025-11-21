"""Custom exception hierarchy."""

from typing import Optional


class MoltresError(Exception):
    """Base exception for Moltres-specific failures."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize exception with message, optional suggestion, and context.

        Args:
            message: Error message
            suggestion: Optional suggestion for fixing the error
            context: Optional dictionary with additional context
        """
        super().__init__(message)
        self.message = message
        self.suggestion = suggestion
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted error message with suggestion if available."""
        msg = self.message
        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f"\n\nContext: {context_str}"
        return msg


class CompilationError(MoltresError):
    """Raised when a logical plan cannot be converted into SQL."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize compilation error.

        Common suggestions:
        - Check if the operation is supported for your SQL dialect
        - Verify column names and table references
        - Ensure join conditions are properly specified
        """
        # Auto-generate suggestions for common errors
        if suggestion is None:
            if "not supported" in message.lower() or "unsupported" in message.lower():
                suggestion = (
                    "This operation may not be supported for your SQL dialect. "
                    "Check the documentation for dialect-specific features."
                )
            elif "join" in message.lower() and "condition" in message.lower():
                suggestion = (
                    "Joins require either an 'on' parameter with column pairs "
                    "or a 'condition' parameter with a Column expression. "
                    "Example: df.join(other, on=[('left_col', 'right_col')])"
                )
            elif "subquery" in message.lower():
                suggestion = (
                    "Subqueries require a DataFrame with a logical plan. "
                    "Make sure you're passing a DataFrame, not a list or other type."
                )
        super().__init__(message, suggestion, context)


class ExecutionError(MoltresError):
    """Raised when SQL execution fails."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize execution error.

        Common suggestions:
        - Check SQL syntax and table/column names
        - Verify database connection is active
        - Check for data type mismatches
        """
        if suggestion is None:
            if "no such table" in message.lower() or "relation" in message.lower():
                suggestion = (
                    "The table does not exist. Check the table name spelling and "
                    "ensure the table has been created. Use db.table('name') to verify."
                )
            elif "no such column" in message.lower():
                suggestion = (
                    "The column does not exist. Check the column name spelling. "
                    "Use df.select() to see available columns."
                )
            elif "syntax error" in message.lower():
                suggestion = (
                    "There's a SQL syntax error. Check your query structure. "
                    "Use df.to_sql() to see the generated SQL."
                )
        super().__init__(message, suggestion, context)


class ValidationError(MoltresError):
    """Raised when input validation fails."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize validation error.

        Common suggestions:
        - Check input types and formats
        - Verify required parameters are provided
        - Check value ranges and constraints
        """
        if suggestion is None:
            if "column name" in message.lower():
                suggestion = (
                    "Column names must be valid identifiers (letters, digits, underscores). "
                    "Avoid SQL keywords and special characters."
                )
            elif "required" in message.lower() or "missing" in message.lower():
                suggestion = "Check that all required parameters are provided."
        super().__init__(message, suggestion, context)


class SchemaError(MoltresError):
    """Raised when schema-related operations fail."""


class DatabaseConnectionError(MoltresError):
    """Raised when database connection operations fail."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize connection error."""
        if suggestion is None:
            suggestion = (
                "Check your database connection string and ensure the database server is running. "
                "Verify network connectivity and credentials."
            )
        super().__init__(message, suggestion, context)


class QueryTimeoutError(ExecutionError):
    """Raised when a query exceeds the configured timeout."""

    def __init__(
        self, message: str, timeout: Optional[float] = None, context: Optional[dict] = None
    ):
        """Initialize query timeout error."""
        suggestion = (
            "The query exceeded the timeout limit. Consider:\n"
            "  - Optimizing the query (add indexes, reduce data scanned)\n"
            "  - Increasing the timeout via query_timeout configuration\n"
            "  - Breaking the query into smaller chunks"
        )
        if timeout is not None:
            context = context or {}
            context["timeout_seconds"] = timeout
        super().__init__(message, suggestion, context)


class ConnectionPoolError(DatabaseConnectionError):
    """Raised when connection pool operations fail."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize connection pool error."""
        if suggestion is None:
            suggestion = (
                "Connection pool error. Consider:\n"
                "  - Increasing pool_size and max_overflow\n"
                "  - Checking for connection leaks (unclosed connections)\n"
                "  - Verifying database server capacity"
            )
        super().__init__(message, suggestion, context)


class TransactionError(ExecutionError):
    """Raised when transaction operations fail."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize transaction error."""
        if suggestion is None:
            suggestion = (
                "Transaction error. Ensure:\n"
                "  - All operations in the transaction are valid\n"
                "  - No conflicting locks exist\n"
                "  - Database supports the transaction isolation level"
            )
        super().__init__(message, suggestion, context)


class UnsupportedOperationError(MoltresError):
    """Raised when an unsupported operation is attempted."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize unsupported operation error."""
        if suggestion is None:
            suggestion = (
                "This operation is not supported. Check the documentation "
                "for supported operations and alternatives."
            )
        super().__init__(message, suggestion, context)
