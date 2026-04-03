"""Custom exception hierarchy."""

from typing import Optional, Sequence


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _suggest_column_name(column_name: str, available_columns: Sequence[str]) -> str:
    """Suggest similar column names when a column is not found."""
    if not available_columns:
        return f"Column '{column_name}' does not exist. No columns are available in this context."
    # Calculate similarity scores
    scores = [
        (col, _levenshtein_distance(column_name.lower(), col.lower())) for col in available_columns
    ]
    scores.sort(key=lambda x: x[1])
    # Get top 3 suggestions
    suggestions = [col for col, _ in scores[:3] if scores[0][1] <= len(column_name)]
    if suggestions:
        if len(suggestions) == 1:
            return f"Column '{column_name}' does not exist. Did you mean: '{suggestions[0]}'?"
        else:
            suggestions_str = ", ".join(f"'{s}'" for s in suggestions)
            return f"Column '{column_name}' does not exist. Did you mean one of: {suggestions_str}?"
    return (
        f"Column '{column_name}' does not exist. "
        f"Available columns: {', '.join(available_columns[:10])}"
        + ("..." if len(available_columns) > 10 else "")
    )


def _suggest_table_name(table_name: str, available_tables: Sequence[str]) -> str:
    """Suggest similar table names when a table is not found."""
    if not available_tables:
        return f"Table '{table_name}' does not exist. No tables are available in this database."
    # Calculate similarity scores
    scores = [
        (tbl, _levenshtein_distance(table_name.lower(), tbl.lower())) for tbl in available_tables
    ]
    scores.sort(key=lambda x: x[1])
    # Get top 3 suggestions
    suggestions = [tbl for tbl, _ in scores[:3] if scores[0][1] <= len(table_name)]
    if suggestions:
        if len(suggestions) == 1:
            return f"Table '{table_name}' does not exist. Did you mean: '{suggestions[0]}'?"
        else:
            suggestions_str = ", ".join(f"'{s}'" for s in suggestions)
            return f"Table '{table_name}' does not exist. Did you mean one of: {suggestions_str}?"
    return (
        f"Table '{table_name}' does not exist. "
        f"Available tables: {', '.join(available_tables[:10])}"
        + ("..." if len(available_tables) > 10 else "")
    )


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

        # Add query context if available
        if self.context:
            query_context = self.context.get("query")
            if query_context:
                msg += f"\n\nQuery: {query_context}"

            operation_context = self.context.get("operation")
            if operation_context:
                msg += f"\n\nOperation: {operation_context}"

            # Add other context (excluding query and operation which are handled above)
            other_context = {
                k: v for k, v in self.context.items() if k not in ("query", "operation")
            }
            if other_context:
                context_str = ", ".join(f"{k}={v}" for k, v in other_context.items())
                msg += f"\n\nContext: {context_str}"

        if self.suggestion:
            msg += f"\n\nSuggestion: {self.suggestion}"

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
                    "Example: df.join(other, on=[col('left_col') == col('right_col')])"
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
                # Try to extract table name and suggest similar ones
                context = context or {}
                if "table_name" in context and "available_tables" in context:
                    suggestion = _suggest_table_name(
                        context["table_name"], context["available_tables"]
                    )
            elif "no such column" in message.lower():
                suggestion = (
                    "The column does not exist. Check the column name spelling. "
                    "Use df.select() to see available columns."
                )
                # Try to extract column name and suggest similar ones
                context = context or {}
                if "column_name" in context and "available_columns" in context:
                    suggestion = _suggest_column_name(
                        context["column_name"], context["available_columns"]
                    )
            elif "syntax error" in message.lower():
                suggestion = (
                    "There's a SQL syntax error. Check your query structure. "
                    "Use df.to_sql() or df.show_sql() to see the generated SQL."
                )
                # Add SQL to context if available
                context = context or {}
                if "sql" in context:
                    sql_preview = str(context["sql"])[:200]
                    if len(str(context["sql"])) > 200:
                        sql_preview += "..."
                    context["sql_preview"] = sql_preview
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


class PandasAPIError(MoltresError):
    """Raised when pandas-style API operations fail."""

    def __init__(
        self, message: str, suggestion: Optional[str] = None, context: Optional[dict] = None
    ):
        """Initialize pandas API error."""
        if suggestion is None:
            suggestion = (
                "This pandas-style operation encountered an error. "
                "Check that column names are correct and operations are valid."
            )
        super().__init__(message, suggestion, context)


class ColumnNotFoundError(ValidationError):
    """Raised when a column is not found in a :class:`DataFrame`."""

    def __init__(
        self,
        column_name: str,
        available_columns: Sequence[str],
        suggestion: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        """Initialize column not found error."""
        message = f"Column '{column_name}' not found"
        if suggestion is None:
            suggestion = _suggest_column_name(column_name, available_columns)
        context = context or {}
        context["column_name"] = column_name
        context["available_columns"] = available_columns
        super().__init__(message, suggestion, context)
