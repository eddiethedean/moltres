"""Custom exception hierarchy."""


class MoltresError(Exception):
    """Base exception for Moltres-specific failures."""


class CompilationError(MoltresError):
    """Raised when a logical plan cannot be converted into SQL."""


class ExecutionError(MoltresError):
    """Raised when SQL execution fails."""


class ValidationError(MoltresError):
    """Raised when input validation fails."""


class SchemaError(MoltresError):
    """Raised when schema-related operations fail."""


class DatabaseConnectionError(MoltresError):
    """Raised when database connection operations fail."""


class UnsupportedOperationError(MoltresError):
    """Raised when an unsupported operation is attempted."""
