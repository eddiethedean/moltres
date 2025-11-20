"""Custom exception hierarchy."""


class MoltresError(Exception):
    """Base exception for Moltres-specific failures."""


class CompilationError(MoltresError):
    """Raised when a logical plan cannot be converted into SQL."""
