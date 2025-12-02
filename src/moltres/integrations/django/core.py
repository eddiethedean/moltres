"""Django integration utilities for Moltres.

This module provides helper functions and utilities to make Moltres
more user-friendly and robust when used with Django.

Key features:
- Middleware for converting Moltres errors to Django HTTP responses
- :class:`Database` connection helpers with Django database routing support
- Integration with Django's transaction management
- Logging integration with Django's logging system
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, cast

if TYPE_CHECKING:
    from django.http import HttpRequest, HttpResponse, JsonResponse  # type: ignore[import-untyped]
    from django.utils.deprecation import MiddlewareMixin  # type: ignore[import-untyped]
    from ...table.table import Database
else:
    Database = Any
    HttpRequest = Any
    HttpResponse = Any
    JsonResponse = Any

try:
    from django.conf import settings  # type: ignore[import-untyped]
    from django.core.exceptions import ImproperlyConfigured  # type: ignore[import-untyped]
    from django.db import connections, transaction  # type: ignore[import-untyped]
    from django.http import HttpRequest, HttpResponse, JsonResponse
    from django.utils.deprecation import MiddlewareMixin

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    # Create stubs for type checking
    MiddlewareMixin = cast(type[Any], None)
    HttpRequest = cast(type[Any], None)
    HttpResponse = cast(type[Any], None)
    JsonResponse = cast(type[Any], None)
    connections = None
    transaction = None
    settings = None
    ImproperlyConfigured = None


logger = logging.getLogger(__name__)


class MoltresExceptionMiddleware:
    """Django middleware that catches Moltres exceptions and converts them to HTTP responses.

    This middleware should be added to Django's MIDDLEWARE setting to automatically
    handle Moltres exceptions throughout your application.

    Example:
        Add to settings.py:
        MIDDLEWARE = [
            # ... other middleware
            'moltres.integrations.django.MoltresExceptionMiddleware',
        ]
    """

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        """Initialize the middleware.

        Args:
            get_response: The next middleware or view in the chain
        """
        if not DJANGO_AVAILABLE:
            raise ImportError(
                "Django is required for MoltresExceptionMiddleware. Install with: pip install django"
            )
        self.get_response = get_response
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Set up logging for Moltres exceptions."""
        # Use Django's logging configuration
        if not hasattr(logger, "handlers") or not logger.handlers:
            # Add a handler if none exists
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    def __call__(self, request: HttpRequest) -> HttpResponse:
        """Process the request and handle Moltres exceptions.

        Args:
            request: Django HttpRequest object

        Returns:
            HttpResponse with appropriate status code and error details
        """
        try:
            response = self.get_response(request)
            return response
        except Exception as exc:
            return self._handle_exception(request, exc)

    def _handle_exception(self, request: HttpRequest, exc: Exception) -> HttpResponse:
        """Handle Moltres exceptions and convert to HTTP responses.

        Args:
            request: Django HttpRequest object
            exc: Exception that was raised

        Returns:
            JsonResponse with appropriate status code and error details
        """
        from ...utils.exceptions import (
            ColumnNotFoundError,
            CompilationError,
            ConnectionPoolError,
            DatabaseConnectionError,
            ExecutionError,
            MoltresError,
            QueryTimeoutError,
            TransactionError,
            ValidationError,
        )

        # Log the exception
        logger.error(
            "Moltres exception in Django view",
            exc_info=True,
            extra={"request_path": request.path, "exception_type": type(exc).__name__},
        )

        # Handle specific Moltres exceptions
        # Check more specific exceptions first (ConnectionPoolError is a subclass of DatabaseConnectionError)
        if isinstance(exc, ConnectionPoolError):
            return JsonResponse(
                {
                    "error": "Connection pool error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=503,  # Service Unavailable
            )

        if isinstance(exc, DatabaseConnectionError):
            return JsonResponse(
                {
                    "error": "Database connection error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=503,  # Service Unavailable
            )

        if isinstance(exc, QueryTimeoutError):
            return JsonResponse(
                {
                    "error": "Query timeout",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "timeout_seconds": exc.context.get("timeout_seconds") if exc.context else None,
                    "detail": exc.context,
                },
                status=504,  # Gateway Timeout
            )

        # Check more specific ExecutionError subclasses first
        if isinstance(exc, QueryTimeoutError):
            return JsonResponse(
                {
                    "error": "Query timeout",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "timeout_seconds": exc.context.get("timeout_seconds") if exc.context else None,
                    "detail": exc.context,
                },
                status=504,  # Gateway Timeout
            )

        if isinstance(exc, TransactionError):
            return JsonResponse(
                {
                    "error": "Transaction error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=500,  # Internal Server Error
            )

        if isinstance(exc, ExecutionError):
            # Check if it's a "not found" type error
            error_msg = str(exc.message).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                status_code = 404  # Not Found
            elif "permission" in error_msg or "access" in error_msg:
                status_code = 403  # Forbidden
            elif "syntax error" in error_msg or "invalid" in error_msg:
                status_code = 400  # Bad Request
            else:
                status_code = 500  # Internal Server Error

            return JsonResponse(
                {
                    "error": "SQL execution error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=status_code,
            )

        if isinstance(exc, CompilationError):
            return JsonResponse(
                {
                    "error": "SQL compilation error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=400,  # Bad Request
            )

        # Check more specific ValidationError subclasses first
        if isinstance(exc, ColumnNotFoundError):
            return JsonResponse(
                {
                    "error": "Column not found",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "column_name": exc.context.get("column_name") if exc.context else None,
                    "available_columns": exc.context.get("available_columns")
                    if exc.context
                    else None,
                },
                status=400,  # Bad Request
            )

        if isinstance(exc, ValidationError):
            return JsonResponse(
                {
                    "error": "Validation error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=400,  # Bad Request
            )

        if isinstance(exc, MoltresError):
            return JsonResponse(
                {
                    "error": "Moltres error",
                    "message": str(exc.message),
                    "suggestion": exc.suggestion,
                    "detail": exc.context,
                },
                status=500,  # Internal Server Error
            )

        # Not a Moltres exception, re-raise it
        raise exc


def get_moltres_db(using: str = "default") -> Database:
    """Get a Moltres :class:`Database` instance from Django's database connection.

    This function creates a Moltres :class:`Database` instance using Django's database
    connection, supporting Django's database routing and transaction management.

    Args:
        using: Django database alias (default: 'default'). Supports database routing.

    Returns:
        Moltres :class:`Database` instance

    Raises:
        ImportError: If Django is not installed
        ImproperlyConfigured: If the database alias is not configured

    Example:
        >>> from moltres.integrations.django import get_moltres_db
        >>>
        >>> def my_view(request):
        ...     db = get_moltres_db(using='default')
        ...     df = db.table("users").select()
        ...     return JsonResponse({'users': df.collect()})
    """
    if not DJANGO_AVAILABLE:
        raise ImportError("Django is required for get_moltres_db. Install with: pip install django")

    # Validate database alias
    if using not in settings.DATABASES:
        raise ImproperlyConfigured(
            f"Database alias '{using}' is not configured in Django settings.DATABASES"
        )

    # Get connection parameters from Django settings
    db_config = settings.DATABASES[using]
    engine_name = db_config.get("ENGINE", "").lower()

    # Build SQLAlchemy connection string from Django settings
    dsn: Optional[str] = None

    if "sqlite" in engine_name:
        # SQLite
        name = db_config.get("NAME", ":memory:")
        # Handle both absolute and relative paths
        if name == ":memory:":
            dsn = "sqlite:///:memory:"
        else:
            dsn = f"sqlite:///{name}"
    elif "postgresql" in engine_name or "postgis" in engine_name:
        # PostgreSQL
        user = db_config.get("USER", "")
        password = db_config.get("PASSWORD", "")
        host = db_config.get("HOST", "localhost")
        port = db_config.get("PORT", "5432")
        name = db_config.get("NAME", "")
        # URL encode password if it contains special characters
        from urllib.parse import quote_plus

        password_encoded = quote_plus(password) if password else ""
        dsn = f"postgresql://{user}:{password_encoded}@{host}:{port}/{name}"
    elif "mysql" in engine_name:
        # MySQL
        user = db_config.get("USER", "")
        password = db_config.get("PASSWORD", "")
        host = db_config.get("HOST", "localhost")
        port = db_config.get("PORT", "3306")
        name = db_config.get("NAME", "")
        # URL encode password if it contains special characters
        from urllib.parse import quote_plus

        password_encoded = quote_plus(password) if password else ""
        dsn = f"mysql://{user}:{password_encoded}@{host}:{port}/{name}"
    else:
        raise ImproperlyConfigured(
            f"Unsupported database engine '{engine_name}' for database '{using}'. "
            "Moltres Django integration currently supports: sqlite3, postgresql, mysql. "
            "For other databases, use connect() directly with a DSN string."
        )

    # Create Moltres Database instance
    from ... import connect

    db = connect(dsn=dsn)

    # Store reference to Django connection for transaction management
    # This allows us to integrate with Django's transaction.atomic()
    django_connection = connections[using]
    db._django_connection = django_connection  # type: ignore[attr-defined]
    db._django_using = using  # type: ignore[attr-defined]

    return db
