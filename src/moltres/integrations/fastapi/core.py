"""FastAPI integration utilities for Moltres.

This module provides helper functions and utilities to make Moltres
more user-friendly and robust when used with FastAPI.

Key features:
- Exception handlers for converting Moltres errors to HTTP responses
- Dependency injection helpers for database connections
- Type-safe helpers for common FastAPI patterns
- Error handling middleware
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import Session

    from ...table.async_table import AsyncDatabase
    from ...table.table import Database

try:
    from fastapi import HTTPException, Request, status, Depends
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create stubs for type checking - use Any to avoid type errors
    HTTPException = None  # type: ignore[assignment, misc]
    Request = None  # type: ignore[assignment, misc]
    JSONResponse = None  # type: ignore[assignment, misc]
    status = None  # type: ignore[assignment]
    Depends = None  # type: ignore[assignment]


def register_exception_handlers(app: "FastAPI") -> None:
    """Register exception handlers for Moltres errors in FastAPI app.

    This converts Moltres-specific exceptions to appropriate HTTP responses,
    making error handling more user-friendly in FastAPI applications.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> from moltres.integrations.fastapi import register_exception_handlers
        >>> app = FastAPI()
        >>> register_exception_handlers(app)
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for exception handlers. Install with: pip install fastapi"
        )

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

    @app.exception_handler(DatabaseConnectionError)
    async def database_connection_error_handler(
        request: "Request", exc: DatabaseConnectionError
    ) -> "JSONResponse":
        """Handle database connection errors."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Database connection error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )

    @app.exception_handler(ConnectionPoolError)
    async def connection_pool_error_handler(
        request: "Request", exc: ConnectionPoolError
    ) -> "JSONResponse":
        """Handle connection pool errors."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Connection pool error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )

    @app.exception_handler(QueryTimeoutError)
    async def query_timeout_error_handler(
        request: "Request", exc: QueryTimeoutError
    ) -> "JSONResponse":
        """Handle query timeout errors."""
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "Query timeout",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "timeout_seconds": exc.context.get("timeout_seconds"),
                "detail": exc.context,
            },
        )

    @app.exception_handler(ExecutionError)
    async def execution_error_handler(request: "Request", exc: ExecutionError) -> "JSONResponse":
        """Handle SQL execution errors."""
        # Check if it's a "not found" type error
        error_msg = str(exc.message).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            status_code = status.HTTP_404_NOT_FOUND
        elif "permission" in error_msg or "access" in error_msg:
            status_code = status.HTTP_403_FORBIDDEN
        elif "syntax error" in error_msg or "invalid" in error_msg:
            status_code = status.HTTP_400_BAD_REQUEST
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        return JSONResponse(
            status_code=status_code,
            content={
                "error": "SQL execution error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )

    @app.exception_handler(CompilationError)
    async def compilation_error_handler(
        request: "Request", exc: CompilationError
    ) -> "JSONResponse":
        """Handle SQL compilation errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "SQL compilation error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )

    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: "Request", exc: ValidationError) -> "JSONResponse":
        """Handle validation errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Validation error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )

    @app.exception_handler(ColumnNotFoundError)
    async def column_not_found_error_handler(
        request: "Request", exc: ColumnNotFoundError
    ) -> "JSONResponse":
        """Handle column not found errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Column not found",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "column_name": exc.context.get("column_name"),
                "available_columns": exc.context.get("available_columns"),
            },
        )

    @app.exception_handler(TransactionError)
    async def transaction_error_handler(
        request: "Request", exc: TransactionError
    ) -> "JSONResponse":
        """Handle transaction errors."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Transaction error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )

    @app.exception_handler(MoltresError)
    async def moltres_error_handler(request: "Request", exc: MoltresError) -> "JSONResponse":
        """Handle generic Moltres errors."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Moltres error",
                "message": str(exc.message),
                "suggestion": exc.suggestion,
                "detail": exc.context,
            },
        )


def get_db(session: "Session") -> "Database":
    """FastAPI dependency helper for sync database connections.

    This is a convenience function for creating Moltres :class:`Database` instances
    from FastAPI dependency-injected SQLAlchemy/SQLModel sessions.

    Args:
        session: SQLAlchemy Session or SQLModel Session from FastAPI dependency

    Returns:
        Moltres :class:`Database` instance

    Example:
        >>> from fastapi import Depends
        >>> from sqlalchemy.orm import Session
        >>> from moltres.integrations.fastapi import get_db
        >>>
        >>> @app.get("/users")
        >>> def get_users(db: :class:`Database` = Depends(lambda: get_db(Depends(get_session)))):
        ...     # Use db for Moltres operations
        ...     df = db.table("users").select()
        ...     return df.collect()
    """
    from ... import connect

    return connect(session=session)


async def get_async_db(session: "AsyncSession") -> "AsyncDatabase":
    """FastAPI dependency helper for async database connections.

    This is a convenience function for creating Moltres :class:`AsyncDatabase` instances
    from FastAPI dependency-injected async SQLAlchemy/SQLModel sessions.

    Args:
        session: SQLAlchemy AsyncSession or SQLModel AsyncSession from FastAPI dependency

    Returns:
        Moltres :class:`AsyncDatabase` instance

    Example:
        >>> from fastapi import Depends
        >>> from sqlalchemy.ext.asyncio import AsyncSession
        >>> from moltres.integrations.fastapi import get_async_db
        >>>
        >>> @app.get("/users")
        >>> async def get_users(db: :class:`AsyncDatabase` = Depends(lambda: get_async_db(Depends(get_async_session)))):
        ...     # Use db for async Moltres operations
        ...     df = (await db.table("users")).select()
        ...     return await df.collect()
    """
    from ... import async_connect

    return async_connect(session=session)


def create_db_dependency(get_session: Callable[[], "Session"]) -> Callable[[], "Database"]:
    """Create a FastAPI dependency for sync database connections.

    This creates a dependency function that can be used directly in FastAPI
    route handlers, making it easier to use Moltres with FastAPI.

    Args:
        get_session: Function that returns a SQLAlchemy Session (typically from Depends)

    Returns:
        Dependency function that returns a Moltres :class:`Database` instance

    Example:
        >>> from fastapi import Depends
        >>> from sqlalchemy.orm import Session
        >>> from moltres.integrations.fastapi import create_db_dependency
        >>>
        >>> def get_session():
        ...     # Your session creation logic
        ...     pass
        >>>
        >>> get_db = create_db_dependency(get_session)
        >>>
        >>> @app.get("/users")
        >>> def get_users(db: :class:`Database` = Depends(get_db)):
        ...     df = db.table("users").select()
        ...     return df.collect()
    """
    from ... import connect
    import inspect

    # Create a dependency that FastAPI can properly inject
    # We need to use Depends inside the function signature for FastAPI to recognize it
    if FASTAPI_AVAILABLE:
        # FastAPI available - use proper Depends signature
        def dependency(session: "Session" = Depends(get_session)) -> Database:
            # FastAPI will inject the session here
            # Check if we got a Depends object (when called directly, not by FastAPI)
            # Depends objects have a 'dependency' attribute
            if hasattr(session, "dependency"):
                # Called directly, not by FastAPI - call get_session ourselves
                session_result = get_session()
                if inspect.isgenerator(session_result):
                    session_obj = next(session_result)
                else:
                    session_obj = session_result
            else:
                # FastAPI injected the session
                if inspect.isgenerator(session):
                    session_obj = next(session)
                else:
                    session_obj = session
            return connect(session=session_obj)
    else:
        # FastAPI not available - use fallback signature
        def dependency(*args: Any, **kwargs: Any) -> Database:  # type: ignore[misc]
            # Fallback for when FastAPI is not available (testing)
            session_result = get_session()
            if inspect.isgenerator(session_result):
                session = next(session_result)
            else:
                session = session_result
            return connect(session=session)

    return dependency


def create_async_db_dependency(
    get_session: Callable[[], "AsyncSession"],
) -> Callable[[], Coroutine[Any, Any, "AsyncDatabase"]]:
    """Create a FastAPI dependency for async database connections.

    This creates an async dependency function that can be used directly in FastAPI
    route handlers, making it easier to use Moltres with FastAPI.

    Args:
        get_session: Async function that returns a SQLAlchemy AsyncSession (typically from Depends).
                     Can be a regular async function or an async generator function (with yield).

    Returns:
        Async dependency function that returns a Moltres :class:`AsyncDatabase` instance.
        This function should be used with Depends(get_db) in route handlers.

    Example:
        >>> from fastapi import Depends
        >>> from sqlalchemy.ext.asyncio import AsyncSession
        >>> from moltres.integrations.fastapi import create_async_db_dependency
        >>>
        >>> async def get_async_session():
        ...     # Your async session creation logic
        ...     pass
        >>>
        >>> get_db = create_async_db_dependency(get_async_session)
        >>>
        >>> @app.get("/users")
        >>> async def get_users(db: :class:`AsyncDatabase` = Depends(get_db)):
        ...     df = (await db.table("users")).select()
        ...     return await df.collect()
    """
    from ... import async_connect
    import inspect

    async def dependency() -> AsyncDatabase:
        # Call get_session - it might return a session directly or an async generator
        session_result = get_session()
        # Handle async generator functions (FastAPI dependency pattern with yield)
        if inspect.isasyncgen(session_result):
            session = await session_result.__anext__()
        elif inspect.iscoroutine(session_result):
            # If it's a coroutine, await it first
            session_result = await session_result
            if inspect.isasyncgen(session_result):
                session = await session_result.__anext__()
            else:
                session = session_result
        else:
            session = session_result
        return async_connect(session=session)

    return dependency


def handle_moltres_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to handle Moltres errors in FastAPI route handlers.

    This decorator catches Moltres exceptions and converts them to appropriate
    HTTPException responses, making error handling more robust.

    Args:
        func: FastAPI route handler function

    Returns:
        Wrapped function with error handling

    Example:
        >>> from fastapi import HTTPException
        >>> from moltres.integrations.fastapi import handle_moltres_errors
        >>>
        >>> @app.get("/users")
        >>> @handle_moltres_errors
        >>> def get_users(db: :class:`Database`):
        ...     df = db.table("users").select()
        ...     return df.collect()
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is required for error handling decorator. Install with: pip install fastapi"
        )

    import asyncio
    from functools import wraps

    from ...utils.exceptions import (
        ColumnNotFoundError,
        CompilationError,
        DatabaseConnectionError,
        ExecutionError,
        MoltresError,
        QueryTimeoutError,
        ValidationError,
    )

    def _handle_error(e: Exception) -> None:
        """Raise appropriate HTTPException for Moltres errors."""
        if isinstance(e, QueryTimeoutError):
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={
                    "error": "Query timeout",
                    "message": str(e.message),
                    "suggestion": e.suggestion,
                },
            ) from e
        elif isinstance(e, DatabaseConnectionError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "Database connection error",
                    "message": str(e.message),
                    "suggestion": e.suggestion,
                },
            ) from e
        elif isinstance(e, (CompilationError, ValidationError, ColumnNotFoundError)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": type(e).__name__,
                    "message": str(e.message),
                    "suggestion": e.suggestion,
                },
            ) from e
        elif isinstance(e, ExecutionError):
            # Check if it's a "not found" error
            error_msg = str(e.message).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                status_code = status.HTTP_404_NOT_FOUND
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            raise HTTPException(
                status_code=status_code,
                detail={
                    "error": "SQL execution error",
                    "message": str(e.message),
                    "suggestion": e.suggestion,
                },
            ) from e
        elif isinstance(e, MoltresError):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Moltres error",
                    "message": str(e.message),
                    "suggestion": getattr(e, "suggestion", "An unexpected error occurred"),
                },
            ) from e

    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTPException (FastAPI exceptions should pass through)
                raise
            except Exception as e:
                _handle_error(e)

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTPException (FastAPI exceptions should pass through)
                raise
            except Exception as e:
                _handle_error(e)

        return sync_wrapper
