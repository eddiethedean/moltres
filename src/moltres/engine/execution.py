"""Execution helpers for running compiled SQL."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
)

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import SAWarning

from ..config import EngineConfig
from ..utils.exceptions import ExecutionError
from .connection import ConnectionManager

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.sql import Select
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)

# Optional performance monitoring hooks
_perf_hooks: dict[str, list[Callable[[str, float, dict[str, Any]], None]]] = {
    "query_start": [],
    "query_end": [],
}

# Type alias for result rows - can be records, pandas DataFrame, or polars DataFrame
if TYPE_CHECKING:
    ResultRows = Union[
        List[dict[str, object]],
        "pd.DataFrame",
        "pl.DataFrame",
    ]
else:
    ResultRows = Any


@dataclass
class QueryResult:
    rows: Optional[ResultRows]
    rowcount: Optional[int]


class QueryExecutor:
    """Thin abstraction over SQL execution for future extensibility."""

    def __init__(self, connection_manager: ConnectionManager, config: EngineConfig):
        self._connections = connection_manager
        self._config = config

    def fetch(
        self,
        stmt: Union[str, "Select"],
        params: Optional[Dict[str, Any]] = None,
        connection: Optional["Connection"] = None,
        model: Optional[Type[Any]] = None,
    ) -> QueryResult:
        """Execute a SELECT query and return results.

        Args:
            stmt: The SQLAlchemy Select statement or SQL string to execute
            params: Optional parameter dictionary for parameterized queries (only used with SQL strings)
            connection: Optional SQLAlchemy Connection to use. If provided, uses this connection
                       directly instead of creating a new one. Useful for executing within
                       existing transactions. The connection's lifecycle is not managed by this method.

        Returns:
            QueryResult containing rows and rowcount

        Raises:
            ExecutionError: If SQL execution fails
        """
        from sqlalchemy.sql import Select

        # Convert SQLAlchemy statement to string for logging
        if isinstance(stmt, Select):
            sql_str = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        else:
            sql_str = str(stmt)[:200] if len(str(stmt)) > 200 else str(stmt)

        logger.debug("Executing query: %s", sql_str)

        # Performance monitoring
        start_time = time.perf_counter()
        _call_hooks("query_start", sql_str, 0.0, {"params": params})

        try:
            # Check if we're using a SQLModel session and have a model
            use_sqlmodel_exec = False
            sqlmodel_session = None
            if model is not None:
                # Check if we have a SQLModel session available
                if (
                    hasattr(self._connections, "_session")
                    and self._connections._session is not None
                ):
                    session = self._connections._session
                    # Check if it's a SQLModel session (has .exec() method)
                    if hasattr(session, "exec"):
                        use_sqlmodel_exec = True
                        sqlmodel_session = session

            # Use provided connection or create a new one
            if connection is not None:
                # Use the provided connection directly (don't manage its lifecycle)
                conn = connection
                # Apply query timeout if configured
                execution_options = {}
                if self._config.query_timeout is not None:
                    execution_options["timeout"] = self._config.query_timeout

                # Use SQLModel .exec() if available and model is provided
                if use_sqlmodel_exec and isinstance(stmt, Select) and model is not None:
                    # SQLModel .exec() returns SQLModel instances directly
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", category=SAWarning, message=".*cartesian product.*"
                            )
                            # SQLModel's exec() works with Select statements
                            exec_result = sqlmodel_session.exec(stmt)
                            # .exec() returns an iterable of SQLModel instances
                            sqlmodel_instances = list(exec_result)
                            return QueryResult(
                                rows=sqlmodel_instances, rowcount=len(sqlmodel_instances)
                            )
                    except (AttributeError, TypeError, ValueError) as e:
                        # SQLModel .exec() may not be available or may fail due to:
                        # - AttributeError: .exec() method doesn't exist on this session type
                        # - TypeError: Incompatible statement or model type
                        # - ValueError: Invalid arguments to .exec()
                        # Fall back to regular execute if exec() fails
                        logger.debug(
                            "SQLModel .exec() failed, falling back to regular execute: %s", e
                        )
                    except Exception as e:
                        # Catch any other unexpected exceptions from SQLModel .exec()
                        # This broad catch is acceptable here because we have a fallback path
                        # and want to ensure query execution continues even if SQLModel integration fails
                        logger.debug(
                            "Unexpected error in SQLModel .exec(), falling back to regular execute: %s",
                            e,
                        )

                # Execute SQLAlchemy statement directly or use text() for SQL strings
                if isinstance(stmt, Select):
                    # Suppress cartesian product warnings for cross joins (intentional)
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", category=SAWarning, message=".*cartesian product.*"
                        )
                        if execution_options:
                            result = conn.execution_options(**execution_options).execute(stmt)
                        else:
                            result = conn.execute(stmt)
                else:
                    if execution_options:
                        result = conn.execution_options(**execution_options).execute(
                            text(stmt), params or {}
                        )
                    else:
                        result = conn.execute(text(stmt), params or {})
            else:
                # Create a new connection (manage its lifecycle)
                # Use SQLModel .exec() if available and model is provided
                if use_sqlmodel_exec and isinstance(stmt, Select) and model is not None:
                    # SQLModel .exec() returns SQLModel instances directly
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", category=SAWarning, message=".*cartesian product.*"
                            )
                            # SQLModel's exec() works with Select statements
                            exec_result = sqlmodel_session.exec(stmt)
                            # .exec() returns an iterable of SQLModel instances
                            sqlmodel_instances = list(exec_result)
                            return QueryResult(
                                rows=sqlmodel_instances, rowcount=len(sqlmodel_instances)
                            )
                    except (AttributeError, TypeError, ValueError) as e:
                        # SQLModel .exec() may not be available or may fail due to:
                        # - AttributeError: .exec() method doesn't exist on this session type
                        # - TypeError: Incompatible statement or model type
                        # - ValueError: Invalid arguments to .exec()
                        # Fall back to regular execute if exec() fails
                        logger.debug(
                            "SQLModel .exec() failed, falling back to regular execute: %s", e
                        )
                    except Exception as e:
                        # Catch any other unexpected exceptions from SQLModel .exec()
                        # This broad catch is acceptable here because we have a fallback path
                        # and want to ensure query execution continues even if SQLModel integration fails
                        logger.debug(
                            "Unexpected error in SQLModel .exec(), falling back to regular execute: %s",
                            e,
                        )

                with self._connections.connect() as conn:
                    # Apply query timeout if configured
                    execution_options = {}
                    if self._config.query_timeout is not None:
                        execution_options["timeout"] = self._config.query_timeout

                    # Execute SQLAlchemy statement directly or use text() for SQL strings
                    if isinstance(stmt, Select):
                        # Suppress cartesian product warnings for cross joins (intentional)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", category=SAWarning, message=".*cartesian product.*"
                            )
                            if execution_options:
                                result = conn.execution_options(**execution_options).execute(stmt)
                            else:
                                result = conn.execute(stmt)
                    else:
                        if execution_options:
                            result = conn.execution_options(**execution_options).execute(
                                text(stmt), params or {}
                            )
                        else:
                            result = conn.execute(text(stmt), params or {})

                    # Fetch rows while still inside the connection context
                    # (important for databases like DuckDB where result sets are tied to connections)
                    rows = result.fetchall()
                    columns = list(result.keys())
                    payload = self._format_rows(rows, columns)
                    rowcount = len(rows) if isinstance(rows, list) else result.rowcount or 0

            # For provided connections, fetch after execution
            if connection is not None:
                rows = result.fetchall()
                columns = list(result.keys())
                payload = self._format_rows(rows, columns)
                rowcount = len(rows) if isinstance(rows, list) else result.rowcount or 0

            elapsed = time.perf_counter() - start_time
            logger.debug("Query returned %d rows in %.3f seconds", rowcount, elapsed)

            _call_hooks(
                "query_end",
                sql_str,
                elapsed,
                {"rowcount": rowcount, "params": params},
            )

            return QueryResult(rows=payload, rowcount=rowcount)
        except SQLAlchemyError as exc:
            elapsed = time.perf_counter() - start_time
            logger.error("SQL execution failed after %.3f seconds: %s", elapsed, exc, exc_info=True)
            _call_hooks("query_end", sql_str, elapsed, {"error": str(exc), "params": params})
            # Check if it's a timeout error
            error_str = str(exc).lower()
            # Include SQL query text in the error message for easier debugging
            sql_preview = sql_str[:500] + "..." if len(sql_str) > 500 else sql_str
            if "timeout" in error_str or "timed out" in error_str:
                from ..utils.exceptions import QueryTimeoutError

                raise QueryTimeoutError(
                    f"Query exceeded timeout: {exc}\nSQL query: {sql_preview}",
                    timeout=self._config.query_timeout,
                ) from exc
            raise ExecutionError(
                f"SQL execution failed: {exc}\nSQL query: {sql_preview}",
                context={"sql": sql_str, "params": params, "elapsed_seconds": elapsed},
            ) from exc

    def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        transaction: Optional["Connection"] = None,
    ) -> QueryResult:
        """Execute a non-SELECT SQL statement (INSERT, UPDATE, DELETE, etc.).

        Args:
            sql: The SQL statement to execute
            params: Optional parameter dictionary for parameterized queries
            transaction: Optional transaction connection to use (if None, uses auto-commit)

        Returns:
            QueryResult with rowcount of affected rows

        Raises:
            ExecutionError: If SQL execution fails
        """
        logger.debug("Executing statement: %s", sql[:200] if len(sql) > 200 else sql)
        try:
            with self._connections.connect(transaction=transaction) as conn:
                result = conn.execute(text(sql), params or {})
                rowcount = result.rowcount or 0
                logger.debug("Statement affected %d rows", rowcount)
                return QueryResult(rows=None, rowcount=rowcount)
        except SQLAlchemyError as exc:
            logger.error("SQL execution failed: %s", exc, exc_info=True)
            raise ExecutionError(f"Failed to execute statement: {exc}") from exc

    def execute_many(
        self,
        sql: str,
        params_list: Sequence[Dict[str, Any]],
        transaction: Optional["Connection"] = None,
    ) -> QueryResult:
        """Execute a SQL statement multiple times with different parameter sets.

        This is more efficient than calling execute() in a loop for batch inserts.

        Args:
            sql: The SQL statement to execute
            params_list: Sequence of parameter dictionaries, one per execution
            transaction: Optional transaction connection to use (if None, uses auto-commit)

        Returns:
            QueryResult with total rowcount across all executions

        Raises:
            ExecutionError: If SQL execution fails
        """
        if not params_list:
            return QueryResult(rows=None, rowcount=0)

        logger.debug(
            "Executing batch statement (%d rows): %s",
            len(params_list),
            sql[:200] if len(sql) > 200 else sql,
        )
        try:
            with self._connections.connect(transaction=transaction) as conn:
                result = conn.execute(text(sql), params_list)
                total_rowcount = result.rowcount or 0
            logger.debug("Batch statement affected %d total rows", total_rowcount)
            return QueryResult(rows=None, rowcount=total_rowcount)
        except SQLAlchemyError as exc:
            logger.error("Batch SQL execution failed: %s", exc, exc_info=True)
            raise ExecutionError(f"Failed to execute batch statement: {exc}") from exc

    def fetch_stream(
        self,
        stmt: Union[str, "Select"],
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 10000,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Fetch query results in streaming chunks."""
        from sqlalchemy.sql import Select
        from sqlalchemy.exc import ProgrammingError

        conn_context = self._connections.connect()
        conn = None
        try:
            conn = conn_context.__enter__()
            # Execute SQLAlchemy statement directly or use text() for SQL strings
            if isinstance(stmt, Select):
                result = conn.execute(stmt)
            else:
                result = conn.execute(text(stmt), params or {})
            columns = list(result.keys())

            try:
                while True:
                    rows = result.fetchmany(chunk_size)
                    if not rows:
                        break
                    # Format rows according to fetch_format
                    if self._config.fetch_format == "records":
                        chunk = [dict(zip(columns, row)) for row in rows]
                        yield chunk
                    else:
                        # For pandas/polars, we'd need to yield DataFrames
                        # For now, convert to records format
                        chunk = [dict(zip(columns, row)) for row in rows]
                        yield chunk
            except GeneratorExit:
                # Generator was closed early (e.g., when db.close() is called)
                # Re-raise to allow proper cleanup, but we'll suppress cleanup errors
                raise
        finally:
            # Always attempt cleanup, but suppress errors if connection is already closed
            if conn is not None:
                try:
                    # Check if connection is still valid before attempting cleanup
                    # If the database was closed, the connection might be invalid
                    conn_context.__exit__(None, None, None)
                except (ProgrammingError, GeneratorExit) as e:
                    # Suppress ProgrammingError about closed database during cleanup
                    # This can happen when db.close() is called while generator is active
                    # Also suppress GeneratorExit to prevent unraisable exception warnings
                    if isinstance(e, ProgrammingError) and "closed" in str(e).lower():
                        # Connection was already closed - this is expected when db.close() is called
                        pass
                    elif isinstance(e, GeneratorExit):
                        # GeneratorExit during cleanup - suppress to prevent unraisable exception
                        pass
                    else:
                        raise
                except Exception:
                    # For other exceptions during cleanup, log but don't raise
                    # to prevent masking the original exception
                    logger.debug("Error during fetch_stream cleanup: %s", exc_info=True)

    def _format_rows(self, rows: Sequence[Sequence[object]], columns: Sequence[str]) -> ResultRows:
        fmt = self._config.fetch_format
        if fmt == "records":
            return [dict(zip(columns, row)) for row in rows]
        if fmt == "pandas":
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Pandas support requested but pandas is not installed") from exc
            # Convert rows to list of lists for pandas DataFrame constructor
            return pd.DataFrame(list(rows), columns=list(columns))
        if fmt == "polars":
            try:
                import polars as pl
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Polars support requested but polars is not installed") from exc
            return pl.DataFrame(rows, schema=columns)
        raise ValueError(
            f"Unknown fetch format '{fmt}'. Supported formats: records, pandas, polars"
        )


def register_performance_hook(
    event: str, callback: Callable[[str, float, dict[str, Any]], None]
) -> None:
    """Register a performance monitoring hook.

    Args:
        event: Event type - "query_start" or "query_end"
        callback: Callback function that receives (sql, elapsed_time, metadata)

    Example:
        >>> def log_slow_queries(sql: str, elapsed: float, metadata: dict):
        ...     if elapsed > 1.0:
        ...         print(f"Slow query ({elapsed:.2f}s): {sql[:100]}")
        >>> register_performance_hook("query_end", log_slow_queries)
    """
    if event not in _perf_hooks:
        raise ValueError(f"Unknown event type: {event}. Valid events: {list(_perf_hooks.keys())}")
    _perf_hooks[event].append(callback)


def unregister_performance_hook(
    event: str, callback: Callable[[str, float, dict[str, Any]], None]
) -> None:
    """Unregister a performance monitoring hook.

    Args:
        event: Event type - "query_start" or "query_end"
        callback: Callback function to remove
    """
    if event in _perf_hooks and callback in _perf_hooks[event]:
        _perf_hooks[event].remove(callback)


def _call_hooks(event: str, sql: str, elapsed: float, metadata: dict[str, Any]) -> None:
    """Call all registered hooks for an event."""
    for hook in _perf_hooks.get(event, []):
        try:
            hook(sql, elapsed, metadata)
        except Exception as exc:
            logger.warning("Performance hook failed: %s", exc, exc_info=True)
