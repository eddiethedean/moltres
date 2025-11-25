"""Execution helpers for running compiled SQL."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.exc import SAWarning

from ..config import EngineConfig
from ..utils.exceptions import ExecutionError
from .connection import ConnectionManager

if TYPE_CHECKING:
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
    rows: ResultRows
    rowcount: Optional[int]


class QueryExecutor:
    """Thin abstraction over SQL execution for future extensibility."""

    def __init__(self, connection_manager: ConnectionManager, config: EngineConfig):
        self._connections = connection_manager
        self._config = config

    def fetch(
        self, stmt: Union[str, "Select"], params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Execute a SELECT query and return results.

        Args:
            stmt: The SQLAlchemy Select statement or SQL string to execute
            params: Optional parameter dictionary for parameterized queries (only used with SQL strings)

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
        self, sql: str, params: Optional[Dict[str, Any]] = None, transaction: Optional[Any] = None
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
        transaction: Optional[Any] = None,
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
        stmt: Union[str, Any],
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 10000,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Fetch query results in streaming chunks."""
        from sqlalchemy.sql import Select

        with self._connections.connect() as conn:
            # Execute SQLAlchemy statement directly or use text() for SQL strings
            if isinstance(stmt, Select):
                result = conn.execute(stmt)
            else:
                result = conn.execute(text(stmt), params or {})
            columns = list(result.keys())

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
