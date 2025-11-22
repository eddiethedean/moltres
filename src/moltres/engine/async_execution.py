"""Async execution helpers for running compiled SQL."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, Union

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# AsyncConnection is used via TYPE_CHECKING, but not directly imported here
# The actual connection is managed by AsyncConnectionManager

from ..config import EngineConfig
from ..utils.exceptions import ExecutionError
from .async_connection import AsyncConnectionManager

logger = logging.getLogger(__name__)

# Optional async performance monitoring hooks
_async_perf_hooks: dict[str, list[Callable[[str, float, dict[str, Any]], None]]] = {
    "query_start": [],
    "query_end": [],
}


@dataclass
class AsyncQueryResult:
    """Result from an async query execution."""

    rows: Any
    rowcount: Optional[int]


class AsyncQueryExecutor:
    """Async abstraction over SQL execution."""

    def __init__(self, connection_manager: AsyncConnectionManager, config: EngineConfig):
        self._connections = connection_manager
        self._config = config

    async def fetch(
        self, stmt: Union[str, Any], params: Optional[Dict[str, Any]] = None
    ) -> AsyncQueryResult:
        """Execute a SELECT query and return results.

        Args:
            stmt: The SQLAlchemy Select statement or SQL string to execute
            params: Optional parameter dictionary (only used with SQL strings)

        Returns:
            AsyncQueryResult containing rows and rowcount

        Raises:
            ExecutionError: If SQL execution fails
        """
        from sqlalchemy.sql import Select

        # Convert SQLAlchemy statement to string for logging
        if isinstance(stmt, Select):
            sql_str = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        else:
            sql_str = str(stmt)[:200] if len(str(stmt)) > 200 else str(stmt)

        logger.debug("Executing async query: %s", sql_str)

        # Performance monitoring
        start_time = time.perf_counter()
        _call_async_hooks("query_start", sql_str, 0.0, {"params": params})

        try:
            async with self._connections.connect() as conn:
                # Execute SQLAlchemy statement directly or use text() for SQL strings
                if isinstance(stmt, Select):
                    result = await conn.execute(stmt)
                else:
                    result = await conn.execute(text(stmt), params or {})
                rows = result.fetchall()
                columns = list(result.keys())
                payload = await self._format_rows(rows, columns)
                rowcount = len(rows) if isinstance(rows, list) else result.rowcount or 0

                elapsed = time.perf_counter() - start_time
                logger.debug("Async query returned %d rows in %.3f seconds", rowcount, elapsed)

                _call_async_hooks(
                    "query_end",
                    sql_str,
                    elapsed,
                    {"rowcount": rowcount, "params": params},
                )

                return AsyncQueryResult(rows=payload, rowcount=result.rowcount)
        except SQLAlchemyError as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "Async SQL execution failed after %.3f seconds: %s", elapsed, exc, exc_info=True
            )
            raise ExecutionError(f"Failed to execute async query: {exc}") from exc

    async def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        transaction: Optional[Any] = None,
    ) -> AsyncQueryResult:
        """Execute a non-SELECT SQL statement (INSERT, UPDATE, DELETE, etc.).

        Args:
            sql: The SQL statement to execute
            params: Optional parameter dictionary for parameterized queries
            transaction: Optional transaction connection to use (if None, uses auto-commit)

        Returns:
            AsyncQueryResult with rowcount of affected rows

        Raises:
            ExecutionError: If SQL execution fails
        """
        logger.debug("Executing async statement: %s", sql[:200] if len(sql) > 200 else sql)
        try:
            async with self._connections.connect(transaction=transaction) as conn:
                result = await conn.execute(text(sql), params or {})
                rowcount = result.rowcount or 0
                logger.debug("Async statement affected %d rows", rowcount)
                return AsyncQueryResult(rows=None, rowcount=rowcount)
        except SQLAlchemyError as exc:
            logger.error("Async SQL execution failed: %s", exc, exc_info=True)
            raise ExecutionError(f"Failed to execute async statement: {exc}") from exc

    async def execute_many(
        self,
        sql: str,
        params_list: Sequence[Dict[str, Any]],
        transaction: Optional[Any] = None,
    ) -> AsyncQueryResult:
        """Execute a SQL statement multiple times with different parameter sets.

        This is more efficient than calling execute() in a loop for batch inserts.

        Args:
            sql: The SQL statement to execute
            params_list: Sequence of parameter dictionaries, one per execution

        Returns:
            AsyncQueryResult with total rowcount across all executions

        Raises:
            ExecutionError: If SQL execution fails
        """
        if not params_list:
            logger.debug("execute_many called with empty params_list, returning 0 affected rows.")
            return AsyncQueryResult(rows=None, rowcount=0)

        logger.debug(
            "Executing async batch statement (%d rows): %s",
            len(params_list),
            sql[:200] if len(sql) > 200 else sql,
        )
        total_rowcount = 0
        try:
            async with self._connections.connect(transaction=transaction) as conn:
                for params in params_list:
                    result = await conn.execute(text(sql), params or {})
                    total_rowcount += result.rowcount or 0
            logger.debug("Async batch statement affected %d total rows", total_rowcount)
            return AsyncQueryResult(rows=None, rowcount=total_rowcount)
        except SQLAlchemyError as exc:
            logger.error("Async batch SQL execution failed: %s", exc, exc_info=True)
            raise ExecutionError(f"Failed to execute async batch statement: {exc}") from exc

    async def fetch_stream(
        self,
        stmt: Union[str, Any],
        params: Optional[Dict[str, Any]] = None,
        chunk_size: int = 10000,
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        """Fetch query results in streaming chunks.

        Args:
            stmt: The SQLAlchemy Select statement or SQL string to execute
            params: Optional parameter dictionary (only used with SQL strings)
            chunk_size: Number of rows per chunk

        Yields:
            Lists of row dictionaries
        """
        from sqlalchemy.sql import Select

        async with self._connections.connect() as conn:
            # Execute SQLAlchemy statement directly or use text() for SQL strings
            if isinstance(stmt, Select):
                result = await conn.stream(stmt)
            else:
                result = await conn.stream(text(stmt), params or {})
            columns: Optional[List[str]] = None
            chunk: List[Dict[str, Any]] = []

            async for row in result:
                if columns is None:
                    # Get column names from result metadata
                    columns = list(result.keys())

                # Convert row to dict - row is a Row object that can be indexed
                row_dict = {col: row[idx] for idx, col in enumerate(columns)}
                chunk.append(row_dict)

                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []

            # Yield remaining rows
            if chunk:
                yield chunk

    async def _format_rows(self, rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> Any:
        """Format rows according to fetch_format configuration."""
        fmt = self._config.fetch_format
        if fmt == "records":
            return [dict(zip(columns, row)) for row in rows]
        if fmt == "pandas":
            try:
                import pandas as pd
            except ModuleNotFoundError as exc:
                raise RuntimeError("Pandas support requested but pandas is not installed") from exc
            # Convert rows to list of lists for pandas DataFrame constructor
            return pd.DataFrame(list(rows), columns=list(columns))
        if fmt == "polars":
            try:
                import polars as pl
            except ModuleNotFoundError as exc:
                raise RuntimeError("Polars support requested but polars is not installed") from exc
            return pl.DataFrame(rows, schema=columns)
        raise ValueError(
            f"Unknown fetch format '{fmt}'. Supported formats: records, pandas, polars"
        )


def register_async_performance_hook(
    event: str, callback: Callable[[str, float, dict[str, Any]], None]
) -> None:
    """Register an async performance monitoring hook.

    Args:
        event: Event type - "query_start" or "query_end"
        callback: Callback function that receives (sql, elapsed_time, metadata)

    Example:
        >>> async def log_slow_queries(sql: str, elapsed: float, metadata: dict):
        ...     if elapsed > 1.0:
        ...         print(f"Slow query ({elapsed:.2f}s): {sql[:100]}")
        >>> register_async_performance_hook("query_end", log_slow_queries)
    """
    if event not in _async_perf_hooks:
        raise ValueError(
            f"Unknown event type: {event}. Valid events: {list(_async_perf_hooks.keys())}"
        )
    _async_perf_hooks[event].append(callback)


def unregister_async_performance_hook(
    event: str, callback: Callable[[str, float, dict[str, Any]], None]
) -> None:
    """Unregister an async performance monitoring hook.

    Args:
        event: Event type - "query_start" or "query_end"
        callback: Callback function to remove
    """
    if event in _async_perf_hooks and callback in _async_perf_hooks[event]:
        _async_perf_hooks[event].remove(callback)


def _call_async_hooks(event: str, sql: str, elapsed: float, metadata: dict[str, Any]) -> None:
    """Call all registered async hooks for an event."""
    for hook in _async_perf_hooks.get(event, []):
        try:
            hook(sql, elapsed, metadata)
        except Exception as exc:
            logger.warning("Async performance hook failed: %s", exc, exc_info=True)
