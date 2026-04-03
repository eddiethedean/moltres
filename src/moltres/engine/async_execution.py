"""Async query execution (moltres facade over :mod:`moltres_core.sql`).

Maps :mod:`moltres_core.exceptions` to :mod:`moltres.utils.exceptions` for parity
with synchronous :mod:`moltres.engine.execution`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type, Union

from moltres_core.exceptions import ExecutionError as _CoreExecutionError
from moltres_core.sql.async_execution import (
    AsyncQueryExecutor as _CoreAsyncQueryExecutor,
    AsyncQueryResult,
    register_async_performance_hook,
    unregister_async_performance_hook,
)
from moltres.utils.exceptions import ExecutionError


class AsyncQueryExecutor(_CoreAsyncQueryExecutor):
    """Like :class:`moltres_core.sql.async_execution.AsyncQueryExecutor` with public Moltres exceptions."""

    async def fetch(
        self,
        stmt: Union[str, Any],
        params: Optional[Dict[str, Any]] = None,
        connection: Any = None,
        model: Optional[Type[Any]] = None,
    ) -> AsyncQueryResult:
        try:
            return await super().fetch(stmt, params=params, connection=connection, model=model)
        except _CoreExecutionError as exc:
            raise ExecutionError(
                exc.message, suggestion=exc.suggestion, context=dict(exc.context)
            ) from exc

    async def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        transaction: Any = None,
    ) -> AsyncQueryResult:
        try:
            return await super().execute(sql, params=params, transaction=transaction)
        except _CoreExecutionError as exc:
            raise ExecutionError(
                exc.message, suggestion=exc.suggestion, context=dict(exc.context)
            ) from exc

    async def execute_many(
        self,
        sql: str,
        params_list: Sequence[Dict[str, Any]],
        transaction: Any = None,
    ) -> AsyncQueryResult:
        try:
            return await super().execute_many(sql, params_list, transaction=transaction)
        except _CoreExecutionError as exc:
            raise ExecutionError(
                exc.message, suggestion=exc.suggestion, context=dict(exc.context)
            ) from exc


__all__ = [
    "AsyncQueryExecutor",
    "AsyncQueryResult",
    "register_async_performance_hook",
    "unregister_async_performance_hook",
]
