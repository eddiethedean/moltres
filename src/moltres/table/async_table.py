"""Async table access primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.async_dataframe import AsyncDataFrame
    from ..dataframe.async_reader import AsyncDataLoader
    from ..io.records import AsyncRecords
from ..engine.async_connection import AsyncConnectionManager
from ..engine.async_execution import AsyncQueryExecutor, AsyncQueryResult
from ..engine.dialects import DialectSpec, get_dialect
from ..expressions.column import Column
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan
from ..sql.ddl import compile_create_table, compile_drop_table
from .schema import ColumnDef, TableSchema


@dataclass
class AsyncTableHandle:
    """Lightweight handle representing a table reference for async operations."""

    name: str
    database: "AsyncDatabase"

    def select(self, *columns: str) -> "AsyncDataFrame":
        from ..dataframe.async_dataframe import AsyncDataFrame

        return AsyncDataFrame.from_table(self, columns=list(columns))

    async def insert(self, rows: Union[Sequence[Mapping[str, object]], "AsyncRecords"]) -> int:
        from .async_mutations import insert_rows_async

        return await insert_rows_async(self, rows)

    async def update(self, *, where: Column, set: Mapping[str, object]) -> int:
        from .async_mutations import update_rows_async

        return await update_rows_async(self, where=where, values=set)

    async def delete(self, where: Column) -> int:
        from .async_mutations import delete_rows_async

        return await delete_rows_async(self, where=where)


class AsyncDatabase:
    """Entry-point object returned by ``moltres.async_connect``."""

    def __init__(self, config: MoltresConfig):
        self.config = config
        self._connections = AsyncConnectionManager(config.engine)
        self._executor = AsyncQueryExecutor(self._connections, config.engine)
        self._dialect = get_dialect(self._dialect_name)

    @property
    def connection_manager(self) -> AsyncConnectionManager:
        return self._connections

    @property
    def executor(self) -> AsyncQueryExecutor:
        return self._executor

    async def table(self, name: str) -> AsyncTableHandle:
        """Get a handle to a table in the database.

        Args:
            name: Name of the table

        Returns:
            AsyncTableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
        """
        from ..utils.exceptions import ValidationError
        from ..sql.builders import quote_identifier

        if not name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(name, self._dialect.quote_char)
        return AsyncTableHandle(name=name, database=self)

    @property
    def load(self) -> "AsyncDataLoader":
        """Return an AsyncDataLoader for loading data from files and tables as AsyncRecords.

        Note: For SQL operations on tables, use await db.table(name).select() instead.
        """
        from ..dataframe.async_reader import AsyncDataLoader

        return AsyncDataLoader(self)

    # -------------------------------------------------------------- DDL operations
    async def create_table(
        self,
        name: str,
        columns: Sequence[ColumnDef],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
    ) -> AsyncTableHandle:
        """Create a new table with the specified schema.

        Args:
            name: Name of the table to create
            columns: Sequence of ColumnDef objects defining the table schema
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)

        Returns:
            AsyncTableHandle for the newly created table

        Raises:
            ValidationError: If table name or columns are invalid
            ExecutionError: If table creation fails
        """
        from ..utils.exceptions import ValidationError

        if not columns:
            raise ValidationError(f"Cannot create table '{name}' with no columns")

        schema = TableSchema(
            name=name,
            columns=columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
        )
        sql = compile_create_table(schema, self._dialect)
        await self._executor.execute(sql)
        return await self.table(name)

    async def drop_table(self, name: str, *, if_exists: bool = True) -> None:
        """Drop a table by name.

        Args:
            name: Name of the table to drop
            if_exists: If True, don't error if table doesn't exist (default: True)

        Raises:
            ValidationError: If table name is invalid
            ExecutionError: If table dropping fails (when if_exists=False and table doesn't exist)
        """
        sql = compile_drop_table(name, self._dialect, if_exists=if_exists)
        await self._executor.execute(sql)

    # -------------------------------------------------------------- query utils
    def compile_plan(self, plan: LogicalPlan) -> Any:
        """Compile a logical plan to SQL."""
        return compile_plan(plan, dialect=self._dialect)

    async def execute_plan(self, plan: LogicalPlan) -> AsyncQueryResult:
        """Execute a logical plan and return results."""
        sql = self.compile_plan(plan)
        return await self._executor.fetch(sql)

    async def execute_plan_stream(
        self, plan: LogicalPlan
    ) -> AsyncIterator[List[Dict[str, object]]]:
        """Execute a plan and return an async iterator of row chunks."""
        sql = self.compile_plan(plan)
        async for chunk in self._executor.fetch_stream(sql):
            yield chunk

    async def execute_sql(
        self, sql: str, params: Optional[Dict[str, Any]] = None
    ) -> AsyncQueryResult:
        """Execute raw SQL and return results."""
        return await self._executor.fetch(sql, params=params)

    @property
    def dialect(self) -> DialectSpec:
        return self._dialect

    async def close(self) -> None:
        """Close the database connection and cleanup resources."""
        await self._connections.close()

    # ----------------------------------------------------------------- internals
    @property
    def _dialect_name(self) -> str:
        if self.config.engine.dialect:
            return self.config.engine.dialect
        # Extract base dialect from DSN (e.g., "sqlite+aiosqlite" -> "sqlite")
        scheme = self.config.engine.dsn.split("://", 1)[0]
        # Remove async driver suffix if present (e.g., "+asyncpg", "+aiomysql", "+aiosqlite")
        if "+" in scheme:
            scheme = scheme.split("+", 1)[0]
        return scheme
