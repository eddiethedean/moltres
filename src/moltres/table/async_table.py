"""Async table access primitives."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.async_dataframe import AsyncDataFrame
    from ..dataframe.async_reader import AsyncDataLoader
    from ..io.records import AsyncRecords
    from .async_actions import (
        AsyncCreateTableOperation,
        AsyncDeleteMutation,
        AsyncDropTableOperation,
        AsyncInsertMutation,
        AsyncMergeMutation,
        AsyncUpdateMutation,
    )
try:
    from sqlalchemy.ext.asyncio.engine import AsyncConnection
except ImportError:
    AsyncConnection = None  # type: ignore[assignment, misc]

from ..engine.async_connection import AsyncConnectionManager
from ..engine.async_execution import AsyncQueryExecutor, AsyncQueryResult
from ..engine.dialects import DialectSpec, get_dialect
from ..expressions.column import Column
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan
from .schema import ColumnDef


@dataclass
class AsyncTableHandle:
    """Lightweight handle representing a table reference for async operations."""

    name: str
    database: "AsyncDatabase"

    def select(self, *columns: str) -> "AsyncDataFrame":
        from ..dataframe.async_dataframe import AsyncDataFrame

        return AsyncDataFrame.from_table(self, columns=list(columns))

    def insert(
        self, rows: Union[Sequence[Mapping[str, object]], "AsyncRecords"]
    ) -> "AsyncInsertMutation":
        """Create a lazy async insert operation.

        Args:
            rows: Sequence of row dictionaries to insert

        Returns:
            AsyncInsertMutation that executes on collect()

        Example:
            >>> mutation = table.insert([{"id": 1, "name": "Alice"}])
            >>> rowcount = await mutation.collect()  # Executes the insert
        """
        from .async_actions import AsyncInsertMutation

        return AsyncInsertMutation(handle=self, rows=rows)

    def update(self, *, where: Column, set: Mapping[str, object]) -> "AsyncUpdateMutation":
        """Create a lazy async update operation.

        Args:
            where: Column expression for the WHERE clause
            set: Dictionary of column names to new values

        Returns:
            AsyncUpdateMutation that executes on collect()

        Example:
            >>> mutation = table.update(where=col("id") == 1, set={"name": "Bob"})
            >>> rowcount = await mutation.collect()  # Executes the update
        """
        from .async_actions import AsyncUpdateMutation

        return AsyncUpdateMutation(handle=self, where=where, values=set)

    def delete(self, where: Column) -> "AsyncDeleteMutation":
        """Create a lazy async delete operation.

        Args:
            where: Column expression for the WHERE clause

        Returns:
            AsyncDeleteMutation that executes on collect()

        Example:
            >>> mutation = table.delete(where=col("id") == 1)
            >>> rowcount = await mutation.collect()  # Executes the delete
        """
        from .async_actions import AsyncDeleteMutation

        return AsyncDeleteMutation(handle=self, where=where)

    def merge(
        self,
        rows: Union[Sequence[Mapping[str, object]], "AsyncRecords"],
        *,
        on: Sequence[str],
        when_matched: Optional[Mapping[str, object]] = None,
        when_not_matched: Optional[Mapping[str, object]] = None,
    ) -> "AsyncMergeMutation":
        """Create a lazy async merge (upsert) operation.

        This implements MERGE/UPSERT operations with dialect-specific SQL:
        - PostgreSQL: INSERT ... ON CONFLICT ... DO UPDATE
        - SQLite: INSERT ... ON CONFLICT ... DO UPDATE
        - MySQL: INSERT ... ON DUPLICATE KEY UPDATE

        Args:
            rows: Sequence of row dictionaries to merge
            on: Sequence of column names that form the conflict key (primary key or unique constraint)
            when_matched: Optional dictionary of column updates when a conflict occurs
                         If None, no update is performed (insert only if not exists)
            when_not_matched: Optional dictionary of default values when inserting new rows
                             If None, uses values from rows

        Returns:
            AsyncMergeMutation that executes on collect()

        Example:
            >>> mutation = table.merge(
            ...     [{"email": "user@example.com", "name": "Updated Name"}],
            ...     on=["email"],
            ...     when_matched={"name": "Updated Name", "updated_at": "NOW()"}
            ... )
            >>> rowcount = await mutation.collect()  # Executes the merge
        """
        from .async_actions import AsyncMergeMutation

        return AsyncMergeMutation(
            handle=self,
            rows=rows,
            on=on,
            when_matched=when_matched,
            when_not_matched=when_not_matched,
        )


class AsyncTransaction:
    """Async transaction context for grouping multiple operations."""

    def __init__(self, database: "AsyncDatabase", connection: AsyncConnection):
        self.database = database
        self.connection = connection
        self._committed = False
        self._rolled_back = False

    async def commit(self) -> None:
        """Explicitly commit the transaction."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        await self.database.connection_manager.commit_transaction(self.connection)
        self._committed = True

    async def rollback(self) -> None:
        """Explicitly rollback the transaction."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        await self.database.connection_manager.rollback_transaction(self.connection)
        self._rolled_back = True

    async def __aenter__(self) -> "AsyncTransaction":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            # Exception occurred, rollback
            if not self._rolled_back and not self._committed:
                await self.rollback()
        else:
            # No exception, commit
            if not self._committed and not self._rolled_back:
                await self.commit()


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
    def create_table(
        self,
        name: str,
        columns: Sequence[ColumnDef],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
    ) -> "AsyncCreateTableOperation":
        """Create a lazy async create table operation.

        Args:
            name: Name of the table to create
            columns: Sequence of ColumnDef objects defining the table schema
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)

        Returns:
            AsyncCreateTableOperation that executes on collect()

        Raises:
            ValidationError: If table name or columns are invalid

        Example:
            >>> op = db.create_table("users", [column("id", "INTEGER")])
            >>> table = await op.collect()  # Executes the CREATE TABLE
        """
        from ..utils.exceptions import ValidationError
        from .async_actions import AsyncCreateTableOperation

        # Validate early (at operation creation time)
        if not columns:
            raise ValidationError(f"Cannot create table '{name}' with no columns")

        return AsyncCreateTableOperation(
            database=self,
            name=name,
            columns=columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
        )

    def drop_table(self, name: str, *, if_exists: bool = True) -> "AsyncDropTableOperation":
        """Create a lazy async drop table operation.

        Args:
            name: Name of the table to drop
            if_exists: If True, don't error if table doesn't exist (default: True)

        Returns:
            AsyncDropTableOperation that executes on collect()

        Example:
            >>> op = db.drop_table("users")
            >>> await op.collect()  # Executes the DROP TABLE
        """
        from .async_actions import AsyncDropTableOperation

        return AsyncDropTableOperation(database=self, name=name, if_exists=if_exists)

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

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncTransaction]:
        """Create an async transaction context for grouping multiple operations.

        All operations within the transaction context share the same transaction.
        If any exception occurs, the transaction is automatically rolled back.
        Otherwise, it is committed on successful exit.

        Yields:
            AsyncTransaction object that can be used for explicit commit/rollback

        Example:
            >>> async with db.transaction() as txn:
            ...     await table.insert([...]).collect()
            ...     await table.update(...).collect()
            ...     # If any operation fails, all are rolled back
            ...     # Otherwise, all are committed on exit
        """
        connection = await self._connections.begin_transaction()
        txn = AsyncTransaction(self, connection)
        try:
            yield txn
            if not txn._committed and not txn._rolled_back:
                await txn.commit()
        except Exception:
            if not txn._rolled_back:
                await txn.rollback()
            raise

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
