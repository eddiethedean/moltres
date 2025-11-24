"""Async table access primitives."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.async_dataframe import AsyncDataFrame
    from ..dataframe.async_reader import AsyncDataLoader, AsyncReadAccessor
    from ..io.records import AsyncLazyRecords, AsyncRecords
    from .async_actions import (
        AsyncCreateTableOperation,
        AsyncDropTableOperation,
    )
try:
    from sqlalchemy.ext.asyncio.engine import AsyncConnection
except ImportError:
    AsyncConnection = None  # type: ignore[assignment, misc]

from ..engine.async_connection import AsyncConnectionManager
from ..engine.async_execution import AsyncQueryExecutor, AsyncQueryResult
from ..engine.dialects import DialectSpec, get_dialect
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
        """Return an AsyncDataLoader for loading data from files and tables as AsyncDataFrames.

        Note: For SQL operations on tables, use await db.table(name).select() instead.
        """
        from ..dataframe.async_reader import AsyncDataLoader

        return AsyncDataLoader(self)

    @property
    def read(self) -> "AsyncReadAccessor":
        """Return an AsyncReadAccessor for accessing read operations.

        Use await db.read.records.* for AsyncRecords-based reads (backward compatibility).
        Use db.load.* for AsyncDataFrame-based reads (PySpark-style).
        """
        from ..dataframe.async_reader import AsyncReadAccessor

        return AsyncReadAccessor(self)

    def sql(self, sql: str, **params: object) -> "AsyncDataFrame":
        """Execute a SQL query and return an AsyncDataFrame.

        Similar to PySpark's `spark.sql()`, this method accepts a raw SQL string
        and returns a lazy AsyncDataFrame that can be chained with further operations.
        The SQL dialect is determined by the database connection.

        Args:
            sql: SQL query string to execute
            **params: Optional named parameters for parameterized queries.
                     Use `:param_name` syntax in SQL and pass values as kwargs.

        Returns:
            Lazy AsyncDataFrame that can be chained with further operations

        Example:
            >>> # Basic SQL query
            >>> df = db.sql("SELECT * FROM users WHERE age > 18")
            >>> results = await df.collect()

            >>> # Parameterized query
            >>> df = db.sql("SELECT * FROM users WHERE id = :id AND status = :status",
            ...             id=1, status="active")
            >>> results = await df.collect()

            >>> # Chaining operations
            >>> df = db.sql("SELECT * FROM orders").where(col("amount") > 100).limit(10)
            >>> results = await df.collect()
        """
        from ..dataframe.async_dataframe import AsyncDataFrame
        from ..logical import operators

        # Convert params dict to the format expected by RawSQL
        params_dict = params if params else None
        plan = operators.raw_sql(sql, params_dict)
        return AsyncDataFrame(plan=plan, database=self)

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
            ...     await df.write.insertInto("table")
            ...     await df.write.update("table", where=..., set={...})
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

    async def createDataFrame(
        self,
        data: Union[
            Sequence[dict[str, object]], Sequence[tuple], "AsyncRecords", "AsyncLazyRecords"
        ],
        schema: Optional[Sequence[ColumnDef]] = None,
        pk: Optional[Union[str, Sequence[str]]] = None,
        auto_pk: Optional[Union[str, Sequence[str]]] = None,
    ) -> "AsyncDataFrame":
        """Create an AsyncDataFrame from Python data (list of dicts, list of tuples, AsyncRecords, or AsyncLazyRecords).

        Creates a temporary table, inserts the data, and returns an AsyncDataFrame querying from that table.
        If AsyncLazyRecords is provided, it will be auto-materialized.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - AsyncRecords object: Extracts data and schema if available
                - AsyncLazyRecords object: Auto-materializes and extracts data and schema
            schema: Optional explicit schema. If not provided, schema is inferred from data.
            pk: Optional column name(s) to mark as primary key. Can be a single string or sequence of strings for composite keys.
            auto_pk: Optional column name(s) to create as auto-incrementing primary key. Can specify same name as pk to make an existing column auto-incrementing.

        Returns:
            AsyncDataFrame querying from the created temporary table

        Raises:
            ValueError: If data is empty and no schema provided, or if primary key requirements are not met
            ValidationError: If list of tuples provided without schema, or other validation errors

        Example:
            >>> # Create AsyncDataFrame from list of dicts
            >>> df = await db.createDataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id")
            >>> # Create AsyncDataFrame with auto-incrementing primary key
            >>> df = await db.createDataFrame([{"name": "Alice"}, {"name": "Bob"}], auto_pk="id")
            >>> # Create AsyncDataFrame from AsyncLazyRecords (auto-materializes)
            >>> lazy_records = db.read.records.csv("data.csv")
            >>> df = await db.createDataFrame(lazy_records, pk="id")
        """
        from ..dataframe.async_dataframe import AsyncDataFrame
        from ..dataframe.create_dataframe import (
            ensure_primary_key,
            generate_unique_table_name,
            get_schema_from_records,
        )
        from ..dataframe.readers.schema_inference import infer_schema_from_rows
        from ..io.records import AsyncLazyRecords, AsyncRecords
        from ..utils.exceptions import ValidationError

        # Normalize data to list of dicts
        # Handle AsyncLazyRecords by auto-materializing
        if isinstance(data, AsyncLazyRecords):
            materialized_records = await data.collect()  # Auto-materialize
            rows = await materialized_records.rows()
            # Use schema from AsyncRecords if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(materialized_records)
        elif isinstance(data, AsyncRecords):
            rows = await data.rows()  # Materialize async records
            # Use schema from AsyncRecords if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(data)
        elif isinstance(data, list):
            if not data:
                rows = []
            elif isinstance(data[0], dict):
                rows = [dict(row) for row in data]
            elif isinstance(data[0], tuple):
                # Handle list of tuples - requires schema
                if schema is None:
                    raise ValidationError(
                        "List of tuples requires a schema with column names. "
                        "Provide schema parameter or use list of dicts instead."
                    )
                # Convert tuples to dicts using schema column names
                column_names = [col.name for col in schema]
                rows = []
                for row_tuple in data:
                    if len(row_tuple) != len(column_names):
                        raise ValueError(
                            f"Tuple length {len(row_tuple)} does not match schema column count {len(column_names)}"
                        )
                    rows.append(dict(zip(column_names, row_tuple)))
            else:
                raise ValueError(f"Unsupported data type in list: {type(data[0])}")
        else:
            raise ValueError(
                f"Unsupported data type: {type(data)}. "
                "Supported types: list of dicts, list of tuples (with schema), AsyncRecords"
            )

        # Validate data is not empty (unless schema provided)
        if not rows and schema is None:
            raise ValueError("Cannot create DataFrame from empty data without a schema")

        # Infer or use schema
        if schema is None:
            if not rows:
                raise ValueError("Cannot infer schema from empty data. Provide schema parameter.")
            inferred_schema_list = list(infer_schema_from_rows(rows))
        else:
            inferred_schema_list = list(schema)

        # Ensure primary key
        inferred_schema_list, new_auto_increment_cols = ensure_primary_key(
            inferred_schema_list,
            pk=pk,
            auto_pk=auto_pk,
            dialect_name=self._dialect_name,
        )

        # Generate unique table name
        table_name = generate_unique_table_name()

        # Create temporary table
        table_handle = await self.create_table(
            table_name,
            inferred_schema_list,
            temporary=True,
            if_not_exists=True,
        ).collect()

        # Insert data (exclude new auto-increment columns from INSERT)
        if rows:
            # Filter rows to only include columns that exist in schema and are not new auto-increment columns
            filtered_rows = []
            for row in rows:
                filtered_row = {
                    k: v
                    for k, v in row.items()
                    if k not in new_auto_increment_cols
                    and any(col.name == k for col in inferred_schema_list)
                }
                filtered_rows.append(filtered_row)

            records_to_insert = AsyncRecords(_data=filtered_rows, _database=self)
            await records_to_insert.insert_into(table_handle)

        # Return AsyncDataFrame querying from the temporary table
        return AsyncDataFrame.from_table(table_handle)

    async def close(self) -> None:
        """Close the database connection and cleanup resources."""
        await self._connections.close()

    # ----------------------------------------------------------------- internals
    @property
    def _dialect_name(self) -> str:
        if self.config.engine.dialect:
            return self.config.engine.dialect
        # Extract base dialect from DSN (e.g., "sqlite+aiosqlite" -> "sqlite")
        dsn = self.config.engine.dsn
        if dsn is None:
            raise ValueError("DSN is required when dialect is not explicitly set")
        scheme = dsn.split("://", 1)[0]
        # Remove async driver suffix if present (e.g., "+asyncpg", "+aiomysql", "+aiosqlite")
        if "+" in scheme:
            scheme = scheme.split("+", 1)[0]
        return scheme
