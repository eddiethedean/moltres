"""Async table access primitives."""

from __future__ import annotations

import asyncio
import atexit
from contextlib import asynccontextmanager
import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from ..dataframe.async_dataframe import AsyncDataFrame
    from ..dataframe.async_reader import AsyncDataLoader, AsyncReadAccessor
    from ..io.records import AsyncLazyRecords, AsyncRecords
    from ..utils.inspector import ColumnInfo
    from .async_actions import (
        AsyncCreateIndexOperation,
        AsyncCreateTableOperation,
        AsyncDropIndexOperation,
        AsyncDropTableOperation,
    )
    from .schema import (
        CheckConstraint,
        ForeignKeyConstraint,
        TableSchema,
        UniqueConstraint,
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

logger = logging.getLogger(__name__)
_ACTIVE_ASYNC_DATABASES: "weakref.WeakSet[AsyncDatabase]" = weakref.WeakSet()


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
        self._ephemeral_tables: set[str] = set()
        self._closed = False
        _ACTIVE_ASYNC_DATABASES.add(self)

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
        constraints: Optional[
            Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]]
        ] = None,
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
            constraints=constraints or (),
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

    def create_index(
        self,
        name: str,
        table: str,
        columns: Union[str, Sequence[str]],
        *,
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> "AsyncCreateIndexOperation":
        """Create a lazy async create index operation.

        Args:
            name: Name of the index to create
            table: Name of the table to create the index on
            columns: Column name(s) to index (single string or sequence)
            unique: If True, create a UNIQUE index (default: False)
            if_not_exists: If True, don't error if index already exists (default: True)

        Returns:
            AsyncCreateIndexOperation that executes on collect()

        Example:
            >>> op = db.create_index("idx_email", "users", "email")
            >>> await op.collect()  # Executes the CREATE INDEX
            >>> # Multi-column index
            >>> op2 = db.create_index("idx_name_age", "users", ["name", "age"], unique=True)
        """
        from .async_actions import AsyncCreateIndexOperation

        return AsyncCreateIndexOperation(
            database=self,
            name=name,
            table_name=table,
            columns=columns,
            unique=unique,
            if_not_exists=if_not_exists,
        )

    def drop_index(
        self,
        name: str,
        table: Optional[str] = None,
        *,
        if_exists: bool = True,
    ) -> "AsyncDropIndexOperation":
        """Create a lazy async drop index operation.

        Args:
            name: Name of the index to drop
            table: Optional table name (required for some dialects like MySQL)
            if_exists: If True, don't error if index doesn't exist (default: True)

        Returns:
            AsyncDropIndexOperation that executes on collect()

        Example:
            >>> op = db.drop_index("idx_email", "users")
            >>> await op.collect()  # Executes the DROP INDEX
        """
        from .async_actions import AsyncDropIndexOperation

        return AsyncDropIndexOperation(
            database=self,
            name=name,
            table_name=table,
            if_exists=if_exists,
        )

    # -------------------------------------------------------------- schema inspection
    async def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of table names in the database.

        Args:
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.

        Returns:
            List of table names

        Raises:
            ValueError: If database connection is not available
            RuntimeError: If inspection fails

        Example:
            >>> tables = await db.get_table_names()
            >>> # Returns: ['users', 'orders', 'products']
        """
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy.ext.asyncio import AsyncEngine

        if self.connection_manager is None:
            raise ValueError("Database connection manager is not available")

        engine = self.connection_manager.engine

        if not isinstance(engine, AsyncEngine):
            raise TypeError("Expected AsyncEngine for async database")

        try:
            async with engine.begin() as conn:

                def _inspect_sync(sync_conn: Any) -> List[str]:
                    inspector = sa_inspect(sync_conn)
                    return inspector.get_table_names(schema=schema)  # type: ignore[no-any-return]

                return await conn.run_sync(_inspect_sync)
        except Exception as e:
            raise RuntimeError(f"Failed to get table names: {e}") from e

    async def get_view_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of view names in the database.

        Args:
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.

        Returns:
            List of view names

        Raises:
            ValueError: If database connection is not available
            RuntimeError: If inspection fails

        Example:
            >>> views = await db.get_view_names()
            >>> # Returns: ['active_users_view', 'order_summary_view']
        """
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy.ext.asyncio import AsyncEngine

        if self.connection_manager is None:
            raise ValueError("Database connection manager is not available")

        engine = self.connection_manager.engine

        if not isinstance(engine, AsyncEngine):
            raise TypeError("Expected AsyncEngine for async database")

        try:
            async with engine.begin() as conn:

                def _inspect_sync(sync_conn: Any) -> List[str]:
                    inspector = sa_inspect(sync_conn)
                    return inspector.get_view_names(schema=schema)  # type: ignore[no-any-return]

                return await conn.run_sync(_inspect_sync)
        except Exception as e:
            raise RuntimeError(f"Failed to get view names: {e}") from e

    async def get_columns(self, table_name: str) -> List["ColumnInfo"]:
        """Get column information for a table.

        Args:
            table_name: Name of the table to inspect

        Returns:
            List of ColumnInfo objects with column metadata

        Raises:
            ValidationError: If table name is invalid
            ValueError: If database connection is not available
            RuntimeError: If table does not exist or cannot be inspected

        Example:
            >>> columns = await db.get_columns("users")
            >>> # Returns: [ColumnInfo(name='id', type_name='INTEGER', ...), ...]
        """
        from ..utils.exceptions import ValidationError
        from ..utils.inspector import ColumnInfo
        from ..sql.builders import quote_identifier
        from sqlalchemy import inspect as sa_inspect
        from sqlalchemy.ext.asyncio import AsyncEngine

        if not table_name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(table_name, self._dialect.quote_char)

        if self.connection_manager is None:
            raise ValueError("Database connection manager is not available")

        engine = self.connection_manager.engine
        if not isinstance(engine, AsyncEngine):
            raise TypeError("Expected AsyncEngine for async database")

        try:
            async with engine.begin() as conn:

                def _inspect_sync(sync_conn: Any) -> List[Dict[str, Any]]:
                    inspector = sa_inspect(sync_conn)
                    return inspector.get_columns(table_name)  # type: ignore[no-any-return]

                columns = await conn.run_sync(_inspect_sync)
        except Exception as e:
            from sqlalchemy.exc import NoSuchTableError

            # Check if the exception or its cause is NoSuchTableError
            is_no_such_table = isinstance(e, NoSuchTableError)
            if not is_no_such_table and e.__cause__:
                is_no_such_table = isinstance(e.__cause__, NoSuchTableError)

            if is_no_such_table:
                raise RuntimeError(f"Failed to inspect table '{table_name}': {e}") from e
            raise

        # Convert to ColumnInfo objects
        result: List[ColumnInfo] = []
        for col_info in columns:
            # Convert SQLAlchemy type to string representation
            type_name = str(col_info["type"])
            # Clean up type string
            if "(" in type_name:
                type_name = type_name.split("(")[0] + "(" + type_name.split("(")[1]
            else:
                type_name = type_name.split(".")[-1].replace("()", "")

            # Extract precision and scale from numeric types
            precision = None
            scale = None
            sa_type = col_info.get("type")
            if (
                sa_type is not None
                and hasattr(sa_type, "precision")
                and sa_type.precision is not None
            ):
                precision = sa_type.precision
            if sa_type is not None and hasattr(sa_type, "scale") and sa_type.scale is not None:
                scale = sa_type.scale

            # Convert primary_key to boolean (SQLAlchemy returns 1/0)
            primary_key = col_info.get("primary_key", False)
            if isinstance(primary_key, int):
                primary_key = bool(primary_key)

            result.append(
                ColumnInfo(
                    name=col_info["name"],
                    type_name=type_name,
                    nullable=col_info.get("nullable", True),
                    default=col_info.get("default"),
                    primary_key=primary_key,
                    precision=precision,
                    scale=scale,
                )
            )

        return result

    async def reflect_table(self, name: str, schema: Optional[str] = None) -> "TableSchema":
        """Reflect a single table from the database.

        Args:
            name: Name of the table to reflect
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.

        Returns:
            TableSchema object with table metadata

        Raises:
            ValidationError: If table name is invalid
            ValueError: If database connection is not available
            RuntimeError: If table does not exist or reflection fails

        Example:
            >>> schema = await db.reflect_table("users")
            >>> # Returns: TableSchema(name='users', columns=[ColumnDef(...), ...])
        """
        from ..utils.exceptions import ValidationError
        from ..sql.builders import quote_identifier
        from .schema import TableSchema

        if not name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(name, self._dialect.quote_char)

        # Use get_columns which handles async properly
        columns = await self.get_columns(name)
        column_defs = [col_info.to_column_def() for col_info in columns]

        return TableSchema(name=name, columns=column_defs)

    async def reflect(
        self, schema: Optional[str] = None, views: bool = False
    ) -> Dict[str, "TableSchema"]:
        """Reflect entire database schema.

        Args:
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.
            views: If True, also reflect views (default: False)

        Returns:
            Dictionary mapping table/view names to TableSchema objects

        Raises:
            ValueError: If database connection is not available
            RuntimeError: If reflection fails

        Example:
            >>> schemas = await db.reflect()
            >>> # Returns: {'users': TableSchema(...), 'orders': TableSchema(...)}
        """

        # Get all table names
        table_names = await self.get_table_names(schema=schema)

        # Optionally get view names
        view_names: List[str] = []
        if views:
            view_names = await self.get_view_names(schema=schema)

        # Reflect each table
        result: Dict[str, TableSchema] = {}
        for table_name in table_names:
            try:
                schema_obj = await self.reflect_table(table_name, schema=schema)
                result[table_name] = schema_obj
            except Exception as e:
                # Log but continue with other tables
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to reflect table '{table_name}': {e}")

        # Reflect views if requested
        for view_name in view_names:
            try:
                view_schema = await self.reflect_table(view_name, schema=schema)
                result[view_name] = view_schema
            except Exception as e:
                # Log but continue with other views
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to reflect view '{view_name}': {e}")

        return result

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
            Sequence[dict[str, object]],
            Sequence[tuple],
            "AsyncRecords",
            "AsyncLazyRecords",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
        schema: Optional[Sequence[ColumnDef]] = None,
        pk: Optional[Union[str, Sequence[str]]] = None,
        auto_pk: Optional[Union[str, Sequence[str]]] = None,
    ) -> "AsyncDataFrame":
        """Create an AsyncDataFrame from Python data (list of dicts, list of tuples, AsyncRecords, AsyncLazyRecords, pandas DataFrame, polars DataFrame, or polars LazyFrame).

        Creates a temporary table, inserts the data, and returns an AsyncDataFrame querying from that table.
        If AsyncLazyRecords is provided, it will be auto-materialized.
        If pandas/polars DataFrame or LazyFrame is provided, it will be converted to Records with lazy conversion.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - AsyncRecords object: Extracts data and schema if available
                - AsyncLazyRecords object: Auto-materializes and extracts data and schema
                - pandas DataFrame: Converts to Records with schema preservation
                - polars DataFrame: Converts to Records with schema preservation
                - polars LazyFrame: Materializes and converts to Records with schema preservation
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
        from ..io.records import (
            AsyncLazyRecords,
            AsyncRecords,
            _is_pandas_dataframe,
            _is_polars_dataframe,
            _is_polars_lazyframe,
            _dataframe_to_records,
        )
        from ..utils.exceptions import ValidationError

        # Convert DataFrame to Records if needed, then extract rows synchronously
        if _is_pandas_dataframe(data) or _is_polars_dataframe(data) or _is_polars_lazyframe(data):
            records = _dataframe_to_records(data)
            rows = records.rows()
            # Use schema from Records if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(records)
        # Normalize data to list of dicts
        # Handle AsyncLazyRecords by auto-materializing
        elif isinstance(data, AsyncLazyRecords):
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
            require_primary_key=False,
        )

        # Generate unique table name
        table_name = generate_unique_table_name()

        # Async workloads frequently hop between pooled connections, so always stage data in
        # regular tables (cleaned up later) instead of relying on connection-scoped temp tables.
        use_temp_tables = False
        table_handle = await self.create_table(
            table_name,
            inferred_schema_list,
            temporary=use_temp_tables,
            if_not_exists=True,
        ).collect()
        if not use_temp_tables:
            self._register_ephemeral_table(table_name)

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
        await self._close_resources()

    async def _close_resources(self) -> None:
        if self._closed:
            return
        await self._cleanup_ephemeral_tables()
        await self._connections.close()
        self._closed = True
        _ACTIVE_ASYNC_DATABASES.discard(self)

    def _register_ephemeral_table(self, name: str) -> None:
        self._ephemeral_tables.add(name)

    def _unregister_ephemeral_table(self, name: str) -> None:
        self._ephemeral_tables.discard(name)

    async def _cleanup_ephemeral_tables(self) -> None:
        if not self._ephemeral_tables:
            return
        for table_name in list(self._ephemeral_tables):
            try:
                await self.drop_table(table_name, if_exists=True).collect()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to drop async ephemeral table %s: %s", table_name, exc)
        self._ephemeral_tables.clear()

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


def _cleanup_all_async_databases() -> None:
    """Best-effort cleanup for AsyncDatabase instances left open at exit.

    Note: This runs in atexit context, so we can't reliably use asyncio.
    Instead, we mark databases as needing cleanup and log a warning.
    In practice, applications should explicitly close databases.
    """
    if not _ACTIVE_ASYNC_DATABASES:
        return

    databases = list(_ACTIVE_ASYNC_DATABASES)
    if databases:
        # Log warning about unclosed databases
        logger.warning(
            "%d AsyncDatabase instance(s) were not explicitly closed. "
            "Ephemeral tables may not be cleaned up. "
            "Always call await db.close() when done with AsyncDatabase instances.",
            len(databases),
        )
        # Mark as closed to prevent further use
        for db in databases:
            db._closed = True
            # Try to clean up ephemeral tables synchronously if possible
            # (this is best-effort and may not work in all scenarios)
            if db._ephemeral_tables:
                logger.debug(
                    "%d ephemeral table(s) may not be cleaned up for AsyncDatabase: %s",
                    len(db._ephemeral_tables),
                    db._ephemeral_tables,
                )


async def _cleanup_all_async_databases_async() -> None:
    """Async version of cleanup that actually drops tables.

    This can be used in tests or when we have an event loop available.
    """
    if not _ACTIVE_ASYNC_DATABASES:
        return

    databases = list(_ACTIVE_ASYNC_DATABASES)
    for db in databases:
        try:
            await db._close_resources()
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("AsyncDatabase cleanup failed: %s", exc)


def _force_async_database_cleanup_for_tests() -> None:
    """Helper used by tests to simulate crash/GC cleanup for async DBs.

    This creates an event loop and actually cleans up async databases.
    """

    if not _ACTIVE_ASYNC_DATABASES:
        return

    async def _cleanup() -> None:
        databases = list(_ACTIVE_ASYNC_DATABASES)
        for db in databases:
            try:
                await db._close_resources()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("AsyncDatabase cleanup during test failed: %s", exc)

    # Try to get existing event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're in a running loop, we can't use run_until_complete
        # Instead, we need to use a different approach
        # For tests, we'll create a new loop in a thread
        import threading
        import queue

        result_queue: queue.Queue[Exception | None] = queue.Queue()

        def run_in_thread() -> None:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(_cleanup())
                result_queue.put(None)
            except Exception as e:
                result_queue.put(e)
            finally:
                new_loop.close()

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=5.0)
        if thread.is_alive():
            logger.warning("Async cleanup thread timed out")
        else:
            result = result_queue.get_nowait()
            if result:
                raise result
    except RuntimeError:
        # No running loop, try to get or create one
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(_cleanup())
        except RuntimeError:
            # No event loop at all, create one
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(_cleanup())
            finally:
                loop.close()


atexit.register(_cleanup_all_async_databases)
