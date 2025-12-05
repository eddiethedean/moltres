"""Async table access primitives."""

from __future__ import annotations

import asyncio
import atexit
from contextlib import asynccontextmanager
import logging
import time
from types import TracebackType
import weakref
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
    cast,
    overload,
    Sequence,
    Type,
)

from ..config import MoltresConfig

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
    from sqlalchemy.orm import DeclarativeBase
    from ..dataframe.core.async_dataframe import AsyncDataFrame
    from ..dataframe.interfaces.async_pandas_dataframe import AsyncPandasDataFrame
    from ..dataframe.interfaces.async_polars_dataframe import AsyncPolarsDataFrame
    from ..dataframe.io.async_reader import AsyncDataLoader, AsyncReadAccessor
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
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio.engine import AsyncConnection
else:
    # Import at runtime - if not available, use Any as fallback type
    try:
        from sqlalchemy.ext.asyncio.engine import AsyncConnection
    except ImportError:
        from typing import Any as AsyncConnection  # type: ignore[assignment, misc]

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
    model: Optional[Type[Any]] = None  # Can be SQLModel, Pydantic, or SQLAlchemy model

    @property
    def model_class(self) -> Optional[Type["DeclarativeBase"]]:
        """Get the SQLAlchemy model class if this handle was created from a model.

        Returns:
            SQLAlchemy model class or None if handle was created from table name
        """
        return self.model

    def select(self, *columns: str) -> "AsyncDataFrame":
        from ..dataframe.core.async_dataframe import AsyncDataFrame

        return AsyncDataFrame.from_table(self, columns=list(columns))

    def polars(self) -> "AsyncPolarsDataFrame":
        """Create an AsyncPolarsDataFrame from this table (Polars-style entry point).

        Returns:
            AsyncPolarsDataFrame querying from this table

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> table = await db.table("users")
            >>> df = table.polars()
            >>> results = await df.collect()
        """
        from ..dataframe.core.async_dataframe import AsyncDataFrame
        from ..dataframe.interfaces.async_polars_dataframe import AsyncPolarsDataFrame

        df = AsyncDataFrame.from_table(self)
        return AsyncPolarsDataFrame.from_dataframe(df)

    def pandas(self) -> "AsyncPandasDataFrame":
        """Create an AsyncPandasDataFrame from this table (Pandas-style entry point).

        Returns:
            AsyncPandasDataFrame querying from this table

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> table = await db.table("users")
            >>> df = table.pandas()
            >>> results = await df.collect()
        """
        from ..dataframe.core.async_dataframe import AsyncDataFrame
        from ..dataframe.interfaces.async_pandas_dataframe import AsyncPandasDataFrame

        df = AsyncDataFrame.from_table(self)
        return AsyncPandasDataFrame.from_dataframe(df)


class AsyncTransaction:
    """Async transaction context for grouping multiple operations."""

    def __init__(
        self,
        database: "AsyncDatabase",
        connection: AsyncConnection,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        is_savepoint: bool = False,
    ):
        """Initialize an async transaction context.

        Args:
            database: The database instance this transaction belongs to
            connection: The SQLAlchemy async connection for this transaction
            readonly: Whether this transaction is read-only
            isolation_level: Transaction isolation level
            is_savepoint: Whether this transaction is actually a savepoint
        """
        self.database = database
        self.connection = connection
        self._committed = False
        self._rolled_back = False
        self._readonly = readonly
        self._isolation_level = isolation_level
        self._is_savepoint = is_savepoint

    async def commit(self) -> None:
        """Explicitly commit the transaction."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        await self.database.connection_manager.commit_transaction(self.connection)
        self._committed = True

        # Execute commit hooks
        from ..utils.transaction_hooks import _execute_hooks_async, _on_commit_hooks_async

        await _execute_hooks_async(_on_commit_hooks_async, self)

    async def rollback(self) -> None:
        """Explicitly rollback the transaction."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        await self.database.connection_manager.rollback_transaction(self.connection)
        self._rolled_back = True

        # Execute rollback hooks
        from ..utils.transaction_hooks import _execute_hooks_async, _on_rollback_hooks_async

        await _execute_hooks_async(_on_rollback_hooks_async, self)

    async def savepoint(self, name: Optional[str] = None) -> str:
        """Create a savepoint within this transaction.

        Args:
            name: Optional savepoint name. If not provided, a unique name is generated.

        Returns:
            The savepoint name (generated or provided)

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back
            ValueError: If savepoints are not supported by the dialect
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        if name is None:
            name = self.database.connection_manager._generate_savepoint_name()
        await self.database.connection_manager.create_savepoint(self.connection, name)
        return name

    async def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a specific savepoint.

        Args:
            name: Savepoint name to rollback to

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back,
                         or if the savepoint doesn't exist
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        await self.database.connection_manager.rollback_to_savepoint(self.connection, name)

    async def release_savepoint(self, name: str) -> None:
        """Release a savepoint.

        Args:
            name: Savepoint name to release

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back,
                         or if the savepoint doesn't exist
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        await self.database.connection_manager.release_savepoint(self.connection, name)

    def is_readonly(self) -> bool:
        """Check if this transaction is read-only.

        Returns:
            True if the transaction is read-only, False otherwise
        """
        return self._readonly

    def isolation_level(self) -> Optional[str]:
        """Get the transaction isolation level.

        Returns:
            Isolation level string or None if not set
        """
        return self._isolation_level

    def is_active(self) -> bool:
        """Check if this transaction is still active.

        Returns:
            True if the transaction is active (not committed or rolled back), False otherwise
        """
        return not self._committed and not self._rolled_back

    async def __aenter__(self) -> "AsyncTransaction":
        # Execute begin hooks
        from ..utils.transaction_hooks import _execute_hooks_async, _on_begin_hooks_async

        await _execute_hooks_async(_on_begin_hooks_async, self)

        # Start metrics tracking (async version needs manual time tracking)
        self._metrics_start_time = time.time()
        self._metrics_has_savepoint = self._is_savepoint
        self._metrics_readonly = self._readonly
        self._metrics_isolation_level = self._isolation_level
        self._metrics_committed = False
        self._metrics_error: Optional[Exception] = None

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if exc_type is not None:
            # Exception occurred, rollback
            if not self._rolled_back and not self._committed:
                await self.rollback()
            # Record metrics after rollback
            if hasattr(self, "_metrics_start_time"):
                from ..utils.transaction_metrics import get_transaction_metrics

                self._metrics_error = exc_val if isinstance(exc_val, Exception) else None
                duration = time.time() - self._metrics_start_time
                metrics = get_transaction_metrics()
                metrics.record_transaction(
                    duration=duration,
                    committed=False,
                    has_savepoint=self._metrics_has_savepoint,
                    readonly=self._metrics_readonly,
                    isolation_level=self._metrics_isolation_level,
                    error=self._metrics_error,
                )
        else:
            # No exception, commit
            if not self._committed and not self._rolled_back:
                await self.commit()
            # Record metrics after commit
            if hasattr(self, "_metrics_start_time"):
                from ..utils.transaction_metrics import get_transaction_metrics

                self._metrics_committed = True
                duration = time.time() - self._metrics_start_time
                metrics = get_transaction_metrics()
                metrics.record_transaction(
                    duration=duration,
                    committed=True,
                    has_savepoint=self._metrics_has_savepoint,
                    readonly=self._metrics_readonly,
                    isolation_level=self._metrics_isolation_level,
                    error=None,
                )


class AsyncDatabase:
    """Entry-point object returned by ``moltres.async_connect``.

    The :class:`AsyncDatabase` class supports async context manager protocol for
    automatic connection cleanup. Use it in an ``async with`` statement to ensure
    the connection is properly closed.

    Example:
        Using async context manager (recommended)::

            >>> async with async_connect("sqlite+aiosqlite:///:memory:") as db:
            ...     df = db.sql("SELECT * FROM users")
            ...     results = await df.collect()
            ...     # await db.close() called automatically on exit
    """

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

    @classmethod
    def from_async_engine(cls, engine: "AsyncEngine", **options: object) -> "AsyncDatabase":
        """Create an :class:`AsyncDatabase` instance from an existing SQLAlchemy AsyncEngine.

        This allows you to use Moltres with an existing SQLAlchemy AsyncEngine,
        enabling integration with existing async SQLAlchemy projects.

        Args:
            engine: SQLAlchemy AsyncEngine instance
            **options: Optional configuration parameters:
                - echo: Enable SQLAlchemy echo mode
                - fetch_format: Result format - "records", "pandas", or "polars"
                - dialect: Override SQL dialect detection
                - query_timeout: Query execution timeout in seconds
                - Other options are stored in config.options

        Returns:
            :class:`AsyncDatabase` instance configured to use the provided AsyncEngine

        Example:
            >>> from sqlalchemy.ext.asyncio import create_async_engine
            >>> from moltres import :class:`AsyncDatabase`
            >>> engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            >>> db = :class:`AsyncDatabase`.from_async_engine(engine)
            >>> # Now use Moltres with your existing async engine
            >>> from moltres.table.schema import column
            >>> await db.create_table("users", [column("id", "INTEGER")]).collect()
        """
        from ..config import create_config
        from typing import cast, Any

        # Type cast needed because mypy doesn't understand **options unpacking
        config = create_config(engine=engine, **cast(dict[str, Any], options))
        return cls(config=config)

    @classmethod
    def from_async_connection(
        cls, connection: "AsyncConnection", **options: object
    ) -> "AsyncDatabase":
        """Create an :class:`AsyncDatabase` instance from an existing SQLAlchemy AsyncConnection.

        This allows you to use Moltres with an existing SQLAlchemy AsyncConnection,
        enabling integration within existing async transactions.

        Note: The :class:`AsyncDatabase` will use the AsyncConnection's engine, but will not manage
        the AsyncConnection's lifecycle. The user is responsible for managing the connection.

        Args:
            connection: SQLAlchemy AsyncConnection instance
            **options: Optional configuration parameters (same as from_async_engine)

        Returns:
            :class:`AsyncDatabase` instance configured to use the AsyncConnection's engine

        Example:
            >>> from sqlalchemy.ext.asyncio import create_async_engine
            >>> from moltres import :class:`AsyncDatabase`
            >>> engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            >>> async with engine.connect() as conn:
            ...     db = :class:`AsyncDatabase`.from_async_connection(conn)
            ...     # Use Moltres within the connection's transaction
            ...     from moltres.table.schema import column
            ...     await db.create_table("users", [column("id", "INTEGER")]).collect()
        """
        # Extract engine from connection
        engine = connection.engine
        return cls.from_async_engine(engine, **options)

    @classmethod
    def from_async_session(cls, session: "AsyncSession", **options: object) -> "AsyncDatabase":
        """Create an :class:`AsyncDatabase` instance from a SQLAlchemy AsyncSession.

        This allows you to use Moltres with an existing SQLAlchemy AsyncSession,
        enabling integration with async ORM-based applications.

        Note: The :class:`AsyncDatabase` will use the AsyncSession's bind/engine, but will not manage
        the AsyncSession's lifecycle. The user is responsible for managing the session.

        Args:
            session: SQLAlchemy AsyncSession instance
            **options: Optional configuration parameters (same as from_async_engine)

        Returns:
            :class:`AsyncDatabase` instance configured to use the AsyncSession's bind/engine

        Example:
            >>> from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
            >>> from moltres import :class:`AsyncDatabase`
            >>> engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            >>> AsyncSession = async_sessionmaker(bind=engine)
            >>> async with AsyncSession() as session:
            ...     db = :class:`AsyncDatabase`.from_async_session(session)
            ...     # Use Moltres with your existing async session
            ...     from moltres.table.schema import column
            ...     await db.create_table("users", [column("id", "INTEGER")]).collect()
        """
        # Extract engine/bind from session
        from sqlalchemy.ext.asyncio import AsyncEngine

        if hasattr(session, "bind") and session.bind is not None:
            engine = session.bind
            # Ensure we have an AsyncEngine
            if not isinstance(engine, AsyncEngine):
                raise ValueError(
                    "Session bind is not a valid AsyncEngine instance. "
                    "Ensure the session is bound to an async engine."
                )
        elif hasattr(session, "connection"):
            # For async sessions, might have connection instead
            # Note: connection() is async, but we can't await here
            # We'll need to get the sync engine from the async engine
            # This is a limitation - we need the engine, not the connection
            raise ValueError(
                "Cannot extract engine from AsyncSession with connection() method. "
                "Use from_async_engine() or from_async_connection() instead, "
                "or ensure the session has a bind attribute."
            )
        else:
            raise ValueError(
                "AsyncSession does not have a bind or connection. "
                "Ensure the session is bound to an engine."
            )
        return cls.from_async_engine(engine, **options)

    @overload
    async def table(self, name: str) -> AsyncTableHandle:
        """Get a handle to a table in the database from table name."""
        ...

    @overload
    async def table(self, model_class: Type["DeclarativeBase"]) -> AsyncTableHandle:
        """Get a handle to a table in the database from SQLAlchemy model class."""
        ...

    async def table(  # type: ignore[misc]
        self, name_or_model: Union[str, Type["DeclarativeBase"], Type[Any]]
    ) -> AsyncTableHandle:
        """Get a handle to a table in the database.

        Args:
            name_or_model: Name of the table, SQLAlchemy model class, or SQLModel model class

        Returns:
            AsyncTableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
            ValueError: If model_class is not a valid SQLAlchemy or SQLModel model

        Example:
            >>> import asyncio
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice"}], _database=db)
            ...     await records.insert_into("users")
            ...     # Get table handle
            ...     users = await db.table("users")
            ...     df = users.select("id", "name")
            ...     results = await df.collect()
            ...     results[0]["name"]
            ...     'Alice'
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        from ..utils.exceptions import ValidationError
        from ..sql.builders import quote_identifier
        from .sqlalchemy_integration import (
            is_sqlalchemy_model,
            get_model_table_name,
        )
        from ..utils.sqlmodel_integration import (
            is_sqlmodel_model,
            get_sqlmodel_table_name,
        )

        # Check if argument is a SQLModel model
        if is_sqlmodel_model(name_or_model):
            sqlmodel_class: Type[Any] = cast(Type[Any], name_or_model)
            table_name = get_sqlmodel_table_name(sqlmodel_class)
            # Validate table name format
            quote_identifier(table_name, self._dialect.quote_char)
            return AsyncTableHandle(name=table_name, database=self, model=sqlmodel_class)
        # Check if argument is a SQLAlchemy model
        elif is_sqlalchemy_model(name_or_model):
            sa_model_class: Type["DeclarativeBase"] = cast(Type["DeclarativeBase"], name_or_model)
            table_name = get_model_table_name(sa_model_class)
            # Validate table name format
            quote_identifier(table_name, self._dialect.quote_char)
            return AsyncTableHandle(name=table_name, database=self, model=sa_model_class)
        else:
            # Type narrowing: after model checks, this must be str
            table_name = cast(str, name_or_model)
            if not table_name:
                raise ValidationError("Table name cannot be empty")
            # Validate table name format
            quote_identifier(table_name, self._dialect.quote_char)
            return AsyncTableHandle(name=table_name, database=self)

    @property
    def load(self) -> "AsyncDataLoader":
        """Return an AsyncDataLoader for loading data from files and tables as AsyncDataFrames.

        Note: For SQL operations on tables, use await db.table(name).select() instead.
        """
        from ..dataframe.io.async_reader import AsyncDataLoader

        return AsyncDataLoader(self)

    @property
    def read(self) -> "AsyncReadAccessor":
        """Return an AsyncReadAccessor for accessing read operations.

        Use await db.read.records.* for :class:`AsyncRecords`-based reads (backward compatibility).
        Use db.load.* for AsyncDataFrame-based reads (PySpark-style).
        """
        from ..dataframe.io.async_reader import AsyncReadAccessor

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
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": "Bob", "age": 17}], _database=db)
            ...     await records.insert_into("users")
            ...     # Basic SQL query
            ...     df = db.sql("SELECT * FROM users WHERE age > 18")
            ...     results = await df.collect()
            ...     len(results)
            ...     1
            ...     # Parameterized query
            ...     df2 = db.sql("SELECT * FROM users WHERE id = :id AND name = :name", id=1, name="Alice")
            ...     results2 = await df2.collect()
            ...     results2[0]["name"]
            ...     'Alice'
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        from ..dataframe.core.async_dataframe import AsyncDataFrame
        from ..logical import operators

        # Convert params dict to the format expected by RawSQL
        params_dict = params if params else None
        plan = operators.raw_sql(sql, params_dict)
        return AsyncDataFrame(plan=plan, database=self)

    async def scan_csv(
        self,
        path: str,
        schema: Optional[Sequence["ColumnDef"]] = None,
        **options: object,
    ) -> "AsyncPolarsDataFrame":
        """Scan a CSV file as an AsyncPolarsDataFrame (Polars-style).

        Args:
            path: Path to the CSV file
            schema: Optional explicit schema
            **options: Format-specific options (e.g., header=True, delimiter=",")

        Returns:
            AsyncPolarsDataFrame containing the CSV data (lazy)

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.scan_csv("data.csv", header=True)
            >>> results = await df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)
        df = await loader.csv(path)
        result = df.polars()

        return cast("AsyncPolarsDataFrame", result)

    async def scan_json(
        self,
        path: str,
        schema: Optional[Sequence["ColumnDef"]] = None,
        **options: object,
    ) -> "AsyncPolarsDataFrame":
        """Scan a JSON file (array of objects) as an AsyncPolarsDataFrame (Polars-style).

        Args:
            path: Path to the JSON file
            schema: Optional explicit schema
            **options: Format-specific options (e.g., multiline=True)

        Returns:
            AsyncPolarsDataFrame containing the JSON data (lazy)

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.scan_json("data.json")
            >>> results = await df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)
        df = await loader.json(path)
        result = df.polars()

        return cast("AsyncPolarsDataFrame", result)

    async def scan_jsonl(
        self,
        path: str,
        schema: Optional[Sequence["ColumnDef"]] = None,
        **options: object,
    ) -> "AsyncPolarsDataFrame":
        """Scan a JSONL file (one JSON object per line) as an AsyncPolarsDataFrame (Polars-style).

        Args:
            path: Path to the JSONL file
            schema: Optional explicit schema
            **options: Format-specific options

        Returns:
            AsyncPolarsDataFrame containing the JSONL data (lazy)

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.scan_jsonl("data.jsonl")
            >>> results = await df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)
        df = await loader.jsonl(path)
        result = df.polars()

        return cast("AsyncPolarsDataFrame", result)

    async def scan_parquet(
        self,
        path: str,
        schema: Optional[Sequence["ColumnDef"]] = None,
        **options: object,
    ) -> "AsyncPolarsDataFrame":
        """Scan a Parquet file as an AsyncPolarsDataFrame (Polars-style).

        Args:
            path: Path to the Parquet file
            schema: Optional explicit schema
            **options: Format-specific options

        Returns:
            AsyncPolarsDataFrame containing the Parquet data (lazy)

        Raises:
            RuntimeError: If pandas or pyarrow are not installed

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.scan_parquet("data.parquet")
            >>> results = await df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)
        df = await loader.parquet(path)
        result = df.polars()

        return cast("AsyncPolarsDataFrame", result)

    async def scan_text(
        self,
        path: str,
        column_name: str = "value",
        schema: Optional[Sequence["ColumnDef"]] = None,
        **options: object,
    ) -> "AsyncPolarsDataFrame":
        """Scan a text file as a single column AsyncPolarsDataFrame (Polars-style).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")
            schema: Optional explicit schema
            **options: Format-specific options

        Returns:
            AsyncPolarsDataFrame containing the text file lines (lazy)

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.scan_text("data.txt", column_name="line")
            >>> results = await df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)
        df = await loader.text(path, column_name=column_name)
        result = df.polars()

        return cast("AsyncPolarsDataFrame", result)

    # -------------------------------------------------------------- DDL operations
    @overload
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
        """Create a lazy async create table operation from table name and columns."""
        ...

    @overload
    def create_table(
        self,
        model_class: Type["DeclarativeBase"],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
    ) -> "AsyncCreateTableOperation":
        """Create a lazy async create table operation from SQLAlchemy model class."""
        ...

    def create_table(  # type: ignore[misc]
        self,
        name_or_model: Union[str, Type["DeclarativeBase"]],
        columns: Optional[Sequence[ColumnDef]] = None,
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
        constraints: Optional[
            Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]]
        ] = None,
    ) -> "AsyncCreateTableOperation":
        """Create a lazy async create table operation.

        Args:
            name_or_model: Name of the table to create, or SQLAlchemy model class
            columns: Sequence of ColumnDef objects defining the table schema (required if name_or_model is str)
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)
            constraints: Optional sequence of constraint objects (UniqueConstraint, CheckConstraint, ForeignKeyConstraint).
                        Ignored if model_class is provided (constraints are extracted from model).

        Returns:
            AsyncCreateTableOperation that executes on collect()

        Example:
            >>> import asyncio
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column, unique, check
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     # Create table with schema
            ...     op = db.create_table(
            ...         "users",
            ...         [column("id", "INTEGER", primary_key=True), column("email", "TEXT")],
            ...         constraints=[unique("email"), check("id > 0", name="ck_positive_id")]
            ...     )
            ...     table = await op.collect()  # Executes the CREATE TABLE
            ...     # Verify table was created
            ...     tables = await db.get_table_names()
            ...     "users" in tables
            ...     True
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP

        Raises:
            ValidationError: If table name or columns are invalid
            ValueError: If model_class is not a valid SQLAlchemy model

        Example:
            >>> op = db.create_table("users", [column("id", "INTEGER")])
            >>> table = await op.collect()  # Executes the CREATE TABLE

            >>> # Or with SQLAlchemy model
            >>> from sqlalchemy.orm import DeclarativeBase
            >>> class User(Base):
            ...     __tablename__ = "users"
            ...     id = :class:`Column`(Integer, primary_key=True)
            >>> op = db.create_table(User)
            >>> table = await op.collect()
        """
        from .async_actions import AsyncCreateTableOperation
        from .table_operations_helpers import build_create_table_params

        params = build_create_table_params(
            name_or_model,
            columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
            constraints=constraints,
        )

        return AsyncCreateTableOperation(
            database=self,
            name=params.name,
            columns=params.columns,
            if_not_exists=params.if_not_exists,
            temporary=params.temporary,
            constraints=params.constraints,
            model=params.model,
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
            columns: :class:`Column` name(s) to index (single string or sequence)
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
            >>> import asyncio
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER")]).collect()
            ...     await db.create_table("orders", [column("id", "INTEGER")]).collect()
            ...     # Get all table names
            ...     tables = await db.get_table_names()
            ...     "users" in tables
            ...     True
            ...     "orders" in tables
            ...     True
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
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
                    return cast("list[str]", inspector.get_table_names(schema=schema))

                return await conn.run_sync(_inspect_sync)
        except (ValueError, RuntimeError):
            # Re-raise ValueError and RuntimeError as-is
            raise
        except Exception as e:
            # Wrap other exceptions (e.g., SQLAlchemy errors) in RuntimeError
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
                    return cast("list[str]", inspector.get_view_names(schema=schema))

                return await conn.run_sync(_inspect_sync)
        except (ValueError, RuntimeError):
            # Re-raise ValueError and RuntimeError as-is
            raise
        except Exception as e:
            # Wrap other exceptions (e.g., SQLAlchemy errors) in RuntimeError
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
                    return cast("list[dict[str, Any]]", inspector.get_columns(table_name))

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

    async def execute_plan(
        self, plan: LogicalPlan, model: Optional[Type[Any]] = None
    ) -> AsyncQueryResult:
        """Execute a logical plan and return results."""
        stmt = self.compile_plan(plan)
        return await self._executor.fetch(stmt, model=model)

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

    def is_in_transaction(self) -> bool:
        """Check if currently in a transaction.

        Returns:
            True if a transaction is active, False otherwise

        Example:
            >>> if db.is_in_transaction():
            ...     status = db.get_transaction_status()
            ...     print(f"Isolation: {status['isolation_level']}")
        """
        return self._connections.active_transaction is not None

    def get_transaction_status(self) -> Optional[dict[str, object]]:
        """Get transaction status and metadata if a transaction is active.

        Returns:
            Dictionary with transaction metadata including:
            - readonly: Whether the transaction is read-only
            - isolation_level: Transaction isolation level (if set)
            - timeout: Transaction timeout in seconds (if set)
            - savepoints: List of active savepoint names
            None if no transaction is active

        Example:
            >>> async with db.transaction(isolation_level="SERIALIZABLE", readonly=True) as txn:
            ...     status = db.get_transaction_status()
            ...     assert status["isolation_level"] == "SERIALIZABLE"
            ...     assert status["readonly"] is True
        """
        if not self.is_in_transaction():
            return None

        metadata = self._connections.transaction_metadata or {}
        return {
            "readonly": metadata.get("readonly", False),
            "isolation_level": metadata.get("isolation_level"),
            "timeout": metadata.get("timeout"),
            "savepoints": self._connections.savepoint_stack,
        }

    @asynccontextmanager
    async def transaction(
        self,
        savepoint: bool = False,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[AsyncTransaction]:
        """Create an async transaction context for grouping multiple operations.

        All operations within the transaction context share the same transaction.
        If any exception occurs, the transaction is automatically rolled back.
        Otherwise, it is committed on successful exit.

        Args:
            savepoint: If True and a transaction is already active, create a savepoint
                      instead of raising an error. This enables nested transactions.
            readonly: If True, set the transaction to read-only mode. Prevents writes.
            isolation_level: Optional isolation level. Must be one of:
                           - "READ UNCOMMITTED"
                           - "READ COMMITTED"
                           - "REPEATABLE READ"
                           - "SERIALIZABLE"
            timeout: Optional transaction timeout in seconds. Database-specific behavior:
                    - PostgreSQL: Sets statement_timeout (milliseconds)
                    - MySQL: Sets innodb_lock_wait_timeout (seconds)
                    - SQLite: Not supported

        Yields:
            AsyncTransaction object that can be used for explicit commit/rollback

        Example:
            Basic transaction::

                >>> async with db.transaction() as txn:
                ...     await df.write.insertInto("table")
                ...     await df.write.update("table", where=..., set={...})
                ...     # If any operation fails, all are rolled back
                ...     # Otherwise, all are committed on exit

            Nested transaction with savepoint::

                >>> async with db.transaction() as outer:
                ...     # ... operations ...
                ...     async with db.transaction(savepoint=True) as inner:
                ...         # ... operations that can be rolled back independently ...
                ...         await inner.savepoint("checkpoint")
                ...         # ... operations ...
                ...         await inner.rollback_to_savepoint("checkpoint")
                ...     # outer transaction continues...

            Read-only transaction::

                >>> async with db.transaction(readonly=True) as txn:
                ...     results = await db.table("users").select().collect()

            Transaction with isolation level::

                >>> async with db.transaction(isolation_level="SERIALIZABLE") as txn:
                ...     # ... critical operations requiring highest isolation ...
        """
        # Check if there's already an active transaction (for savepoint detection)
        had_active_transaction = self._connections.active_transaction is not None

        connection = await self._connections.begin_transaction(
            savepoint=savepoint,
            readonly=readonly,
            isolation_level=isolation_level,
            timeout=timeout,
        )
        metadata = self._connections.transaction_metadata or {}

        # If savepoint=True was requested and there was already an active transaction,
        # this is a savepoint transaction
        is_savepoint_txn = savepoint and had_active_transaction

        txn = AsyncTransaction(
            self,
            connection,
            readonly=bool(metadata.get("readonly", False)) if metadata else readonly,
            isolation_level=cast(Optional[str], metadata.get("isolation_level"))
            if metadata
            else isolation_level,
            is_savepoint=is_savepoint_txn,
        )

        # Track savepoint name if this is a savepoint
        savepoint_name: Optional[str] = None
        if is_savepoint_txn:
            savepoint_stack = self._connections.savepoint_stack
            if savepoint_stack:
                savepoint_name = savepoint_stack[-1]

        try:
            yield txn
            if not txn._committed and not txn._rolled_back:
                if is_savepoint_txn and savepoint_name:
                    # For savepoints, we don't commit - the outer transaction handles it
                    # But we should release the savepoint
                    try:
                        await txn.release_savepoint(savepoint_name)
                    except RuntimeError:
                        # Savepoint may have already been released
                        pass
                else:
                    await txn.commit()
        except Exception:
            if not txn._rolled_back:
                if is_savepoint_txn and savepoint_name:
                    # For savepoints, rollback to the savepoint
                    try:
                        await txn.rollback_to_savepoint(savepoint_name)
                    except RuntimeError:
                        # Fallback to regular rollback if savepoint rollback fails
                        await txn.rollback()
                else:
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
        """Create an AsyncDataFrame from Python data (list of dicts, list of tuples, :class:`AsyncRecords`, AsyncLazyRecords, pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame).

        Creates a temporary table, inserts the data, and returns an AsyncDataFrame querying from that table.
        If AsyncLazyRecords is provided, it will be auto-materialized.
        If pandas/polars :class:`DataFrame` or LazyFrame is provided, it will be converted to :class:`Records` with lazy conversion.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - :class:`AsyncRecords` object: Extracts data and schema if available
                - AsyncLazyRecords object: Auto-materializes and extracts data and schema
                - pandas :class:`DataFrame`: Converts to :class:`Records` with schema preservation
                - polars :class:`DataFrame`: Converts to :class:`Records` with schema preservation
                - polars LazyFrame: Materializes and converts to :class:`Records` with schema preservation
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
        from ..dataframe.core.async_dataframe import AsyncDataFrame
        from ..dataframe.core.create_dataframe import (
            ensure_primary_key,
            generate_unique_table_name,
            get_schema_from_records,
        )
        from ..dataframe.io.readers.schema_inference import infer_schema_from_rows
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

    async def __aenter__(self) -> "AsyncDatabase":
        """Enter the async database context manager.

        Returns:
            AsyncDatabase: This database instance

        Example:
            >>> async with async_connect("sqlite+aiosqlite:///:memory:") as db:
            ...     df = db.sql("SELECT * FROM users")
            ...     results = await df.collect()
            ...     # await db.close() called automatically on exit
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the async database context manager.

        Automatically closes the database connection when exiting the context,
        even if an exception occurred.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise
        """
        await self.close()

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
        # If session is provided, extract dialect from session's bind (engine)
        if self.config.engine.session is not None:
            session = self.config.engine.session
            if hasattr(session, "get_bind"):
                bind = session.get_bind()
            elif hasattr(session, "bind"):
                bind = session.bind
            else:
                bind = None
            if bind is not None:
                dialect_name = getattr(getattr(bind, "dialect", None), "name", None)
                if dialect_name:
                    # Normalize driver variants (e.g., "mysql+aiomysql" -> "mysql")
                    if "+" in dialect_name:
                        dialect_name = dialect_name.split("+", 1)[0]
                    return str(dialect_name)
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
    """Best-effort cleanup for :class:`AsyncDatabase` instances left open at exit.

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

        result_queue: queue.Queue[Union[Exception, None]] = queue.Queue()

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
        # Modern approach: avoid deprecated get_event_loop(), just create new loop
        try:
            # Try to get existing event loop (may be None or closed)
            # Use get_event_loop() only as fallback, but prefer new_event_loop()
            try:
                # Check if there's an event loop set (even if not running)
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Loop exists but is closed - create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                # No event loop exists, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(_cleanup())
        except RuntimeError:
            # No event loop at all, create one
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(_cleanup())
            finally:
                loop.close()


atexit.register(_cleanup_all_async_databases)
