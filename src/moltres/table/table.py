"""Table access primitives.

This module provides the core database and table access functionality:

- :class:`Database` - Main database connection and query interface
- :class:`TableHandle` - Lightweight reference to a database table
- :class:`Transaction` - Transaction context manager

The :class:`Database` class is the primary entry point for all database operations,
including table creation, querying, and data mutations.
"""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import logging
import signal
from types import FrameType, TracebackType
import weakref
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
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
    from sqlalchemy.engine import Engine
    from sqlalchemy.orm import DeclarativeBase, Session
    from sqlalchemy.sql import Select
    from ..dataframe.core.dataframe import DataFrame
    from ..dataframe.interfaces.pandas_dataframe import PandasDataFrame
    from ..dataframe.interfaces.polars_dataframe import PolarsDataFrame
    from ..dataframe.io.reader import DataLoader, ReadAccessor
    from ..expressions.column import Column
    from ..io.records import LazyRecords, Records
    from ..utils.inspector import ColumnInfo
    from .actions import (
        CreateIndexOperation,
        CreateTableOperation,
        DropIndexOperation,
        DropTableOperation,
    )
    from .batch import OperationBatch
    from .schema import (
        CheckConstraint,
        ForeignKeyConstraint,
        TableSchema,
        UniqueConstraint,
    )

    # Type alias for table name or model
    TableNameOrModel = Union[str, Type[DeclarativeBase], Type[Any]]
from sqlalchemy.engine import Connection

from ..engine.connection import ConnectionManager
from ..engine.dialects import DialectSpec, get_dialect
from ..engine.execution import QueryExecutor, QueryResult
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan
from .schema import ColumnDef

logger = logging.getLogger(__name__)
_ACTIVE_DATABASES: "weakref.WeakSet[Database]" = weakref.WeakSet()


@dataclass
class TableHandle:
    """Lightweight handle representing a table reference.

    A :class:`TableHandle` provides access to a specific table in the database.
    It can be created from a table name or from a SQLModel/Pydantic model class.

    Attributes:
        name: Name of the table
        database: The :class:`Database` instance this handle belongs to
        model: Optional SQLModel, Pydantic, or SQLAlchemy model class

    Example:
        >>> db = connect("sqlite:///:memory:")
        >>> handle = db.table("users")
        >>> df = handle.select()
    """

    name: str
    database: "Database"
    model: Optional[Type[Any]] = None  # Can be SQLModel, Pydantic, or SQLAlchemy model

    def __repr__(self) -> str:
        """Return a user-friendly string representation of the TableHandle."""
        if self.model:
            model_name = getattr(self.model, "__name__", str(self.model))
            return f"TableHandle('{self.name}', model={model_name})"
        return f"TableHandle('{self.name}')"

    @property
    def model_class(self) -> Optional[Type["DeclarativeBase"]]:
        """Get the SQLAlchemy model class if this handle was created from a model.

        Returns:
            SQLAlchemy model class or None if handle was created from table name
        """
        return self.model

    def select(self, *columns: str) -> "DataFrame":
        """Select columns from this table.

        Args:
            *columns: Optional column names to select. If empty, selects all columns.

        Returns:
            :class:`DataFrame`: DataFrame with selected columns

        Example:
            >>> db = connect("sqlite:///:memory:")
            >>> handle = db.table("users")
            >>> df = handle.select("id", "name")
        """
        from ..dataframe.core.dataframe import DataFrame

        return DataFrame.from_table(self, columns=list(columns))

    def columns(self) -> List[str]:
        """Get the list of column names for this table.

        Returns:
            List of column names

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> handle = db.table("users")
            >>> cols = handle.columns()
            >>> "id" in cols
            True
            >>> "name" in cols
            True
        """
        column_infos = self.database.get_columns(self.name)
        return [col.name for col in column_infos]

    def pandas(self, *columns: str) -> "PandasDataFrame":
        """Create a :class:`PandasDataFrame` from this table.

        Args:
            *columns: Optional column names to select

        Returns:
            :class:`PandasDataFrame`: :class:`PandasDataFrame` for pandas-style operations

        Example:
            >>> df = db.table('users').pandas()
            >>> df = db.table('users').pandas('id', 'name')
        """
        from ..dataframe.interfaces.pandas_dataframe import PandasDataFrame
        from ..dataframe.core.dataframe import DataFrame

        df = DataFrame.from_table(self, columns=list(columns) if columns else None)
        return PandasDataFrame.from_dataframe(df)

    def polars(self, *columns: str) -> "PolarsDataFrame":
        """Create a :class:`PolarsDataFrame` from this table.

        Args:
            *columns: Optional column names to select

        Returns:
            :class:`PolarsDataFrame`: :class:`PolarsDataFrame` for Polars-style operations

        Example:
            >>> df = db.table('users').polars()
            >>> df = db.table('users').polars('id', 'name')
        """
        from ..dataframe.interfaces.polars_dataframe import PolarsDataFrame
        from ..dataframe.core.dataframe import DataFrame

        df = DataFrame.from_table(self, columns=list(columns) if columns else None)
        return PolarsDataFrame.from_dataframe(df)


class Transaction:
    """:class:`Transaction` context for grouping multiple operations."""

    def __init__(
        self,
        database: "Database",
        connection: Connection,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        is_savepoint: bool = False,
    ):
        """Initialize a transaction context.

        Args:
            database: The database instance this transaction belongs to
            connection: The SQLAlchemy connection for this transaction
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

    def commit(self) -> None:
        """Explicitly commit the transaction.

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        self.database.connection_manager.commit_transaction(self.connection)
        self._committed = True

        # Execute commit hooks
        from ..utils.transaction_hooks import _execute_hooks, _on_commit_hooks

        _execute_hooks(_on_commit_hooks, self)

    def rollback(self) -> None:
        """Explicitly rollback the transaction.

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        self.database.connection_manager.rollback_transaction(self.connection)
        self._rolled_back = True

        # Execute rollback hooks
        from ..utils.transaction_hooks import _execute_hooks, _on_rollback_hooks

        _execute_hooks(_on_rollback_hooks, self)

    def savepoint(self, name: Optional[str] = None) -> str:
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
        self.database.connection_manager.create_savepoint(self.connection, name)
        return name

    def rollback_to_savepoint(self, name: str) -> None:
        """Rollback to a specific savepoint.

        Args:
            name: Savepoint name to rollback to

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back,
                         or if the savepoint doesn't exist
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        self.database.connection_manager.rollback_to_savepoint(self.connection, name)

    def release_savepoint(self, name: str) -> None:
        """Release a savepoint.

        Args:
            name: Savepoint name to release

        Raises:
            RuntimeError: If the transaction has already been committed or rolled back,
                         or if the savepoint doesn't exist
        """
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        self.database.connection_manager.release_savepoint(self.connection, name)

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

    def __enter__(self) -> "Transaction":
        """Enter the transaction context.

        Returns:
            :class:`Transaction`: This transaction instance
        """
        # Execute begin hooks
        from ..utils.transaction_hooks import _execute_hooks, _on_begin_hooks

        _execute_hooks(_on_begin_hooks, self)

        # Start metrics tracking
        import time

        self._metrics_start_time = time.time()
        self._metrics_has_savepoint = self._is_savepoint
        self._metrics_readonly = self._readonly
        self._metrics_isolation_level = self._isolation_level
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the transaction context.

        Automatically commits if no exception occurred, or rolls back if an exception was raised.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise
        """
        if exc_type is not None:
            # Exception occurred, rollback
            if not self._rolled_back and not self._committed:
                self.rollback()
        else:
            # No exception, commit
            if not self._committed and not self._rolled_back:
                self.commit()

        # Record metrics after commit/rollback
        if hasattr(self, "_metrics_start_time"):
            import time
            from ..utils.transaction_metrics import get_transaction_metrics

            duration = time.time() - self._metrics_start_time
            committed = exc_type is None and not self._rolled_back
            metrics = get_transaction_metrics()
            metrics.record_transaction(
                duration=duration,
                committed=committed,
                has_savepoint=self._metrics_has_savepoint,
                readonly=self._metrics_readonly,
                isolation_level=self._metrics_isolation_level,
                error=exc_val if (exc_type and isinstance(exc_val, Exception)) else None,
            )


class Database:
    """Entry-point object returned by :func:`moltres.connect`.

    The :class:`Database` class provides the main interface for all database operations
    in Moltres. It handles connections, query execution, table management, and data
    mutations.

    The :class:`Database` class supports context manager protocol for automatic
    connection cleanup. Use it in a ``with`` statement to ensure the connection
    is properly closed.

    Attributes:
        config: The :class:`MoltresConfig` instance used for this database
        dialect: The SQL dialect being used (e.g., "sqlite", "postgresql")

    Example:
        Basic usage::

            >>> from moltres import connect, col
            >>> db = connect("sqlite:///example.db")
            >>> df = db.table("users").select().where(col("active") == True)
            >>> results = df.collect()
            >>> db.close()

        Using context manager (recommended)::

            >>> with connect("sqlite:///example.db") as db:
            ...     df = db.table("users").select().where(col("active") == True)
            ...     results = df.collect()
            ...     # db.close() called automatically on exit
    """

    def __init__(self, config: MoltresConfig):
        self.config = config
        self._connections = ConnectionManager(config.engine)
        self._executor = QueryExecutor(self._connections, config.engine)
        self._dialect = get_dialect(self._dialect_name)
        self._ephemeral_tables: set[str] = set()
        self._closed = False
        _ACTIVE_DATABASES.add(self)

    def __repr__(self) -> str:
        """Return a user-friendly string representation of the Database."""
        dialect_name = self._dialect_name
        status = "closed" if self._closed else "open"

        # Try to get DSN from config, but sanitize it
        dsn = None
        if self.config.engine.dsn:
            dsn = self.config.engine.dsn
            # Sanitize DSN to hide passwords
            if "://" in dsn:
                parts = dsn.split("://", 1)
                if "@" in parts[1]:
                    # Has credentials - hide password
                    scheme = parts[0]
                    rest = parts[1]
                    if "/" in rest:
                        # Format: user:pass@host/db
                        creds_and_host, db_part = rest.rsplit("/", 1)
                        if "@" in creds_and_host:
                            user_pass, host = creds_and_host.rsplit("@", 1)
                            if ":" in user_pass:
                                user, _ = user_pass.split(":", 1)
                                dsn = f"{scheme}://{user}:***@{host}/{db_part}"
                            else:
                                dsn = f"{scheme}://{user_pass}@{host}/{db_part}"
                    else:
                        # No database part
                        if "@" in rest:
                            user_pass, host = rest.rsplit("@", 1)
                            if ":" in user_pass:
                                user, _ = user_pass.split(":", 1)
                                dsn = f"{scheme}://{user}:***@{host}"
                            else:
                                dsn = f"{scheme}://{user_pass}@{host}"

        if dsn:
            return f"Database(dialect='{dialect_name}', dsn='{dsn}', status='{status}')"
        else:
            return f"Database(dialect='{dialect_name}', status='{status}')"

    @property
    def connection_manager(self) -> ConnectionManager:
        """Get the connection manager for this database.

        Returns:
            ConnectionManager: The connection manager instance
        """
        return self._connections

    @property
    def executor(self) -> QueryExecutor:
        """Get the query executor for this database.

        Returns:
            QueryExecutor: The query executor instance
        """
        return self._executor

    @classmethod
    def from_engine(cls, engine: Engine, **options: object) -> "Database":
        """Create a :class:`Database` instance from an existing SQLAlchemy Engine.

        This allows you to use Moltres with an existing SQLAlchemy Engine,
        enabling integration with existing SQLAlchemy projects.

        Args:
            engine: SQLAlchemy Engine instance
            **options: Optional configuration parameters:
                - echo: Enable SQLAlchemy echo mode
                - fetch_format: Result format - "records", "pandas", or "polars"
                - dialect: Override SQL dialect detection
                - query_timeout: Query execution timeout in seconds
                - Other options are stored in config.options

        Returns:
            :class:`Database`: Database instance configured to use the provided Engine

        Example:
            >>> from sqlalchemy import create_engine
            >>> from moltres import :class:`Database`
            >>> engine = create_engine("sqlite:///:memory:")
            >>> db = :class:`Database`.from_engine(engine)
            >>> # Now use Moltres with your existing engine
            >>> from moltres.table.schema import column
            >>> _ = db.create_table("users", [column("id", "INTEGER")]).collect()
        """
        from ..config import create_config
        from typing import cast, Any

        # Type cast needed because mypy doesn't understand **options unpacking
        config = create_config(engine=engine, **cast(dict[str, Any], options))
        return cls(config=config)

    @classmethod
    def from_connection(cls, connection: Connection, **options: object) -> "Database":
        """Create a :class:`Database` instance from an existing SQLAlchemy Connection.

        This allows you to use Moltres with an existing SQLAlchemy Connection,
        enabling integration within existing transactions.

        Note: The :class:`Database` will use the Connection's engine, but will not manage
        the Connection's lifecycle. The user is responsible for managing the connection.

        Args:
            connection: SQLAlchemy Connection instance
            **options: Optional configuration parameters (same as from_engine)

        Returns:
            :class:`Database` instance configured to use the Connection's engine

        Example:
            >>> from sqlalchemy import create_engine
            >>> from moltres import :class:`Database`
            >>> engine = create_engine("sqlite:///:memory:")
            >>> with engine.connect() as conn:
            ...     db = :class:`Database`.from_connection(conn)
            ...     # Use Moltres within the connection's transaction
            ...     from moltres.table.schema import column
            ...     _ = db.create_table("users", [column("id", "INTEGER")]).collect()
        """
        # Extract engine from connection
        engine = connection.engine
        return cls.from_engine(engine, **options)

    @classmethod
    def from_session(cls, session: "Session", **options: object) -> "Database":
        """Create a :class:`Database` instance from a SQLAlchemy ORM Session.

        This allows you to use Moltres with an existing SQLAlchemy ORM Session,
        enabling integration with ORM-based applications.

        Note: The :class:`Database` will use the Session's bind/engine, but will not manage
        the Session's lifecycle. The user is responsible for managing the session.

        Args:
            session: SQLAlchemy ORM Session instance
            **options: Optional configuration parameters (same as from_engine)

        Returns:
            :class:`Database` instance configured to use the Session's bind/engine

        Example:
            >>> from sqlalchemy import create_engine
            >>> from sqlalchemy.orm import sessionmaker
            >>> from moltres import :class:`Database`
            >>> engine = create_engine("sqlite:///:memory:")
            >>> Session = sessionmaker(bind=engine)
            >>> with Session() as session:
            ...     db = :class:`Database`.from_session(session)
            ...     # Use Moltres with your existing session
            ...     from moltres.table.schema import column
            ...     _ = db.create_table("users", [column("id", "INTEGER")]).collect()
        """
        # Extract engine/bind from session
        if hasattr(session, "bind") and session.bind is not None:
            bind = session.bind
            # Ensure we have an Engine, not a Connection
            if isinstance(bind, Engine):
                engine = bind
            else:
                # If bind is a Connection, get its engine
                if hasattr(bind, "engine"):
                    engine = bind.engine
                    if not isinstance(engine, Engine):
                        raise ValueError(
                            "Session bind's engine is not a valid Engine instance. "
                            "Ensure the session is bound to an engine."
                        )
                else:
                    raise ValueError(
                        "Session bind is not an Engine or Connection. "
                        "Ensure the session is bound to an engine."
                    )
        elif hasattr(session, "connection"):
            # For async sessions, might have connection instead
            conn = session.connection()
            engine = conn.engine
            # Ensure we have an Engine, not a Connection
            if not isinstance(engine, Engine):
                raise ValueError(
                    "Session connection's engine is not a valid Engine instance. "
                    "Ensure the session is bound to an engine."
                )
        else:
            raise ValueError(
                "Session does not have a bind or connection. "
                "Ensure the session is bound to an engine."
            )
        return cls.from_engine(engine, **options)

    def close(self) -> None:
        """Close all database connections and dispose of the engine.

        This should be called when done with the database connection,
        especially for ephemeral test databases.

        Note: After calling close(), the :class:`Database` instance should not be used.
        """
        self._close_resources()

    def __enter__(self) -> "Database":
        """Enter the database context manager.

        Returns:
            Database: This database instance

        Example:
            >>> with connect("sqlite:///example.db") as db:
            ...     df = db.table("users").select()
            ...     results = df.collect()
            ...     # db.close() called automatically on exit
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the database context manager.

        Automatically closes the database connection when exiting the context,
        even if an exception occurred.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise
            exc_val: Exception value if an exception occurred, None otherwise
            exc_tb: Exception traceback if an exception occurred, None otherwise
        """
        self.close()

    def _close_resources(self) -> None:
        if self._closed:
            return
        self._cleanup_ephemeral_tables()
        engine = getattr(self._connections, "_engine", None)
        if engine is not None:
            try:
                engine.dispose(close=True)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error disposing engine during close: %s", exc)
            finally:
                self._connections._engine = None
        self._closed = True
        _ACTIVE_DATABASES.discard(self)

    def _register_ephemeral_table(self, name: str) -> None:
        self._ephemeral_tables.add(name)

    def _unregister_ephemeral_table(self, name: str) -> None:
        self._ephemeral_tables.discard(name)

    def _cleanup_ephemeral_tables(self) -> None:
        """Clean up all ephemeral tables.

        Delegates to :class:`EphemeralTableManager`.
        """
        if not self._ephemeral_tables:
            return
        for table_name in list(self._ephemeral_tables):
            try:
                self.drop_table(table_name, if_exists=True).collect()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to drop ephemeral table %s: %s", table_name, exc)
        self._ephemeral_tables.clear()

    @overload
    def table(self, name: str) -> TableHandle:
        """Get a handle to a table in the database from table name.

        Args:
            name: Name of the table

        Returns:
            :class:`TableHandle`: Handle to the specified table
        """
        ...

    @overload
    def table(self, model_class: Type["DeclarativeBase"]) -> TableHandle:
        """Get a handle to a table in the database from SQLAlchemy model class.

        Args:
            model_class: SQLAlchemy or SQLModel model class

        Returns:
            :class:`TableHandle`: Handle to the table corresponding to the model
        """
        ...

    def table(  # type: ignore[misc]
        self, name_or_model: "TableNameOrModel"
    ) -> TableHandle:
        """Get a handle to a table in the database.

        Delegates to :class:`TableManager`.

        Args:
            name_or_model: Name of the table, SQLAlchemy model class, or SQLModel model class

        Returns:
            TableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
            ValueError: If model_class is not a valid SQLAlchemy or SQLModel model

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> _ = db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
            >>> # Get table handle by name
            >>> users = db.table("users")
            >>> df = users.select("id", "name")
            >>> results = df.collect()
            >>> results[0]["name"]
            'Alice'
            >>> db.close()
        """
        from .table_manager import TableManager

        table_manager = TableManager(self)
        return table_manager.table(name_or_model)

    def insert(
        self,
        table_name: str,
        rows: Union[
            Sequence[Mapping[str, object]],
            "Records",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
    ) -> int:
        """Insert rows into a table.

        Delegates to :class:`TableManager`.

        Args:
            table_name: Name of the table to insert into
            rows: Sequence of row dictionaries, Records, pandas DataFrame, polars DataFrame, or polars LazyFrame

        Returns:
            Number of rows inserted

        Raises:
            ValidationError: If table name is invalid or rows are empty

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> # Insert rows
            >>> count = db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> count
            2
            >>> # Verify insertion
            >>> df = db.table("users").select()
            >>> results = df.collect()
            >>> len(results)
            2
            >>> results[0]["name"]
            'Alice'
            >>> db.close()
        """
        from .table_manager import TableManager

        table_manager = TableManager(self)
        return table_manager.insert(table_name, rows)

    def update(
        self,
        table_name: str,
        *,
        where: "Column",
        set: Mapping[str, object],  # noqa: A002
    ) -> int:
        """Update rows in a table.

        Convenience method for updating data in a table.

        Args:
            table_name: Name of the table to update
            where: :class:`Column` expression for the WHERE clause
            set: Dictionary of column names to new values

        Returns:
            Number of rows updated

        Raises:
            ValidationError: If table name is invalid or set dictionary is empty

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
            >>> # Update rows
            >>> count = db.update("users", where=col("id") == 1, set={"name": "Alice Updated"})
            >>> count
            1
            >>> # Verify update
            >>> df = db.table("users").select()
            >>> results = df.collect()
            >>> results[0]["name"]
            'Alice Updated'
            >>> db.close()
        """
        from .mutations import update_rows

        handle = self.table(table_name)
        return update_rows(handle, where=where, values=set)

    def delete(self, table_name: str, *, where: "Column") -> int:
        """Delete rows from a table.

        Convenience method for deleting data from a table.

        Args:
            table_name: Name of the table to delete from
            where: :class:`Column` expression for the WHERE clause

        Returns:
            Number of rows deleted

        Raises:
            ValidationError: If table name is invalid

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
            >>> # Delete rows
            >>> count = db.delete("users", where=col("id") == 1)
            >>> count
            1
            >>> # Verify deletion
            >>> df = db.table("users").select()
            >>> results = df.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Bob'
            >>> db.close()
        """
        from .mutations import delete_rows

        handle = self.table(table_name)
        return delete_rows(handle, where=where)

    def merge(
        self,
        table_name: str,
        rows: Union[
            Sequence[Mapping[str, object]],
            "Records",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
        *,
        on: Sequence[str],
        when_matched: Optional[Mapping[str, object]] = None,
        when_not_matched: Optional[Mapping[str, object]] = None,
    ) -> int:
        """Merge (upsert) rows into a table.

        Convenience method for merging data into a table with conflict resolution.

        Args:
            table_name: Name of the table to merge into
            rows: Sequence of row dictionaries, :class:`Records`, pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame
            on: Sequence of column names that form the conflict key
            when_matched: Optional dictionary of column updates when a conflict occurs
            when_not_matched: Optional dictionary of default values when inserting new rows

        Returns:
            Number of rows affected (inserted or updated)

        Raises:
            ValidationError: If table name is invalid, rows are empty, or on columns are invalid

        Example:
            >>> db.merge(
            ...     "users",
            ...     [{"id": 1, "name": "Alice", "email": "alice@example.com"}],
            ...     on=["id"],
            ...     when_matched={"name": "Alice Updated"}
            ... )
        """
        from .mutations import merge_rows

        handle = self.table(table_name)
        return merge_rows(
            handle, rows, on=on, when_matched=when_matched, when_not_matched=when_not_matched
        )

    @property
    def load(self) -> "DataLoader":
        """Return a DataLoader for loading data from files and tables as DataFrames.

        Note: For SQL operations on tables, use db.table(name).select() instead.
        """
        from ..dataframe.io.reader import DataLoader

        return DataLoader(self)

    @property
    def read(self) -> "ReadAccessor":
        """Return a ReadAccessor for accessing read operations.

        Use db.read.records.* for :class:`Records`-based reads (backward compatibility).
        Use db.load.* for :class:`DataFrame`-based reads (PySpark-style).
        """
        from ..dataframe.io.reader import ReadAccessor

        return ReadAccessor(self)

    def sql(self, sql: str, **params: object) -> "DataFrame":
        """Execute a SQL query and return a :class:`DataFrame`.

        Similar to PySpark's `spark.sql()`, this method accepts a raw SQL string
        and returns a lazy :class:`DataFrame` that can be chained with further operations.
        The SQL dialect is determined by the database connection.

        Args:
            sql: SQL query string to execute
            **params: Optional named parameters for parameterized queries.
                     Use `:param_name` syntax in SQL and pass values as kwargs.

        Returns:
            Lazy :class:`DataFrame` that can be chained with further operations

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": "Bob", "age": 17}], _database=db).insert_into("users")
            >>> # Basic SQL query
            >>> df = db.sql("SELECT * FROM users WHERE age > 18")
            >>> results = df.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Alice'
            >>> # Parameterized query
            >>> df2 = db.sql("SELECT * FROM users WHERE id = :id", id=1)
            >>> results2 = df2.collect()
            >>> results2[0]["name"]
            'Alice'
            >>> # Chaining operations
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 150.0}, {"id": 2, "amount": 50.0}], _database=db).insert_into("orders")
            >>> df3 = db.sql("SELECT * FROM orders").where(col("amount") > 100).limit(1)
            >>> results3 = df3.collect()
            >>> len(results3)
            1
            >>> results3[0]["amount"]
            150.0
            >>> db.close()
        """
        from ..dataframe.core.dataframe import DataFrame
        from ..logical import operators

        # Convert params dict to the format expected by RawSQL
        params_dict = params if params else None
        plan = operators.raw_sql(sql, params_dict)
        return DataFrame(plan=plan, database=self)

    def scan_csv(
        self,
        path: str,
        schema: Optional[Sequence[ColumnDef]] = None,
        **options: object,
    ) -> "PolarsDataFrame":
        """Scan a CSV file as a :class:`PolarsDataFrame` (Polars-style).

        Args:
            path: Path to the CSV file
            schema: Optional explicit schema
            **options: Format-specific options (e.g., header=True, delimiter=",")

        Returns:
            :class:`PolarsDataFrame` containing the CSV data (lazy)

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.scan_csv("data.csv", header=True)
            >>> results = df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)

        return cast("PolarsDataFrame", loader.csv(path).polars())

    def scan_json(
        self,
        path: str,
        schema: Optional[Sequence[ColumnDef]] = None,
        **options: object,
    ) -> "PolarsDataFrame":
        """Scan a JSON file (array of objects) as a :class:`PolarsDataFrame` (Polars-style).

        Args:
            path: Path to the JSON file
            schema: Optional explicit schema
            **options: Format-specific options (e.g., multiline=True)

        Returns:
            :class:`PolarsDataFrame` containing the JSON data (lazy)

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.scan_json("data.json")
            >>> results = df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)

        return cast("PolarsDataFrame", loader.json(path).polars())

    def scan_jsonl(
        self,
        path: str,
        schema: Optional[Sequence[ColumnDef]] = None,
        **options: object,
    ) -> "PolarsDataFrame":
        """Scan a JSONL file (one JSON object per line) as a :class:`PolarsDataFrame` (Polars-style).

        Args:
            path: Path to the JSONL file
            schema: Optional explicit schema
            **options: Format-specific options

        Returns:
            :class:`PolarsDataFrame` containing the JSONL data (lazy)

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.scan_jsonl("data.jsonl")
            >>> results = df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)

        return cast("PolarsDataFrame", loader.jsonl(path).polars())

    def scan_parquet(
        self,
        path: str,
        schema: Optional[Sequence[ColumnDef]] = None,
        **options: object,
    ) -> "PolarsDataFrame":
        """Scan a Parquet file as a :class:`PolarsDataFrame` (Polars-style).

        Args:
            path: Path to the Parquet file
            schema: Optional explicit schema
            **options: Format-specific options

        Returns:
            :class:`PolarsDataFrame` containing the Parquet data (lazy)

        Raises:
            RuntimeError: If pandas or pyarrow are not installed

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.scan_parquet("data.parquet")
            >>> results = df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)

        return cast("PolarsDataFrame", loader.parquet(path).polars())

    def scan_text(
        self,
        path: str,
        column_name: str = "value",
        schema: Optional[Sequence[ColumnDef]] = None,
        **options: object,
    ) -> "PolarsDataFrame":
        """Scan a text file as a single column :class:`PolarsDataFrame` (Polars-style).

        Args:
            path: Path to the text file
            column_name: Name of the column to create (default: "value")
            schema: Optional explicit schema
            **options: Format-specific options

        Returns:
            :class:`PolarsDataFrame` containing the text file lines (lazy)

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.scan_text("data.txt", column_name="line")
            >>> results = df.collect()
        """
        from .table_operations_helpers import build_scan_loader_chain

        loader = build_scan_loader_chain(self.read, schema, **options)

        return cast("PolarsDataFrame", loader.text(path, column_name=column_name).polars())

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
    ) -> "CreateTableOperation":
        """Create a lazy create table operation from table name and columns."""
        ...

    @overload
    def create_table(
        self,
        model_class: Type["DeclarativeBase"],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
    ) -> "CreateTableOperation":
        """Create a lazy create table operation from SQLAlchemy model class."""
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
    ) -> "CreateTableOperation":
        """Create a lazy create table operation.

        Delegates to :class:`DDLManager`.

        Args:
            name_or_model: Name of the table to create, or SQLAlchemy model class
            columns: Sequence of ColumnDef objects defining the table schema (required if name_or_model is str)
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)
            constraints: Optional sequence of constraint objects (UniqueConstraint, CheckConstraint, ForeignKeyConstraint).
                        Ignored if model_class is provided (constraints are extracted from model).

        Returns:
            CreateTableOperation that executes on collect()

        Raises:
            ValidationError: If table name or columns are invalid
            ValueError: If model_class is not a valid SQLAlchemy model

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column, unique, check
            >>> db = connect("sqlite:///:memory:")
            >>> # Create table with constraints
            >>> op = db.create_table(
            ...     "users",
            ...     [column("id", "INTEGER", primary_key=True), column("email", "TEXT")],
            ...     constraints=[unique("email"), check("id > 0", name="ck_positive_id")]
            ... )
            >>> table = op.collect()  # Executes the CREATE TABLE
            >>> # Verify table was created
            >>> tables = db.get_table_names()
            >>> "users" in tables
            True
            >>> db.close()
        """
        from .ddl_manager import DDLManager

        ddl_manager = DDLManager(self)
        return ddl_manager.create_table(
            name_or_model,
            columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
            constraints=constraints,
        )

    def drop_table(self, name: str, *, if_exists: bool = True) -> "DropTableOperation":
        """Create a lazy drop table operation.

        Delegates to :class:`DDLManager`.

        Args:
            name: Name of the table to drop
            if_exists: If True, don't error if table doesn't exist (default: True)

        Returns:
            DropTableOperation that executes on collect()

        Example:
            >>> op = db.drop_table("users")
            >>> op.collect()  # Executes the DROP TABLE
        """
        from .ddl_manager import DDLManager

        ddl_manager = DDLManager(self)
        return ddl_manager.drop_table(name, if_exists=if_exists)

    def create_index(
        self,
        name: str,
        table: str,
        columns: Union[str, Sequence[str]],
        *,
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> "CreateIndexOperation":
        """Create a lazy create index operation.

        Delegates to :class:`DDLManager`.

        Args:
            name: Name of the index to create
            table: Name of the table to create the index on
            columns: Column name(s) to index (single string or sequence)
            unique: If True, create a UNIQUE index (default: False)
            if_not_exists: If True, don't error if index already exists (default: True)

        Returns:
            CreateIndexOperation that executes on collect()

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("email", "TEXT"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> # Create single-column index
            >>> op = db.create_index("idx_email", "users", "email")
            >>> op.collect()  # Executes the CREATE INDEX
            >>> # Multi-column index
            >>> op2 = db.create_index("idx_name_age", "users", ["name", "age"], unique=True)
            >>> op2.collect()
            >>> db.close()
        """
        from .ddl_manager import DDLManager

        ddl_manager = DDLManager(self)
        return ddl_manager.create_index(
            name, table, columns, unique=unique, if_not_exists=if_not_exists
        )

    def drop_index(
        self,
        name: str,
        table: Optional[str] = None,
        *,
        if_exists: bool = True,
    ) -> "DropIndexOperation":
        """Create a lazy drop index operation.

        Args:
            name: Name of the index to drop
            table: Optional table name (required for some dialects like MySQL)
            if_exists: If True, don't error if index doesn't exist (default: True)

        Returns:
            DropIndexOperation that executes on collect()

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("email", "TEXT")]).collect()
            >>> db.create_index("idx_email", "users", "email").collect()
            >>> # Drop index
            >>> op = db.drop_index("idx_email", "users")
            >>> op.collect()  # Executes the DROP INDEX
            >>> db.close()
        """
        from .ddl_manager import DDLManager

        ddl_manager = DDLManager(self)
        return ddl_manager.drop_index(name, table=table, if_exists=if_exists)

    # -------------------------------------------------------------- schema inspection
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
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
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER")]).collect()
            >>> # Get all table names
            >>> tables = db.get_table_names()
            >>> "users" in tables
            True
            >>> "orders" in tables
            True
            >>> db.close()
        """
        from ..utils.inspector import get_table_names

        return get_table_names(self, schema=schema)

    def get_view_names(self, schema: Optional[str] = None) -> List[str]:
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
            >>> views = db.get_view_names()
            >>> # Returns: ['active_users_view', 'order_summary_view']
        """
        from ..utils.inspector import get_view_names

        return get_view_names(self, schema=schema)

    def schema(self, table_name: str) -> List[ColumnDef]:
        """Get the schema (column definitions) for a table.

        Args:
            table_name: Name of the table

        Returns:
            List of :class:`ColumnDef` objects describing the table's columns

        Raises:
            ValueError: If table does not exist

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> schema = db.schema("users")
            >>> len(schema)
            2
            >>> schema[0].name
            'id'
            >>> schema[0].type_name
            'INTEGER'
        """
        columns = self.get_columns(table_name)
        return [
            ColumnDef(
                name=col.name,
                type_name=col.type_name,
                nullable=col.nullable,
                default=col.default,
                primary_key=col.primary_key,
                precision=col.precision,
                scale=col.scale,
            )
            for col in columns
        ]

    def tables(self, schema: Optional[str] = None) -> Dict[str, List[ColumnDef]]:
        """Get all tables in the database with their schemas.

        Args:
            schema: Optional schema name (for databases that support schemas)

        Returns:
            Dictionary mapping table names to their column definitions

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER")]).collect()
            >>> tables = db.tables()
            >>> "users" in tables
            True
            >>> "orders" in tables
            True
            >>> len(tables["users"])
            1
        """
        table_names = self.get_table_names(schema=schema)
        result = {}
        for table_name in table_names:
            try:
                result[table_name] = self.schema(table_name)
            except Exception as exc:
                logger.debug("Failed to get schema for table %s: %s", table_name, exc)
                # Continue with other tables even if one fails
                result[table_name] = []
        return result

    def get_columns(self, table_name: str) -> List["ColumnInfo"]:
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
            >>> columns = db.get_columns("users")
            >>> # Returns: [ColumnInfo(name='id', type_name='INTEGER', ...), ...]
        """
        from ..utils.exceptions import ValidationError
        from ..utils.inspector import get_table_columns
        from ..sql.builders import quote_identifier

        if not table_name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(table_name, self._dialect.quote_char)

        return get_table_columns(self, table_name)

    def reflect_table(self, name: str, schema: Optional[str] = None) -> "TableSchema":
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
            >>> schema = db.reflect_table("users")
            >>> # Returns: TableSchema(name='users', columns=[ColumnDef(...), ...])
        """
        from ..utils.exceptions import ValidationError
        from ..utils.inspector import reflect_table
        from ..sql.builders import quote_identifier
        from .schema import TableSchema

        if not name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(name, self._dialect.quote_char)

        reflected = reflect_table(self, name, schema=schema)
        column_defs = reflected[name]

        return TableSchema(name=name, columns=column_defs)

    def reflect(
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
            >>> schemas = db.reflect()
            >>> # Returns: {'users': TableSchema(...), 'orders': TableSchema(...)}
        """
        from ..utils.inspector import reflect_database
        from .schema import TableSchema

        reflected = reflect_database(self, schema=schema, views=views)

        # Convert to TableSchema objects
        result: Dict[str, TableSchema] = {}
        for table_name, column_defs in reflected.items():
            result[table_name] = TableSchema(name=table_name, columns=column_defs)

        return result

    # -------------------------------------------------------------- query utils
    def compile_plan(self, plan: LogicalPlan) -> "Select":
        """Compile a logical plan to a SQLAlchemy Select statement."""
        return compile_plan(plan, dialect=self._dialect)

    def execute_plan(self, plan: LogicalPlan, model: Optional[Type[Any]] = None) -> QueryResult:
        """Execute a logical plan and return results.

        Delegates to :class:`DatabaseQueryExecutor`.
        """
        from .query_executor import DatabaseQueryExecutor

        executor = DatabaseQueryExecutor(self)
        return executor.execute_plan(plan, model=model)

    def execute_plan_stream(self, plan: LogicalPlan) -> Iterator[List[Dict[str, object]]]:
        """Execute a plan and return an iterator of row chunks.

        Delegates to :class:`DatabaseQueryExecutor`.
        """
        from .query_executor import DatabaseQueryExecutor

        executor = DatabaseQueryExecutor(self)
        return executor.execute_plan_stream(plan)

    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a raw SQL query.

        Delegates to :class:`DatabaseQueryExecutor`.
        """
        from .query_executor import DatabaseQueryExecutor

        executor = DatabaseQueryExecutor(self)
        return executor.execute_sql(sql, params=params)

    def explain(self, sql: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Get the execution plan for a SQL query.

        Delegates to :class:`DatabaseQueryExecutor`.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            Execution plan as a string (dialect-specific)

        Example:
            >>> plan = db.explain("SELECT * FROM users WHERE id = :id", params={"id": 1})
            >>> print(plan)
        """
        from .query_executor import DatabaseQueryExecutor

        executor = DatabaseQueryExecutor(self)
        return executor.explain(sql, params=params)

    def show_tables(self, schema: Optional[str] = None) -> None:
        """Print a formatted list of tables in the database.

        Convenience method for interactive exploration.

        Args:
            schema: Optional schema name

        Example:
            >>> db.show_tables()
            Tables in database:
            - users
            - orders
            - products
        """
        tables = self.get_table_names(schema=schema)
        if tables:
            print("Tables in database:")
            for table in sorted(tables):
                print(f"  - {table}")
        else:
            print("No tables found in database.")

    def show_schema(self, table_name: str) -> None:
        """Print a formatted schema for a table.

        Convenience method for interactive exploration.

        Args:
            table_name: Name of the table

        Example:
            >>> db.show_schema("users")
            Schema for table 'users':
            - id: INTEGER (primary_key=True)
            - name: TEXT
            - email: TEXT
        """
        from ..utils.exceptions import ValidationError

        if not table_name:
            raise ValidationError("Table name cannot be empty")

        columns = self.get_columns(table_name)
        if columns:
            print(f"Schema for table '{table_name}':")
            for col_info in columns:
                attrs = []
                if col_info.primary_key:
                    attrs.append("primary_key=True")
                if col_info.nullable is False:
                    attrs.append("nullable=False")
                if col_info.default is not None:
                    attrs.append(f"default={col_info.default}")
                attr_str = f" ({', '.join(attrs)})" if attrs else ""
                print(f"  - {col_info.name}: {col_info.type_name}{attr_str}")
        else:
            print(f"No columns found for table '{table_name}'.")

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
            >>> with db.transaction(isolation_level="SERIALIZABLE", readonly=True) as txn:
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

    def batch(self) -> "OperationBatch":
        """Create a batch context for grouping multiple operations.

        All operations within the batch context are executed together in a single transaction
        when the context exits. If any exception occurs, all operations are rolled back.

        Returns:
            OperationBatch context manager

        Example:
            >>> with db.batch():
            ...     db.create_table("users", [...])
            ...     # All operations execute together on exit
        """
        from .batch import OperationBatch

        return OperationBatch(self)

    @contextmanager
    def transaction(
        self,
        savepoint: bool = False,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[Transaction]:
        """Create a transaction context for grouping multiple operations.

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
            :class:`Transaction` object that can be used for explicit commit/rollback

        Example:
            Basic transaction::

                >>> with db.transaction() as txn:
                ...     df.write.insertInto("table")
                ...     df.write.update("table", where=..., set={...})
                ...     # If any operation fails, all are rolled back
                ...     # Otherwise, all are committed on exit

            Nested transaction with savepoint::

                >>> with db.transaction() as outer:
                ...     # ... operations ...
                ...     with db.transaction(savepoint=True) as inner:
                ...         # ... operations that can be rolled back independently ...
                ...         inner.savepoint("checkpoint")
                ...         # ... operations ...
                ...         inner.rollback_to_savepoint("checkpoint")
                ...     # outer transaction continues...

            Read-only transaction::

                >>> with db.transaction(readonly=True) as txn:
                ...     results = db.table("users").select().collect()

            Transaction with isolation level::

                >>> with db.transaction(isolation_level="SERIALIZABLE") as txn:
                ...     # ... critical operations requiring highest isolation ...
        """
        # Check if there's already an active transaction (for savepoint detection)
        had_active_transaction = self._connections.active_transaction is not None

        connection = self._connections.begin_transaction(
            savepoint=savepoint,
            readonly=readonly,
            isolation_level=isolation_level,
            timeout=timeout,
        )
        metadata = self._connections.transaction_metadata or {}

        # If savepoint=True was requested and there was already an active transaction,
        # this is a savepoint transaction
        is_savepoint_txn = savepoint and had_active_transaction

        txn = Transaction(
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

        # Call Transaction's __enter__ to set up hooks and metrics
        from ..utils.transaction_hooks import _execute_hooks, _on_begin_hooks

        # Execute begin hooks
        _execute_hooks(_on_begin_hooks, txn)

        # Start metrics tracking
        import time

        metrics_start_time = time.time()

        exc_info = None
        committed = False
        try:
            yield txn
            if not txn._committed and not txn._rolled_back:
                if is_savepoint_txn and savepoint_name:
                    # For savepoints, we don't commit - the outer transaction handles it
                    # But we should release the savepoint
                    try:
                        txn.release_savepoint(savepoint_name)
                    except RuntimeError:
                        # Savepoint may have already been released
                        pass
                else:
                    txn.commit()
                    committed = True
            else:
                committed = txn._committed
        except Exception as exc:
            exc_info = exc
            if not txn._rolled_back:
                if is_savepoint_txn and savepoint_name:
                    # For savepoints, rollback to the savepoint
                    try:
                        txn.rollback_to_savepoint(savepoint_name)
                    except RuntimeError:
                        # Fallback to regular rollback if savepoint rollback fails
                        txn.rollback()
                else:
                    txn.rollback()
            raise
        finally:
            # Record metrics (always called, even on exception)
            from ..utils.transaction_metrics import get_transaction_metrics

            duration = time.time() - metrics_start_time
            final_committed = committed if not exc_info else False
            metrics = get_transaction_metrics()
            metrics.record_transaction(
                duration=duration,
                committed=final_committed,
                has_savepoint=is_savepoint_txn,
                readonly=readonly,
                isolation_level=isolation_level,
                error=exc_info if exc_info else None,
            )

    def createDataFrame(
        self,
        data: Union[
            Sequence[dict[str, object]],
            Sequence[tuple],
            Records,
            "LazyRecords",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
        schema: Optional[Sequence[ColumnDef]] = None,
        pk: Optional[Union[str, Sequence[str]]] = None,
        auto_pk: Optional[Union[str, Sequence[str]]] = None,
    ) -> "DataFrame":
        """Create a DataFrame from Python data.

        Delegates to :class:`EphemeralTableManager`.

        Creates a temporary table, inserts the data, and returns a DataFrame querying from that table.
        If LazyRecords is provided, it will be auto-materialized.
        If pandas/polars DataFrame or LazyFrame is provided, it will be converted to Records with lazy conversion.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - Records object: Extracts data and schema if available
                - LazyRecords object: Auto-materializes and extracts data and schema
                - pandas DataFrame: Converts to Records with schema preservation
                - polars DataFrame: Converts to Records with schema preservation
                - polars LazyFrame: Materializes and converts to Records with schema preservation
            schema: Optional explicit schema. If not provided, schema is inferred from data.
            pk: Optional column name(s) to mark as primary key. Can be a single string or sequence of strings for composite keys.
            auto_pk: Optional column name(s) to create as auto-incrementing primary key. Can specify same name as pk to make an existing column auto-incrementing.

        Returns:
            DataFrame querying from the created temporary table

        Raises:
            ValueError: If data is empty and no schema provided, or if primary key requirements are not met
            ValidationError: If list of tuples provided without schema, or other validation errors

        Example:
            >>> # Create DataFrame from list of dicts
            >>> df = db.createDataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id")
            >>> # Create DataFrame with auto-incrementing primary key
            >>> df = db.createDataFrame([{"name": "Alice"}, {"name": "Bob"}], auto_pk="id")
            >>> # Create DataFrame from Records
            >>> from moltres.io.records import Records
            >>> records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
            >>> df = db.createDataFrame(records, pk="id")
            >>> # Create DataFrame from LazyRecords (auto-materializes)
            >>> lazy_records = db.read.records.csv("data.csv")
            >>> df = db.createDataFrame(lazy_records, pk="id")
            >>> # Create DataFrame from pandas DataFrame
            >>> import pandas as pd
            >>> pdf = pd.DataFrame([{"id": 1, "name": "Alice"}])
            >>> df = db.createDataFrame(pdf, pk="id")
            >>> # Create DataFrame from polars DataFrame
            >>> import polars as pl
            >>> plf = pl.DataFrame([{"id": 1, "name": "Alice"}])
            >>> df = db.createDataFrame(plf, pk="id")
        """
        from .ephemeral_manager import EphemeralTableManager

        manager = EphemeralTableManager(self)
        return manager.create_dataframe(data, schema=schema, pk=pk, auto_pk=auto_pk)

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
        # Extract dialect from DSN, normalizing driver variants (e.g., "mysql+pymysql" -> "mysql")
        dsn = self.config.engine.dsn
        if not dsn:
            return "ansi"
        dialect_part = dsn.split(":", 1)[0]
        # Normalize driver variants: "mysql+pymysql" -> "mysql", "postgresql+psycopg2" -> "postgresql"
        if "+" in dialect_part:
            dialect_part = dialect_part.split("+", 1)[0]
        return dialect_part


def _cleanup_all_databases() -> None:
    """Best-effort cleanup for any :class:`Database` instances left open at exit.

    This is called on normal interpreter shutdown and on signal handlers
    for crash scenarios (SIGTERM, SIGINT).
    """
    for db in list(_ACTIVE_DATABASES):
        try:
            db._close_resources()
        except Exception as exc:  # pragma: no cover - atexit safeguard
            logger.debug("Database cleanup during interpreter shutdown failed: %s", exc)


def _signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """Handle signals (SIGTERM, SIGINT) by cleaning up databases before exit."""
    logger.info("Received signal %d, cleaning up databases...", signum)
    _cleanup_all_databases()
    # Re-raise the signal with default handler
    signal.signal(signum, signal.SIG_DFL)
    import os

    os.kill(os.getpid(), signum)


# Register signal handlers for crash scenarios (only on main thread)
try:
    # Check if we can register signal handlers (main thread only)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except (ValueError, OSError):
    # Signal handlers can only be registered on the main thread
    # This is expected in some contexts (e.g., subprocesses, threads)
    pass


def _force_database_cleanup_for_tests() -> None:
    """Helper used by tests to simulate crash/GC cleanup."""
    _cleanup_all_databases()


atexit.register(_cleanup_all_databases)
