"""SQLAlchemy connection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

# Import duckdb_engine to register the dialect with SQLAlchemy
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from ..config import EngineConfig
from ..engine.dialects import DialectSpec, get_dialect


class ConnectionManager:
    """Creates and caches SQLAlchemy engines for Moltres sessions."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine: Engine | None = None
        self._session: object | None = None  # SQLAlchemy Session
        self._active_transaction: Optional[Connection] = None
        self._savepoint_stack: list[str] = []
        self._transaction_metadata: Optional[dict[str, object]] = None

    def _create_engine(self) -> Engine:
        # If a session is provided, extract engine from it
        if self.config.session is not None:
            session = self.config.session
            # Check if it's a SQLAlchemy Session or SQLModel Session
            if hasattr(session, "get_bind"):
                # SQLAlchemy 2.0 style
                bind = session.get_bind()
            elif hasattr(session, "bind"):
                # SQLAlchemy 1.x style
                bind = session.bind
            else:
                raise TypeError(
                    "session must be a SQLAlchemy Session or SQLModel Session instance. "
                    f"Got: {type(session).__name__}"
                )
            if not isinstance(bind, Engine):
                raise TypeError(
                    "Session's bind must be a synchronous Engine, not AsyncEngine. "
                    "Use async_connect() for async sessions."
                )
            self._session = session
            return bind

        # If an engine is provided in config, use it directly
        if self.config.engine is not None:
            if not isinstance(self.config.engine, Engine):
                raise TypeError("config.engine must be a synchronous Engine, not AsyncEngine")
            return self.config.engine

        # Otherwise, create a new engine from DSN
        if self.config.dsn is None:
            raise ValueError(
                "Either 'dsn', 'engine', or 'session' must be provided in EngineConfig"
            )

        kwargs: dict[str, object] = {"echo": self.config.echo, "future": self.config.future}
        if self.config.pool_size is not None:
            kwargs["pool_size"] = self.config.pool_size
        if self.config.max_overflow is not None:
            kwargs["max_overflow"] = self.config.max_overflow
        if self.config.pool_timeout is not None:
            kwargs["pool_timeout"] = self.config.pool_timeout
        if self.config.pool_recycle is not None:
            kwargs["pool_recycle"] = self.config.pool_recycle
        if self.config.pool_pre_ping:
            kwargs["pool_pre_ping"] = self.config.pool_pre_ping
        return create_engine(self.config.dsn, **kwargs)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @contextmanager
    def connect(self, transaction: Optional[Connection] = None) -> Iterator[Connection]:
        """Get a database connection.

        Args:
            transaction: If provided, use this transaction connection instead of creating a new one.
                        This allows operations to share a transaction.
                        If None and an active transaction exists, uses the active transaction.

        Yields:
            :class:`Database` connection
        """
        if transaction is not None:
            # Use the provided transaction connection
            yield transaction
        elif self._active_transaction is not None:
            # Use the active transaction connection automatically
            yield self._active_transaction
        elif self._session is not None:
            # Use the session's connection
            # SQLAlchemy sessions have a connection() method
            if hasattr(self._session, "connection"):
                # Get connection from session
                connection = self._session.connection()
                yield connection
            else:
                # Fallback: use session's bind to create a connection
                with self.engine.begin() as connection:
                    yield connection
        else:
            # Create a new connection with auto-commit (default behavior)
            with self.engine.begin() as connection:
                yield connection

    def begin_transaction(
        self,
        savepoint: bool = False,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Connection:
        """Begin a new transaction and return the connection.

        Args:
            savepoint: If True and a transaction is already active, create a savepoint instead.
            readonly: If True, set transaction to read-only mode.
            isolation_level: Optional isolation level (READ UNCOMMITTED, READ COMMITTED,
                           REPEATABLE READ, SERIALIZABLE).
            timeout: Optional transaction timeout in seconds.

        Returns:
            Connection that is part of a transaction (not auto-committed)

        Raises:
            RuntimeError: If savepoint=False and a transaction is already active.
            ValueError: If isolation level or readonly is requested but not supported by dialect.
        """
        if self._active_transaction is not None:
            if savepoint:
                # Create a savepoint instead of a new transaction
                savepoint_name = self._generate_savepoint_name()
                return self.create_savepoint(self._active_transaction, savepoint_name)
            else:
                raise RuntimeError(
                    "Transaction already active. Use savepoint=True for nested transactions."
                )

        # Get dialect for feature checking
        dialect_name = self.engine.dialect.name
        try:
            dialect_spec = get_dialect(dialect_name)
        except ValueError:
            # Unknown dialect, use conservative defaults
            dialect_spec = DialectSpec(name=dialect_name)

        self._active_transaction = self.engine.connect()
        self._savepoint_stack = []
        self._transaction_metadata = {
            "readonly": readonly,
            "isolation_level": isolation_level,
            "timeout": timeout,
        }

        # Set isolation level if specified
        if isolation_level:
            if not dialect_spec.supports_isolation_levels:
                self._active_transaction.close()
                self._active_transaction = None
                raise ValueError(
                    f"Dialect '{dialect_name}' does not support isolation levels. "
                    "SQLite only supports SERIALIZABLE and READ UNCOMMITTED via PRAGMA."
                )
            self._set_isolation_level(self._active_transaction, isolation_level)

        # Set read-only mode if specified
        if readonly:
            if not dialect_spec.supports_read_only_transactions:
                self._active_transaction.close()
                self._active_transaction = None
                raise ValueError(
                    f"Dialect '{dialect_name}' does not support read-only transactions."
                )
            self._set_readonly(self._active_transaction, True)

        # Set timeout if specified
        if timeout:
            self._set_timeout(self._active_transaction, timeout, dialect_name)

        self._active_transaction.begin()
        return self._active_transaction

    def _generate_savepoint_name(self) -> str:
        """Generate a unique savepoint name."""
        return f"sp_{len(self._savepoint_stack)}"

    def _set_isolation_level(self, connection: Connection, isolation_level: str) -> None:
        """Set transaction isolation level."""
        # Normalize isolation level names
        level_map = {
            "READ UNCOMMITTED": "READ UNCOMMITTED",
            "READ COMMITTED": "READ COMMITTED",
            "REPEATABLE READ": "REPEATABLE READ",
            "SERIALIZABLE": "SERIALIZABLE",
        }
        normalized = level_map.get(isolation_level.upper())
        if not normalized:
            raise ValueError(
                f"Invalid isolation level '{isolation_level}'. "
                "Must be one of: READ UNCOMMITTED, READ COMMITTED, REPEATABLE READ, SERIALIZABLE"
            )

        stmt = text(f"SET TRANSACTION ISOLATION LEVEL {normalized}")
        connection.execute(stmt)

    def _set_readonly(self, connection: Connection, readonly: bool) -> None:
        """Set transaction to read-only mode."""
        mode = "READ ONLY" if readonly else "READ WRITE"
        stmt = text(f"SET TRANSACTION {mode}")
        connection.execute(stmt)

    def _set_timeout(self, connection: Connection, timeout: float, dialect_name: str) -> None:
        """Set transaction timeout (database-specific)."""
        # PostgreSQL uses statement_timeout (in milliseconds)
        if "postgresql" in dialect_name:
            stmt = text(f"SET statement_timeout = {int(timeout * 1000)}")
            connection.execute(stmt)
        # MySQL uses innodb_lock_wait_timeout (in seconds)
        elif "mysql" in dialect_name:
            stmt = text(f"SET innodb_lock_wait_timeout = {int(timeout)}")
            connection.execute(stmt)
        # SQLite doesn't support transaction timeouts directly
        # Other databases may need specific implementations

    def create_savepoint(self, connection: Connection, name: str) -> Connection:
        """Create a savepoint in the current transaction.

        Args:
            connection: The transaction connection
            name: Savepoint name

        Returns:
            The same connection (for compatibility)

        Raises:
            RuntimeError: If no transaction is active or connection doesn't match active transaction.
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        if not self._active_transaction:
            raise RuntimeError("No active transaction")

        # Get dialect for feature checking
        dialect_name = self.engine.dialect.name
        try:
            dialect_spec = get_dialect(dialect_name)
        except ValueError:
            dialect_spec = DialectSpec(name=dialect_name)

        if not dialect_spec.supports_savepoints:
            raise ValueError(f"Dialect '{dialect_name}' does not support savepoints.")

        stmt = text(f"SAVEPOINT {name}")
        connection.execute(stmt)
        self._savepoint_stack.append(name)
        return connection

    def rollback_to_savepoint(self, connection: Connection, name: str) -> None:
        """Rollback to a specific savepoint.

        Args:
            connection: The transaction connection
            name: Savepoint name to rollback to

        Raises:
            RuntimeError: If no transaction is active, connection doesn't match, or savepoint not found.
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        if not self._active_transaction:
            raise RuntimeError("No active transaction")
        if name not in self._savepoint_stack:
            raise RuntimeError(f"Savepoint '{name}' not found in current transaction")

        stmt = text(f"ROLLBACK TO SAVEPOINT {name}")
        connection.execute(stmt)

        # Remove all savepoints after the one we're rolling back to
        index = self._savepoint_stack.index(name)
        self._savepoint_stack = self._savepoint_stack[: index + 1]

    def release_savepoint(self, connection: Connection, name: str) -> None:
        """Release a savepoint.

        Args:
            connection: The transaction connection
            name: Savepoint name to release

        Raises:
            RuntimeError: If no transaction is active, connection doesn't match, or savepoint not found.
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        if not self._active_transaction:
            raise RuntimeError("No active transaction")
        if name not in self._savepoint_stack:
            raise RuntimeError(f"Savepoint '{name}' not found in current transaction")

        stmt = text(f"RELEASE SAVEPOINT {name}")
        connection.execute(stmt)
        self._savepoint_stack.remove(name)

    def commit_transaction(self, connection: Connection) -> None:
        """Commit a transaction.

        Args:
            connection: The transaction connection to commit
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        try:
            connection.commit()
        finally:
            # Always close connection, even if commit fails
            connection.close()
            self._active_transaction = None
            self._savepoint_stack = []
            self._transaction_metadata = None

    def rollback_transaction(self, connection: Connection) -> None:
        """Rollback a transaction.

        Args:
            connection: The transaction connection to rollback
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        try:
            connection.rollback()
        finally:
            # Always close connection, even if rollback fails
            connection.close()
            self._active_transaction = None
            self._savepoint_stack = []
            self._transaction_metadata = None

    @property
    def active_transaction(self) -> Optional[Connection]:
        """Get the active transaction connection if one exists."""
        return self._active_transaction

    @property
    def transaction_metadata(self) -> Optional[dict[str, object]]:
        """Get transaction metadata if a transaction is active."""
        return self._transaction_metadata

    @property
    def savepoint_stack(self) -> list[str]:
        """Get the current savepoint stack."""
        return self._savepoint_stack.copy()
