"""Async SQLAlchemy connection helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import shlex
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Import duckdb_engine to register the dialect with SQLAlchemy
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass

try:
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
    from sqlalchemy.ext.asyncio.engine import AsyncConnection
except ImportError as exc:
    raise ImportError(
        "Async support requires SQLAlchemy 2.0+ with async extensions. "
        "Install with: pip install 'SQLAlchemy>=2.0'"
    ) from exc

from sqlalchemy import text

from ..config import EngineConfig
from ..engine.dialects import DialectSpec, get_dialect


def _extract_postgres_server_settings(dsn: str) -> tuple[str, dict[str, str]]:
    """Convert Postgres DSN ``options`` into asyncpg server settings.

    asyncpg does not support the ``options`` keyword argument that psycopg does.
    SQLAlchemy forwards query parameters from the DSN (e.g. ?options=-csearch_path=foo)
    directly to asyncpg, which raises ``TypeError``. We translate any ``-cKEY=VALUE``
    tokens into asyncpg ``server_settings`` entries and drop the ``options`` query param.
    """

    split = urlsplit(dsn)
    scheme = split.scheme.split("+")[0]
    if scheme != "postgresql":
        return dsn, {}

    query_items = parse_qsl(split.query, keep_blank_values=True)
    if not query_items:
        return dsn, {}

    server_settings: dict[str, str] = {}
    filtered_query: list[tuple[str, str]] = []

    for key, value in query_items:
        if key != "options" or not value:
            filtered_query.append((key, value))
            continue

        for token in shlex.split(value):
            if not token.startswith("-c"):
                continue
            setting = token[2:]
            if "=" not in setting:
                continue
            name, setting_value = setting.split("=", 1)
            name = name.strip()
            if not name:
                continue
            server_settings[name] = setting_value.strip()

    if not server_settings:
        return dsn, {}

    new_query = urlencode(filtered_query, doseq=True)
    normalized = urlunsplit((split.scheme, split.netloc, split.path, new_query, split.fragment))
    return normalized, server_settings


class AsyncConnectionManager:
    """Creates and caches async SQLAlchemy engines for Moltres sessions."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine: AsyncEngine | None = None
        self._session: object | None = None  # SQLAlchemy AsyncSession
        self._active_transaction: Optional[AsyncConnection] = None
        self._savepoint_stack: list[str] = []
        self._transaction_metadata: Optional[dict[str, object]] = None

    def _create_engine(self) -> AsyncEngine:
        """Create an async SQLAlchemy engine.

        Args:
            config: Engine configuration

        Returns:
            AsyncEngine instance

        Raises:
            ValueError: If DSN doesn't support async (missing +asyncpg, +aiomysql, etc.)
        """
        # If a session is provided, extract engine from it
        if self.config.session is not None:
            session = self.config.session
            # Check if it's a SQLAlchemy AsyncSession or SQLModel AsyncSession
            bind = None
            # For AsyncSession, prefer .bind over .get_bind() because get_bind() returns sync Engine
            if hasattr(session, "bind"):
                bind = session.bind
            elif hasattr(session, "get_bind"):
                # SQLAlchemy 2.0 style - but get_bind() might return sync Engine for async sessions
                try:
                    bind = session.get_bind()
                    # If get_bind() returned a sync Engine, it's not what we want
                    if not isinstance(bind, AsyncEngine):
                        bind = None
                except (TypeError, AttributeError):
                    # get_bind() might require arguments
                    pass
            if bind is None:
                # Try to get from sessionmaker or other attributes
                if hasattr(session, "maker") and hasattr(session.maker, "bind"):
                    bind = session.maker.bind
            if bind is None:
                raise TypeError(
                    "session must be a SQLAlchemy AsyncSession or SQLModel AsyncSession instance "
                    f"with a bind (engine) attached. Got: {type(session).__name__}"
                )
            if not isinstance(bind, AsyncEngine):
                raise TypeError(
                    "Session's bind must be an AsyncEngine, not a synchronous Engine. "
                    f"Got: {type(bind).__name__}. Use connect() for sync sessions."
                )
            self._session = session
            return bind

        # If an engine is provided in config, use it directly
        if self.config.engine is not None:
            if not isinstance(self.config.engine, AsyncEngine):
                raise TypeError("engine must be a SQLAlchemy AsyncEngine instance")
            return self.config.engine

        # Otherwise, create a new engine from DSN
        if self.config.dsn is None:
            raise ValueError(
                "Either 'dsn', 'engine', or 'session' must be provided in EngineConfig"
            )

        dsn = self.config.dsn

        # Check if DSN already has async driver specified
        dsn_parts = dsn.split("://", 1)
        if len(dsn_parts) < 2:
            raise ValueError(f"Invalid DSN format: {dsn}")

        scheme = dsn_parts[0]
        # Check if scheme already has an async driver
        has_async_driver = "+" in scheme and any(
            driver in scheme for driver in ["asyncpg", "aiomysql", "aiosqlite"]
        )

        if not has_async_driver:
            # Auto-detect and add async driver based on database type
            # Handle schemes with sync drivers (e.g., mysql+pymysql -> mysql+aiomysql)
            base_scheme = scheme.split("+")[
                0
            ]  # Get base scheme (e.g., "mysql" from "mysql+pymysql")

            if base_scheme == "postgresql":
                # Replace any existing driver with asyncpg
                if "+" in scheme:
                    dsn = dsn.replace(scheme, "postgresql+asyncpg", 1)
                else:
                    dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif base_scheme in ("mysql", "mariadb"):
                # Replace any existing driver with aiomysql
                if "+" in scheme:
                    dsn = dsn.replace(scheme, f"{base_scheme}+aiomysql", 1)
                else:
                    dsn = dsn.replace("mysql://", "mysql+aiomysql://", 1).replace(
                        "mariadb://", "mariadb+aiomysql://", 1
                    )
            elif base_scheme == "sqlite":
                # Replace any existing driver with aiosqlite
                if "+" in scheme:
                    dsn = dsn.replace(scheme, "sqlite+aiosqlite", 1)
                else:
                    dsn = dsn.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
            else:
                raise ValueError(
                    f"DSN '{dsn}' does not specify an async driver. "
                    "Use format like 'postgresql+asyncpg://...' or 'mysql+aiomysql://...'"
                )

        # Refresh scheme after any driver normalization
        scheme = dsn.split("://", 1)[0]

        kwargs: dict[str, object] = {"echo": self.config.echo}
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

        if "postgresql+asyncpg" in scheme:
            normalized_dsn, server_settings = _extract_postgres_server_settings(dsn)
            if server_settings:
                dsn = normalized_dsn
                connect_args_obj = kwargs.setdefault("connect_args", {})
                if not isinstance(connect_args_obj, dict):
                    raise TypeError("connect_args must be a dict")
                server_settings_container = connect_args_obj.setdefault("server_settings", {})
                if not isinstance(server_settings_container, dict):
                    raise TypeError("server_settings connect arg must be a dict")
                server_settings_container.update(server_settings)

        return create_async_engine(dsn, **kwargs)

    @property
    def engine(self) -> AsyncEngine:
        """Get or create the async engine."""
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @asynccontextmanager
    async def connect(
        self, transaction: Optional[AsyncConnection] = None
    ) -> AsyncIterator[AsyncConnection]:
        """Get an async database connection.

        Args:
            transaction: If provided, use this transaction connection instead of creating a new one.
                        This allows operations to share a transaction.
                        If None and an active transaction exists, uses the active transaction.

        Yields:
            AsyncConnection instance
        """
        if transaction is not None:
            # Use the provided transaction connection
            yield transaction
        elif self._active_transaction is not None:
            # Use the active transaction connection automatically
            yield self._active_transaction
        elif self._session is not None:
            # Use the session's connection
            # SQLAlchemy async sessions have a connection() method that returns a coroutine
            if hasattr(self._session, "connection"):
                # Get connection from session (async)
                connection = await self._session.connection()
                yield connection
            else:
                # Fallback: use session's bind to create a connection
                async with self.engine.begin() as connection:
                    yield connection
        else:
            # Create a new connection with auto-commit (default behavior)
            async with self.engine.begin() as connection:
                yield connection

    async def begin_transaction(
        self,
        savepoint: bool = False,
        readonly: bool = False,
        isolation_level: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> AsyncConnection:
        """Begin a new transaction and return the connection.

        Args:
            savepoint: If True and a transaction is already active, create a savepoint instead.
            readonly: If True, set transaction to read-only mode.
            isolation_level: Optional isolation level (READ UNCOMMITTED, READ COMMITTED,
                           REPEATABLE READ, SERIALIZABLE).
            timeout: Optional transaction timeout in seconds.

        Returns:
            AsyncConnection that is part of a transaction (not auto-committed)

        Raises:
            RuntimeError: If savepoint=False and a transaction is already active.
            ValueError: If isolation level or readonly is requested but not supported by dialect.
        """
        if self._active_transaction is not None:
            if savepoint:
                # Create a savepoint instead of a new transaction
                savepoint_name = self._generate_savepoint_name()
                return await self.create_savepoint(self._active_transaction, savepoint_name)
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

        self._active_transaction = await self.engine.connect()
        self._savepoint_stack = []
        self._transaction_metadata = {
            "readonly": readonly,
            "isolation_level": isolation_level,
            "timeout": timeout,
        }

        # Set isolation level if specified
        if isolation_level:
            if not dialect_spec.supports_isolation_levels:
                await self._active_transaction.close()
                self._active_transaction = None
                raise ValueError(
                    f"Dialect '{dialect_name}' does not support isolation levels. "
                    "SQLite only supports SERIALIZABLE and READ UNCOMMITTED via PRAGMA."
                )
            await self._set_isolation_level(self._active_transaction, isolation_level)

        # Set read-only mode if specified
        if readonly:
            if not dialect_spec.supports_read_only_transactions:
                await self._active_transaction.close()
                self._active_transaction = None
                raise ValueError(
                    f"Dialect '{dialect_name}' does not support read-only transactions."
                )
            await self._set_readonly(self._active_transaction, True)

        # Set timeout if specified
        if timeout:
            await self._set_timeout(self._active_transaction, timeout, dialect_name)

        await self._active_transaction.begin()
        return self._active_transaction

    def _generate_savepoint_name(self) -> str:
        """Generate a unique savepoint name."""
        return f"sp_{len(self._savepoint_stack)}"

    async def _set_isolation_level(self, connection: AsyncConnection, isolation_level: str) -> None:
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
        await connection.execute(stmt)

    async def _set_readonly(self, connection: AsyncConnection, readonly: bool) -> None:
        """Set transaction to read-only mode."""
        mode = "READ ONLY" if readonly else "READ WRITE"
        stmt = text(f"SET TRANSACTION {mode}")
        await connection.execute(stmt)

    async def _set_timeout(
        self, connection: AsyncConnection, timeout: float, dialect_name: str
    ) -> None:
        """Set transaction timeout (database-specific)."""
        # PostgreSQL uses statement_timeout (in milliseconds)
        if "postgresql" in dialect_name:
            stmt = text(f"SET statement_timeout = {int(timeout * 1000)}")
            await connection.execute(stmt)
        # MySQL uses innodb_lock_wait_timeout (in seconds)
        elif "mysql" in dialect_name:
            stmt = text(f"SET innodb_lock_wait_timeout = {int(timeout)}")
            await connection.execute(stmt)
        # SQLite doesn't support transaction timeouts directly
        # Other databases may need specific implementations

    async def create_savepoint(self, connection: AsyncConnection, name: str) -> AsyncConnection:
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
        await connection.execute(stmt)
        self._savepoint_stack.append(name)
        return connection

    async def rollback_to_savepoint(self, connection: AsyncConnection, name: str) -> None:
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
        await connection.execute(stmt)

        # Remove all savepoints after the one we're rolling back to
        index = self._savepoint_stack.index(name)
        self._savepoint_stack = self._savepoint_stack[: index + 1]

    async def release_savepoint(self, connection: AsyncConnection, name: str) -> None:
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
        await connection.execute(stmt)
        self._savepoint_stack.remove(name)

    async def commit_transaction(self, connection: AsyncConnection) -> None:
        """Commit a transaction.

        Args:
            connection: The transaction connection to commit
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        try:
            await connection.commit()
        finally:
            # Always close connection, even if commit fails
            await connection.close()
            self._active_transaction = None
            self._savepoint_stack = []
            self._transaction_metadata = None

    async def rollback_transaction(self, connection: AsyncConnection) -> None:
        """Rollback a transaction.

        Args:
            connection: The transaction connection to rollback
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        try:
            await connection.rollback()
        finally:
            # Always close connection, even if rollback fails
            await connection.close()
            self._active_transaction = None
            self._savepoint_stack = []
            self._transaction_metadata = None

    @property
    def active_transaction(self) -> Optional[AsyncConnection]:
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

    async def close(self) -> None:
        """Close the engine and all connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
