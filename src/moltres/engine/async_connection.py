"""Async SQLAlchemy connection helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import shlex
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

try:
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
    from sqlalchemy.ext.asyncio.engine import AsyncConnection
except ImportError as exc:
    raise ImportError(
        "Async support requires SQLAlchemy 2.0+ with async extensions. "
        "Install with: pip install 'SQLAlchemy>=2.0'"
    ) from exc

from ..config import EngineConfig


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
            raise ValueError("Either 'dsn', 'engine', or 'session' must be provided in EngineConfig")

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

        Yields:
            AsyncConnection instance
        """
        if transaction is not None:
            # Use the provided transaction connection
            yield transaction
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

    async def begin_transaction(self) -> AsyncConnection:
        """Begin a new transaction and return the connection.

        Returns:
            AsyncConnection that is part of a transaction (not auto-committed)
        """
        if self._active_transaction is not None:
            raise RuntimeError("Transaction already active. Nested transactions not yet supported.")
        self._active_transaction = await self.engine.connect()
        await self._active_transaction.begin()
        return self._active_transaction

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

    @property
    def active_transaction(self) -> Optional[AsyncConnection]:
        """Get the active transaction connection if one exists."""
        return self._active_transaction

    async def close(self) -> None:
        """Close the engine and all connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
