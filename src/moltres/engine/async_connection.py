"""Async SQLAlchemy connection helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Optional

try:
    from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
    from sqlalchemy.ext.asyncio.engine import AsyncConnection
except ImportError as exc:
    raise ImportError(
        "Async support requires SQLAlchemy 2.0+ with async extensions. "
        "Install with: pip install 'SQLAlchemy>=2.0'"
    ) from exc

from ..config import EngineConfig


class AsyncConnectionManager:
    """Creates and caches async SQLAlchemy engines for Moltres sessions."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine: AsyncEngine | None = None
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
        # If an engine is provided in config, use it directly
        if self.config.engine is not None:
            if not isinstance(self.config.engine, AsyncEngine):
                raise TypeError("engine must be a SQLAlchemy AsyncEngine instance")
            return self.config.engine

        # Otherwise, create a new engine from DSN
        if self.config.dsn is None:
            raise ValueError("Either 'dsn' or 'engine' must be provided in EngineConfig")

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
        await connection.commit()
        await connection.close()
        self._active_transaction = None

    async def rollback_transaction(self, connection: AsyncConnection) -> None:
        """Rollback a transaction.

        Args:
            connection: The transaction connection to rollback
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        await connection.rollback()
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
