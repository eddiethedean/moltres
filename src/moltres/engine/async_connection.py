"""Async SQLAlchemy connection helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Dict, Optional

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
        self._engine: Optional[AsyncEngine] = None

    def _create_engine(self) -> AsyncEngine:
        """Create an async SQLAlchemy engine.

        Args:
            config: Engine configuration

        Returns:
            AsyncEngine instance

        Raises:
            ValueError: If DSN doesn't support async (missing +asyncpg, +aiomysql, etc.)
        """
        dsn = self.config.dsn

        # Check if DSN already has async driver specified
        dsn_parts = dsn.split("://", 1)
        if len(dsn_parts) < 2:
            raise ValueError(f"Invalid DSN format: {dsn}")

        scheme = dsn_parts[0]
        if "+" not in scheme:
            # Auto-detect and add async driver based on database type
            if scheme == "postgresql":
                dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif scheme in ("mysql", "mariadb"):
                dsn = dsn.replace("mysql://", "mysql+aiomysql://", 1).replace(
                    "mariadb://", "mariadb+aiomysql://", 1
                )
            elif scheme == "sqlite":
                dsn = dsn.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
            else:
                raise ValueError(
                    f"DSN '{dsn}' does not specify an async driver. "
                    "Use format like 'postgresql+asyncpg://...' or 'mysql+aiomysql://...'"
                )

        kwargs: Dict[str, object] = {"echo": self.config.echo}
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
    async def connect(self) -> AsyncIterator[AsyncConnection]:
        """Get an async database connection.

        Yields:
            AsyncConnection instance
        """
        async with self.engine.begin() as connection:
            yield connection

    async def close(self) -> None:
        """Close the engine and all connections."""
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
