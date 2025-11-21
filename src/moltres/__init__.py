"""Public Moltres API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import MoltresConfig, create_config
from .expressions import col, lit
from .table.schema import column
from .table.table import Database

__version__ = "0.3.0"

__all__ = [
    "AsyncDatabase",
    "Database",
    "MoltresConfig",
    "__version__",
    "async_connect",
    "col",
    "column",
    "connect",
    "lit",
]

# Async imports - only available if async dependencies are installed
if TYPE_CHECKING:
    from .table.async_table import AsyncDatabase
else:
    try:
        from .table.async_table import AsyncDatabase
    except ImportError:
        AsyncDatabase = None  # type: ignore[assignment]


def connect(dsn: str | None = None, **options: object) -> Database:
    """Connect to a SQL database and return a ``Database`` handle.

    Configuration can be provided via arguments or environment variables:
    - MOLTRES_DSN: Database connection string (if dsn is None)
    - MOLTRES_ECHO: Enable SQLAlchemy echo mode (true/false)
    - MOLTRES_FETCH_FORMAT: "records", "pandas", or "polars"
    - MOLTRES_DIALECT: Override SQL dialect detection
    - MOLTRES_POOL_SIZE: Connection pool size
    - MOLTRES_MAX_OVERFLOW: Maximum pool overflow connections
    - MOLTRES_POOL_TIMEOUT: Pool timeout in seconds
    - MOLTRES_POOL_RECYCLE: Connection recycle time in seconds
    - MOLTRES_POOL_PRE_PING: Enable connection health checks (true/false)

    Args:
        dsn: Database connection string. Examples:
            - SQLite: "sqlite:///path/to/database.db"
            - PostgreSQL: "postgresql://user:pass@host:port/dbname"
            - MySQL: "mysql://user:pass@host:port/dbname"
            If None, will use MOLTRES_DSN environment variable.
        **options: Optional configuration parameters (can also be set via environment variables):
            - echo: Enable SQLAlchemy echo mode for debugging (default: False)
            - fetch_format: Result format - "records", "pandas", or "polars" (default: "records")
            - dialect: Override SQL dialect detection (e.g., "postgresql", "mysql")
            - pool_size: Connection pool size (default: None, uses SQLAlchemy default)
            - max_overflow: Maximum pool overflow connections (default: None)
            - pool_timeout: Pool timeout in seconds (default: None)
            - pool_recycle: Connection recycle time in seconds (default: None)
            - pool_pre_ping: Enable connection health checks (default: False)
            - future: Use SQLAlchemy 2.0 style (default: True)

    Returns:
        Database instance for querying and table operations

    Example:
        >>> db = connect("sqlite:///example.db")
        >>> df = db.table("users").select().where(col("active") == True)
        >>> results = df.collect()
    """
    config: MoltresConfig = create_config(dsn, **options)
    return Database(config=config)


def async_connect(dsn: str | None = None, **options: object) -> AsyncDatabase:
    """Connect to a SQL database asynchronously and return an ``AsyncDatabase`` handle.

    This function requires async dependencies. Install with:
    - `pip install moltres[async]` - for core async support (aiofiles)
    - `pip install moltres[async,async-postgresql]` - for PostgreSQL async support
    - `pip install moltres[async,async-mysql]` - for MySQL async support
    - `pip install moltres[async,async-sqlite]` - for SQLite async support

    Configuration can be provided via arguments or environment variables:
    - MOLTRES_DSN: Database connection string (if dsn is None)
    - MOLTRES_ECHO: Enable SQLAlchemy echo mode (true/false)
    - MOLTRES_FETCH_FORMAT: "records", "pandas", or "polars"
    - MOLTRES_DIALECT: Override SQL dialect detection
    - MOLTRES_POOL_SIZE: Connection pool size
    - MOLTRES_MAX_OVERFLOW: Maximum pool overflow connections
    - MOLTRES_POOL_TIMEOUT: Pool timeout in seconds
    - MOLTRES_POOL_RECYCLE: Connection recycle time in seconds
    - MOLTRES_POOL_PRE_PING: Enable connection health checks (true/false)

    Args:
        dsn: Database connection string. Examples:
            - SQLite: "sqlite+aiosqlite:///path/to/database.db"
            - PostgreSQL: "postgresql+asyncpg://user:pass@host:port/dbname"
            - MySQL: "mysql+aiomysql://user:pass@host:port/dbname"
            If None, will use MOLTRES_DSN environment variable.
            Note: DSN should include async driver (e.g., +asyncpg, +aiomysql, +aiosqlite)
        **options: Optional configuration parameters (can also be set via environment variables):
            - echo: Enable SQLAlchemy echo mode for debugging (default: False)
            - fetch_format: Result format - "records", "pandas", or "polars" (default: "records")
            - dialect: Override SQL dialect detection (e.g., "postgresql", "mysql")
            - pool_size: Connection pool size (default: None, uses SQLAlchemy default)
            - max_overflow: Maximum pool overflow connections (default: None)
            - pool_timeout: Pool timeout in seconds (default: None)
            - pool_recycle: Connection recycle time in seconds (default: None)
            - pool_pre_ping: Enable connection health checks (default: False)

    Returns:
        AsyncDatabase instance for async querying and table operations

    Raises:
        ImportError: If async dependencies are not installed

    Example:
        >>> import asyncio
        >>> from moltres import async_connect
        >>>
        >>> async def main():
        ...     db = async_connect("postgresql+asyncpg://user:pass@localhost/db")
        ...     df = await db.read.table("users")
        ...     results = await df.collect()
        ...     print(results)
        >>>
        >>> asyncio.run(main())
    """
    try:
        from .table.async_table import AsyncDatabase
    except ImportError as exc:
        raise ImportError(
            "Async support requires async dependencies. Install with: pip install moltres[async]"
        ) from exc

    config: MoltresConfig = create_config(dsn, **options)
    return AsyncDatabase(config=config)
