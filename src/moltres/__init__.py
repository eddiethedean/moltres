"""Public Moltres API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import MoltresConfig, create_config
from .expressions import col, lit
from .table.schema import column
from .table.table import Database

__version__ = "0.14.0"

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
        AsyncDatabase = None


def connect(
    dsn: str | None = None,
    engine: object | None = None,
    **options: object,
) -> Database:
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
            Cannot be provided if engine is provided.
        engine: SQLAlchemy Engine instance to use. If provided, dsn is ignored.
                This gives users more flexibility to configure the engine themselves.
                Pool configuration options (pool_size, max_overflow, etc.) are ignored
                when using an existing engine.
        **options: Optional configuration parameters (can also be set via environment variables):
            - echo: Enable SQLAlchemy echo mode for debugging (default: False)
            - fetch_format: Result format - "records", "pandas", or "polars" (default: "records")
            - dialect: Override SQL dialect detection (e.g., "postgresql", "mysql")
            - pool_size: Connection pool size (default: None, uses SQLAlchemy default)
                         Ignored if engine is provided.
            - max_overflow: Maximum pool overflow connections (default: None)
                            Ignored if engine is provided.
            - pool_timeout: Pool timeout in seconds (default: None)
                           Ignored if engine is provided.
            - pool_recycle: Connection recycle time in seconds (default: None)
                           Ignored if engine is provided.
            - pool_pre_ping: Enable connection health checks (default: False)
                            Ignored if engine is provided.
            - future: Use SQLAlchemy 2.0 style (default: True)

    Returns:
        Database instance for querying and table operations

    Raises:
        ValueError: If neither dsn nor engine is provided and MOLTRES_DSN is not set
        ValueError: If both dsn and engine are provided

    Example:
        >>> # Using connection string
        >>> db = connect("sqlite:///:memory:")  # doctest: +SKIP
        >>> # Create table first
        >>> from moltres.table.schema import ColumnDef
        >>> table = db.create_table("users", [ColumnDef("id", "INTEGER"), ColumnDef("active", "BOOLEAN")])  # doctest: +SKIP
        >>> # Insert data using Records
        >>> from moltres.io.records import Records
        >>> records = Records(_data=[{"id": 1, "active": True}], _database=db)  # doctest: +SKIP
        >>> records.insert_into("users")  # doctest: +SKIP
        >>> df = db.table("users").select().where(col("active") == True)  # doctest: +SKIP
        >>> results = df.collect()  # doctest: +SKIP
        >>> len(results)  # doctest: +SKIP
        1

        >>> # Using SQLAlchemy Engine
        >>> from sqlalchemy import create_engine  # doctest: +SKIP
        >>> engine = create_engine("sqlite:///:memory:", echo=True)  # doctest: +SKIP
        >>> db = connect(engine=engine)  # doctest: +SKIP
    """
    from sqlalchemy.engine import Engine as SQLAlchemyEngine

    # Check if engine is provided in kwargs (for backward compatibility)
    engine_obj: SQLAlchemyEngine | None = None
    if engine is not None:
        if not isinstance(engine, SQLAlchemyEngine):
            raise TypeError("engine must be a SQLAlchemy Engine instance")
        engine_obj = engine
    elif "engine" in options:
        engine_from_options = options.pop("engine")
        if not isinstance(engine_from_options, SQLAlchemyEngine):
            raise TypeError("engine must be a SQLAlchemy Engine instance")
        engine_obj = engine_from_options

    config: MoltresConfig = create_config(dsn=dsn, engine=engine_obj, **options)
    return Database(config=config)


def async_connect(
    dsn: str | None = None, engine: object | None = None, **options: object
) -> AsyncDatabase:
    """Connect to a SQL database asynchronously and return an ``AsyncDatabase`` handle.

    This function requires async dependencies. Install with:
    - `pip install moltres[async]` - for core async support (aiofiles)
    - `pip install moltres[async-postgresql]` - for PostgreSQL async support (includes async + asyncpg)
    - `pip install moltres[async-mysql]` - for MySQL async support (includes async + aiomysql)
    - `pip install moltres[async-sqlite]` - for SQLite async support (includes async + aiosqlite)

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
            Cannot be provided if engine is provided.
        engine: SQLAlchemy async Engine instance to use. If provided, dsn is ignored.
                This gives users more flexibility to configure the engine themselves.
                Pool configuration options (pool_size, max_overflow, etc.) are ignored
                when using an existing engine.
        **options: Optional configuration parameters (can also be set via environment variables):
            - echo: Enable SQLAlchemy echo mode for debugging (default: False)
            - fetch_format: Result format - "records", "pandas", or "polars" (default: "records")
            - dialect: Override SQL dialect detection (e.g., "postgresql", "mysql")
            - pool_size: Connection pool size (default: None, uses SQLAlchemy default)
                         Ignored if engine is provided.
            - max_overflow: Maximum pool overflow connections (default: None)
                            Ignored if engine is provided.
            - pool_timeout: Pool timeout in seconds (default: None)
                           Ignored if engine is provided.
            - pool_recycle: Connection recycle time in seconds (default: None)
                           Ignored if engine is provided.
            - pool_pre_ping: Enable connection health checks (default: False)
                            Ignored if engine is provided.

    Returns:
        AsyncDatabase instance for async querying and table operations

    Raises:
        ImportError: If async dependencies are not installed
        ValueError: If neither dsn nor engine is provided and MOLTRES_DSN is not set
        ValueError: If both dsn and engine are provided

    Example:
        >>> import asyncio  # doctest: +SKIP
        >>> from moltres import async_connect  # doctest: +SKIP
        >>>
        >>> async def main():  # doctest: +SKIP
        ...     # Using connection string
        ...     db = async_connect("sqlite+aiosqlite:///:memory:")  # doctest: +SKIP
        ...     from moltres.table.schema import ColumnDef  # doctest: +SKIP
        ...     table = await db.create_table("users", [ColumnDef("id", "INTEGER")])  # doctest: +SKIP
        ...     from moltres.io.records import AsyncRecords  # doctest: +SKIP
        ...     records = AsyncRecords(_data=[{"id": 1}], _database=db)  # doctest: +SKIP
        ...     await records.insert_into("users")  # doctest: +SKIP
        ...     table_handle = await db.table("users")  # doctest: +SKIP
        ...     df = table_handle.select()  # doctest: +SKIP
        ...     results = await df.collect()  # doctest: +SKIP
        ...     await db.close()  # doctest: +SKIP
        >>>
        >>>     # Using SQLAlchemy async Engine
        >>>     from sqlalchemy.ext.asyncio import create_async_engine  # doctest: +SKIP
        >>>     engine = create_async_engine("sqlite+aiosqlite:///:memory:")  # doctest: +SKIP
        >>>     db = async_connect(engine=engine)  # doctest: +SKIP
        >>>
        >>> # asyncio.run(main())  # doctest: +SKIP
    """
    try:
        from .table.async_table import AsyncDatabase
    except ImportError as exc:
        raise ImportError(
            "Async support requires async dependencies. Install with: pip install moltres[async]"
        ) from exc

    from sqlalchemy.ext.asyncio import AsyncEngine as SQLAlchemyAsyncEngine

    # Check if engine is provided in kwargs (for backward compatibility)
    engine_obj: SQLAlchemyAsyncEngine | None = None
    if engine is not None:
        if not isinstance(engine, SQLAlchemyAsyncEngine):
            raise TypeError("engine must be a SQLAlchemy AsyncEngine instance")
        engine_obj = engine
    elif "engine" in options:
        engine_from_options = options.pop("engine")
        if not isinstance(engine_from_options, SQLAlchemyAsyncEngine):
            raise TypeError("engine must be a SQLAlchemy AsyncEngine instance")
        engine_obj = engine_from_options

    config: MoltresConfig = create_config(dsn=dsn, engine=engine_obj, **options)
    return AsyncDatabase(config=config)
