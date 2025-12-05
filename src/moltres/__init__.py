"""Public Moltres API.

This module provides the main entry points for using Moltres:

- :func:`connect` - Create a synchronous database connection
- :func:`async_connect` - Create an asynchronous database connection
- :class:`Database` - Main database interface for querying and table operations
- :class:`AsyncDatabase` - Async version of :class:`Database`
- :func:`col` - Create column expressions
- :func:`lit` - Create literal values
- :func:`column` - Define table columns for schema creation

Example:
    Basic usage::

        from moltres import connect, col
        from moltres.expressions import functions as F

        db = connect("sqlite:///example.db")
        df = db.table("users").select().where(col("age") > 25)
        results = df.collect()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import duckdb_engine early to register the dialect with SQLAlchemy
try:
    import duckdb_engine  # noqa: F401
except ImportError:
    pass

from .config import EngineOptionValue, MoltresConfig, create_config
from .expressions import col, lit
from .table.schema import column
from .table.table import Database

# Optional pandas interface - only import if available
try:
    from .dataframe.interfaces.pandas_dataframe import PandasDataFrame
except ImportError:
    PandasDataFrame = None  # type: ignore

# Optional polars interface - only import if available
try:
    from .dataframe.interfaces.polars_dataframe import PolarsDataFrame
except ImportError:
    PolarsDataFrame = None  # type: ignore

# Optional async polars interface - only import if available
try:
    from .dataframe.interfaces.async_polars_dataframe import AsyncPolarsDataFrame
except ImportError:
    AsyncPolarsDataFrame = None  # type: ignore

# Optional async pandas interface - only import if available
try:
    from .dataframe.interfaces.async_pandas_dataframe import AsyncPandasDataFrame
except ImportError:
    AsyncPandasDataFrame = None  # type: ignore


def _validate_connection_string(dsn: str, is_async: bool = False) -> None:
    """Validate connection string format and provide helpful error messages.

    Args:
        dsn: Connection string to validate
        is_async: Whether this is for async connection

    Raises:
        DatabaseConnectionError: If connection string is invalid
    """
    from .utils.exceptions import DatabaseConnectionError

    if not dsn or not isinstance(dsn, str):
        raise DatabaseConnectionError(
            f"Connection string must be a non-empty string, got: {type(dsn).__name__}"
        )

    dsn_lower = dsn.lower()
    # Check for common connection string patterns
    if dsn_lower.startswith("sqlite"):
        if is_async and "+aiosqlite" not in dsn_lower:
            raise DatabaseConnectionError(
                f"Async SQLite connection requires 'sqlite+aiosqlite://' prefix. "
                f"Got: {dsn[:50]}...",
                suggestion="Use 'sqlite+aiosqlite:///path/to/db.db' for async SQLite connections.",
            )
    elif dsn_lower.startswith("postgresql"):
        if is_async and "+asyncpg" not in dsn_lower:
            raise DatabaseConnectionError(
                f"Async PostgreSQL connection requires 'postgresql+asyncpg://' prefix. "
                f"Got: {dsn[:50]}...",
                suggestion="Use 'postgresql+asyncpg://user:pass@host:port/dbname' for async PostgreSQL connections.",
            )
    elif dsn_lower.startswith("mysql"):
        if is_async and "+aiomysql" not in dsn_lower:
            raise DatabaseConnectionError(
                f"Async MySQL connection requires 'mysql+aiomysql://' prefix. Got: {dsn[:50]}...",
                suggestion="Use 'mysql+aiomysql://user:pass@host:port/dbname' for async MySQL connections.",
            )

    # Check for basic URL structure
    if "://" not in dsn:
        raise DatabaseConnectionError(
            f"Connection string must include '://' separator. Got: {dsn[:50]}...",
            suggestion="Connection strings should follow the format: 'dialect://user:pass@host:port/dbname'",
        )


__version__ = "0.23.1"

__all__ = [
    "AsyncDatabase",
    "AsyncPandasDataFrame",
    "AsyncPolarsDataFrame",
    "Database",
    "MoltresConfig",
    "PandasDataFrame",
    "PolarsDataFrame",
    "__version__",
    "async_connect",
    "col",
    "column",
    "connect",
    "lit",
]

# Optional FastAPI integration - only import if available
try:
    from .integrations import fastapi as fastapi_integration

    __all__.append("fastapi_integration")
except ImportError:
    fastapi_integration = None  # type: ignore[assignment]

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
    session: object | None = None,
    **options: EngineOptionValue,
) -> Database:
    """Connect to a SQL database and return a :class:`Database` handle.

    Configuration can be provided via arguments or environment variables:
    - MOLTRES_DSN: :class:`Database` connection string (if dsn is None)
    - MOLTRES_ECHO: Enable SQLAlchemy echo mode (true/false)
    - MOLTRES_FETCH_FORMAT: "records", "pandas", or "polars"
    - MOLTRES_DIALECT: Override SQL dialect detection
    - MOLTRES_POOL_SIZE: Connection pool size
    - MOLTRES_MAX_OVERFLOW: Maximum pool overflow connections
    - MOLTRES_POOL_TIMEOUT: Pool timeout in seconds
    - MOLTRES_POOL_RECYCLE: Connection recycle time in seconds
    - MOLTRES_POOL_PRE_PING: Enable connection health checks (true/false)

    Args:
        dsn: :class:`Database` connection string. Examples:
            - SQLite: "sqlite:///path/to/database.db"
            - PostgreSQL: "postgresql://user:pass@host:port/dbname"
            - MySQL: "mysql://user:pass@host:port/dbname"
            If None, will use MOLTRES_DSN environment variable.
            Cannot be provided if engine or session is provided.
        engine: SQLAlchemy Engine instance to use. If provided, dsn is ignored.
                This gives users more flexibility to configure the engine themselves.
                Pool configuration options (pool_size, max_overflow, etc.) are ignored
                when using an existing engine.
                Cannot be provided if session is provided.
        session: SQLAlchemy Session or SQLModel Session instance to use. If provided,
                dsn and engine are ignored. The session's bind (engine) will be used.
                This allows using Moltres with FastAPI's dependency-injected sessions.
                Cannot be provided if dsn or engine is provided.
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
        :class:`Database`: Database instance for querying and table operations

    Raises:
        ValueError: If neither dsn, engine, nor session is provided and MOLTRES_DSN is not set
        ValueError: If multiple of dsn, engine, and session are provided
        TypeError: If session is not a SQLAlchemy Session or SQLModel Session instance

    Example:
        >>> # Using connection string with context manager (recommended)
        >>> with connect("sqlite:///:memory:") as db:
        ...     from moltres.table.schema import column
        ...     _ = db.create_table("users", [column("id", "INTEGER"), column("active", "BOOLEAN")]).collect()  # doctest: +ELLIPSIS
        ...     from moltres.io.records import Records
        ...     _ = Records(_data=[{"id": 1, "active": True}], _database=db).insert_into("users")
        ...     df = db.table("users").select().where(col("active") == True)
        ...     results = df.collect()
        ...     # db.close() called automatically on exit

        >>> # Using connection string (manual close)
        >>> db = connect("sqlite:///:memory:")
        >>> from moltres.table.schema import column
        >>> _ = db.create_table("users", [column("id", "INTEGER"), column("active", "BOOLEAN")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import Records
        >>> _ = Records(_data=[{"id": 1, "active": True}], _database=db).insert_into("users")
        >>> df = db.table("users").select().where(col("active") == True)
        >>> results = df.collect()
        >>> db.close()

        >>> # Using SQLAlchemy Engine
        >>> from sqlalchemy import create_engine
        >>> engine = create_engine("sqlite:///:memory:")
        >>> db2 = connect(engine=engine)
        >>> _ = db2.create_table("test", [column("x", "INTEGER")]).collect()  # doctest: +ELLIPSIS
        >>> db2.close()

        >>> # Using SQLAlchemy Session (e.g., from FastAPI dependency injection)
        >>> from sqlalchemy.orm import Session, sessionmaker
        >>> SessionLocal = sessionmaker(bind=create_engine("sqlite:///:memory:"))
        >>> with SessionLocal() as session:
        ...     db3 = connect(session=session)
        ...     _ = db3.create_table("test2", [column("x", "INTEGER")]).collect()  # doctest: +ELLIPSIS
    """
    from sqlalchemy.engine import Engine as SQLAlchemyEngine

    # Validate connection string format if provided
    if dsn is not None:
        _validate_connection_string(dsn, is_async=False)

    # Check if session is provided
    session_obj: object | None = None
    if session is not None:
        # Validate it's a Session-like object
        if not (
            hasattr(session, "get_bind")
            or hasattr(session, "bind")
            or hasattr(session, "connection")
        ):
            raise TypeError(
                "session must be a SQLAlchemy Session or SQLModel Session instance. "
                f"Got: {type(session).__name__}"
            )
        session_obj = session
    elif "session" in options:
        session_from_options = options.pop("session")
        if not (
            hasattr(session_from_options, "get_bind")
            or hasattr(session_from_options, "bind")
            or hasattr(session_from_options, "connection")
        ):
            raise TypeError(
                "session must be a SQLAlchemy Session or SQLModel Session instance. "
                f"Got: {type(session_from_options).__name__}"
            )
        session_obj = session_from_options

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

    config: MoltresConfig = create_config(
        dsn=dsn, engine=engine_obj, session=session_obj, **options
    )
    return Database(config=config)


def async_connect(
    dsn: str | None = None,
    engine: object | None = None,
    session: object | None = None,
    **options: EngineOptionValue,
) -> AsyncDatabase:
    """Connect to a SQL database asynchronously and return an :class:`AsyncDatabase` handle.

    This function requires async dependencies. Install with:
    - `pip install moltres[async]` - for core async support (aiofiles)
    - `pip install moltres[async-postgresql]` - for PostgreSQL async support (includes async + asyncpg)
    - `pip install moltres[async-mysql]` - for MySQL async support (includes async + aiomysql)
    - `pip install moltres[async-sqlite]` - for SQLite async support (includes async + aiosqlite)

    Configuration can be provided via arguments or environment variables:
    - MOLTRES_DSN: :class:`Database` connection string (if dsn is None)
    - MOLTRES_ECHO: Enable SQLAlchemy echo mode (true/false)
    - MOLTRES_FETCH_FORMAT: "records", "pandas", or "polars"
    - MOLTRES_DIALECT: Override SQL dialect detection
    - MOLTRES_POOL_SIZE: Connection pool size
    - MOLTRES_MAX_OVERFLOW: Maximum pool overflow connections
    - MOLTRES_POOL_TIMEOUT: Pool timeout in seconds
    - MOLTRES_POOL_RECYCLE: Connection recycle time in seconds
    - MOLTRES_POOL_PRE_PING: Enable connection health checks (true/false)

    Args:
        dsn: :class:`Database` connection string. Examples:
            - SQLite: "sqlite+aiosqlite:///path/to/database.db"
            - PostgreSQL: "postgresql+asyncpg://user:pass@host:port/dbname"
            - MySQL: "mysql+aiomysql://user:pass@host:port/dbname"
            If None, will use MOLTRES_DSN environment variable.
            Note: DSN should include async driver (e.g., +asyncpg, +aiomysql, +aiosqlite)
            Cannot be provided if engine or session is provided.
        engine: SQLAlchemy async Engine instance to use. If provided, dsn is ignored.
                This gives users more flexibility to configure the engine themselves.
                Pool configuration options (pool_size, max_overflow, etc.) are ignored
                when using an existing engine.
                Cannot be provided if session is provided.
        session: SQLAlchemy AsyncSession or SQLModel AsyncSession instance to use. If provided,
                dsn and engine are ignored. The session's bind (async engine) will be used.
                This allows using Moltres with FastAPI's dependency-injected async sessions.
                Cannot be provided if dsn or engine is provided.
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
        :class:`AsyncDatabase`: :class:`AsyncDatabase` instance for async querying and table operations

    Raises:
        ImportError: If async dependencies are not installed
        ValueError: If neither dsn, engine, nor session is provided and MOLTRES_DSN is not set
        ValueError: If multiple of dsn, engine, and session are provided
        TypeError: If session is not a SQLAlchemy AsyncSession or SQLModel AsyncSession instance

    Example:
        >>> import asyncio
        >>> async def example():
        ...     # Using connection string with async context manager (recommended)
        ...     async with async_connect("sqlite+aiosqlite:///:memory:") as db:
        ...         from moltres.table.schema import column
        ...         await db.create_table("users", [column("id", "INTEGER")]).collect()
        ...         from moltres.io.records import :class:`AsyncRecords`
        ...         records = :class:`AsyncRecords`(_data=[{"id": 1}], _database=db)
        ...         await records.insert_into("users")
        ...         table_handle = await db.table("users")
        ...         df = table_handle.select()
        ...         results = await df.collect()
        ...         assert len(results) == 1
        ...         assert results[0]["id"] == 1
        ...         # await db.close() called automatically on exit
        ...
        ...     # Using connection string (manual close)
        ...     db = async_connect("sqlite+aiosqlite:///:memory:")
        ...     await db.create_table("users", [column("id", "INTEGER")]).collect()
        ...     await db.close()
        ...     # Note: async examples require running in async context
        ...     # asyncio.run(example())  # doctest: +SKIP
    """
    try:
        from .table.async_table import AsyncDatabase
    except ImportError as exc:
        raise ImportError(
            "Async support requires async dependencies. Install with: pip install moltres[async]"
        ) from exc

    from sqlalchemy.ext.asyncio import AsyncEngine as SQLAlchemyAsyncEngine

    # Validate connection string format if provided
    if dsn is not None:
        _validate_connection_string(dsn, is_async=True)

    # Check if session is provided
    session_obj: object | None = None
    if session is not None:
        # Validate it's an AsyncSession-like object
        if not (
            hasattr(session, "get_bind")
            or hasattr(session, "bind")
            or hasattr(session, "connection")
        ):
            raise TypeError(
                "session must be a SQLAlchemy AsyncSession or SQLModel AsyncSession instance. "
                f"Got: {type(session).__name__}"
            )
        session_obj = session
    elif "session" in options:
        session_from_options = options.pop("session")
        if not (
            hasattr(session_from_options, "get_bind")
            or hasattr(session_from_options, "bind")
            or hasattr(session_from_options, "connection")
        ):
            raise TypeError(
                "session must be a SQLAlchemy AsyncSession or SQLModel AsyncSession instance. "
                f"Got: {type(session_from_options).__name__}"
            )
        session_obj = session_from_options

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

    config: MoltresConfig = create_config(
        dsn=dsn, engine=engine_obj, session=session_obj, **options
    )
    return AsyncDatabase(config=config)
