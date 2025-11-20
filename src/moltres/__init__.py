"""Public Moltres API."""

from __future__ import annotations

from .config import MoltresConfig, create_config
from .expressions import col, lit
from .table.schema import column
from .table.table import Database

__version__ = "0.2.0"

__all__ = ["connect", "Database", "MoltresConfig", "col", "lit", "column", "__version__"]


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
