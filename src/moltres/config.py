"""Runtime configuration objects for Moltres."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine
else:
    AsyncEngine = None  # type: ignore[assignment, misc]

FetchFormat = Literal["pandas", "polars", "records"]


@dataclass
class EngineConfig:
    """Connection + execution options for SQLAlchemy engines."""

    dsn: str | None = None
    engine: Engine | "AsyncEngine" | None = None
    echo: bool = False
    fetch_format: FetchFormat = "records"
    dialect: str | None = None
    pool_size: int | None = None
    max_overflow: int | None = None
    pool_timeout: int | None = None
    pool_recycle: int | None = None
    pool_pre_ping: bool = False
    query_timeout: float | None = None  # Query execution timeout in seconds
    future: bool = True

    def __post_init__(self) -> None:
        """Validate that either dsn or engine is provided, but not both."""
        if self.dsn is None and self.engine is None:
            raise ValueError("Either 'dsn' or 'engine' must be provided")
        if self.dsn is not None and self.engine is not None:
            raise ValueError(
                "Cannot provide both 'dsn' and 'engine'. Provide either a connection string or an Engine instance."
            )


@dataclass
class MoltresConfig:
    """Container for all runtime configuration knobs."""

    engine: EngineConfig
    default_schema: str | None = None
    include_metadata: bool = False
    options: dict[str, object] = field(default_factory=dict)


def _load_env_config() -> dict[str, object]:
    """Load configuration from environment variables.

    Returns:
        Dictionary of configuration values from environment
    """
    config: dict[str, object] = {}

    # DSN is handled separately in create_config
    if "MOLTRES_ECHO" in os.environ:
        config["echo"] = os.environ["MOLTRES_ECHO"].lower() in ("true", "1", "yes", "on")

    if "MOLTRES_FETCH_FORMAT" in os.environ:
        config["fetch_format"] = os.environ["MOLTRES_FETCH_FORMAT"]

    if "MOLTRES_DIALECT" in os.environ:
        config["dialect"] = os.environ["MOLTRES_DIALECT"]

    if "MOLTRES_POOL_SIZE" in os.environ:
        config["pool_size"] = int(os.environ["MOLTRES_POOL_SIZE"])

    if "MOLTRES_MAX_OVERFLOW" in os.environ:
        config["max_overflow"] = int(os.environ["MOLTRES_MAX_OVERFLOW"])

    if "MOLTRES_POOL_TIMEOUT" in os.environ:
        config["pool_timeout"] = int(os.environ["MOLTRES_POOL_TIMEOUT"])

    if "MOLTRES_POOL_RECYCLE" in os.environ:
        config["pool_recycle"] = int(os.environ["MOLTRES_POOL_RECYCLE"])

    if "MOLTRES_POOL_PRE_PING" in os.environ:
        config["pool_pre_ping"] = os.environ["MOLTRES_POOL_PRE_PING"].lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

    if "MOLTRES_QUERY_TIMEOUT" in os.environ:
        config["query_timeout"] = float(os.environ["MOLTRES_QUERY_TIMEOUT"])

    return config


def create_config(
    dsn: str | None = None,
    engine: Engine | "AsyncEngine | None" = None,
    **kwargs: object,
) -> MoltresConfig:
    """Convenience helper used by ``moltres.connect``.

    Supports environment variables for configuration:
    - MOLTRES_DSN: Database connection string
    - MOLTRES_ECHO: Enable SQLAlchemy echo mode (true/false)
    - MOLTRES_FETCH_FORMAT: "records", "pandas", or "polars"
    - MOLTRES_DIALECT: Override SQL dialect detection
    - MOLTRES_POOL_SIZE: Connection pool size
    - MOLTRES_MAX_OVERFLOW: Maximum pool overflow connections
    - MOLTRES_POOL_TIMEOUT: Pool timeout in seconds
    - MOLTRES_POOL_RECYCLE: Connection recycle time in seconds
    - MOLTRES_POOL_PRE_PING: Enable connection health checks (true/false)
    - MOLTRES_QUERY_TIMEOUT: Query execution timeout in seconds

    Args:
        dsn: Database connection string (e.g., "sqlite:///example.db").
             If None, will try MOLTRES_DSN environment variable.
             Cannot be provided if engine is provided.
        engine: SQLAlchemy Engine instance to use. If provided, dsn is ignored.
                This gives users more flexibility to configure the engine themselves.
        **kwargs: Additional configuration options. Valid keys include:
            - echo: Enable SQLAlchemy echo mode
            - fetch_format: "records", "pandas", or "polars"
            - dialect: Override SQL dialect detection
            - pool_size: Connection pool size (ignored if engine is provided)
            - max_overflow: Maximum pool overflow connections (ignored if engine is provided)
            - pool_timeout: Pool timeout in seconds (ignored if engine is provided)
            - pool_recycle: Connection recycle time in seconds (ignored if engine is provided)
            - pool_pre_ping: Enable connection health checks (ignored if engine is provided)
            - query_timeout: Query execution timeout in seconds
            - future: Use SQLAlchemy 2.0 style (default: True)
            - Other options are stored in config.options

    Returns:
        MoltresConfig instance with parsed configuration

    Raises:
        ValueError: If neither dsn nor engine is provided and MOLTRES_DSN is not set
        ValueError: If both dsn and engine are provided
    """
    # Check if engine is provided in kwargs (for backward compatibility)
    if engine is None and "engine" in kwargs:
        engine_obj = kwargs.pop("engine")
        if not isinstance(engine_obj, Engine):
            raise TypeError("engine must be a SQLAlchemy Engine instance")
        engine = engine_obj

    # Get DSN from argument or environment variable (only if engine is not provided)
    if engine is None:
        if dsn is None:
            dsn = os.environ.get("MOLTRES_DSN")
            if dsn is None:
                raise ValueError(
                    "Either 'dsn' or 'engine' must be provided as argument, or MOLTRES_DSN environment variable must be set"
                )
        # Normalize SQLite paths: convert backslashes to forward slashes for URLs
        # SQLite URLs always use forward slashes, even on Windows
        if dsn and (dsn.startswith("sqlite:///") or dsn.startswith("sqlite+aiosqlite:///")):
            # Replace backslashes with forward slashes in the path part
            dsn = dsn.replace("\\", "/")
    else:
        # If engine is provided, ignore dsn
        if dsn is not None:
            raise ValueError(
                "Cannot provide both 'dsn' and 'engine'. Provide either a connection string or an Engine instance."
            )
        dsn = None

    # Load configuration from environment variables if not provided in kwargs
    env_config = _load_env_config()

    # Merge: kwargs override env vars, env vars override defaults
    merged_kwargs = {**env_config, **kwargs}

    # If an engine object is provided, infer dialect name unless explicitly overridden
    inferred_engine_dialect: str | None = None
    if engine is not None:
        inferred_engine_dialect = getattr(getattr(engine, "dialect", None), "name", None)
        if inferred_engine_dialect and "+" in inferred_engine_dialect:
            # Normalize driver variants similar to DSN parsing (e.g., "mysql+aiomysql")
            inferred_engine_dialect = inferred_engine_dialect.split("+", 1)[0]

    if "dialect" not in merged_kwargs and inferred_engine_dialect:
        merged_kwargs["dialect"] = inferred_engine_dialect

    engine_kwargs: dict[str, object] = {
        k: merged_kwargs.pop(k)
        for k in list(merged_kwargs)
        if k in EngineConfig.__dataclass_fields__
    }
    # Construct EngineConfig with validated kwargs
    # Using **kwargs with type checking would be ideal, but dataclass doesn't support it directly
    # This approach extracts known fields and passes them safely
    engine_config = EngineConfig(dsn=dsn, engine=engine, **engine_kwargs)  # type: ignore[arg-type]
    return MoltresConfig(engine=engine_config, options=merged_kwargs)


DEFAULT_CONFIG = create_config("sqlite:///:memory:")
