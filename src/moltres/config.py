"""Runtime configuration objects for Moltres."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

FetchFormat = Literal["pandas", "polars", "records"]


@dataclass
class EngineConfig:
    """Connection + execution options for SQLAlchemy engines."""

    dsn: str
    echo: bool = False
    fetch_format: FetchFormat = "records"
    dialect: Optional[str] = None
    pool_size: Optional[int] = None
    max_overflow: Optional[int] = None
    pool_timeout: Optional[int] = None
    pool_recycle: Optional[int] = None
    pool_pre_ping: bool = False
    future: bool = True


@dataclass
class MoltresConfig:
    """Container for all runtime configuration knobs."""

    engine: EngineConfig
    default_schema: Optional[str] = None
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

    return config


def create_config(dsn: str | None = None, **kwargs: object) -> MoltresConfig:
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

    Args:
        dsn: Database connection string (e.g., "sqlite:///example.db").
             If None, will try MOLTRES_DSN environment variable.
        **kwargs: Additional configuration options. Valid keys include:
            - echo: Enable SQLAlchemy echo mode
            - fetch_format: "records", "pandas", or "polars"
            - dialect: Override SQL dialect detection
            - pool_size: Connection pool size
            - max_overflow: Maximum pool overflow connections
            - pool_timeout: Pool timeout in seconds
            - pool_recycle: Connection recycle time in seconds
            - pool_pre_ping: Enable connection health checks
            - future: Use SQLAlchemy 2.0 style (default: True)
            - Other options are stored in config.options

    Returns:
        MoltresConfig instance with parsed configuration

    Raises:
        ValueError: If dsn is not provided and MOLTRES_DSN is not set
    """
    # Get DSN from argument or environment variable
    if dsn is None:
        dsn = os.environ.get("MOLTRES_DSN")
        if dsn is None:
            raise ValueError(
                "dsn must be provided as argument or MOLTRES_DSN environment variable must be set"
            )

    # Load configuration from environment variables if not provided in kwargs
    env_config = _load_env_config()

    # Merge: kwargs override env vars, env vars override defaults
    merged_kwargs = {**env_config, **kwargs}

    engine_kwargs: dict[str, object] = {
        k: merged_kwargs.pop(k)
        for k in list(merged_kwargs)
        if k in EngineConfig.__dataclass_fields__
    }
    # Construct EngineConfig with validated kwargs
    # Using **kwargs with type checking would be ideal, but dataclass doesn't support it directly
    # This approach extracts known fields and passes them safely
    engine = EngineConfig(dsn=dsn, **engine_kwargs)  # type: ignore[arg-type]
    return MoltresConfig(engine=engine, options=merged_kwargs)


DEFAULT_CONFIG = create_config("sqlite:///:memory:")
