"""Runtime configuration objects for Moltres."""

from __future__ import annotations

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
    future: bool = True


@dataclass
class MoltresConfig:
    """Container for all runtime configuration knobs."""

    engine: EngineConfig
    default_schema: Optional[str] = None
    include_metadata: bool = False
    options: dict[str, object] = field(default_factory=dict)


def create_config(dsn: str, **kwargs: object) -> MoltresConfig:
    """Convenience helper used by ``moltres.connect``."""

    engine_kwargs = {
        k: kwargs.pop(k) for k in list(kwargs) if k in EngineConfig.__dataclass_fields__
    }
    engine = EngineConfig(dsn=dsn, **engine_kwargs)  # type: ignore[arg-type]
    return MoltresConfig(engine=engine, options=kwargs)


DEFAULT_CONFIG = create_config("sqlite:///:memory:")
