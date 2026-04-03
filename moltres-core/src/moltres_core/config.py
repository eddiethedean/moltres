"""Minimal engine configuration for moltres-core (SQLAlchemy)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Union

from sqlalchemy.engine import Engine

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine
else:
    AsyncEngine = None  # type: ignore[assignment, misc]

FetchFormat = Literal["pandas", "polars", "records"]
EngineOptionValue = Union[bool, int, float, str, None]


@dataclass
class EngineConfig:
    """Connection + execution options for SQLAlchemy engines."""

    dsn: str | None = None
    engine: Engine | "AsyncEngine" | None = None
    session: object | None = None
    echo: bool = False
    fetch_format: FetchFormat = "records"
    dialect: str | None = None
    pool_size: int | None = None
    max_overflow: int | None = None
    pool_timeout: int | None = None
    pool_recycle: int | None = None
    pool_pre_ping: bool = False
    query_timeout: float | None = None
    future: bool = True

    def __post_init__(self) -> None:
        provided = [self.dsn, self.engine, self.session]
        if all(x is None for x in provided):
            raise ValueError("Either 'dsn', 'engine', or 'session' must be provided")
        if sum(1 for x in provided if x is not None) > 1:
            raise ValueError(
                "Cannot provide multiple of 'dsn', 'engine', and 'session'."
            )
