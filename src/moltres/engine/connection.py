"""SQLAlchemy connection helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from ..config import EngineConfig


class ConnectionManager:
    """Creates and caches SQLAlchemy engines for Moltres sessions."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine: Optional[Engine] = None

    def _create_engine(self) -> Engine:
        kwargs: Dict[str, object] = {"echo": self.config.echo, "future": self.config.future}
        if self.config.pool_size is not None:
            kwargs["pool_size"] = self.config.pool_size
        return create_engine(self.config.dsn, **kwargs)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @contextmanager
    def connect(self) -> Iterator[Connection]:
        with self.engine.begin() as connection:
            yield connection
