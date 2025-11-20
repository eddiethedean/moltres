"""Public Moltres API."""
from __future__ import annotations

from .config import MoltresConfig, create_config
from .expressions import col, lit
from .table.table import Database

__all__ = ["connect", "Database", "MoltresConfig", "col", "lit"]


def connect(dsn: str, **options: object) -> Database:
    """Connect to a SQL database and return a ``Database`` handle."""

    config: MoltresConfig = create_config(dsn, **options)
    return Database(config=config)
