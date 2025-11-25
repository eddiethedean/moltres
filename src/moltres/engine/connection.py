"""SQLAlchemy connection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from ..config import EngineConfig


class ConnectionManager:
    """Creates and caches SQLAlchemy engines for Moltres sessions."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine: Engine | None = None
        self._active_transaction: Optional[Connection] = None

    def _create_engine(self) -> Engine:
        # If an engine is provided in config, use it directly
        if self.config.engine is not None:
            if not isinstance(self.config.engine, Engine):
                raise TypeError("config.engine must be a synchronous Engine, not AsyncEngine")
            return self.config.engine

        # Otherwise, create a new engine from DSN
        if self.config.dsn is None:
            raise ValueError("Either 'dsn' or 'engine' must be provided in EngineConfig")

        kwargs: dict[str, object] = {"echo": self.config.echo, "future": self.config.future}
        if self.config.pool_size is not None:
            kwargs["pool_size"] = self.config.pool_size
        if self.config.max_overflow is not None:
            kwargs["max_overflow"] = self.config.max_overflow
        if self.config.pool_timeout is not None:
            kwargs["pool_timeout"] = self.config.pool_timeout
        if self.config.pool_recycle is not None:
            kwargs["pool_recycle"] = self.config.pool_recycle
        if self.config.pool_pre_ping:
            kwargs["pool_pre_ping"] = self.config.pool_pre_ping
        return create_engine(self.config.dsn, **kwargs)

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = self._create_engine()
        return self._engine

    @contextmanager
    def connect(self, transaction: Optional[Connection] = None) -> Iterator[Connection]:
        """Get a database connection.

        Args:
            transaction: If provided, use this transaction connection instead of creating a new one.
                        This allows operations to share a transaction.

        Yields:
            Database connection
        """
        if transaction is not None:
            # Use the provided transaction connection
            yield transaction
        else:
            # Create a new connection with auto-commit (default behavior)
            with self.engine.begin() as connection:
                yield connection

    def begin_transaction(self) -> Connection:
        """Begin a new transaction and return the connection.

        Returns:
            Connection that is part of a transaction (not auto-committed)
        """
        if self._active_transaction is not None:
            raise RuntimeError("Transaction already active. Nested transactions not yet supported.")
        self._active_transaction = self.engine.connect()
        self._active_transaction.begin()
        return self._active_transaction

    def commit_transaction(self, connection: Connection) -> None:
        """Commit a transaction.

        Args:
            connection: The transaction connection to commit
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        try:
            connection.commit()
        finally:
            # Always close connection, even if commit fails
            connection.close()
            self._active_transaction = None

    def rollback_transaction(self, connection: Connection) -> None:
        """Rollback a transaction.

        Args:
            connection: The transaction connection to rollback
        """
        if connection is not self._active_transaction:
            raise RuntimeError("Connection is not the active transaction")
        try:
            connection.rollback()
        finally:
            # Always close connection, even if rollback fails
            connection.close()
            self._active_transaction = None

    @property
    def active_transaction(self) -> Optional[Connection]:
        """Get the active transaction connection if one exists."""
        return self._active_transaction
