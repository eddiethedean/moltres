"""Unit tests for async executor timeout handling using stubs."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from moltres.config import EngineConfig
from moltres.engine.async_execution import AsyncQueryExecutor


class _StubResult:
    def __init__(self):
        self.rowcount = 1

    def fetchall(self):
        return [(1,)]

    def keys(self):
        return ["value"]


class _StubAsyncConnection:
    def __init__(self):
        self.timeout = None

    def execution_options(self, **options):
        self.timeout = options.get("timeout")
        return self

    async def execute(self, *args, **kwargs):
        return _StubResult()


class _StubConnectionManager:
    def __init__(self):
        self.connection = _StubAsyncConnection()

    @asynccontextmanager
    async def connect(self, transaction=None):
        yield self.connection


@pytest.mark.asyncio
async def test_async_executor_applies_query_timeout():
    """AsyncQueryExecutor should pass query_timeout to SQLAlchemy connections."""
    config = EngineConfig(dsn="sqlite:///:memory:", query_timeout=3.5)
    manager = _StubConnectionManager()
    executor = AsyncQueryExecutor(manager, config)

    result = await executor.fetch("SELECT 1")

    assert result.rowcount == 1
    assert manager.connection.timeout == pytest.approx(3.5)
