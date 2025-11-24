"""Comprehensive tests for async execution covering edge cases."""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect
from moltres.engine.async_execution import (
    AsyncQueryResult,
    register_async_performance_hook,
    unregister_async_performance_hook,
)
from moltres.io.records import AsyncRecords


@pytest.mark.asyncio
class TestAsyncQueryExecutor:
    """Test AsyncQueryExecutor class."""

    async def test_fetch_with_sql_string(self, tmp_path):
        """Test fetch with SQL string."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
        await records.insert_into("test")

        result = await db.executor.fetch("SELECT * FROM test")
        assert result.rows is not None
        assert len(result.rows) == 1

        await db.close()

    async def test_fetch_with_params(self, tmp_path):
        """Test fetch with parameters."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        records = AsyncRecords(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db
        )
        await records.insert_into("test")

        result = await db.executor.fetch("SELECT * FROM test WHERE id = :id", params={"id": 1})
        assert len(result.rows) == 1
        assert result.rows[0]["name"] == "Alice"

        await db.close()

    async def test_execute_with_params(self, tmp_path):
        """Test execute with parameters."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        result = await db.executor.execute(
            "INSERT INTO test (id, name) VALUES (:id, :name)", params={"id": 1, "name": "Alice"}
        )
        assert result.rowcount == 1

        await db.close()

    async def test_execute_many_empty_list(self, tmp_path):
        """Test execute_many with empty params_list."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        result = await db.executor.execute_many(
            "INSERT INTO test (id, name) VALUES (:id, :name)", []
        )
        assert result.rowcount == 0

        await db.close()

    async def test_execute_many_multiple_rows(self, tmp_path):
        """Test execute_many with multiple parameter sets."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        params_list = [{"id": i, "name": f"User{i}"} for i in range(1, 6)]
        result = await db.executor.execute_many(
            "INSERT INTO test (id, name) VALUES (:id, :name)", params_list
        )
        assert result.rowcount == 5

        await db.close()

    async def test_fetch_stream(self, tmp_path):
        """Test fetch_stream method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "INTEGER")]
        ).collect()

        records = AsyncRecords(
            _data=[{"id": i, "value": i * 2} for i in range(1, 21)], _database=db
        )
        await records.insert_into("test")

        chunks = []
        async for chunk in db.executor.fetch_stream("SELECT * FROM test", chunk_size=5):
            chunks.extend(chunk)

        assert len(chunks) == 20

        await db.close()

    async def test_fetch_stream_with_chunk_size(self, tmp_path):
        """Test fetch_stream with custom chunk size."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "INTEGER")]
        ).collect()

        records = AsyncRecords(_data=[{"id": i, "value": i} for i in range(1, 11)], _database=db)
        await records.insert_into("test")

        chunk_count = 0
        total_rows = 0
        async for chunk in db.executor.fetch_stream("SELECT * FROM test", chunk_size=3):
            chunk_count += 1
            total_rows += len(chunk)

        assert total_rows == 10
        assert chunk_count >= 3  # At least 3 chunks with chunk_size=3

        await db.close()

    async def test_fetch_stream_empty_result(self, tmp_path):
        """Test fetch_stream with empty result."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        chunks = []
        async for chunk in db.executor.fetch_stream("SELECT * FROM test"):
            chunks.extend(chunk)

        assert len(chunks) == 0

        await db.close()


@pytest.mark.asyncio
class TestAsyncQueryResult:
    """Test AsyncQueryResult dataclass."""

    async def test_query_result_with_rows(self, tmp_path):
        """Test AsyncQueryResult with rows."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
        await records.insert_into("test")

        result = await db.executor.fetch("SELECT * FROM test")
        assert isinstance(result, AsyncQueryResult)
        assert result.rows is not None
        assert result.rowcount is not None

        await db.close()

    async def test_query_result_with_rowcount_only(self, tmp_path):
        """Test AsyncQueryResult with rowcount only (no rows)."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        result = await db.executor.execute("INSERT INTO test (id, name) VALUES (1, 'Alice')")
        assert isinstance(result, AsyncQueryResult)
        assert result.rows is None
        assert result.rowcount == 1

        await db.close()


@pytest.mark.asyncio
class TestAsyncPerformanceHooks:
    """Test async performance hooks."""

    async def test_register_performance_hook(self, tmp_path):
        """Test registering performance hook."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        hook_called = []

        def test_hook(sql: str, elapsed: float, metadata: dict) -> None:
            hook_called.append((sql, elapsed, metadata))

        register_async_performance_hook("query_start", test_hook)

        from moltres.table.schema import column

        await db.create_table("test", [column("id", "INTEGER", primary_key=True)]).collect()

        # Execute a query to trigger hook
        await db.executor.fetch("SELECT 1")

        assert len(hook_called) > 0

        # Cleanup
        unregister_async_performance_hook("query_start", test_hook)

        await db.close()

    async def test_register_invalid_event_error(self):
        """Test registering hook with invalid event raises error."""

        def test_hook(sql: str, elapsed: float, metadata: dict) -> None:
            pass

        with pytest.raises(ValueError, match="Unknown event type"):
            register_async_performance_hook("invalid_event", test_hook)

    async def test_unregister_performance_hook(self, tmp_path):
        """Test unregistering performance hook."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        hook_called = []

        def test_hook(sql: str, elapsed: float, metadata: dict) -> None:
            hook_called.append((sql, elapsed, metadata))

        register_async_performance_hook("query_end", test_hook)

        from moltres.table.schema import column

        await db.create_table("test", [column("id", "INTEGER", primary_key=True)]).collect()

        # Execute a query
        await db.executor.fetch("SELECT 1")
        initial_count = len(hook_called)

        # Unregister hook
        unregister_async_performance_hook("query_end", test_hook)

        # Execute another query - hook should not be called
        await db.executor.fetch("SELECT 2")

        # Count should not have increased
        assert len(hook_called) == initial_count

        await db.close()

    async def test_performance_hook_with_exception(self, tmp_path):
        """Test that hook exceptions don't break execution."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        def failing_hook(sql: str, elapsed: float, metadata: dict) -> None:
            raise ValueError("Hook error")

        register_async_performance_hook("query_start", failing_hook)

        from moltres.table.schema import column

        await db.create_table("test", [column("id", "INTEGER", primary_key=True)]).collect()

        # Query should still execute despite hook error
        result = await db.executor.fetch("SELECT 1")
        assert result.rows is not None

        # Cleanup
        unregister_async_performance_hook("query_start", failing_hook)

        await db.close()
