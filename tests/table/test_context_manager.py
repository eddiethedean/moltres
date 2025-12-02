"""Tests for database context manager functionality."""

from __future__ import annotations

import pytest

from moltres import async_connect, connect, col
from moltres.table.schema import column


class TestDatabaseContextManager:
    """Test sync Database context manager."""

    def test_context_manager_normal_exit(self, tmp_path):
        """Test that database is closed on normal exit."""
        db_path = tmp_path / "test.db"

        with connect(f"sqlite:///{db_path}") as db:
            # Database should be open
            assert not db._closed

            # Operations should work
            db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            from moltres.io.records import Records

            Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
            df = db.table("users").select()
            results = df.collect()
            assert len(results) == 1
            assert results[0]["name"] == "Alice"

        # Database should be closed after exiting context
        assert db._closed

    def test_context_manager_exception_handling(self, tmp_path):
        """Test that database is closed even when exception occurs."""
        db_path = tmp_path / "test.db"
        db = None

        with pytest.raises(ValueError, match="test exception"):
            with connect(f"sqlite:///{db_path}") as db:
                # Database should be open
                assert not db._closed
                db.create_table("users", [column("id", "INTEGER")]).collect()
                # Raise an exception
                raise ValueError("test exception")

        # Database should still be closed despite exception
        assert db is not None
        assert db._closed

    def test_context_manager_operations_work(self, tmp_path):
        """Test that database operations work within context."""
        db_path = tmp_path / "test.db"

        with connect(f"sqlite:///{db_path}") as db:
            # Create table
            db.create_table(
                "products", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
            ).collect()

            # Insert data
            from moltres.io.records import Records

            Records(
                _data=[{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}],
                _database=db,
            ).insert_into("products")

            # Query data
            df = db.table("products").select().where(col("id") == 1)
            results = df.collect()
            assert len(results) == 1
            assert results[0]["name"] == "Widget"

            # Use SQL method
            df2 = db.sql("SELECT COUNT(*) as count FROM products")
            count_results = df2.collect()
            assert count_results[0]["count"] == 2

        # Verify database is closed
        assert db._closed

    def test_context_manager_idempotency(self, tmp_path):
        """Test that calling close() after context exit is safe."""
        db_path = tmp_path / "test.db"

        with connect(f"sqlite:///{db_path}") as db:
            db.create_table("test", [column("id", "INTEGER")]).collect()

        # Database should be closed
        assert db._closed

        # Calling close() again should be safe (idempotent)
        db.close()
        assert db._closed

    def test_context_manager_closed_flag(self, tmp_path):
        """Test that database is marked as closed after context exit."""
        db_path = tmp_path / "test.db"

        with connect(f"sqlite:///{db_path}") as db:
            # Should not be closed while in context
            assert not db._closed
            # Verify _closed is the flag used
            assert hasattr(db, "_closed")

        # Should be closed after context
        assert db._closed


class TestAsyncDatabaseContextManager:
    """Test async AsyncDatabase context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager_normal_exit(self, tmp_path):
        """Test that async database is closed on normal exit."""
        db_path = tmp_path / "test_async.db"

        async with async_connect(f"sqlite+aiosqlite:///{db_path}") as db:
            # Database should be open
            assert not db._closed

            # Operations should work
            await db.create_table(
                "users", [column("id", "INTEGER"), column("name", "TEXT")]
            ).collect()
            from moltres.io.records import AsyncRecords

            records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
            await records.insert_into("users")
            table_handle = await db.table("users")
            df = table_handle.select()
            results = await df.collect()
            assert len(results) == 1
            assert results[0]["name"] == "Alice"

        # Database should be closed after exiting context
        assert db._closed

    @pytest.mark.asyncio
    async def test_async_context_manager_exception_handling(self, tmp_path):
        """Test that async database is closed even when exception occurs."""
        db_path = tmp_path / "test_async.db"
        db = None

        with pytest.raises(ValueError, match="test exception"):
            async with async_connect(f"sqlite+aiosqlite:///{db_path}") as db:
                # Database should be open
                assert not db._closed
                await db.create_table("users", [column("id", "INTEGER")]).collect()
                # Raise an exception
                raise ValueError("test exception")

        # Database should still be closed despite exception
        assert db is not None
        assert db._closed

    @pytest.mark.asyncio
    async def test_async_context_manager_operations_work(self, tmp_path):
        """Test that async database operations work within context."""
        db_path = tmp_path / "test_async.db"

        async with async_connect(f"sqlite+aiosqlite:///{db_path}") as db:
            # Create table
            await db.create_table(
                "products", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
            ).collect()

            # Insert data
            from moltres.io.records import AsyncRecords

            records = AsyncRecords(
                _data=[{"id": 1, "name": "Widget"}, {"id": 2, "name": "Gadget"}],
                _database=db,
            )
            await records.insert_into("products")

            # Query data
            table_handle = await db.table("products")
            df = table_handle.select()
            results = await df.collect()
            assert len(results) == 2
            assert results[0]["name"] in ("Widget", "Gadget")

            # Use SQL method
            df2 = db.sql("SELECT COUNT(*) as count FROM products")
            count_results = await df2.collect()
            assert count_results[0]["count"] == 2

        # Verify database is closed
        assert db._closed

    @pytest.mark.asyncio
    async def test_async_context_manager_idempotency(self, tmp_path):
        """Test that calling await close() after context exit is safe."""
        db_path = tmp_path / "test_async.db"

        async with async_connect(f"sqlite+aiosqlite:///{db_path}") as db:
            await db.create_table("test", [column("id", "INTEGER")]).collect()

        # Database should be closed
        assert db._closed

        # Calling close() again should be safe (idempotent)
        await db.close()
        assert db._closed

    @pytest.mark.asyncio
    async def test_async_context_manager_closed_flag(self, tmp_path):
        """Test that async database is marked as closed after context exit."""
        db_path = tmp_path / "test_async.db"

        async with async_connect(f"sqlite+aiosqlite:///{db_path}") as db:
            # Should not be closed while in context
            assert not db._closed
            # Verify _closed is the flag used
            assert hasattr(db, "_closed")

        # Should be closed after context
        assert db._closed
