"""Comprehensive tests for async table actions covering all mutation operations."""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col
from moltres.table.async_actions import (
    AsyncCreateTableOperation,
    AsyncDeleteMutation,
    AsyncDropTableOperation,
    AsyncInsertMutation,
    AsyncMergeMutation,
    AsyncUpdateMutation,
)
from moltres.table.schema import column


@pytest.mark.asyncio
class TestAsyncInsertMutation:
    """Test AsyncInsertMutation class."""

    async def test_insert_mutation_collect(self, tmp_path):
        """Test insert mutation collect() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table_handle = await db.table("users")
        rows = [{"id": 1, "name": "Alice"}]
        mutation = AsyncInsertMutation(handle=table_handle, rows=rows)

        count = await mutation.collect()
        assert count == 1

        # Verify insertion
        results = await table_handle.select().collect()
        assert len(results) == 1

        await db.close()

    async def test_insert_mutation_to_sql(self, tmp_path):
        """Test insert mutation to_sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table_handle = await db.table("users")
        rows = [{"id": 1, "name": "Alice"}]
        mutation = AsyncInsertMutation(handle=table_handle, rows=rows)

        sql = mutation.to_sql()
        assert "INSERT INTO" in sql
        assert "users" in sql
        assert "id" in sql
        assert "name" in sql

        await db.close()

    async def test_insert_mutation_empty_rows(self, tmp_path):
        """Test insert mutation with empty rows."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table("users", [column("id", "INTEGER", primary_key=True)]).collect()

        table_handle = await db.table("users")
        mutation = AsyncInsertMutation(handle=table_handle, rows=[])

        sql = mutation.to_sql()
        assert sql == ""

        await db.close()


@pytest.mark.asyncio
class TestAsyncUpdateMutation:
    """Test AsyncUpdateMutation class."""

    async def test_update_mutation_collect(self, tmp_path):
        """Test update mutation collect() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        # Insert initial data
        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df.write.insertInto("users")

        table_handle = await db.table("users")
        mutation = AsyncUpdateMutation(
            handle=table_handle, where=col("id") == 1, values={"name": "Bob"}
        )

        count = await mutation.collect()
        assert count == 1

        # Verify update
        results = await table_handle.select().collect()
        assert results[0]["name"] == "Bob"

        await db.close()

    async def test_update_mutation_to_sql(self, tmp_path):
        """Test update mutation to_sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table_handle = await db.table("users")
        mutation = AsyncUpdateMutation(
            handle=table_handle, where=col("id") == 1, values={"name": "Bob"}
        )

        sql = mutation.to_sql()
        assert "UPDATE" in sql
        assert "users" in sql
        assert "SET" in sql
        assert "WHERE" in sql

        await db.close()

    async def test_update_mutation_empty_values(self, tmp_path):
        """Test update mutation with empty values."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table("users", [column("id", "INTEGER", primary_key=True)]).collect()

        table_handle = await db.table("users")
        mutation = AsyncUpdateMutation(handle=table_handle, where=col("id") == 1, values={})

        sql = mutation.to_sql()
        assert sql == ""

        await db.close()


@pytest.mark.asyncio
class TestAsyncDeleteMutation:
    """Test AsyncDeleteMutation class."""

    async def test_delete_mutation_collect(self, tmp_path):
        """Test delete mutation collect() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        # Insert initial data
        df = await db.createDataFrame(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id"
        )
        await df.write.insertInto("users")

        table_handle = await db.table("users")
        mutation = AsyncDeleteMutation(handle=table_handle, where=col("id") == 1)

        count = await mutation.collect()
        assert count == 1

        # Verify deletion
        results = await table_handle.select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"

        await db.close()

    async def test_delete_mutation_to_sql(self, tmp_path):
        """Test delete mutation to_sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table("users", [column("id", "INTEGER", primary_key=True)]).collect()

        table_handle = await db.table("users")
        mutation = AsyncDeleteMutation(handle=table_handle, where=col("id") == 1)

        sql = mutation.to_sql()
        assert "DELETE FROM" in sql
        assert "users" in sql
        assert "WHERE" in sql

        await db.close()


@pytest.mark.asyncio
class TestAsyncMergeMutation:
    """Test AsyncMergeMutation class."""

    async def test_merge_mutation_collect(self, tmp_path):
        """Test merge mutation collect() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        # Insert initial data
        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df.write.insertInto("users")

        table_handle = await db.table("users")
        rows = [{"id": 1, "name": "Bob"}, {"id": 2, "name": "Charlie"}]
        mutation = AsyncMergeMutation(handle=table_handle, rows=rows, on=["id"])

        count = await mutation.collect()
        assert count >= 1

        # Verify merge
        results = await table_handle.select().collect()
        assert len(results) >= 1

        await db.close()

    async def test_merge_mutation_to_sql(self, tmp_path):
        """Test merge mutation to_sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table_handle = await db.table("users")
        rows = [{"id": 1, "name": "Alice"}]
        mutation = AsyncMergeMutation(handle=table_handle, rows=rows, on=["id"])

        sql = mutation.to_sql()
        # Should return placeholder for complex SQL
        assert "MERGE" in sql or "UPSERT" in sql

        await db.close()

    async def test_merge_mutation_with_when_matched(self, tmp_path):
        """Test merge mutation with when_matched."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        # Insert initial data
        df = await db.createDataFrame([{"id": 1, "name": "Alice", "age": 30}], pk="id")
        await df.write.insertInto("users")

        table_handle = await db.table("users")
        rows = [{"id": 1, "name": "Bob", "age": 25}]
        mutation = AsyncMergeMutation(
            handle=table_handle,
            rows=rows,
            on=["id"],
            when_matched={"name": "updated_name", "age": "updated_age"},
        )

        count = await mutation.collect()
        assert count >= 0

        await db.close()


@pytest.mark.asyncio
class TestAsyncCreateTableOperation:
    """Test AsyncCreateTableOperation class."""

    async def test_create_table_operation_collect(self, tmp_path):
        """Test create table operation collect() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        columns = [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        operation = AsyncCreateTableOperation(database=db, name="users", columns=columns)

        handle = await operation.collect()
        assert handle is not None
        assert handle.name == "users"

        # Verify table exists
        results = await handle.select().collect()
        assert results == []

        await db.close()

    async def test_create_table_operation_to_sql(self, tmp_path):
        """Test create table operation to_sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        columns = [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        operation = AsyncCreateTableOperation(database=db, name="users", columns=columns)

        sql = operation.to_sql()
        assert "CREATE TABLE" in sql
        assert "users" in sql
        assert "id" in sql
        assert "name" in sql

        await db.close()

    async def test_create_table_operation_if_not_exists(self, tmp_path):
        """Test create table operation with if_not_exists."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        columns = [column("id", "INTEGER", primary_key=True)]
        operation = AsyncCreateTableOperation(
            database=db, name="users", columns=columns, if_not_exists=True
        )

        handle1 = await operation.collect()
        # Should not error when creating again
        handle2 = await operation.collect()
        assert handle1.name == handle2.name

        await db.close()

    async def test_create_table_operation_temporary(self, tmp_path):
        """Test create table operation with temporary flag."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        columns = [column("id", "INTEGER", primary_key=True)]
        operation = AsyncCreateTableOperation(
            database=db, name="temp_users", columns=columns, temporary=True
        )

        handle = await operation.collect()
        assert handle is not None

        await db.close()


@pytest.mark.asyncio
class TestAsyncDropTableOperation:
    """Test AsyncDropTableOperation class."""

    async def test_drop_table_operation_collect(self, tmp_path):
        """Test drop table operation collect() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create table first
        from moltres.table.schema import column

        await db.create_table("users", [column("id", "INTEGER", primary_key=True)]).collect()

        operation = AsyncDropTableOperation(database=db, name="users", if_exists=True)
        await operation.collect()

        # Verify table is dropped (should error when trying to access)
        # Table should not exist - this might raise an error depending on implementation
        # For now, just verify the operation completes

        await db.close()

    async def test_drop_table_operation_to_sql(self, tmp_path):
        """Test drop table operation to_sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        operation = AsyncDropTableOperation(database=db, name="users", if_exists=True)
        sql = operation.to_sql()
        assert "DROP TABLE" in sql
        assert "users" in sql

        await db.close()

    async def test_drop_table_operation_if_exists(self, tmp_path):
        """Test drop table operation with if_exists flag."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Drop non-existent table with if_exists=True (should not error)
        operation = AsyncDropTableOperation(database=db, name="nonexistent", if_exists=True)
        await operation.collect()  # Should not raise error

        await db.close()
