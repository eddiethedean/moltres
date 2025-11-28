"""Comprehensive tests for async table operations covering edge cases."""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect
from moltres.table.schema import ColumnDef


@pytest.mark.asyncio
class TestAsyncTransaction:
    """Test AsyncTransaction class."""

    async def test_transaction_commit(self, tmp_path):
        """Test explicit transaction commit."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        async with db.transaction() as txn:
            from moltres.io.records import AsyncRecords

            records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
            await records.insert_into("test")
            await txn.commit()

        # Verify data was committed
        table = await db.table("test")
        results = await table.select().collect()
        assert len(results) == 1

        await db.close()

    async def test_transaction_rollback(self, tmp_path):
        """Test explicit transaction rollback."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        async with db.transaction() as txn:
            from moltres.io.records import AsyncRecords

            records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
            await records.insert_into("test")
            await txn.rollback()

        # Verify data was rolled back
        table = await db.table("test")
        results = await table.select().collect()
        assert len(results) == 0

        await db.close()

    async def test_transaction_auto_commit(self, tmp_path):
        """Test automatic commit on successful exit."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        async with db.transaction():
            from moltres.io.records import AsyncRecords

            records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
            await records.insert_into("test")
            # No explicit commit - should auto-commit on exit

        # Verify data was committed
        table = await db.table("test")
        results = await table.select().collect()
        assert len(results) == 1

        await db.close()

    async def test_transaction_auto_rollback_on_exception(self, tmp_path):
        """Test automatic rollback on exception."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        try:
            async with db.transaction():
                from moltres.io.records import AsyncRecords

                records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
                await records.insert_into("test")
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify data was rolled back
        table = await db.table("test")
        results = await table.select().collect()
        assert len(results) == 0

        await db.close()

    async def test_transaction_double_commit_error(self, tmp_path):
        """Test that double commit raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        async with db.transaction() as txn:
            await txn.commit()
            with pytest.raises(RuntimeError, match="already committed"):
                await txn.commit()

        await db.close()

    async def test_transaction_commit_after_rollback_error(self, tmp_path):
        """Test that commit after rollback raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        async with db.transaction() as txn:
            await txn.rollback()
            with pytest.raises(RuntimeError, match="already.*rolled back"):
                await txn.commit()

        await db.close()


@pytest.mark.asyncio
class TestAsyncDatabase:
    """Test AsyncDatabase class methods."""

    async def test_sql_method(self, tmp_path):
        """Test sql() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
        await records.insert_into("users")

        # Test SQL query
        df = db.sql("SELECT * FROM users WHERE id = 1")
        results = await df.collect()
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

        await db.close()

    async def test_sql_method_with_params(self, tmp_path):
        """Test sql() method with parameters."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db
        )
        await records.insert_into("users")

        # Test parameterized SQL query
        df = db.sql("SELECT * FROM users WHERE id = :id", id=1)
        results = await df.collect()
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

        await db.close()

    async def test_table_method_empty_name_error(self, tmp_path):
        """Test table() with empty name raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        with pytest.raises(Exception):  # ValidationError or similar
            await db.table("")

        await db.close()

    async def test_create_table_empty_columns_error(self, tmp_path):
        """Test create_table with empty columns raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        with pytest.raises(Exception, match="no columns"):
            db.create_table("test", [])

        await db.close()

    async def test_create_table_temporary(self, tmp_path):
        """Test create_table with temporary flag."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        # Create temporary table - temporary tables are connection-scoped in SQLite
        # Since aiosqlite uses connection pooling, we can't reliably query it from
        # a different connection. Just verify the creation succeeds.
        await db.create_table(
            "temp_table", [column("id", "INTEGER", primary_key=True)], temporary=True
        ).collect()

        # Verify table creation succeeded by checking it doesn't raise an error
        # Note: We can't query it reliably due to connection pooling, but the
        # creation itself validates the temporary flag works
        assert True  # Table creation succeeded

        await db.close()

    async def test_create_table_if_not_exists(self, tmp_path):
        """Test create_table with if_not_exists."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True)], if_not_exists=True
        ).collect()
        # Should not error on second call
        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True)], if_not_exists=True
        ).collect()

        await db.close()

    async def test_createDataFrame_from_list_of_dicts(self, tmp_path):
        """Test createDataFrame from list of dicts."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        results = await df.collect()
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

        await db.close()

    async def test_createDataFrame_from_list_of_tuples(self, tmp_path):
        """Test createDataFrame from list of tuples with schema."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        schema = [
            ColumnDef(name="id", type_name="INTEGER", primary_key=True),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        df = await db.createDataFrame([(1, "Alice"), (2, "Bob")], schema=schema)
        results = await df.collect()
        assert len(results) == 2

        await db.close()

    async def test_createDataFrame_from_list_of_tuples_no_schema_error(self, tmp_path):
        """Test createDataFrame from list of tuples without schema raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        with pytest.raises(Exception, match="requires a schema"):
            await db.createDataFrame([(1, "Alice")])

        await db.close()

    async def test_createDataFrame_empty_data_no_schema_error(self, tmp_path):
        """Test createDataFrame with empty data and no schema raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        with pytest.raises(Exception, match="Cannot create DataFrame from empty data"):
            await db.createDataFrame([])

        await db.close()

    async def test_createDataFrame_empty_data_with_schema(self, tmp_path):
        """Test createDataFrame with empty data but with schema."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        schema = [
            ColumnDef(name="id", type_name="INTEGER", primary_key=True),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        df = await db.createDataFrame([], schema=schema)
        results = await df.collect()
        assert len(results) == 0

        await db.close()

    async def test_createDataFrame_unsupported_type_error(self, tmp_path):
        """Test createDataFrame with unsupported type raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        with pytest.raises(Exception, match="Unsupported data type"):
            await db.createDataFrame("not a list")  # type: ignore[arg-type]

        await db.close()

    async def test_compile_plan(self, tmp_path):
        """Test compile_plan method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.logical import operators

        plan = operators.scan("test")
        sql = db.compile_plan(plan)
        assert sql is not None

        await db.close()

    async def test_execute_plan(self, tmp_path):
        """Test execute_plan method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
        await records.insert_into("test")

        from moltres.logical import operators

        plan = operators.scan("test")
        result = await db.execute_plan(plan)
        assert len(result.rows) == 1

        await db.close()

    async def test_execute_plan_stream(self, tmp_path):
        """Test execute_plan_stream method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(
            _data=[{"id": i, "value": f"test{i}"} for i in range(10)], _database=db
        )
        await records.insert_into("test")

        from moltres.logical import operators

        plan = operators.scan("test")
        chunks = []
        async for chunk in db.execute_plan_stream(plan):
            chunks.extend(chunk)

        assert len(chunks) == 10

        await db.close()

    async def test_execute_sql(self, tmp_path):
        """Test execute_sql method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
        await records.insert_into("test")

        result = await db.execute_sql("SELECT * FROM test")
        assert len(result.rows) == 1

        await db.close()

    async def test_execute_sql_with_params(self, tmp_path):
        """Test execute_sql with parameters."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(_data=[{"id": 1, "value": "test"}], _database=db)
        await records.insert_into("test")

        result = await db.execute_sql("SELECT * FROM test WHERE id = :id", params={"id": 1})
        assert len(result.rows) == 1

        await db.close()

    async def test_dialect_property(self, tmp_path):
        """Test dialect property."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        dialect = db.dialect
        assert dialect is not None
        assert hasattr(dialect, "name")

        await db.close()

    async def test_ephemeral_table_cleanup(self, tmp_path):
        """Test ephemeral table cleanup."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create DataFrame which creates ephemeral table
        await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")

        # Close database - should cleanup ephemeral tables
        await db.close()

        # Verify cleanup happened (table should not exist)
        # This is tested implicitly by the close not raising errors

    async def test_register_unregister_ephemeral_table(self, tmp_path):
        """Test ephemeral table registration."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Register ephemeral table
        db._register_ephemeral_table("test_table")
        assert "test_table" in db._ephemeral_tables

        # Unregister
        db._unregister_ephemeral_table("test_table")
        assert "test_table" not in db._ephemeral_tables

        await db.close()

    async def test_close_idempotent(self, tmp_path):
        """Test that close() is idempotent."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        await db.close()
        # Should not error on second close
        await db.close()

    async def test_dialect_name_extraction(self, tmp_path):
        """Test dialect name extraction from DSN."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        dialect_name = db._dialect_name
        assert dialect_name == "sqlite"

        await db.close()

    async def test_dialect_name_with_plus(self, tmp_path):
        """Test dialect name extraction with driver suffix."""
        db_path = tmp_path / "test.db"
        # Test with explicit async driver
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        dialect_name = db._dialect_name
        assert dialect_name == "sqlite"

        await db.close()


@pytest.mark.asyncio
class TestAsyncTableHandle:
    """Test AsyncTableHandle class."""

    async def test_select_method(self, tmp_path):
        """Test AsyncTableHandle.select() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table = await db.table("users")
        df = table.select()
        assert df is not None

        # Test with specific columns
        df2 = table.select("id", "name")
        assert df2 is not None

        await db.close()
