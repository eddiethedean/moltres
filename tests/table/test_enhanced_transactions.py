"""Tests for enhanced transaction control features."""

import pytest

from moltres import col, connect, async_connect
from moltres.io.records import Records, AsyncRecords
from moltres.table.schema import column


class TestSavepoints:
    """Test savepoint functionality."""

    def test_savepoint_creation(self, tmp_path):
        """Test creating a savepoint within a transaction."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        with db.transaction() as txn:
            # Insert first record
            Records(_data=[{"id": 1, "value": "first"}], _database=db).insert_into("test")

            # Create a savepoint
            sp_name = txn.savepoint()
            assert sp_name is not None

            # Insert more data after savepoint
            Records(_data=[{"id": 2, "value": "second"}], _database=db).insert_into("test")

            # Verify we have both inserts before rollback
            results = db.table("test").select().collect()
            assert len(results) == 2

            # Rollback to savepoint
            txn.rollback_to_savepoint(sp_name)

            # Verify only first insert remains after rollback
            results = db.table("test").select().collect()
            # After rollback to savepoint, only the first insert should remain
            assert len(results) == 1
            assert results[0]["id"] == 1
            assert results[0]["value"] == "first"

        # Transaction commits - verify final state
        results = db.table("test").select().collect()
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_nested_transaction_with_savepoint(self, tmp_path):
        """Test nested transactions using savepoints."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        with db.transaction() as outer:
            Records(_data=[{"id": 1, "value": "outer"}], _database=db).insert_into("test")

            # Nested transaction with savepoint
            with db.transaction(savepoint=True) as inner:
                Records(_data=[{"id": 2, "value": "inner"}], _database=db).insert_into("test")

                # Verify both inserts visible in nested transaction
                results = db.table("test").select().collect()
                assert len(results) == 2

            # Inner transaction (savepoint) is released on exit, but changes remain
            # in outer transaction. Outer transaction sees both inserts
            results = db.table("test").select().collect()
            # The inner transaction's changes are still in the outer transaction
            assert len(results) == 2

    def test_savepoint_rollback_multiple_savepoints(self, tmp_path):
        """Test rolling back to an earlier savepoint."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "test", [column("id", "INTEGER", primary_key=True), column("value", "TEXT")]
        ).collect()

        with db.transaction() as txn:
            Records(_data=[{"id": 1, "value": "first"}], _database=db).insert_into("test")

            sp1 = txn.savepoint("checkpoint1")
            Records(_data=[{"id": 2, "value": "second"}], _database=db).insert_into("test")

            # Verify we have both before second savepoint
            results = db.table("test").select().collect()
            assert len(results) == 2

            sp2 = txn.savepoint("checkpoint2")
            Records(_data=[{"id": 3, "value": "third"}], _database=db).insert_into("test")

            # Verify we have all three
            results = db.table("test").select().collect()
            assert len(results) == 3

            # Rollback to first savepoint
            txn.rollback_to_savepoint(sp1)

            # Should only have first insert after rollback
            results = db.table("test").select().collect()
            assert len(results) == 1
            assert results[0]["id"] == 1

    def test_savepoint_release(self, tmp_path):
        """Test releasing a savepoint."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("test", [column("id", "INTEGER"), column("value", "TEXT")]).collect()

        with db.transaction() as txn:
            Records(_data=[{"id": 1, "value": "first"}], _database=db).insert_into("test")

            sp_name = txn.savepoint("test_sp")
            Records(_data=[{"id": 2, "value": "second"}], _database=db).insert_into("test")

            # Release savepoint
            txn.release_savepoint(sp_name)

            # Cannot rollback to released savepoint
            with pytest.raises(RuntimeError, match="Savepoint 'test_sp' not found"):
                txn.rollback_to_savepoint(sp_name)


class TestTransactionStateInspection:
    """Test transaction state inspection methods."""

    def test_is_in_transaction(self, tmp_path):
        """Test checking if currently in a transaction."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        assert not db.is_in_transaction()

        with db.transaction():
            assert db.is_in_transaction()

        assert not db.is_in_transaction()

    def test_get_transaction_status(self, tmp_path):
        """Test getting transaction status."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        assert db.get_transaction_status() is None

        # SQLite doesn't support isolation levels, so just test readonly=False
        with db.transaction(readonly=False) as txn:
            status = db.get_transaction_status()
            assert status is not None
            assert status["readonly"] is False

        assert db.get_transaction_status() is None


class TestReadOnlyTransactions:
    """Test read-only transactions."""

    @pytest.mark.skipif(
        True, reason="SQLite doesn't support read-only transactions"
    )  # SQLite limitation
    def test_readonly_transaction(self, tmp_path):
        """Test read-only transaction prevents writes."""
        db_path = tmp_path / "test.db"
        db = connect("postgresql:///test")  # Would need PostgreSQL

        db.create_table("test", [column("id", "INTEGER"), column("value", "TEXT")]).collect()

        with db.transaction(readonly=True) as txn:
            # Should be able to read
            results = db.table("test").select().collect()
            assert isinstance(results, list)

            # Writing should fail in read-only transaction
            # (Database-specific behavior)


class TestIsolationLevels:
    """Test transaction isolation levels."""

    @pytest.mark.skipif(
        True, reason="SQLite has limited isolation level support"
    )  # SQLite limitation
    def test_isolation_level_setting(self, tmp_path):
        """Test setting transaction isolation level."""
        db_path = tmp_path / "test.db"
        db = connect("postgresql:///test")  # Would need PostgreSQL

        with db.transaction(isolation_level="SERIALIZABLE") as txn:
            assert txn.isolation_level() == "SERIALIZABLE"
            status = db.get_transaction_status()
            assert status["isolation_level"] == "SERIALIZABLE"

    def test_invalid_isolation_level(self, tmp_path):
        """Test that invalid isolation level raises error."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        with pytest.raises(ValueError, match="does not support isolation levels"):
            with db.transaction(isolation_level="SERIALIZABLE"):
                pass


class TestRowLevelLocking:
    """Test row-level locking (FOR UPDATE/FOR SHARE)."""

    def test_select_for_update(self, tmp_path):
        """Test FOR UPDATE locking."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "orders", [column("id", "INTEGER", primary_key=True), column("status", "TEXT")]
        ).collect()
        Records(_data=[{"id": 1, "status": "pending"}], _database=db).insert_into("orders")

        with db.transaction() as txn:
            # Calling .select() without arguments creates a Project plan
            df = db.table("orders").select().where(col("status") == "pending")
            locked_df = df.select_for_update()
            results = locked_df.collect()
            assert len(results) == 1

    def test_select_for_share(self, tmp_path):
        """Test FOR SHARE locking."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "products", [column("id", "INTEGER", primary_key=True), column("stock", "INTEGER")]
        ).collect()
        Records(_data=[{"id": 1, "stock": 10}], _database=db).insert_into("products")

        with db.transaction() as txn:
            # Calling .select() without arguments creates a Project plan
            df = db.table("products").select().where(col("id") == 1)
            locked_df = df.select_for_share()
            results = locked_df.collect()
            assert len(results) == 1

    def test_select_for_update_nowait_unsupported(self, tmp_path):
        """Test that NOWAIT raises error on unsupported dialects."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "orders", [column("id", "INTEGER", primary_key=True), column("status", "TEXT")]
        ).collect()

        # Calling .select() creates a Project plan
        df = db.table("orders").select()
        with pytest.raises(ValueError, match="does not support FOR UPDATE NOWAIT"):
            df.select_for_update(nowait=True)


class TestAsyncEnhancedTransactions:
    """Test async enhanced transaction features."""

    @pytest.mark.asyncio
    async def test_async_savepoint(self, tmp_path):
        """Test async savepoint creation."""
        db_path = tmp_path / "test_async.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        await db.create_table("test", [column("id", "INTEGER"), column("value", "TEXT")]).collect()

        async with db.transaction() as txn:
            records = AsyncRecords(_data=[{"id": 1, "value": "first"}], _database=db)
            await records.insert_into("test")

            # Create a savepoint
            sp_name = await txn.savepoint()
            assert sp_name is not None

            # Insert more data after savepoint
            records2 = AsyncRecords(_data=[{"id": 2, "value": "second"}], _database=db)
            await records2.insert_into("test")

            # Verify both inserts before rollback
            table_handle = await db.table("test")
            df = table_handle.select()
            results = await df.collect()
            assert len(results) == 2

            # Rollback to savepoint
            await txn.rollback_to_savepoint(sp_name)

            # Verify only first insert remains after rollback
            results = await df.collect()
            assert len(results) == 1
            assert results[0]["id"] == 1

        await db.close()

    @pytest.mark.asyncio
    async def test_async_transaction_state_inspection(self, tmp_path):
        """Test async transaction state inspection."""
        db_path = tmp_path / "test_async.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        assert not db.is_in_transaction()
        assert db.get_transaction_status() is None

        async with db.transaction():
            assert db.is_in_transaction()
            status = db.get_transaction_status()
            assert status is not None

        assert not db.is_in_transaction()
        await db.close()

    @pytest.mark.asyncio
    async def test_async_row_level_locking(self, tmp_path):
        """Test async row-level locking."""
        db_path = tmp_path / "test_async.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        await db.create_table(
            "orders", [column("id", "INTEGER"), column("status", "TEXT")]
        ).collect()
        records = AsyncRecords(_data=[{"id": 1, "status": "pending"}], _database=db)
        await records.insert_into("orders")

        async with db.transaction() as txn:
            table_handle = await db.table("orders")
            # Calling .select() creates a Project plan
            df = table_handle.select().where(col("status") == "pending")
            locked_df = df.select_for_update()
            results = await locked_df.collect()
            assert len(results) == 1

        await db.close()
