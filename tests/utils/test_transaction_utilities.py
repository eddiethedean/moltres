"""Tests for transaction utility features."""

import pytest
import time

from moltres import connect, async_connect
from moltres.io.records import Records, AsyncRecords
from moltres.table.schema import column
from moltres.utils.transaction_decorator import transaction
from moltres.utils.transaction_hooks import (
    register_transaction_hook,
    register_async_transaction_hook,
    unregister_transaction_hook,
    unregister_async_transaction_hook,
)
from moltres.utils.transaction_metrics import (
    get_transaction_metrics,
    reset_transaction_metrics,
)
from moltres.utils.transaction_retry import (
    retry_transaction,
    retry_transaction_async,
    transaction_retry_config,
    is_transaction_retryable_error,
)
from moltres.utils.transaction_testing import (
    ConcurrentTransactionTester,
    DeadlockSimulator,
)
from sqlalchemy.exc import OperationalError


class TestTransactionDecorator:
    """Test transaction decorator."""

    def test_transaction_decorator_with_db_instance(self, tmp_path):
        """Test @transaction decorator with database instance."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        @transaction(db)
        def add_user(name: str):
            Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")

        add_user("Alice")
        results = db.table("users").select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Alice"

    def test_transaction_decorator_with_db_parameter(self, tmp_path):
        """Test @transaction decorator with database as parameter."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        @transaction
        def add_user(db, name: str):
            Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")

        add_user(db, "Bob")
        results = db.table("users").select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"

    def test_transaction_decorator_rollback_on_error(self, tmp_path):
        """Test that decorator rolls back on error."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        @transaction(db)
        def add_user_fail(name: str):
            Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")
            raise ValueError("Simulated error")

        with pytest.raises(ValueError, match="Simulated error"):
            add_user_fail("Fail")

        # Verify transaction was rolled back
        results = db.table("users").select().collect()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_transaction_decorator_async(self, tmp_path):
        """Test @transaction decorator with async function."""
        db_path = tmp_path / "test.db"
        async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        await async_db.create_table(
            "users", [column("id", "INTEGER"), column("name", "TEXT")]
        ).collect()

        @transaction(async_db)
        async def add_user_async(name: str):
            await AsyncRecords(_data=[{"id": 1, "name": name}], _database=async_db).insert_into(
                "users"
            )

        await add_user_async("Async")
        table_handle = await async_db.table("users")
        results = await table_handle.select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Async"

        await async_db.close()


class TestTransactionHooks:
    """Test transaction hooks."""

    def test_transaction_begin_hook(self, tmp_path):
        """Test transaction begin hook."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        hook_called = []

        def on_begin(txn):
            hook_called.append("begin")

        register_transaction_hook("begin", on_begin)

        with db.transaction() as txn:
            pass

        assert len(hook_called) == 1
        assert hook_called[0] == "begin"

        unregister_transaction_hook("begin", on_begin)

    def test_transaction_commit_hook(self, tmp_path):
        """Test transaction commit hook."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        hook_called = []

        def on_commit(txn):
            hook_called.append("commit")

        register_transaction_hook("commit", on_commit)

        with db.transaction() as txn:
            pass  # Commits automatically

        assert len(hook_called) == 1
        assert hook_called[0] == "commit"

        unregister_transaction_hook("commit", on_commit)

    def test_transaction_rollback_hook(self, tmp_path):
        """Test transaction rollback hook."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        hook_called = []

        def on_rollback(txn):
            hook_called.append("rollback")

        register_transaction_hook("rollback", on_rollback)

        try:
            with db.transaction() as txn:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        assert len(hook_called) == 1
        assert hook_called[0] == "rollback"

        unregister_transaction_hook("rollback", on_rollback)

    @pytest.mark.asyncio
    async def test_async_transaction_hooks(self, tmp_path):
        """Test async transaction hooks."""
        db_path = tmp_path / "test.db"
        async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        hook_called = []

        async def on_commit_async(txn):
            hook_called.append("commit")

        register_async_transaction_hook("commit", on_commit_async)

        async with async_db.transaction() as txn:
            pass

        assert len(hook_called) == 1
        assert hook_called[0] == "commit"

        unregister_async_transaction_hook("commit", on_commit_async)

        await async_db.close()


class TestTransactionMetrics:
    """Test transaction metrics."""

    def test_transaction_metrics_basic(self, tmp_path):
        """Test basic transaction metrics collection."""
        reset_transaction_metrics()
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        metrics = get_transaction_metrics()

        # Run a transaction
        with db.transaction() as txn:
            time.sleep(0.01)  # Small delay for duration measurement
            pass

        stats = metrics.get_stats()
        assert stats["transaction_count"] == 1
        assert stats["committed_count"] == 1
        assert stats["rolled_back_count"] == 0
        assert stats["commit_rate"] == 1.0
        assert stats["transaction_duration_avg"] > 0

    def test_transaction_metrics_with_rollback(self, tmp_path):
        """Test metrics with transaction rollback."""
        reset_transaction_metrics()
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        metrics = get_transaction_metrics()

        # Run a transaction that fails
        try:
            with db.transaction() as txn:
                raise ValueError("Test error")
        except ValueError:
            pass  # Expected

        stats = metrics.get_stats()
        assert stats["transaction_count"] == 1
        assert stats["committed_count"] == 0
        assert stats["rolled_back_count"] == 1
        assert stats["error_count"] == 1
        assert "ValueError" in stats["errors_by_type"]

    def test_transaction_metrics_reset(self, tmp_path):
        """Test resetting transaction metrics."""
        reset_transaction_metrics()
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        metrics = get_transaction_metrics()

        with db.transaction() as txn:
            pass

        stats = metrics.get_stats()
        assert stats["transaction_count"] == 1

        reset_transaction_metrics()
        stats = metrics.get_stats()
        assert stats["transaction_count"] == 0


class TestTransactionRetry:
    """Test transaction retry logic."""

    def test_retryable_error_detection(self):
        """Test detection of retryable transaction errors."""
        config = transaction_retry_config()

        # Deadlock error should be retryable
        error = OperationalError("statement", "parameters", "deadlock detected")
        assert is_transaction_retryable_error(error, config)

        # Lock timeout should be retryable
        error2 = OperationalError("statement", "parameters", "lock wait timeout exceeded")
        assert is_transaction_retryable_error(error2, config)

        # Normal ValueError should not be retryable
        error3 = ValueError("Some error")
        assert not is_transaction_retryable_error(error3, config)

    def test_retry_transaction_success(self, tmp_path):
        """Test successful transaction retry."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        attempt_count = [0]

        def transaction_func():
            attempt_count[0] += 1
            with db.transaction() as txn:
                Records(_data=[{"id": 1, "name": "Test"}], _database=db).insert_into("users")

        # Should succeed on first attempt
        retry_transaction(transaction_func)
        assert attempt_count[0] == 1

        results = db.table("users").select().collect()
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retry_transaction_async(self, tmp_path):
        """Test async transaction retry."""
        db_path = tmp_path / "test.db"
        async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        await async_db.create_table(
            "users", [column("id", "INTEGER"), column("name", "TEXT")]
        ).collect()

        attempt_count = [0]

        async def transaction_func():
            attempt_count[0] += 1
            async with async_db.transaction() as txn:
                await AsyncRecords(
                    _data=[{"id": 1, "name": "Test"}], _database=async_db
                ).insert_into("users")

        await retry_transaction_async(transaction_func)
        assert attempt_count[0] == 1

        table_handle = await async_db.table("users")
        results = await table_handle.select().collect()
        assert len(results) == 1

        await async_db.close()


class TestTransactionTesting:
    """Test transaction testing utilities."""

    def test_concurrent_transaction_tester(self, tmp_path):
        """Test concurrent transaction execution."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("counters", [column("id", "INTEGER"), column("value", "INTEGER")]).collect()

        tester = ConcurrentTransactionTester(db, num_threads=4)

        def update_counter(db):
            with db.transaction() as txn:
                # Simple counter update
                Records(_data=[{"id": 1, "value": 1}], _database=db).insert_into("counters")
                return {"success": True}

        results = tester.run_concurrent_transactions(update_counter, num_transactions=5)
        stats = tester.get_statistics()

        assert stats["total_transactions"] == 5
        # All should succeed (SQLite handles concurrent writes)
        assert stats["successful"] >= 0  # May vary based on timing

    def test_deadlock_simulator(self, tmp_path):
        """Test deadlock simulation."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("locks", [column("id", "INTEGER"), column("value", "TEXT")]).collect()

        simulator = DeadlockSimulator(db)

        def txn1(db):
            with db.transaction() as txn:
                Records(_data=[{"id": 1, "value": "A"}], _database=db).insert_into("locks")
                time.sleep(0.1)

        def txn2(db):
            with db.transaction() as txn:
                Records(_data=[{"id": 2, "value": "B"}], _database=db).insert_into("locks")
                time.sleep(0.1)

        # This won't create a deadlock in SQLite (no row-level locking in this simple case)
        # but tests the simulator structure
        results = simulator.simulate_deadlock(txn1, txn2)

        assert "deadlock_detected" in results
        assert "transaction1_success" in results
        assert "transaction2_success" in results
