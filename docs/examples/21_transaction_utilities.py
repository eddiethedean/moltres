"""Example: Transaction Utilities

This example demonstrates Moltres's transaction utility features:
- Transaction decorator
- Transaction hooks
- Transaction metrics
- Transaction retry
- Transaction testing utilities

Run this example:
    python docs/examples/21_transaction_utilities.py
"""

from moltres import connect, async_connect
from moltres.utils.transaction_decorator import transaction
from moltres.utils.transaction_hooks import (
    register_transaction_hook,
    unregister_transaction_hook,
)
from moltres.utils.transaction_metrics import (
    get_transaction_metrics,
    reset_transaction_metrics,
)
from moltres.utils.transaction_retry import (
    retry_transaction,
    transaction_retry_config,
)
from moltres.utils.transaction_testing import ConcurrentTransactionTester
from moltres.io.records import Records, AsyncRecords
from moltres.table.schema import column
import time


def example_transaction_decorator():
    """Demonstrate transaction decorator."""
    print("\n=== Transaction Decorator ===")

    db = connect("sqlite:///:memory:")
    db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

    # Method 1: Database instance provided to decorator
    @transaction(db)
    def create_user(name: str):
        Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")
        print(f"Created user: {name}")

    # Method 2: Database as parameter
    @transaction
    def create_user_with_db(db, name: str):
        Records(_data=[{"id": 2, "name": name}], _database=db).insert_into("users")
        print(f"Created user with db param: {name}")

    create_user("Alice")
    create_user_with_db(db, "Bob")

    results = db.table("users").select().collect()
    print(f"Users in database: {len(results)}")
    for user in results:
        print(f"  - {user['name']}")

    db.close()


def example_transaction_hooks():
    """Demonstrate transaction hooks."""
    print("\n=== Transaction Hooks ===")

    db = connect("sqlite:///:memory:")
    db.create_table("logs", [column("id", "INTEGER"), column("message", "TEXT")]).collect()

    hook_calls = []

    def on_begin(txn):
        hook_calls.append("begin")
        print("Hook: Transaction began")

    def on_commit(txn):
        hook_calls.append("commit")
        print("Hook: Transaction committed")

    def on_rollback(txn):
        hook_calls.append("rollback")
        print("Hook: Transaction rolled back")

    register_transaction_hook("begin", on_begin)
    register_transaction_hook("commit", on_commit)
    register_transaction_hook("rollback", on_rollback)

    # Successful transaction
    with db.transaction() as txn:
        Records(_data=[{"id": 1, "message": "Success"}], _database=db).insert_into("logs")

    # Failed transaction
    try:
        with db.transaction() as txn:
            Records(_data=[{"id": 2, "message": "Fail"}], _database=db).insert_into("logs")
            raise ValueError("Simulated error")
    except ValueError:
        pass

    print(f"\nHook calls: {hook_calls}")

    # Cleanup
    unregister_transaction_hook("begin", on_begin)
    unregister_transaction_hook("commit", on_commit)
    unregister_transaction_hook("rollback", on_rollback)

    db.close()


def example_transaction_metrics():
    """Demonstrate transaction metrics."""
    print("\n=== Transaction Metrics ===")

    reset_transaction_metrics()
    db = connect("sqlite:///:memory:")
    db.create_table("counters", [column("id", "INTEGER"), column("value", "INTEGER")]).collect()

    metrics = get_transaction_metrics()

    # Run some transactions
    for i in range(5):
        with db.transaction() as txn:
            Records(_data=[{"id": i, "value": i * 10}], _database=db).insert_into("counters")
            time.sleep(0.01)  # Small delay

    # One read-only transaction
    with db.transaction(readonly=True) as txn:
        results = db.table("counters").select().collect()

    # One that fails
    try:
        with db.transaction() as txn:
            Records(_data=[{"id": 99, "value": 999}], _database=db).insert_into("counters")
            raise ValueError("Test error")
    except ValueError:
        pass

    # Get statistics
    stats = metrics.get_stats()
    print(f"Total transactions: {stats['transaction_count']}")
    print(f"Committed: {stats['committed_count']}")
    print(f"Rolled back: {stats['rolled_back_count']}")
    print(f"Average duration: {stats['transaction_duration_avg']:.4f}s")
    print(f"Max duration: {stats['transaction_duration_max']:.4f}s")
    print(f"Read-only transactions: {stats['readonly_count']}")
    print(f"Error count: {stats['error_count']}")
    print(f"Error rate: {stats['error_rate']:.2%}")

    db.close()


def example_transaction_retry():
    """Demonstrate transaction retry."""
    print("\n=== Transaction Retry ===")

    db = connect("sqlite:///:memory:")
    db.create_table(
        "attempts", [column("id", "INTEGER"), column("attempt_num", "INTEGER")]
    ).collect()

    attempt_count = [0]

    def transaction_func():
        attempt_count[0] += 1
        with db.transaction() as txn:
            Records(
                _data=[{"id": 1, "attempt_num": attempt_count[0]}],
                _database=db,
            ).insert_into("attempts")
            print(f"Attempt {attempt_count[0]}")

    config = transaction_retry_config(max_attempts=3, initial_delay=0.1)

    retry_transaction(transaction_func, config=config)

    results = db.table("attempts").select().collect()
    print(f"Records inserted: {len(results)}")
    print(f"Total attempts: {attempt_count[0]}")

    db.close()


def example_concurrent_transactions():
    """Demonstrate concurrent transaction testing."""
    print("\n=== Concurrent Transaction Testing ===")

    db = connect("sqlite:///:memory:")
    db.create_table("concurrent", [column("id", "INTEGER"), column("value", "TEXT")]).collect()

    tester = ConcurrentTransactionTester(db, num_threads=4)

    def update_operation(db):
        with db.transaction() as txn:
            Records(_data=[{"id": 1, "value": "test"}], _database=db).insert_into("concurrent")
            return {"success": True}

    results = tester.run_concurrent_transactions(update_operation, num_transactions=10)

    stats = tester.get_statistics()
    print(f"Total transactions: {stats['total_transactions']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Success rate: {stats['success_rate']:.2%}")

    db.close()


async def example_async_transaction_decorator():
    """Demonstrate async transaction decorator."""
    print("\n=== Async Transaction Decorator ===")

    async_db = async_connect("sqlite+aiosqlite:///:memory:")
    await async_db.create_table(
        "async_users", [column("id", "INTEGER"), column("name", "TEXT")]
    ).collect()

    @transaction(async_db)
    async def create_user_async(name: str):
        await AsyncRecords(_data=[{"id": 1, "name": name}], _database=async_db).insert_into(
            "async_users"
        )
        print(f"Created async user: {name}")

    await create_user_async("AsyncUser")

    results = await async_db.table("async_users").select().collect()
    print(f"Async users in database: {len(results)}")

    await async_db.close()


def main():
    """Run all examples."""
    print("Moltres Transaction Utilities Examples")
    print("=" * 50)

    example_transaction_decorator()
    example_transaction_hooks()
    example_transaction_metrics()
    example_transaction_retry()
    example_concurrent_transactions()

    # Run async example if asyncio is available
    try:
        import asyncio

        asyncio.run(example_async_transaction_decorator())
    except ImportError:
        print("\nSkipping async examples (asyncio not available)")

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
