# Transaction Utilities Guide

Moltres provides a comprehensive set of utilities for working with database transactions, making it easier to manage transaction lifecycles, handle retries, monitor performance, and test concurrent scenarios.

## Overview

This guide covers:

- **Transaction Decorator** - Automatically wrap functions in transactions
- **Transaction Hooks** - Register callbacks for transaction lifecycle events
- **Transaction Metrics** - Monitor transaction performance and statistics
- **Transaction Retry** - Automatic retry on transient failures
- **Transaction Testing** - Utilities for testing concurrent scenarios

## Transaction Decorator

The `@transaction` decorator automatically wraps functions in database transactions, eliminating the need for manual `with db.transaction()` blocks.

### Basic Usage

```python
from moltres import connect
from moltres.utils.transaction_decorator import transaction
from moltres.io.records import Records
from moltres.table.schema import column

db = connect("sqlite:///:memory:")
db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

# Method 1: Provide database instance to decorator
@transaction(db)
def create_user(name: str):
    Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")

create_user("Alice")

# Method 2: Database as function parameter
@transaction
def create_user_with_db(db, name: str):
    Records(_data=[{"id": 2, "name": name}], _database=db).insert_into("users")

create_user_with_db(db, "Bob")
```

### Decorator Parameters

The decorator accepts the same parameters as `db.transaction()`:

```python
@transaction(db, readonly=True, isolation_level="SERIALIZABLE")
def read_heavy_operation():
    # ... read-only operations ...
    pass

@transaction(db, savepoint=True)
def nested_operation():
    # ... operations with savepoint support ...
    pass
```

### Async Support

The decorator works with async functions:

```python
from moltres import async_connect
from moltres.utils.transaction_decorator import transaction

async def example():
    async_db = async_connect("sqlite+aiosqlite:///:memory:")

    @transaction(async_db)
    async def create_user_async(name: str):
        await AsyncRecords(_data=[{"id": 1, "name": name}], _database=async_db).insert_into("users")

    await create_user_async("Async")

# Run with: await example()
```

### Automatic Rollback on Errors

If an exception occurs, the transaction is automatically rolled back:

```python
@transaction(db)
def risky_operation():
    Records(_data=[{"id": 1}], _database=db).insert_into("users")
    raise ValueError("Something went wrong")
    # Transaction automatically rolls back
```

## Transaction Hooks

Transaction hooks allow you to register callbacks that execute at specific points in the transaction lifecycle: begin, commit, or rollback.

### Registering Hooks

```python
from moltres.utils.transaction_hooks import register_transaction_hook

def on_begin(txn):
    print(f"Transaction started: {txn}")

def on_commit(txn):
    print(f"Transaction committed: {txn}")

def on_rollback(txn):
    print(f"Transaction rolled back: {txn}")

register_transaction_hook("begin", on_begin)
register_transaction_hook("commit", on_commit)
register_transaction_hook("rollback", on_rollback)

# Now all transactions will trigger these hooks
with db.transaction() as txn:
    # ... operations ...
    pass  # on_commit will be called
```

### Async Hooks

For async transactions, use async hooks:

```python
from moltres.utils.transaction_hooks import register_async_transaction_hook

async def on_commit_async(txn):
    print(f"Async transaction committed: {txn}")
    # Can perform async operations here

register_async_transaction_hook("commit", on_commit_async)
```

### Unregistering Hooks

```python
from moltres.utils.transaction_hooks import unregister_transaction_hook

# Unregister a specific hook
unregister_transaction_hook("commit", on_commit)

# Unregister all hooks for an event
unregister_transaction_hook("commit")
```

### Use Cases

- **Audit Logging**: Log all transaction commits/rollbacks
- **Performance Monitoring**: Track transaction start/end times
- **Cache Invalidation**: Invalidate caches on commit
- **Event Publishing**: Publish events after successful commits

## Transaction Metrics

Transaction metrics provide detailed statistics about transaction performance and behavior.

### Getting Metrics

```python
from moltres.utils.transaction_metrics import get_transaction_metrics, reset_transaction_metrics

# Reset metrics before starting
reset_transaction_metrics()

# Run some transactions
with db.transaction() as txn:
    # ... operations ...
    pass

with db.transaction(readonly=True) as txn:
    # ... read operations ...
    pass

# Get statistics
metrics = get_transaction_metrics()
stats = metrics.get_stats()

print(f"Total transactions: {stats['transaction_count']}")
print(f"Committed: {stats['committed_count']}")
print(f"Rolled back: {stats['rolled_back_count']}")
print(f"Average duration: {stats['transaction_duration_avg']:.3f}s")
print(f"Max duration: {stats['transaction_duration_max']:.3f}s")
print(f"Error rate: {stats['error_rate']:.2%}")
```

### Metrics Available

- `transaction_count` - Total number of transactions
- `transaction_duration_avg` - Average transaction duration
- `transaction_duration_min` - Minimum transaction duration
- `transaction_duration_max` - Maximum transaction duration
- `committed_count` - Number of committed transactions
- `rolled_back_count` - Number of rolled back transactions
- `commit_rate` - Percentage of transactions that committed
- `savepoint_count` - Number of transactions using savepoints
- `readonly_count` - Number of read-only transactions
- `isolation_levels` - Dictionary of isolation level usage
- `error_count` - Number of transactions that failed
- `error_rate` - Percentage of transactions that failed
- `errors_by_type` - Breakdown of errors by exception type

### Resetting Metrics

```python
from moltres.utils.transaction_metrics import reset_transaction_metrics

reset_transaction_metrics()  # Clear all metrics
```

## Transaction Retry

The transaction retry utilities automatically retry transactions on transient failures like deadlocks, lock timeouts, and connection errors.

### Basic Retry

```python
from moltres.utils.transaction_retry import retry_transaction, transaction_retry_config

def update_counter():
    with db.transaction() as txn:
        # ... operations that might deadlock ...
        Records(_data=[{"id": 1, "value": 100}], _database=db).insert_into("counters")

# Retry with default configuration (3 attempts, exponential backoff)
retry_transaction(update_counter)
```

### Custom Retry Configuration

```python
from moltres.utils.transaction_retry import retry_transaction, transaction_retry_config

config = transaction_retry_config(
    max_attempts=5,
    initial_delay=0.1,  # Start with 100ms delay
    max_delay=5.0,      # Cap at 5 seconds
    exponential_base=2.0,
    jitter=True
)

def risky_operation():
    with db.transaction() as txn:
        # ... operations ...
        pass

retry_transaction(risky_operation, config=config)
```

### Retry Callbacks

```python
def on_retry(error, attempt):
    print(f"Retry attempt {attempt} after error: {error}")

retry_transaction(risky_operation, config=config, on_retry=on_retry)
```

### Async Retry

```python
from moltres.utils.transaction_retry import retry_transaction_async

async def example():
    async def update_counter_async():
        async with async_db.transaction() as txn:
            # ... async operations ...
            pass

    await retry_transaction_async(update_counter_async)

# Run with: await example()
```

### Retryable Errors

The retry logic automatically detects and retries on:

- Deadlocks (PostgreSQL, MySQL, SQLite)
- Lock timeouts (PostgreSQL, MySQL)
- Connection errors
- Database locked errors (SQLite)
- Serialization failures

## Transaction Testing

Transaction testing utilities help test concurrent scenarios and isolation levels.

### Concurrent Transaction Testing

```python
from moltres.utils.transaction_testing import ConcurrentTransactionTester

tester = ConcurrentTransactionTester(db, num_threads=4)

def update_counter(db):
    with db.transaction() as txn:
        # ... update operations ...
        return {"success": True}

# Run 10 transactions concurrently
results = tester.run_concurrent_transactions(update_counter, num_transactions=10)

# Get statistics
stats = tester.get_statistics()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Failed: {stats['failed']}")
print(f"Error types: {stats['error_types']}")
```

### Async Concurrent Testing

```python
async def example():
    async def update_counter_async(db):
        async with db.transaction() as txn:
            # ... async operations ...
            return {"success": True}

    results = await tester.run_concurrent_transactions_async(
        update_counter_async,
        num_transactions=10
    )

# Run with: await example()
```

### Deadlock Simulation

```python
from moltres.utils.transaction_testing import DeadlockSimulator

simulator = DeadlockSimulator(db)

def txn1(db):
    with db.transaction() as txn:
        # Lock row A, then try to lock row B
        Records(_data=[{"id": 1, "value": "A"}], _database=db).insert_into("locks")
        time.sleep(0.1)

def txn2(db):
    with db.transaction() as txn:
        # Lock row B, then try to lock row A (potential deadlock)
        Records(_data=[{"id": 2, "value": "B"}], _database=db).insert_into("locks")
        time.sleep(0.1)

results = simulator.simulate_deadlock(txn1, txn2)
print(f"Deadlock detected: {results['deadlock_detected']}")
```

### Isolation Level Testing

```python
from moltres.utils.transaction_testing import test_isolation_level

def test_phantom_reads(db):
    # ... test for phantom reads ...
    return {"phantom_reads_detected": False}

results = test_isolation_level(db, "SERIALIZABLE", test_phantom_reads)
print(f"Test passed: {results['success']}")
```

## Best Practices

1. **Use Decorators for Clean Code**: The `@transaction` decorator reduces boilerplate and makes transaction boundaries explicit.

2. **Monitor Performance**: Regularly check transaction metrics to identify slow or problematic transactions.

3. **Handle Retries Appropriately**: Use transaction retry for operations that might encounter deadlocks or lock timeouts, but avoid retrying on non-retryable errors.

4. **Test Concurrent Scenarios**: Use testing utilities to verify your code handles concurrent access correctly.

5. **Use Hooks for Observability**: Register hooks for logging, monitoring, or audit purposes.

## Complete Example

```python
from moltres import connect
from moltres.utils.transaction_decorator import transaction
from moltres.utils.transaction_hooks import register_transaction_hook
from moltres.utils.transaction_metrics import get_transaction_metrics, reset_transaction_metrics
from moltres.utils.transaction_retry import retry_transaction, transaction_retry_config
from moltres.io.records import Records
from moltres.table.schema import column

# Setup
db = connect("sqlite:///:memory:")
db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

# Register hooks
def on_commit(txn):
    print("Transaction committed!")

register_transaction_hook("commit", on_commit)

# Reset metrics
reset_transaction_metrics()

# Use decorator with retry
config = transaction_retry_config(max_attempts=3)

@transaction(db)
def create_user(name: str):
    Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")

retry_transaction(create_user, config=config)

# Check metrics
metrics = get_transaction_metrics()
stats = metrics.get_stats()
print(f"Transactions: {stats['transaction_count']}")
print(f"Average duration: {stats['transaction_duration_avg']:.3f}s")
```

