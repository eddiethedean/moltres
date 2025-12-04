# Transaction Control Guide

Moltres provides advanced transaction control features including savepoints, nested transactions, isolation levels, read-only transactions, timeouts, state inspection, and row-level locking. This guide demonstrates how to use these features effectively.

## Basic Transactions

The simplest way to use transactions is with the context manager:

```python
from moltres import connect

db = connect("sqlite:///example.db")

with db.transaction() as txn:
    # All operations within this block are in a single transaction
    # Commit happens automatically on successful exit
    # Rollback happens automatically if an exception occurs
    pass
```

## Savepoints and Nested Transactions

Savepoints allow you to create nested transactions that can be rolled back independently without affecting the outer transaction.

### Creating Savepoints

```python
from moltres import connect
from moltres.io.records import Records
from moltres.table.schema import column

db = connect("sqlite:///example.db")
db.create_table("orders", [
    column("id", "INTEGER"),
    column("amount", "REAL"),
    column("status", "TEXT")
]).collect()

with db.transaction() as txn:
    # Initial insert
    Records(_data=[{"id": 1, "amount": 100.0}], _database=db).insert_into("orders")
    
    # Create a savepoint
    checkpoint = txn.savepoint("checkpoint1")
    
    # More operations
    Records(_data=[{"id": 2, "amount": 200.0}], _database=db).insert_into("orders")
    
    # Rollback to savepoint (only rolls back operations after the savepoint)
    txn.rollback_to_savepoint(checkpoint)
    
    # Only the first insert remains
```

### Nested Transactions with Savepoints

You can use nested transactions with the `savepoint=True` parameter:

```python
with db.transaction() as outer:
    Records(_data=[{"id": 1, "amount": 100.0}], _database=db).insert_into("orders")
    
    # Nested transaction using a savepoint
    with db.transaction(savepoint=True) as inner:
        Records(_data=[{"id": 2, "amount": 200.0}], _database=db).insert_into("orders")
        
        # If this fails, only the inner transaction is rolled back
        # The outer transaction continues
```

### Releasing Savepoints

Savepoints can be explicitly released:

```python
with db.transaction() as txn:
    sp = txn.savepoint("my_savepoint")
    
    # ... operations ...
    
    # Release the savepoint (commits it)
    txn.release_savepoint(sp)
```

## Isolation Levels

Transaction isolation levels control how concurrent transactions interact. Moltres supports:

- `READ UNCOMMITTED` - Lowest isolation, allows dirty reads
- `READ COMMITTED` - Prevents dirty reads (default for most databases)
- `REPEATABLE READ` - Prevents non-repeatable reads
- `SERIALIZABLE` - Highest isolation, prevents all concurrency issues

```python
# Use SERIALIZABLE for critical operations
with db.transaction(isolation_level="SERIALIZABLE") as txn:
    # Critical operations that require highest isolation
    pass
```

**Note:** SQLite has limited isolation level support. PostgreSQL and MySQL fully support all isolation levels.

## Read-Only Transactions

Read-only transactions prevent any writes and can be optimized by the database:

```python
with db.transaction(readonly=True) as txn:
    # Only read operations allowed
    results = db.table("users").select().collect()
    # Writes would fail in read-only mode
```

**Note:** SQLite and MySQL don't support read-only transactions. PostgreSQL does.

## Transaction Timeouts

Set a timeout for transactions to prevent long-running operations:

```python
# 30 second timeout
with db.transaction(timeout=30.0) as txn:
    # Operations must complete within 30 seconds
    pass
```

Timeout behavior is database-specific:
- **PostgreSQL**: Sets `statement_timeout` (in milliseconds)
- **MySQL**: Sets `innodb_lock_wait_timeout` (in seconds)
- **SQLite**: Not supported

## Transaction State Inspection

Check if you're currently in a transaction and get its status:

```python
# Check if in transaction
if db.is_in_transaction():
    status = db.get_transaction_status()
    print(f"Isolation: {status['isolation_level']}")
    print(f"Readonly: {status['readonly']}")
    print(f"Savepoints: {status['savepoints']}")
```

The status dictionary includes:
- `readonly`: Whether the transaction is read-only
- `isolation_level`: Transaction isolation level (if set)
- `timeout`: Transaction timeout in seconds (if set)
- `savepoints`: List of active savepoint names

## Row-Level Locking

Row-level locking prevents concurrent modifications to specific rows.

### FOR UPDATE

Locks rows for exclusive update:

```python
from moltres import col

with db.transaction() as txn:
    df = db.table("orders").select().where(col("status") == "pending")
    locked_df = df.select_for_update()
    results = locked_df.collect()
    
    # Rows are now locked for update
    # Other transactions cannot modify them until this transaction commits
```

### FOR SHARE

Locks rows for shared (read) access:

```python
with db.transaction() as txn:
    df = db.table("products").select().where(col("id") == 1)
    locked_df = df.select_for_share()
    results = locked_df.collect()
    
    # Rows are locked for read
    # Other transactions can read but not modify
```

### NOWAIT and SKIP LOCKED

Don't wait for locks:

```python
# Raise error immediately if rows are locked
df.select_for_update(nowait=True)

# Skip locked rows instead of waiting
df.select_for_update(skip_locked=True)
```

**Note:** `NOWAIT` and `SKIP LOCKED` require PostgreSQL or MySQL 8.0+. SQLite doesn't support these options.

## Async Transactions

All transaction features work with async databases:

```python
from moltres import async_connect

async def example():
    db = await async_connect("sqlite+aiosqlite:///example.db")

    async with db.transaction(savepoint=True, readonly=False) as txn:
        # Async operations
        table_handle = await db.table("users")
        df = table_handle.select()
        results = await df.collect()

    await db.close()

# Run with: await example()
```

## Transaction Methods

### Transaction.savepoint(name=None)

Create a savepoint within the transaction. Returns the savepoint name.

### Transaction.rollback_to_savepoint(name)

Rollback to a specific savepoint.

### Transaction.release_savepoint(name)

Release a savepoint (commits operations since the savepoint).

### Transaction.is_readonly()

Check if the transaction is read-only.

### Transaction.isolation_level()

Get the transaction isolation level.

### Transaction.is_active()

Check if the transaction is still active (not committed or rolled back).

## Best Practices

1. **Always use context managers** - They ensure proper cleanup:

```python
# Good
with db.transaction() as txn:
    # operations
    pass

# Bad - manual management is error-prone
txn = db.transaction()
# ... what if an exception occurs?
```

2. **Use savepoints for error recovery** - Allow partial rollbacks:

```python
with db.transaction() as txn:
    # Critical operation
    process_payment()
    
    # Create checkpoint
    checkpoint = txn.savepoint()
    
    try:
        # Non-critical operation
        send_notification()
    except Exception:
        # Rollback only the notification, keep payment
        txn.rollback_to_savepoint(checkpoint)
```

3. **Use isolation levels appropriately** - Balance consistency vs performance:

```python
# High isolation for critical operations
with db.transaction(isolation_level="SERIALIZABLE"):
    transfer_money(from_account, to_account, amount)

# Lower isolation for read-heavy operations
with db.transaction(isolation_level="READ COMMITTED"):
    generate_report()
```

4. **Use row-level locking for concurrent access** - Prevent race conditions:

```python
# Process pending orders
with db.transaction() as txn:
    # Lock and process
    df = db.table("orders").select().where(col("status") == "pending")
    locked_df = df.select_for_update(nowait=True)
    orders = locked_df.collect()
    
    for order in orders:
        process_order(order)
```

## Limitations

- **SQLite**: Limited isolation level support, no read-only transactions, no NOWAIT/SKIP LOCKED
- **MySQL**: No read-only transactions (MySQL 8.0+ supports SKIP LOCKED)
- **PostgreSQL**: Full feature support

Check dialect capabilities:

```python
dialect = db.dialect
print(f"Supports savepoints: {dialect.supports_savepoints}")
print(f"Supports isolation levels: {dialect.supports_isolation_levels}")
print(f"Supports read-only: {dialect.supports_read_only_transactions}")
print(f"Supports row locking: {dialect.supports_row_locking}")
```

## Error Handling

Transactions automatically rollback on exceptions:

```python
try:
    with db.transaction() as txn:
        # Operations
        risky_operation()
except Exception as e:
    # Transaction already rolled back
    print(f"Transaction failed: {e}")
```

For savepoints, only the inner transaction rolls back:

```python
with db.transaction() as outer:
    critical_operation()
    
    try:
        with db.transaction(savepoint=True) as inner:
            risky_operation()
    except Exception:
        # Only inner transaction rolled back
        # Outer transaction continues
        pass
```

