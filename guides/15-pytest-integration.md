# Pytest Integration Guide

This guide shows you how to use Moltres with Pytest for comprehensive testing of database operations and DataFrame queries.

## Installation

The Pytest integration is included with Moltres. Pytest itself should already be installed:

```bash
pip install pytest pytest-asyncio
```

## Quick Start

### Basic Database Fixture

The `moltres_db` fixture provides an isolated test database for each test:

```python
from moltres.integrations.pytest import moltres_db

def test_user_operations(moltres_db):
    db = moltres_db
    
    # Create a table
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    ).collect()
    
    # Query and verify
    df = db.table("users").select()
    results = df.collect()
    assert len(results) == 0
```

### Async Database Fixture

For async tests, use `moltres_async_db`:

```python
import pytest
from moltres.integrations.pytest import moltres_async_db

@pytest.mark.asyncio
async def test_async_operations(moltres_async_db):
    db = await moltres_async_db
    
    await db.create_table("users", [...])
    table = await db.table("users")
    df = table.select()
    results = await df.collect()
    assert len(results) == 0
```

## Features

### Database Fixtures

#### `moltres_db` Fixture

Creates an isolated test database for each test function. By default, uses SQLite.

**Features:**
- Automatic cleanup after each test
- Isolated database per test
- Support for multiple database backends via markers

**Example:**
```python
def test_isolated_db(moltres_db):
    db = moltres_db
    # Each test gets its own database
    db.create_table("test", [...]).collect()
```

#### `moltres_async_db` Fixture

Async version of the database fixture for async tests.

**Example:**
```python
@pytest.mark.asyncio
async def test_async(moltres_async_db):
    db = await moltres_async_db
    # Use async operations
```

### Test Data Fixtures

#### `test_data` Fixture

Loads test data from CSV and JSON files in a `test_data/` directory.

**Setup:**
Create a `test_data/` directory next to your test file:

```text
tests/
  test_data/
    users.csv
    orders.json
```

**Example:**
```python
def test_with_data(moltres_db, test_data):
    db = moltres_db
    
    # Load data from files
    users = test_data["users"]  # From users.csv
    
    # Create table and insert
    db.create_table("users", [...]).collect()
    Records(_data=users, _database=db).insert_into("users")
```

#### `create_test_df()` Helper

Creates a Moltres DataFrame from test data:

```python
from moltres.integrations.pytest import create_test_df

data = [{"id": 1, "name": "Alice"}]
df = create_test_df(data, database=db)
results = df.collect()
```

### Custom Assertions

#### `assert_dataframe_equal()`

Compare two DataFrames for equality:

```python
from moltres.integrations.pytest import assert_dataframe_equal

df1 = db.table("source").select()
df2 = db.table("target").select()

# Compare schema and data
assert_dataframe_equal(df1, df2)

# Ignore row order
assert_dataframe_equal(df1, df2, ignore_order=True)

# Skip schema check
assert_dataframe_equal(df1, df2, check_schema=False)
```

#### `assert_schema_equal()`

Compare DataFrame schemas:

```python
from moltres.integrations.pytest import assert_schema_equal

assert_schema_equal(df1.schema, expected_schema)
```

#### `assert_query_results()`

Validate query results:

```python
from moltres.integrations.pytest import assert_query_results

df = db.table("users").select()

# Exact count
assert_query_results(df, expected_count=10)

# Range
assert_query_results(df, min_count=5, max_count=20)

# Expected rows
assert_query_results(df, expected_rows=[{"id": 1, "name": "Alice"}])
```

### Query Logging

#### `query_logger` Fixture

Track all SQL queries executed during tests:

```python
from moltres.integrations.pytest_plugin import query_logger

def test_query_debugging(moltres_db, query_logger):
    db = moltres_db
    df = db.table("users").select()
    df.collect()
    
    # Check queries
    assert query_logger.count == 1
    assert "SELECT" in query_logger.queries[0]
    
    # Check performance
    assert query_logger.get_average_time() < 0.1
```

### Database-Specific Tests

Use markers to specify database backends:

```python
import pytest

@pytest.mark.moltres_db("postgresql")
def test_postgresql_feature(moltres_db):
    # Only runs with PostgreSQL
    # Configure via: TEST_POSTGRES_HOST, TEST_POSTGRES_PORT, etc.
    pass

@pytest.mark.moltres_db("mysql")
def test_mysql_feature(moltres_db):
    # Only runs with MySQL
    pass
```

### Performance Tests

Mark tests as performance tests:

```python
@pytest.mark.moltres_performance
def test_query_performance(moltres_db):
    import time
    start = time.time()
    # Run query
    elapsed = time.time() - start
    assert elapsed < 1.0
```

## Configuration

### Environment Variables

For PostgreSQL tests:
- `TEST_POSTGRES_HOST` (default: localhost)
- `TEST_POSTGRES_PORT` (default: 5432)
- `TEST_POSTGRES_USER` (default: postgres)
- `TEST_POSTGRES_PASSWORD` (default: "")
- `TEST_POSTGRES_DB` (default: test_moltres)

For MySQL tests:
- `TEST_MYSQL_HOST` (default: localhost)
- `TEST_MYSQL_PORT` (default: 3306)
- `TEST_MYSQL_USER` (default: root)
- `TEST_MYSQL_PASSWORD` (default: "")
- `TEST_MYSQL_DB` (default: test_moltres)

### Pytest Markers

Markers are registered in `pyproject.toml`:

```toml
markers = [
    "moltres_db(db_type): marks tests to use specific database backend",
    "moltres_performance: marks tests as performance tests with timing",
]
```

## Best Practices

1. **Use fixtures for database setup**: Always use `moltres_db` or `moltres_async_db` for isolated tests
2. **Load test data from files**: Use the `test_data` fixture for reusable test data
3. **Use custom assertions**: Use `assert_dataframe_equal()` for clear test failures
4. **Log queries for debugging**: Use `query_logger` to track SQL execution
5. **Mark database-specific tests**: Use markers to skip tests when databases aren't available

## Examples

See `docs/examples/26_pytest_integration.py` for comprehensive examples.

## See Also

- [Pytest Documentation](https://docs.pytest.org/)
- [Moltres DataFrame API](../README.md)
- [Testing Guide](../docs/TESTING.md)

