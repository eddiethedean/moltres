# Testing Guide

Moltres includes comprehensive test coverage across multiple database backends.

## Test Structure

Tests are organized in the `tests/` directory:

- `tests/dataframe/` - DataFrame operation tests
- `tests/expressions/` - Expression and function tests
- `tests/table/` - Table and CRUD operation tests
- `tests/utils/` - Utility function tests

## Database Support

Moltres tests run against multiple database backends:

### SQLite (Default)

SQLite tests run by default and don't require any additional setup. All tests use SQLite unless marked otherwise.

### PostgreSQL

PostgreSQL tests use the `testing.postgresql` library to create ephemeral PostgreSQL instances. These tests are marked with `@pytest.mark.postgres`.

To run PostgreSQL tests:

```bash
pytest -m postgres
```

**Note**: PostgreSQL tests require the `testing.postgresql` package and a PostgreSQL installation. They will be skipped if these are not available.

### MySQL

MySQL tests use the `testing.mysqld` library to create ephemeral MySQL instances. These tests are marked with `@pytest.mark.mysql`.

To run MySQL tests:

```bash
pytest -m mysql
```

**Note**: MySQL tests require the `testing.mysqld` package and a MySQL installation. They will be skipped if these are not available.

## Test Markers

The following pytest markers are available:

- `@pytest.mark.postgres` - Tests that require PostgreSQL
- `@pytest.mark.mysql` - Tests that require MySQL
- `@pytest.mark.multidb` - Tests that run against multiple databases
- `@pytest.mark.asyncio` - Async tests

## Running Tests

### Run all tests (SQLite only)

```bash
pytest
```

### Run specific test markers

```bash
# PostgreSQL tests only
pytest -m postgres

# MySQL tests only
pytest -m mysql

# Multi-database tests
pytest -m multidb

# Exclude database-specific tests
pytest -m "not postgres and not mysql"
```

### Run with coverage

```bash
pytest --cov=src/moltres --cov-report=html
```

## Test Fixtures

### Database Fixtures

- `sqlite_db` - SQLite database connection (default)
- `postgresql_connection` - PostgreSQL database connection
- `mysql_connection` - MySQL database connection
- `parametrize_db` - Parametrized fixture that runs tests against all databases

### Helper Functions

- `create_sample_table(db, table_name)` - Creates a sample users table
- `seed_customers_orders(db)` - Seeds customers and orders tables for join tests

## Writing Tests

Need a refresher on the database fixtures or unique table helpers? See
[`docs/TEST_HARNESSES.md`](./TEST_HARNESSES.md) for the full harness reference.

### Basic Test Example

```python
def test_basic_select(sqlite_db):
    """Test basic SELECT operation."""
    from moltres.table.schema import column
    
    sqlite_db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    )
    
    table = sqlite_db.table("users")
    table.insert([{"id": 1, "name": "Alice"}])
    
    result = table.select().collect()
    assert len(result) == 1
    assert result[0]["name"] == "Alice"
```

### Multi-Database Test Example

```python
@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_select_multidb(request, db_fixture):
    """Test SELECT across all databases."""
    db = request.getfixturevalue(db_fixture)
    # ... test code ...
```

### PostgreSQL-Specific Test Example

```python
@pytest.mark.postgres
def test_jsonb_type(postgresql_connection):
    """Test PostgreSQL JSONB type."""
    from moltres.table.schema import json
    
    db = postgresql_connection
    db.create_table("test_jsonb", [json("data", jsonb=True)])
    # ... test code ...
```

## CI/CD Integration

The CI workflow runs:

1. All SQLite tests (default)
2. PostgreSQL tests (if available, with `continue-on-error`)
3. MySQL tests (if available, with `continue-on-error`)

This ensures that:
- Core functionality is always tested
- Database-specific features are tested when possible
- CI doesn't fail if database servers aren't available

## Troubleshooting

### PostgreSQL tests are skipped

- Ensure `testing.postgresql` is installed: `pip install testing.postgresql`
- Ensure PostgreSQL is installed and accessible
- Check that the `postgres` user has necessary permissions

### MySQL tests are skipped

- Ensure `testing.mysqld` is installed: `pip install testing.mysqld`
- Ensure MySQL is installed and accessible
- Check that the `mysql` user has necessary permissions

### Tests fail with connection errors

- Verify database servers are running
- Check connection strings in test fixtures
- Ensure test databases can be created/dropped

