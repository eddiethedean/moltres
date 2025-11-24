# Test Harness Reference

Moltres ships with database-aware pytest fixtures (our “test harnesses”) that spin
up isolated database instances, wire in async drivers, and keep staging tables tidy.
This guide shows how to plug into those harnesses when writing new tests and how to
avoid table-name collisions with the built-in unique-table utilities.

## When to Use Which Harness

| Fixture | Scope | Marks Needed | Purpose |
| --- | --- | --- | --- |
| `sqlite_db` | function | *(none)* | Lightweight default database stored in a tmp path. |
| `postgresql_connection` | function | `@pytest.mark.postgres` | Sync PostgreSQL handle with per-test schema isolation. |
| `postgresql_async_connection` | function | `@pytest.mark.postgres` + `@pytest.mark.asyncio` | Async PostgreSQL handle that automatically upgrades the DSN to `postgresql+asyncpg://` and propagates `?options=-csearch_path=...`. |
| `mysql_connection` | function | `@pytest.mark.mysql` | Sync MySQL handle with a per-test database. |
| `mysql_async_connection` | function | `@pytest.mark.mysql` + `@pytest.mark.asyncio` | Async MySQL handle using `aiomysql`. |
| `sample_table`, `create_sample_table`, `seed_customers_orders`, `seed_customers_orders_async` | helper | varies | Pre-populate standard fixtures for CRUD/join scenarios regardless of backend. |

All of these live in `tests/conftest.py`. Use the fixture name as a test argument;
pytest takes care of bootstrapping and cleanup.

```python
import pytest
from moltres import col

@pytest.mark.postgres
def test_array_ops(postgresql_connection):
    db = postgresql_connection
    # Safe to use simple table names because every test gets its own schema.
    db.create_table("items", [...]).collect()
    ...

@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_percentile(postgresql_async_connection):
    db = postgresql_async_connection
    await db.create_table("metrics", [...])
    ...
```

## Unique Table Management

The harnesses are designed so you rarely need to invent ad‑hoc table names:

1. **Per-test schema/database isolation**  
   - `postgresql_schema` fixture creates a brand-new schema via `CREATE SCHEMA test_<uuid>` and injects it into the DSN’s `search_path`.  
   - `mysql_database` fixture creates a unique database per test.  
   - SQLite tests write into a temporary file under `tmp_path`.

2. **Automatic staging names inside Moltres APIs**  
   `db.createDataFrame()` and writer operations call `generate_unique_table_name()` (see `src/moltres/dataframe/create_dataframe.py`) to stage data in tables like `__moltres_df_<hash>__`. These are registered as “ephemeral” tables and dropped when the `Database`/`AsyncDatabase` is closed, so tests can safely chain multiple staging steps.

3. **Manual unique names when needed**  
If you are issuing raw SQL or need multiple temp tables in a single test, import the helper:

```python
from moltres.dataframe.create_dataframe import generate_unique_table_name

table_name = generate_unique_table_name()
db.create_table(table_name, [...]).collect()
```

Because PostgreSQL schemas and MySQL databases are already unique per test, most authors still prefer human-readable names. Reserve `generate_unique_table_name()` for cases where tests dynamically create many tables or work outside the provided schema/database fixtures.

## Patterns for New Tests

### 1. Sync tests against multiple backends

```python
import pytest
from moltres.table.schema import column

@pytest.mark.multidb
@pytest.mark.parametrize("db_fixture", ["sqlite_db", "postgresql_connection", "mysql_connection"])
def test_merge_round_trip(request, db_fixture):
    db = request.getfixturevalue(db_fixture)
    table = "orders"
    db.create_table(table, [column("id", "INTEGER", primary_key=True)]).collect()
    ...
```

*Tips:*
- Always parametrize on fixture names instead of manually creating engines. This ensures each backend benefits from its isolation harness.
- Keep table names simple (`"orders"`, `"customers"`)—the schema/database isolation prevents collisions.

### 2. Async tests

```python
import pytest
from moltres.table.schema import column

@pytest.mark.asyncio
@pytest.mark.postgres
async def test_async_writer_handles_streams(postgresql_async_connection):
    db = postgresql_async_connection
    await db.create_table("streaming", [column("id", "INTEGER", primary_key=True)])
    ...
```

*Tips:*
- Combine `@pytest.mark.asyncio` with the relevant backend marker.
- Use the async helper functions (`seed_customers_orders_async`) if you need sample data.
- Async fixtures already translate the DSN to driver-specific forms (`postgresql+asyncpg`, `mysql+aiomysql`). Do **not** hard-code async drivers in tests.

### 3. File / staging heavy tests

When a test needs multiple temporary tables or writes intermediate results to disk:

1. Use `tmp_path`/`tmp_path_factory` to hold files.
2. Use `generate_unique_table_name()` if you have to create more than one staging table manually.
3. Prefer `db.createDataFrame([...])` for quick staging—it auto-manages table names and cleanup.

```python
from moltres.dataframe.create_dataframe import generate_unique_table_name

table_name = generate_unique_table_name()
db.create_table(table_name, schema).collect()
```

## Checklist Before Submitting a New Test

- [ ] Pick the right fixture(s) and pytest markers.
- [ ] Let the harness manage schemas/databases; avoid hard-coding database names.
- [ ] Use helper functions (`create_sample_table`, `seed_customers_orders`, async variants) whenever possible.
- [ ] For raw SQL or custom staging, call `generate_unique_table_name()` instead of `f"tmp_{uuid.uuid4().hex}"`.
- [ ] Async tests await `.collect()`/`.write` operations and close resources by relying on the fixture teardown.

Refer back to this file whenever you add or review tests so the harness remains the single source of truth for database setup and cleanup.

