# Runtime & Platform Support

This document captures the currently supported Python versions, operating systems, database backends, and driver requirements for running Moltres in production. The matrix mirrors our CI coverage so you can rely on it when planning deployments.

## Python & Operating Systems

| Python | Ubuntu (22.04) | macOS (13/14) | Windows (2025) |
|--------|----------------|---------------|----------------|
| 3.9    | ✅ Full test suite | ✅ Full test suite | ✅ Full test suite |
| 3.10   | ✅ Full test suite | ✅ Full test suite | ✅ Full test suite |
| 3.11   | ✅ Full test suite | ✅ Full test suite | ✅ Full test suite |
| 3.12   | ✅ Full test suite | ✅ Full test suite | ✅ Full test suite |

- The CI workflow (`.github/workflows/ci.yml`) runs the unit test suite, linting, typing, documentation validation, and smoke DB tests across every entry in the table above.
- We require **Python 3.9+** as defined in `pyproject.toml`. Earlier Python versions are not tested or supported.

## Database Backends

| Backend          | Driver(s)                            | Notes                                                                 |
|------------------|--------------------------------------|-----------------------------------------------------------------------|
| SQLite           | Built-in                             | Default backend for development and quick smoke tests.                 |
| PostgreSQL       | `psycopg2-binary` (sync), `asyncpg` (async) | Postgres fixtures rely on `initdb` and the `testing.postgresql` package. |
| MySQL/MariaDB    | `pymysql` (sync), `aiomysql` (async) | Requires `mysqld` tooling plus the `testing.mysqld` package.           |
| Async variants   | `aiofiles`, `aiosqlite`, `greenlet`  | Install via extras: `moltres[async]`, `moltres[async-postgresql]`, etc. |

### Fixture expectations

- **PostgreSQL**: `testing.postgresql` spins up ephemeral clusters. Ensure `initdb` and `postgres` binaries are installed (the workflow skips gracefully if missing).
- **MySQL**: `testing.mysqld` shells out to `mysqld`/`mysql_install_db`. Install a server locally or rely on containerized MySQL.
- **SQLite**: No extra dependencies; used for the majority of CI coverage.

## Optional Dependencies

Use extras to install the relevant driver stack:

- `pip install "moltres[async]"` – core async support (file IO).
- `pip install "moltres[async-postgresql]"` – async Postgres (asyncpg + core async extras).
- `pip install "moltres[async-mysql]"` – async MySQL (aiomysql + core async extras).
- `pip install "moltres[async-sqlite]"` – async SQLite (aiosqlite + core async extras).
- `pip install "moltres[pandas]"`, `moltres[polars]` – DataFrame integrations for export paths.

## Verifying Your Environment

1. Install the required extras for your database backend.
2. Run `pytest -p pytest_asyncio -m "not postgres and not mysql"` to validate the core suite.
3. Enable the Postgres/MySQL markers once their binaries are installed:
   ```bash
   pytest -m postgres
   pytest -m mysql
   ```
4. For async clusters, run `pytest -m asyncio` to ensure drivers, event loop policies, and staging table cleanup behave as expected.

If your deployment targets a platform that is not listed above, add it to the CI matrix first so we can formally support it.

