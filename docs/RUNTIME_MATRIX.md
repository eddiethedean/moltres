# Runtime Support Matrix

This document details the supported Python versions, database backends, and driver combinations for Moltres.

## Python Versions

Moltres supports Python 3.9 and above:

- **Python 3.9** - Minimum supported version
- **Python 3.10** - Fully supported
- **Python 3.11** - Fully supported
- **Python 3.12** - Fully supported

### Python Version Support Policy

- Minimum version is determined by dependencies (SQLAlchemy 2.0+ requires Python 3.9+)
- New Python versions are tested and added within 3 months of release
- Deprecated Python versions are supported for at least 12 months after EOL

## Database Backends

### SQLite

- **Driver**: Built-in (`sqlite3` module)
- **Async Driver**: `aiosqlite>=0.19.0` (optional dependency)
- **SQLAlchemy Dialect**: `sqlite`
- **Status**: Fully supported, tested in CI
- **Notes**: 
  - Default database for development and testing
  - No additional setup required
  - File-based, no server required

### PostgreSQL

- **Sync Driver**: `psycopg2-binary>=2.9.0` (via SQLAlchemy)
- **Async Driver**: `asyncpg>=0.29.0` (optional dependency)
- **SQLAlchemy Dialect**: `postgresql` (sync), `postgresql+asyncpg` (async)
- **Status**: Fully supported, tested in CI
- **Minimum PostgreSQL Version**: 9.6+
- **Recommended PostgreSQL Version**: 12+
- **Notes**:
  - Full feature support including JSONB, arrays, window functions
  - DSN `options` parameter (e.g., `?options=-csearch_path=schema`) automatically translated to asyncpg `server_settings`
  - Staging tables use regular tables (not TEMPORARY) for connection pool compatibility

### MySQL / MariaDB

- **Sync Driver**: `pymysql>=1.0.0` (via SQLAlchemy)
- **Async Driver**: `aiomysql>=0.2.0` (optional dependency)
- **SQLAlchemy Dialect**: `mysql` (sync), `mysql+aiomysql` (async)
- **Status**: Fully supported, tested in CI
- **Minimum MySQL Version**: 5.7+
- **Minimum MariaDB Version**: 10.2+
- **Recommended Version**: MySQL 8.0+ or MariaDB 10.5+
- **Notes**:
  - Full feature support including JSON, window functions (MySQL 8.0+)
  - Staging tables use regular tables (not TEMPORARY) for connection pool compatibility
  - Each test gets isolated database via `CREATE DATABASE test_<uuid>`

### Other SQLAlchemy-Supported Databases

Moltres uses SQLAlchemy's dialect system, so any database with a SQLAlchemy driver should work with basic functionality:

- **Oracle** - Via `cx_Oracle` or `oracledb` (not tested in CI)
- **SQL Server** - Via `pyodbc` or `pymssql` (not tested in CI)
- **BigQuery** - Via `sqlalchemy-bigquery` (not tested in CI)
- **Snowflake** - Via `snowflake-sqlalchemy` (not tested in CI)
- **Redshift** - Via `sqlalchemy-redshift` (not tested in CI)
- **DuckDB** - Via `duckdb-engine` (not tested in CI)

**Note**: Advanced features (JSON, arrays, window functions) may not be available on all backends.

## Operating Systems

Tested and supported on:

- **Linux** (Ubuntu 20.04+, tested via GitHub Actions `ubuntu-latest`)
- **macOS** (10.15+, tested via GitHub Actions `macos-latest`)
- **Windows** (Windows Server 2019+, tested via GitHub Actions `windows-latest`)

## Driver Version Requirements

### Core Dependencies

- **SQLAlchemy**: `>=2.0` (required)
- **typing-extensions**: `>=4.5` (required)

### Optional Dependencies

#### Async Support
- **aiofiles**: `>=23.0` (for async file I/O)
- **greenlet**: `>=3.0.0` (required for async SQLAlchemy)

#### Database-Specific Async Drivers
- **asyncpg**: `>=0.29.0` (PostgreSQL async)
- **aiomysql**: `>=0.2.0` (MySQL async)
- **aiosqlite**: `>=0.19.0` (SQLite async)

#### Data Format Support
- **pandas**: `>=2.1` (optional, for pandas DataFrame results)
- **polars**: `>=1.0` (optional, for Polars DataFrame results)
- **pyarrow**: `>=10.0` (optional, for Parquet file support)

#### Testing Dependencies
- **pytest**: `>=8.0`
- **pytest-asyncio**: `>=0.21.0`
- **testing.postgresql**: `>=1.3.0` (for PostgreSQL test fixtures)
- **testing.mysqld**: `>=1.4.0` (for MySQL test fixtures)
- **psycopg2-binary**: `>=2.9.0` (for PostgreSQL tests)
- **pymysql**: `>=1.0.0` (for MySQL tests)

## CI Testing Matrix

The following combinations are tested in CI:

### Python × OS Matrix

- Python 3.9 × Ubuntu, macOS, Windows
- Python 3.10 × Ubuntu, macOS, Windows
- Python 3.11 × Ubuntu, macOS, Windows
- Python 3.12 × Ubuntu, macOS, Windows

### Database Backend Testing

- **SQLite**: Tested on all Python/OS combinations (always available)
- **PostgreSQL**: Tested when `testing.postgresql` is available (Linux/macOS primarily)
- **MySQL**: Tested when `testing.mysqld` is available (Linux/macOS primarily)

**Note**: PostgreSQL and MySQL tests are marked `continue-on-error: true` in CI to allow tests to run even if database services are unavailable.

## Installation

### Minimal Installation

```bash
pip install moltres
```

This installs only core dependencies (SQLAlchemy, typing-extensions).

### With Async Support

```bash
pip install moltres[async]
```

### With Database-Specific Async Drivers

```bash
# PostgreSQL async
pip install moltres[async-postgresql]

# MySQL async
pip install moltres[async-mysql]

# SQLite async
pip install moltres[async-sqlite]
```

### With Data Format Support

```bash
# Pandas support
pip install moltres[pandas]

# Polars support
pip install moltres[polars]
```

### Development Installation

```bash
pip install -e ".[dev]"
```

This includes all optional dependencies for development and testing.

## Version Compatibility Notes

### SQLAlchemy 2.0+

Moltres requires SQLAlchemy 2.0+ which introduced:
- New async engine API
- Improved type hints
- Better connection pooling

### Python 3.9+

Python 3.9 is the minimum version due to:
- Type hint improvements (PEP 585, PEP 604)
- SQLAlchemy 2.0 requirements
- Modern async/await support

## Known Limitations

1. **Temporary Tables**: PostgreSQL and MySQL use regular tables (not TEMPORARY) for staging operations to ensure visibility across connection pool connections. These tables are automatically cleaned up when the Database instance is closed.

2. **Database-Specific Features**: Some advanced features (JSONB, arrays, window functions) are only available on specific backends. Check dialect documentation for feature availability.

3. **Driver Compatibility**: Some database drivers may have version-specific quirks. Always use the minimum versions specified above.

## Support and Reporting Issues

If you encounter issues with a specific Python version, database backend, or driver combination:

1. Check this document to ensure your combination is supported
2. Verify you're using minimum required versions
3. Open an issue on GitHub with:
   - Python version (`python --version`)
   - Database type and version
   - Driver version
   - Error traceback
   - Minimal reproduction code

