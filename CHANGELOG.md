# Changelog

All notable changes to Moltres will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2025-01-21

### Added
- **Compressed File Reading** - Automatic detection and support for gzip, bz2, and xz compression formats
  - Support for compressed CSV, JSON, JSONL, and text files
  - Works with both synchronous and asynchronous file readers
  - Compression detection from file extension (`.gz`, `.bz2`, `.xz`) or explicit `compression` option
  - New `compression.py` module with `open_compressed()` and `read_compressed_async()` utilities
- **Array/JSON Functions** - New functions for working with JSON and array data:
  - `json_extract(column, path)` - Extract values from JSON columns (SQLite JSON1, PostgreSQL JSONB, MySQL JSON)
  - `array(elements)` - Create array literals
  - `array_length(column)` - Get array length
  - `array_contains(column, value)` - Check if array contains value
  - `array_position(column, value)` - Find position of value in array
  - All functions include dialect-specific SQL compilation for optimal database support
- **Collect Aggregations** - New aggregation functions for collecting values into arrays:
  - `collect_list(column)` - Collect values into a list/array (uses `ARRAY_AGG` in PostgreSQL, `json_group_array` in SQLite, `GROUP_CONCAT` in MySQL)
  - `collect_set(column)` - Collect distinct values into a set/array
- **Semi-Join and Anti-Join Operations** - New DataFrame methods for efficient filtering:
  - `semi_join(other, on=[...])` - Filter rows that have matches in another DataFrame (compiles to INNER JOIN with DISTINCT)
  - `anti_join(other, on=[...])` - Filter rows that don't have matches in another DataFrame (compiles to LEFT JOIN with IS NULL)
  - Both methods support column-based joins and custom conditions
- **MERGE/UPSERT Operations** - New table method for upsert operations:
  - `table.merge(source_df, on=[...], when_matched={...}, when_not_matched={...})` - Merge/upsert rows with conflict resolution
  - Dialect-specific SQL compilation:
    - SQLite: `INSERT ... ON CONFLICT DO UPDATE`
    - PostgreSQL: Full `MERGE` statement
    - MySQL: `INSERT ... ON DUPLICATE KEY UPDATE` (planned)
  - Supports both update-on-match and insert-on-no-match scenarios
- **Comprehensive Test Coverage** - Added execution tests for all new features:
  - Tests for compressed file reading (gzip, bz2, xz) in sync and async modes
  - Tests for array/JSON functions with SQLite limitations handled
  - Tests for collect aggregations
  - Tests for semi-join and anti-join operations
  - Tests for MERGE/UPSERT operations

### Changed
- Improved join compilation to handle column qualification properly in semi-join and anti-join operations
- Enhanced type safety with proper type annotations for new functions and methods

### Fixed
- Fixed type checking issues in compression utilities
- Fixed column qualification in semi-join and anti-join to avoid ambiguous column errors

## [0.4.0] - 2025-01-21

### Added
- **Strict Type Checking** - Enabled mypy strict mode with comprehensive type annotations across the entire codebase
- **Type Stubs for PyArrow** - Custom type stubs (`stubs/pyarrow/`) to provide type information for pyarrow library
- **PEP 561 Compliance** - Added `py.typed` marker file to signal that the package is fully typed
- **Mypy Configuration** - Comprehensive mypy configuration in `pyproject.toml` with strict checking enabled

### Changed
- **Type Safety:** All functions and methods now have complete type annotations
- **Type Safety:** Removed all unused type ignore comments and fixed type inference issues
- **Type Safety:** Improved type hints for async operations and Records classes

### Fixed
- Fixed `AsyncRecords` import issue in `async_mutations.py` for proper runtime type checking
- Fixed missing pytest fixtures for example tests by creating `conftest.py`
- Fixed all mypy type errors to achieve strict mode compliance
- Fixed duplicate class and function definitions in `logical/plan.py` and `logical/operators.py`
- Fixed missing function imports in `expressions/__init__.py` (removed non-existent `date_add`, `date_sub`, `len`, `substr`, `pow`, `power`, `trunc`)

## [0.3.0] - 2024-12-19

### Added
- **Full async/await support** for all database operations, file I/O, and DataFrame operations
- **Async API** with `async_connect()` function returning `AsyncDatabase` instance
- **Async DataFrame operations** - All DataFrame methods now support async execution (`collect()`, `select()`, `where()`, `join()`, etc.)
- **Async file readers** - Async support for CSV, JSON, JSONL, Parquet, and text file reading
- **Async file writers** - Async support for writing DataFrames to files and tables
- **Async table mutations** - Async `insert()`, `update()`, and `delete()` operations
- **Optional async dependencies** - Grouped optional dependencies for async support:
  - `moltres[async]` - Core async support (aiofiles)
  - `moltres[async-postgresql]` - PostgreSQL async driver (includes async + asyncpg)
  - `moltres[async-mysql]` - MySQL async driver (includes async + aiomysql)
  - `moltres[async-sqlite]` - SQLite async driver (includes async + aiosqlite)
- **Async streaming support** - Async iterators for processing large datasets in chunks
- **SQLAlchemy async engine integration** - Automatic async driver detection and configuration
- Comprehensive async test suite (8 new async tests)

### Changed
- Improved type safety with proper async type hints
- Enhanced error messages for async operations
- Better separation between sync and async APIs

## [0.2.0] - 2024-12-19

### Added
- Comprehensive exception hierarchy with specific exception types (`ExecutionError`, `ValidationError`, `SchemaError`, `DatabaseConnectionError`, `UnsupportedOperationError`)
- Input validation for SQL identifiers to prevent injection attacks
- Batch insert support for improved performance (`execute_many()` method)
- Connection pooling configuration options (max_overflow, pool_timeout, pool_recycle, pool_pre_ping)
- Structured logging throughout the execution layer
- Enhanced error messages with context (table names, operations, etc.)
- Comprehensive docstrings for public APIs
- Pre-commit hooks configuration
- EditorConfig for consistent code formatting
- Contributing guide (`CONTRIBUTING.md`)
- CI/CD workflow with multi-OS and multi-Python version testing
- **Environment variable support** for all configuration options (MOLTRES_DSN, MOLTRES_POOL_SIZE, etc.)
- **Performance monitoring hooks** for query execution tracking (`register_performance_hook()`, `unregister_performance_hook()`)
- **Security best practices documentation** (`docs/SECURITY.md`)
- **Troubleshooting guide** (`docs/TROUBLESHOOTING.md`)
- **Common patterns and examples** (`docs/EXAMPLES.md`)
- Comprehensive test suite with 113 test cases covering edge cases, security, and error handling
- **Modular file reader architecture** - Refactored large `reader.py` (699 lines) into organized `readers/` subdirectory with format-specific modules

### Changed
- Improved type hints throughout the codebase
- Enhanced error messages with more context
- Better documentation for limit() behavior
- **Code organization**: Split `dataframe/reader.py` into modular format-specific readers (`csv_reader.py`, `json_reader.py`, `parquet_reader.py`, `text_reader.py`, `schema_inference.py`)

### Fixed
- Typo in pyproject.toml (deV → dev)
- Incomplete insert_rows implementation in io/write.py
- Missing CI/CD workflow file
- Type checking issues with optional dependencies (pandas, pyarrow)

### Security
- Added SQL identifier validation to prevent injection attacks
- Comprehensive security testing and documentation

## [0.1.0] - Initial Release

### Added
- PySpark-like DataFrame API
- SQL compilation from logical plans
- Support for SQLite, PostgreSQL, MySQL
- File format readers (CSV, JSON, JSONL, Parquet, Text)
- Streaming support for large datasets
- Table mutations (insert, update, delete)
- Column expressions and functions
- Joins, aggregations, filtering, sorting
- Type hints and mypy support

[Unreleased]: https://github.com/eddiethedean/moltres/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/eddiethedean/moltres/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/eddiethedean/moltres/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/eddiethedean/moltres/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/eddiethedean/moltres/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/eddiethedean/moltres/releases/tag/v0.1.0

