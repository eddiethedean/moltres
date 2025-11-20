# Changelog

All notable changes to Moltres will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - `moltres[async,async-postgresql]` - PostgreSQL async driver (asyncpg)
  - `moltres[async,async-mysql]` - MySQL async driver (aiomysql)
  - `moltres[async,async-sqlite]` - SQLite async driver (aiosqlite)
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

[Unreleased]: https://github.com/eddiethedean/moltres/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/eddiethedean/moltres/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/eddiethedean/moltres/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/eddiethedean/moltres/releases/tag/v0.1.0

