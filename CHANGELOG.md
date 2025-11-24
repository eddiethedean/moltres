# Changelog

All notable changes to Moltres will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Chunked File Reading for Large Files** - Files are now read in chunks by default to safely handle files larger than available memory:
  - **Default Streaming Mode** - All file reads (`db.read.csv()`, `db.read.json()`, etc.) now use chunked reading by default, similar to PySpark's partition-based approach
  - **Opt-Out Mechanism** - Users can disable chunked reading for small files by setting `stream=False`: `db.read.option("stream", False).csv("small_file.csv")`
  - **Memory Safety** - Prevents out-of-memory errors when processing large datasets by reading and inserting data incrementally in chunks
  - **Schema Inference from First Chunk** - Schema is inferred from the first chunk of data, then applied consistently to all subsequent chunks
  - **Error Recovery** - Temporary tables are automatically cleaned up if chunk insertion fails
  - **Empty File Handling** - Gracefully handles empty files with or without explicit schemas
  - **Both Sync and Async** - Full support for both synchronous (`DataFrame`) and asynchronous (`AsyncDataFrame`) operations
  - This matches PySpark's behavior where files are read in partitions across the cluster, adapted for single-machine processing
- **PySpark Read API Parity** - Enhanced read API to match PySpark's DataFrameReader with comprehensive option support:
  - **Builder Methods** - Added `options()` method to set multiple read options at once (PySpark-compatible):
    - `db.read.options(header=True, delimiter=",").csv("data.csv")`
    - Works with all read methods: `csv()`, `json()`, `parquet()`, `text()`, etc.
    - Available on both sync (`DataLoader`, `ReadAccessor`) and async (`AsyncDataLoader`, `AsyncReadAccessor`) APIs
  - **Text File Method** - Added `textFile()` method as PySpark-compatible alias for `text()`:
    - `db.read.textFile("log.txt")` - Same as `db.read.text("log.txt")`
    - Available in both sync and async APIs
  - **CSV Options** - Comprehensive CSV reading options matching PySpark:
    - `mode` - Read mode: "PERMISSIVE" (default), "DROPMALFORMED", or "FAILFAST"
    - `encoding` - File encoding (default: "UTF-8")
    - `quote` - Quote character (default: '"')
    - `escape` - Escape character (default: "\\")
    - `nullValue` - String representation of null (default: "")
    - `nanValue` - String representation of NaN (default: "NaN")
    - `dateFormat` - Date format string for parsing dates
    - `timestampFormat` - Timestamp format string for parsing timestamps
    - `samplingRatio` - Fraction of rows used for schema inference (default: 1.0)
    - `columnNameOfCorruptRecord` - Column name for corrupt records
    - `sep` - Alias for `delimiter`
    - `quoteAll` - Quote all fields (default: False)
    - `ignoreLeadingWhiteSpace` - Ignore leading whitespace (default: False)
    - `ignoreTrailingWhiteSpace` - Ignore trailing whitespace (default: False)
    - `comment` - Comment character to skip lines
    - `enforceSchema` - Enforce schema even if it doesn't match data (default: True)
    - All options work with both sync and async CSV readers
  - **JSON Options** - Enhanced JSON reading options matching PySpark:
    - `mode` - Read mode: "PERMISSIVE" (default), "DROPMALFORMED", or "FAILFAST"
    - `encoding` - File encoding (default: "UTF-8")
    - `multiLine` - Alias for `multiline` (PySpark-compatible)
    - `dateFormat` - Date format string for parsing dates
    - `timestampFormat` - Timestamp format string for parsing timestamps
    - `samplingRatio` - Fraction of rows used for schema inference (default: 1.0)
    - `columnNameOfCorruptRecord` - Column name for corrupt records
    - `lineSep` - Line separator for multiline JSON
    - `dropFieldIfAllNull` - Drop fields if all values are null (default: False)
    - All options work with both sync and async JSON readers
    - Note: Some JSON parsing options (e.g., `allowComments`, `allowUnquotedFieldNames`) are not supported by Python's `json` module and are ignored
  - **Parquet Options** - Parquet reading options matching PySpark:
    - `mergeSchema` - Merge schemas from multiple files (default: False)
    - `rebaseDatetimeInRead` - Rebase datetime values during read (default: True)
    - `datetimeRebaseMode` - Datetime rebase mode (default: "EXCEPTION")
    - `int96RebaseMode` - INT96 rebase mode (default: "EXCEPTION")
    - All options work with both sync and async Parquet readers
  - **Text Options** - Enhanced text file reading options:
    - `encoding` - File encoding (default: "UTF-8")
    - `wholetext` - If True, read entire file as single value (default: False)
    - `lineSep` - Line separator (default: newline)
    - All options work with both sync and async text readers
  - **Read Modes** - Comprehensive error handling modes for CSV and JSON:
    - `PERMISSIVE` (default) - Sets other fields to null when encountering corrupted records and puts malformed strings into a field configured by `columnNameOfCorruptRecord`
    - `DROPMALFORMED` - Ignores the whole corrupted records
    - `FAILFAST` - Throws an exception when it meets corrupted records
  - **Schema Inference Enhancements** - Enhanced schema inference with date/timestamp format support:
    - `dateFormat` and `timestampFormat` options now properly influence schema inference
    - Date and timestamp columns are correctly inferred when formats are provided
    - Works with CSV, JSON, and JSONL readers
- **PySpark Write API Parity & Chunked Output**:
  - **Builder Enhancements** - Added `.format()`, `.options()`, `.bucketBy()`, `.sortBy()` and camel-case mode aliases to `DataFrameWriter` and `AsyncDataFrameWriter`
  - **Save Modes** - `mode("ignore")` now skips both table and file targets when they already exist; file targets also honor `error_if_exists`/`overwrite`
  - **File Writers Stream by Default** - Any sink that requires materialization (CSV/JSON/JSONL/Text/Parquet) now streams chunks automatically unless `.stream(False)` is specified
  - **New Sinks & Options** - Added `.text()` helper, `format("csv").options(...)` parity, JSON streaming (when no indent), Parquet streaming via `pyarrow.ParquetWriter`, and explicit mode handling for partitioned outputs
  - **Async Parity** - All improvements apply to the async writer as well, using `aiofiles` for file sinks
  - **Safety Checks** - Unsupported combinations (e.g., bucketing into files, partitioned async text/parquet writes) now raise `NotImplementedError` instead of silently misbehaving
- **Extended Function Library** - Added 38+ new PySpark-compatible functions across multiple categories:
  - **Mathematical Functions** - `pow()`, `power()`, `asin()`, `acos()`, `atan()`, `atan2()`, `signum()`, `sign()`, `log2()`, `hypot()` for advanced mathematical operations
  - **String Functions** - `initcap()`, `instr()`, `locate()`, `translate()` for enhanced string manipulation
  - **Date/Time Functions** - `to_timestamp()`, `unix_timestamp()`, `from_unixtime()`, `date_trunc()`, `quarter()`, `weekofyear()`, `week()`, `dayofyear()`, `last_day()`, `months_between()` for comprehensive date/time operations
  - **Window Functions** - `first_value()`, `last_value()` for window-based analytics
  - **Array Functions** - `array_append()`, `array_prepend()`, `array_remove()`, `array_distinct()`, `array_sort()`, `array_max()`, `array_min()`, `array_sum()` for array manipulation
  - **JSON Functions** - `json_tuple()`, `from_json()`, `to_json()`, `json_array_length()` for JSON data processing
  - **Utility Functions** - `rand()`, `randn()`, `hash()`, `md5()`, `sha1()`, `sha2()`, `base64()` for random number generation and hashing
  - **Additional Functions** - `monotonically_increasing_id()`, `crc32()`, `soundex()` for ID generation and data processing
  - All functions include dialect-specific SQL compilation for PostgreSQL, MySQL, and SQLite
  - Functions are available from `moltres.expressions.functions` and can be imported directly
  - Example: `from moltres.expressions.functions import pow, asin, to_timestamp; df.select(pow(col("x"), 2), asin(col("y")))`
- **98% PySpark API Compatibility** - Major improvements to match PySpark's DataFrame API:
  - **Raw SQL Query Support** - New `db.sql()` method for executing raw SQL queries, similar to PySpark's `spark.sql()`:
    - Accepts raw SQL strings with optional named parameters (`:param_name` syntax)
    - Returns lazy `DataFrame` objects that can be chained with other operations
    - Supports parameterized queries for security and flexibility
    - Raw SQL is automatically wrapped in subqueries when chained, enabling full DataFrame API compatibility
    - Works with both synchronous (`db.sql()`) and asynchronous (`await db.sql()`) databases
    - SQL dialect is determined by the database connection
    - Example: `db.sql("SELECT * FROM users WHERE id = :id", id=1).where(col("age") > 18).collect()`
  - **SQL Expression Selection** - New `selectExpr()` method on DataFrame for writing SQL expressions directly, matching PySpark's `selectExpr()` API:
    - Accepts SQL expression strings (e.g., `"amount * 1.1 as with_tax"`)
    - Parses SQL expressions into Column expressions automatically
    - Supports arithmetic operations, functions, comparisons, literals, and aliases
    - Full SQL expression parser with operator precedence handling
    - Works with both synchronous (`df.selectExpr()`) and asynchronous (`await df.selectExpr()`) DataFrames
    - Returns lazy `DataFrame` objects that can be chained with other operations
    - Example: `df.selectExpr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")`
  - **Select All Columns with `select("*")`** - Support for `df.select("*")` to explicitly select all columns, matching PySpark's API:
    - `select("*")` alone is equivalent to `select()` (selects all columns)
    - Can combine `"*"` with other columns: `select("*", col("new_col"))` or `select("*", "column_name")`
    - Works with both synchronous and asynchronous DataFrames
    - Example: `df.select("*", (col("amount") * 1.1).alias("with_tax"))`
  - **SQL String Predicates in `filter()` and `where()`** - Support for SQL string predicates in filtering methods, matching PySpark's API:
    - `filter()` and `where()` now accept both `Column` expressions and SQL strings
    - Supports basic comparison operators (`>`, `<`, `>=`, `<=`, `=`, `!=`)
    - Works with both synchronous and asynchronous DataFrames
    - Complex predicates can be achieved by chaining multiple filters or using Column API
    - Example: `df.filter("age > 18")` or `df.where("amount >= 100 AND status = 'active'")`
  - **String Column Names in Aggregations** - Support for string column names in `agg()`, matching PySpark's convenience:
    - String column names default to `sum()` aggregation
    - More convenient than PySpark's requirement for explicit aggregation functions
    - Works with both synchronous and asynchronous DataFrames
    - Example: `df.group_by("category").agg("amount")` (equivalent to `sum(col("amount"))`)
  - **Dictionary Syntax in Aggregations** - Support for dictionary syntax in `agg()`, matching PySpark's API:
    - Dictionary maps column names to aggregation function names
    - Supports multiple aggregations in a single call
    - Can be mixed with Column expressions and string column names
    - Example: `df.group_by("category").agg({"amount": "sum", "price": "avg"})`
  - **Pivot on GroupBy** - PySpark-style `pivot()` chaining on `groupBy()`, matching PySpark's API:
    - Supports `df.group_by("category").pivot("status").agg("amount")` syntax
    - Automatic pivot value inference from data (like PySpark)
    - Explicit pivot values when needed: `pivot("status", values=["active", "inactive"])`
    - Works with both synchronous and asynchronous DataFrames
    - Example: `df.group_by("category").pivot("status").agg("amount")`
  - **Explode Function** - PySpark-style `explode()` function for array/JSON column expansion:
    - Function-based API: `df.select(explode(col("array_col")).alias("value"))`
    - Matches PySpark's `from pyspark.sql.functions import explode` pattern
    - Works with both synchronous and asynchronous DataFrames
    - Note: SQL compilation for `explode()` is in progress (requires table-valued function support)
    - Example: `from moltres.expressions.functions import explode; df.select(explode(col("tags")).alias("tag"))`
  - **PySpark-style Aliases** - Additional camelCase aliases for better PySpark compatibility:
    - `orderBy()` and `sort()` - PySpark-style aliases for `order_by()`
    - `saveAsTable()` - PySpark-style alias for `save_as_table()`
    - Both camelCase and snake_case methods available throughout the API
    - Example: `df.orderBy(col("name"))` or `df.write.saveAsTable("results")`
  - **Improved `withColumn()`** - Enhanced `withColumn()` to correctly handle both adding and replacing columns:
    - Adding new columns: `df.withColumn("new_col", col("old_col") * 2)`
    - Replacing existing columns: `df.withColumn("existing_col", col("existing_col") + 1)`
    - Matches PySpark's behavior exactly
    - Works with both synchronous and asynchronous DataFrames

### Changed
- **API Compatibility** - Moltres now achieves ~98% API compatibility with PySpark for core DataFrame operations
- All major DataFrame transformation methods now match PySpark's API
- Both camelCase (PySpark-style) and snake_case (Python-style) naming conventions supported throughout the API

### Fixed
- Fixed `withColumn()` to correctly replace existing columns instead of duplicating them
- Fixed pivot value inference to work automatically when values are not provided
- Fixed column replacement logic in `withColumn()` to match PySpark's behavior
- Fixed `select("*")` to work correctly when combined with other columns
- Fixed async PostgreSQL connections that forwarded DSN `?options=-csearch_path=...` parameters to asyncpg (which rejects unknown keywords) by translating them into asyncpg `server_settings`.
- Fixed async PostgreSQL staging tables so `createDataFrame()` and file readers now create regular tables instead of connection-scoped temp tables, preventing `UndefinedTableError` when inserts execute on different pooled connections.

## [0.8.0] - 2025-11-22

### Added
- **Lazy CRUD and DDL Operations** - All DataFrame CRUD and DDL operations are now lazy, requiring an explicit `.collect()` call for execution:
  - `insert()`, `update()`, `delete()`, `merge()` now return lazy `Mutation` objects
  - `create_table()`, `drop_table()` now return lazy `DDLOperation` objects
  - Operations build a logical plan that only executes when `.collect()` is called
  - DataFrame write operations remain eager (similar to PySpark's behavior)
  - New `to_sql()` method on lazy operations for SQL inspection without execution
- **Transaction Management** - All operations within a single `.collect()` call are part of a single session that rolls back all changes if any failure occurs:
  - Automatic transaction management for lazy operations
  - Rollback on any exception during execution
  - Explicit transaction support via `db.transaction()` context manager
- **Batch Operation API** - New `db.batch()` context manager to queue multiple lazy operations and execute them together within a single transaction:
  - Queue multiple insert, update, delete, and DDL operations
  - Execute all queued operations atomically in a single transaction
  - Automatic rollback if any operation fails
  - Supports both synchronous and asynchronous batch operations
- **Type Checking Improvements** - Enhanced type safety and CI compatibility:
  - Added `pandas-stubs>=2.1` to dev dependencies for proper mypy type checking
  - Fixed pandas DataFrame constructor type compatibility issues
  - Improved type annotations for lazy operation classes

### Changed
- **Breaking Change**: CRUD and DDL operations now require `.collect()` to execute:
  - `table.insert([...])` → `table.insert([...]).collect()`
  - `table.update(...)` → `table.update(...).collect()`
  - `table.delete(...)` → `table.delete(...).collect()`
  - `db.create_table(...)` → `db.create_table(...).collect()`
  - `db.drop_table(...)` → `db.drop_table(...).collect()`
- Improved composability of operations by making them lazy
- Enhanced transaction safety with automatic rollback on failures
- Better alignment with PySpark's lazy evaluation model

### Fixed
- Fixed mypy type checking errors related to pandas DataFrame constructor
- Fixed unused type ignore comments after adding pandas-stubs
- Fixed transaction management to ensure atomicity of operations
- Fixed async operation handling in batch context

### Internal
- Added `OperationBatch` and `async_OperationBatch` classes for batch operation management
- Created `Mutation` and `DDLOperation` base classes for lazy operations
- Enhanced test coverage for lazy operations and batch API
- Improved code quality with proper type annotations and mypy strict checking

## [0.7.0] - 2025-01-22

### Added
- **PostgreSQL and MySQL Testing Infrastructure** - Comprehensive test support for multiple database backends:
  - Added `testing.postgresql` and `testing.mysqld` dependencies for ephemeral database instances in tests
  - New pytest markers: `@pytest.mark.postgres`, `@pytest.mark.mysql`, `@pytest.mark.multidb`
  - Database fixtures: `postgresql_db`, `postgresql_connection`, `mysql_db`, `mysql_connection`
  - Async database fixtures: `postgresql_async_connection`, `mysql_async_connection`
  - Test helpers: `seed_customers_orders()` for consistent test data across databases
  - New test suites: `test_postgresql_features.py`, `test_mysql_features.py`, `test_multidb.py`
  - Async test suites: `test_async_postgresql_features.py`, `test_async_mysql_features.py`, `test_async_integration.py`
- **Type Overloads for collect() Methods** - Enhanced type safety with `@overload` decorators:
  - `collect(stream=False)` returns `List[Dict[str, object]]`
  - `collect(stream=True)` returns `Iterator[List[Dict[str, object]]]` or `AsyncIterator[...]`
  - Improved type inference for better IDE support and type checking

### Changed
- Enhanced type annotations throughout the codebase with proper `@overload` decorators
- Improved type safety in SQL compiler with explicit type casting using `typing_cast`
- Better type inference for DataFrame operations with overloaded method signatures

### Fixed
- Fixed mypy type checking errors related to `ColumnElement[Any]` return types in expression compiler
- Fixed ruff linting errors for name conflicts between `typing.cast` and `sqlalchemy.cast`
- Fixed async DSN parsing to correctly convert `mysql+pymysql://` to `mysql+aiomysql://` for async connections
- Fixed database connection cleanup - added `close()` methods to `Database` and `AsyncDatabase` classes
- Fixed test hanging issues by ensuring proper engine disposal in pytest fixtures
- Fixed column qualification after joins to handle unqualified column names correctly
- Fixed PostgreSQL JSON extraction to use `->>` operator for direct JSONB path extraction
- Fixed PostgreSQL array literal syntax to use `ARRAY[...]` format
- Fixed MySQL JSON array functions to handle literal values correctly
- Fixed MySQL `JSON_CONTAINS` to properly quote values with `json_quote()`

### Internal
- Added comprehensive type stubs and type annotations for better IDE support
- Improved code quality with ruff formatting and mypy strict type checking
- Enhanced test coverage with 301 passing tests across multiple database backends

## [0.6.0] - 2025-01-21

### Added
- **Null Handling Convenience Methods** - New `na` property on DataFrame for convenient null handling:
  - `df.na.drop()` - Drop rows with null values (wrapper around `dropna()`)
  - `df.na.fill(value)` - Fill null values with a specified value (wrapper around `fillna()`)
  - Available on both synchronous and asynchronous DataFrames
- **Random Sampling** - New `sample(fraction, seed=None)` method for random row sampling:
  - Uses `TABLESAMPLE` clause for PostgreSQL and SQL Server
  - Falls back to `ORDER BY RANDOM() LIMIT` for SQLite and other dialects
  - Supports optional seed for reproducible sampling
- **Enhanced Type System** - New data types with full SQL support:
  - **Decimal/Numeric with Precision** - `decimal(name, precision, scale)` helper for creating columns with specific precision and scale
  - **UUID Type** - `uuid(name)` helper with dialect-specific compilation (PostgreSQL `UUID`, MySQL `CHAR(36)`, SQLite `TEXT`)
  - **JSON/JSONB Type** - `json(name)` helper with dialect-specific compilation (PostgreSQL `JSONB`, MySQL `JSON`, SQLite `TEXT`)
  - **Date/Time Interval Types** - `interval(name)` helper with interval arithmetic support
  - All types support precision/scale where applicable and proper casting
- **Interval Arithmetic Functions** - New functions for date/time interval operations:
  - `date_add(column, interval)` - Add interval to date/time column (e.g., `date_add(col("date"), "1 DAY")`)
  - `date_sub(column, interval)` - Subtract interval from date/time column
  - Dialect-specific compilation with proper interval handling
- **Join Hints** - New `hints` parameter for `join()` method to provide query optimization hints:
  - Supports dialect-specific join hints (e.g., `USE_INDEX`, `FORCE_INDEX`, `IGNORE_INDEX`)
  - Hints are passed through to the SQL compiler for database-specific optimizations
- **Complex Join Conditions** - Enhanced `join()` method to support arbitrary Column expressions in join conditions:
  - Beyond simple column pairs, now supports complex predicates and expressions
  - Enables more sophisticated join logic while maintaining SQL pushdown
- **Query Plan Analysis** - New `explain(analyze=False)` method on DataFrame:
  - Returns query execution plan as SQL `EXPLAIN` output
  - Supports `analyze=True` for execution plan with statistics (PostgreSQL `EXPLAIN ANALYZE`)
  - Helps with query optimization and debugging
- **Pivot/Unpivot Operations** - New `pivot()` method for data reshaping:
  - `df.pivot(pivot_column, value_column, agg_func="sum", pivot_values=None)` - Reshape data by pivoting columns
  - Compiles to `CASE WHEN` with aggregation for cross-dialect compatibility
  - Supports custom aggregation functions (sum, avg, count, min, max)
  - Automatically detects pivot values if not specified

### Changed
- Enhanced `cast()` function to support more SQL types with precision/scale (DECIMAL, TIMESTAMP, DATE, TIME, INTERVAL)
- Improved type annotations throughout the codebase for better IDE support and type safety

### Fixed
- Fixed mypy type checking errors related to type annotations in compiler and DDL modules
- Fixed ruff linting errors for unused imports and code formatting

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

[Unreleased]: https://github.com/eddiethedean/moltres/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/eddiethedean/moltres/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/eddiethedean/moltres/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/eddiethedean/moltres/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/eddiethedean/moltres/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/eddiethedean/moltres/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/eddiethedean/moltres/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/eddiethedean/moltres/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/eddiethedean/moltres/releases/tag/v0.1.0

