# TODO

This file tracks planned features, improvements, and tasks for Moltres.

> **Note:** Moltres focuses on providing full SQL feature support through a DataFrame API, not replicating every PySpark feature. Features are included only if they map to SQL/SQLAlchemy capabilities and align with SQL pushdown execution.

## 🎯 High Priority

### Core Features
- [ ] `UNNEST()` / table-valued functions for array/JSON expansion in FROM clause (`UNNEST(array)`, `jsonb_array_elements()`, `jsonb_each()` in FROM) - dialect-specific
  - Initial implementation covers SQLite (`json_each`) and PostgreSQL (`jsonb_array_elements`) via the `explode()` API. Additional dialect support will be tracked separately.
- [x] `FILTER` clause for conditional aggregation (`COUNT(*) FILTER (WHERE condition)`) - SQL standard (PostgreSQL, MySQL 8.0+, SQL Server, Oracle)
- [ ] `DISTINCT ON` for selecting distinct rows based on specific columns (PostgreSQL-specific)

### SQL Dialects
- [ ] Better MySQL-specific optimizations
- [ ] Oracle database support
- [ ] SQL Server support
- [ ] BigQuery support
- [ ] Redshift support
- [ ] Snowflake support
- [ ] DuckDB support

### Schema Management ✅ **COMPLETED**
- [x] Foreign key constraints (`FOREIGN KEY ... REFERENCES`) - SQL standard
- [x] Unique constraints (`UNIQUE`) - SQL standard
- [x] Check constraints (`CHECK`) - SQL standard
- [x] Indexes (`CREATE INDEX`, `DROP INDEX`) - SQL standard

## 📊 Medium Priority

### Advanced SQL Features
- [ ] Advanced JSON functions (`jsonb_set()`, `jsonb_insert()`, `jsonb_delete_path()`, etc.) - dialect-specific (PostgreSQL JSONB, MySQL 8.0+)
- [ ] Full-text search functions (`to_tsvector()`, `to_tsquery()`, `ts_rank()`, etc.) - dialect-specific (PostgreSQL, MySQL, SQL Server)
- [ ] PIVOT / UNPIVOT SQL operations (dialect-specific, e.g., SQL Server, Oracle)
- [ ] Stored procedure support (dialect-specific via SQLAlchemy)

### Aggregations
- [ ] `skewness()` / `kurtosis()` higher-order statistics (dialect-specific)
- [ ] Array aggregation with ordering (`ARRAY_AGG(column ORDER BY column)`) - dialect-specific

### Data Types
- [ ] Better support for complex types (arrays, maps, structs) - SQL standard/dialect-specific
- [x] SQLAlchemy TypeEngine integration - Leverage SQLAlchemy's type system for better type mapping ✅ **COMPLETED**
- [ ] Type coercion utilities - Automatic type conversion based on SQLAlchemy TypeEngine
- [x] Dialect-specific type mapping - Use SQLAlchemy's dialect-specific type mapping ✅ **COMPLETED**

### File Formats
- [ ] Excel/ODS file support
- [ ] Avro file support
- [ ] ORC file support
- [ ] Delta Lake support
- [ ] Arrow IPC format support
- [ ] Better compression options for all formats

### Schema Inspection ✅ **COMPLETED**
- [x] Table reflection (`db.reflect_table(name)`) - Automatically introspect table schemas from database
- [x] Database reflection (`db.reflect()`) - Introspect all tables, views, indexes in database schema
- [x] Schema introspection utilities (`db.get_table_names()`, `db.get_view_names()`, etc.)
- [x] Column metadata introspection (`db.get_columns(table_name)`, etc.)

### Transaction Control
- [ ] Explicit transaction control improvements (`BEGIN`, `COMMIT`, `ROLLBACK`) - expand API
- [ ] Nested transaction context manager (`with db.transaction(savepoint=True):`)
- [ ] Read-only transactions (`with db.transaction(readonly=True):`)
- [ ] Transaction timeout (`with db.transaction(timeout=30):`)
- [ ] Savepoints (`SAVEPOINT`, `ROLLBACK TO SAVEPOINT`) - SQL standard
- [ ] Transaction isolation levels (`SET TRANSACTION ISOLATION LEVEL`) - SQL standard
- [ ] Transaction state inspection (`db.is_in_transaction()`, `db.get_transaction_status()`)
- [ ] Locking (`SELECT ... FOR UPDATE`, `SELECT ... FOR SHARE`) - SQL standard/dialect-specific

## 🔧 Developer Experience

### Type Safety
- [ ] Better type inference for schemas
- [ ] Generic DataFrame types with schema
- [ ] Type-safe column references
- [x] Better mypy coverage (reduce Any types) - Fixed all type errors in async implementations and examples ✅ **COMPLETED**

### Error Handling
- [ ] Error recovery strategies
- [ ] Validation error aggregation

### Documentation
- [ ] Video tutorials
- [ ] Enhanced docs/index.md with better organization and search-friendly structure

### Testing
- [ ] Property-based testing with Hypothesis
- [ ] Load testing
- [ ] Test coverage improvements (currently ~75%, target 80%+)

## 🛠️ Infrastructure

### CI/CD
- [ ] Automated release process
- [ ] Version bump automation
- [ ] Changelog generation
- [ ] Documentation deployment
- [ ] Performance regression testing
- [x] Pre-commit CI checks script improvements - Updated to use `python -m pytest` for correct environment detection ✅ **COMPLETED**

### Code Quality
- [ ] More comprehensive linting rules
- [ ] Code complexity analysis
- [ ] Security scanning
- [ ] Dependency vulnerability scanning

### Distribution
- [ ] PyPI package optimization
- [ ] Conda package support
- [ ] Docker image with examples
- [ ] Pre-built wheels for common platforms

## 📈 Performance

### Optimizations
- [ ] Lazy evaluation improvements
- [ ] Query plan caching
- [ ] Better batch size heuristics
- [ ] Parallel file reading where possible
- [ ] Streaming optimizations

### Monitoring
- [ ] Query plan visualization
- [ ] Execution time breakdown
- [ ] Resource usage tracking
- [ ] Query profiling tools
- [ ] SQLAlchemy event system integration
- [ ] Connection pool event hooks
- [ ] Execution event hooks
- [ ] Connection pool statistics access

## 🔐 Security

### Enhancements
- [ ] SQL injection testing suite expansion
- [ ] Rate limiting support
- [ ] Connection encryption verification
- [ ] Audit logging
- [ ] Access control integration points

## 🌐 Ecosystem

### Integrations
- [ ] Jupyter notebook integration
- [ ] Streamlit integration
- [ ] FastAPI integration examples
- [ ] Django integration
- [ ] Flask integration
- [ ] Polars interop improvements (more seamless)
- [ ] Pandas interop improvements (more seamless)

### Tools
- [ ] CLI tool for common operations
- [ ] Query builder GUI
- [ ] Schema migration tool
- [ ] Data validation framework integration

## 🎯 Advocacy & Marketing

### Migration Guides
- [ ] Create docs/MIGRATION_PANDAS.md guide showing how to convert common Pandas workflows to Moltres
- [ ] Create docs/MIGRATION_SQLALCHEMY.md guide showing how to replace SQLAlchemy ORM operations with Moltres DataFrame CRUD (Note: SQLAlchemy model integration now allows using models directly with Moltres)
- [ ] Create docs/MIGRATION_SPARK.md guide for teams moving from PySpark to Moltres

### Integration Examples
- [ ] Create example showing Moltres integration with FastAPI for building data APIs
- [ ] Create Jupyter notebook examples demonstrating interactive data analysis with Moltres
- [ ] Create example showing how Moltres can complement or replace dbt transformations

### Performance & Benchmarks
- [ ] Create docs/PERFORMANCE.md with detailed benchmarks comparing memory usage and query time vs Pandas/Ibis/SQLAlchemy

### Marketing & Outreach
- [ ] Draft blog post announcing Moltres unique positioning: The Missing DataFrame Layer for SQL in Python
- [ ] Create script/storyboard for video tutorial demonstrating memory-efficient operations and CRUD workflows
- [ ] Create template for documenting real-world use cases and case studies
- [ ] Create social media content (Twitter/LinkedIn posts) highlighting unique positioning

## 📝 Code Refactoring

### Improvements
- [ ] Reduce code duplication between sync/async implementations
- [ ] Better abstraction for SQL dialect differences
- [ ] Improve logical plan optimization
- [ ] Refactor compiler for better maintainability
- [ ] Better handling of edge cases in file readers

## ✅ Recently Completed

### Documentation Organization & Read the Docs Optimization ✅ **COMPLETED**
- ✅ **Examples Directory Reorganization** – Moved example directories to `docs/` for better organization:
  - ✅ Moved `example_data/` → `docs/example_data/`
  - ✅ Moved `example_output/` → `docs/example_output/`
  - ✅ Moved `examples/` → `docs/examples/`
  - ✅ Updated all references throughout codebase (guides, documentation, CHANGELOG, etc.)
  - ✅ Updated test files to use new `docs/examples/` path
  - ✅ Updated example files to use relative paths from `__file__` for data/output directories
- ✅ **Docstring Optimization for Read the Docs** – Enhanced all docstrings for optimal Read the Docs deployment:
  - ✅ Added Sphinx cross-references (`:class:`, `:func:`, `:meth:`) throughout codebase
  - ✅ Ensured Google-style format consistency across all modules
  - ✅ Added proper type annotations in Returns sections with Sphinx references
  - ✅ Completed Args/Returns/Raises sections for all public API functions and classes
  - ✅ Enhanced module-level docstrings with comprehensive descriptions
  - ✅ Updated 948+ docstrings across 133 source files
  - ✅ Created helper scripts for docstring management (`scripts/update_docstrings_for_rtd.py`, `scripts/find_missing_docstrings.py`)
- ✅ **README Quick Start Example Fix** – Fixed join example in README:
  - ✅ Added `.select()` to `db.table("customers")` before joining (TableHandle must be converted to DataFrame)
  - ✅ Fixed join condition to use proper column references: `on=[col("orders.customer_id") == col("customers.id")]`

### Polars-Style Interface ✅ **COMPLETED**
- ✅ **Polars LazyFrame API** – Comprehensive Polars-style interface (`PolarsDataFrame`):
  - ✅ **Lazy Evaluation** – All operations build logical plans that execute only on `collect()` or `fetch()`
  - ✅ **Core Operations** – `select()`, `filter()`, `with_columns()`, `with_column()`, `drop()`, `rename()`, `sort()`, `limit()`, `head()`, `tail()`, `sample()`
  - ✅ **GroupBy Operations** – `group_by()` returns `PolarsGroupBy` with aggregation methods (`agg()`, `mean()`, `sum()`, `min()`, `max()`, `count()`, `std()`, `var()`, `first()`, `last()`, `n_unique()`)
  - ✅ **Join Operations** – Full join support (`inner`, `left`, `right`, `outer`, `anti`, `semi`) with `on` parameter
  - ✅ **Column Access** – `df['col']` returns `PolarsColumn` with `.str` and `.dt` accessors
  - ✅ **String Accessor** – `.str` accessor with methods: `upper()`, `lower()`, `strip()`, `contains()`, `startswith()`, `endswith()`, `replace()`, `split()`, `len()`
  - ✅ **DateTime Accessor** – `.dt` accessor with methods: `year()`, `month()`, `day()`, `hour()`, `minute()`, `second()`, `day_of_week()`, `day_of_year()`, `quarter()`, `week()`
  - ✅ **Window Functions** – `row_number()`, `rank()`, `dense_rank()`, `percent_rank()`, `ntile()`, `lead()`, `lag()`, `first_value()`, `last_value()` with `over()` clause
  - ✅ **Conditional Expressions** – `when().then().otherwise()` for SQL CASE statements
  - ✅ **Data Reshaping** – `explode()`, `unnest()`, `pivot()`, `slice()`
  - ✅ **Set Operations** – `concat()`, `vstack()`, `hstack()`, `union()`, `intersect()`, `difference()`, `cross_join()`
  - ✅ **SQL Expressions** – `select_expr()` for raw SQL expression selection
  - ✅ **CTEs** – `cte()`, `with_recursive()` for Common Table Expressions
  - ✅ **Utility Methods** – `gather_every()`, `quantile()`, `describe()`, `explain()`, `with_row_count()`, `with_context()`, `with_columns_renamed()`
  - ✅ **File I/O** – Polars-style read/write operations: `db.scan_csv()`, `db.scan_json()`, `db.scan_jsonl()`, `db.scan_parquet()`, `db.scan_text()`, `df.write_csv()`, `df.write_json()`, `df.write_jsonl()`, `df.write_parquet()`
  - ✅ **Schema Properties** – `columns`, `width`, `height`, `schema` properties with lazy evaluation
  - ✅ Comprehensive test coverage (all tests passing)
  - ✅ Example file (`docs/examples/19_polars_interface.py`) and guide (`guides/10-polars-interface.md`)

### Async Polars and Pandas DataFrames ✅ **COMPLETED**
- ✅ **AsyncPolarsDataFrame** – Async version of Polars-style interface:
  - ✅ Wraps `AsyncDataFrame` with Polars-style API
  - ✅ All database-interactive methods are `async` (`collect()`, `fetch()`, `height`, `schema`, `describe()`, `explain()`, `write_*`)
  - ✅ Integrated via `.polars()` method on `AsyncDataFrame` and `AsyncTableHandle`
  - ✅ `scan_*` methods on `AsyncDatabase` return `AsyncPolarsDataFrame`
  - ✅ Comprehensive test coverage
- ✅ **AsyncPandasDataFrame** – Async version of Pandas-style interface:
  - ✅ Wraps `AsyncDataFrame` with Pandas-style API
  - ✅ All database-interactive methods are `async` (`collect()`, `shape`, `dtypes`, `empty`, `describe()`, `info()`, `nunique()`, `value_counts()`)
  - ✅ Integrated via `.pandas()` method on `AsyncDataFrame` and `AsyncTableHandle`
  - ✅ `_AsyncLocIndexer` and `_AsyncILocIndexer` for async pandas-style indexing
  - ✅ Comprehensive test coverage

### Type Safety & Code Quality Improvements ✅ **COMPLETED**
- ✅ Fixed mypy type errors in async DataFrame implementations (`async_polars_dataframe.py`, `async_pandas_dataframe.py`, `async_table.py`)
- ✅ Fixed redundant cast errors in `mutations.py`
- ✅ Fixed mypy errors in example files (`18_pandas_interface.py`, `19_polars_interface.py`)
- ✅ Removed unused type ignore comments that became unnecessary when checking `src` and `examples` together
- ✅ All pre-commit CI checks now pass with Python 3.11 (ruff, mypy, tests, documentation validation)
- ✅ Verified all mypy checks pass (101 source files, no errors)

### Pandas-Style Interface Enhancements ✅ **COMPLETED**
- ✅ Enhanced pandas-style interface (`PandasDataFrame`) with comprehensive improvements:
  - ✅ **String Accessor** - Added `.str` accessor for pandas-style string operations (`upper()`, `lower()`, `strip()`, `contains()`, `startswith()`, `endswith()`, `replace()`, `split()`, `len()`)
  - ✅ **Improved Query Syntax** - Enhanced `query()` method to support both `=` and `==` for equality, and `AND`/`OR` keywords in addition to `&`/`|`
  - ✅ **Proper dtypes Property** - Implemented real schema inspection to return pandas-compatible dtype strings (e.g., 'int64', 'object', 'float64')
  - ✅ **Shape and Empty Properties** - Added caching for `shape` property to avoid redundant database queries
  - ✅ **Data Inspection Methods** - Added `head()`, `tail()`, `describe()`, `info()`, `nunique()`, `value_counts()` methods
  - ✅ **Fixed drop_duplicates** - Corrected implementation to properly handle `subset` parameter using GROUP BY
  - ✅ **Early Column Validation** - Added column existence validation with helpful error messages and typo suggestions
  - ✅ **Enhanced GroupBy** - Added pandas-style aggregation methods (`sum()`, `mean()`, `min()`, `max()`, `nunique()`, `first()`, `last()`)
  - ✅ **Column Access Improvements** - `df['col']` now returns `PandasColumn` with `.str` accessor support
  - ✅ Comprehensive test coverage (all tests passing)
  - ✅ Updated examples and documentation

### Documentation & Examples Improvements ✅ **COMPLETED**
- ✅ **Runnable Guide Code** - Updated all guide code blocks to be fully runnable using `sqlite:///:memory:`
  - ✅ Created automated script (`scripts/make_guides_runnable.py`) to update 131 code blocks across 8 guides
  - ✅ All code examples now include complete setup (imports, database creation, data insertion)
  - ✅ All 567 examples validated and passing
  - ✅ Guides updated: 01-getting-started, 02-migrating-from-pandas, 03-migrating-from-pyspark, 04-performance-optimization, 06-error-handling, 07-advanced-topics, 08-best-practices, 09-pandas-interface
- ✅ **New Pandas Interface Guide** - Created comprehensive guide (`guides/09-pandas-interface.md`) covering:
  - Getting started with `PandasDataFrame`
  - Column access and string operations
  - Query filtering with improved syntax
  - Data inspection methods
  - GroupBy operations
  - Merging, sorting, and data manipulation
- ✅ **Updated README and Examples** - Enhanced documentation with new pandas-style interface features

### Type Safety & Code Quality Improvements ✅ **COMPLETED**
- ✅ Fixed mypy type errors in join condition normalization (`_normalize_join_condition` in both `DataFrame` and `AsyncDataFrame`)
- ✅ Fixed redundant cast errors in `mutations.py` (`insert_rows` and `merge_rows` functions)
- ✅ Updated pre-commit CI checks script to use `python -m pytest` instead of `pytest` for correct Python environment detection
- ✅ Verified all mypy checks pass (87 source files, no errors)
- ✅ Installed all project dependencies in Python 3.11 environment (including duckdb-engine)

### SQLAlchemy ORM Model Integration ✅ **COMPLETED**
- ✅ SQLAlchemy model detection and table name extraction
- ✅ Type mapping between SQLAlchemy types and Moltres type names (bidirectional)
- ✅ Constraint extraction from SQLAlchemy models (primary keys, foreign keys, unique, check)
- ✅ Model-to-schema conversion (`model_to_schema()`) - Convert SQLAlchemy models to Moltres TableSchema
- ✅ Schema-to-table conversion (`schema_to_table()`) - Convert Moltres TableSchema to SQLAlchemy Table
- ✅ Extended `Database.create_table()` to accept SQLAlchemy model classes
- ✅ Extended `Database.table()` to accept SQLAlchemy model classes
- ✅ Extended `AsyncDatabase.create_table()` and `AsyncDatabase.table()` for async support
- ✅ Model reference storage in `TableHandle` and `AsyncTableHandle`
- ✅ Comprehensive test suite (19 tests, all passing)
- ✅ Example file (`docs/examples/17_sqlalchemy_models.py`) with real outputs
- ✅ README.md documentation with SQLAlchemy integration section
- ✅ Full backward compatibility maintained
- ✅ All mypy type errors fixed

### v0.13.0

#### Schema Management - Constraints & Indexes
- ✅ Unique constraints (`UNIQUE`) - Single and multi-column support via `unique()` helper
- ✅ Check constraints (`CHECK`) - SQL expression-based validation via `check()` helper
- ✅ Foreign key constraints (`FOREIGN KEY ... REFERENCES`) - Single and multi-column with cascade options via `foreign_key()` helper
- ✅ Indexes (`CREATE INDEX`, `DROP INDEX`) - Single and multi-column indexes, unique indexes via `create_index()` and `drop_index()` methods
- ✅ SQLAlchemy DDL Integration - All DDL operations now use SQLAlchemy's declarative API instead of raw SQL strings
- ✅ Async support for all constraint and index operations
- ✅ Comprehensive test coverage (41 tests) for constraints, indexes, and edge cases
- ✅ Updated examples demonstrating constraint and index usage

#### Schema Inspection & Reflection
- ✅ Table reflection (`db.reflect_table(name)`) - Automatically introspect table schemas from database
- ✅ Database reflection (`db.reflect()`) - Introspect all tables, views, indexes in database schema
- ✅ Schema introspection utilities (`db.get_table_names()`, `db.get_view_names()`, etc.)
- ✅ Column metadata introspection (`db.get_columns(table_name)`, etc.)
- ✅ Enhanced `ColumnInfo` dataclass with full metadata (nullable, default, primary_key, precision, scale)
- ✅ Comprehensive test coverage for both sync and async reflection methods
- ✅ Example file demonstrating reflection features (`docs/examples/14_reflection.py`)

#### FILTER Clause for Conditional Aggregation
- ✅ Extended `Expression` dataclass with `_filter` field for storing FILTER clause conditions
- ✅ Added `filter()` method to `Column` class for conditional aggregation
- ✅ Updated `ExpressionCompiler` to handle FILTER clause for all aggregation functions (sum, avg, count, min, max, count_distinct, collect_list, collect_set, corr, covar, stddev, variance)
- ✅ Implemented dialect-specific support (PostgreSQL, MySQL 8.0+ use native FILTER; SQLite uses CASE WHEN fallback)
- ✅ Added comprehensive test coverage (18 tests covering all aggregation functions, edge cases, and error handling)
- ✅ Updated documentation with examples in `docs/examples/05_groupby.py` and README.md

### v0.12.0

### Examples & Documentation
- ✅ Created comprehensive examples directory with 13 example files
- ✅ All examples verified to run with real outputs documented
- ✅ README streamlined (37% reduction, removed verbose content)
- ✅ All examples use PySpark-style function imports (`from moltres.expressions import functions as F`)

### Core Features
- ✅ 98% PySpark API compatibility for core DataFrame operations
- ✅ Raw SQL query support (`db.sql()`)
- ✅ SQL expression selection (`selectExpr()`)
- ✅ Chunked file reading for large files
- ✅ Extended function library (130+ functions)
- ✅ Comprehensive test coverage (~75%)

### Infrastructure
- ✅ Async PostgreSQL connection fixes (DSN options translation)
- ✅ Pooled async staging tables
- ✅ Performance monitoring hooks
- ✅ Comprehensive test suite with real database integration

## 📝 Notes

- Items are roughly prioritized but not strictly ordered
- Some features may depend on others
- Community feedback will help prioritize
- Breaking changes should be carefully considered
- Focus on SQL feature support rather than PySpark feature parity

## 🤝 Contributing

If you'd like to work on any of these items, please:
1. Check existing issues/PRs to avoid duplication
2. Open an issue to discuss the approach
3. Follow the contributing guidelines
4. Submit a PR with tests and documentation
