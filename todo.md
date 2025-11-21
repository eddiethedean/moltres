# TODO

This file tracks planned features, improvements, and tasks for Moltres.

## 🚀 Features

> **Note:** Moltres focuses on providing full SQL feature support through a DataFrame API, not replicating every PySpark feature. Features are included only if they map to SQL/SQLAlchemy capabilities and align with SQL pushdown execution. PySpark-specific features that don't translate to SQL (e.g., cluster distribution, in-memory caching) are not included.

### DataFrame Operations
- [x] `select()` - Project columns [tested]
- [x] `where()` / `filter()` - Filter rows [tested]
- [x] `join()` - Join with other DataFrames (inner, left, right, outer) [tested]
- [x] `group_by()` / `groupBy()` - Group rows [tested]
- [x] `agg()` - Aggregate functions [tested]
- [x] `order_by()` / `orderBy()` - Sort rows [tested]
- [x] `limit()` - Limit number of rows [tested]
- [x] `withColumn()` - Add or replace columns
- [x] `withColumnRenamed()` - Rename columns
- [x] `drop()` - Drop columns
- [x] `union()` / `unionAll()` - Union operations
- [x] `distinct()` - Distinct rows
- [x] `dropDuplicates()` - Remove duplicate rows
- [x] `dropna()` - Remove null values [tested]
- [x] `intersect()` and `except()` operations for set operations (SQL INTERSECT/EXCEPT) [tested]
- [ ] `pivot()` / `unpivot()` for data reshaping (SQL PIVOT/UNPIVOT where supported)
- [ ] `explode()` / `flatten()` for array/JSON column expansion (SQL JSON functions)
- [ ] `sample()` for random sampling (SQL TABLESAMPLE where supported)
- [x] `fillna()` for null handling (SQL COALESCE/CASE expressions) [tested]
- [ ] `na.drop()` / `na.fill()` for null value operations (SQL WHERE/COALESCE)

### Window Functions
- [x] Window frame specification (ROWS/RANGE BETWEEN) - `rowsBetween()`, `rangeBetween()` [tested]
- [x] `over()` - Window function support [tested]
- [x] `row_number()` - Row number window function [tested]
- [x] `rank()` - Rank window function [tested]
- [x] `dense_rank()` - Dense rank window function [tested]
- [x] `lag()` - Lag window function [tested]
- [x] `lead()` - Lead window function [tested]
- [x] `first_value()` - First value window function [tested]
- [x] `last_value()` - Last value window function [tested]
- [x] `percent_rank()`, `cume_dist()` window functions (SQL standard window functions) [tested]
- [x] `nth_value()` window function (SQL standard window function) [tested]
- [x] `ntile()` window function for quantile bucketing (SQL standard window function) [tested]

### Column Expressions
- [x] `cast()` - Type casting [tested]
- [x] `is_null()` / `is_not_null()` - Null checking functions [tested]
- [x] `coalesce()` - Coalesce function (supports multiple arguments) [tested]
- [x] `like()` / `ilike()` - Pattern matching [tested]
- [x] `between()` - Range checking [tested]
- [x] `greatest()` / `least()` functions for multiple values (SQL standard functions - already implemented) [tested]
- [ ] Array/JSON functions (`array()`, `array_length()`, `json_extract()`, etc.) - SQL standard/dialect-specific
- [x] Regular expression functions (`regexp_extract()`, `regexp_replace()`, etc.) - SQL standard/dialect-specific [tested]
- [x] More string functions (`split()`, `array_join()`, `repeat()`, etc.) - SQL standard/dialect-specific [tested - split()]
- [ ] Type casting improvements (`cast()` with more type support) - SQL standard
- [ ] `isnull()` / `isnotnull()` / `isnan()` null checking functions (aliases for is_null/is_not_null) - Note: `isnan()` is implemented, but `isnull()`/`isnotnull()` aliases are not yet

### Joins
- [x] Inner join (default) [tested]
- [x] Left join (`how="left"`) [tested]
- [x] Right join (`how="right"`) [tested]
- [x] Outer join (`how="outer"`) [tested]
- [x] Multiple join conditions with column pairs [tested]
- [x] Cross join support (SQL CROSS JOIN) [tested]
- [ ] Semi-join and anti-join support (SQL EXISTS/NOT EXISTS subqueries)
- [ ] Join hints for query optimization (dialect-specific, e.g., PostgreSQL, MySQL)
- [ ] Multiple join conditions with complex predicates (beyond column pairs) - SQL standard

### Aggregations
- [x] `sum()` - Sum aggregation [tested]
- [x] `avg()` - Average aggregation [tested]
- [x] `count()` - Count aggregation [tested]
- [x] `min()` - Minimum aggregation [tested]
- [x] `max()` - Maximum aggregation [tested]
- [x] `count_distinct()` - Distinct count aggregation [tested]
- [ ] `collect_list()` / `collect_set()` for array aggregation (SQL ARRAY_AGG where supported, e.g., PostgreSQL)
- [ ] `percentile()` / `percentile_approx()` functions (SQL PERCENTILE_CONT/PERCENTILE_DISC where supported)
- [x] `stddev()` / `variance()` statistical functions (SQL STDDEV/VARIANCE - standard aggregate functions) [tested - SQLite incompatible]
- [ ] `skewness()` / `kurtosis()` higher-order statistics (dialect-specific, may require custom SQL)
- [x] `corr()` / `covar()` correlation functions (SQL CORR/COVAR - standard aggregate functions) [tested - SQLite incompatible]

### Data Types
- [ ] Better support for complex types (arrays, maps, structs) - SQL standard/dialect-specific
- [ ] Decimal/Numeric type with precision (SQL DECIMAL/NUMERIC - standard)
- [ ] UUID type support (dialect-specific, e.g., PostgreSQL UUID)
- [ ] JSON/JSONB type support (dialect-specific, e.g., PostgreSQL JSONB)
- [ ] Date/Time interval types (SQL INTERVAL - standard)

## 📊 File Formats

### Reading
- [x] CSV file support [tested]
- [x] JSON file support (array of objects) [tested]
- [x] JSONL file support (one JSON object per line) [tested]
- [x] Parquet file support (requires pandas and pyarrow) [tested]
- [x] Text file support (one line per row) [tested]
- [x] Streaming support for all formats [tested]
- [ ] Excel/ODS file support
- [ ] Avro file support
- [ ] ORC file support
- [ ] Delta Lake support
- [ ] Arrow IPC format support
- [ ] Compressed file reading (gzip, bz2, xz, etc.)

### Writing
- [x] CSV file writing [tested]
- [x] JSON file writing [tested]
- [x] JSONL file writing [tested]
- [x] Parquet file writing (requires pandas and pyarrow) [tested]
- [x] Streaming support for all formats [tested]
- [x] Partitioned writing (`partitionBy()`) [tested]
- [ ] Excel/ODS file writing
- [ ] Avro file writing
- [ ] ORC file writing
- [ ] Better compression options for all formats
- [ ] Partitioned writing improvements (more strategies)

## 🗄️ Database Features

### CRUD Operations
- [x] `insert()` - Insert rows (batch optimized) [tested]
- [x] `update()` - Update rows with WHERE clause [tested]
- [x] `delete()` - Delete rows with WHERE clause [tested]
- [x] Async CRUD support (`async insert/update/delete`) [tested]

### SQL Dialects
- [x] SQLite support [tested]
- [x] PostgreSQL support [tested - via SQLAlchemy]
- [x] MySQL support (basic) [tested - via SQLAlchemy]
- [x] ANSI SQL fallback for other SQLAlchemy-supported databases [tested - via SQLAlchemy]
- [ ] Better MySQL-specific optimizations
- [ ] Oracle database support
- [ ] SQL Server support
- [ ] BigQuery support
- [ ] Redshift support
- [ ] Snowflake support
- [ ] DuckDB support

### SQL Features
- [x] Common Table Expressions (CTEs) - WITH clauses (SQL standard) [tested]
- [ ] Recursive CTEs (SQL standard WITH RECURSIVE)
- [ ] Subqueries in SELECT, FROM, WHERE clauses (SQL standard)
- [ ] EXISTS / NOT EXISTS subqueries (SQL standard)
- [ ] LATERAL joins (SQL standard, PostgreSQL/MySQL support)
- [ ] PIVOT / UNPIVOT SQL operations (dialect-specific, e.g., SQL Server, Oracle)
- [ ] MERGE / UPSERT operations (SQL standard MERGE, PostgreSQL INSERT ... ON CONFLICT)
- [ ] Stored procedure support (dialect-specific via SQLAlchemy)

### Performance
- [x] Performance monitoring hooks (`register_performance_hook()`) [tested]
- [x] Connection pooling (`pool_size`, `pool_pre_ping`, `pool_recycle`, etc.) [tested]
- [x] Batch inserts for better performance [tested]
- [x] Streaming support for large datasets [tested]
- [x] Environment variable configuration (12-factor app friendly) [tested]
- [ ] Query plan optimization (SQL EXPLAIN integration)
- [ ] Index hints (dialect-specific, e.g., MySQL USE INDEX, PostgreSQL)
- [ ] Query result caching (application-level, not SQL feature)
- [ ] Connection pool monitoring (SQLAlchemy pool metrics)
- [ ] Query timeout configuration (SQLAlchemy/dialect-specific)
- [ ] Batch size auto-tuning (application-level optimization)

## 🔧 Developer Experience

### Type Safety
- [x] Full type hints across codebase
- [x] Mypy strict mode enabled
- [x] Type stubs for PyArrow (`stubs/pyarrow/`)
- [x] PEP 561 compliance (`py.typed` marker)
- [x] Comprehensive type annotations
- [ ] Better type inference for schemas
- [ ] Generic DataFrame types with schema
- [ ] Type-safe column references
- [ ] Better mypy coverage (reduce Any types)

### Error Handling
- [x] ValidationError for validation issues
- [x] ExecutionError for SQL execution failures
- [ ] More specific error types
- [ ] Better error messages with suggestions
- [ ] Error recovery strategies
- [ ] Validation error aggregation

### Documentation
- [ ] API reference documentation
- [x] More examples in docs/ (added showcase examples and audience-specific use cases)
- [ ] Migration guide from PySpark
- [ ] Performance tuning guide
- [ ] Best practices guide
- [ ] Video tutorials

### Testing
- [x] Comprehensive test suite (113+ test cases)
- [x] Integration tests with real databases (SQLite, PostgreSQL, MySQL)
- [x] Security tests (SQL injection prevention)
- [x] Example validation tests
- [ ] Property-based testing with Hypothesis
- [ ] Performance benchmarks
- [ ] Load testing
- [ ] Test coverage improvements

## 🛠️ Infrastructure

### CI/CD
- [ ] Automated release process
- [ ] Version bump automation
- [ ] Changelog generation
- [ ] Documentation deployment
- [ ] Performance regression testing

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

## 🐛 Bug Fixes & Improvements

### Known Issues
- [x] Fixed all mypy warnings (strict mode compliance achieved in 0.4.0)
- [ ] Improve error messages for common mistakes
- [ ] Better handling of edge cases in file readers
- [x] Memory optimization for large file reads (streaming support implemented)

### Code Refactoring
- [ ] Reduce code duplication between sync/async implementations
- [ ] Better abstraction for SQL dialect differences
- [ ] Improve logical plan optimization
- [ ] Refactor compiler for better maintainability

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

## 🔐 Security

### Enhancements
- [x] SQL injection prevention (parameterized queries, identifier validation) [tested]
- [x] SQL injection testing suite (`tests/security/test_sql_injection.py`) [tested]
- [ ] SQL injection testing suite expansion
- [ ] Rate limiting support
- [ ] Connection encryption verification
- [ ] Audit logging
- [ ] Access control integration points

## 🌐 Ecosystem

### Integrations
- [x] Pandas interop (fetch_format="pandas") [tested]
- [x] Polars interop (fetch_format="polars") [tested]
- [x] Async/await support (full async API) [tested]
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

### Testing & Validation
- [ ] Add tests for new showcase examples (memory-efficient operations, CRUD workflows) in tests/examples/ to ensure they compile and work correctly
- [ ] Create examples/ directory with standalone runnable scripts demonstrating key use cases (memory efficiency, CRUD workflows, ETL pipelines)
- [ ] Create performance comparison script showing memory usage and execution time vs Pandas for large datasets (10M+ rows)

### Migration Guides
- [ ] Create docs/MIGRATION_PANDAS.md guide showing how to convert common Pandas workflows to Moltres with before/after examples
- [ ] Create docs/MIGRATION_SQLALCHEMY.md guide showing how to replace SQLAlchemy ORM operations with Moltres DataFrame CRUD
- [ ] Create docs/MIGRATION_SPARK.md guide for teams moving from PySpark to Moltres (no cluster required, same API style)

### Integration Examples
- [ ] Create example showing Moltres integration with FastAPI for building data APIs with DataFrame operations
- [ ] Create Jupyter notebook examples demonstrating interactive data analysis with Moltres (show SQL pushdown in action)
- [ ] Create example showing how Moltres can complement or replace dbt transformations using Python DataFrame API

### Performance & Benchmarks
- [ ] Create docs/PERFORMANCE.md with detailed benchmarks comparing memory usage and query time vs Pandas/Ibis/SQLAlchemy with actual numbers

### Documentation Enhancements
- [ ] Create docs/FAQ.md addressing common questions: when to use Moltres vs Pandas vs SQLAlchemy vs Ibis, performance considerations, etc.
- [ ] Create docs/DEBUGGING.md guide on error handling and debugging, showing how to interpret SQL errors from DataFrame operations
- [ ] Create docs/DEPLOYMENT.md guide for production use: connection pooling, monitoring, best practices, scaling considerations
- [ ] Ensure all CRUD operations (insert, update, delete) have comprehensive docstrings with examples in the codebase
- [ ] Review and enhance DataFrame method docstrings to include examples showing SQL pushdown execution
- [x] Created docs/WHY_MOLTRES.md explaining the gap Moltres fills and unique positioning
- [x] Added showcase examples (memory-efficient operations, CRUD workflows) to docs/EXAMPLES.md
- [x] Added targeted examples for different audiences (Data Engineers, Backend Developers, Analytics Engineers, etc.) to docs/EXAMPLES.md
- [x] Added SQL query logging example to docs/EXAMPLES.md demonstrating pushdown execution
- [ ] Enhance docs/index.md with better organization, search-friendly structure, and clear navigation to all documentation

### Marketing & Outreach
- [ ] Update CHANGELOG.md to highlight unique positioning (DataFrame + SQL pushdown + CRUD) in upcoming release notes
- [ ] Add badges to README for PyPI downloads, GitHub stars, test coverage to build credibility and social proof
- [x] Updated README with advocacy-focused messaging emphasizing unique DataFrame + SQL pushdown + CRUD combination
- [x] Added comparison table to README showing Moltres vs Pandas/Polars/Ibis/SQLAlchemy
- [x] Enhanced "Why Moltres" section in README explaining the gap in Python ecosystem
- [x] Updated pyproject.toml description and keywords to reflect unique positioning
- [x] Added CRUD examples to Quick Start section in README
- [ ] Draft blog post announcing Moltres unique positioning: The Missing DataFrame Layer for SQL in Python (for Medium/Dev.to)
- [ ] Create script/storyboard for video tutorial demonstrating memory-efficient operations and CRUD workflows (can be animated GIF or video)
- [ ] Create template for documenting real-world use cases and case studies showing Moltres solving actual problems
- [x] Added "Use Cases by Audience" section to docs/EXAMPLES.md with examples for Data Engineers, Backend Developers, Analytics Engineers, Product Engineers, and Spark migration teams
- [ ] Create social media content (Twitter/LinkedIn posts) highlighting unique positioning and key differentiators

## 📝 Notes

- Items are roughly prioritized but not strictly ordered
- Some features may depend on others
- Community feedback will help prioritize
- Breaking changes should be carefully considered

## 🤝 Contributing

If you'd like to work on any of these items, please:
1. Check existing issues/PRs to avoid duplication
2. Open an issue to discuss the approach
3. Follow the contributing guidelines
4. Submit a PR with tests and documentation

