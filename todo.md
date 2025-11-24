# TODO

This file tracks planned features, improvements, and tasks for Moltres.

> **Note:** Moltres focuses on providing full SQL feature support through a DataFrame API, not replicating every PySpark feature. Features are included only if they map to SQL/SQLAlchemy capabilities and align with SQL pushdown execution.

## 🎯 High Priority

### Core Features
- [ ] `QUALIFY` clause for filtering window function results without subqueries (SQL standard, PostgreSQL 12+, BigQuery, Snowflake)
- [ ] `UNNEST()` / table-valued functions for array/JSON expansion in FROM clause (`UNNEST(array)`, `jsonb_array_elements()`, `jsonb_each()` in FROM) - dialect-specific
  - Note: `explode()` API is complete, but SQL compilation needs table-valued function support
- [ ] `FILTER` clause for conditional aggregation (`COUNT(*) FILTER (WHERE condition)`) - SQL standard (PostgreSQL, MySQL 8.0+, SQL Server, Oracle)
- [ ] `DISTINCT ON` for selecting distinct rows based on specific columns (PostgreSQL-specific)

### SQL Dialects
- [ ] Better MySQL-specific optimizations
- [ ] Oracle database support
- [ ] SQL Server support
- [ ] BigQuery support
- [ ] Redshift support
- [ ] Snowflake support
- [ ] DuckDB support

### Schema Management
- [ ] Foreign key constraints (`FOREIGN KEY ... REFERENCES`) - SQL standard
- [ ] Unique constraints (`UNIQUE`) - SQL standard
- [ ] Check constraints (`CHECK`) - SQL standard
- [ ] Indexes (`CREATE INDEX`, `DROP INDEX`) - SQL standard
- [ ] Views (`CREATE VIEW`, `DROP VIEW`) - SQL standard
- [ ] Alter table operations (`ALTER TABLE ADD COLUMN`, `DROP COLUMN`, etc.) - SQL standard

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
- [ ] SQLAlchemy TypeEngine integration - Leverage SQLAlchemy's type system for better type mapping
- [ ] Type coercion utilities - Automatic type conversion based on SQLAlchemy TypeEngine
- [ ] Dialect-specific type mapping - Use SQLAlchemy's dialect-specific type mapping

### File Formats
- [ ] Excel/ODS file support
- [ ] Avro file support
- [ ] ORC file support
- [ ] Delta Lake support
- [ ] Arrow IPC format support
- [ ] Better compression options for all formats

### Schema Inspection
- [ ] Table reflection (`db.reflect_table(name)`) - Automatically introspect table schemas from database
- [ ] Database reflection (`db.reflect()`) - Introspect all tables, views, indexes in database schema
- [ ] Schema introspection utilities (`db.get_table_names()`, `db.get_view_names()`, etc.)
- [ ] Column metadata introspection (`db.get_columns(table_name)`, etc.)

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
- [ ] Better mypy coverage (reduce Any types)

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
- [ ] Create docs/MIGRATION_SQLALCHEMY.md guide showing how to replace SQLAlchemy ORM operations with Moltres DataFrame CRUD
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

## ✅ Recently Completed (v0.12.0)

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
