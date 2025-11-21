# TODO

This file tracks planned features, improvements, and tasks for Moltres.

## 🚀 Features

### DataFrame Operations
- [ ] `intersect()` and `except()` operations for set operations
- [ ] `pivot()` / `unpivot()` for data reshaping
- [ ] `explode()` / `flatten()` for array/JSON column expansion
- [ ] `cache()` / `persist()` for materializing DataFrames
- [ ] `repartition()` / `coalesce()` for data distribution (if applicable)
- [ ] `sample()` for random sampling
- [ ] `fillna()` / `dropna()` for null handling
- [ ] `na.drop()` / `na.fill()` for null value operations

### Window Functions
- [ ] Window frame specification (ROWS/RANGE BETWEEN)
- [ ] `percent_rank()`, `cume_dist()` window functions
- [ ] `nth_value()` window function
- [ ] `ntile()` window function for quantile bucketing

### Column Expressions
- [ ] Array/JSON functions (`array()`, `array_length()`, `json_extract()`, etc.)
- [ ] Regular expression functions (`regexp_extract()`, `regexp_replace()`, etc.)
- [ ] More string functions (`split()`, `array_join()`, `repeat()`, etc.)
- [ ] Type casting improvements (`cast()` with more type support)
- [ ] `isnull()` / `isnotnull()` / `isnan()` null checking functions
- [ ] `coalesce()` improvements with multiple arguments
- [ ] `greatest()` / `least()` functions for multiple values

### Joins
- [ ] Cross join support
- [ ] Semi-join and anti-join support
- [ ] Join hints for query optimization
- [ ] Multiple join conditions with complex predicates

### Aggregations
- [ ] `collect_list()` / `collect_set()` for array aggregation
- [ ] `percentile()` / `percentile_approx()` functions
- [ ] `stddev()` / `variance()` statistical functions
- [ ] `skewness()` / `kurtosis()` higher-order statistics
- [ ] `corr()` / `covar()` correlation functions

### Data Types
- [ ] Better support for complex types (arrays, maps, structs)
- [ ] Decimal/Numeric type with precision
- [ ] UUID type support
- [ ] JSON/JSONB type support
- [ ] Date/Time interval types

## 📊 File Formats

### Reading
- [ ] Excel/ODS file support
- [ ] Avro file support
- [ ] ORC file support
- [ ] Delta Lake support
- [ ] Arrow IPC format support
- [ ] Compressed file reading (gzip, bz2, xz, etc.)

### Writing
- [ ] Excel/ODS file writing
- [ ] Avro file writing
- [ ] ORC file writing
- [ ] Better compression options for all formats
- [ ] Partitioned writing improvements (more strategies)

## 🗄️ Database Features

### SQL Dialects
- [ ] Better MySQL-specific optimizations
- [ ] Oracle database support
- [ ] SQL Server support
- [ ] BigQuery support
- [ ] Redshift support
- [ ] Snowflake support
- [ ] DuckDB support

### SQL Features
- [ ] Common Table Expressions (CTEs) - WITH clauses
- [ ] Recursive CTEs
- [ ] Subqueries in SELECT, FROM, WHERE clauses
- [ ] EXISTS / NOT EXISTS subqueries
- [ ] LATERAL joins
- [ ] PIVOT / UNPIVOT SQL operations
- [ ] MERGE / UPSERT operations
- [ ] Stored procedure support

### Performance
- [ ] Query plan optimization
- [ ] Index hints
- [ ] Query result caching
- [ ] Connection pool monitoring
- [ ] Query timeout configuration
- [ ] Batch size auto-tuning

## 🔧 Developer Experience

### Type Safety
- [ ] Better type inference for schemas
- [ ] Generic DataFrame types with schema
- [ ] Type-safe column references
- [ ] Better mypy coverage (reduce Any types)
- [ ] Type stubs for better IDE support

### Error Handling
- [ ] More specific error types
- [ ] Better error messages with suggestions
- [ ] Error recovery strategies
- [ ] Validation error aggregation

### Documentation
- [ ] API reference documentation
- [ ] More examples in docs/
- [ ] Migration guide from PySpark
- [ ] Performance tuning guide
- [ ] Best practices guide
- [ ] Video tutorials

### Testing
- [ ] Property-based testing with Hypothesis
- [ ] Performance benchmarks
- [ ] Load testing
- [ ] Integration tests with real databases
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
- [ ] Review and fix remaining mypy warnings
- [ ] Improve error messages for common mistakes
- [ ] Better handling of edge cases in file readers
- [ ] Memory optimization for large file reads

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
- [ ] Polars interop improvements
- [ ] Pandas interop improvements

### Tools
- [ ] CLI tool for common operations
- [ ] Query builder GUI
- [ ] Schema migration tool
- [ ] Data validation framework integration

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

