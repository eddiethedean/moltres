# Performance Service Level Agreements (SLAs)

This document defines performance expectations and SLAs for Moltres operations. These benchmarks are based on representative datasets and typical production workloads.

## Benchmark Methodology

- **Test Environment**: Standard development machine (8GB RAM, 4 CPU cores)
- **Database**: SQLite (in-memory) for baseline, PostgreSQL for production benchmarks
- **Dataset Sizes**: 
  - Small: 1K-10K rows
  - Medium: 10K-100K rows
  - Large: 100K-1M rows
  - Very Large: 1M+ rows (streaming mode)

## Query Performance SLAs

### Simple SELECT Operations

**Target**: < 100ms for 10K rows, < 1s for 100K rows

- **Small datasets (1K-10K rows)**: < 100ms
- **Medium datasets (10K-100K rows)**: < 500ms
- **Large datasets (100K-1M rows)**: < 2s
- **Very large datasets (1M+ rows)**: Streaming mode recommended, < 10s per chunk

**Notes**:
- Performance depends heavily on database engine and indexing
- SQLite (in-memory) provides baseline performance
- PostgreSQL/MySQL with proper indexing can achieve better results
- Network latency not included in these benchmarks

### Filter Operations

**Target**: < 150ms for 10K rows with indexed filter column

- **Indexed column filter**: < 150ms for 10K rows
- **Non-indexed column filter**: < 500ms for 10K rows
- **Complex predicates (AND/OR)**: < 300ms for 10K rows

**Recommendations**:
- Create indexes on frequently filtered columns
- Use parameterized queries for repeated filters
- Consider partitioning for very large tables

### Aggregation Operations

**Target**: < 200ms for 100K rows grouped by indexed column

- **Simple aggregations (SUM, AVG, COUNT)**: < 200ms for 100K rows
- **Group by indexed column**: < 200ms for 100K rows
- **Group by non-indexed column**: < 1s for 100K rows
- **Multiple aggregations**: < 500ms for 100K rows

**Recommendations**:
- Index GROUP BY columns for better performance
- Use approximate aggregations (e.g., `percentile_approx`) for very large datasets
- Consider materialized views for frequently aggregated data

### Join Operations

**Target**: < 500ms for 10K × 10K row joins with indexed join keys

- **Inner join (indexed keys)**: < 500ms for 10K × 10K rows
- **Left/Right join (indexed keys)**: < 600ms for 10K × 10K rows
- **Join without indexes**: < 2s for 10K × 10K rows
- **Multiple joins**: < 1s per additional join

**Recommendations**:
- Always index foreign key columns
- Use appropriate join types (INNER vs LEFT) based on data requirements
- Consider denormalization for frequently joined tables

### Window Functions

**Target**: < 1s for 100K rows with simple window

- **ROW_NUMBER, RANK**: < 1s for 100K rows
- **LAG, LEAD**: < 1s for 100K rows
- **Complex windows (PARTITION BY + ORDER BY)**: < 2s for 100K rows

**Notes**:
- Window function performance varies significantly by database engine
- PostgreSQL generally provides better window function performance than SQLite
- Consider materialized views for expensive window calculations

## Write Performance SLAs

### INSERT Operations

**Target**: > 1K rows/second for batch inserts

- **Single row inserts**: Not recommended for production (use batch)
- **Batch inserts (100 rows)**: < 100ms
- **Batch inserts (1K rows)**: < 500ms
- **Batch inserts (10K rows)**: < 2s

**Recommendations**:
- Always use batch inserts for multiple rows
- Use `createDataFrame()` for bulk data loading
- Consider `COPY` or bulk insert for very large datasets (database-specific)

### UPDATE Operations

**Target**: < 500ms for 10K rows with indexed WHERE clause

- **Single row update (indexed WHERE)**: < 10ms
- **Bulk update (10K rows, indexed WHERE)**: < 500ms
- **Bulk update (100K rows, indexed WHERE)**: < 5s

**Recommendations**:
- Index columns used in WHERE clauses
- Use transactions for multiple updates
- Consider batch updates for very large datasets

### DELETE Operations

**Target**: < 500ms for 10K rows with indexed WHERE clause

- **Single row delete (indexed WHERE)**: < 10ms
- **Bulk delete (10K rows, indexed WHERE)**: < 500ms
- **Bulk delete (100K rows, indexed WHERE)**: < 5s

**Recommendations**:
- Index columns used in WHERE clauses
- Use transactions for multiple deletes
- Consider soft deletes (UPDATE flag) instead of hard deletes for audit trails

## File I/O Performance SLAs

### CSV Reading

**Target**: > 10K rows/second for streaming reads

- **Small files (< 10MB)**: < 1s
- **Medium files (10-100MB)**: < 10s (streaming)
- **Large files (100MB-1GB)**: < 2 minutes (streaming)
- **Very large files (1GB+)**: Streaming mode required, < 5 minutes per GB

**Recommendations**:
- Use streaming mode (`stream=True`) for files > 10MB
- Compress files (gzip) for better I/O performance
- Consider Parquet format for better compression and performance

### JSON Reading

**Target**: > 5K rows/second for streaming reads

- **Small files (< 10MB)**: < 2s
- **Medium files (10-100MB)**: < 20s (streaming)
- **Large files (100MB-1GB)**: < 4 minutes (streaming)

**Recommendations**:
- Use JSONL format (one object per line) for better streaming performance
- Use streaming mode for files > 10MB
- Consider schema inference optimization for large files

### Parquet Reading

**Target**: > 50K rows/second

- **Small files (< 10MB)**: < 500ms
- **Medium files (10-100MB)**: < 5s
- **Large files (100MB-1GB)**: < 30s

**Recommendations**:
- Parquet provides best performance for analytical workloads
- Use column pruning to read only needed columns
- Consider partitioning for very large datasets

### CSV Writing

**Target**: > 10K rows/second

- **Small datasets (< 10K rows)**: < 1s
- **Medium datasets (10K-100K rows)**: < 10s
- **Large datasets (100K-1M rows)**: < 2 minutes

### JSON Writing

**Target**: > 5K rows/second

- **Small datasets (< 10K rows)**: < 2s
- **Medium datasets (10K-100K rows)**: < 20s
- **Large datasets (100K-1M rows)**: < 4 minutes

### Parquet Writing

**Target**: > 20K rows/second

- **Small datasets (< 10K rows)**: < 500ms
- **Medium datasets (10K-100K rows)**: < 5s
- **Large datasets (100K-1M rows)**: < 30s

## Connection Pool Performance

### Connection Acquisition

**Target**: < 10ms from pool

- **Connection from pool**: < 10ms
- **New connection (if pool exhausted)**: < 100ms
- **Connection timeout**: 30s default

### Pool Sizing Recommendations

- **Small applications**: `pool_size=5, max_overflow=10`
- **Medium applications**: `pool_size=10, max_overflow=20`
- **Large applications**: `pool_size=20, max_overflow=40`

**Formula**: `pool_size = (expected_concurrent_requests / avg_query_time) * 1.2`

## Memory Usage SLAs

### Query Execution

**Target**: Constant memory usage regardless of dataset size (with streaming)

- **Streaming queries**: O(1) memory (constant)
- **Non-streaming queries**: O(n) memory (proportional to result size)
- **Recommended**: Always use streaming for datasets > 100K rows

### File Reading

**Target**: Constant memory usage with streaming

- **Streaming mode**: O(chunk_size) memory (default: 10K rows)
- **Non-streaming mode**: O(file_size) memory (not recommended for large files)

## Performance Monitoring

### Metrics to Track

1. **Query Latency**: P50, P95, P99 percentiles
2. **Throughput**: Queries per second
3. **Error Rate**: Failed queries per total queries
4. **Connection Pool Usage**: Active connections, pool size, wait time
5. **Memory Usage**: Peak memory per operation
6. **File I/O**: Read/write throughput

### Performance Hooks

Moltres provides performance monitoring hooks:

```python
from moltres import register_performance_hook

def my_hook(event: str, sql: str, duration: float, metadata: dict):
    if duration > 1.0:  # Log slow queries
        print(f"Slow query ({duration:.2f}s): {sql}")

register_performance_hook(my_hook)
```

## Performance Tuning Tips

1. **Indexing**: Create indexes on frequently filtered, joined, or grouped columns
2. **Connection Pooling**: Configure appropriate pool sizes for your workload
3. **Streaming**: Always use streaming for large datasets
4. **Batch Operations**: Use batch inserts/updates instead of single-row operations
5. **Query Optimization**: Use EXPLAIN to analyze query plans
6. **Database-Specific Optimizations**: Leverage database-specific features (e.g., PostgreSQL JSONB, MySQL JSON functions)

## Regression Testing

Performance benchmarks are run as part of CI to detect regressions:

- **Baseline**: SQLite in-memory database
- **Frequency**: On every commit to main branch
- **Threshold**: 20% performance degradation triggers alert

## Notes

- All benchmarks assume optimal conditions (indexed columns, sufficient memory, no network latency)
- Real-world performance may vary based on:
  - Database engine and version
  - Hardware specifications
  - Network latency (for remote databases)
  - Concurrent load
  - Data distribution and cardinality
- These SLAs are targets, not guarantees. Actual performance depends on many factors outside Moltres's control.

