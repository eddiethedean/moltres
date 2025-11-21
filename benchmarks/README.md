# Moltres Benchmarks

This directory contains benchmark scripts for comparing Moltres performance with other data processing libraries.

## Running Benchmarks

### Basic Usage

```bash
# Run with SQLite (in-memory)
python benchmarks/benchmark.py

# Run with SQLite file
python benchmarks/benchmark.py sqlite:///benchmark.db

# Run with PostgreSQL
python benchmarks/benchmark.py postgresql://user:pass@host/dbname
```

### Prerequisites

Install optional dependencies for comparisons:

```bash
# For Pandas comparison
pip install pandas

# For Ibis comparison
pip install ibis-framework
```

## Benchmark Results

The benchmark script tests:

1. **Simple SELECT** - Selecting rows with LIMIT
2. **Filter** - Filtering rows by condition
3. **Aggregation** - Group by with aggregations
4. **Join** - Joining two tables

### Expected Results

- **Moltres:** Executes queries in the database (SQL pushdown)
  - Faster for large datasets
  - Lower memory usage
  - Scales with database capabilities

- **Pandas:** Loads data into Python memory
  - Faster for small datasets already in memory
  - Higher memory usage
  - Limited by available RAM

### Performance Characteristics

Moltres performance depends on:
- Database engine optimization
- Indexes on queried columns
- Query complexity
- Data size

For large datasets, Moltres typically outperforms Pandas because:
- Operations execute in optimized database engines
- No need to load all data into Python memory
- Database handles filtering, aggregation, and joins efficiently

## Custom Benchmarks

You can create custom benchmarks by modifying `benchmark.py`:

```python
def my_custom_benchmark(db):
    """Your custom benchmark."""
    df = db.table("my_table").select().where(col("value") > 100)
    return df.collect()

# Add to run_benchmarks function
results.append(benchmark_query(
    "My Custom Benchmark",
    lambda: my_custom_benchmark(db),
))
```

## Interpreting Results

- **Mean time:** Average execution time across iterations
- **Median time:** Middle execution time (less affected by outliers)
- **Min/Max:** Best and worst case times
- **Standard deviation:** Consistency of results

Lower times are better. Compare Moltres results with Pandas/Ibis to understand trade-offs.

## Notes

- Results vary based on hardware, database engine, and data size
- First query may be slower (connection setup, query plan caching)
- Database indexes significantly impact performance
- Network latency affects remote database connections

For production performance tuning, see [Performance Guide](../docs/PERFORMANCE.md).

