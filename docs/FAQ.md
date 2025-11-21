# Frequently Asked Questions (FAQ)

Common questions about Moltres and their answers.

## General Questions

### What is Moltres?

Moltres is a Python library that provides a PySpark-like DataFrame API for SQL databases. It compiles DataFrame operations to SQL and executes them in your database, providing SQL pushdown without requiring a Spark cluster.

### How is Moltres different from PySpark?

- **No cluster required:** Moltres works with any SQL database, no Spark cluster needed
- **SQL pushdown:** Operations are compiled to SQL and executed in the database
- **Simpler deployment:** Just connect to your existing database
- **Lower overhead:** No cluster management or resource allocation

### How is Moltres different from Pandas?

- **SQL execution:** Moltres executes queries in the database, not in Python memory
- **Lazy evaluation:** Queries are optimized before execution
- **Scalability:** Can handle larger datasets (limited by database, not Python memory)
- **SQL pushdown:** Operations compile to SQL for better performance

### Which databases are supported?

Moltres supports any database with a SQLAlchemy driver, including:
- SQLite
- PostgreSQL
- MySQL
- SQL Server
- Oracle
- And many others

## Usage Questions

### How do I connect to a database?

```python
from moltres import connect

# SQLite
db = connect("sqlite:///example.db")

# PostgreSQL
db = connect("postgresql://user:pass@host/dbname")

# MySQL
db = connect("mysql://user:pass@host/dbname")
```

### How do I read data from a table?

```python
# Select all columns
df = db.table("users").select()

# Select specific columns
df = db.table("users").select("id", "name", "email")

# With filtering
from moltres import col
df = db.table("users").select().where(col("active") == True)
```

### How do I read data from files?

```python
# CSV
df = db.load.csv("data.csv")

# JSON
df = db.load.json("data.json")

# Parquet
df = db.load.parquet("data.parquet")

# With options
df = db.load.csv("data.csv", header=True, delimiter=",")
```

### How do I filter rows?

```python
from moltres import col

# Single condition
df = db.table("users").select().where(col("age") > 18)

# Multiple conditions
df = db.table("users").select().where(
    (col("age") > 18) & (col("active") == True)
)
```

### How do I join tables?

```python
customers = db.table("customers").select()
orders = db.table("orders").select()

# Inner join
df = customers.join(orders, on=[("id", "customer_id")], how="inner")

# Left join
df = customers.join(orders, on=[("id", "customer_id")], how="left")
```

### How do I aggregate data?

```python
from moltres import col
from moltres.expressions.functions import sum, avg, count

df = (
    db.table("orders")
    .select()
    .group_by("category")
    .agg(
        sum(col("amount")).alias("total"),
        avg(col("amount")).alias("average"),
        count("*").alias("count")
    )
)
```

### How do I execute a query and get results?

```python
# Collect all results
results = df.collect()  # Returns list of dicts

# Stream results (for large datasets)
for chunk in df.collect(stream=True):
    process_chunk(chunk)

# Show first few rows
df.show(10)
```

### How do I see the generated SQL?

```python
sql = df.to_sql()
print(sql)
```

## Performance Questions

### Why is my query slow?

Common causes:
1. **Missing indexes** - Create indexes on filtered/joined columns
2. **Full table scans** - Add WHERE clauses to filter early
3. **Large result sets** - Use LIMIT or streaming
4. **Inefficient joins** - Ensure join columns are indexed

See [Performance Guide](./PERFORMANCE.md) for details.

### How do I optimize queries?

1. Create indexes on frequently queried columns
2. Filter early in the query chain
3. Select only needed columns
4. Use LIMIT for exploration
5. Configure connection pooling

See [Performance Guide](./PERFORMANCE.md) for more tips.

### Does Moltres cache results?

No, Moltres doesn't cache results. Each query executes fresh in the database. For caching, use:
- Database query cache (if supported)
- Application-level caching (Redis, Memcached)
- Materialized views in the database

## Error Questions

### What does "CompilationError" mean?

A `CompilationError` means Moltres couldn't compile your DataFrame operation to SQL. Common causes:
- Unsupported operation for your SQL dialect
- Invalid column references
- Missing join conditions

Check the error message and suggestion for details.

### What does "ExecutionError" mean?

An `ExecutionError` means the SQL query failed to execute. Common causes:
- Table/column doesn't exist
- SQL syntax error
- Data type mismatch
- Constraint violation

Check the error message and suggestion for details.

### How do I debug query errors?

1. Use `df.to_sql()` to see the generated SQL
2. Check the error message and suggestion
3. Verify table/column names exist
4. Test the SQL directly in your database
5. See [Debugging Guide](./DEBUGGING.md) for more help

## Configuration Questions

### How do I configure connection pooling?

```python
db = connect(
    "postgresql://user:pass@host/dbname",
    pool_size=10,        # Number of connections
    max_overflow=5,      # Additional connections
    pool_timeout=30,     # Timeout in seconds
    pool_pre_ping=True,  # Verify connections
)
```

### How do I set query timeout?

```python
# Via configuration
db = connect(
    "postgresql://user:pass@host/dbname",
    query_timeout=30.0  # 30 seconds
)

# Via environment variable
# MOLTRES_QUERY_TIMEOUT=30.0
```

### How do I enable SQL logging?

```python
db = connect(
    "postgresql://user:pass@host/dbname",
    echo=True  # Enable SQL logging
)
```

## Feature Questions

### Does Moltres support UDFs (User-Defined Functions)?

No, Moltres doesn't support Python UDFs since operations are pushed down to SQL. Instead:
- Use SQL functions available in your database
- Use Moltres expression functions (see `moltres.expressions.functions`)
- For complex logic, use raw SQL expressions

### Does Moltres support window functions?

Yes! Moltres supports window functions:

```python
from moltres.expressions.window import Window
from moltres.expressions.functions import row_number, rank

window = Window.partition_by(col("category")).order_by(col("amount").desc())
df = df.with_column("rank", rank().over(window))
```

### Does Moltres support recursive queries?

Yes! Use recursive CTEs:

```python
initial = db.table("seed").select(...)
recursive = initial.select(...)  # References CTE
df = initial.recursive_cte("name", recursive)
```

### Does Moltres support LATERAL joins?

Yes, for PostgreSQL and MySQL 8.0+:

```python
df = customers.join(
    orders.select().where(col("customer_id") == col("customers.id")),
    how="left",
    lateral=True
)
```

### Can I use raw SQL?

Yes, you can execute raw SQL:

```python
# Execute raw SQL
results = db.execute("SELECT * FROM users WHERE age > 18")

# Use in subqueries
subquery = db.execute("SELECT id FROM active_users")
df = db.table("orders").select().where(col("user_id").isin(subquery))
```

## Migration Questions

### How do I migrate from PySpark?

See the [Migration Guide](./MIGRATION_SPARK.md) for detailed instructions.

Key differences:
- Replace `SparkSession` with `connect()`
- Update imports (`pyspark.sql.functions` → `moltres.expressions.functions`)
- Use `col()` consistently
- Replace UDFs with SQL functions

### How do I migrate from Pandas?

1. Replace `pd.read_csv()` with `db.load.csv()`
2. Replace pandas operations with Moltres DataFrame operations
3. Use `df.collect()` to get results as list of dicts
4. Convert to pandas if needed: `pd.DataFrame(df.collect())`

## Troubleshooting

### My query returns no results

1. Check if the table has data: `db.table("users").select().count()`
2. Verify filter conditions: `df.to_sql()` to see SQL
3. Check for NULL values: Use `isnull()` / `isnotnull()`

### My query is too slow

1. Check indexes: `EXPLAIN` the query
2. Filter early: Apply WHERE clauses before joins
3. Use LIMIT: Test with small result sets first
4. See [Performance Guide](./PERFORMANCE.md)

### I get "table does not exist" error

1. Verify table name spelling
2. Check database connection
3. Ensure table is in the correct schema
4. Use `db.execute("SHOW TABLES")` to list tables

### Column names are case-sensitive

SQL column names may be case-sensitive depending on your database:
- SQLite: Case-insensitive
- PostgreSQL: Case-sensitive (use quotes)
- MySQL: Depends on configuration

Use exact column names as they appear in your database.

## Getting Help

- **Documentation:** See [README](../README.md) and other guides
- **Examples:** See [Examples Guide](./EXAMPLES.md)
- **Issues:** Report on GitHub
- **Questions:** Check this FAQ first

For more help:
- [Performance Guide](./PERFORMANCE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Debugging Guide](./DEBUGGING.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

