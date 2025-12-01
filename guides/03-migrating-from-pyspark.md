# Migrating from PySpark to Moltres

This guide helps you transition from PySpark to Moltres, highlighting similarities and differences.

## Why Migrate?

- **No cluster required**: Run on any SQL database without Spark infrastructure
- **Simpler deployment**: No need for Spark clusters, YARN, or Kubernetes
- **Direct SQL execution**: Operations compile directly to SQL
- **Familiar API**: 98% PySpark API compatibility
- **Real CRUD**: Full INSERT, UPDATE, DELETE support (not just SELECT)

## Key Similarities

Moltres maintains high compatibility with PySpark:

| PySpark | Moltres | Notes |
|---------|---------|-------|
| `df.select()` | `df.select()` | ✅ Same |
| `df.where()` / `df.filter()` | `df.where()` / `df.filter()` | ✅ Same |
| `df.join()` | `df.join()` | ✅ Same syntax |
| `df.groupBy()` | `df.group_by()` | ✅ Same (also supports `groupBy()`) |
| `df.agg()` | `df.agg()` | ✅ Same |
| `df.orderBy()` | `df.order_by()` | ✅ Same (also supports `orderBy()`) |
| `df.limit()` | `df.limit()` | ✅ Same |
| `df.collect()` | `df.collect()` | ✅ Same |
| `df.show()` | `df.show()` | ✅ Same |
| `df.explain()` | `df.explain()` | ✅ Same |

## Key Differences

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Execution** | Spark cluster | Direct SQL on database |
| **Data Sources** | HDFS, S3, Parquet, etc. | SQL tables, files (CSV, JSON, Parquet) |
| **CRUD Operations** | Limited (mostly SELECT) | Full INSERT, UPDATE, DELETE |
| **Connection** | `SparkSession.builder` | `connect("database://...")` |
| **DataFrames** | Spark DataFrames | Moltres DataFrames (SQL-backed) |

## Migration Patterns

### 1. Initialization

**PySpark:**
```python
# Note: PySpark requires a Spark cluster to run
# from pyspark.sql import SparkSession
#
# spark = SparkSession.builder \
#     .appName("MyApp") \
#     .config("spark.some.config.option", "some-value") \
#     .getOrCreate()
```

**Moltres:**
```python
from moltres import connect

# Simple connection
db = connect("postgresql://user:pass@localhost/mydb")

# With configuration
db = connect(
    "postgresql://user:pass@localhost/mydb",
    echo=False,  # SQL logging
    fetch_format="records",  # Result format
    pool_size=5  # Connection pool
)
```

### 2. Reading Data

**PySpark:**
```python
# Note: PySpark requires a Spark cluster to run
# From table/view
# df = spark.table("users")
#
# From CSV
# df = spark.read.csv("data.csv", header=True, inferSchema=True)
#
# From Parquet
# df = spark.read.parquet("data.parquet")
#
# From SQL
# df = spark.read.jdbc(url, table, properties)
```

**Moltres:**
```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
], database=db).insert_into("users")
# From table
df = db.table("users").select()

# From CSV (returns Records, insert into table first)
from moltres.io.records import Records
records = Records.from_csv("data.csv", database=db)
records.insert_into("users")
df = db.table("users").select()

# From Parquet
records = Records.from_parquet("data.parquet", database=db)
records.insert_into("users")
df = db.table("users").select()

# From SQL query
df = db.sql("SELECT * FROM users WHERE age > 25")

```

### 3. Basic Operations

**PySpark:**
```python
from moltres import connect
from pyspark.sql.functions import col, sum as spark_sum
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

df = spark.table("users")
result = (
    df.select("id", "name", "email")
    .where(col("age") > 25)
    .groupBy("country")
    .agg(spark_sum(col("amount")).alias("total"))
    .orderBy(col("total").desc())
    .limit(10)
)

```

**Moltres:**
```python
from moltres import connect
from moltres import col
from moltres.expressions import functions as F
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

df = db.table("users").select()
result = (
    df.select("id", "name", "email")
    .where(col("age") > 25)
    .group_by("country")
    .agg(F.sum(col("amount")).alias("total"))
    .order_by(col("total").desc())
    .limit(10)
)

```

### 4. Joins

**PySpark:**
```python
# Note: PySpark requires a Spark cluster to run
# Assume df1 and df2 are already loaded
# Inner join
# result = df1.join(df2, on="key", how="inner")
#
# Left join
# result = df1.join(df2, on="key", how="left")
#
# Using column expressions
# result = df1.join(df2, df1.id == df2.user_id, how="inner")
```

**Moltres:**
```python
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Setup: Create tables and data
db = connect("sqlite:///:memory:")
db.create_table("table1", [column("key", "INTEGER"), column("value1", "TEXT")]).collect()
db.create_table("table2", [column("key", "INTEGER"), column("value2", "TEXT")]).collect()

df1 = db.table("table1").select()
df2 = db.table("table2").select()

# Inner join (default)
result = df1.join(df2, on=[col("table1.key") == col("table2.key")])

# Left join
result = df1.join(df2, on=[col("df1.key") == col("df2.key")], how="left")

# Using column expressions (same as PySpark)
result = df1.join(df2, on=[col("df1.id") == col("df2.user_id")], how="inner")
```

### 5. Window Functions

**PySpark:**
```python
from moltres import connect
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

window = Window.partitionBy("country").orderBy(col("amount").desc())
df = df.withColumn("rank", rank().over(window))

```

**Moltres:**
```python
from moltres import connect
from moltres.expressions import functions as F
from moltres.expressions.window import Window
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

window = Window.partition_by("country").order_by(col("amount").desc())
df = df.withColumn("rank", F.rank().over(window))

```

### 6. UDFs (User-Defined Functions)

**PySpark:**
```python
from moltres import connect
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

@udf(returnType=StringType())
def my_udf(value):
    return value.upper()

df = df.withColumn("upper_name", my_udf(col("name")))

```

**Moltres:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Moltres doesn't support Python UDFs directly
# Instead, use SQL functions or create database functions

# Option 1: Use built-in SQL functions
from moltres.expressions import functions as F
df = df.withColumn("upper_name", F.upper(col("name")))

# Option 2: Use raw SQL expressions
df = df.selectExpr("UPPER(name) as upper_name")

# Option 3: Create database function (PostgreSQL example)
# CREATE FUNCTION my_upper(text) RETURNS text AS $$ SELECT UPPER($1) $$ LANGUAGE SQL;
# Then use in Moltres: F.func("my_upper", col("name"))

```

### 7. Writing Data

**PySpark:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Write to table
df.write.saveAsTable("results")

# Write to Parquet
df.write.parquet("output.parquet", mode="overwrite")

# Write to CSV
df.write.csv("output.csv", mode="overwrite", header=True)

```

**Moltres:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Write to table
df.write.save_as_table("results")

# Write to Parquet
df.write.parquet("output.parquet", mode="overwrite")

# Write to CSV
df.write.csv("output.csv", mode="overwrite", header=True)

```

### 8. CRUD Operations (Moltres Advantage)

**PySpark:**
```python
# PySpark doesn't support UPDATE/DELETE directly
# You'd need to use SQL or rewrite tables
```

**Moltres:**
```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Update rows
result = db.update(
    "users",
    where=col("status") == "pending",
    set={"status": "active", "updated_at": "2024-01-15"}
)

# Delete rows
result = db.delete(
    "users",
    where=col("age") < 18
)

# Insert from DataFrame
df.write.insertInto("users")

```

## Performance Considerations

### PySpark Optimizations
- Catalyst optimizer
- Partition pruning
- Column pruning
- Broadcast joins

### Moltres Optimizations
- SQL pushdown (database optimizer handles it)
- Query compilation (single SQL statement)
- Connection pooling
- Streaming for large results

**Key Difference**: PySpark optimizes at the Spark level, while Moltres relies on the database optimizer.

## Migration Strategy

### Step 1: Identify Data Sources

Map your PySpark data sources to Moltres equivalents:

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Insert sample data
Records.from_list([
    {"id": 1, "name": "Alice"},
    {"id": 2, "name": "Bob"},
], database=db).insert_into("users")
# PySpark
df = spark.read.parquet("s3://bucket/data.parquet")

# Moltres
# Option 1: Load into database first
records = Records.from_parquet("s3://bucket/data.parquet", database=db)
records.insert_into("data_table")
df = db.table("data_table").select()

# Option 2: Use database external tables (if supported)
# PostgreSQL: CREATE FOREIGN TABLE ...
# Then: df = db.table("data_table").select()

```

### Step 2: Convert Transformations

Most PySpark transformations map directly:

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()
# PySpark
result = (
    spark.table("users")
    .select("id", "name")
    .where(col("age") > 25)
    .groupBy("country")
    .agg(sum(col("amount")))
)

# Moltres (almost identical)
result = (
    db.table("users")
    .select("id", "name")
    .where(col("age") > 25)
    .group_by("country")
    .agg(F.sum(col("amount")))
)

```

### Step 3: Handle UDFs

Replace Python UDFs with SQL functions or database functions:

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# PySpark UDF
@udf(returnType=StringType())
def process_name(name):
    return name.strip().upper()

# Moltres equivalent
from moltres.expressions import functions as F
df = df.withColumn(
    "processed_name",
    F.upper(F.trim(col("name")))
)

```

### Step 4: Test Incrementally

Start with simple queries and gradually migrate complex pipelines:

1. ✅ Simple SELECT queries
2. ✅ Filters and aggregations
3. ✅ Joins
4. ✅ Window functions
5. ✅ Complex transformations

## Common Migration Challenges

### Challenge 1: Distributed Processing

**PySpark**: Processes data across cluster nodes
**Moltres**: Processes in single database (but database may be distributed)

**Solution**: Use database-specific features (e.g., PostgreSQL partitioning, MySQL sharding)

### Challenge 2: Large File Processing

**PySpark**: Handles large files via Spark's distributed processing
**Moltres**: Use streaming or load into database first

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Moltres streaming approach
async def process_streaming(df):
    async for chunk in await df.collect(stream=True):
        process_chunk(chunk)

```

### Challenge 3: Complex UDFs

**PySpark**: Supports Python UDFs
**Moltres**: Use SQL functions or create database functions

**Solution**: Rewrite logic in SQL or create database stored procedures

## Migration Checklist

- [ ] Map PySpark data sources to Moltres equivalents
- [ ] Replace `SparkSession` with `connect()`
- [ ] Update `groupBy()` to `group_by()` (or keep `groupBy()` - both work)
- [ ] Update `orderBy()` to `order_by()` (or keep `orderBy()` - both work)
- [ ] Replace PySpark functions with Moltres functions
- [ ] Handle UDFs (convert to SQL functions)
- [ ] Test query results match PySpark output
- [ ] Update deployment (remove Spark dependencies)
- [ ] Configure database connection pooling
- [ ] Set up monitoring (database query logs)

## Next Steps

- **Performance**: See [Performance Optimization Guide](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)
- **Patterns**: Check [Common Patterns Guide](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)
- **Best Practices**: Read [Best Practices Guide](https://moltres.readthedocs.io/en/latest/guides/best-practices.html)

