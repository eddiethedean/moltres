# PySpark to Moltres Migration Guide

This guide helps you migrate from PySpark to Moltres, highlighting key differences and providing migration examples.

## Overview

Moltres provides a PySpark-like API but compiles operations to SQL instead of executing on a Spark cluster. This means:

- **No cluster setup required** - works with any SQL database
- **SQL pushdown** - operations are compiled to SQL and executed in the database
- **Lazy evaluation** - similar to PySpark's lazy evaluation model
- **Familiar API** - many methods have the same names and signatures

## Key Differences

### 1. Initialization

**PySpark:**
```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("app").getOrCreate()
df = spark.read.table("users")
```

**Moltres:**
```python
from moltres import connect
db = connect("postgresql://user:pass@host/dbname")
df = db.table("users").select()
```

### 2. Reading Data

**PySpark:**
```python
df = spark.read.csv("data.csv")
df = spark.read.json("data.json")
df = spark.read.parquet("data.parquet")
```

**Moltres:**
```python
df = db.load.csv("data.csv")
df = db.load.json("data.json")
df = db.load.parquet("data.parquet")
```

### 3. Basic Operations

Most DataFrame operations work similarly:

**PySpark:**
```python
df.select("id", "name").where(col("age") > 18).order_by(col("name"))
```

**Moltres:**
```python
from moltres import col
df.select("id", "name").where(col("age") > 18).order_by(col("name"))
```

### 4. Aggregations

**PySpark:**
```python
from pyspark.sql.functions import sum, avg, count
df.group_by("category").agg(
    sum("amount").alias("total"),
    avg("amount").alias("average"),
    count("*").alias("count")
)
```

**Moltres:**
```python
from moltres.expressions.functions import sum, avg, count
df.group_by("category").agg(
    sum(col("amount")).alias("total"),
    avg(col("amount")).alias("average"),
    count("*").alias("count")
)
```

### 5. Joins

**PySpark:**
```python
df1.join(df2, on="id", how="left")
df1.join(df2, on=[("id", "customer_id")], how="inner")
```

**Moltres:**
```python
df1.join(df2, on="id", how="left")
df1.join(df2, on=[("id", "customer_id")], how="inner")
```

### 6. Window Functions

**PySpark:**
```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank

window = Window.partition_by("category").order_by(col("amount").desc())
df.with_column("rank", rank().over(window))
```

**Moltres:**
```python
from moltres.expressions.window import Window
from moltres.expressions.functions import rank

window = Window.partition_by(col("category")).order_by(col("amount").desc())
df.with_column("rank", rank().over(window))
```

### 7. UDFs (User-Defined Functions)

**PySpark:**
```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def my_udf(x):
    return x.upper()

df.with_column("upper", my_udf(col("name")))
```

**Moltres:**
Moltres doesn't support UDFs directly since operations are pushed down to SQL. Instead, use SQL functions:

```python
from moltres.expressions.functions import upper
df.with_column("upper", upper(col("name")))
```

Or use raw SQL expressions for complex logic.

### 8. Collecting Results

**PySpark:**
```python
rows = df.collect()  # Returns list of Row objects
df.show()  # Prints first 20 rows
```

**Moltres:**
```python
rows = df.collect()  # Returns list of dicts
df.show()  # Prints first 20 rows
```

### 9. Writing Data

**PySpark:**
```python
df.write.mode("overwrite").parquet("output.parquet")
df.write.mode("append").csv("output.csv")
```

**Moltres:**
```python
df.write.parquet("output.parquet", mode="overwrite")
df.write.csv("output.csv", mode="append")
```

## Migration Checklist

1. **Replace SparkSession with Database connection**
   - Use `connect()` instead of `SparkSession.builder`
   - Configure connection string for your database

2. **Update imports**
   - Replace `pyspark.sql.functions` with `moltres.expressions.functions`
   - Replace `pyspark.sql.window` with `moltres.expressions.window`

3. **Column references**
   - Use `col("name")` consistently (PySpark sometimes allows strings directly)
   - Import `col` from `moltres`

4. **UDFs**
   - Replace UDFs with SQL functions where possible
   - For complex logic, consider using SQL expressions or database functions

5. **Data types**
   - Moltres uses SQL types (INTEGER, TEXT, REAL, etc.)
   - Check type compatibility with your database

6. **Testing**
   - Test queries with `df.to_sql()` to see generated SQL
   - Verify results match PySpark output

## Common Patterns

### Pattern 1: Filtering and Aggregation

**PySpark:**
```python
df.filter(col("status") == "active") \
  .group_by("category") \
  .agg(sum("amount").alias("total"))
```

**Moltres:**
```python
df.where(col("status") == "active") \
  .group_by("category") \
  .agg(sum(col("amount")).alias("total"))
```

### Pattern 2: Window Functions

**PySpark:**
```python
window = Window.partition_by("department").order_by(col("salary").desc())
df.with_column("rank", row_number().over(window))
```

**Moltres:**
```python
window = Window.partition_by(col("department")).order_by(col("salary").desc())
df.with_column("rank", row_number().over(window))
```

### Pattern 3: Complex Joins

**PySpark:**
```python
df1.join(df2, df1.id == df2.customer_id, "left")
```

**Moltres:**
```python
df1.join(df2, on=[("id", "customer_id")], how="left")
```

## Limitations

1. **No distributed processing** - Moltres executes in the database, not on a cluster
2. **No UDFs** - Use SQL functions instead
3. **Limited complex types** - Array/Map support varies by database
4. **SQL dialect differences** - Some operations may not work on all databases

## Benefits

1. **No infrastructure** - No Spark cluster needed
2. **Better performance** - SQL pushdown to optimized database engines
3. **Simpler deployment** - Just connect to your existing database
4. **Cost effective** - No cluster maintenance costs

## Getting Help

- Check the [API documentation](../README.md)
- Review [examples](../docs/EXAMPLES.md)
- See [troubleshooting guide](./TROUBLESHOOTING.md)

