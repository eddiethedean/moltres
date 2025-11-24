# Moltres vs PySpark API Comparison Report

## Executive Summary

This report provides a comprehensive comparison between Moltres and PySpark APIs, analyzing their similarities, differences, strengths, and areas for improvement. Both libraries provide DataFrame APIs for data manipulation, but they differ significantly in their execution models, target use cases, and feature sets.

### Key Findings

- **API Similarity**: Moltres closely mirrors PySpark's DataFrame API, making it familiar for PySpark users
- **Execution Model**: Moltres compiles to SQL and executes on traditional databases; PySpark uses distributed computing
- **CRUD Operations**: Moltres provides UPDATE and DELETE operations that PySpark lacks
- **Async Support**: Moltres has comprehensive async/await support; PySpark has limited async capabilities
- **File Reading**: Both return lazy DataFrames, but Moltres materializes files into temporary tables for SQL pushdown
- **Feature Scope**: PySpark has broader feature set; Moltres focuses on SQL-mappable features

### Use Case Recommendations

- **Choose Moltres** when:
  - Working with traditional SQL databases (PostgreSQL, MySQL, SQLite)
  - Need UPDATE/DELETE operations with DataFrame syntax
  - Want SQL pushdown without a cluster
  - Require async/await support
  - Prefer lightweight, dependency-minimal solution

- **Choose PySpark** when:
  - Need distributed computing across clusters
  - Working with very large datasets requiring Spark's optimizations
  - Need machine learning (MLlib) integration
  - Require streaming data processing at scale
  - Working with Hadoop ecosystem

---

## 1. Initialization & Connection

### PySpark

```python
from pyspark.sql import SparkSession

# Create SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# Or with existing SparkContext
from pyspark import SparkContext
sc = SparkContext("local", "MyApp")
spark = SparkSession(sc)
```

### Moltres

```python
from moltres import connect, async_connect

# Synchronous connection
db = connect("sqlite:///example.db")
db = connect("postgresql://user:pass@localhost/db")
db = connect("mysql://user:pass@localhost/db")

# Async connection
db = async_connect("postgresql+asyncpg://user:pass@localhost/db")
db = async_connect("sqlite+aiosqlite:///example.db")
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Connection Type** | SparkSession (cluster-based) | Database connection (direct to DB) |
| **Configuration** | Extensive Spark config options | Database connection string + env vars |
| **Session Management** | Single SparkSession per app | Multiple Database instances possible |
| **Async Support** | Limited (mostly synchronous) | Full async/await support |
| **Resource Management** | Spark cluster resources | Database connection pooling |

**Key Differences:**
- PySpark requires a Spark cluster or local Spark installation
- Moltres connects directly to SQL databases (no cluster needed)
- Moltres supports async operations natively
- PySpark has extensive configuration options; Moltres uses simpler connection strings

---

## 2. DataFrame Creation

### PySpark

```python
# From existing DataFrame
df = spark.createDataFrame([(1, "Alice"), (2, "Bob")], ["id", "name"])

# From RDD
rdd = sc.parallelize([(1, "Alice"), (2, "Bob")])
df = spark.createDataFrame(rdd, ["id", "name"])

# From files (returns DataFrame)
df = spark.read.csv("data.csv")
df = spark.read.json("data.json")
df = spark.read.parquet("data.parquet")

# From SQL
df = spark.sql("SELECT * FROM table")
df = spark.read.table("table_name")
```

### Moltres

```python
# From Python data
df = db.createDataFrame(
    [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    schema=[ColumnDef(name="id", type_name="INTEGER"), ...]
)

# From database table
df = db.table("users").select()
df = db.table("users").select("id", "name", "email")

# From files (returns DataFrame - lazy)
df = db.load.csv("data.csv")
df = db.load.json("data.json")
df = db.load.parquet("data.parquet")
df = db.load.jsonl("data.jsonl")
df = db.load.text("log.txt", column_name="line")

# For backward compatibility: get Records directly
records = db.read.records.csv("data.csv")  # Returns Records (materialized)
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **From Lists** | `createDataFrame()` with schema | `createDataFrame()` with schema |
| **From Tables** | `read.table()` or `sql()` | `db.table().select()` |
| **From Files** | `read.csv/json/parquet()` | `db.load.csv/json/parquet()` |
| **Return Type** | DataFrame (lazy) | DataFrame (lazy) or Records (materialized) |
| **File Materialization** | Handled by Spark | Materialized to temp tables on `.collect()` |
| **Schema Inference** | Automatic | Automatic or explicit |

**Key Differences:**
- Both return lazy DataFrames from files
- Moltres materializes files into temporary tables for SQL pushdown
- Moltres provides `db.read.records.*` for backward compatibility
- PySpark has `sql()` method; Moltres uses `db.table().select()`

---

## 3. DataFrame Transformations

### 3.1 Select Operations

#### PySpark

```python
# Select columns
df.select("id", "name", "email")
df.select(df.id, df.name)

# Select with expressions
df.select(col("id"), (col("amount") * 1.1).alias("with_tax"))
df.selectExpr("id", "amount * 1.1 as with_tax")

# Select all
df.select("*")
```

#### Moltres

```python
# Select columns
df.select("id", "name", "email")
df.select(col("id"), col("name"))

# Select with expressions
df.select(col("id"), (col("amount") * 1.1).alias("with_tax"))

# Select all
df.select()  # Empty select means all columns
```

**Comparison:** Nearly identical, except Moltres uses empty `select()` for all columns.

### 3.2 Filter Operations

#### PySpark

```python
df.filter(col("age") > 18)
df.where(col("age") > 18)
df.filter("age > 18")  # SQL string
```

#### Moltres

```python
df.filter(col("age") > 18)
df.where(col("age") > 18)  # Alias for filter
```

**Comparison:** Identical functionality. PySpark supports SQL strings; Moltres requires Column expressions.

### 3.3 Join Operations

#### PySpark

```python
# Inner join
df1.join(df2, "id")
df1.join(df2, df1.id == df2.id)
df1.join(df2, ["id", "name"])

# Join types
df1.join(df2, "id", "inner")
df1.join(df2, "id", "left")
df1.join(df2, "id", "right")
df1.join(df2, "id", "outer")
df1.join(df2, "id", "full")
df1.join(df2, "id", "left_semi")
df1.join(df2, "id", "left_anti")
df1.crossJoin(df2)

# Join hints
df1.join(df2.hint("broadcast"), "id")
```

#### Moltres

```python
# Inner join
df1.join(df2, on="id")
df1.join(df2, on=[("left_col", "right_col")])
df1.join(df2, on=["id", "name"])

# Join types
df1.join(df2, on="id", how="inner")
df1.join(df2, on="id", how="left")
df1.join(df2, on="id", how="right")
df1.join(df2, on="id", how="full")
df1.join(df2, on="id", how="cross")

# Specialized joins
df1.semi_join(df2, on="id")
df1.anti_join(df2, on="id")

# Lateral joins (PostgreSQL, MySQL 8.0+)
df1.join(df2, on="id", lateral=True)

# Join hints (dialect-specific)
df1.join(df2, on="id", hints=["USE_INDEX(idx_name)"])
```

**Comparison:** Very similar. Moltres has `semi_join()` and `anti_join()` as separate methods. Both support join hints, but Moltres uses dialect-specific syntax.

### 3.4 GroupBy and Aggregations

#### PySpark

```python
from pyspark.sql.functions import sum, avg, count

# Group by
df.groupBy("category").agg(sum("amount").alias("total"))
df.groupBy("category", "region").agg(
    sum("amount").alias("total"),
    avg("price").alias("avg_price"),
    count("*").alias("count")
)

# Pivot
df.groupBy("category").pivot("status").agg(sum("amount"))
```

#### Moltres

```python
from moltres.expressions.functions import sum, avg, count

# Group by
df.group_by("category").agg(sum(col("amount")).alias("total"))
df.group_by("category", "region").agg(
    sum(col("amount")).alias("total"),
    avg(col("price")).alias("avg_price"),
    count("*").alias("count")
)

# Pivot (separate method)
df.pivot("status", values=["active", "inactive"]).agg(sum(col("amount")))
```

**Comparison:** Nearly identical. Moltres requires `col()` wrapper for column references in aggregations. Pivot is a separate method in Moltres.

### 3.5 Sorting

#### PySpark

```python
df.orderBy("name")
df.orderBy(col("name").desc())
df.orderBy(["category", col("amount").desc()])
df.sort("name")
df.sort(col("name").desc())
```

#### Moltres

```python
df.order_by(col("name"))
df.order_by(col("name").desc())
df.order_by(col("category"), col("amount").desc())
```

**Comparison:** Identical functionality. PySpark has both `orderBy()` and `sort()`; Moltres uses `order_by()`.

### 3.6 Distinct Operations

#### PySpark

```python
df.distinct()
df.dropDuplicates()
df.dropDuplicates(["col1", "col2"])
```

#### Moltres

```python
df.distinct()
df.dropDuplicates(["col1", "col2"])  # Simplified implementation
```

**Comparison:** Similar. Moltres `dropDuplicates()` with subset is simplified.

### 3.7 Column Operations

#### PySpark

```python
df.withColumn("new_col", col("old_col") * 2)
df.withColumnRenamed("old_name", "new_name")
df.drop("col1", "col2")
df.select("col1", "col2").alias("new_name")  # DataFrame alias
```

#### Moltres

```python
df.withColumn("new_col", col("old_col") * 2)
df.drop("col1", "col2")
# Column renaming done via select with alias
df.select(col("old_name").alias("new_name"), ...)
```

**Comparison:** Moltres lacks `withColumnRenamed()`; use `select()` with aliases instead.

### 3.8 Window Functions

#### PySpark

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, lag, lead

window = Window.partitionBy("category").orderBy("amount")
df.withColumn("row_num", row_number().over(window))
df.withColumn("rank", rank().over(window))
df.withColumn("prev_amount", lag("amount", 1).over(window))
```

#### Moltres

```python
from moltres.expressions.functions import row_number, rank, lag, lead
from moltres.expressions.window import Window

window = Window.partition_by("category").order_by("amount")
df.select(
    col("category"),
    col("amount"),
    row_number().over(window).alias("row_num"),
    rank().over(window).alias("rank"),
    lag(col("amount"), 1).over(window).alias("prev_amount")
)
```

**Comparison:** Similar API, but Moltres requires window functions in `select()` rather than `withColumn()`.

---

## 4. Read Operations

### PySpark

```python
# CSV
df = spark.read.csv("data.csv")
df = spark.read.option("header", True).csv("data.csv")
df = spark.read.schema(schema).csv("data.csv")

# JSON
df = spark.read.json("data.json")
df = spark.read.option("multiline", True).json("data.json")

# Parquet
df = spark.read.parquet("data.parquet")

# Text
df = spark.read.text("log.txt")

# Generic format
df = spark.read.format("csv").load("data.csv")
df = spark.read.format("json").option("multiline", True).load("data.json")

# From table
df = spark.read.table("table_name")
df = spark.sql("SELECT * FROM table_name")
```

### Moltres

```python
# CSV - returns DataFrame (lazy)
df = db.load.csv("data.csv")
df = db.load.option("header", True).csv("data.csv")
df = db.load.schema(schema).csv("data.csv")

# JSON - returns DataFrame (lazy)
df = db.load.json("data.json")
df = db.load.option("multiline", True).json("data.json")

# JSONL - returns DataFrame (lazy)
df = db.load.jsonl("data.jsonl")

# Parquet - returns DataFrame (lazy)
df = db.load.parquet("data.parquet")

# Text - returns DataFrame (lazy)
df = db.load.text("log.txt", column_name="line")

# Generic format - returns DataFrame (lazy)
df = db.load.format("csv").option("header", True).load("data.csv")

# From table
df = db.table("table_name").select()
df = db.load.table("table_name")

# For backward compatibility: get Records directly
records = db.read.records.csv("data.csv")  # Returns Records (materialized)
records = db.read.records.json("data.json")
records = db.read.records.jsonl("data.jsonl")
records = db.read.records.parquet("data.parquet")
records = db.read.records.text("log.txt")
records = db.read.records.dicts([{"id": 1, "name": "Alice"}])
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Return Type** | DataFrame (lazy) | DataFrame (lazy) or Records (materialized) |
| **File Formats** | CSV, JSON, Parquet, Text, ORC, Avro, etc. | CSV, JSON, JSONL, Parquet, Text |
| **Schema** | `.schema()` or inference | `.schema()` or inference |
| **Options** | `.option()` builder | `.option()` builder |
| **Streaming** | `.stream()` for streaming | `.stream()` for Records, `collect(stream=True)` for DataFrames |
| **Materialization** | Handled by Spark | Files materialized to temp tables on `.collect()` |
| **Backward Compatibility** | N/A | `db.read.records.*` for Records |

**Key Differences:**
- Both return lazy DataFrames
- Moltres materializes files into temporary tables for SQL pushdown
- Moltres has `db.read.records.*` for backward compatibility
- PySpark supports more file formats (ORC, Avro, etc.)
- Moltres has JSONL support

---

## 5. Write Operations

### PySpark

```python
# To table
df.write.saveAsTable("table_name")
df.write.mode("overwrite").saveAsTable("table_name")
df.write.mode("append").saveAsTable("table_name")
df.write.insertInto("existing_table")  # Append only

# To files
df.write.csv("output.csv")
df.write.json("output.json")
df.write.parquet("output.parquet")

# With options
df.write.option("header", True).csv("output.csv")
df.write.option("compression", "gzip").parquet("output.parquet")

# Partitioned writes
df.write.partitionBy("country", "year").parquet("partitioned_data")

# Write modes
df.write.mode("overwrite").csv("output.csv")
df.write.mode("append").csv("output.csv")
df.write.mode("ignore").csv("output.csv")  # Skip if exists
df.write.mode("error").csv("output.csv")  # Error if exists
```

### Moltres

```python
# To table
df.write.save_as_table("table_name")
df.write.mode("overwrite").save_as_table("table_name")
df.write.mode("append").save_as_table("table_name")
df.write.mode("error_if_exists").save_as_table("table_name")
df.write.insertInto("existing_table")  # Append only

# To files
df.write.csv("output.csv")
df.write.json("output.json")
df.write.jsonl("output.jsonl")
df.write.parquet("output.parquet")

# With options
df.write.option("header", True).csv("output.csv")
df.write.option("compression", "gzip").parquet("output.parquet")

# Partitioned writes
df.write.partitionBy("country", "year").csv("partitioned_data")

# CRUD operations (eager execution)
df.write.update("table_name", where=col("id") == 1, set={"name": "Updated"})
df.write.delete("table_name", where=col("id") == 1)
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Save to Table** | `saveAsTable()` | `save_as_table()` |
| **Insert to Table** | `insertInto()` | `insertInto()` / `insert_into()` |
| **Write Modes** | overwrite, append, ignore, error | overwrite, append, error_if_exists |
| **File Formats** | CSV, JSON, Parquet, ORC, Avro, etc. | CSV, JSON, JSONL, Parquet |
| **Partitioning** | `partitionBy()` | `partitionBy()` / `partition_by()` |
| **UPDATE Operation** | ❌ Not available | ✅ `df.write.update()` |
| **DELETE Operation** | ❌ Not available | ✅ `df.write.delete()` |
| **Execution Model** | Eager (writes execute immediately) | Eager (writes execute immediately) |

**Key Differences:**
- Moltres provides UPDATE and DELETE operations that PySpark lacks
- PySpark has more write modes (ignore)
- Both execute writes eagerly (immediately)

---

## 6. CRUD Operations

### PySpark

```python
# Insert only (append)
df.write.insertInto("table_name")

# No UPDATE or DELETE operations
# Must use SQL or manual DataFrame operations
spark.sql("UPDATE table SET col = value WHERE condition")
spark.sql("DELETE FROM table WHERE condition")
```

### Moltres

```python
# Insert
df.write.insertInto("table_name")
records.insert_into("table_name")  # From Records

# Update (eager execution)
df.write.update(
    "table_name",
    where=col("id") == 1,
    set={"name": "Updated", "active": 1}
)

# Delete (eager execution)
df.write.delete("table_name", where=col("id") == 1)

# All operations participate in transactions
with db.transaction():
    df.write.insertInto("table1")
    df.write.update("table2", where=..., set={...})
    df.write.delete("table3", where=...)
```

### Comparison

| Operation | PySpark | Moltres |
|-----------|---------|---------|
| **INSERT** | ✅ `insertInto()` | ✅ `insertInto()`, `Records.insert_into()` |
| **UPDATE** | ❌ Must use SQL | ✅ `df.write.update()` |
| **DELETE** | ❌ Must use SQL | ✅ `df.write.delete()` |
| **Transaction Support** | Limited | ✅ Full transaction support |

**Key Differences:**
- **Moltres's major advantage**: Provides UPDATE and DELETE with DataFrame syntax
- PySpark requires raw SQL for updates/deletes
- Moltres has comprehensive transaction support

---

## 7. Expression System

### PySpark

```python
from pyspark.sql.functions import col, lit, when, sum, avg, count
from pyspark.sql.types import StringType

# Column references
col("name")
df["name"]

# Literals
lit(42)
lit("string")

# Conditional expressions
when(col("age") > 18, "adult").otherwise("minor")

# Aggregations
sum(col("amount"))
avg(col("price"))
count("*")

# String functions
concat(col("first"), lit(" "), col("last"))
upper(col("name"))
substring(col("text"), 1, 10)

# Math functions
col("amount") * 1.1
col("price") + col("tax")

# UDFs
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf

def my_udf(x):
    return x.upper()

udf_func = udf(my_udf, StringType())
df.withColumn("upper_name", udf_func(col("name")))
```

### Moltres

```python
from moltres import col, lit
from moltres.expressions.functions import sum, avg, count, concat, upper, substring, when

# Column references
col("name")

# Literals
lit(42)
lit("string")

# Conditional expressions
when(col("age") > 18, "adult").otherwise("minor")

# Aggregations
sum(col("amount"))
avg(col("price"))
count("*")

# String functions
concat(col("first"), lit(" "), col("last"))
upper(col("name"))
substring(col("text"), 1, 10)

# Math functions
col("amount") * 1.1
col("price") + col("tax")

# UDFs - Not directly supported (SQL pushdown focus)
# Must use SQL functions or expressions
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Column References** | `col()`, `df["col"]` | `col()` |
| **Literals** | `lit()` | `lit()` |
| **Conditionals** | `when().otherwise()` | `when().otherwise()` |
| **Aggregations** | Extensive | SQL-mappable only |
| **String Functions** | Extensive | SQL-mappable only |
| **Math Functions** | Extensive | SQL-mappable only |
| **UDFs** | ✅ Python UDFs, Pandas UDFs | ❌ Not supported (SQL pushdown) |
| **Window Functions** | ✅ Full support | ✅ Full support |

**Key Differences:**
- Moltres focuses on SQL-mappable functions only
- PySpark supports Python UDFs; Moltres does not (by design)
- Both have similar expression APIs for SQL-mappable operations

---

## 8. Execution Model

### PySpark

```python
# Lazy evaluation - builds logical plan
df = spark.read.csv("data.csv").filter(col("age") > 18)

# Actions trigger execution
results = df.collect()  # Returns list of Row objects
df.show()  # Prints first 20 rows
df.count()  # Returns count
df.first()  # Returns first row
df.take(10)  # Returns first 10 rows
df.head()  # Returns first row

# Streaming
df_stream = spark.readStream.csv("data.csv")
query = df_stream.writeStream.outputMode("append").start()
```

### Moltres

```python
# Lazy evaluation - builds logical plan
df = db.load.csv("data.csv").where(col("age") > 18)

# Actions trigger execution
results = df.collect()  # Returns list of dicts
df.show()  # Prints first 20 rows
df.count()  # Returns count
df.first()  # Returns first dict or None
df.take(10)  # Returns first 10 rows
df.head()  # Returns first 5 rows

# Streaming
for chunk in df.collect(stream=True):  # Returns iterator
    process_chunk(chunk)

# File materialization happens on collect()
# Files are read and materialized into temp tables
# Then SQL operations execute on the temp table
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Lazy Evaluation** | ✅ Yes | ✅ Yes |
| **Action Triggers** | `collect()`, `show()`, `count()`, etc. | `collect()`, `show()`, `count()`, etc. |
| **Return Type** | List of Row objects | List of dicts |
| **File Materialization** | Handled by Spark | Materialized to temp tables on `.collect()` |
| **SQL Compilation** | Catalyst optimizer | SQL compiler (SQLAlchemy) |
| **Streaming** | Structured Streaming API | Iterator-based streaming |
| **Execution Context** | Spark cluster | Database connection |

**Key Differences:**
- PySpark uses Row objects; Moltres uses dicts
- Moltres materializes files to temp tables for SQL pushdown
- PySpark has Structured Streaming; Moltres uses iterator-based streaming
- Both use lazy evaluation

---

## 9. Async Support

### PySpark

```python
# Limited async support
# Mostly synchronous operations
df = spark.read.csv("data.csv")
results = df.collect()  # Synchronous
```

### Moltres

```python
# Full async support
import asyncio
from moltres import async_connect

async def main():
    db = async_connect("postgresql+asyncpg://user:pass@localhost/db")
    
    # All operations are async
    df = await db.load.csv("data.csv")
    results = await df.collect()
    
    # Streaming
    async for chunk in await df.collect(stream=True):
        process_chunk(chunk)
    
    await db.close()

asyncio.run(main())
```

### Comparison

| Aspect | PySpark | Moltres |
|--------|---------|---------|
| **Async Support** | ❌ Limited | ✅ Full async/await |
| **Async Operations** | N/A | All operations async |
| **Async Drivers** | N/A | asyncpg, aiomysql, aiosqlite |
| **Performance** | Cluster-based parallelism | Database connection pooling |

**Key Differences:**
- Moltres has comprehensive async support; PySpark does not
- Moltres async operations use async database drivers
- This is a significant advantage for Moltres in async Python applications

---

## 10. Advanced Features

### 10.1 Window Functions

Both libraries support window functions with similar APIs. Moltres requires window functions in `select()` rather than `withColumn()`.

### 10.2 Pivoting

**PySpark:**
```python
df.groupBy("category").pivot("status").agg(sum("amount"))
```

**Moltres:**
```python
df.pivot("status", values=["active", "inactive"]).agg(sum(col("amount")))
```

Moltres requires explicit values list; PySpark infers from data.

### 10.3 Explode/Array Operations

**PySpark:**
```python
from pyspark.sql.functions import explode
df.select(explode(col("array_col")).alias("value"))
```

**Moltres:**
```python
df.explode(col("array_col"), alias="value")
```

Similar functionality, different API.

### 10.4 CTEs (Common Table Expressions)

**PySpark:**
```python
df.createOrReplaceTempView("temp_view")
spark.sql("WITH cte AS (SELECT * FROM temp_view) SELECT * FROM cte")
```

**Moltres:**
```python
df.cte("cte_name")
df.recursive_cte("cte_name", initial=..., recursive=...)
```

Moltres has native CTE support; PySpark uses SQL strings.

---

## 11. API Design Patterns

### Method Chaining

Both libraries support fluent method chaining:

**PySpark:**
```python
df.select("id", "name").filter(col("age") > 18).orderBy("name")
```

**Moltres:**
```python
df.select("id", "name").where(col("age") > 18).order_by(col("name"))
```

### Naming Conventions

| Pattern | PySpark | Moltres |
|---------|---------|---------|
| **camelCase** | `groupBy()`, `orderBy()`, `saveAsTable()` | `groupBy()`, `order_by()`, `save_as_table()` |
| **snake_case** | `drop_duplicates()` | `drop_duplicates()`, `insert_into()` |
| **Consistency** | Mixed | More consistent (prefers snake_case) |

### Return Type Consistency

- **PySpark**: Returns DataFrame for transformations, specific types for actions
- **Moltres**: Returns DataFrame for transformations, list/dict for actions

Both are consistent within their models.

---

## 12. Feature Gaps Analysis

### What PySpark Has That Moltres Doesn't

1. **Machine Learning (MLlib)**
   - PySpark: Full MLlib integration
   - Moltres: Not applicable (SQL focus)

2. **Structured Streaming**
   - PySpark: Dedicated streaming API
   - Moltres: Iterator-based streaming only

3. **UDFs (User-Defined Functions)**
   - PySpark: Python UDFs, Pandas UDFs
   - Moltres: Not supported (SQL pushdown design)

4. **More File Formats**
   - PySpark: ORC, Avro, Delta Lake, etc.
   - Moltres: CSV, JSON, JSONL, Parquet, Text

5. **Distributed Computing**
   - PySpark: Cluster-based execution
   - Moltres: Single database connection

6. **Catalyst Optimizer**
   - PySpark: Advanced query optimization
   - Moltres: SQL compiler (relies on database optimizer)

### What Moltres Has That PySpark Doesn't

1. **UPDATE Operations**
   - Moltres: `df.write.update()` with DataFrame syntax
   - PySpark: Must use SQL

2. **DELETE Operations**
   - Moltres: `df.write.delete()` with DataFrame syntax
   - PySpark: Must use SQL

3. **Full Async Support**
   - Moltres: Comprehensive async/await
   - PySpark: Limited async

4. **Direct SQL Database Integration**
   - Moltres: Works directly with PostgreSQL, MySQL, SQLite
   - PySpark: Requires Spark cluster

5. **Transaction Support**
   - Moltres: Full transaction context
   - PySpark: Limited transaction support

6. **JSONL Support**
   - Moltres: Native JSONL reader
   - PySpark: Requires workaround

7. **Lateral Joins**
   - Moltres: Native lateral join support
   - PySpark: Limited support

---

## 13. Migration Guide

### Converting PySpark Code to Moltres

#### Example 1: Basic Operations

**PySpark:**
```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg

spark = SparkSession.builder.appName("MyApp").getOrCreate()
df = spark.read.csv("data.csv", header=True)
result = (
    df.select("category", "amount")
    .filter(col("amount") > 100)
    .groupBy("category")
    .agg(sum("amount").alias("total"), avg("amount").alias("avg"))
    .orderBy("total")
)
result.show()
```

**Moltres:**
```python
from moltres import connect, col
from moltres.expressions.functions import sum, avg

db = connect("sqlite:///example.db")
df = db.load.option("header", True).csv("data.csv")
result = (
    df.select("category", "amount")
    .where(col("amount") > 100)
    .group_by("category")
    .agg(sum(col("amount")).alias("total"), avg(col("amount")).alias("avg"))
    .order_by(col("total"))
)
result.show()
```

**Key Changes:**
- `SparkSession` → `connect()`
- `read.csv()` → `load.csv()`
- `filter()` → `where()` (or use `filter()`)
- `groupBy()` → `group_by()` (or use `groupBy()`)
- `orderBy()` → `order_by()`
- Column references in aggregations need `col()` wrapper

#### Example 2: Joins

**PySpark:**
```python
df1 = spark.read.table("customers")
df2 = spark.read.table("orders")
result = df1.join(df2, df1.id == df2.customer_id, "left")
```

**Moltres:**
```python
df1 = db.table("customers").select()
df2 = db.table("orders").select()
result = df1.join(df2, on=[("id", "customer_id")], how="left")
```

**Key Changes:**
- `read.table()` → `db.table().select()`
- Join condition syntax differs

#### Example 3: Updates

**PySpark:**
```python
# Must use SQL
spark.sql("UPDATE customers SET active = 1 WHERE id = 1")
```

**Moltres:**
```python
# DataFrame syntax
df = db.table("customers").select()
df.write.update("customers", where=col("id") == 1, set={"active": 1})
```

**Key Changes:**
- Moltres provides DataFrame syntax for updates

---

## 14. Recommendations for Moltres

### High Priority

1. **Add `withColumnRenamed()` Method**
   - Currently requires `select()` with aliases
   - Would improve API consistency with PySpark

2. **Support SQL String Filters**
   - PySpark allows `filter("age > 18")`
   - Moltres requires Column expressions
   - Consider adding SQL string support for convenience

3. **Improve `dropDuplicates()` Implementation**
   - Current implementation is simplified
   - Should match PySpark's behavior more closely

4. **Add More File Format Support**
   - Consider ORC, Avro if there's demand
   - Focus on formats that map well to SQL

### Medium Priority

5. **Add `selectExpr()` Method**
   - PySpark's `selectExpr()` allows SQL expressions
   - Would provide convenience for complex selects

6. **Improve Error Messages**
   - Add suggestions for common mistakes
   - Better error messages when operations can't be compiled to SQL

7. **Add DataFrame Alias Support**
   - PySpark allows `df.alias("name")`
   - Useful for self-joins and subqueries

### Low Priority

8. **Consider UDF Support (with caveats)**
   - Only if it can be done without breaking SQL pushdown
   - Could compile Python functions to SQL where possible
   - Mark as "experimental" or "limited"

9. **Add More Aggregation Functions**
   - As SQL databases add support
   - Focus on SQL-mappable functions

10. **Improve Documentation**
    - Add more PySpark migration examples
    - Side-by-side comparison examples
    - Common patterns guide

---

## 15. Conclusion

Moltres successfully provides a PySpark-like DataFrame API that compiles to SQL, making it familiar for PySpark users while offering unique advantages:

### Strengths of Moltres

- ✅ **UPDATE/DELETE Operations**: Unique DataFrame syntax for updates and deletes
- ✅ **Async Support**: Comprehensive async/await support
- ✅ **SQL Pushdown**: All operations compile to SQL and execute on database
- ✅ **No Cluster Required**: Works with traditional SQL databases
- ✅ **Transaction Support**: Full transaction context for operations
- ✅ **Lightweight**: Minimal dependencies, easy to deploy

### Strengths of PySpark

- ✅ **Distributed Computing**: Cluster-based execution for very large datasets
- ✅ **Machine Learning**: MLlib integration
- ✅ **Structured Streaming**: Dedicated streaming API
- ✅ **UDF Support**: Python and Pandas UDFs
- ✅ **More File Formats**: ORC, Avro, Delta Lake, etc.
- ✅ **Catalyst Optimizer**: Advanced query optimization

### When to Choose Each

**Choose Moltres when:**
- Working with traditional SQL databases
- Need UPDATE/DELETE operations with DataFrame syntax
- Want SQL pushdown without a cluster
- Require async/await support
- Prefer lightweight, dependency-minimal solution
- Working with datasets that fit in database memory/disk

**Choose PySpark when:**
- Need distributed computing across clusters
- Working with very large datasets requiring Spark's optimizations
- Need machine learning (MLlib) integration
- Require structured streaming at scale
- Working with Hadoop ecosystem
- Need UDF support for complex transformations

### Final Thoughts

Moltres fills a valuable niche in the Python data ecosystem by providing a PySpark-like API for traditional SQL databases. While it doesn't replicate every PySpark feature, it focuses on SQL-mappable operations and provides unique capabilities (UPDATE/DELETE, async support) that PySpark lacks.

The API similarity makes migration from PySpark straightforward, and the SQL pushdown execution model provides excellent performance for database-backed operations. The addition of UPDATE and DELETE operations with DataFrame syntax is a significant advantage for applications that need to modify data.

For teams working with SQL databases who want a DataFrame API without the overhead of a Spark cluster, Moltres is an excellent choice.

---

## Appendix: Method Comparison Tables

### DataFrame Transformation Methods

| Method | PySpark | Moltres | Notes |
|--------|---------|---------|-------|
| `select()` | ✅ | ✅ | Identical |
| `filter()` | ✅ | ✅ | Identical |
| `where()` | ✅ | ✅ | Alias for filter |
| `groupBy()` | ✅ | ✅ | Also `group_by()` |
| `orderBy()` | ✅ | ✅ | Also `order_by()` |
| `sort()` | ✅ | ❌ | Use `order_by()` |
| `join()` | ✅ | ✅ | Similar API |
| `union()` | ✅ | ✅ | Identical |
| `intersect()` | ✅ | ✅ | Identical |
| `except()` | ✅ | ✅ | Also `except_()` |
| `distinct()` | ✅ | ✅ | Identical |
| `dropDuplicates()` | ✅ | ✅ | Simplified in Moltres |
| `withColumn()` | ✅ | ✅ | Identical |
| `withColumnRenamed()` | ✅ | ❌ | Use `select()` with alias |
| `drop()` | ✅ | ✅ | Identical |
| `limit()` | ✅ | ✅ | Identical |
| `sample()` | ✅ | ✅ | Identical |
| `pivot()` | ✅ | ✅ | Different API |
| `explode()` | ✅ | ✅ | Different API |
| `semi_join()` | ✅ | ✅ | Separate method in Moltres |
| `anti_join()` | ✅ | ✅ | Separate method in Moltres |
| `cte()` | ❌ | ✅ | Native CTE support |
| `recursive_cte()` | ❌ | ✅ | Native recursive CTE |

### Action Methods

| Method | PySpark | Moltres | Return Type |
|--------|---------|---------|------------|
| `collect()` | ✅ | ✅ | List[Row] vs List[dict] |
| `show()` | ✅ | ✅ | Prints rows |
| `count()` | ✅ | ✅ | int |
| `first()` | ✅ | ✅ | Row vs dict |
| `take()` | ✅ | ✅ | List[Row] vs List[dict] |
| `head()` | ✅ | ✅ | List[Row] vs List[dict] |
| `toPandas()` | ✅ | ❌ | N/A |
| `toPolars()` | ❌ | ❌ | N/A |

### Read Operations

| Method | PySpark | Moltres | Notes |
|--------|---------|---------|-------|
| `read.csv()` | ✅ | ✅ `load.csv()` | Returns DataFrame |
| `read.json()` | ✅ | ✅ `load.json()` | Returns DataFrame |
| `read.parquet()` | ✅ | ✅ `load.parquet()` | Returns DataFrame |
| `read.text()` | ✅ | ✅ `load.text()` | Returns DataFrame |
| `read.jsonl()` | ❌ | ✅ `load.jsonl()` | Moltres only |
| `read.table()` | ✅ | ✅ `table().select()` | Different API |
| `sql()` | ✅ | ❌ | Use `table().select()` |

### Write Operations

| Method | PySpark | Moltres | Notes |
|--------|---------|---------|-------|
| `write.saveAsTable()` | ✅ | ✅ `save_as_table()` | Identical |
| `write.insertInto()` | ✅ | ✅ `insertInto()` | Identical |
| `write.csv()` | ✅ | ✅ | Identical |
| `write.json()` | ✅ | ✅ | Identical |
| `write.parquet()` | ✅ | ✅ | Identical |
| `write.jsonl()` | ❌ | ✅ | Moltres only |
| `write.update()` | ❌ | ✅ | Moltres only |
| `write.delete()` | ❌ | ✅ | Moltres only |
| `write.partitionBy()` | ✅ | ✅ `partitionBy()` | Identical |

---

*Report generated: 2024*
*Moltres Version: 0.8.0+*
*PySpark Version: 3.x+*

