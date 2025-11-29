# PySpark to Moltres Migration: Working Around Inconsistencies

This guide helps you work around known API inconsistencies when migrating from PySpark to Moltres. For a complete comparison, see [MOLTRES_VS_PYSPARK_COMPARISON.md](MOLTRES_VS_PYSPARK_COMPARISON.md).

---

## Overview

Moltres achieves **100% API compatibility** with PySpark for core DataFrame operations (v0.16.0+). This guide documents the fixes that were made and any remaining considerations.

---

## High Priority Inconsistencies

### 1. `order_by()` / `orderBy()` / `sort()` - String Parameter Support ✅ **FIXED in v0.16.0**

**Status:** Fixed! Moltres now accepts both strings and `Column` objects, matching PySpark behavior.

#### PySpark Code

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("MyApp").getOrCreate()
df = spark.read.table("orders")

# Both of these work in PySpark
result1 = df.orderBy("amount")  # ✅ String
result2 = df.orderBy(col("amount"))  # ✅ Column object
result3 = df.sort("amount")  # ✅ String
result4 = df.sort(col("amount").desc())  # ✅ Column object
```

#### Moltres Code (v0.16.0+)

```python
from moltres import connect, col

db = connect("sqlite:///example.db")
df = db.table("orders").select()

# Both strings and Column objects work in Moltres (v0.16.0+)
result1 = df.order_by("amount")  # ✅ String works!
result2 = df.order_by(col("amount"))  # ✅ Column object works
result3 = df.order_by(col("amount").desc())  # ✅ Descending works
result4 = df.orderBy("amount")  # ✅ PySpark-style alias with string works
result5 = df.sort("amount")  # ✅ PySpark-style alias with string works
```

#### Migration Pattern

**Before (PySpark):**
```python
df.orderBy("category", "amount")
df.sort("name")
```

**After (Moltres v0.16.0+):**
```python
# Works exactly the same - no changes needed!
df.order_by("category", "amount")
df.sort("name")

# Or use PySpark-style aliases:
df.orderBy("category", "amount")
df.sort("name")
```

**Status:** ✅ Fixed in v0.16.0 - No workaround needed!

---

## Medium Priority Inconsistencies

### 2. Window Functions Usage Pattern ✅ **FIXED in v0.16.0**

**Status:** Fixed! Moltres now supports window functions in `withColumn()`, matching PySpark behavior.

#### PySpark Code

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

window = Window.partitionBy("category").orderBy("amount")
df = df.withColumn("row_num", row_number().over(window))
```

#### Moltres Code (v0.16.0+)

```python
from moltres.expressions import functions as F
from moltres import col

# Works exactly the same - no changes needed!
df = df.withColumn(
    "row_num", F.row_number().over(partition_by=col("category"), order_by=col("amount"))
)
```

#### Migration Pattern

**Before (PySpark):**
```python
df = df.withColumn("row_num", row_number().over(window))
df = df.withColumn("rank", rank().over(window))
```

**After (Moltres v0.16.0+):**
```python
# Works exactly the same - no changes needed!
df = df.withColumn(
    "row_num", F.row_number().over(partition_by=col("category"), order_by=col("amount"))
)
df = df.withColumn(
    "rank", F.rank().over(partition_by=col("category"), order_by=col("amount"))
)
```

**Status:** ✅ Fixed in v0.16.0 - No workaround needed!

---

## Low Priority Inconsistencies

### 3. `drop()` - Column Object Support ✅ **FIXED in v0.16.0**

**Status:** Fixed! Moltres now accepts both strings and `Column` objects, matching PySpark behavior.

#### PySpark Code

```python
# Both work in PySpark
df.drop("col1", "col2")  # ✅ Strings
df.drop(col("col1"), col("col2"))  # ✅ Column objects
```

#### Moltres Code (v0.16.0+)

```python
# Both work in Moltres (v0.16.0+)
df.drop("col1", "col2")  # ✅ Strings work
df.drop(col("col1"), col("col2"))  # ✅ Column objects now work!
df.drop("col1", col("col2"))  # ✅ Mixed usage works
```

#### Migration Pattern

**Before (PySpark):**
```python
df.drop(col("col1"), col("col2"))
```

**After (Moltres v0.16.0+):**
```python
# Works exactly the same - no changes needed!
df.drop(col("col1"), col("col2"))
# Or use strings:
df.drop("col1", "col2")
```

**Status:** ✅ Fixed in v0.16.0 - No workaround needed!

---

## General Migration Tips

### 1. Always Import `col` from Moltres

```python
from moltres import col  # Essential for column operations
```

### 2. Use `col()` Wrapper for Column References

When in doubt, wrap column names with `col()`:

```python
# Safe pattern
df.select(col("name"), col("amount"))
df.filter(col("age") > 18)
df.group_by(col("category"))  # Works, but strings also work
df.order_by(col("name"))  # Required for order_by
```

### 3. Check Method Signatures

If a method doesn't work as expected, check the method signature:

```python
# Check what types are accepted
help(df.order_by)  # Shows: order_by(*columns: Column)
help(df.group_by)  # Shows: group_by(*columns: Union[Column, str])
```

### 4. Use PySpark-Style Aliases

Moltres provides PySpark-style camelCase aliases:

```python
# Both work
df.order_by(col("name"))
df.orderBy(col("name"))  # PySpark-style alias

df.group_by("category")
df.groupBy("category")  # PySpark-style alias
```

---

## Complete Migration Example

### PySpark Code

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg

spark = SparkSession.builder.appName("MyApp").getOrCreate()
df = spark.read.table("orders")

result = (
    df.select("category", "amount")
    .filter(col("amount") > 100)
    .groupBy("category")
    .agg(sum("amount").alias("total"), avg("amount").alias("avg"))
    .orderBy("total")  # String column name
)
result.show()
```

### Moltres Code (With Workarounds)

```python
from moltres import connect, col
from moltres.expressions import functions as F

db = connect("sqlite:///example.db")
df = db.table("orders").select()

result = (
    df.select("category", "amount")
    .where(col("amount") > 100)
    .group_by("category")
    .agg(F.sum(col("amount")).alias("total"), F.avg(col("amount")).alias("avg"))
    .order_by(col("total"))  # Must use col() wrapper
)
result.show()
```

**Key Changes:**
1. `SparkSession` → `connect()`
2. `read.table()` → `table().select()`
3. `filter()` → `where()` (or use `filter()`)
4. `groupBy()` → `group_by()` (or use `groupBy()`)
5. `orderBy("total")` → `order_by(col("total"))` ⚠️ **Must use col()**

---

## Testing Your Migration

After migrating, test your code to ensure it works:

```python
# Test basic operations
df = db.table("orders").select()
assert df.count() > 0

# Test sorting (with col() wrapper)
sorted_df = df.order_by(col("amount"))
results = sorted_df.collect()
assert len(results) > 0

# Test grouping
grouped = df.group_by("category").agg(F.sum(col("amount")))
results = grouped.collect()
assert len(results) > 0
```

---

## Getting Help

- **Full API Comparison:** See [MOLTRES_VS_PYSPARK_COMPARISON.md](MOLTRES_VS_PYSPARK_COMPARISON.md)
- **Detailed Audit:** See [PYSPARK_INTERFACE_AUDIT.md](PYSPARK_INTERFACE_AUDIT.md)
- **Examples:** See `docs/examples/` directory for working code samples
- **Issues:** Report inconsistencies or request features on GitHub

---

## Recent Improvements (v0.16.0)

The Moltres team has recently fixed all PySpark compatibility issues:

1. ✅ **String support in `order_by()`** - Fixed in v0.16.0
2. ✅ **Column object support in `drop()`** - Fixed in v0.16.0
3. ✅ **Window functions in `withColumn()`** - Fixed in v0.16.0

**Result:** Moltres now achieves **100% PySpark API compatibility** for core DataFrame operations!

See [PYSPARK_INTERFACE_AUDIT.md](PYSPARK_INTERFACE_AUDIT.md) for the complete roadmap.

---

*Last Updated: 2025*
*Moltres Version: 0.16.0*

