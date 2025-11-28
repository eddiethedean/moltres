# PySpark Feature Comparison

This document provides a comprehensive comparison of PySpark DataFrame API features against all 6 Moltres interfaces:

1. **PySpark-Style (Sync)** - `DataFrame` - Primary PySpark-compatible API
2. **PySpark-Style (Async)** - `AsyncDataFrame` - Async version of PySpark-style API
3. **Pandas-Style (Sync)** - `PandasDataFrame` - Pandas-compatible API
4. **Pandas-Style (Async)** - `AsyncPandasDataFrame` - Async version of Pandas-style API
5. **Polars-Style (Sync)** - `PolarsDataFrame` - Polars LazyFrame-compatible API
6. **Polars-Style (Async)** - `AsyncPolarsDataFrame` - Async version of Polars-style API

## Status Indicators

- ✅ **Supported** - Fully implemented with full feature parity
- ⚠️ **Partial** - Partially implemented or with limitations
- ❌ **Not Implemented** - Not available in this interface
- 🔄 **Different API** - Available but with different method name/API signature
- 📝 **Notes** - Additional implementation details, differences, or limitations

## Selection & Projection

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `select(*cols)` | ✅ | ✅ | ✅ (`select()`) | ✅ (`select()`) | ✅ | ✅ | All interfaces support column selection |
| `selectExpr(*exprs)` | ✅ | ✅ | ✅ (`select_expr()`) | ✅ (`select_expr()`) | ✅ (`select_expr()`) | ✅ (`select_expr()`) | SQL expression selection |
| Column access `df.col` | ✅ (`__getattr__`) | ✅ (`__getattr__`) | 🔄 (`df['col']`) | 🔄 (`df['col']`) | 🔄 (`df['col']`) | 🔄 (`df['col']`) | PySpark-style supports dot notation; Pandas/Polars use bracket notation `df['col']` instead |
| `df["col"]` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Pandas/Polars-style column access (bracket notation) |
| `df[["col1", "col2"]]` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Multi-column selection (Pandas/Polars) |
| `df[df["col"] > 5]` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Boolean indexing (Pandas/Polars) |

## Filtering

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `where(condition)` | ✅ | ✅ | 🔄 (`query()`) | 🔄 (`query()`) | ✅ (`filter()`) | ✅ (`filter()`) | All support filtering |
| `filter(condition)` | ✅ | ✅ | 🔄 (`query()`) | 🔄 (`query()`) | ✅ | ✅ | Alias for `where()` in PySpark-style |
| `query(expr)` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | Pandas-style string query syntax |
| `isin(values)` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | Pandas-style membership check |
| `between(start, end)` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | Pandas-style range check |

## Joins

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `join(other, on, how)` | ✅ | ✅ | ✅ (`merge()`) | ✅ (`merge()`) | ✅ | ✅ | All support joins |
| `crossJoin(other)` | ✅ | ✅ | ❌ | ❌ | ✅ (`cross_join()`) | ✅ (`cross_join()`) | Cross join support |
| `semi_join(other, on)` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Semi-join (filter rows with matches) |
| `anti_join(other, on)` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Anti-join (filter rows without matches) |
| Join types: `inner`, `left`, `right`, `outer` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | All join types supported |

## GroupBy & Aggregations

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `groupBy(*cols)` | ✅ | ✅ | ✅ (`groupby()`) | ✅ (`groupby()`) | ✅ (`group_by()`) | ✅ (`group_by()`) | All support grouping |
| `agg(*exprs)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Aggregation expressions |
| `agg({"col": "func"})` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Dictionary syntax (PySpark/Pandas) |
| `sum()`, `mean()`, `min()`, `max()`, `count()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Basic aggregations |
| `first()`, `last()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | First/last value per group |
| `nunique()` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Count distinct (Pandas/Polars) |
| `std()`, `var()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Statistical aggregations |
| `pivot(pivot_col)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Pivot operation |

## Sorting

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `orderBy(*cols)` | ✅ | ✅ | ✅ (`sort_values()`) | ✅ (`sort_values()`) | ✅ (`sort()`) | ✅ (`sort()`) | All support sorting |
| `sort(*cols)` | ✅ | ✅ | ✅ (`sort_values()`) | ✅ (`sort_values()`) | ✅ | ✅ | Alias for `orderBy()` |
| `sort_values(by, ascending)` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | Pandas-style sorting |
| `sort(by, descending)` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Polars-style sorting |

## Window Functions

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `over(windowSpec)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Window function support |
| `row_number()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Row number window function |
| `rank()`, `dense_rank()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Ranking functions |
| `lead()`, `lag()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Lead/lag functions |
| `first_value()`, `last_value()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | First/last value in window |

## Set Operations

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `union(other)` | ✅ | ✅ | ✅ (`concat()`) | ✅ (`concat()`) | ✅ | ✅ | Union operation |
| `unionAll(other)` | ✅ | ✅ | ✅ (`concat()`) | ✅ (`concat()`) | ✅ | ✅ | Union all (same as union) |
| `intersect(other)` | ✅ | ✅ | ✅ (`concat()` + dedup) | ✅ (`concat()` + dedup) | ✅ | ✅ | Intersection |
| `except(other)` | ✅ | ✅ | ✅ (`concat()` + filter) | ✅ (`concat()` + filter) | ✅ (`difference()`) | ✅ (`difference()`) | Set difference |
| `distinct()` | ✅ | ✅ | ✅ (`drop_duplicates()`) | ✅ (`drop_duplicates()`) | ✅ | ✅ | Remove duplicates |
| `dropDuplicates(subset)` | ✅ | ✅ | ✅ | ✅ | ✅ (`unique()`) | ✅ (`unique()`) | Drop duplicates with subset |

## Column Manipulation

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `withColumn(name, expr)` | ✅ | ✅ | ✅ (`assign()`) | ✅ (`assign()`) | ✅ (`with_column()`) | ✅ (`with_column()`) | Add/replace column |
| `withColumns({name: expr})` | ❌ | ❌ | ✅ (`assign()`) | ✅ (`assign()`) | ✅ (`with_columns()`) | ✅ (`with_columns()`) | Multiple columns |
| `drop(*cols)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Drop columns |
| `withColumnRenamed(old, new)` | ✅ | ✅ | ✅ (`rename()`) | ✅ (`rename()`) | ✅ (`rename()`) | ✅ (`rename()`) | Rename column |
| `alias(name)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Column alias |
| `assign(**kwargs)` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | Pandas-style column assignment |

## Null Handling

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `fillna(value)` | ✅ | ✅ | ✅ | ✅ | ✅ (`fill_null()`) | ✅ (`fill_null()`) | Fill null values |
| `fillna({col: value})` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Column-specific fill |
| `dropna(how, subset)` | ✅ | ✅ | ✅ | ✅ | ✅ (`drop_nulls()`) | ✅ (`drop_nulls()`) | Drop null rows |
| `na.drop()` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Null handling property |
| `na.fill(value)` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Null handling property |

## String Operations

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `df["col"].str.upper()` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String accessor |
| `df["col"].str.lower()` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String accessor |
| `df["col"].str.contains(pattern)` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String contains |
| `df["col"].str.startswith(pattern)` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String startswith |
| `df["col"].str.endswith(pattern)` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String endswith |
| `df["col"].str.replace(old, new)` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String replace |
| `df["col"].str.split(delimiter)` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String split |
| `df["col"].str.len()` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | String length |
| `upper(col)`, `lower(col)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | String functions (all) |
| `substring(col, pos, len)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Substring function |

## Date/Time Operations

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `df["col"].dt.year` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Date accessor |
| `df["col"].dt.month` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Date accessor |
| `df["col"].dt.day` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Date accessor |
| `df["col"].dt.hour` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Date accessor |
| `year(col)`, `month(col)`, etc. | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Date functions (all) |
| `to_date(col)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Date conversion |
| `to_timestamp(col)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Timestamp conversion |
| `date_add(col, days)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Date arithmetic |
| `date_sub(col, days)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Date arithmetic |

## File I/O

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `spark.read.csv(path)` | ✅ (`db.read.csv()`) | ✅ (`db.read.csv()`) | ❌ | ❌ | ✅ (`db.scan_csv()`) | ✅ (`db.scan_csv()`) | Read CSV |
| `spark.read.json(path)` | ✅ (`db.read.json()`) | ✅ (`db.read.json()`) | ❌ | ❌ | ✅ (`db.scan_json()`) | ✅ (`db.scan_json()`) | Read JSON |
| `spark.read.parquet(path)` | ✅ (`db.read.parquet()`) | ✅ (`db.read.parquet()`) | ❌ | ❌ | ✅ (`db.scan_parquet()`) | ✅ (`db.scan_parquet()`) | Read Parquet |
| `spark.read.text(path)` | ✅ (`db.read.text()`) | ✅ (`db.read.text()`) | ❌ | ❌ | ✅ (`db.scan_text()`) | ✅ (`db.scan_text()`) | Read text |
| `df.write.csv(path)` | ✅ | ✅ | ❌ | ❌ | ✅ (`write_csv()`) | ✅ (`write_csv()`) | Write CSV |
| `df.write.json(path)` | ✅ | ✅ | ❌ | ❌ | ✅ (`write_json()`) | ✅ (`write_json()`) | Write JSON |
| `df.write.parquet(path)` | ✅ | ✅ | ❌ | ❌ | ✅ (`write_parquet()`) | ✅ (`write_parquet()`) | Write Parquet |
| `df.write.saveAsTable(name)` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Save to table |
| `df.write.insertInto(table)` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Insert into table |
| `df.write.mode("overwrite")` | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | Write mode |

## Schema Operations

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `df.columns` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Column names |
| `df.schema` | ✅ | ✅ | 🔄 (`dtypes`) | 🔄 (`dtypes`) | ✅ | ✅ | Schema information |
| `df.dtypes` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Column types (Pandas-style) |
| `df.printSchema()` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | Print schema tree |
| `df.schema` (Polars) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Polars schema format |

## Execution Methods

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `collect()` | ✅ | ✅ (`await collect()`) | ✅ | ✅ (`await collect()`) | ✅ | ✅ (`await collect()`) | Execute and return results |
| `collect()` (streaming) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Streaming execution |
| `show(n, truncate)` | ✅ | ✅ (`await show()`) | ❌ | ❌ | ❌ | ❌ | Display results |
| `take(n)` | ✅ | ✅ (`await take()`) | ❌ | ❌ | ✅ (`fetch()`) | ✅ (`await fetch()`) | Take n rows |
| `first()` | ✅ | ✅ (`await first()`) | ❌ | ❌ | ❌ | ❌ | First row |
| `head(n)` | ✅ | ✅ (`await head()`) | ✅ | ✅ | ✅ | ✅ | First n rows |
| `tail(n)` | ✅ | ✅ (`await tail()`) | ✅ | ✅ | ✅ | ✅ | Last n rows |
| `count()` | ✅ | ✅ (`await count()`) | ❌ | ❌ | ❌ | ❌ | Row count |
| `fetch(n)` (Polars) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ (`await fetch()`) | Polars-style fetch |

## Statistics & Descriptive

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `describe(*cols)` | ✅ | ✅ (`await describe()`) | ✅ | ✅ (`await describe()`) | ✅ (`await describe()`) | ✅ (`await describe()`) | Statistical summary |
| `summary(*stats)` | ✅ | ✅ (`await summary()`) | ❌ | ❌ | ❌ | ❌ | Custom statistics |
| `nunique(column)` | ❌ | ❌ | ✅ | ✅ (`await nunique()`) | ✅ | ✅ | Count unique values |
| `value_counts(column)` | ❌ | ❌ | ✅ | ✅ (`await value_counts()`) | ❌ | ❌ | Value frequency (Pandas) |
| `info()` | ❌ | ❌ | ✅ | ✅ (`await info()`) | ❌ | ❌ | DataFrame info (Pandas) |
| `empty` | ❌ | ❌ | ✅ | ✅ (`await empty`) | ❌ | ❌ | Check if empty (Pandas) |
| `shape` | ❌ | ❌ | ✅ | ✅ (`await shape`) | ❌ | ❌ | DataFrame shape (Pandas) |
| `width`, `height` (Polars) | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ (`await height`) | Polars dimensions |

## Data Reshaping

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `pivot(pivot_col, values)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Pivot operation |
| `explode(col)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Explode array/JSON |
| `melt(id_vars, value_vars)` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | Unpivot (Pandas/Polars) |
| `unnest(cols)` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Unnest nested structures (Polars) |
| `slice(offset, length)` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Slice rows (Polars) |

## Sampling & Limiting

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `sample(fraction, seed)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Random sampling |
| `limit(n)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Limit rows |

## CTEs & SQL

| PySpark Method | PySpark-Style (Sync) | PySpark-Style (Async) | Pandas-Style (Sync) | Pandas-Style (Async) | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------------|---------------------|----------------------|-------------------|---------------------|-------------------|---------------------|-------|
| `cte(name)` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Common Table Expression |
| `with_recursive(name)` | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | Recursive CTE (Polars) |
| `spark.sql(query)` | ✅ (`db.sql()`) | ✅ (`await db.sql()`) | ❌ | ❌ | ❌ | ❌ | Raw SQL execution |
| `to_sql()` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | Get SQL string |
| `to_sqlalchemy()` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Get SQLAlchemy statement |

## Interface-Specific Features

### Pandas-Style Unique Features

| Feature | Pandas-Style (Sync) | Pandas-Style (Async) | Notes |
|---------|-------------------|---------------------|-------|
| `query(expr)` | ✅ | ✅ | String-based query syntax |
| `loc` indexer | ✅ | ✅ (`await loc`) | Label-based indexing |
| `iloc` indexer | ✅ | ✅ (`await iloc`) | Integer-based indexing |
| `append(other)` | ✅ | ✅ | Append DataFrames |
| `isin(values)` | ✅ | ✅ | Membership check |
| `between(start, end)` | ✅ | ✅ | Range check |
| `assign(**kwargs)` | ✅ | ✅ | Column assignment |
| `sort_values(by, ascending)` | ✅ | ✅ | Sorting with parameters |
| `value_counts(column)` | ✅ | ✅ | Value frequency |
| `info()` | ✅ | ✅ | DataFrame info |
| `melt()` | ✅ | ✅ | Unpivot operation |

### Polars-Style Unique Features

| Feature | Polars-Style (Sync) | Polars-Style (Async) | Notes |
|---------|-------------------|---------------------|-------|
| `lazy()` | ✅ | ✅ | Mark as lazy (no-op in Moltres) |
| `fetch(n)` | ✅ | ✅ (`await fetch()`) | Fetch n rows |
| `with_columns(*exprs)` | ✅ | ✅ | Multiple column operations |
| `with_columns_renamed(mapping)` | ✅ | ✅ | Rename multiple columns |
| `with_row_count(name)` | ✅ | ✅ | Add row number column |
| `with_context(df)` | ✅ | ✅ | Add context DataFrame |
| `with_recursive(name)` | ✅ | ✅ | Recursive CTE |
| `unnest(cols)` | ✅ | ✅ | Unnest nested structures |
| `slice(offset, length)` | ✅ | ✅ | Slice rows |
| `gather_every(n, offset)` | ✅ | ✅ | Sample every nth row |
| `interpolate(method)` | ✅ | ✅ | Interpolate missing values |
| `quantile(quantile)` | ✅ | ✅ | Quantile calculation |
| `hstack(other)` | ✅ | ✅ | Horizontal stack |
| `vstack(other)` | ✅ | ✅ | Vertical stack |
| `difference(other)` | ✅ | ✅ | Set difference |
| `cross_join(other)` | ✅ | ✅ | Cross join |
| `drop_nulls(subset)` | ✅ | ✅ | Drop nulls |
| `fill_null(value)` | ✅ | ✅ | Fill nulls |
| `unique(subset)` | ✅ | ✅ | Unique rows |
| `explain(format)` | ✅ | ✅ (`await explain()`) | Query plan explanation |

### Async-Specific Features

| Feature | PySpark-Style (Async) | Pandas-Style (Async) | Polars-Style (Async) | Notes |
|---------|---------------------|---------------------|---------------------|-------|
| `await collect()` | ✅ | ✅ | ✅ | Async execution |
| `await collect(stream=True)` | ✅ | ✅ | ✅ | Async streaming |
| `await show()` | ✅ | ❌ | ❌ | Async display |
| `await take()` | ✅ | ❌ | ✅ (`await fetch()`) | Async take |
| `await first()` | ✅ | ❌ | ❌ | Async first row |
| `await head()` | ✅ | ✅ | ✅ | Async head |
| `await tail()` | ✅ | ✅ | ✅ | Async tail |
| `await count()` | ✅ | ❌ | ❌ | Async count |
| `await describe()` | ✅ | ✅ | ✅ | Async describe |
| `await summary()` | ✅ | ❌ | ❌ | Async summary |
| `await nunique()` | ❌ | ✅ | ✅ | Async nunique |
| `await value_counts()` | ❌ | ✅ | ❌ | Async value_counts |
| `await info()` | ❌ | ✅ | ❌ | Async info |
| `await shape` | ❌ | ✅ | ❌ | Async shape |
| `await empty` | ❌ | ✅ | ❌ | Async empty |
| `await height` | ❌ | ❌ | ✅ | Async height |
| `await schema` | ❌ | ❌ | ✅ | Async schema |
| `await dtypes` | ❌ | ✅ | ❌ | Async dtypes |
| `await fetch()` | ❌ | ❌ | ✅ | Async fetch |
| `await write_csv()` | ❌ | ❌ | ✅ | Async write CSV |
| `await write_json()` | ❌ | ❌ | ✅ | Async write JSON |
| `await write_parquet()` | ❌ | ❌ | ✅ | Async write Parquet |
| `await explain()` | ❌ | ❌ | ✅ | Async explain |
| `await loc` | ❌ | ✅ | ❌ | Async loc indexer |
| `await iloc` | ❌ | ✅ | ❌ | Async iloc indexer |

## Summary

### Overall Coverage

- **PySpark-Style (Sync)**: ~98% API compatibility with PySpark DataFrame API
- **PySpark-Style (Async)**: Full async support for all sync methods
- **Pandas-Style (Sync)**: Comprehensive Pandas DataFrame API with SQL pushdown
- **Pandas-Style (Async)**: Full async support for all Pandas-style methods
- **Polars-Style (Sync)**: Comprehensive Polars LazyFrame API with SQL pushdown
- **Polars-Style (Async)**: Full async support for all Polars-style methods

### Key Differences

1. **Column Access**: PySpark uses `df.col` attribute access, while Pandas/Polars use `df['col']` bracket notation
2. **Filtering**: PySpark uses `where()`/`filter()`, Pandas uses `query()`, Polars uses `filter()`
3. **GroupBy**: PySpark uses `groupBy()`, Pandas uses `groupby()`, Polars uses `group_by()`
4. **Sorting**: PySpark uses `orderBy()`, Pandas uses `sort_values()`, Polars uses `sort()`
5. **Null Handling**: PySpark uses `na` property, Pandas/Polars use direct methods
6. **File I/O**: PySpark uses `spark.read.*`, Moltres uses `db.read.*` or `db.scan_*` (Polars)
7. **Async**: All async interfaces require `await` for execution methods

### Implementation Notes

- All interfaces maintain lazy evaluation until execution
- SQL pushdown execution for all operations
- Type safety with proper type hints
- Comprehensive error handling and validation
- Support for multiple database dialects (SQLite, PostgreSQL, MySQL, DuckDB)

