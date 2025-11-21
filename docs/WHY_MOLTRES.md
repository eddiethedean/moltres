# Why Moltres?

## The Missing Piece in Python's Data Ecosystem

Moltres fills a major, long-standing gap in the Python data ecosystem. It provides a **DataFrame API** whose operations are **pushed down into SQL**, **without loading data into memory**, **while also supporting real SQL CRUD** (INSERT, UPDATE, DELETE).

**This combination does not exist anywhere else in Python today.**

## The Gap in Python's Ecosystem

Python has powerful DataFrame tools and powerful SQL tools—but **no library connects them in a unified, ergonomic way**.

### What Currently Exists

| Category | Examples | Limitation |
|----------|-----------|------------|
| **DataFrame libraries** | Pandas, Polars, Modin | In-memory only. No SQL CRUD. |
| **SQL libraries** | SQLAlchemy, SQLModel, Databases | Row-level CRUD but *not* DataFrame-style. |
| **SQL query builders** | Ibis, SQLGlot, PyPika | Excellent SELECT support, but **no updates/deletes/inserts**. |
| **Distributed DataFrames** | PySpark | Heavy, clustered environment required. |

Across all of these, developers repeatedly ask for:

> "A Pandas/DataFrame-like interface backed by SQL instead of memory."

But until Moltres, **nobody built it**.

## What Makes Moltres Unique

Moltres is the **only** Python library that provides:

| Feature | Pandas/Polars | Ibis | SQLAlchemy | SQLModel | **Moltres** |
|--------|----------------|------|-------------|-----------|-------------|
| DataFrame API | ✔ | ✔ | ❌ | ❌ | **✔** |
| SQL Pushdown Execution | ❌ | ✔ | ✔ | ✔ | **✔** |
| **Row-Level INSERT/UPDATE/DELETE** | ❌ | ❌ | ✔ | ✔ | **✔** |
| Lazy query building | ✔ (Polars) | ✔ | ⚠️ | ⚠️ | **✔** |
| Operates directly on SQL tables | ⚠️ limited | ✔ | ✔ | ✔ | **✔** |
| Column-oriented transformations | ✔ | ✔ | ❌ | ❌ | **✔** |

## Who Needs Moltres?

Moltres solves real problems for:

### Data Engineers

**Problem:** Need to update millions of rows, but loading data into memory is impractical.

**Solution:** Use Moltres DataFrame operations that compile to SQL UPDATE statements. No data loading required.

```python
# Update millions of rows without loading into memory
orders = db.table("orders")
orders.update(
    where=col("status") == "pending",
    set={"status": "processing", "updated_at": "2024-01-15"}
)
```

### Backend Developers

**Problem:** ORM operations are verbose and don't support column-aware bulk operations well.

**Solution:** Replace many ORM operations with cleaner, column-aware DataFrame syntax.

```python
# Instead of row-by-row ORM updates
users.update(
    where=col("status") == "pending",
    set={"status": "active", "updated_at": "2024-01-15"}
)
```

### Analytics Engineers / dbt Users

**Problem:** Want to express SQL models in Python code with DataFrame chaining.

**Solution:** Build analytics pipelines using composable DataFrame operations that compile to SQL.

```python
# Build models like dbt, but in Python
customer_metrics = (
    db.table("orders")
    .group_by("customer_id")
    .agg(sum(col("amount")).alias("lifetime_value"))
)
```

### Product Engineers

**Problem:** Need validated, type-safe CRUD without hand-writing SQL.

**Solution:** Moltres provides type-safe CRUD operations with DataFrame-style syntax.

```python
# Type-safe, validated CRUD
users.insert([{"name": "Alice", "email": "alice@example.com"}])
users.update(where=col("id") == 1, set={"status": "active"})
```

### Teams Migrating Off Spark

**Problem:** Want Spark-like DataFrame API but for traditional SQL databases—no cluster required.

**Solution:** Moltres provides a Spark-like DataFrame API that works with existing SQL infrastructure.

```python
# Familiar Spark-style operations
df = (
    db.table("orders")
    .select()
    .where(col("status") == "completed")
    .group_by("country")
    .agg(sum(col("amount")).alias("total"))
)
```

## Why This Matters

### Pain Today

Developers must juggle:
- Pandas or Polars for DataFrame transformations  
- SQLAlchemy/ORMs for persistence  
- Raw SQL for updates/deletes  
- Custom glue to keep everything in sync  

### Moltres Fixes This

With Moltres:

- Transformations are DataFrame-style  
- Execution happens in SQL  
- No massive DataFrame materialization  
- CRUD is first-class  
- Types and schemas stay consistent  
- Code becomes composable and readable  

## Why Moltres Is Important for the Future of Python Data

The industry is moving toward:

- **pushdown execution**  
- **lazy query planning**  
- **typed models**  
- **server-side compute**  
- **Python as a declarative DSL for data**  

Moltres is aligned perfectly with this direction.

It acts as the **SQL-powered backbone** of typed, validated, Pythonic data pipelines.

## Summary

**Moltres is not "another DataFrame library."**  
It provides a core capability missing from Python:

> **A DataFrame layer directly backed by SQL with full CRUD support.**

This makes it uniquely powerful for modern data engineering, backend services, analytics, and hybrid workflows where SQL is the source of truth.

If you work with SQL and Python—**Moltres solves problems you've had for years.**

## See Also

- [Examples](EXAMPLES.md) - Practical examples for each use case
- [Advocacy Document](moltres_advocacy.md) - Detailed positioning and comparison
- [Design Notes](moltres_plan.md) - Architecture and design decisions

