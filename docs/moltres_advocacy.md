# 🐦🔥 Moltres: The Missing DataFrame Layer for SQL in Python  
*A modern DataFrame interface with real SQL CRUD execution*

```{admonition} Archived
:class: warning

This document captures early advocacy material for Moltres.
It is kept for maintainers and is not part of the primary user docs.
```

## 🌟 Overview

**Moltres fills a major, long-standing gap in the Python data ecosystem:**  
It provides a **DataFrame API** whose operations are **pushed down into SQL**, **without loading data into memory**, **while also supporting real SQL CRUD** (INSERT, UPDATE, DELETE).

This combination does **not exist anywhere else in Python** today.

## 🧩 The Gap in Python’s Ecosystem

Python has powerful DataFrame tools and powerful SQL tools — but **no library connects them in a unified, ergonomic way**.

### What currently exists:

| Category | Examples | Limitation |
|----------|-----------|------------|
| **DataFrame libraries** | Pandas, Polars, Modin | In-memory only. No SQL CRUD. |
| **SQL libraries** | SQLAlchemy, SQLModel, Databases | Row-level CRUD but *not* DataFrame-style. |
| **SQL query builders** | Ibis, SQLGlot, PyPika | Excellent SELECT support, but **no updates/deletes/inserts**. |
| **Distributed DataFrames** | PySpark | Heavy, clustered environment required. |

Across all of these, developers repeatedly ask for:

> “A Pandas/DataFrame-like interface backed by SQL instead of memory.”

But until Moltres, **nobody built it**.

## 🔥 What Makes Moltres Unique

Moltres is the **only** Python library that provides:

| Feature | Pandas/ Polars | Ibis | SQLAlchemy | SQLModel | **Moltres** |
|--------|----------------|------|-------------|-----------|-------------|
| DataFrame API | ✔ | ✔ | ❌ | ❌ | **✔** |
| SQL Pushdown Execution | ❌ | ✔ | ✔ | ✔ | **✔** |
| **Row-Level INSERT/UPDATE/DELETE** | ❌ | ❌ | ✔ | ✔ | **✔** |
| Lazy query building | ✔ (Polars) | ✔ | ⚠️ | ⚠️ | **✔** |
| Operates directly on SQL tables | ⚠️ limited | ✔ | ✔ | ✔ | **✔** |
| Column-oriented transformations | ✔ | ✔ | ❌ | ❌ | **✔** |

## 🎯 Who Needs Moltres?

Moltres solves real problems for:

### **Data Engineers**
Avoid loading millions of rows into memory just to update a subset.

### **Backend Developers**
Replace many ORM operations with cleaner, column-aware DataFrame syntax.

### **Analytics Engineers / dbt Users**
Express SQL models in Python code with DataFrame chaining.

### **Product Engineers**
Validated, type-safe CRUD without hand-writing SQL.

### **Teams migrating off Spark**
A Spark-like DataFrame API but for traditional SQL databases — no cluster required.

## 🛠 Why This Matters

### **Pain Today**
Developers must juggle:
- Pandas or Polars for DataFrame transformations  
- SQLAlchemy/ORMs for persistence  
- Raw SQL for updates/deletes  
- Custom glue to keep everything in sync  

### **Moltres fixes this**
With Moltres:

- Transformations are DataFrame-style  
- Execution happens in SQL  
- No massive DataFrame materialization  
- CRUD is first-class  
- Types and schemas stay consistent  
- Code becomes composable and readable  

## 🧠 Why Moltres Is Important for the Future of Python Data

The industry is moving toward:

- **pushdown execution**  
- **lazy query planning**  
- **typed models**  
- **server-side compute**  
- **Python as a declarative DSL for data**  

Moltres is aligned perfectly with this direction.

It acts as the **SQL-powered backbone** of typed, validated, Pythonic data pipelines.

## 🚀 Summary

**Moltres is not “another DataFrame library.”**  
It provides a core capability missing from Python:

> **A DataFrame layer directly backed by SQL with full CRUD support.**

This makes it uniquely powerful for modern data engineering, backend services, analytics, and hybrid workflows where SQL is the source of truth.

If you work with SQL and Python — **Moltres solves problems you’ve had for years.**
