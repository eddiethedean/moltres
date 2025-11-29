# Moltres Package Plan

```{admonition} Archived
:class: warning

This document outlines an early package and architecture plan for Moltres.
It is kept for maintainers and is not part of the primary user docs.
```

## 1. Name
**Moltres** – inspired by the legendary fire Pokémon, evoking speed, power, and a spark-like DataFrame API.

## 2. Project Goal
Provide a **PySpark DataFrame API** that executes lazily on **real SQL databases**, supporting **select, joins, aggregations, inserts, updates, deletes**, without needing Spark.

## 3. High-Level Architecture

### Core Layers (Bottom → Top)

1. **Expression System**: Columns, literals, functions → produce a symbolic expression tree (like PySpark Column or Polars Expr)
2. **Logical Plan Builder**: select, join, filter, groupby → produce a logical plan tree (Project, Filter, Join, Aggregate, etc.)
3. **SQL Compiler**: Converts the logical plan into SQL (ANSI + dialect adaptations)
4. **Execution Engine**: Uses SQLAlchemy for DB connections, executes SQL, returns DataFrame (pandas/polars)
5. **Mutation Engine**: Supports `insert`, `update`, `delete` on underlying SQL tables

## 4. Package Directory Structure

```text
moltres/
    __init__.py
    config.py
    engine/
        connection.py
        execution.py
        dialects.py
    expressions/
        column.py
        expr.py
        functions.py
    logical/
        plan.py
        operators.py
    dataframe/
        dataframe.py
        groupby.py
    sql/
        compiler.py
        builders.py
    io/
        read.py
        write.py
    table/
        table.py
        mutations.py
    utils/
        exceptions.py
        typing.py
        inspector.py
    tests/
        ...
```

## 5. API Design

### Connect to DB
```python
from moltres import connect

db = connect("postgresql://user:pass@host/db")
```

### Select & Filter
```python
t = db.table("customers")

df = (
    t.select("id", "name", (col("spend") * 1.1).alias("adj_spend"))
     .where(col("active") == True)
     .order_by(col("created_at").desc())
)

df.show()
```

### Joins
```python
df = orders.join(customers, on="customer_id").select(
    customers["name"],
    orders["total"],
)
```

### GroupBy & Aggregations
```python
df = t.groupBy("country").agg(
    sum(col("spend")).alias("total_spend"),
    count("*").alias("n"),
)
```

### Insert
```python
t.insert(new_df)
```

### Update
```python
t.update(
    where=col("status") == "new",
    set={"status": lit("processed")}
)
```

### Delete
```python
t.delete(col("created_at") < "2024-01-01")
```

## 6. Internal Components

### Expression System
- Column expressions (`col("spend") + 1`, `col("country").like("%US%")`)
- Functions (`sum`, `avg`, `upper`, `concat`, etc.)

### Logical Plan Nodes
- Project, Filter, Aggregate, Join, Limit, Sort, TableScan

### SQL Compiler
- Translates logical plan → SQL  
- Handles aliases, expressions, joins, groupings  
- Supports multiple SQL dialects

### Execution Engine
- Uses SQLAlchemy for connections  
- Executes SQL  
- Returns pandas/polars DataFrame

### Mutation Engine
- INSERT, UPDATE, DELETE  
- Returns row count or status

## 7. Development Roadmap

### Week 1: Foundation
- Project structure  
- Column, Literal, basic expressions  
- TableScan  
- Minimal DataFrame wrapper

### Week 2: Logical Plan + Compiler
- select / where / limit  
- Expression → SQL conversion  
- Execute queries → pandas

### Week 3: Full Query Support
- joins  
- groupBy / aggregates  
- orderBy  
- Dialect support (Postgres + SQLite)

### Week 4: Mutation
- insert DataFrame  
- update  
- delete  

### Week 5: Polars Integration
- Convert query results to polars  
- Accept polars DataFrames for insert

### Week 6: Stabilize API + Docs
- Documentation (sphinx/mkdocs)  
- Examples  
- Benchmark suite  
- PyPI packaging

## 8. Testing Plan
- Unit tests: expressions, logical plan, SQL compiler, mutation queries  
- Integration tests: SQLite + PostgreSQL  
- Performance tests: compare Sparklet vs raw SQL

## 9. Value Proposition
- PySpark familiarity, SQL execution  
- No cluster required  
- Safer than writing dynamic SQL  
- Composable & testable  
- Wraps existing SQL databases  
- Lets analysts/engineers write PySpark-style code anywhere
