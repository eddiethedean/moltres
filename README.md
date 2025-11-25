# Moltres

<div align="center">

[![CI](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml/badge.svg)](https://github.com/eddiethedean/moltres/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://github.com/eddiethedean/moltres)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/eddiethedean/moltres/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**The Missing DataFrame Layer for SQL in Python**

**MOLTRES**: **M**odern **O**perations **L**ayer for **T**ransformations, **R**elational **E**xecution, and **S**QL

[Installation](#-installation) • [Quick Start](#-quick-start) • [Examples](#-examples) • [Documentation](#-documentation)

</div>

---

**Moltres** combines a DataFrame API (like Pandas/Polars), SQL pushdown execution (no data loading into memory), and real SQL CRUD operations (INSERT, UPDATE, DELETE) in one unified interface.

Transform millions of rows using familiar DataFrame operations—all executed directly in SQL without materializing data. Update, insert, and delete with column-aware, type-safe operations.

## ✨ Features

- 🚀 **DataFrame API** - Familiar operations (select, filter, join, groupBy, etc.) like Pandas/Polars/PySpark
- 🎯 **98% PySpark API Compatibility** - Near-complete compatibility for seamless migration
- 🗄️ **SQL Pushdown Execution** - All operations compile to SQL and run on your database—no data loading into memory
- ✏️ **Real SQL CRUD** - INSERT, UPDATE, DELETE operations with DataFrame-style syntax
- 📊 **Multiple Formats** - Read/write CSV, JSON, JSONL, Parquet, and more
- 🐼 **Pandas & Polars Integration** - Pass pandas/polars DataFrames directly to moltres operations (insert, createDataFrame, etc.)
- 🌊 **Streaming Support** - Handle datasets larger than memory with chunked processing
- ⚡ **Async Support** - Full async/await support for all operations
- 🔒 **Security First** - Built-in SQL injection prevention and validation

## 📦 Installation

```bash
pip install moltres

# Optional: For async support
pip install moltres[async-postgresql]  # PostgreSQL
pip install moltres[async-mysql]       # MySQL
pip install moltres[async-sqlite]     # SQLite

# Optional: For pandas/polars result formats
pip install moltres[pandas,polars]
```

## 🚀 Quick Start

### Basic DataFrame Operations

```python
from moltres import col, connect
from moltres.expressions import functions as F
from moltres.table.schema import column

# Connect to your database
db = connect("sqlite:///example.db")

# Create tables and insert data (setup)
db.create_table("orders", [
    column("id", "INTEGER", primary_key=True),
    column("customer_id", "INTEGER"),
    column("amount", "REAL"),
    column("country", "TEXT"),
]).collect()

db.create_table("customers", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("active", "INTEGER"),
    column("country", "TEXT"),
]).collect()

from moltres.io.records import Records
Records.from_list([
    {"id": 1, "customer_id": 1, "amount": 150.0, "country": "UK"},
    {"id": 2, "customer_id": 2, "amount": 300.0, "country": "USA"},
], database=db).insert_into("orders")

Records.from_list([
    {"id": 1, "name": "Alice", "active": 1, "country": "UK"},
    {"id": 2, "name": "Bob", "active": 1, "country": "USA"},
], database=db).insert_into("customers")

# DataFrame operations with SQL pushdown (no data loading into memory)
df = (
    db.table("orders")
    .select()
    .join(db.table("customers").select(), on=[("customer_id", "id")])
    .where(col("active") == True)  # noqa: E712
    .group_by("country")
    .agg(F.sum(col("amount")).alias("total_amount"))
)

# Execute and get results (SQL is compiled and executed here)
results = df.collect()  # Returns list of dicts by default
print(results)
# Output: [{'country': 'UK', 'total_amount': 150.0}, {'country': 'USA', 'total_amount': 300.0}]
```

### Raw SQL & SQL Expressions

```python
from moltres import col, connect
from moltres.table.schema import column

db = connect("sqlite:///example.db")

# Create tables and insert data (setup)
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]).collect()

db.create_table("orders", [
    column("id", "INTEGER", primary_key=True),
    column("amount", "REAL"),
]).collect()

from moltres.io.records import Records
Records.from_list([
    {"id": 1, "name": "Alice", "age": 25},
    {"id": 2, "name": "Bob", "age": 17},
    {"id": 3, "name": "Charlie", "age": 30},
], database=db).insert_into("users")

Records.from_list([
    {"id": 1, "amount": 50.0},
    {"id": 2, "amount": 150.0},
], database=db).insert_into("orders")

# Raw SQL queries (PySpark-style)
df = db.sql("SELECT * FROM users WHERE age > 18")
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 30}]

df = db.sql("SELECT * FROM orders WHERE id = :id", id=1).where(col("amount") > 100)
results = df.collect()
print(results)
# Output: [] (empty if amount <= 100)

# SQL expression selection
df = db.table("orders").select()
results = df.selectExpr("amount * 1.1 as with_tax", "amount as amount_original").collect()
print(results)
# Output: [{'with_tax': 55.00000000000001, 'amount_original': 50.0}, {'with_tax': 165.0, 'amount_original': 150.0}]
```

### CRUD Operations

Moltres provides multiple ways to perform CRUD operations:

**Using Records API:**
```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

db = connect("sqlite:///example.db")

# Create table (setup)
db.create_table("customers", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("active", "INTEGER"),
]).collect()

# Create Records from list (recommended)
records = Records.from_list(
    [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": 1},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": 0},
    ],
    database=db,
)
result = records.insert_into("customers")  # Executes immediately
print(result)
# Output: 2 (number of rows inserted)

# Create Records from multiple dicts (convenience method)
records = Records.from_dicts(
    {"id": 3, "name": "Charlie"},
    {"id": 4, "name": "Diana"},
    database=db,
)
result = records.insert_into("customers")
print(result)
# Output: 2 (number of rows inserted)
```

**Using Database convenience methods:**
```python
from moltres import col, connect
from moltres.table.schema import column

db = connect("sqlite:///example.db")

# Create table (setup)
db.create_table("customers", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("active", "INTEGER"),
]).collect()

# Insert rows directly
result = db.insert("customers", [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
])
print(result)
# Output: 2 (number of rows inserted)

# Update rows
result = db.update("customers", where=col("active") == 0, set={"active": 1})
print(result)
# Output: 1 (number of rows updated)

# Delete rows (example: delete rows where email is null)
# First insert a row without email for demonstration
db.insert("customers", [{"id": 5, "name": "Eve"}])
result = db.delete("customers", where=col("email").is_null())
print(result)
# Output: 1 (number of rows deleted, if any rows have null email)

# Merge (upsert) rows
result = db.merge(
    "customers",
    [{"id": 1, "name": "Alice Updated"}],
    on=["id"],
    when_matched={"name": "Alice Updated"},
)
print(result)
# Output: 1 (number of rows affected)
```

**Using DataFrame write API:**
```python
from moltres import col, connect
from moltres.table.schema import column

db = connect("sqlite:///example.db")

# Create table (setup)
db.create_table("customers", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
    column("active", "INTEGER"),
]).collect()

# Insert some data
from moltres.io.records import Records
Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com", "active": 0},
    {"id": 2, "name": "Bob", "email": None, "active": 1},
], database=db).insert_into("customers")

# Update rows
df = db.table("customers").select()
df.write.update(
    "customers",
    where=col("active") == 0,
    set={"active": 1}
)  # Executes immediately
# Note: Returns None (operation executes immediately)

# Delete rows
df.write.delete("customers", where=col("email").is_null())  # Executes immediately
# Note: Returns None (operation executes immediately)
```

### Pandas & Polars DataFrame Integration

Moltres seamlessly integrates with pandas and polars DataFrames. You can pass DataFrames directly to moltres operations without manual conversion:

```python
import pandas as pd
import polars as pl
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

db = connect("sqlite:///example.db")

# Create table (setup)
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]).collect()

# Create pandas DataFrame
pandas_df = pd.DataFrame([
    {"id": 1, "name": "Alice", "age": 30},
    {"id": 2, "name": "Bob", "age": 25},
])

# Create polars DataFrame
polars_df = pl.DataFrame([
    {"id": 3, "name": "Charlie", "age": 35},
    {"id": 4, "name": "Diana", "age": 28},
])

# Pass pandas DataFrame directly to insert_into
records = Records.from_dataframe(pandas_df, database=db)
result = records.insert_into("users")  # Schema is automatically inferred!
print(f"Inserted {result} rows from pandas DataFrame")
# Output: Inserted 2 rows from pandas DataFrame

# Pass polars DataFrame to createDataFrame
df = db.createDataFrame(polars_df, pk="id")
df.write.insertInto("users")
# Note: Returns None (operation executes immediately)

# Polars LazyFrame support (lazy conversion)
# Note: Requires a CSV file named "data.csv" to exist
# lazy_df = pl.scan_csv("data.csv")
# df = db.createDataFrame(lazy_df)  # Conversion happens lazily
# df.write.insertInto("users")

# Direct insertion with pandas/polars DataFrames
# pandas_df = pd.read_csv("data.csv")
# Records.from_dataframe(pandas_df, database=db).insert_into("users")
```

**Key Features:**
- **Lazy Conversion** - DataFrames are converted to Records only when data is accessed
- **Schema Preservation** - Column types and nullability are automatically inferred
- **No Manual Conversion** - Pass DataFrames directly to `insert_into()`, `createDataFrame()`, etc.
- **Polars LazyFrame Support** - Works with both eager and lazy polars DataFrames

### Convenient Data Inspection

Moltres provides convenient methods for exploring your data:

```python
from moltres import connect
from moltres.table.schema import column

db = connect("sqlite:///example.db")

# Create table and insert data (setup)
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("age", "INTEGER"),
]).collect()

from moltres.io.records import Records
Records.from_list([
    {"id": i, "name": f"User{i}", "age": 20 + i} for i in range(1, 16)
], database=db).insert_into("users")

df = db.table("users").select()

# Get first few rows
first_rows = df.head(10)  # Returns first 10 rows
print(f"First row: {first_rows[0] if first_rows else None}")
# Output: First row: {'id': 1, 'name': 'User1', 'age': 21}

# Get last few rows (requires materializing entire DataFrame)
last_rows = df.tail(5)  # Returns last 5 rows
print(f"Last row: {last_rows[-1] if last_rows else None}")
# Output: Last row: {'id': 15, 'name': 'User15', 'age': 35}

# Show DataFrame contents (formatted output)
df.show(20)  # Prints first 20 rows in a formatted table
# Output:
# id | name   | age
# -----------------
# 1  | User1  | 21
# 2  | User2  | 22
# ... (truncated)

# Print schema
df.printSchema()  # Prints schema in tree format
# Output:
# root
#  |-- id: INTEGER (nullable = true)
#  |-- name: TEXT (nullable = true)
#  |-- age: INTEGER (nullable = true)

# Get query execution plan
plan = df.explain()  # Estimated plan
print(f"Plan length: {len(plan)}")
# Output: Plan length: 60 (varies by database)

plan = df.explain(analyze=True)  # Actual execution stats (PostgreSQL)
# Note: SQLite uses EXPLAIN QUERY PLAN, not EXPLAIN ANALYZE
```

### SQLAlchemy ORM Model Integration

Moltres provides seamless integration with SQLAlchemy ORM models, allowing you to create tables and query using your existing model classes:

```python
from sqlalchemy import Column, ForeignKey, Integer, String, Numeric, DateTime
from sqlalchemy.orm import DeclarativeBase
from moltres import col, connect
from moltres.io.records import Records

# Define SQLAlchemy models
class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    age = Column(Integer, nullable=True)

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Numeric(10, 2))
    created_at = Column(DateTime)

db = connect("sqlite:///example.db")

# Create tables directly from model classes
db.create_table(User).collect()
db.create_table(Order).collect()

# Insert data
Records.from_list([
    {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
    {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
], database=db).insert_into("users")

# Query using model classes
df = db.table(User).select().where(col("age") > 25)
results = df.collect()
print(results)
# Output: [{'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 30}]

# Model-based joins
df = (
    db.table(Order)
    .select()
    .join(db.table(User), on=[("user_id", "id")])
)
results = df.collect()
print(results)
# Output: [{'id': 1, 'user_id': 1, 'amount': 100.50, ...}, ...]

# Access model class from table handle
user_handle = db.table(User)
print(user_handle.model_class)  # <class '__main__.User'>
print(user_handle.name)  # 'users'

# Backward compatibility: traditional API still works
from moltres.table.schema import column
db.create_table("products", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()
```

**Key Features:**
- ✅ Create tables from SQLAlchemy models automatically
- ✅ Query using model classes instead of table names
- ✅ Model-based joins and relationships
- ✅ Automatic constraint extraction (primary keys, foreign keys, unique, check)
- ✅ Full backward compatibility with existing API
- ✅ Async support for SQLAlchemy models

**See more examples:** `examples/17_sqlalchemy_models.py`

### Async Support

```python
import asyncio
from moltres import async_connect, col

async def main():
    # Note: Requires async dependencies: pip install moltres[async-postgresql]
    db = await async_connect("postgresql+asyncpg://user:pass@localhost/db")
    
    df = await db.table("orders").select()
    results = await df.collect()
    print(f"Results: {results}")
    # Output: Results: [{'id': 1, ...}, {'id': 2, ...}]  # Actual output depends on data
    
    # Streaming support
    async for chunk in await df.collect(stream=True):
        print(f"Chunk: {chunk}")
        # Output: Chunk: [{'id': 1, ...}, ...]  # Processed in chunks
    
    await db.close()

# Uncomment to run:
# asyncio.run(main())
```

## 📖 Core Concepts

### Lazy Evaluation

All DataFrame query operations are lazy—they build a logical plan that only executes when you call `collect()`. DataFrame write operations (insertInto, update, delete) execute eagerly (immediately), matching PySpark's behavior.

```python
# This doesn't execute any SQL yet
df = db.table("users").select().where(col("age") > 18)

# SQL is compiled and executed here
results = df.collect()
# Output: [{'id': 1, 'name': 'Alice', 'age': 25}, {'id': 3, 'name': 'Charlie', 'age': 30}]
# Note: Requires users table with data to produce this output
```

### Column Expressions

Moltres supports multiple ways to reference columns:
- **String names**: `df.select("id", "name")`
- **Dot notation**: `df.select(df.id, df.name)` (PySpark-style)
- **col() function**: `df.select(col("id"), col("name"))`
- **Mix and match**: Combine all three methods in the same query

**📚 See detailed examples:**
- [Column expressions and functions](https://github.com/eddiethedean/moltres/blob/main/examples/06_expressions.py)

## 📥 Reading Data

Moltres supports reading from database tables, raw SQL queries, and files (CSV, JSON, Parquet, etc.). All readers return lazy `DataFrame` objects that can be transformed before execution.

**Key Features:**
- Read from tables: `db.table("table_name").select()` or `db.read.table("table_name")`
- Raw SQL queries: `db.sql("SELECT * FROM users WHERE age > 18")`
- SQL expressions: `df.selectExpr("amount * 1.1 as with_tax")`
- File formats: CSV, JSON, JSONL, Parquet, Text
- Schema inference or explicit schemas
- Lazy evaluation - files materialize only when `.collect()` is called

**📚 See detailed examples:**
- [Reading from tables and SQL](https://github.com/eddiethedean/moltres/blob/main/examples/02_dataframe_basics.py)
- [Reading files (CSV, JSON, Parquet, Text)](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)
- [Raw SQL operations](https://github.com/eddiethedean/moltres/blob/main/examples/12_sql_operations.py)

## 📤 Writing Data

Write DataFrames to database tables or files (CSV, JSON, Parquet, etc.) using the `write` API.

**Key Features:**
- Save to tables: `df.write.save_as_table("table_name")`
- Insert into existing tables: `df.write.insertInto("table_name")`
- Update/Delete operations: `df.write.update()` / `df.write.delete()`
- Multiple file formats: CSV, JSON, JSONL, Parquet, Text
- Write modes: `append`, `overwrite`, `ignore`, `error_if_exists`
- Partitioned writes and streaming support

**📚 See detailed examples:**
- [Writing to files](https://github.com/eddiethedean/moltres/blob/main/examples/08_file_writing.py)
- [Table operations and mutations](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)

## 🌊 Streaming for Large Datasets

Moltres supports streaming for datasets larger than memory. Process data in chunks without loading everything into RAM.

**Key Features:**
- Stream reads: `async for chunk in await df.collect(stream=True)`
- Stream writes: `df.write.stream().csv("output.csv")`
- Configurable chunk sizes
- Works with both sync and async operations

**📚 See detailed examples:**
- [Async DataFrame operations with streaming](https://github.com/eddiethedean/moltres/blob/main/examples/03_async_dataframe.py)
- [File reading with streaming](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)

## 🗄️ Table Management

Create, drop, and manage database tables with explicit schemas or from DataFrames.

**Key Features:**
- Create tables: `db.create_table("name", [column(...)])`
- Create from DataFrames: `df.write.save_as_table("table_name")`
- Drop tables: `db.drop_table("name", if_exists=True)`
- **Constraints**: UNIQUE, CHECK, and FOREIGN KEY constraints
- **Indexes**: Create and drop indexes for better query performance
- Temporary tables, primary keys, and schema validation

**Example:**
```python
from moltres import connect
from moltres.table.schema import column, unique, check, foreign_key

db = connect("sqlite:///example.db")

# Create table with constraints
db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("email", "TEXT"),
        column("age", "INTEGER"),
    ],
    constraints=[
        unique("email", name="uq_user_email"),
        check("age >= 0", name="ck_positive_age"),
    ],
).collect()
# Note: Returns TableHandle, executes on .collect()

# Create table with foreign key
db.create_table(
    "orders",
    [
        column("id", "INTEGER", primary_key=True),
        column("user_id", "INTEGER"),
        column("total", "REAL"),
        column("status", "TEXT"),  # Added for composite index example
    ],
    constraints=[
        foreign_key("user_id", "users", "id", on_delete="CASCADE"),
    ],
).collect()

# Create indexes
db.create_index("idx_user_email", "users", "email").collect()
db.create_index("idx_order_user", "orders", "user_id").collect()
db.create_index("idx_order_user_status", "orders", ["user_id", "status"]).collect()
# Note: All return CreateIndexOperation, execute on .collect()

# Drop index
db.drop_index("idx_user_email", "users").collect()
# Note: Returns DropIndexOperation, executes on .collect()
```

**📚 See detailed examples:**
- [Table operations](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)
- [Creating DataFrames from Python data](https://github.com/eddiethedean/moltres/blob/main/examples/10_create_dataframe.py)

## 🔍 Schema Inspection & Reflection

Inspect and reflect existing database schemas without manually defining them.

**Key Features:**
- List tables: `db.get_table_names()` or `db.show_tables()` (formatted output)
- List views: `db.get_view_names()`
- Get column metadata: `db.get_columns("table_name")` or `db.show_schema("table_name")` (formatted output)
- Reflect single table: `db.reflect_table("table_name")`
- Reflect entire database: `db.reflect()`
- Query execution plans: `db.explain(sql)` or `df.explain()`
- Full async support: All methods available on `AsyncDatabase`

**Example:**
```python
from moltres import connect, col
from moltres.table.schema import column

db = connect("sqlite:///example.db")

# Create tables (setup)
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
    column("email", "TEXT"),
]).collect()

db.create_table("orders", [
    column("id", "INTEGER", primary_key=True),
    column("user_id", "INTEGER"),
]).collect()

db.create_table("products", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Get list of tables (programmatic)
tables = db.get_table_names()
print(tables)
# Output: ['orders', 'products', 'users']  # Order may vary

# Show tables (formatted for interactive use)
db.show_tables()
# Output:
# Tables in database:
#   - orders
#   - products
#   - users

# Get column information (programmatic)
columns = db.get_columns("users")
for col_info in columns:
    print(f"{col_info.name}: {col_info.type_name} (nullable={col_info.nullable}, pk={col_info.primary_key})")
# Output:
# id: INTEGER (nullable=True, pk=True)
# name: TEXT (nullable=True, pk=False)
# email: TEXT (nullable=True, pk=False)

# Show schema (formatted for interactive use)
db.show_schema("users")
# Output:
# Schema for table 'users':
#   - id: INTEGER (primary_key=True)
#   - name: TEXT
#   - email: TEXT

# Reflect a single table
schema = db.reflect_table("users")
print(f"Table: {schema.name}, Columns: {len(schema.columns)}")
# Returns: TableSchema(name='users', columns=[ColumnDef(...), ...])

# Reflect entire database
all_schemas = db.reflect()
for table_name, schema in all_schemas.items():
    print(f"{table_name}: {len(schema.columns)} columns")
# Output:
# orders: 2 columns
# products: 2 columns
# users: 3 columns

# Get query execution plan
plan = db.explain("SELECT * FROM users WHERE id = :id", params={"id": 1})
print(f"Plan length: {len(plan)}")
# Output: Plan length: 24 (varies by database)

# Or from a DataFrame
df = db.table("users").select().where(col("id") == 1)
plan = df.explain()  # Shows estimated plan
plan = df.explain(analyze=True)  # Shows actual execution stats (PostgreSQL)
# Note: SQLite uses EXPLAIN QUERY PLAN, not EXPLAIN ANALYZE
```

**📚 See detailed examples:**
- [Schema inspection and reflection](https://github.com/eddiethedean/moltres/blob/main/examples/14_reflection.py)

## ✏️ Data Mutations

Type-safe INSERT, UPDATE, DELETE, and MERGE operations with DataFrame-style syntax.

**Key Features:**
- Insert: `records.insert_into("table")` or `df.write.insertInto("table")`
- Update: `update_rows(table, where=..., values={...})` or `df.write.update()`
- Delete: `delete_rows(table, where=...)` or `df.write.delete()`
- Merge (Upsert): `merge_rows(table, data, on=[...], when_matched={...}, when_not_matched={...})`
- Transactions: `with db.transaction() as txn: ...`
- Automatic batch operations for multiple rows

**📚 See detailed examples:**
- [Table mutations (insert, update, delete, merge)](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)
- [Transaction management](https://github.com/eddiethedean/moltres/blob/main/examples/13_transactions.py)

## 📊 Result Formats

Moltres supports multiple result formats:
- **Records** (default): List of dictionaries `[{"id": 1, "name": "Alice"}, ...]`
- **Pandas**: `df.collect(format="pandas")` (requires pandas)
- **Polars**: `df.collect(format="polars")` (requires polars)

Configure default format: `db = connect("sqlite:///example.db", fetch_format="pandas")`

## ⚙️ Configuration

Configure Moltres programmatically or via environment variables:

**Programmatic:**
```python
db = connect(
    "sqlite:///example.db",
    echo=False,  # Enable SQL logging
    fetch_format="records",  # Default result format
    pool_size=5,  # Connection pool size
)
# Note: No output - connection is created silently
```

**Environment Variables:**
- `MOLTRES_DSN` - Database connection string
- `MOLTRES_ECHO` - Enable SQL logging (true/false)
- `MOLTRES_FETCH_FORMAT` - Result format: "records", "pandas", or "polars"
- `MOLTRES_POOL_SIZE`, `MOLTRES_MAX_OVERFLOW`, etc. - Connection pool settings

**Connection String Validation:**
Moltres validates connection strings and provides helpful error messages:
- Validates format (must include `://` separator)
- Checks for async driver requirements (`+asyncpg`, `+aiomysql`, `+aiosqlite`)
- Provides suggestions for fixing connection string issues

```python
# Invalid connection string - helpful error message
try:
    from moltres.utils.exceptions import DatabaseConnectionError
    db = connect("invalid-connection-string")
except DatabaseConnectionError as e:
    print(e)  # Clear error message with suggestions
    # Output: Connection string must include '://' separator. Got: invalid-connection-string...
    #         Suggestion: Connection strings should follow the format: 'dialect://user:pass@host:port/dbname'

# Missing async driver - helpful error message
try:
    from moltres import async_connect
    db = async_connect("sqlite:///example.db")  # Missing +aiosqlite
except DatabaseConnectionError as e:
    print(e)  # Suggests using 'sqlite+aiosqlite://'
    # Output: Async SQLite connection requires 'sqlite+aiosqlite://' prefix. Got: sqlite:///example.db...
    #         Suggestion: Use 'sqlite+aiosqlite:///path/to/db.db' for async SQLite connections.
```

**Error Messages with Suggestions:**
Moltres provides helpful "Did you mean?" suggestions for common errors:
- Column name typos suggest similar column names
- Table name typos suggest similar table names
- Connection string issues provide format guidance

See [connection examples](https://github.com/eddiethedean/moltres/blob/main/examples/01_connecting.py) for more details.

## 📈 Performance Monitoring

Optional performance monitoring hooks to track query execution:

```python
from moltres.engine import register_performance_hook

def log_query(sql: str, elapsed: float, metadata: dict):
    print(f"Query took {elapsed:.3f}s, returned {metadata.get('rowcount', 0)} rows")

register_performance_hook("query_end", log_query)
# Note: Output appears when queries execute, e.g.:
# Query took 0.000s, returned 2 rows
```

See the [telemetry module](https://github.com/eddiethedean/moltres/blob/main/src/moltres/utils/telemetry.py) for more details.

## 🔒 Security

Moltres includes built-in security features to prevent SQL injection:
- **SQL Identifier Validation** - All table and column names are validated
- **Parameterized Queries** - All user data is passed as parameters, never string concatenation
- **Input Sanitization** - Comprehensive validation of identifiers and inputs

See [`docs/SECURITY.md`](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md) for security best practices and guidelines.

## 📚 Examples

Comprehensive examples demonstrating all Moltres features:

- **[01_connecting.py](https://github.com/eddiethedean/moltres/blob/main/examples/01_connecting.py)** - Database connections (sync and async)
- **[02_dataframe_basics.py](https://github.com/eddiethedean/moltres/blob/main/examples/02_dataframe_basics.py)** - Basic DataFrame operations (select, filter, order by, limit)
- **[03_async_dataframe.py](https://github.com/eddiethedean/moltres/blob/main/examples/03_async_dataframe.py)** - Asynchronous DataFrame operations
- **[04_joins.py](https://github.com/eddiethedean/moltres/blob/main/examples/04_joins.py)** - Join operations (inner, left, with conditions)
- **[05_groupby.py](https://github.com/eddiethedean/moltres/blob/main/examples/05_groupby.py)** - GroupBy and aggregation operations
- **[06_expressions.py](https://github.com/eddiethedean/moltres/blob/main/examples/06_expressions.py)** - Column expressions, functions, and operators
- **[07_file_reading.py](https://github.com/eddiethedean/moltres/blob/main/examples/07_file_reading.py)** - Reading files (CSV, JSON, JSONL, Parquet, Text)
- **[08_file_writing.py](https://github.com/eddiethedean/moltres/blob/main/examples/08_file_writing.py)** - Writing DataFrames to files
- **[09_table_operations.py](https://github.com/eddiethedean/moltres/blob/main/examples/09_table_operations.py)** - Table operations (create, drop, mutations)
- **[10_create_dataframe.py](https://github.com/eddiethedean/moltres/blob/main/examples/10_create_dataframe.py)** - Creating DataFrames from Python data
- **[11_window_functions.py](https://github.com/eddiethedean/moltres/blob/main/examples/11_window_functions.py)** - Window functions for analytical queries
- **[12_sql_operations.py](https://github.com/eddiethedean/moltres/blob/main/examples/12_sql_operations.py)** - Raw SQL and SQL operations (CTEs, unions, etc.)
- **[13_transactions.py](https://github.com/eddiethedean/moltres/blob/main/examples/13_transactions.py)** - Transaction management
- **[14_reflection.py](https://github.com/eddiethedean/moltres/blob/main/examples/14_reflection.py)** - Schema inspection and reflection
- **[15_pandas_polars_dataframes.py](https://github.com/eddiethedean/moltres/blob/main/examples/15_pandas_polars_dataframes.py)** - Using pandas and polars DataFrames with moltres
- **[16_ux_features.py](https://github.com/eddiethedean/moltres/blob/main/examples/16_ux_features.py)** - UX improvements and convenience methods

See the [examples directory](https://github.com/eddiethedean/moltres/tree/main/examples) for all example files.

## 🛠️ Supported Operations

### DataFrame Operations (PySpark-Compatible)
- `select()` / `selectExpr()` - Project columns or SQL expressions
- `where()` / `filter()` - Filter rows (supports SQL strings)
- `join()` - Join with other DataFrames
- `group_by()` / `groupBy()` - Group rows
- `agg()` - Aggregate functions (supports strings and dictionaries)
- `order_by()` / `orderBy()` / `sort()` - Sort rows
- `limit()` - Limit number of rows
- `distinct()` - Remove duplicate rows
- `withColumn()` / `withColumnRenamed()` - Add or rename columns
- `pivot()` - Pivot operations (including `groupBy().pivot()`)
- `explode()` - Explode array/JSON columns
- `db.sql()` - Execute raw SQL queries

### DataFrame Write Operations
- `df.write.insertInto("table")` - Insert DataFrame into existing table (eager execution)
- `df.write.update("table", where=..., set={...})` - Update rows in table (eager execution)
- `df.write.delete("table", where=...)` - Delete rows from table (eager execution)
- `df.write.save_as_table("table")` / `saveAsTable()` - Write DataFrame to table (eager execution)

### Column Expressions
- **Arithmetic**: `+`, `-`, `*`, `/`, `%`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Boolean**: `&`, `|`, `~`
- **Functions**: Comprehensive function library with 130+ functions including:
  - **Mathematical**: `pow()`, `sqrt()`, `abs()`, `floor()`, `ceil()`, `round()`, `sin()`, `cos()`, `tan()`, `log()`, `exp()`, etc.
  - **String**: `concat()`, `upper()`, `lower()`, `substring()`, `trim()`, `length()`, `replace()`, `regexp_extract()`, `split()`, etc.
  - **Date/Time**: `year()`, `month()`, `day()`, `hour()`, `minute()`, `second()`, `date_format()`, `to_date()`, `datediff()`, `date_add()`, etc.
  - **Aggregate**: `sum()`, `avg()`, `min()`, `max()`, `count()`, `count_distinct()`, `stddev()`, `variance()`, etc.
    - **FILTER clause**: Conditional aggregation with `.filter()` method (e.g., `F.sum(col("amount")).filter(col("status") == "active")`)
  - **Window**: `row_number()`, `rank()`, `dense_rank()`, `lag()`, `lead()`, etc.
  - **Array**: `array()`, `array_length()`, `array_contains()`, `array_position()`, etc.
  - **JSON**: `json_extract()`, `from_json()`, `to_json()`, etc.
  - **Utility**: `coalesce()`, `greatest()`, `least()`, `when()`, `isnull()`, `isnotnull()`, etc.
- **Window Functions**: `over()`, `partition_by()`, `order_by()`

### Supported SQL Dialects
- ✅ **SQLite** - Full support
- ✅ **PostgreSQL** - Full support with dialect-specific optimizations
- ✅ **MySQL** - Full support with dialect-specific optimizations
- ✅ **Other SQLAlchemy-supported databases** - ANSI SQL fallback

## 🧪 Development

### Setup

```bash
# Clone the repository
git clone https://github.com/eddiethedean/moltres.git
cd moltres

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n 9

# Run with coverage
pytest --cov=src/moltres --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
ruff format .

# Type checking (strict mode enabled)
mypy src
```

## 📖 Documentation

Additional documentation is available:

- **[Examples Directory](https://github.com/eddiethedean/moltres/tree/main/examples)** - 14 comprehensive example files covering all features
- **[Examples Guide](https://github.com/eddiethedean/moltres/blob/main/docs/EXAMPLES.md)** - Common patterns and use cases
- **[Why Moltres?](https://github.com/eddiethedean/moltres/blob/main/docs/WHY_MOLTRES.md)** - Understanding the gap Moltres fills
- **[Security Guide](https://github.com/eddiethedean/moltres/blob/main/docs/SECURITY.md)** - Security best practices
- **[Troubleshooting](https://github.com/eddiethedean/moltres/blob/main/docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🤝 Contributing

Contributions are welcome! Please see [`CONTRIBUTING.md`](https://github.com/eddiethedean/moltres/blob/main/CONTRIBUTING.md) for guidelines.

**Quick Start:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Before submitting:**
- Run tests: `pytest`
- Check code quality: `ruff check . && mypy src`
- Update documentation if needed

## 👤 Author

**Odos Matthews**

- GitHub: [@eddiethedean](https://github.com/eddiethedean)
- Email: odosmatthews@gmail.com

## 🙏 Acknowledgments

- Inspired by PySpark's DataFrame API style, but focused on SQL feature support rather than PySpark feature parity
- Built on SQLAlchemy for database connectivity and SQL compilation
- Thanks to all contributors and users

## 📄 License

MIT License - see [LICENSE](https://github.com/eddiethedean/moltres/blob/main/LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the Python data community**

[⬆ Back to Top](#moltres)

</div>
