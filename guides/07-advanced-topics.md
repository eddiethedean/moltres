# Advanced Topics

Advanced features and techniques for power users.

## Table of Contents

1. [Async Operations](#async-operations)
2. [Streaming Large Datasets](#streaming-large-datasets)
3. [Custom SQL Functions](#custom-sql-functions)
4. [Transactions](#transactions)
5. [Schema Management](#schema-management)
6. [SQLAlchemy Integration](#sqlalchemy-integration)
7. [Window Functions](#window-functions)
8. [CTEs and Subqueries](#ctes-and-subqueries)

## Async Operations

Moltres supports full async/await for all operations.

**See also:** [Async DataFrame examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Basic Async Usage

```python
import asyncio
from moltres import async_connect, col

async def main():
    # Connect asynchronously
    db = await async_connect("postgresql+asyncpg://user:pass@localhost/mydb")
    
    # Async table operations
    table_handle = await db.table("users")
    df = table_handle.select().where(col("age") > 25)
    
    # Async collect
    results = await df.collect()
    print(results)
    
    # Close connection
    await db.close()

# Run async function
asyncio.run(main())
```

### Async Streaming

```python
async def process_large_dataset():
    db = await async_connect("postgresql+asyncpg://...")
    df = db.table("large_table").select()
    
    async for chunk in await df.collect(stream=True):
        # Process each chunk
        process_chunk(chunk)
    
    await db.close()
```

### Async CRUD Operations

```python
async def update_users():
    db = await async_connect("postgresql+asyncpg://...")
    
    # Async update
    result = await db.update(
        "users",
        where=col("status") == "pending",
        set={"status": "active"}
    )
    
    # Async insert
    from moltres.io.records import AsyncRecords
    records = AsyncRecords(
        _data=[{"name": "Alice", "email": "alice@example.com"}],
        _database=db
    )
    result = await records.insert_into("users")
    
    await db.close()
```

## Streaming Large Datasets

Process datasets larger than memory using streaming.

**See also:** [Async DataFrame examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html) and [File reading examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Streaming Reads

```python
# Sync streaming
df = db.table("large_table").select()
for chunk in df.collect(stream=True):
    process_chunk(chunk)

# Async streaming
async def stream_data():
    db = await async_connect("postgresql+asyncpg://...")
    df = db.table("large_table").select()
    
    async for chunk in await df.collect(stream=True):
        await process_chunk(chunk)
    
    await db.close()
```

### Streaming Writes

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
# Stream DataFrame to file
df = db.table("large_table").select()
df.write.stream().csv("output.csv", mode="overwrite")

# Stream with custom chunk size
df.write.stream(chunk_size=5000).parquet("output.parquet")

```

### Chunked Processing

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
def process_in_chunks(db, table_name, chunk_size=1000):
    """Process table in chunks using row numbers."""
    from moltres import col
    from moltres.expressions import functions as F
    from moltres.expressions.window import Window
    
    # Add row numbers for pagination
    df_with_row_num = (
        db.table(table_name)
        .select()
        .withColumn(
            "row_num",
            F.row_number().over(Window.order_by("id"))  # Assuming id column exists
        )
    )
    
    chunk_num = 1
    while True:
        # Get chunk using row number range
        chunk_df = (
            df_with_row_num
            .where(
                (col("row_num") > (chunk_num - 1) * chunk_size) &
                (col("row_num") <= chunk_num * chunk_size)
            )
            .select("*")  # Exclude row_num from results
        )
        results = chunk_df.collect()
        
        if not results:
            break
        
        process_chunk(results)
        chunk_num += 1

```

## Custom SQL Functions

Use database-specific functions or create custom functions.

### Using Database Functions

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

# PostgreSQL JSONB functions
df = db.table("users").select(
    F.func("jsonb_extract_path_text", col("metadata"), "key").alias("value")
)

# MySQL date functions
df = db.table("orders").select(
    F.func("DATE_FORMAT", col("created_at"), "%Y-%m-%d").alias("date")
)

```

### Creating Database Functions

```python
from moltres import connect
from sqlalchemy import text

# Create PostgreSQL function using SQLAlchemy engine directly
# Note: This example requires PostgreSQL. For SQLite, use a simpler approach.
db = connect("sqlite:///:memory:")

# For PostgreSQL, you would use:
# db = connect("postgresql://user:pass@localhost/dbname")

with db.connection_manager.engine.connect() as conn:
    conn.execute(text("""
        CREATE OR REPLACE FUNCTION my_custom_function(value TEXT)
        RETURNS TEXT AS $$
        BEGIN
            RETURN UPPER(value);
        END;
        $$ LANGUAGE plpgsql;
    """))
    conn.commit()

# Use in Moltres
from moltres import col
from moltres.expressions import functions as F

df = db.table("users").select(
    F.func("my_custom_function", col("name")).alias("upper_name")
)
```

## Transactions

Ensure data consistency with transactions.

**See also:** [Transaction examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Basic Transactions

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Sync transaction
with db.transaction() as txn:
    db.insert("users", [{"name": "Alice"}])
    db.insert("orders", [{"user_id": 1, "amount": 100}])
    # Both inserts are committed together, or both rolled back on error

# Async transaction
async def transaction_example():
    async with db.transaction() as txn:
        await db.insert("users", [{"name": "Alice"}])
        await db.insert("orders", [{"user_id": 1, "amount": 100}])

```

### Manual Transaction Control

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Begin transaction
txn = db.begin_transaction()

try:
    db.insert("users", [{"name": "Alice"}])
    db.insert("orders", [{"user_id": 1, "amount": 100}])
    txn.commit()
except Exception as e:
    txn.rollback()
    raise

```

### Nested Transactions

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Outer transaction
with db.transaction() as outer:
    db.insert("users", [{"name": "Alice"}])
    
    # Inner transaction (savepoint)
    with db.transaction() as inner:
        db.insert("orders", [{"user_id": 1, "amount": 100}])
        # Can rollback inner without affecting outer

```

## Schema Management

Programmatically manage database schemas.

**See also:** [Schema reflection examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Reflecting Existing Schemas

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# Reflect single table
schema = db.reflect_table("users")
print(f"Table: {schema.name}")
for col_def in schema.columns:
    print(f"  {col_def.name}: {col_def.type_name}")

# Reflect entire database
all_schemas = db.reflect()
for table_name, schema in all_schemas.items():
    print(f"{table_name}: {len(schema.columns)} columns")

```

### Schema Comparison

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
def compare_schemas(db, table1, table2):
    schema1 = db.reflect_table(table1)
    schema2 = db.reflect_table(table2)
    
    cols1 = {c.name: c.type_name for c in schema1.columns}
    cols2 = {c.name: c.type_name for c in schema2.columns}
    
    # Find differences
    only_in_1 = set(cols1.keys()) - set(cols2.keys())
    only_in_2 = set(cols2.keys()) - set(cols1.keys())
    different_types = {
        col: (cols1[col], cols2[col])
        for col in set(cols1.keys()) & set(cols2.keys())
        if cols1[col] != cols2[col]
    }
    
    return {
        "only_in_1": only_in_1,
        "only_in_2": only_in_2,
        "different_types": different_types
    }

```

### Schema Migration

```python
from moltres import connect
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
def migrate_schema(db, table_name, new_columns):
    """Add new columns to existing table."""
    existing = db.get_columns(table_name)
    existing_names = {c.name for c in existing}
    
    for col_def in new_columns:
        if col_def.name not in existing_names:
            # Add column
            db.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_def.name} {col_def.type_name}")

```

## SQLAlchemy Integration

Integrate with existing SQLAlchemy code.

**See also:** [SQLAlchemy model integration examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Using SQLAlchemy Models

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase
from moltres import connect, col
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(100))

# Create table from model
db = connect("postgresql://...")
db.create_table(User).collect()

# Query using model
df = db.table(User).select().where(col("name") == "Alice")
results = df.collect()

```

### Using SQLAlchemy Engine

```python
from sqlalchemy import create_engine
from moltres import connect

# Use existing SQLAlchemy engine
engine = create_engine("postgresql://...")
db = connect(engine=engine)

# Now use Moltres API with your existing engine
df = db.table("users").select()
results = df.collect()
```

### Mixing SQLAlchemy and Moltres

```python
from sqlalchemy import text

# Use raw SQLAlchemy for complex queries
with db._engine.connect() as conn:
    result = conn.execute(text("SELECT * FROM complex_view"))
    data = result.fetchall()

# Use Moltres for DataFrame operations
df = db.table("users").select().where(col("age") > 25)
results = df.collect()
```

## Window Functions

Advanced analytical queries with window functions.

**See also:** [Window function examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Ranking

```python
from moltres import connect
from moltres.expressions import functions as F
from moltres.expressions.window import Window
from moltres.table.schema import column

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")

# Create sample table
db.create_table("users", [
    column("id", "INTEGER", primary_key=True),
    column("name", "TEXT"),
]).collect()

# Rank products by revenue
df = (
    db.table("sales")
    .select()
    .group_by("product_id")
    .agg(F.sum(col("amount")).alias("revenue"))
    .withColumn(
        "rank",
        F.rank().over(Window.order_by(col("revenue").desc()))
    )
)

```

### Moving Averages

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
# 7-day moving average
df = (
    db.table("daily_sales")
    .select(
        col("date"),
        col("revenue"),
        F.avg(col("revenue")).over(
            Window.order_by("date").rows_between(-6, 0)
        ).alias("ma_7day")
    )
)

```

### Partitioned Windows

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
# Rank within each category
df = (
    db.table("products")
    .select()
    .withColumn(
        "category_rank",
        F.rank().over(
            Window.partition_by("category").order_by(col("price").desc())
        )
    )
)

```

### Cumulative Sums

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
# Running total
df = (
    db.table("transactions")
    .select(
        col("date"),
        col("amount"),
        F.sum(col("amount")).over(
            Window.order_by("date").rows_between(None, 0)
        ).alias("running_total")
    )
)

```

## CTEs and Subqueries

Use Common Table Expressions for complex queries.

**See also:** [SQL operations and CTE examples](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

### Simple CTE

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
# Using CTE
df = (
    db.table("users")
    .select()
    .where(col("age") > 25)
    .cte("adult_users")
)

# Use CTE in another query
result = (
    db.table("orders")
    .select()
    .join(df, on=[col("orders.user_id") == col("adult_users.id")])
)

```

### Recursive CTE

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
# Recursive CTE for hierarchical data
initial = (
    db.table("employees")
    .select(col("id"), col("name"), col("manager_id"))
    .where(col("manager_id").is_null())
)

recursive = (
    db.table("employees")
    .select(col("id"), col("name"), col("manager_id"))
    .join(
        initial,
        on=[col("employees.manager_id") == col("initial.id")]
    )
)

# Create recursive CTE
hierarchy = db.recursive_cte("employee_hierarchy", initial, recursive)
results = hierarchy.collect()

```

### Subqueries

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
# Subquery in WHERE clause
subquery = (
    db.table("orders")
    .select(col("user_id"))
    .group_by("user_id")
    .agg(F.sum(col("amount")).alias("total"))
    .where(col("total") > 1000)
)

df = (
    db.table("users")
    .select()
    .where(col("id").isin(subquery.select("user_id")))
)

```

## Performance Tuning

### Query Hints

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# PostgreSQL query hints (via raw SQL)
df = db.sql("""
    SELECT /*+ INDEX(users idx_user_email) */ *
    FROM users
    WHERE email = :email
""", email="alice@example.com")

```

### Connection Pooling

```python
# Optimize connection pool
db = connect(
    "postgresql://...",
    pool_size=20,        # Larger pool for high concurrency
    max_overflow=40,      # Allow more overflow connections
    pool_timeout=30,      # Timeout for getting connection
    pool_recycle=3600,    # Recycle connections after 1 hour
    pool_pre_ping=True   # Verify connections before use
)
```

### Batch Operations

```python
# Batch inserts for better performance
from moltres.io.records import Records

# Large batch insert (automatically batched)
large_list = [{"name": f"User{i}", "email": f"user{i}@example.com"} for i in range(10000)]
records = Records.from_list(large_list, database=db)
result = records.insert_into("users")  # Efficiently batched
```

## Next Steps

- **Performance**: See [Performance Optimization Guide](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)
- **Patterns**: Check [Common Patterns Guide](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)
- **Best Practices**: Read [Best Practices Guide](https://moltres.readthedocs.io/en/latest/guides/best-practices.html)

