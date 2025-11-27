# SQLAlchemy Integration Guide

This guide demonstrates how to integrate Moltres with existing SQLAlchemy projects, allowing you to use Moltres DataFrames with your existing SQLAlchemy infrastructure.

## Overview

Moltres provides several ways to integrate with existing SQLAlchemy projects:

1. **Using existing SQLAlchemy objects** - Create Moltres Database instances from existing Engines, Connections, or Sessions
2. **Converting to SQLAlchemy statements** - Convert Moltres DataFrames to SQLAlchemy Select statements
3. **Creating from SQLAlchemy statements** - Create Moltres DataFrames from existing SQLAlchemy Select statements
4. **Executing with existing connections** - Execute Moltres queries using existing SQLAlchemy connections or sessions

## Using Existing SQLAlchemy Objects

### From SQLAlchemy Engine

If you already have a SQLAlchemy Engine, you can create a Moltres Database from it:

```python
from sqlalchemy import create_engine
from moltres import Database

# Your existing engine
engine = create_engine("sqlite:///:memory:")

# Create Moltres Database from engine
db = Database.from_engine(engine)

# Now use Moltres with your existing engine
df = db.table("users").select().where(col("age") > 25)
results = df.collect()
```

### From SQLAlchemy Connection

You can also create a Database from an existing Connection:

```python
from sqlalchemy import create_engine
from moltres import Database

engine = create_engine("sqlite:///:memory:")

with engine.connect() as conn:
    # Create Database from connection
    db = Database.from_connection(conn)
    
    # Use Moltres within the connection's transaction
    df = db.table("users").select()
    results = df.collect()
```

### From SQLAlchemy Session

For ORM-based applications, you can create a Database from a Session:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from moltres import Database

engine = create_engine("sqlite:///:memory:")
Session = sessionmaker(bind=engine)

with Session() as session:
    # Create Database from session
    db = Database.from_session(session)
    
    # Use Moltres with your existing session
    df = db.table("users").select()
    results = df.collect()
```

### Async Versions

For async SQLAlchemy projects, use the async factory methods:

```python
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from moltres import AsyncDatabase

# Async Engine
engine = create_async_engine("sqlite+aiosqlite:///:memory:")
db = AsyncDatabase.from_async_engine(engine)

# Async Connection
async with engine.connect() as conn:
    db = AsyncDatabase.from_async_connection(conn)

# Async Session
AsyncSession = async_sessionmaker(bind=engine)
async with AsyncSession() as session:
    db = AsyncDatabase.from_async_session(session)
```

## Converting DataFrames to SQLAlchemy Statements

### Using `to_sqlalchemy()` Method

You can convert any Moltres DataFrame to a SQLAlchemy Select statement:

```python
from moltres import connect, col

db = connect("sqlite:///:memory:")
df = db.table("users").select().where(col("age") > 25)

# Convert to SQLAlchemy Select statement
stmt = df.to_sqlalchemy()

# Now execute with any SQLAlchemy connection
from sqlalchemy import create_engine
engine = create_engine("sqlite:///:memory:")
with engine.connect() as conn:
    result = conn.execute(stmt)
    rows = result.fetchall()
```

### Using Convenience Function

You can also use the convenience function from the integration module:

```python
from moltres.integration import to_sqlalchemy_select

stmt = to_sqlalchemy_select(df)
```

### With Pandas/Polars Interfaces

The `to_sqlalchemy()` method also works with PandasDataFrame and PolarsDataFrame:

```python
# Pandas interface
df = db.table("users").pandas()
stmt = df.to_sqlalchemy()

# Polars interface
df = db.table("users").polars()
stmt = df.to_sqlalchemy()
```

## Creating DataFrames from SQLAlchemy Statements

You can create Moltres DataFrames from existing SQLAlchemy Select statements:

```python
from sqlalchemy import create_engine, select, table, column
from moltres import DataFrame

# Create a SQLAlchemy Select statement
users = table("users", column("id"), column("name"), column("age"))
sa_stmt = select(users.c.id, users.c.name).where(users.c.age > 25)

# Convert to Moltres DataFrame
df = DataFrame.from_sqlalchemy(sa_stmt)

# Can now chain Moltres operations
df2 = df.select("name").where(col("name") != "Alice")
results = df2.collect()
```

### Using Convenience Function

```python
from moltres.integration import from_sqlalchemy_select

df = from_sqlalchemy_select(sa_stmt)
```

**Note:** When creating a DataFrame from a SQLAlchemy statement, the statement is wrapped as a `RawSQL` logical plan. This means you can chain additional Moltres operations, but the original SQLAlchemy statement structure is preserved as SQL text.

## Executing with Existing Connections

### Using `execute_with_connection()`

Execute a Moltres DataFrame using an existing SQLAlchemy Connection:

```python
from moltres.integration import execute_with_connection
from sqlalchemy import create_engine

engine = create_engine("sqlite:///:memory:")

# Create DataFrame
df = db.table("users").select().where(col("age") > 25)

# Execute with existing connection
with engine.connect() as conn:
    results = execute_with_connection(df, conn)
```

### Using `execute_with_session()`

Execute a Moltres DataFrame using an existing SQLAlchemy Session:

```python
from moltres.integration import execute_with_session
from sqlalchemy.orm import sessionmaker

Session = sessionmaker(bind=engine)

with Session() as session:
    results = execute_with_session(df, session)
```

## Working with Transactions

You can use Moltres within existing SQLAlchemy transactions:

```python
from sqlalchemy import create_engine
from moltres import Database

engine = create_engine("sqlite:///:memory:")
db = Database.from_engine(engine)

# Use Moltres within a transaction
with engine.begin() as conn:
    # Create DataFrame
    df = db.table("users").select().where(col("id") == 1)
    
    # Convert to SQLAlchemy statement and execute within transaction
    stmt = df.to_sqlalchemy()
    result = conn.execute(stmt)
    rows = result.fetchall()
```

## Best Practices

### Connection Lifecycle

When using existing connections or sessions, Moltres does **not** manage their lifecycle. You are responsible for:

- Opening and closing connections
- Managing transaction boundaries
- Handling connection errors

### Transaction Handling

When using existing connections or sessions:

- Moltres respects the transaction state of the connection/session
- Operations execute within the existing transaction context
- No auto-commit is performed (respects your transaction management)

### Dialect Detection

Moltres automatically detects the SQL dialect from:

- The Engine's dialect (when using `from_engine()`)
- The Connection's engine dialect (when using `from_connection()`)
- The Session's bind dialect (when using `from_session()`)

You can override the dialect by passing it explicitly to `to_sqlalchemy()`:

```python
stmt = df.to_sqlalchemy(dialect="postgresql")
```

## Async Integration

All integration features are available for async SQLAlchemy:

```python
from sqlalchemy.ext.asyncio import create_async_engine
from moltres import AsyncDatabase, AsyncDataFrame
from moltres.integration.async_integration import (
    execute_with_async_connection,
    execute_with_async_session,
)

engine = create_async_engine("sqlite+aiosqlite:///:memory:")

# Create async database
db = AsyncDatabase.from_async_engine(engine)

# Convert async DataFrame to SQLAlchemy statement
table_handle = await db.table("users")
df = table_handle.select()
stmt = df.to_sqlalchemy()

# Execute with async connection
async with engine.connect() as conn:
    results = await execute_with_async_connection(df, conn)
```

## Examples

See [`examples/20_sqlalchemy_integration.py`](../examples/20_sqlalchemy_integration.py) for comprehensive examples of all integration patterns.

## Summary

Moltres provides seamless integration with existing SQLAlchemy projects:

- ✅ Use existing Engines, Connections, and Sessions
- ✅ Convert DataFrames to SQLAlchemy statements
- ✅ Create DataFrames from SQLAlchemy statements
- ✅ Execute within existing transactions
- ✅ Full async support
- ✅ Works with Pandas and Polars interfaces

This allows you to gradually adopt Moltres in existing SQLAlchemy projects without requiring a complete rewrite.

