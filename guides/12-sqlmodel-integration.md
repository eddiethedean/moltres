# SQLModel and Pydantic Integration Guide

SQLModel is a library that combines SQLAlchemy and Pydantic, providing type-safe database models with automatic validation. This guide shows you how to use Moltres with SQLModel models and pure Pydantic models.

## Overview

When you attach a SQLModel or Pydantic model to a Moltres DataFrame, the `collect()` method will return model instances instead of dictionaries. This provides:

- **Type Safety**: Get full type hints and IDE autocomplete
- **Validation**: Automatic Pydantic validation of data
- **Seamless Integration**: Works with existing SQLModel and Pydantic models
- **Flexibility**: Use SQLModel for database-backed models or Pydantic for validation-only models

## Installation

SQLModel and Pydantic are optional dependencies. Install them with:

```bash
# For SQLModel (includes Pydantic)
pip install sqlmodel

# For pure Pydantic (validation only, no database)
pip install pydantic
```

Or install Moltres with SQLModel support:

```bash
pip install moltres[sqlmodel]
```

**Note**: SQLModel includes Pydantic, so installing SQLModel gives you both. If you only need Pydantic validation without database models, you can install just Pydantic.

## Basic Usage

### Using SQLModel Models

SQLModel models combine SQLAlchemy and Pydantic, providing database-backed models with validation:

```python
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    __tablename__ = "users"
    
    id: int = Field(primary_key=True)
    name: str
    email: str
    age: int
```

### Using Pure Pydantic Models

You can also use pure Pydantic models for validation and type safety without database tables:

```python
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str
    email: str
    age: int
```

**When to use each:**
- **SQLModel**: When you need database-backed models with SQLAlchemy integration
- **Pydantic**: When you only need validation and type safety, without database tables

### Method 1: Using `table()` with SQLModel

The simplest way is to pass the SQLModel class directly to `table()`:

```python
from moltres import connect

db = connect("sqlite:///:memory:")

# Create table (you can use SQLModel's metadata or Moltres schema)
# ... create table ...

# Get DataFrame with model attached
users_table = db.table(User)  # Pass SQLModel class
df = users_table.select()
results = df.collect()  # Returns list of User instances

# Now results are SQLModel instances
for user in results:
    print(user.name)  # Type-safe access
    print(user.email)
```

### Method 2: Using `with_model()`

You can also attach a model to an existing DataFrame. This works with both SQLModel and Pydantic models:

```python
# With SQLModel
df = db.table("users").select()
df_with_model = df.with_model(User)
results = df_with_model.collect()  # Returns list of User instances

# With Pydantic
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str
    email: str

df = db.table("users").select()
df_with_pydantic = df.with_model(UserData)
results = df_with_pydantic.collect()  # Returns list of UserData instances with validation
```

### Method 3: Using Integration Helpers

Use the convenience function from the integration module. This works with both SQLModel and Pydantic:

```python
from moltres.integration import with_sqlmodel

# With SQLModel
df = db.table("users").select()
df_with_model = with_sqlmodel(df, User)
results = df_with_model.collect()

# With Pydantic
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str

df_with_pydantic = with_sqlmodel(df, UserData)
results = df_with_pydantic.collect()
```

## Chaining Operations

You can chain Moltres operations with SQLModel models attached:

```python
from moltres import col

df = (
    db.table(User)
    .select()
    .where(col("age") > 25)
    .order_by("age")
)
results = df.collect()  # Returns list of User instances

for user in results:
    print(f"{user.name} is {user.age} years old")
```

## Streaming with SQLModel

Streaming also works with SQLModel models:

```python
df = db.table(User).select()
stream_results = df.collect(stream=True)

for chunk in stream_results:
    for user in chunk:  # Each user is a SQLModel instance
        print(user.name)
```

## Integration with SQLAlchemy

You can use SQLModel with existing SQLAlchemy infrastructure:

### Using with SQLAlchemy Sessions

```python
from sqlalchemy.orm import sessionmaker
from moltres.integration import execute_with_session_model

engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(bind=engine)

# Create table using SQLModel
User.__table__.create(engine, checkfirst=True)

# Use Moltres with existing session
db = connect(engine=engine)
df = db.table("users").select()

with SessionLocal() as session:
    results = execute_with_session_model(df, session, User)
    # results is a list of User instances
```

### Using with SQLAlchemy Connections

```python
from moltres.integration import execute_with_connection_model

db = connect(engine=engine)
df = db.table("users").select()

with engine.connect() as conn:
    results = execute_with_connection_model(df, conn, User)
    # results is a list of User instances
```

## Async Support

SQLModel integration also works with async DataFrames:

```python
from moltres import async_connect
from moltres.integration.async_integration import with_sqlmodel_async

async def example():
    db = await async_connect("sqlite+aiosqlite:///:memory:")
    table_handle = await db.table(User)
    df = table_handle.select()
    df_with_model = df.with_model(User)
    results = await df_with_model.collect()  # Returns list of User instances
```

## Best Practices

### 1. Use Models for Type Safety

Always attach SQLModel or Pydantic models when you need type-safe access to results:

```python
# Good: Type-safe with SQLModel
df = db.table(User).select()
results = df.collect()  # List[User]
user = results[0]
print(user.name)  # IDE autocomplete works

# Good: Type-safe with Pydantic
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str

df = db.table("users").select()
df_with_model = df.with_model(UserData)
results = df_with_model.collect()  # List[UserData]
user = results[0]
print(user.name)  # IDE autocomplete works, Pydantic validation applied

# Less ideal: Dictionary access
df = db.table("users").select()
results = df.collect()  # List[Dict[str, Any]]
user = results[0]
print(user["name"])  # No type hints
```

### 2. Reuse Models Across Operations

Attach the model once and reuse it:

```python
# Good: Model attached at table level
users_table = db.table(User)
df1 = users_table.select().where(col("age") > 25)
df2 = users_table.select().where(col("age") < 30)

# Both df1 and df2 will return User instances when collected
```

### 3. Choose the Right Model Type

- **Use SQLModel** when you need database-backed models with SQLAlchemy integration
- **Use Pydantic** when you only need validation and type safety without database tables

```python
# SQLModel: Database-backed with validation
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int = Field(primary_key=True)
    name: str

# Pydantic: Validation and type safety only
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str
    # No database table needed
```

### 4. Handle Optional Dependencies

If SQLModel or Pydantic might not be installed, handle it gracefully:

```python
try:
    from sqlmodel import SQLModel, Field
    
    class User(SQLModel, table=True):
        # ...
        pass
    
    df = db.table(User).select()
except ImportError:
    try:
        from pydantic import BaseModel
        
        class UserData(BaseModel):
            # ...
            pass
        
        df = db.table("users").select().with_model(UserData)
    except ImportError:
        # Fallback to dictionary-based approach
        df = db.table("users").select()
```

## Common Patterns

### Filtering and Sorting

```python
df = (
    db.table(User)
    .select()
    .where(col("age") > 25)
    .where(col("email").like("%@example.com"))
    .order_by("age")
    .limit(10)
)
results = df.collect()
```

### Aggregations

```python
from moltres import col, func

df = (
    db.table(User)
    .select(
        func.count(col("id")).alias("total"),
        func.avg(col("age")).alias("avg_age"),
    )
    .group_by(col("email"))
)
results = df.collect()  # Returns dictionaries (aggregations don't map to models)
```

### Joins

```python
class Post(SQLModel, table=True):
    __tablename__ = "posts"
    id: int = Field(primary_key=True)
    user_id: int
    title: str
    content: str

# Join users and posts
df = (
    db.table(User)
    .join(db.table("posts"), col("users.id") == col("posts.user_id"))
    .select()
)
# Note: Joins return dictionaries, not model instances
results = df.collect()
```

## Limitations

1. **Joins and Aggregations**: When you perform joins or aggregations, the result may not map directly to a single model. In these cases, `collect()` will return dictionaries.

2. **Custom Selections**: If you select specific columns that don't match the model structure, you'll get dictionaries.

3. **Optional Dependencies**: SQLModel or Pydantic must be installed for this feature to work. The code will raise an `ImportError` if neither is available.

4. **Pydantic Models and `table()`**: Pure Pydantic models don't have database tables, so you can't use `db.table(PydanticModel)`. Use `with_model()` instead:

```python
# SQLModel: Can use with table()
df = db.table(User).select()  # ✅ Works

# Pydantic: Must use with_model()
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str

df = db.table("users").select().with_model(UserData)  # ✅ Works
df = db.table(UserData).select()  # ❌ Won't work - no table name
```

## Troubleshooting

### ImportError: SQLModel not installed

If you see this error, install SQLModel:

```bash
pip install sqlmodel
```

### TypeError: Expected SQLModel or Pydantic class

Make sure you're passing a model class, not an instance:

```python
# Wrong
user = User(id=1, name="Alice")
df = df.with_model(user)  # TypeError

# Correct - SQLModel
df = df.with_model(User)  # Pass the class

# Correct - Pydantic
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str

df = df.with_model(UserData)  # Pass the class
```

### Results are dictionaries instead of models

Check that:
1. The model is properly attached to the DataFrame
2. The selected columns match the model structure
3. You're not performing operations that break the model mapping (joins, aggregations, etc.)

## See Also

- [SQLAlchemy Integration Guide](./11-sqlalchemy-integration.md) - For SQLAlchemy-specific integration
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/) - Official SQLModel documentation
- [Pydantic Documentation](https://docs.pydantic.dev/) - Official Pydantic documentation
- [Examples](../examples/21_sqlmodel_integration.py) - Complete working examples

