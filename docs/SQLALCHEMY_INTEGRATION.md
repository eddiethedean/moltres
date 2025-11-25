# SQLAlchemy Models Integration Plan

## Overview
Enable comprehensive bidirectional integration between SQLAlchemy ORM models (declarative models) and Moltres, allowing users to:
- Create tables from SQLAlchemy models
- Query using SQLAlchemy models as table references
- Convert between SQLAlchemy models and Moltres TableSchema/ColumnDef
- Use model-based joins and relationships

## Implementation Components

### 1. Schema Conversion Module (`src/moltres/table/sqlalchemy_integration.py`)

Create a new module for SQLAlchemy integration with the following functions:

**SQLAlchemy → Moltres Conversion:**
- `model_to_schema(model_class: Type[DeclarativeBase]) -> TableSchema`
  - Extract table name from `__tablename__` or model class name
  - Convert SQLAlchemy Column objects to Moltres ColumnDef
  - Extract constraints (primary keys, unique, foreign keys, check)
  - Handle relationships (extract foreign key info)
  - Map SQLAlchemy types to Moltres type names

**Moltres → SQLAlchemy Conversion:**
- `schema_to_table(schema: TableSchema, metadata: MetaData) -> Table`
  - Convert Moltres TableSchema to SQLAlchemy Table object
  - Map Moltres type names to SQLAlchemy types
  - Reconstruct constraints

**Type Mapping:**
- `sqlalchemy_type_to_moltres_type(sa_type: TypeEngine) -> str`
  - Map common SQLAlchemy types (Integer, String, DateTime, Boolean, etc.) to Moltres type names
  - Handle dialect-specific types
  - Support precision/scale for DECIMAL/NUMERIC

- `moltres_type_to_sqlalchemy_type(type_name: str, nullable: bool, precision: Optional[int], scale: Optional[int]) -> TypeEngine`
  - Reverse mapping from Moltres types to SQLAlchemy types

**Helper Functions:**
- `is_sqlalchemy_model(obj: Any) -> bool` - Detect if object is a SQLAlchemy model
- `get_model_table_name(model_class: Type) -> str` - Extract table name from model
- `extract_foreign_keys(model_class: Type) -> List[ForeignKeyConstraint]` - Extract FK info from relationships

### 2. Extended Database API (`src/moltres/table/table.py`)

**Modify `create_table()` method:**
- Add overload to accept SQLAlchemy model class
- Detect if first argument is a model class vs table name
- Convert model to TableSchema internally
- Maintain backward compatibility with existing API

**Modify `table()` method:**
- Accept SQLAlchemy model class in addition to string table name
- Extract table name from model if model is provided
- Return TableHandle with model metadata attached

**Add convenience methods:**
- `create_table_from_model(model_class: Type, **kwargs) -> CreateTableOperation`
- `reflect_model(model_class: Type) -> TableSchema` - Reflect existing model schema

### 3. TableHandle Extensions (`src/moltres/table/table.py`)

**Extend TableHandle:**
- Store optional model class reference
- Add `model` property to access the model class
- Support model-based column references in joins

### 4. Type Mapping Implementation

**Create type mapping dictionary:**
- Map SQLAlchemy core types to Moltres type names:
  - `Integer` → `"INTEGER"`
  - `String(length)` → `"TEXT"` or `"VARCHAR(n)"`
  - `DateTime` → `"TIMESTAMP"`
  - `Date` → `"DATE"`
  - `Boolean` → `"BOOLEAN"` or `"INTEGER"` (SQLite)
  - `Float` → `"REAL"`
  - `Numeric(precision, scale)` → `"DECIMAL"`
  - `Text` → `"TEXT"`
  - `JSON` → `"JSON"` or `"TEXT"` (SQLite)
  - Handle dialect-specific types (PostgreSQL JSONB, etc.)

### 5. Constraint Extraction

**Extract from SQLAlchemy models:**
- Primary keys: From `primary_key=True` or `__table_args__`
- Unique constraints: From `UniqueConstraint` in `__table_args__`
- Foreign keys: From `ForeignKey` column definitions and `relationship()` attributes
- Check constraints: From `CheckConstraint` in `__table_args__`
- Indexes: From `Index` in `__table_args__`

### 6. Testing (`tests/table/test_sqlalchemy_integration.py`)

**Test cases:**
- Test `model_to_schema()` with various model definitions
- Test `schema_to_table()` conversion
- Test type mapping in both directions
- Test constraint extraction (PK, unique, FK, check)
- Test `create_table()` with model class
- Test `table()` with model class
- Test model-based queries and joins
- Test relationship handling
- Test dialect-specific type handling
- Test backward compatibility (existing API still works)

### 7. Documentation Updates

**README.md:**
- Add SQLAlchemy integration section with examples
- Show model-based table creation
- Show model-based querying
- Show conversion examples

**New example file (`examples/17_sqlalchemy_models.py`):**
- Define sample SQLAlchemy models
- Create tables from models
- Query using models
- Convert between models and schemas
- Model-based joins

### 8. Async Support

**AsyncDatabase (`src/moltres/table/async_table.py`):**
- Mirror all SQLAlchemy integration features for async operations
- Support async model-based operations

## Key Files to Modify

1. **New file:** `src/moltres/table/sqlalchemy_integration.py` - Core integration logic
2. **Modify:** `src/moltres/table/table.py` - Extend Database and TableHandle classes
3. **Modify:** `src/moltres/table/async_table.py` - Add async support
4. **Modify:** `src/moltres/__init__.py` - Export integration utilities if needed
5. **New file:** `tests/table/test_sqlalchemy_integration.py` - Comprehensive tests
6. **New file:** `examples/17_sqlalchemy_models.py` - Usage examples
7. **Modify:** `README.md` - Add integration documentation

## Implementation Notes

- Use SQLAlchemy's inspection API (`inspect()`) to extract model metadata
- Handle both Core and ORM SQLAlchemy usage patterns
- Support SQLAlchemy 2.0+ (project already uses SQLAlchemy>=2.0)
- Maintain full backward compatibility with existing Moltres API
- Type hints should use `TYPE_CHECKING` for SQLAlchemy imports to avoid runtime dependencies
- Handle edge cases: models without `__tablename__`, abstract base classes, etc.
- Consider relationship handling for joins (extract FK info from relationships)

## Dependencies

- SQLAlchemy>=2.0 (already a dependency)
- No additional dependencies required

## Example Usage (Post-Implementation)

```python
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(100), unique=True)

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Integer)

# Create tables from models
db.create_table(User).collect()
db.create_table(Order).collect()

# Query using models
df = db.table(User).select().where(col("name") == "Alice")
results = df.collect()

# Model-based joins
df = (
    db.table(Order)
    .select()
    .join(db.table(User), on=[("user_id", "id")])
)
```

## Progress

### Completed

1. ✅ **create_integration_module** - Created `src/moltres/table/sqlalchemy_integration.py` with:
   - `is_sqlalchemy_model()` - Detects SQLAlchemy ORM model classes
   - `get_model_table_name()` - Extracts table name from model
   - `sqlalchemy_type_to_moltres_type()` - Converts SQLAlchemy types to Moltres type names
   - `moltres_type_to_sqlalchemy_type()` - Converts Moltres types to SQLAlchemy TypeEngine
   - `extract_foreign_keys()` - Extracts foreign key constraints from models
   - `model_to_schema()` - Converts SQLAlchemy models to Moltres TableSchema
   - `schema_to_table()` - Converts Moltres TableSchema to SQLAlchemy Table
   - All functions include proper type hints and error handling
   - No linting errors

### In Progress

- None currently

### Pending

2. **extend_database_api** - Extend `Database.create_table()` and `Database.table()` methods to accept SQLAlchemy model classes
3. **extend_table_handle** - Extend TableHandle to store and expose model class references
4. **implement_type_mapping** - Type mapping is implemented in sqlalchemy_integration.py, but may need refinement based on testing
5. **extract_constraints** - Constraint extraction is implemented in sqlalchemy_integration.py, but may need refinement based on testing
6. **async_support** - Add SQLAlchemy model support to AsyncDatabase class
7. **write_tests** - Create comprehensive test suite in `tests/table/test_sqlalchemy_integration.py`
8. **create_examples** - Create `examples/17_sqlalchemy_models.py` with usage examples
9. **update_documentation** - Update README.md with SQLAlchemy integration section and examples

## Implementation Todos

1. ✅ **create_integration_module** - Create `src/moltres/table/sqlalchemy_integration.py` with `model_to_schema()`, `schema_to_table()`, and type mapping functions
2. **extend_database_api** - Extend `Database.create_table()` and `Database.table()` methods to accept SQLAlchemy model classes
3. **extend_table_handle** - Extend TableHandle to store and expose model class references
4. **implement_type_mapping** - Implement comprehensive type mapping between SQLAlchemy types and Moltres type names
5. **extract_constraints** - Implement constraint extraction from SQLAlchemy models (PK, unique, FK, check, indexes)
6. **async_support** - Add SQLAlchemy model support to AsyncDatabase class
7. **write_tests** - Create comprehensive test suite in `tests/table/test_sqlalchemy_integration.py`
8. **create_examples** - Create `examples/17_sqlalchemy_models.py` with usage examples
9. **update_documentation** - Update README.md with SQLAlchemy integration section and examples

