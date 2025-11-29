# Best Practices Guide

Essential best practices for writing maintainable and efficient Moltres code.

## Code Organization

### 1. Separate Connection Logic

```python
# ✅ Good: Centralized connection management
# config.py
from moltres import connect

def get_database():
    return connect("postgresql://user:pass@localhost/mydb")

# main.py
from config import get_database

db = get_database()
df = db.table("users").select()
```

### 2. Use Type Hints

```python
from moltres import connect
from typing import List, Dict
from moltres import Database
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

def get_active_users(db: Database) -> List[Dict[str, object]]:
    df = db.table("users").select().where(col("active") == 1)
    return df.collect()

```

### 3. Organize by Functionality

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
# ✅ Good: Group related operations
class UserService:
    def __init__(self, db: Database):
        self.db = db
    
    def get_active_users(self):
        return self.db.table("users").select().where(col("active") == 1).collect()
    
    def update_user_status(self, user_id: int, status: str):
        return self.db.update("users", where=col("id") == user_id, set={"status": status})

```

## Query Writing

### 1. Use Descriptive Variable Names

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
# ✅ Good
active_users_query = db.table("users").select().where(col("active") == 1)
high_value_orders = db.table("orders").select().where(col("amount") > 1000)

# ❌ Bad
q1 = db.table("users").select().where(col("active") == 1)
q2 = db.table("orders").select().where(col("amount") > 1000)

```

### 2. Build Queries Incrementally

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
# ✅ Good: Clear, readable query building
base_query = db.table("users").select()
active_users = base_query.where(col("active") == 1)
adult_active_users = active_users.where(col("age") >= 18)
results = adult_active_users.collect()

```

### 3. Use Column Aliases

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
# ✅ Good: Clear column names
df = db.table("orders").select(
    col("id").alias("order_id"),
    col("amount").alias("order_amount"),
    (col("amount") * 1.1).alias("amount_with_tax")
)

```

## Error Handling

### 1. Always Handle Errors

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# ✅ Good
def safe_collect(df):
    try:
        results = df.collect()
        return results
    except ExecutionError as e:
        logger.error(f"Query failed: {e}")
        return []

```

### 2. Validate Inputs

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
# ✅ Good
def query_users(db: Database, min_age: int):
    if not isinstance(min_age, int):
        raise ValueError("min_age must be an integer")
    if min_age < 0:
        raise ValueError("min_age must be non-negative")
    
    return db.table("users").select().where(col("age") >= min_age).collect()

```

### 3. Use Context Managers

```python
# ✅ Good: Automatic connection cleanup
with connect("postgresql://...") as db:
    results = db.table("users").select().collect()
```

## Performance

### 1. Filter Early

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
# ✅ Good: Filter before expensive operations
df = (
    db.table("orders")
    .select()
    .where(col("date") >= "2024-01-01")  # Filter early
    .join(db.table("users").select(), on=[...])
)

# ❌ Bad: Filter after join
df = (
    db.table("orders")
    .select()
    .join(db.table("users").select(), on=[...])
    .where(col("date") >= "2024-01-01")  # Filter late
)

```

### 2. Select Only Needed Columns

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
# ✅ Good
df = db.table("users").select("id", "name", "email")

# ❌ Bad
df = db.table("users").select()  # Selects all columns

```

### 3. Use Indexes

```python
from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records

# Use in-memory SQLite for easy setup (no file needed)
db = connect("sqlite:///:memory:")
# ✅ Good: Create indexes on frequently queried columns
db.create_index("idx_user_email", "users", "email").collect()
db.create_index("idx_order_date", "orders", "date").collect()

```

### 4. Avoid N+1 Queries

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
# ✅ Good: Single query with IN clause
user_ids = [1, 2, 3, 4, 5]
df = db.table("orders").select().where(col("user_id").isin(user_ids))
results = df.collect()

# ❌ Bad: Multiple queries
results = []
for user_id in user_ids:
    df = db.table("orders").select().where(col("user_id") == user_id)
    results.append(df.collect())

```

## Security

### 1. Never Use String Formatting for SQL

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
# ❌ Bad: SQL injection risk
user_input = "'; DROP TABLE users; --"
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# ✅ Good: Use parameterized queries
df = db.table("users").select().where(col("name") == user_input)

```

### 2. Validate Table/Column Names

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
# ✅ Good: Moltres validates automatically, but be aware
# Moltres will raise ValidationError for invalid identifiers
try:
    df = db.table("users; DROP TABLE users; --").select()
except ValidationError:
    # Invalid table name caught
    pass

```

### 3. Use Least Privilege

```python
# ✅ Good: Use database user with minimal required permissions
# Don't use root/admin user for application queries
db = connect("postgresql://app_user:pass@localhost/mydb")
```

## Testing

### 1. Use Test Databases

```python
# ✅ Good: Separate test database
import pytest

@pytest.fixture
def test_db():
    db = connect("sqlite:///:memory:")  # In-memory for tests
    yield db
    db.close()
```

### 2. Test Query Logic Separately

```python
# ✅ Good: Test query building separately from execution
def test_user_query_building():
    db = connect("sqlite:///:memory:")
    df = db.table("users").select().where(col("age") > 25)
    
    # Test plan structure
    assert isinstance(df.plan, Filter)
    assert df.plan.predicate.op == "gt"
```

### 3. Use Fixtures for Test Data

```python
# ✅ Good: Reusable test data
@pytest.fixture
def sample_users(db):
    from moltres.io.records import Records
    users = Records.from_list([
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ], database=db)
    users.insert_into("users")
    return users
```

## Documentation

### 1. Document Complex Queries

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
# ✅ Good: Document complex logic
def get_user_revenue_by_country(db: Database) -> List[Dict]:
    """
    Calculate total revenue per user grouped by country.
    
    Returns:
        List of dicts with keys: user_id, country, total_revenue
    """
    df = (
        db.table("orders")
        .select()
        .join(db.table("users").select(), on=[col("orders.user_id") == col("users.id")])
        .group_by("user_id", "country")
        .agg(F.sum(col("amount")).alias("total_revenue"))
    )
    return df.collect()

```

### 2. Use Type Hints

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
# ✅ Good: Type hints improve code clarity
from typing import List, Dict, Optional
from moltres import Database, col

def find_user(db: Database, user_id: int) -> Optional[Dict[str, object]]:
    results = db.table("users").select().where(col("id") == user_id).collect()
    return results[0] if results else None

```

## Configuration

### 1. Use Environment Variables

```python
# ✅ Good: Configuration via environment
import os
from moltres import connect

db = connect(
    os.getenv("DATABASE_URL", "sqlite:///:memory:"),
    echo=os.getenv("MOLTRES_ECHO", "false").lower() == "true"
)
```

### 2. Centralize Configuration

```python
# ✅ Good: Centralized config
# config.py
class DatabaseConfig:
    DSN = "postgresql://user:pass@localhost/mydb"
    POOL_SIZE = 10
    ECHO = False

# main.py
from config import DatabaseConfig
db = connect(
    DatabaseConfig.DSN,
    pool_size=DatabaseConfig.POOL_SIZE,
    echo=DatabaseConfig.ECHO
)
```

## Code Reusability

### 1. Create Query Builders

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
# ✅ Good: Reusable query builders
class UserQueries:
    @staticmethod
    def active_users(db: Database):
        return db.table("users").select().where(col("active") == 1)
    
    @staticmethod
    def users_by_country(db: Database, country: str):
        return db.table("users").select().where(col("country") == country)

```

### 2. Use Helper Functions

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
# ✅ Good: Helper functions for common patterns
def filter_by_date_range(df, start_date: str, end_date: str):
    return df.where(
        (col("date") >= start_date) & (col("date") <= end_date)
    )

# Usage
df = db.table("orders").select()
df_filtered = filter_by_date_range(df, "2024-01-01", "2024-12-31")

```

## Maintenance

### 1. Keep Queries Simple

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
# ✅ Good: Simple, readable queries
df = (
    db.table("users")
    .select("id", "name")
    .where(col("active") == 1)
    .order_by("name")
)

# ❌ Bad: Overly complex one-liner
df = db.table("users").select("id", "name").where(col("active") == 1).order_by("name").join(db.table("orders").select(), on=[col("users.id") == col("orders.user_id")]).group_by("users.id").agg(F.sum(col("orders.amount")))

```

### 2. Refactor Complex Queries

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
# ✅ Good: Break complex queries into steps
# Step 1: Get active users
active_users = db.table("users").select().where(col("active") == 1)

# Step 2: Join with orders
user_orders = active_users.join(
    db.table("orders").select(),
    on=[col("users.id") == col("orders.user_id")]
)

# Step 3: Aggregate
summary = user_orders.group_by("users.id").agg(
    F.sum(col("orders.amount")).alias("total")
)

```

## Next Steps

- **Getting Started**: See [Getting Started Guide](https://moltres.readthedocs.io/en/latest/guides/getting-started.html)
- **Performance**: Read [Performance Optimization Guide](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)
- **Patterns**: Check [Common Patterns Guide](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)

