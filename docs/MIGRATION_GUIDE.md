# Migration Guide

This guide helps you migrate to Moltres from other data processing libraries.

## Migrating from Pandas

### Basic Operations

**Pandas:**
```python
import pandas as pd

df = pd.read_csv("data.csv")
filtered = df[df["age"] > 18]
result = filtered.groupby("category").sum()
```

**Moltres:**
```python
from moltres import connect, col

db = connect("sqlite:///data.db")
df = db.read.csv("data.csv")
filtered = df.where(col("age") > 18)
result = filtered.group_by("category").agg(sum(col("amount")))
```

### Key Differences

1. **Lazy Evaluation**: Moltres operations are lazy until `.collect()`
2. **SQL Pushdown**: Operations execute in the database
3. **No In-Memory Data**: Data stays in the database

## Migrating from SQLAlchemy ORM

### Query Building

**SQLAlchemy ORM:**
```python
from sqlalchemy.orm import Session
from models import User

session = Session()
users = session.query(User).filter(User.age > 18).all()
```

**Moltres:**
```python
from moltres import connect, col

db = connect("postgresql://...")
users = db.table("users").select().where(col("age") > 18).collect()
```

### CRUD Operations

**SQLAlchemy ORM:**
```python
# Create
user = User(name="Alice", age=30)
session.add(user)
session.commit()

# Update
user.age = 31
session.commit()

# Delete
session.delete(user)
session.commit()
```

**Moltres:**
```python
# Create
db.createDataFrame([{"name": "Alice", "age": 30}]).write.insertInto("users")

# Update
df = db.table("users").select()
df.write.update("users", where=col("name") == "Alice", set={"age": 31})

# Delete
df.write.delete("users", where=col("name") == "Alice")
```

## Migrating from PySpark

### DataFrame Operations

**PySpark:**
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
df = spark.read.csv("data.csv")
result = df.filter(df.age > 18).groupBy("category").sum("amount")
```

**Moltres:**
```python
from moltres import connect, col

db = connect("postgresql://...")
df = db.read.csv("data.csv")
result = df.where(col("age") > 18).group_by("category").agg(sum(col("amount")))
```

### Key Differences

1. **No Cluster**: Moltres works with existing databases, no cluster needed
2. **Same API**: 98% API compatibility with PySpark
3. **SQL Pushdown**: All operations compile to SQL

## Migrating from Ibis

### Query Building

**Ibis:**
```python
import ibis

con = ibis.postgres.connect(...)
table = con.table("users")
result = table.filter(table.age > 18).group_by("category").aggregate(...)
```

**Moltres:**
```python
from moltres import connect, col

db = connect("postgresql://...")
df = db.table("users").select().where(col("age") > 18)
result = df.group_by("category").agg(...)
```

### Key Differences

1. **DataFrame API**: Moltres uses DataFrame API (like Pandas/PySpark)
2. **CRUD Operations**: Moltres supports INSERT/UPDATE/DELETE
3. **Type Safety**: Full type hints throughout

## Migration Checklist

### Pre-Migration

- [ ] Identify all data sources
- [ ] Map current operations to Moltres equivalents
- [ ] Identify breaking changes
- [ ] Plan migration strategy (big bang vs. gradual)

### Migration Steps

1. **Setup**
   - Install Moltres
   - Configure database connections
   - Test connectivity

2. **Data Migration**
   - Migrate data to target database
   - Verify data integrity
   - Set up indexes

3. **Code Migration**
   - Replace library imports
   - Update API calls
   - Update data access patterns

4. **Testing**
   - Test all operations
   - Verify results match
   - Performance testing

5. **Deployment**
   - Deploy to staging
   - Monitor for issues
   - Deploy to production

### Post-Migration

- [ ] Monitor performance
- [ ] Verify data correctness
- [ ] Update documentation
- [ ] Train team members

## Common Migration Patterns

### Pattern 1: Gradual Migration

1. Keep existing system running
2. Migrate one module at a time
3. Use Moltres for new features
4. Gradually replace old code

### Pattern 2: Big Bang Migration

1. Migrate entire system at once
2. Requires thorough testing
3. Higher risk but faster completion

### Pattern 3: Hybrid Approach

1. Use Moltres for new features
2. Keep existing code as-is
3. Migrate when touching old code

## Troubleshooting

### Common Issues

1. **Performance Differences**
   - Add indexes
   - Optimize queries
   - Use connection pooling

2. **API Differences**
   - Check documentation
   - Use type hints for IDE help
   - Review examples

3. **Data Type Mismatches**
   - Verify schema
   - Check type mappings
   - Use explicit casting

## Getting Help

- Check documentation
- Search GitHub issues
- Ask questions in discussions
- Review examples in docs/

