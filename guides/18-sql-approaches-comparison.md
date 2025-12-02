# SQL Approaches Comparison Guide

This guide demonstrates three different approaches to executing the same complex SQL query in Python, comparing raw SQL with a database driver, SQLAlchemy Core, and Moltres DataFrame API.

## Table of Contents

1. [Overview](#overview)
2. [Database Setup](#database-setup)
3. [Approach 1: Raw SQL](#approach-1-raw-sql)
4. [Approach 2: SQLAlchemy Core](#approach-2-sqlalchemy-core)
5. [Approach 3: Moltres DataFrame API](#approach-3-moltres-dataframe-api)
6. [Side-by-Side Comparison](#side-by-side-comparison)
7. [When to Use Each Approach](#when-to-use-each-approach)
8. [Conclusion](#conclusion)

## Overview

When working with SQL databases in Python, you have several options for executing queries. This guide compares three common approaches using a real-world example: **finding the top 10 customers by total order amount in the last 30 days, including their order count**.

The query we'll implement demonstrates:
- Multiple table joins (customers, orders, order_items)
- Aggregations (SUM, COUNT)
- Filtering (WHERE clauses with date ranges)
- Grouping (GROUP BY)
- Ordering (ORDER BY)
- Column aliasing

Each approach has its strengths and trade-offs, which we'll explore in detail.

## Database Setup

Before we dive into the three approaches, let's set up our database schema and sample data. All three approaches will use the same database structure.

### Schema

We'll create three tables:
- `customers` - Customer information (id, name, email, created_at)
- `orders` - Order records (id, customer_id, order_date, status)
- `order_items` - Order line items (id, order_id, product_id, quantity, price)

### Target Query

The SQL query we want to execute is:

```sql
SELECT 
    c.id,
    c.name,
    c.email,
    SUM(oi.price * oi.quantity) AS total_spent,
    COUNT(DISTINCT o.id) AS order_count
FROM customers c
INNER JOIN orders o ON c.id = o.customer_id
INNER JOIN order_items oi ON o.id = oi.order_id
WHERE o.order_date >= DATE('now', '-30 days')
    AND o.status = 'completed'
GROUP BY c.id, c.name, c.email
ORDER BY total_spent DESC
LIMIT 10
```

This query finds the top 10 customers by total spending in the last 30 days, showing their customer ID, name, email, total amount spent, and number of orders.

## Approach 1: Raw SQL

The most direct approach is to write and execute raw SQL using a database driver like `sqlite3` (for SQLite) or `psycopg2` (for PostgreSQL).

### Pros
- Full control over SQL syntax
- No abstraction layer overhead
- Direct access to database-specific features
- Minimal dependencies

### Cons
- Manual SQL string construction (error-prone)
- Requires careful parameter binding to prevent SQL injection
- No type safety or compile-time checking
- Verbose connection and error handling code
- Database-specific SQL may not be portable

### Implementation

```python
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any

def get_top_customers_raw_sql(db_path: str) -> List[Dict[str, Any]]:
    """
    Execute the query using raw SQL with sqlite3.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        List of dictionaries containing customer data
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    # Enable row factory to get dictionaries instead of tuples
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        
        # Calculate date 30 days ago
        # SQLite uses DATE('now', '-30 days') in the query
        # For parameterized queries, we'd need to calculate in Python
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Construct SQL query with parameter binding for safety
        # Note: SQLite DATE() function is used directly in SQL
        # For other databases, you'd pass the date as a parameter
        query = """
        SELECT 
            c.id,
            c.name,
            c.email,
            SUM(oi.price * oi.quantity) AS total_spent,
            COUNT(DISTINCT o.id) AS order_count
        FROM customers c
        INNER JOIN orders o ON c.id = o.customer_id
        INNER JOIN order_items oi ON o.id = oi.order_id
        WHERE o.order_date >= DATE('now', '-30 days')
            AND o.status = ?
        GROUP BY c.id, c.name, c.email
        ORDER BY total_spent DESC
        LIMIT ?
        """
        
        # Execute query with parameters (prevents SQL injection)
        # Parameters: status='completed', limit=10
        cursor.execute(query, ('completed', 10))
        
        # Fetch all results
        rows = cursor.fetchall()
        
        # Convert Row objects to dictionaries
        results = [dict(row) for row in rows]
        
        return results
        
    except sqlite3.Error as e:
        # Handle database errors
        print(f"Database error: {e}")
        raise
    finally:
        # Always close the connection
        conn.close()

# Usage example
results = get_top_customers_raw_sql("example.db")
for customer in results:
    print(f"{customer['name']}: ${customer['total_spent']:.2f} ({customer['order_count']} orders)")
```

### Key Points

1. **Connection Management**: Manual connection creation and cleanup required
2. **Parameter Binding**: Use `?` placeholders (SQLite) or `%s` (PostgreSQL) to prevent SQL injection
3. **Row Factory**: Set `row_factory` to get dictionaries instead of tuples
4. **Error Handling**: Must manually catch and handle database exceptions
5. **SQL Portability**: SQLite-specific functions like `DATE('now', '-30 days')` won't work on other databases

### Common Pitfalls

- **SQL Injection**: Never use string formatting (`f"SELECT * FROM {table}"`) - always use parameter binding
- **Connection Leaks**: Always close connections in a `finally` block or use context managers
- **Type Conversion**: Database types may not match Python types (e.g., DECIMAL vs float)
- **Date Handling**: Database-specific date functions require different syntax for different databases

## Approach 2: SQLAlchemy Core

SQLAlchemy Core provides a Pythonic way to build SQL queries programmatically while maintaining control over the generated SQL.

### Pros
- Programmatic query building (less error-prone than string concatenation)
- Database abstraction (works with multiple databases)
- Parameter binding handled automatically
- Better error messages than raw SQL
- Can inspect generated SQL

### Cons
- More verbose than raw SQL for simple queries
- Learning curve for SQLAlchemy API
- Still requires understanding of SQL concepts
- Some database-specific features may be harder to express

### Implementation

```python
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, Date, MetaData, select, func
from sqlalchemy.engine import Row
from datetime import datetime, timedelta
from typing import List, Dict, Any

def get_top_customers_sqlalchemy(db_url: str) -> List[Dict[str, Any]]:
    """
    Execute the query using SQLAlchemy Core.
    
    Args:
        db_url: Database connection URL (e.g., 'sqlite:///example.db')
        
    Returns:
        List of dictionaries containing customer data
    """
    # Create engine
    engine = create_engine(db_url)
    
    # Define table metadata
    metadata = MetaData()
    
    # Define tables (reflects the schema)
    customers = Table(
        'customers', metadata,
        Column('id', Integer, primary_key=True),
        Column('name', String),
        Column('email', String),
        Column('created_at', Date)
    )
    
    orders = Table(
        'orders', metadata,
        Column('id', Integer, primary_key=True),
        Column('customer_id', Integer),
        Column('order_date', Date),
        Column('status', String)
    )
    
    order_items = Table(
        'order_items', metadata,
        Column('id', Integer, primary_key=True),
        Column('order_id', Integer),
        Column('product_id', Integer),
        Column('quantity', Integer),
        Column('price', Float)
    )
    
    # Calculate date 30 days ago
    thirty_days_ago = datetime.now() - timedelta(days=30)
    
    # Build query using SQLAlchemy Core
    query = (
        select(
            customers.c.id,
            customers.c.name,
            customers.c.email,
            func.sum(order_items.c.price * order_items.c.quantity).label('total_spent'),
            func.count(func.distinct(orders.c.id)).label('order_count')
        )
        .select_from(
            customers
            .join(orders, customers.c.id == orders.c.customer_id)
            .join(order_items, orders.c.id == order_items.c.order_id)
        )
        .where(orders.c.order_date >= thirty_days_ago)
        .where(orders.c.status == 'completed')
        .group_by(customers.c.id, customers.c.name, customers.c.email)
        .order_by(func.sum(order_items.c.price * order_items.c.quantity).desc())
        .limit(10)
    )
    
    # Execute query
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
        
        # Convert Row objects to dictionaries
        results = [dict(row._mapping) for row in rows]
        
    return results

# Usage example
results = get_top_customers_sqlalchemy("sqlite:///example.db")
for customer in results:
    print(f"{customer['name']}: ${customer['total_spent']:.2f} ({customer['order_count']} orders)")
```

### Key Points

1. **Table Definitions**: Define tables using `Table()` and `Column()` objects
2. **Query Building**: Use `select()` to build queries programmatically
3. **Joins**: Use `.join()` method with join conditions
4. **Aggregations**: Use `func.sum()`, `func.count()`, etc. for aggregate functions
5. **Parameter Binding**: SQLAlchemy handles parameter binding automatically
6. **Database Abstraction**: Same code works with SQLite, PostgreSQL, MySQL, etc.

### Common Pitfalls

- **Table Reflection**: For existing databases, you can use `Table(..., autoload_with=engine)` instead of manual definitions
- **Label vs Alias**: Use `.label()` for column aliases in SELECT, `.alias()` for table aliases
- **Date Handling**: Pass Python `datetime` objects - SQLAlchemy converts them appropriately
- **Connection Management**: Use context managers (`with engine.connect()`) for proper cleanup

## Approach 3: Moltres DataFrame API

Moltres provides a PySpark-like DataFrame API that compiles to SQL, offering a familiar interface for data engineers.

### Pros
- Familiar DataFrame API (similar to PySpark/Pandas)
- Method chaining for readable, composable queries
- Automatic SQL generation and optimization
- Type-safe column expressions
- Built-in connection management
- Works with any SQLAlchemy-supported database

### Cons
- Additional dependency (Moltres library)
- Learning curve if not familiar with DataFrame APIs
- Less control over exact SQL generated (though you can inspect it)

### Implementation

```python
from moltres import connect, col
from moltres.expressions import functions as F
from datetime import datetime, timedelta
from typing import List, Dict, Any

def get_top_customers_moltres(db_url: str) -> List[Dict[str, Any]]:
    """
    Execute the query using Moltres DataFrame API.
    
    Args:
        db_url: Database connection URL (e.g., 'sqlite:///example.db')
        
    Returns:
        List of dictionaries containing customer data
    """
    # Connect to database using context manager (automatically closes on exit)
    with connect(db_url) as db:
        # Calculate date 30 days ago
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        # Build query using DataFrame API
        customers_df = db.table("customers").select()
        orders_df = db.table("orders").select()
        order_items_df = db.table("order_items").select()
        
        # Join tables and build query
        result_df = (
            customers_df
            .join(orders_df, on=[col("customers.id") == col("orders.customer_id")], how="inner")
            .join(order_items_df, on=[col("orders.id") == col("order_items.order_id")], how="inner")
            .where(col("orders.order_date") >= thirty_days_ago)
            .where(col("orders.status") == "completed")
            .group_by("customers.id", "customers.name", "customers.email")
            .agg(
                F.sum(col("order_items.price") * col("order_items.quantity")).alias("total_spent"),
                F.count_distinct(col("orders.id")).alias("order_count")
            )
            .order_by(col("total_spent").desc())
            .limit(10)
        )
        
        # Execute and collect results
        results = result_df.collect()
        
        return results

# Usage example
results = get_top_customers_moltres("sqlite:///example.db")
for customer in results:
    print(f"{customer['name']}: ${customer['total_spent']:.2f} ({customer['order_count']} orders)")
```

### Alternative: More Concise Version

```python
from moltres import connect, col
from moltres.expressions import functions as F
from datetime import datetime, timedelta

def get_top_customers_moltres_concise(db_url: str):
    """More concise version using method chaining."""
    with connect(db_url) as db:
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        return (
            db.table("customers")
            .select()
            .join(
                db.table("orders").select(),
                on=[col("customers.id") == col("orders.customer_id")],
                how="inner"
            )
            .join(
                db.table("order_items").select(),
                on=[col("orders.id") == col("order_items.order_id")],
                how="inner"
            )
            .where(col("orders.order_date") >= thirty_days_ago)
            .where(col("orders.status") == "completed")
            .group_by("customers.id", "customers.name", "customers.email")
            .agg(
                F.sum(col("order_items.price") * col("order_items.quantity")).alias("total_spent"),
                F.count_distinct(col("orders.id")).alias("order_count")
            )
            .order_by(col("total_spent").desc())
            .limit(10)
            .collect()
        )
```

### Key Points

1. **Method Chaining**: Build queries by chaining methods together
2. **Column Expressions**: Use `col()` for column references and operations
3. **Functions**: Use `F.sum()`, `F.count_distinct()`, etc. for aggregations
4. **Automatic SQL**: Moltres generates SQL automatically - you can inspect it with `.to_sql()`
5. **Connection Management**: Use context manager (`with connect()`) for automatic connection cleanup
6. **Type Safety**: Column expressions provide better IDE support and type checking

### Inspecting Generated SQL

You can see the SQL that Moltres generates:

```python
result_df = (
    db.table("customers")
    .select()
    .join(db.table("orders").select(), on=[col("customers.id") == col("orders.customer_id")])
    # ... rest of query
)

# View the generated SQL
print(result_df.to_sql(pretty=True))

# Or get a preview
result_df.sql_preview()
```

### Common Pitfalls

- **Column References**: Use `col("table.column")` format for joined tables to avoid ambiguity
- **Date Handling**: Pass date strings or use `F.to_date()` for date conversions
- **Aggregation Aliases**: Use `.alias()` on aggregate expressions to name result columns
- **Connection Cleanup**: Use context manager (`with connect()`) for automatic cleanup, or manually call `db.close()` if not using context manager

## Side-by-Side Comparison

Here's a comprehensive comparison of the three approaches:

| Feature | Raw SQL | SQLAlchemy Core | Moltres DataFrame API |
|---------|---------|-----------------|----------------------|
| **Lines of Code** | ~40-50 | ~60-70 | ~30-40 |
| **Readability** | Medium (SQL is familiar) | Medium-High (programmatic) | High (method chaining) |
| **Maintainability** | Low (string manipulation) | Medium (structured) | High (composable) |
| **Type Safety** | None | Partial (column types) | High (column expressions) |
| **SQL Injection Protection** | Manual (must use parameters) | Automatic | Automatic |
| **Error Handling** | Manual | Manual | Built-in |
| **Database Portability** | Low (SQL-specific) | High (abstraction layer) | High (SQLAlchemy-based) |
| **Learning Curve** | Low (if you know SQL) | Medium | Medium (if familiar with DataFrames) |
| **IDE Support** | Limited | Good | Excellent (type hints) |
| **Query Inspection** | Direct (it's SQL) | `.compile()` method | `.to_sql()` method |
| **Performance** | Fastest (no abstraction) | Fast (minimal overhead) | Fast (compiles to SQL) |
| **Debugging** | Easy (see exact SQL) | Medium (can inspect SQL) | Easy (can inspect SQL) |
| **Complex Queries** | Verbose | Verbose | Concise |
| **Dependencies** | Database driver only | SQLAlchemy | Moltres (uses SQLAlchemy) |

### Code Complexity Comparison

**Raw SQL** (40 lines):
- Connection management: 5 lines
- SQL string: 15 lines
- Parameter binding: 3 lines
- Execution and error handling: 10 lines
- Result conversion: 5 lines

**SQLAlchemy Core** (65 lines):
- Engine and metadata: 5 lines
- Table definitions: 25 lines
- Query building: 20 lines
- Execution: 10 lines
- Result conversion: 5 lines

**Moltres** (30 lines):
- Connection (context manager): 1 line
- Query building: 25 lines
- Execution: 3 lines
- Connection cleanup: Automatic (context manager handles it)

### Readability Comparison

**Raw SQL**: SQL is familiar to most developers, but string concatenation can be error-prone.

**SQLAlchemy Core**: More verbose, but structured and explicit about what's happening.

**Moltres**: Most readable for those familiar with DataFrame APIs (PySpark, Pandas), with clear method chaining.

## When to Use Each Approach

### Use Raw SQL When:

1. **Simple, one-off queries** that don't need abstraction
2. **Database-specific features** that aren't well-supported by abstractions
3. **Performance-critical code** where every microsecond counts
4. **Legacy codebases** already using raw SQL extensively
5. **SQL expertise** is available and SQL is preferred

**Example scenarios:**
- Quick data exploration scripts
- Database administration tasks
- Complex stored procedures
- Performance tuning with database-specific optimizations

### Use SQLAlchemy Core When:

1. **Existing SQLAlchemy projects** where Core is already in use
2. **Need database portability** but want explicit SQL control
3. **Complex query building** with dynamic conditions
4. **Integration with SQLAlchemy ORM** in the same codebase
5. **Team familiarity** with SQLAlchemy

**Example scenarios:**
- ETL pipelines with multiple database targets
- Applications using SQLAlchemy ORM that need raw queries
- Dynamic query builders based on user input
- Migration scripts that work across databases

### Use Moltres When:

1. **DataFrame-style workflows** are preferred (PySpark/Pandas background)
2. **Rapid development** with less boilerplate
3. **Type safety** and IDE support are important
4. **Composable queries** that are built incrementally
5. **Modern Python projects** starting fresh

**Example scenarios:**
- Data analysis and exploration
- ETL pipelines with complex transformations
- Analytics applications
- Teams migrating from PySpark to SQL
- Projects where readability and maintainability are priorities

### Migration Paths

#### From Raw SQL to SQLAlchemy Core

1. Replace connection code with SQLAlchemy engine
2. Define tables using `Table()` and `Column()`
3. Convert SQL strings to `select()` statements
4. Replace parameter placeholders with SQLAlchemy parameters
5. Update result handling to use SQLAlchemy Row objects

#### From Raw SQL to Moltres

1. Replace connection code with `connect()`
2. Convert SQL to DataFrame operations
3. Replace WHERE clauses with `.where()`
4. Replace JOINs with `.join()`
5. Replace aggregations with `.agg()`

#### From SQLAlchemy Core to Moltres

1. Replace `select()` statements with DataFrame operations
2. Convert table references to `db.table()`
3. Replace SQLAlchemy functions with Moltres `F.*` functions
4. Simplify query building with method chaining

## Conclusion

All three approaches have their place in Python database development:

- **Raw SQL** offers maximum control and performance for simple queries
- **SQLAlchemy Core** provides a good balance of control and abstraction
- **Moltres** offers the most developer-friendly API for complex data workflows

The best choice depends on your team's expertise, project requirements, and long-term maintenance considerations. For new projects focused on data analysis and ETL, Moltres provides an excellent balance of readability, type safety, and productivity.

### Quick Decision Guide

```text
Do you need database-specific SQL features?
├─ Yes → Raw SQL
└─ No → Continue

Is your team already using SQLAlchemy?
├─ Yes → SQLAlchemy Core
└─ No → Continue

Do you prefer DataFrame-style APIs?
├─ Yes → Moltres
└─ No → SQLAlchemy Core
```

### Further Reading

- [Moltres Getting Started Guide](getting-started.md)
- [SQLAlchemy Core Documentation](https://docs.sqlalchemy.org/en/20/core/)
- [Python Database API Specification](https://peps.python.org/pep-0249/)

