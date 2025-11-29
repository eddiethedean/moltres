# Streamlit Integration Guide

This guide demonstrates how to use Moltres with Streamlit to build interactive data applications. The Streamlit integration provides utilities for displaying DataFrames, building queries interactively, caching results, managing database connections, and visualizing queries.

## Installation

Install Moltres with Streamlit support:

```bash
pip install moltres[streamlit]
```

Or install Streamlit separately:

```bash
pip install moltres streamlit
```

## Quick Start

Here's a simple example to get started:

```python
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import moltres_dataframe, get_db_from_session

# Get database connection from session state
db = get_db_from_session()

# Create a query
df = db.table("users").select()

# Display the DataFrame with query information
moltres_dataframe(df, show_query_info=True)
```

## Features

### 1. DataFrame Display

The `moltres_dataframe()` function displays Moltres DataFrames in Streamlit with optional query information.

#### Basic Usage

```python
from moltres.integrations.streamlit import moltres_dataframe

df = db.table("users").select()
moltres_dataframe(df)
```

#### With Query Information

```python
moltres_dataframe(df, show_query_info=True)
```

This displays:
- The SQL query
- Row count
- Additional metadata

#### Custom Display Options

```python
moltres_dataframe(
    df,
    show_query_info=True,
    height=400,
    use_container_width=True,
    hide_index=True
)
```

All `st.dataframe()` keyword arguments are supported.

### 2. Interactive Query Builder

The `query_builder()` widget provides an interactive UI for building queries.

```python
from moltres.integrations.streamlit import query_builder

df = query_builder(db)
if df:
    results = df.collect()
    st.dataframe(results)
```

**Note**: The query builder is a basic implementation. For complex queries, use the DataFrame API directly.

### 3. Query Result Caching

Use the `@cached_query` decorator to cache query results, improving performance for repeated queries.

#### Basic Caching

```python
from moltres.integrations.streamlit import cached_query

@cached_query()
def get_users():
    return db.table("users").select().collect()
```

#### With TTL (Time-To-Live)

```python
@cached_query(ttl=3600)  # Cache for 1 hour
def get_user_stats():
    return db.table("users").select().agg(...).collect()
```

#### With Max Entries

```python
@cached_query(ttl=3600, max_entries=10)  # Cache max 10 entries
def get_user_stats():
    return db.table("users").select().agg(...).collect()
```

**Important**: The decorator automatically materializes DataFrames (calls `collect()`) before caching, as DataFrames themselves cannot be cached.

#### Cache Management

```python
from moltres.integrations.streamlit import clear_moltres_cache, invalidate_query_cache

# Clear all caches
if st.button("Clear Cache"):
    clear_moltres_cache()
    st.success("Cache cleared!")

# Invalidate specific query (clears all caches)
invalidate_query_cache("SELECT * FROM users")
```

**Note**: Streamlit's `cache_data` doesn't support selective invalidation by key, so `invalidate_query_cache()` clears all caches.

### 4. Session State Management

Streamlit's session state is used to manage database connections across reruns.

#### Automatic Connection Management

```python
from moltres.integrations.streamlit import get_db_from_session

# Automatically manages connection in session state
db = get_db_from_session()
```

This function:
- Creates a connection if it doesn't exist
- Reuses existing connection across reruns
- Uses Streamlit secrets for connection string if available
- Falls back to SQLite in-memory if no DSN configured

#### Using Streamlit Secrets

Create a `.streamlit/secrets.toml` file:

```toml
[moltres]
dsn = "postgresql://user:password@localhost/dbname"
```

The `get_db_from_session()` function will automatically use this DSN.

#### Manual Connection Management

```python
from moltres.integrations.streamlit import init_db_connection, close_db_connection

# Initialize connection
db = init_db_connection("sqlite:///example.db", key="my_db")

# Use the connection
df = db.table("users").select()

# Close connection when done
close_db_connection(key="my_db")
```

#### Multiple Database Connections

```python
# Primary database
db1 = get_db_from_session(key="primary_db")

# Secondary database
db2 = get_db_from_session(key="secondary_db")
```

### 5. Query Visualization

The `visualize_query()` function displays query information including SQL, execution plan, and performance metrics.

#### Basic Visualization

```python
from moltres.integrations.streamlit import visualize_query

df = db.table("users").select().where(col("age") > 25)
visualize_query(df)
```

#### With All Options

```python
visualize_query(
    df,
    show_sql=True,      # Display SQL query
    show_plan=True,     # Display query execution plan
    show_metrics=True   # Display performance metrics
)
```

This displays:
- **SQL Query**: The generated SQL with syntax highlighting
- **Query Plan**: Execution plan from `EXPLAIN`
- **Performance Metrics**: Execution time and row count

### 6. Error Handling

The `display_moltres_error()` function formats Moltres exceptions for display in Streamlit.

```python
from moltres.integrations.streamlit import display_moltres_error

try:
    df = db.table("nonexistent").select()
    df.collect()
except Exception as e:
    display_moltres_error(e)
```

This displays:
- Error message
- Suggestions (if available)
- Error context (if available)

## Complete Example

Here's a complete example combining all features:

```python
import streamlit as st
from moltres import connect, col
from moltres.integrations.streamlit import (
    moltres_dataframe,
    query_builder,
    cached_query,
    get_db_from_session,
    visualize_query,
    display_moltres_error,
)

# Page configuration
st.set_page_config(page_title="Moltres Dashboard", layout="wide")

st.title("Data Analysis Dashboard")

# Get database connection
try:
    db = get_db_from_session()
except Exception as e:
    display_moltres_error(e)
    st.stop()

# Sidebar for filters
st.sidebar.header("Filters")
min_age = st.sidebar.slider("Minimum Age", 0, 100, 18)

# Create query
df = db.table("users").select().where(col("age") >= min_age)

# Visualize query
with st.expander("Query Details"):
    visualize_query(df, show_sql=True, show_plan=True, show_metrics=True)

# Display results with caching
@cached_query(ttl=300)  # Cache for 5 minutes
def get_filtered_users(min_age):
    return db.table("users").select().where(col("age") >= min_age).collect()

results = get_filtered_users(min_age)

# Display DataFrame
moltres_dataframe(
    db.table("users").select().where(col("age") >= min_age),
    show_query_info=True,
    height=400
)

# Statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Users", len(results))
with col2:
    avg_age = sum(r.get("age", 0) for r in results) / len(results) if results else 0
    st.metric("Average Age", f"{avg_age:.1f}")
with col3:
    cities = len(set(r.get("city", "") for r in results))
    st.metric("Unique Cities", cities)
```

## Best Practices

### 1. Connection Management

- Use `get_db_from_session()` for automatic connection management
- Close connections explicitly when switching databases
- Use Streamlit secrets for sensitive connection strings

### 2. Caching

- Cache expensive queries with appropriate TTL
- Remember that DataFrames are automatically materialized before caching
- Clear caches when data changes

### 3. Error Handling

- Always wrap database operations in try/except blocks
- Use `display_moltres_error()` for user-friendly error messages
- Provide fallback UI when errors occur

### 4. Performance

- Use `visualize_query()` to understand query performance
- Enable metrics to track execution times
- Cache frequently accessed data

### 5. Async DataFrames

Async DataFrames must be collected before use with Streamlit components:

```python
async def example():
    # ❌ Don't do this
    async_df = await db.table("users").select()
    moltres_dataframe(async_df)  # Won't work

    # ✅ Do this instead
    async_df = await db.table("users").select()
    results = await async_df.collect()
    st.dataframe(results)
```

## Troubleshooting

### Import Errors

If you get `ImportError: Streamlit is required`, install Streamlit:

```bash
pip install streamlit
```

### Connection Issues

- Check that your DSN is correct
- Verify database credentials in Streamlit secrets
- Ensure the database is accessible

### Caching Issues

- Remember that caches persist across reruns
- Use `clear_moltres_cache()` to reset caches
- Check TTL settings if data seems stale

### Performance Issues

- Use `visualize_query()` to identify slow queries
- Enable caching for expensive operations
- Consider using `limit()` for large result sets

## API Reference

### `moltres_dataframe(df, show_query_info=True, **kwargs)`

Display a Moltres DataFrame in Streamlit.

**Parameters:**
- `df`: Moltres DataFrame to display
- `show_query_info`: If True, display query SQL and row count
- `**kwargs`: Additional arguments passed to `st.dataframe()`

### `query_builder(db)`

Interactive query builder widget.

**Parameters:**
- `db`: Moltres Database instance

**Returns:**
- DataFrame if query was built, None otherwise

### `cached_query(ttl=None, max_entries=None)`

Decorator for caching query results.

**Parameters:**
- `ttl`: Time-to-live in seconds (None = never expires)
- `max_entries`: Maximum cache entries (None = no limit)

### `get_db_from_session(key="db")`

Get or create Database instance from session state.

**Parameters:**
- `key`: Session state key for the connection

**Returns:**
- Database instance

### `init_db_connection(dsn, key="db")`

Initialize database connection in session state.

**Parameters:**
- `dsn`: Database connection string
- `key`: Session state key for the connection

**Returns:**
- Database instance

### `close_db_connection(key="db")`

Close and remove database connection from session state.

**Parameters:**
- `key`: Session state key for the connection

### `visualize_query(df, show_sql=True, show_plan=True, show_metrics=False)`

Visualize query with SQL, plan, and metrics.

**Parameters:**
- `df`: Moltres DataFrame to visualize
- `show_sql`: Display SQL query
- `show_plan`: Display query execution plan
- `show_metrics`: Display performance metrics

### `display_moltres_error(error)`

Display Moltres error in Streamlit-friendly format.

**Parameters:**
- `error`: Exception to display

### `clear_moltres_cache()`

Clear all Moltres-related caches.

### `invalidate_query_cache(query_sql)`

Invalidate cache for a specific query (clears all caches).

**Parameters:**
- `query_sql`: SQL query string

## See Also

- [Moltres Documentation](https://moltres.readthedocs.io/en/latest/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Example: Streamlit Integration](https://moltres.readthedocs.io/en/latest/EXAMPLES.html)

