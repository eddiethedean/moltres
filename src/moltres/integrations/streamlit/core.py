"""Streamlit integration utilities for Moltres.

This module provides helper functions and utilities to make Moltres
more user-friendly and robust when used with Streamlit applications.

Key features:
- :class:`DataFrame` display components with query information
- Interactive query builder widget
- Caching utilities for query results
- Session state helpers for database connections
- Query visualization components
- Error handling helpers
"""

from __future__ import annotations

import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    import streamlit as st

    from ...dataframe.core.async_dataframe import AsyncDataFrame
    from ...dataframe.core.dataframe import DataFrame
    from ...table.table import Database

try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create stubs for type checking
    st = None


def _check_streamlit_available() -> None:
    """Check if Streamlit is available, raise ImportError if not."""
    if not STREAMLIT_AVAILABLE:
        raise ImportError(
            "Streamlit is required for Streamlit integration. Install with: pip install streamlit"
        )


def moltres_dataframe(
    df: "DataFrame | AsyncDataFrame",
    show_query_info: bool = True,
    **kwargs: Any,
) -> None:
    """Display a Moltres :class:`DataFrame` in Streamlit with optional query information.

    This function automatically collects the :class:`DataFrame` results and displays them
    using Streamlit's dataframe component, with optional query information display.

    Args:
        df: Moltres :class:`DataFrame` (sync or async) to display
        show_query_info: If True, display query SQL and row count information
        **kwargs: Additional arguments passed to st.dataframe() (height, width, use_container_width, etc.)

    Example:
        >>> import streamlit as st
        >>> from moltres import connect
        >>> from moltres.integrations.streamlit import moltres_dataframe
        >>>
        >>> db = connect("sqlite:///example.db")
        >>> df = db.table("users").select()
        >>> moltres_dataframe(df, show_query_info=True)
    """
    _check_streamlit_available()

    # Handle async DataFrames
    if hasattr(df, "__class__") and "Async" in df.__class__.__name__:
        st.warning(
            "Async DataFrames must be collected before display. Use await df.collect() first."
        )
        return

    try:
        # Collect DataFrame results
        results = df.collect()

        # Convert to format suitable for st.dataframe()
        # st.dataframe() can handle list of dicts, pandas DataFrame, or polars DataFrame
        display_data = results

        # Display the DataFrame
        st.dataframe(display_data, **kwargs)

        # Show query information if requested
        if show_query_info:
            with st.expander("Query Information", expanded=False):
                try:
                    sql = df.to_sql()
                    st.code(sql, language="sql")
                except Exception:
                    st.text("SQL query not available")

                # Show row count
                if isinstance(results, list):
                    row_count: Union[int, str] = len(results)
                elif hasattr(results, "__len__"):
                    row_count = int(len(results))  # Convert to int for type safety
                else:
                    row_count = "Unknown"

                st.metric("Rows", row_count)

    except Exception as e:
        display_moltres_error(e)


def query_builder(db: "Database") -> Optional["DataFrame"]:
    """Interactive query builder widget for constructing Moltres queries.

    This function provides a Streamlit UI for building queries interactively,
    including table selection, column selection, filtering, and ordering.

    Args:
        db: Moltres :class:`Database` instance

    Returns:
        :class:`DataFrame` if a query was built, None otherwise

    Example:
        >>> import streamlit as st
        >>> from moltres import connect
        >>> from moltres.integrations.streamlit import query_builder
        >>>
        >>> db = connect("sqlite:///example.db")
        >>> df = query_builder(db)
        >>> if df:
        ...     results = df.collect()
        ...     st.dataframe(results)
    """
    _check_streamlit_available()

    # Get available tables
    try:
        tables = db.get_table_names()
        if not tables:
            st.info("No tables available in the database.")
            return None
    except Exception as e:
        display_moltres_error(e)
        return None

    # Table selection
    selected_table: str = st.selectbox("Select Table", tables)

    if not selected_table:
        return None

    # Get table columns
    try:
        table_handle = db.table(selected_table)
        # Get column info - we'll need to inspect the table
        # For now, create a simple select to get columns
        sample_df = table_handle.select()
        # Try to get column names from the plan or by executing a limit 0 query
        # This is a simplified approach - in practice, you'd want to use table inspection
        st.info(
            "Column selection and filtering coming soon. For now, use the DataFrame API directly."
        )
        return sample_df
    except Exception as e:
        display_moltres_error(e)
        return None

    # Note: Full query builder implementation would include:
    # - Column multi-select
    # - Filter builder (column, operator, value)
    # - Order by selection
    # - Limit input
    # This is a basic implementation that can be extended


def cached_query(
    ttl: Optional[int] = None,
    max_entries: Optional[int] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for caching Moltres query results in Streamlit.

    This decorator wraps functions that return DataFrames or query results,
    automatically caching them using Streamlit's cache_data mechanism.

    Args:
        ttl: Time-to-live for cache entries in seconds. If None, cache never expires.
        max_entries: Maximum number of cache entries. If None, no limit.

    Returns:
        Decorator function

    Example:
        >>> import streamlit as st
        >>> from moltres import connect
        >>> from moltres.integrations.streamlit import cached_query
        >>>
        >>> db = connect("sqlite:///example.db")
        >>>
        >>> @cached_query(ttl=3600)
        >>> def get_user_stats():
        ...     return db.table("users").select().agg(...).collect()
        >>>
        >>> results = get_user_stats()  # Cached for 1 hour
    """
    _check_streamlit_available()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Create cache_data decorator with TTL and max_entries
        cache_decorator = st.cache_data(ttl=ttl, max_entries=max_entries)

        @wraps(func)
        @cache_decorator
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # If result is a DataFrame, collect it for caching
            # (DataFrames can't be cached directly, need to materialize)
            if hasattr(result, "collect") and callable(result.collect):
                # Check if it's async
                if hasattr(result, "__class__") and "Async" in result.__class__.__name__:
                    st.warning(
                        "Async DataFrames cannot be cached directly. "
                        "Collect the results first, then cache the collected data."
                    )
                    return result
                # Materialize the DataFrame
                return result.collect()

            return result

        return wrapper

    return decorator


def clear_moltres_cache() -> None:
    """Clear all Moltres-related caches in Streamlit.

    This function clears Streamlit's cache_data cache, which will invalidate
    all cached query results decorated with @cached_query.

    Example:
        >>> import streamlit as st
        >>> from moltres.integrations.streamlit import clear_moltres_cache
        >>>
        >>> if st.button("Clear Cache"):
        ...     clear_moltres_cache()
        ...     st.success("Cache cleared!")
    """
    _check_streamlit_available()
    st.cache_data.clear()


def invalidate_query_cache(query_sql: str) -> None:
    """Invalidate cache for a specific query.

    Note: Streamlit's cache_data doesn't support selective invalidation by key.
    This function clears all caches. For selective invalidation, use clear_moltres_cache()
    or implement custom cache key management.

    Args:
        query_sql: SQL query string to invalidate (currently clears all caches)

    Example:
        >>> import streamlit as st
        >>> from moltres.integrations.streamlit import invalidate_query_cache
        >>>
        >>> invalidate_query_cache("SELECT * FROM users")
    """
    _check_streamlit_available()
    # Streamlit doesn't support selective cache invalidation
    # So we clear all caches
    st.warning("Streamlit cache_data doesn't support selective invalidation. Clearing all caches.")
    clear_moltres_cache()


def get_db_from_session(key: str = "db") -> "Database":
    """Get or create a :class:`Database` instance from Streamlit session state.

    This helper manages database connections in Streamlit's session state,
    ensuring connections are reused across reruns and properly cleaned up.

    Args:
        key: Key to use in session state for storing the database connection

    Returns:
        :class:`Database` instance

    Example:
        >>> import streamlit as st
        >>> from moltres.integrations.streamlit import get_db_from_session
        >>>
        >>> db = get_db_from_session()
        >>> df = db.table("users").select()
    """
    _check_streamlit_available()

    if key not in st.session_state:
        # Try to get connection string from Streamlit secrets or config
        dsn = None
        try:
            # Check Streamlit secrets
            if hasattr(st, "secrets") and hasattr(st.secrets, "get"):
                dsn = st.secrets.get("moltres", {}).get("dsn")
        except Exception:
            pass

        if not dsn:
            # Default to SQLite in-memory if no DSN configured
            dsn = "sqlite:///:memory:"
            st.info(f"No database DSN configured. Using default: {dsn}")

        from ... import connect

        st.session_state[key] = connect(dsn)

    db = st.session_state[key]
    assert isinstance(db, Database), "Expected Database instance"
    return db


def init_db_connection(dsn: str, key: str = "db") -> "Database":
    """Initialize a database connection in Streamlit session state.

    Args:
        dsn: :class:`Database` connection string
        key: Key to use in session state for storing the database connection

    Returns:
        :class:`Database` instance

    Example:
        >>> import streamlit as st
        >>> from moltres.integrations.streamlit import init_db_connection
        >>>
        >>> db = init_db_connection("sqlite:///example.db")
        >>> df = db.table("users").select()
    """
    _check_streamlit_available()

    from ... import connect

    # Close existing connection if any
    if key in st.session_state:
        try:
            db = st.session_state[key]
            if hasattr(db, "close"):
                db.close()
        except Exception:
            pass

    # Create new connection
    db = connect(dsn)
    st.session_state[key] = db
    return db


def close_db_connection(key: str = "db") -> None:
    """Close and remove a database connection from Streamlit session state.

    Args:
        key: Key used in session state for the database connection

    Example:
        >>> import streamlit as st
        >>> from moltres.integrations.streamlit import close_db_connection
        >>>
        >>> close_db_connection()
    """
    _check_streamlit_available()

    if key in st.session_state:
        try:
            db = st.session_state[key]
            if hasattr(db, "close"):
                db.close()
        except Exception:
            pass
        del st.session_state[key]


def visualize_query(
    df: "DataFrame | AsyncDataFrame",
    show_sql: bool = True,
    show_plan: bool = True,
    show_metrics: bool = False,
) -> None:
    """Visualize a Moltres query with SQL, execution plan, and performance metrics.

    This function displays query information in an organized format using
    Streamlit expanders for SQL, query plan, and performance metrics.

    Args:
        df: Moltres :class:`DataFrame` (sync or async) to visualize
        show_sql: If True, display the SQL query
        show_plan: If True, display the query execution plan
        show_metrics: If True, display performance metrics (execution time, row count)

    Example:
        >>> import streamlit as st
        >>> from moltres import connect, col
        >>> from moltres.integrations.streamlit import visualize_query
        >>>
        >>> db = connect("sqlite:///example.db")
        >>> df = db.table("users").select().where(col("age") > 25)
        >>> visualize_query(df, show_sql=True, show_plan=True, show_metrics=True)
    """
    _check_streamlit_available()

    # Handle async DataFrames
    if hasattr(df, "__class__") and "Async" in df.__class__.__name__:
        st.warning(
            "Async DataFrames must be collected before visualization. Use await df.collect() first."
        )
        return

    # Type guard: after async check, df is definitely a DataFrame
    from ...dataframe.core.dataframe import DataFrame as DataFrameType

    if not isinstance(df, DataFrameType):
        st.error("Unsupported DataFrame type")
        return

    # SQL Display
    if show_sql:
        with st.expander("SQL Query", expanded=True):
            try:
                sql = df.to_sql()
                st.code(sql, language="sql")
            except Exception as e:
                st.error(f"Could not generate SQL: {e}")

    # Query Plan Display
    if show_plan:
        with st.expander("Query Plan", expanded=False):
            try:
                plan = df.explain()
                st.text(plan)
            except Exception as e:
                st.warning(f"Could not get query plan: {e}")

    # Performance Metrics
    if show_metrics:
        with st.expander("Performance Metrics", expanded=False):
            try:
                start_time = time.time()
                results = df.collect()
                execution_time = time.time() - start_time

                if isinstance(results, list):
                    row_count: Union[int, str] = len(results)
                elif hasattr(results, "__len__"):
                    row_count = int(len(results))  # Convert to int for type safety
                else:
                    row_count = "Unknown"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Execution Time", f"{execution_time:.3f}s")
                with col2:
                    st.metric("Rows Returned", row_count)
            except Exception as e:
                st.error(f"Could not get performance metrics: {e}")


def display_moltres_error(error: Exception) -> None:
    """Display a Moltres error in a Streamlit-friendly format.

    This helper function formats Moltres exceptions for display in Streamlit,
    showing error messages and suggestions using appropriate Streamlit components.

    Args:
        error: Exception to display

    Example:
        >>> import streamlit as st
        >>> from moltres.integrations.streamlit import display_moltres_error
        >>>
        >>> try:
        ...     df = db.table("nonexistent").select()
        ...     df.collect()
        ... except Exception as e:
        ...     display_moltres_error(e)
    """
    _check_streamlit_available()

    from ...utils.exceptions import MoltresError

    if isinstance(error, MoltresError):
        st.error(f"**{type(error).__name__}**: {error.message}")

        if error.suggestion:
            st.warning(f"💡 **Suggestion**: {error.suggestion}")

        if error.context:
            with st.expander("Error Details"):
                st.json(error.context)
    else:
        st.error(f"**Error**: {str(error)}")
