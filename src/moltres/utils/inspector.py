"""Schema inspector utilities.

This module provides utilities for inspecting database schemas.
Supports both sync and async database connections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase
    from ..table.schema import ColumnDef
    from ..table.table import Database


@dataclass
class ColumnInfo:
    """Information about a database column.

    Attributes:
        name: The name of the column
        type_name: The SQL type name (e.g., "INTEGER", "TEXT", "VARCHAR(255)")
        nullable: Whether the column allows NULL values (default: True)
        default: Default value for the column (default: None)
        primary_key: Whether this is a primary key column (default: False)
        precision: Precision for numeric types like DECIMAL (default: None)
        scale: Scale for numeric types like DECIMAL (default: None)
    """

    name: str
    type_name: str
    nullable: bool = True
    default: object | None = None
    primary_key: bool = False
    precision: int | None = None
    scale: int | None = None

    def to_column_def(self) -> "ColumnDef":
        """Convert to ColumnDef for schema operations.

        Returns:
            ColumnDef object with the same metadata
        """
        from ..table.schema import ColumnDef

        return ColumnDef(
            name=self.name,
            type_name=self.type_name,
            nullable=self.nullable,
            default=self.default,
            primary_key=self.primary_key,
            precision=self.precision,
            scale=self.scale,
        )


def get_table_columns(db: Union["Database", "AsyncDatabase"], table_name: str) -> List[ColumnInfo]:
    """Get column information for a table from the database.

    Uses SQLAlchemy Inspector to query database metadata and retrieve
    column names and types. Works with both sync and async databases.

    Args:
        db: :class:`Database` instance to query (:class:`Database` or :class:`AsyncDatabase`)
        table_name: Name of the table to inspect

    Returns:
        List of ColumnInfo objects with column names and types

    Raises:
        ValueError: If database connection is not available
        RuntimeError: If table does not exist or cannot be inspected

    Example:
        >>> columns = get_table_columns(db, "users")
        >>> # Returns: [ColumnInfo(name='id', type_name='INTEGER'), ...]
    """
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy.ext.asyncio import AsyncEngine

    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    engine = db.connection_manager.engine

    # Declare columns variable
    columns: List[Dict[str, Any]]

    # Handle async engines - SQLAlchemy Inspector doesn't work directly with AsyncEngine
    if isinstance(engine, AsyncEngine):
        import asyncio
        import threading
        import concurrent.futures

        async def _get_columns_async() -> List[Dict[str, Any]]:
            async with engine.begin() as conn:
                # Use run_sync to call inspect on the underlying sync connection
                def _inspect_sync(sync_conn: Any) -> List[Dict[str, Any]]:
                    inspector = sa_inspect(sync_conn)
                    return inspector.get_columns(table_name)  # type: ignore[no-any-return]

                cols = await conn.run_sync(_inspect_sync)
                return cols

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run in a separate thread with new event loop
            future: concurrent.futures.Future[List[Dict[str, Any]]] = concurrent.futures.Future()

            def run_in_new_loop() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(_get_columns_async())
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
            try:
                columns = future.result()
            except Exception as e:
                raise RuntimeError(f"Failed to inspect table '{table_name}': {e}") from e
        except RuntimeError:
            # No running loop, we can create one
            try:
                columns = asyncio.run(_get_columns_async())
            except Exception as e:
                raise RuntimeError(f"Failed to inspect table '{table_name}': {e}") from e
    else:
        # Sync engine - use inspector directly
        inspector = sa_inspect(engine)
        try:
            columns = inspector.get_columns(table_name)  # type: ignore[assignment]
        except Exception as e:
            raise RuntimeError(f"Failed to inspect table '{table_name}': {e}") from e

    result: List[ColumnInfo] = []
    for col_info in columns:
        # Convert SQLAlchemy type to string representation
        type_name = str(col_info["type"])
        # Clean up type string (remove module paths, keep just the type name)
        # e.g., "INTEGER()" -> "INTEGER", "VARCHAR(255)" -> "VARCHAR(255)"
        if "(" in type_name:
            # Keep the full type with parameters
            type_name = type_name.split("(")[0] + "(" + type_name.split("(")[1]
        else:
            # Remove any module path prefixes
            type_name = type_name.split(".")[-1].replace("()", "")

        # Extract precision and scale from numeric types
        precision = None
        scale = None
        sa_type = col_info.get("type")
        if sa_type is not None and hasattr(sa_type, "precision") and sa_type.precision is not None:
            precision = sa_type.precision
        if sa_type is not None and hasattr(sa_type, "scale") and sa_type.scale is not None:
            scale = sa_type.scale

        # Convert primary_key to boolean (SQLAlchemy returns 1/0)
        primary_key = col_info.get("primary_key", False)
        if isinstance(primary_key, int):
            primary_key = bool(primary_key)

        result.append(
            ColumnInfo(
                name=col_info["name"],
                type_name=type_name,
                nullable=col_info.get("nullable", True),
                default=col_info.get("default"),
                primary_key=primary_key,
                precision=precision,
                scale=scale,
            )
        )

    return result


def get_table_schema(db: Union["Database", "AsyncDatabase"], table_name: str) -> List[ColumnInfo]:
    """Get schema information for a table.

    Alias for get_table_columns() for consistency with PySpark terminology.

    Args:
        db: :class:`Database` instance to query
        table_name: Name of the table to inspect

    Returns:
        List of ColumnInfo objects with column names and types
    """
    return get_table_columns(db, table_name)


def _get_inspector(db: Union["Database", "AsyncDatabase"]) -> Any:
    """Get SQLAlchemy Inspector for the database.

    Handles both sync and async engines.

    Args:
        db: :class:`Database` instance

    Returns:
        SQLAlchemy Inspector instance

    Raises:
        ValueError: If database connection is not available
    """
    from sqlalchemy import inspect as sa_inspect
    from sqlalchemy.ext.asyncio import AsyncEngine

    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    engine = db.connection_manager.engine

    # For async engines, we need to use run_sync pattern
    if isinstance(engine, AsyncEngine):
        # This function should be called from within an async context
        # that uses conn.run_sync() to get the inspector
        raise RuntimeError(
            "Cannot get inspector directly from async engine. Use async helper functions instead."
        )

    return sa_inspect(engine)


def _get_inspector_async(engine: Any) -> Any:
    """Get SQLAlchemy Inspector from async engine using run_sync.

    Args:
        engine: AsyncEngine instance

    Returns:
        Coroutine that yields SQLAlchemy Inspector
    """
    from sqlalchemy import inspect as sa_inspect

    async def _get_inspector_coro() -> Any:
        async with engine.begin() as conn:
            # Use run_sync to call inspect on the underlying sync connection
            def _inspect_sync(sync_conn: Any) -> Any:
                return sa_inspect(sync_conn)

            inspector = await conn.run_sync(_inspect_sync)
            return inspector

    return _get_inspector_coro()


def get_table_names(
    db: Union["Database", "AsyncDatabase"], schema: Optional[str] = None
) -> List[str]:
    """Get list of table names in the database.

    Uses SQLAlchemy Inspector to query database metadata. Works with both
    sync and async databases.

    Args:
        db: :class:`Database` instance to query (:class:`Database` or :class:`AsyncDatabase`)
        schema: Optional schema name (for multi-schema databases like PostgreSQL).
                If None, uses default schema.

    Returns:
        List of table names

    Raises:
        ValueError: If database connection is not available
        RuntimeError: If inspection fails

    Example:
        >>> tables = get_table_names(db)
        >>> # Returns: ['users', 'orders', 'products']
    """
    from sqlalchemy.ext.asyncio import AsyncEngine

    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    engine = db.connection_manager.engine

    # Handle async engines
    if isinstance(engine, AsyncEngine):
        import asyncio
        import threading
        import concurrent.futures

        async def _get_table_names_async() -> List[str]:
            async with engine.begin() as conn:

                def _inspect_sync(sync_conn: Any) -> List[str]:
                    from sqlalchemy import inspect as sa_inspect

                    inspector = sa_inspect(sync_conn)
                    return inspector.get_table_names(schema=schema)  # type: ignore[no-any-return]

                table_names = await conn.run_sync(_inspect_sync)
                return table_names

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run in a separate thread with new event loop
            future: concurrent.futures.Future[List[str]] = concurrent.futures.Future()

            def run_in_new_loop() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(_get_table_names_async())
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
            try:
                return future.result()
            except Exception as e:
                raise RuntimeError(f"Failed to get table names: {e}") from e
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(_get_table_names_async())
    else:
        # Sync engine - use inspector directly
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(engine)
        try:
            return inspector.get_table_names(schema=schema)
        except Exception as e:
            raise RuntimeError(f"Failed to get table names: {e}") from e


def get_view_names(
    db: Union["Database", "AsyncDatabase"], schema: Optional[str] = None
) -> List[str]:
    """Get list of view names in the database.

    Uses SQLAlchemy Inspector to query database metadata. Works with both
    sync and async databases.

    Args:
        db: :class:`Database` instance to query (:class:`Database` or :class:`AsyncDatabase`)
        schema: Optional schema name (for multi-schema databases like PostgreSQL).
                If None, uses default schema.

    Returns:
        List of view names

    Raises:
        ValueError: If database connection is not available
        RuntimeError: If inspection fails

    Example:
        >>> views = get_view_names(db)
        >>> # Returns: ['active_users_view', 'order_summary_view']
    """
    from sqlalchemy.ext.asyncio import AsyncEngine

    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    engine = db.connection_manager.engine

    # Handle async engines
    if isinstance(engine, AsyncEngine):
        import asyncio
        import threading
        import concurrent.futures

        async def _get_view_names_async() -> List[str]:
            async with engine.begin() as conn:

                def _inspect_sync(sync_conn: Any) -> List[str]:
                    from sqlalchemy import inspect as sa_inspect

                    inspector = sa_inspect(sync_conn)
                    return inspector.get_view_names(schema=schema)  # type: ignore[no-any-return]

                view_names = await conn.run_sync(_inspect_sync)
                return view_names

        # Check if we're in an async context
        try:
            asyncio.get_running_loop()
            # We're in an async context - run in a separate thread with new event loop
            future: concurrent.futures.Future[List[str]] = concurrent.futures.Future()

            def run_in_new_loop() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result = new_loop.run_until_complete(_get_view_names_async())
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=run_in_new_loop)
            thread.start()
            thread.join()
            try:
                return future.result()
            except Exception as e:
                raise RuntimeError(f"Failed to get view names: {e}") from e
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(_get_view_names_async())
    else:
        # Sync engine - use inspector directly
        from sqlalchemy import inspect as sa_inspect

        inspector = sa_inspect(engine)
        try:
            return inspector.get_view_names(schema=schema)
        except Exception as e:
            raise RuntimeError(f"Failed to get view names: {e}") from e


def reflect_table(
    db: Union["Database", "AsyncDatabase"],
    table_name: str,
    schema: Optional[str] = None,
) -> Dict[str, List["ColumnDef"]]:
    """Reflect a single table from the database.

    Uses SQLAlchemy Inspector to reflect table metadata and convert it to
    Moltres ColumnDef objects. Works with both sync and async databases.

    Args:
        db: :class:`Database` instance to query (:class:`Database` or :class:`AsyncDatabase`)
        table_name: Name of the table to reflect
        schema: Optional schema name (for multi-schema databases like PostgreSQL).
                If None, uses default schema.

    Returns:
        Dictionary mapping table name to list of ColumnDef objects

    Raises:
        ValueError: If database connection is not available
        RuntimeError: If table does not exist or reflection fails

    Example:
        >>> table_schema = reflect_table(db, "users")
        >>> # Returns: {'users': [ColumnDef(name='id', type_name='INTEGER', ...), ...]}
    """

    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    # Get column information
    columns = get_table_columns(db, table_name)

    # Convert ColumnInfo to ColumnDef

    column_defs = [col_info.to_column_def() for col_info in columns]

    return {table_name: column_defs}


def reflect_database(
    db: Union["Database", "AsyncDatabase"],
    schema: Optional[str] = None,
    views: bool = False,
) -> Dict[str, List["ColumnDef"]]:
    """Reflect entire database schema.

    Uses SQLAlchemy Inspector to reflect all tables (and optionally views)
    in the database. Works with both sync and async databases.

    Args:
        db: :class:`Database` instance to query (:class:`Database` or :class:`AsyncDatabase`)
        schema: Optional schema name (for multi-schema databases like PostgreSQL).
                If None, uses default schema.
        views: If True, also reflect views (default: False)

    Returns:
        Dictionary mapping table/view names to lists of ColumnDef objects

    Raises:
        ValueError: If database connection is not available
        RuntimeError: If reflection fails

    Example:
        >>> schemas = reflect_database(db)
        >>> # Returns: {'users': [ColumnDef(...), ...], 'orders': [ColumnDef(...), ...]}
    """
    if db.connection_manager is None:
        raise ValueError("Database connection manager is not available")

    # Get all table names
    table_names = get_table_names(db, schema=schema)

    # Optionally get view names
    view_names: List[str] = []
    if views:
        view_names = get_view_names(db, schema=schema)

    # Reflect each table
    result: Dict[str, List["ColumnDef"]] = {}
    for table_name in table_names:
        try:
            table_schema = reflect_table(db, table_name, schema=schema)
            result.update(table_schema)
        except Exception as e:
            # Log but continue with other tables
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to reflect table '{table_name}': {e}")

    # Reflect views if requested
    for view_name in view_names:
        try:
            # Views are queried like tables for column information
            view_schema = reflect_table(db, view_name, schema=schema)
            result.update(view_schema)
        except Exception as e:
            # Log but continue with other views
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to reflect view '{view_name}': {e}")

    return result


def sql_type_to_pandas_dtype(sql_type: str) -> str:
    """Map SQL type names to pandas dtype strings.

    Args:
        sql_type: SQL type name (e.g., "INTEGER", "TEXT", "VARCHAR(255)", "REAL")

    Returns:
        pandas dtype string (e.g., "int64", "object", "float64")

    Example:
        >>> sql_type_to_pandas_dtype("INTEGER")
        'int64'
        >>> sql_type_to_pandas_dtype("TEXT")
        'object'
        >>> sql_type_to_pandas_dtype("REAL")
        'float64'
    """
    # Normalize the type name - remove parameters and convert to uppercase
    type_upper = sql_type.upper().strip()
    # Remove parameters if present (e.g., "VARCHAR(255)" -> "VARCHAR")
    if "(" in type_upper:
        type_upper = type_upper.split("(")[0].strip()

    # Remove parentheses suffix if present
    type_upper = type_upper.replace("()", "")

    # Map SQL types to pandas dtypes
    type_mapping: Dict[str, str] = {
        # Integer types
        "INTEGER": "int64",
        "INT": "int64",
        "BIGINT": "int64",
        "SMALLINT": "int64",
        "TINYINT": "int64",
        "SERIAL": "int64",
        "BIGSERIAL": "int64",
        # Floating point types
        "REAL": "float64",
        "FLOAT": "float64",
        "DOUBLE": "float64",
        "DOUBLE PRECISION": "float64",
        "NUMERIC": "float64",
        "DECIMAL": "float64",
        "MONEY": "float64",
        # Text types
        "TEXT": "object",
        "VARCHAR": "object",
        "CHAR": "object",
        "CHARACTER": "object",
        "STRING": "object",
        "CLOB": "object",
        # Binary types
        "BLOB": "object",
        "BYTEA": "object",
        "BINARY": "object",
        "VARBINARY": "object",
        # Boolean
        "BOOLEAN": "bool",
        "BOOL": "bool",
        # Date/Time types
        "DATE": "datetime64[ns]",
        "TIME": "datetime64[ns]",
        "TIMESTAMP": "datetime64[ns]",
        "DATETIME": "datetime64[ns]",
        "TIMESTAMPTZ": "datetime64[ns]",
        # JSON types
        "JSON": "object",
        "JSONB": "object",
        # UUID
        "UUID": "object",
    }

    # Return mapped type or default to 'object' for unknown types
    return type_mapping.get(type_upper, "object")
