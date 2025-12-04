"""Transaction decorator for automatic transaction management.

This module provides a decorator that automatically wraps functions in database transactions,
simplifying transaction management for common use cases.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union, overload

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..table.table import Database
    from ..table.async_table import AsyncDatabase

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@overload
def transaction(
    database: "Database",
    *,
    savepoint: bool = False,
    readonly: bool = False,
    isolation_level: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Callable[[F], F]: ...


@overload
def transaction(
    database: "AsyncDatabase",
    *,
    savepoint: bool = False,
    readonly: bool = False,
    isolation_level: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Callable[[F], F]: ...


@overload
def transaction(
    *,
    savepoint: bool = False,
    readonly: bool = False,
    isolation_level: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Callable[[F], F]: ...


def transaction(
    database: Optional[Union["Database", "AsyncDatabase"]] = None,
    *,
    savepoint: bool = False,
    readonly: bool = False,
    isolation_level: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Union[Callable[[F], F], Callable[[Callable[..., Any]], Callable[..., Any]]]:
    """Decorator to automatically wrap a function in a database transaction.

    The decorator can be used in two ways:

    1. With a database instance (recommended for class methods or functions with db parameter):
       ```python
       db = connect("sqlite:///example.db")

       @transaction(db)
       def create_user(name: str):
           Records(_data=[{"name": name}], _database=db).insert_into("users")
       ```

    2. Without a database instance (database is expected as first parameter or keyword arg):
       ```python
       @transaction
       def create_user(db: Database, name: str):
           Records(_data=[{"name": name}], _database=db).insert_into("users")
       ```

    For async functions, the decorator will automatically detect and handle async transactions:
       ```python
       @transaction(async_db)
       async def create_user_async(name: str):
           await AsyncRecords(_data=[{"name": name}], _database=async_db).insert_into("users")
       ```

    Args:
        database: Optional database instance. If provided, this database will be used for
                 all transactions. If not provided, the database is expected as the first
                 parameter or a 'db' keyword argument in the decorated function.
        savepoint: Use savepoint for nested transactions (default: False)
        readonly: Create read-only transaction (default: False)
        isolation_level: Set transaction isolation level (e.g., "SERIALIZABLE", "READ COMMITTED")
        timeout: Transaction timeout in seconds (default: None)

    Returns:
        Decorated function that automatically runs in a transaction

    Example:
        >>> from moltres import connect
        >>> from moltres.utils.transaction_decorator import transaction
        >>> from moltres.io.records import Records
        >>>
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
        >>>
        >>> # Method 1: Provide database instance
        >>> @transaction(db)
        ... def add_user(name: str):
        ...     Records(_data=[{"id": 1, "name": name}], _database=db).insert_into("users")
        >>>
        >>> # Method 2: Database as parameter
        >>> @transaction
        ... def add_user_with_db(db: Database, name: str):
        ...     Records(_data=[{"id": 2, "name": name}], _database=db).insert_into("users")
        >>>
        >>> add_user("Alice")
        >>> add_user_with_db(db, "Bob")
        >>> results = db.table("users").select().collect()
        >>> assert len(results) == 2
    """
    # If database is None, this is the decorator being called without arguments
    # We need to return a decorator that expects the function
    if database is None:
        # Called as @transaction
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return _wrap_with_transaction(
                func,
                database=None,
                savepoint=savepoint,
                readonly=readonly,
                isolation_level=isolation_level,
                timeout=timeout,
            )

        return decorator

    # If database is provided, check if it's actually a function (called as @transaction(db))
    # or if it's the database instance (called as @transaction(db))
    if callable(database):
        # Called as @transaction with function (database is actually the function)
        func = database
        return _wrap_with_transaction(
            func,
            database=None,
            savepoint=savepoint,
            readonly=readonly,
            isolation_level=isolation_level,
            timeout=timeout,
        )

    # Called as @transaction(db) - database is the Database instance
    def db_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return _wrap_with_transaction(
            func,
            database=database,
            savepoint=savepoint,
            readonly=readonly,
            isolation_level=isolation_level,
            timeout=timeout,
        )

    return db_decorator


def _wrap_with_transaction(
    func: Callable[..., Any],
    database: Optional[Union["Database", "AsyncDatabase"]] = None,
    savepoint: bool = False,
    readonly: bool = False,
    isolation_level: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Callable[..., Any]:
    """Wrap a function to run in a transaction."""
    import inspect

    is_async = inspect.iscoroutinefunction(func)

    if is_async:
        # Async function

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine which database to use
            db: Any = database
            if db is None:
                # Try to find database in args or kwargs
                for arg in args:
                    from ..table.async_table import AsyncDatabase

                    if isinstance(arg, AsyncDatabase):
                        db = arg
                        break
                if db is None:
                    db = kwargs.get("db")
                    if db is None:
                        raise ValueError(
                            "Database instance not found. Provide database as first argument "
                            "or 'db' keyword argument, or pass database to @transaction decorator."
                        )

            # Run function in transaction
            async with db.transaction(
                savepoint=savepoint,
                readonly=readonly,
                isolation_level=isolation_level,
                timeout=timeout,
            ):
                return await func(*args, **kwargs)

        return async_wrapper
    else:
        # Sync function

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine which database to use
            db: Any = database
            if db is None:
                # Try to find database in args or kwargs
                for arg in args:
                    from ..table.table import Database

                    if isinstance(arg, Database):
                        db = arg
                        break
                if db is None:
                    db = kwargs.get("db")
                    if db is None:
                        raise ValueError(
                            "Database instance not found. Provide database as first argument "
                            "or 'db' keyword argument, or pass database to @transaction decorator."
                        )

            # Run function in transaction
            with db.transaction(
                savepoint=savepoint,
                readonly=readonly,
                isolation_level=isolation_level,
                timeout=timeout,
            ):
                return func(*args, **kwargs)

        return sync_wrapper
