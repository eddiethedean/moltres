"""Async SQLAlchemy integration helpers for Moltres.

This module provides async helper functions for integrating Moltres with existing
SQLAlchemy async projects, allowing you to use Moltres AsyncDataFrames with existing
SQLAlchemy async connections, sessions, and infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession
    from sqlalchemy.sql import Select
    from ...dataframe.core.async_dataframe import AsyncDataFrame
    from ...table.async_table import AsyncDatabase


async def execute_with_async_connection(
    df: "AsyncDataFrame", connection: "AsyncConnection"
) -> List[Dict[str, Any]]:
    """Execute a Moltres AsyncDataFrame using a provided SQLAlchemy AsyncConnection.

    This allows you to execute Moltres queries within an existing SQLAlchemy
    async transaction or connection context.

    Args:
        df: Moltres AsyncDataFrame to execute
        connection: SQLAlchemy AsyncConnection to use for execution

    Returns:
        List of dictionaries representing rows

    Example:
        >>> from sqlalchemy.ext.asyncio import create_async_engine
        >>> from moltres import async_connect, col
        >>> from moltres.integrations.sqlalchemy.async_integration import execute_with_async_connection
        >>> async def example():
        ...     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        ...     db = await async_connect(engine=engine)
        ...     table_handle = await db.table("users")
        ...     df = table_handle.select().where(col("id") > 1)
        ...     async with engine.connect() as conn:
        ...         results = await execute_with_async_connection(df, conn)
    """
    # Convert AsyncDataFrame to SQLAlchemy statement
    stmt = df.to_sqlalchemy()

    # Execute using the provided connection
    result = await connection.execute(stmt)
    rows = result.fetchall()
    columns = list(result.keys())

    # Format as list of dicts
    return [dict(zip(columns, row)) for row in rows]


async def execute_with_async_session(
    df: "AsyncDataFrame", session: "AsyncSession"
) -> List[Dict[str, Any]]:
    """Execute a Moltres AsyncDataFrame using a SQLAlchemy AsyncSession.

    This allows you to execute Moltres queries within an existing SQLAlchemy
    async ORM session context.

    Args:
        df: Moltres AsyncDataFrame to execute
        session: SQLAlchemy AsyncSession to use for execution

    Returns:
        List of dictionaries representing rows

    Example:
        >>> from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        >>> from moltres import async_connect, col
        >>> from moltres.integrations.sqlalchemy.async_integration import execute_with_async_session
        >>> async def example():
        ...     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        ...     AsyncSession = async_sessionmaker(bind=engine)
        ...     db = await async_connect(engine=engine)
        ...     table_handle = await db.table("users")
        ...     df = table_handle.select().where(col("id") > 1)
        ...     async with AsyncSession() as session:
        ...         results = await execute_with_async_session(df, session)
    """
    # Get connection from session
    # Note: session.connection() is async in SQLAlchemy 2.0
    connection = await session.connection()
    return await execute_with_async_connection(df, connection)


def to_sqlalchemy_select_async(df: "AsyncDataFrame", dialect: Optional[str] = None) -> "Select":
    """Convert a Moltres AsyncDataFrame to a SQLAlchemy Select statement.

    This is a convenience function that wraps AsyncDataFrame.to_sqlalchemy().

    Args:
        df: Moltres AsyncDataFrame to convert
        dialect: Optional SQL dialect name. If not provided, uses the dialect
                from the AsyncDataFrame's attached :class:`AsyncDatabase`, or defaults to "ansi"

    Returns:
        SQLAlchemy Select statement

    Example:
        >>> from moltres import async_connect, col
        >>> from moltres.integrations.sqlalchemy.async_integration import to_sqlalchemy_select_async
        >>> async def example():
        ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
        ...     table_handle = await db.table("users")
        ...     df = table_handle.select().where(col("id") > 1)
        ...     stmt = to_sqlalchemy_select_async(df)
        ...     # Now use stmt with any SQLAlchemy async connection
    """
    return df.to_sqlalchemy(dialect=dialect)


def from_sqlalchemy_select_async(
    select_stmt: "Select", database: Optional["AsyncDatabase"] = None
) -> "AsyncDataFrame":
    """Create a Moltres AsyncDataFrame from a SQLAlchemy Select statement.

    This is a convenience function that wraps AsyncDataFrame.from_sqlalchemy().

    Args:
        select_stmt: SQLAlchemy Select statement to convert
        database: Optional :class:`AsyncDatabase` instance to attach to the AsyncDataFrame

    Returns:
        Moltres AsyncDataFrame that can be further chained with Moltres operations

    Example:
        >>> from sqlalchemy.ext.asyncio import create_async_engine
        >>> from sqlalchemy import select, table, column
        >>> from moltres.integrations.sqlalchemy.async_integration import from_sqlalchemy_select_async
        >>> async def example():
        ...     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        ...     users = table("users", column("id"), column("name"))
        ...     sa_stmt = select(users.c.id, users.c.name).where(users.c.id > 1)
        ...     df = from_sqlalchemy_select_async(sa_stmt)
        ...     # Can now chain Moltres operations
    """
    from ...dataframe.core.async_dataframe import AsyncDataFrame

    return AsyncDataFrame.from_sqlalchemy(select_stmt, database=database)


def with_sqlmodel_async(df: "AsyncDataFrame", model: Type[Any]) -> "AsyncDataFrame":
    """Attach a SQLModel or Pydantic model to an AsyncDataFrame.

    This is a convenience function that wraps AsyncDataFrame.with_model().

    Args:
        df: Moltres AsyncDataFrame
        model: SQLModel or Pydantic model class to attach

    Returns:
        AsyncDataFrame with the model attached

    Example:
        >>> from sqlmodel import SQLModel, Field
        >>> from moltres import async_connect
        >>> from moltres.integrations.sqlalchemy.async_integration import with_sqlmodel_async
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        >>> async def example():
        ...     db = async_connect("sqlite+aiosqlite:///:memory:")
        ...     table_handle = await db.table("users")
        ...     df = table_handle.select()
        ...     df_with_model = with_sqlmodel_async(df, User)
        ...     results = await df_with_model.collect()  # Returns list of User instances

        >>> from pydantic import BaseModel
        >>> class UserData(BaseModel):
        ...     id: int
        ...     name: str
        >>> async def example():
        ...     df_with_pydantic = with_sqlmodel_async(df, UserData)
        ...     results = await df_with_pydantic.collect()  # Returns list of UserData instances
    """
    return df.with_model(model)


async def execute_with_async_connection_model(
    df: "AsyncDataFrame", connection: "AsyncConnection", model: Type[Any]
) -> List[Any]:
    """Execute a Moltres AsyncDataFrame using a provided SQLAlchemy AsyncConnection and return SQLModel instances.

    Args:
        df: Moltres AsyncDataFrame to execute
        connection: SQLAlchemy AsyncConnection to use for execution
        model: SQLModel model class to instantiate results as

    Returns:
        List of SQLModel instances

    Example:
        >>> from sqlmodel import SQLModel, Field
        >>> from sqlalchemy.ext.asyncio import create_async_engine
        >>> from moltres import async_connect, col
        >>> from moltres.integrations.sqlalchemy.async_integration import execute_with_async_connection_model
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        >>> async def example():
        ...     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        ...     db = await async_connect(engine=engine)
        ...     table_handle = await db.table("users")
        ...     df = table_handle.select().where(col("id") > 1)
        ...     async with engine.connect() as conn:
        ...         results = await execute_with_async_connection_model(df, conn, User)
    """
    df_with_model = df.with_model(model)
    # Convert AsyncDataFrame to SQLAlchemy statement
    stmt = df_with_model.to_sqlalchemy()

    # Execute using the provided connection
    result = await connection.execute(stmt)
    rows = result.fetchall()
    columns = list(result.keys())

    # Format as list of dicts
    dict_rows = [dict(zip(columns, row)) for row in rows]

    # Convert to SQLModel instances
    from ...utils.sqlmodel_integration import rows_to_sqlmodels

    return rows_to_sqlmodels(dict_rows, model)


async def execute_with_async_session_model(
    df: "AsyncDataFrame", session: "AsyncSession", model: Type[Any]
) -> List[Any]:
    """Execute a Moltres AsyncDataFrame using a SQLAlchemy AsyncSession and return SQLModel instances.

    Args:
        df: Moltres AsyncDataFrame to execute
        session: SQLAlchemy AsyncSession to use for execution
        model: SQLModel model class to instantiate results as

    Returns:
        List of SQLModel instances

    Example:
        >>> from sqlmodel import SQLModel, Field
        >>> from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
        >>> from moltres import async_connect, col
        >>> from moltres.integrations.sqlalchemy.async_integration import execute_with_async_session_model
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        >>> async def example():
        ...     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        ...     AsyncSession = async_sessionmaker(bind=engine)
        ...     db = await async_connect(engine=engine)
        ...     table_handle = await db.table("users")
        ...     df = table_handle.select().where(col("id") > 1)
        ...     async with AsyncSession() as session:
        ...         results = await execute_with_async_session_model(df, session, User)
    """
    # Get connection from session
    connection = await session.connection()
    return await execute_with_async_connection_model(df, connection, model)


__all__ = [
    "execute_with_async_connection",
    "execute_with_async_session",
    "to_sqlalchemy_select_async",
    "from_sqlalchemy_select_async",
    "with_sqlmodel_async",
    "execute_with_async_connection_model",
    "execute_with_async_session_model",
]
