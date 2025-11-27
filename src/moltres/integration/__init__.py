"""SQLAlchemy integration helpers for Moltres.

This module provides helper functions for integrating Moltres with existing
SQLAlchemy projects, allowing you to use Moltres DataFrames with existing
SQLAlchemy connections, sessions, and infrastructure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection
    from sqlalchemy.orm import Session
    from sqlalchemy.sql import Select
    from ..dataframe.dataframe import DataFrame
    from ..table.table import Database


def execute_with_connection(df: "DataFrame", connection: "Connection") -> List[Dict[str, Any]]:
    """Execute a Moltres DataFrame using a provided SQLAlchemy Connection.

    This allows you to execute Moltres queries within an existing SQLAlchemy
    transaction or connection context.

    Args:
        df: Moltres DataFrame to execute
        connection: SQLAlchemy Connection to use for execution

    Returns:
        List of dictionaries representing rows

    Example:
        >>> from sqlalchemy import create_engine
        >>> from moltres import connect, col
        >>> from moltres.integration import execute_with_connection
        >>> engine = create_engine("sqlite:///:memory:")
        >>> db = connect(engine=engine)
        >>> df = db.table("users").select().where(col("id") > 1)
        >>> with engine.connect() as conn:
        ...     results = execute_with_connection(df, conn)
    """
    # Convert DataFrame to SQLAlchemy statement
    stmt = df.to_sqlalchemy()

    # Execute using the provided connection
    result = connection.execute(stmt)
    rows = result.fetchall()
    columns = list(result.keys())

    # Format as list of dicts
    return [dict(zip(columns, row)) for row in rows]


def execute_with_session(df: "DataFrame", session: "Session") -> List[Dict[str, Any]]:
    """Execute a Moltres DataFrame using a SQLAlchemy ORM Session.

    This allows you to execute Moltres queries within an existing SQLAlchemy
    ORM session context.

    Args:
        df: Moltres DataFrame to execute
        session: SQLAlchemy ORM Session to use for execution

    Returns:
        List of dictionaries representing rows

    Example:
        >>> from sqlalchemy import create_engine
        >>> from sqlalchemy.orm import sessionmaker
        >>> from moltres import connect, col
        >>> from moltres.integration import execute_with_session
        >>> engine = create_engine("sqlite:///:memory:")
        >>> Session = sessionmaker(bind=engine)
        >>> db = connect(engine=engine)
        >>> df = db.table("users").select().where(col("id") > 1)
        >>> with Session() as session:
        ...     results = execute_with_session(df, session)
    """
    # Get connection from session
    connection = session.connection()
    return execute_with_connection(df, connection)


def to_sqlalchemy_select(df: "DataFrame", dialect: Optional[str] = None) -> "Select":
    """Convert a Moltres DataFrame to a SQLAlchemy Select statement.

    This is a convenience function that wraps DataFrame.to_sqlalchemy().

    Args:
        df: Moltres DataFrame to convert
        dialect: Optional SQL dialect name. If not provided, uses the dialect
                from the DataFrame's attached Database, or defaults to "ansi"

    Returns:
        SQLAlchemy Select statement

    Example:
        >>> from moltres import connect, col
        >>> from moltres.integration import to_sqlalchemy_select
        >>> db = connect("sqlite:///:memory:")
        >>> df = db.table("users").select().where(col("id") > 1)
        >>> stmt = to_sqlalchemy_select(df)
        >>> # Now use stmt with any SQLAlchemy connection
    """
    return df.to_sqlalchemy(dialect=dialect)


def from_sqlalchemy_select(
    select_stmt: "Select", database: Optional["Database"] = None
) -> "DataFrame":
    """Create a Moltres DataFrame from a SQLAlchemy Select statement.

    This is a convenience function that wraps DataFrame.from_sqlalchemy().

    Args:
        select_stmt: SQLAlchemy Select statement to convert
        database: Optional Database instance to attach to the DataFrame

    Returns:
        Moltres DataFrame that can be further chained with Moltres operations

    Example:
        >>> from sqlalchemy import create_engine, select, table, column
        >>> from moltres.integration import from_sqlalchemy_select
        >>> engine = create_engine("sqlite:///:memory:")
        >>> users = table("users", column("id"), column("name"))
        >>> sa_stmt = select(users.c.id, users.c.name).where(users.c.id > 1)
        >>> df = from_sqlalchemy_select(sa_stmt)
        >>> # Can now chain Moltres operations
    """
    from ..dataframe.dataframe import DataFrame

    return DataFrame.from_sqlalchemy(select_stmt, database=database)


def with_sqlmodel(df: "DataFrame", model: Type[Any]) -> "DataFrame":
    """Attach a SQLModel or Pydantic model to a DataFrame.

    This is a convenience function that wraps DataFrame.with_model().

    Args:
        df: Moltres DataFrame
        model: SQLModel or Pydantic model class to attach

    Returns:
        DataFrame with the model attached

    Example:
        >>> from sqlmodel import SQLModel, Field
        >>> from moltres import connect
        >>> from moltres.integration import with_sqlmodel
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        >>> db = connect("sqlite:///:memory:")
        >>> df = db.table("users").select()
        >>> df_with_model = with_sqlmodel(df, User)
        >>> results = df_with_model.collect()  # Returns list of User instances

        >>> from pydantic import BaseModel
        >>> class UserData(BaseModel):
        ...     id: int
        ...     name: str
        >>> df_with_pydantic = with_sqlmodel(df, UserData)
        >>> results = df_with_pydantic.collect()  # Returns list of UserData instances
    """
    return df.with_model(model)


def execute_with_connection_model(
    df: "DataFrame", connection: "Connection", model: Type[Any]
) -> List[Any]:
    """Execute a Moltres DataFrame using a provided SQLAlchemy Connection and return SQLModel instances.

    Args:
        df: Moltres DataFrame to execute
        connection: SQLAlchemy Connection to use for execution
        model: SQLModel model class to instantiate results as

    Returns:
        List of SQLModel instances

    Example:
        >>> from sqlmodel import SQLModel, Field
        >>> from sqlalchemy import create_engine
        >>> from moltres import connect, col
        >>> from moltres.integration import execute_with_connection_model
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        >>> engine = create_engine("sqlite:///:memory:")
        >>> db = connect(engine=engine)
        >>> df = db.table("users").select().where(col("id") > 1)
        >>> with engine.connect() as conn:
        ...     results = execute_with_connection_model(df, conn, User)
    """
    df_with_model = df.with_model(model)
    # Convert DataFrame to SQLAlchemy statement
    stmt = df_with_model.to_sqlalchemy()

    # Execute using the provided connection
    result = connection.execute(stmt)
    rows = result.fetchall()
    columns = list(result.keys())

    # Format as list of dicts
    dict_rows = [dict(zip(columns, row)) for row in rows]

    # Convert to SQLModel instances
    from ..utils.sqlmodel_integration import rows_to_sqlmodels

    return rows_to_sqlmodels(dict_rows, model)


def execute_with_session_model(df: "DataFrame", session: "Session", model: Type[Any]) -> List[Any]:
    """Execute a Moltres DataFrame using a SQLAlchemy ORM Session and return SQLModel instances.

    Args:
        df: Moltres DataFrame to execute
        session: SQLAlchemy ORM Session to use for execution
        model: SQLModel model class to instantiate results as

    Returns:
        List of SQLModel instances

    Example:
        >>> from sqlmodel import SQLModel, Field
        >>> from sqlalchemy import create_engine
        >>> from sqlalchemy.orm import sessionmaker
        >>> from moltres import connect, col
        >>> from moltres.integration import execute_with_session_model
        >>> class User(SQLModel, table=True):
        ...     id: int = Field(primary_key=True)
        ...     name: str
        >>> engine = create_engine("sqlite:///:memory:")
        >>> Session = sessionmaker(bind=engine)
        >>> db = connect(engine=engine)
        >>> df = db.table("users").select().where(col("id") > 1)
        >>> with Session() as session:
        ...     results = execute_with_session_model(df, session, User)
    """
    # Get connection from session
    connection = session.connection()
    return execute_with_connection_model(df, connection, model)


__all__ = [
    "execute_with_connection",
    "execute_with_session",
    "to_sqlalchemy_select",
    "from_sqlalchemy_select",
    "with_sqlmodel",
    "execute_with_connection_model",
    "execute_with_session_model",
]
