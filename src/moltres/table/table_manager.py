"""Database table operations manager.

This module handles table access and data insertion operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence, Type, Union

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from sqlalchemy.orm import DeclarativeBase
    from .table import Database, TableHandle
    from ..io.records import Records


class TableManager:
    """Handles table access and data insertion operations for Database."""

    def __init__(self, database: "Database"):
        """Initialize table manager with a Database.

        Args:
            database: The Database instance to manage
        """
        self._db = database

    def table(self, name_or_model: Union[str, Type["DeclarativeBase"], Type[Any]]) -> "TableHandle":
        """Get a handle to a table in the database.

        Args:
            name_or_model: Name of the table, SQLAlchemy model class, or SQLModel model class

        Returns:
            TableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
            ValueError: If model_class is not a valid SQLAlchemy or SQLModel model
        """
        from typing import cast
        from ..utils.exceptions import ValidationError
        from ..sql.builders import quote_identifier
        from .sqlalchemy_integration import (
            is_sqlalchemy_model,
            get_model_table_name,
        )
        from ..utils.sqlmodel_integration import (
            is_sqlmodel_model,
            get_sqlmodel_table_name,
        )
        from .table import TableHandle

        # Check if argument is a SQLModel model
        if is_sqlmodel_model(name_or_model):
            sqlmodel_class: Type[Any] = cast(Type[Any], name_or_model)
            table_name = get_sqlmodel_table_name(sqlmodel_class)
            # Validate table name format
            quote_identifier(table_name, self._db._dialect.quote_char)
            return TableHandle(name=table_name, database=self._db, model=sqlmodel_class)
        # Check if argument is a SQLAlchemy model
        elif is_sqlalchemy_model(name_or_model):
            sa_model_class: Type["DeclarativeBase"] = cast(Type["DeclarativeBase"], name_or_model)
            table_name = get_model_table_name(sa_model_class)
            # Validate table name format
            quote_identifier(table_name, self._db._dialect.quote_char)
            return TableHandle(name=table_name, database=self._db, model=sa_model_class)
        else:
            # Type narrowing: after model checks, this must be str
            table_name = cast(str, name_or_model)
            if not table_name:
                raise ValidationError("Table name cannot be empty")
            # Validate table name format
            quote_identifier(table_name, self._db._dialect.quote_char)
            return TableHandle(name=table_name, database=self._db)

    def insert(
        self,
        table_name: str,
        rows: Union[
            Sequence[Mapping[str, object]],
            "Records",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
    ) -> int:
        """Insert rows into a table.

        Convenience method for inserting data into a table.

        Args:
            table_name: Name of the table to insert into
            rows: Sequence of row dictionaries, Records, pandas DataFrame, polars DataFrame, or polars LazyFrame

        Returns:
            Number of rows inserted

        Raises:
            ValidationError: If table name is invalid or rows are empty
        """
        from .mutations import insert_rows

        handle = self.table(table_name)
        return insert_rows(handle, rows)
