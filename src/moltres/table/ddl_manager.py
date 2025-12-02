"""Database DDL operations manager.

This module handles Data Definition Language (DDL) operations like creating and dropping tables and indexes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Type, Union

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase
    from .actions import (
        CreateIndexOperation,
        CreateTableOperation,
        DropIndexOperation,
        DropTableOperation,
    )
    from .schema import CheckConstraint, ForeignKeyConstraint, UniqueConstraint
    from .table import Database
    from ..table.schema import ColumnDef


class DDLManager:
    """Handles DDL operations for Database."""

    def __init__(self, database: "Database"):
        """Initialize DDL manager with a Database.

        Args:
            database: The Database instance to manage
        """
        self._db = database

    def create_table(
        self,
        name_or_model: Union[str, Type["DeclarativeBase"]],
        columns: Optional[Sequence["ColumnDef"]] = None,
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
        constraints: Optional[
            Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]]
        ] = None,
    ) -> "CreateTableOperation":
        """Create a lazy create table operation.

        Args:
            name_or_model: Name of the table to create, or SQLAlchemy model class
            columns: Sequence of ColumnDef objects defining the table schema (required if name_or_model is str)
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)
            constraints: Optional sequence of constraint objects (UniqueConstraint, CheckConstraint, ForeignKeyConstraint).
                        Ignored if model_class is provided (constraints are extracted from model).

        Returns:
            CreateTableOperation that executes on collect()
        """
        from .actions import CreateTableOperation
        from .batch import get_active_batch
        from .table_operations_helpers import build_create_table_params

        params = build_create_table_params(
            name_or_model,
            columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
            constraints=constraints,
        )

        op = CreateTableOperation(
            database=self._db,
            name=params.name,
            columns=params.columns,
            if_not_exists=params.if_not_exists,
            temporary=params.temporary,
            constraints=params.constraints,
            model=params.model,
        )

        # Add to active batch if one exists
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def drop_table(self, name: str, *, if_exists: bool = True) -> "DropTableOperation":
        """Create a lazy drop table operation.

        Args:
            name: Name of the table to drop
            if_exists: If True, don't error if table doesn't exist (default: True)

        Returns:
            DropTableOperation that executes on collect()
        """
        from .actions import DropTableOperation
        from .batch import get_active_batch

        op = DropTableOperation(database=self._db, name=name, if_exists=if_exists)
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def create_index(
        self,
        name: str,
        table: str,
        columns: Union[str, Sequence[str]],
        *,
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> "CreateIndexOperation":
        """Create a lazy create index operation.

        Args:
            name: Name of the index to create
            table: Name of the table to create the index on
            columns: Column name(s) to index (single string or sequence)
            unique: If True, create a UNIQUE index (default: False)
            if_not_exists: If True, don't error if index already exists (default: True)

        Returns:
            CreateIndexOperation that executes on collect()
        """
        from .actions import CreateIndexOperation
        from .batch import get_active_batch

        op = CreateIndexOperation(
            database=self._db,
            name=name,
            table_name=table,
            columns=columns,
            unique=unique,
            if_not_exists=if_not_exists,
        )
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def drop_index(
        self,
        name: str,
        table: Optional[str] = None,
        *,
        if_exists: bool = True,
    ) -> "DropIndexOperation":
        """Create a lazy drop index operation.

        Args:
            name: Name of the index to drop
            table: Optional table name (required for some dialects like MySQL)
            if_exists: If True, don't error if index doesn't exist (default: True)

        Returns:
            DropIndexOperation that executes on collect()
        """
        from .actions import DropIndexOperation
        from .batch import get_active_batch

        op = DropIndexOperation(database=self._db, name=name, table_name=table, if_exists=if_exists)
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op
