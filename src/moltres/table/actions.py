"""Lazy operation classes for mutations and DDL operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Type, Union

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from sqlalchemy.orm import DeclarativeBase
    from ..expressions.column import Column
    from ..io.records import Records
    from .schema import (
        ColumnDef,
        UniqueConstraint,
        CheckConstraint,
        ForeignKeyConstraint,
    )
    from .table import Database, TableHandle


@dataclass(frozen=True)
class InsertMutation:
    """Lazy insert operation that executes on collect()."""

    handle: "TableHandle"
    rows: Union[
        Sequence[Mapping[str, object]], "Records", "pd.DataFrame", "pl.DataFrame", "pl.LazyFrame"
    ]

    def collect(self) -> int:
        """Execute the insert operation and return number of rows affected.

        Returns:
            Number of rows inserted

        Raises:
            ValidationError: If rows are empty or have inconsistent schemas
            ExecutionError: If SQL execution fails
        """
        from .mutations import insert_rows

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return insert_rows(self.handle, self.rows, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL INSERT statement as a string
        """
        from ..sql.builders import comma_separated, quote_identifier

        if not self.rows:
            return ""
        columns = list(self.rows[0].keys())
        if not columns:
            return ""
        table_sql = quote_identifier(self.handle.name, self.handle.database.dialect.quote_char)
        column_sql = comma_separated(
            quote_identifier(col, self.handle.database.dialect.quote_char) for col in columns
        )
        placeholder_sql = comma_separated(f":{col}" for col in columns)
        return f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"


@dataclass(frozen=True)
class UpdateMutation:
    """Lazy update operation that executes on collect()."""

    handle: "TableHandle"
    where: "Column"
    values: Mapping[str, object]

    def collect(self) -> int:
        """Execute the update operation and return number of rows affected.

        Returns:
            Number of rows updated

        Raises:
            ValidationError: If values dictionary is empty
            ExecutionError: If SQL execution fails
        """
        from .mutations import update_rows

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return update_rows(
            self.handle, where=self.where, values=self.values, transaction=transaction
        )

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL UPDATE statement as a string
        """
        from ..sql.builders import quote_identifier
        from ..sql.compiler import ExpressionCompiler

        if not self.values:
            return ""
        assignments: list[str] = []
        quote = self.handle.database.dialect.quote_char
        for column in self.values.keys():
            assignments.append(f"{quote_identifier(column, quote)} = :val_{len(assignments)}")
        compiler = ExpressionCompiler(self.handle.database.dialect)
        condition_sql = compiler.emit(self.where)
        table_sql = quote_identifier(self.handle.name, quote)
        return f"UPDATE {table_sql} SET {', '.join(assignments)} WHERE {condition_sql}"


@dataclass(frozen=True)
class DeleteMutation:
    """Lazy delete operation that executes on collect()."""

    handle: "TableHandle"
    where: "Column"

    def collect(self) -> int:
        """Execute the delete operation and return number of rows affected.

        Returns:
            Number of rows deleted

        Raises:
            ExecutionError: If SQL execution fails
        """
        from .mutations import delete_rows

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return delete_rows(self.handle, where=self.where, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL DELETE statement as a string
        """
        from ..sql.builders import quote_identifier
        from ..sql.compiler import ExpressionCompiler

        compiler = ExpressionCompiler(self.handle.database.dialect)
        condition_sql = compiler.emit(self.where)
        table_sql = quote_identifier(self.handle.name, self.handle.database.dialect.quote_char)
        return f"DELETE FROM {table_sql} WHERE {condition_sql}"


@dataclass(frozen=True)
class MergeMutation:
    """Lazy merge (upsert) operation that executes on collect()."""

    handle: "TableHandle"
    rows: Union[
        Sequence[Mapping[str, object]], "Records", "pd.DataFrame", "pl.DataFrame", "pl.LazyFrame"
    ]
    on: Sequence[str]
    when_matched: Optional[Mapping[str, object]] = None
    when_not_matched: Optional[Mapping[str, object]] = None

    def collect(self) -> int:
        """Execute the merge operation and return number of rows affected.

        Returns:
            Number of rows inserted or updated

        Raises:
            ValidationError: If rows are empty, on columns are invalid, etc.
            ExecutionError: If SQL execution fails
        """
        from .mutations import merge_rows

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return merge_rows(
            self.handle,
            self.rows,
            on=self.on,
            when_matched=self.when_matched,
            when_not_matched=self.when_not_matched,
            transaction=transaction,
        )

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL MERGE/UPSERT statement as a string
        """
        # This is complex SQL generation, so we'll just return a placeholder
        # The actual SQL is generated in merge_rows function
        return "MERGE/UPSERT (SQL generation in merge_rows)"


@dataclass(frozen=True)
class CreateTableOperation:
    """Lazy create table operation that executes on collect()."""

    database: "Database"
    name: str
    columns: Sequence["ColumnDef"]
    if_not_exists: bool = True
    temporary: bool = False
    constraints: Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]] = ()
    model: Optional[Type["DeclarativeBase"]] = None

    def collect(self) -> "TableHandle":
        """Execute the create table operation and return :class:`TableHandle`.

        Returns:
            :class:`TableHandle` for the newly created table

        Raises:
            ExecutionError: If table creation fails
        """
        from .schema import TableSchema
        from .table import TableHandle

        schema = TableSchema(
            name=self.name,
            columns=self.columns,
            if_not_exists=self.if_not_exists,
            temporary=self.temporary,
            constraints=self.constraints,
        )
        from ..sql.ddl import compile_create_table

        # Pass engine to use SQLAlchemy Table API
        engine = self.database.connection_manager.engine
        sql = compile_create_table(schema, self.database.dialect, engine=engine)
        # Check for active transaction
        transaction = self.database.connection_manager.active_transaction
        self.database.executor.execute(sql, transaction=transaction)
        return TableHandle(name=self.name, database=self.database, model=self.model)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL CREATE TABLE statement as a string
        """
        from .schema import TableSchema
        from ..sql.ddl import compile_create_table

        schema = TableSchema(
            name=self.name,
            columns=self.columns,
            if_not_exists=self.if_not_exists,
            temporary=self.temporary,
            constraints=self.constraints,
        )
        # Pass engine to use SQLAlchemy Table API
        engine = self.database.connection_manager.engine
        return compile_create_table(schema, self.database.dialect, engine=engine)


@dataclass(frozen=True)
class DropTableOperation:
    """Lazy drop table operation that executes on collect()."""

    database: "Database"
    name: str
    if_exists: bool = True

    def collect(self) -> None:
        """Execute the drop table operation.

        Raises:
            ValidationError: If table name is invalid
            ExecutionError: If table dropping fails (when if_exists=False and table doesn't exist)
        """
        from ..sql.ddl import compile_drop_table

        engine = self.database.connection_manager.engine
        sql = compile_drop_table(
            self.name, self.database.dialect, if_exists=self.if_exists, engine=engine
        )
        # Check for active transaction
        transaction = self.database.connection_manager.active_transaction
        self.database.executor.execute(sql, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL DROP TABLE statement as a string
        """
        from ..sql.ddl import compile_drop_table

        engine = self.database.connection_manager.engine
        return compile_drop_table(
            self.name, self.database.dialect, if_exists=self.if_exists, engine=engine
        )


@dataclass(frozen=True)
class CreateIndexOperation:
    """Lazy create index operation that executes on collect()."""

    database: "Database"
    name: str
    table_name: str
    columns: Union[str, Sequence[str]]
    unique: bool = False
    if_not_exists: bool = True

    def collect(self) -> None:
        """Execute the create index operation.

        Raises:
            ExecutionError: If index creation fails
        """
        from ..sql.ddl import compile_create_index

        engine = self.database.connection_manager.engine
        sql = compile_create_index(
            self.name,
            self.table_name,
            self.columns,
            unique=self.unique,
            engine=engine,
            if_not_exists=self.if_not_exists,
        )
        # Check for active transaction
        transaction = self.database.connection_manager.active_transaction
        self.database.executor.execute(sql, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL CREATE INDEX statement as a string
        """
        from ..sql.ddl import compile_create_index

        engine = self.database.connection_manager.engine
        return compile_create_index(
            self.name,
            self.table_name,
            self.columns,
            unique=self.unique,
            engine=engine,
            if_not_exists=self.if_not_exists,
        )


@dataclass(frozen=True)
class DropIndexOperation:
    """Lazy drop index operation that executes on collect()."""

    database: "Database"
    name: str
    table_name: Optional[str] = None
    if_exists: bool = True

    def collect(self) -> None:
        """Execute the drop index operation.

        Raises:
            ExecutionError: If index dropping fails (when if_exists=False and index doesn't exist)
        """
        from ..sql.ddl import compile_drop_index

        engine = self.database.connection_manager.engine
        sql = compile_drop_index(
            self.name,
            table_name=self.table_name,
            engine=engine,
            if_exists=self.if_exists,
        )
        # Check for active transaction
        transaction = self.database.connection_manager.active_transaction
        self.database.executor.execute(sql, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL DROP INDEX statement as a string
        """
        from ..sql.ddl import compile_drop_index

        engine = self.database.connection_manager.engine
        return compile_drop_index(
            self.name,
            table_name=self.table_name,
            engine=engine,
            if_exists=self.if_exists,
        )
