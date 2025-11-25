"""Async lazy operation classes for mutations and DDL operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Type, Union

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from sqlalchemy.orm import DeclarativeBase
    from ..expressions.column import Column
    from ..io.records import AsyncRecords
    from .schema import (
        ColumnDef,
        UniqueConstraint,
        CheckConstraint,
        ForeignKeyConstraint,
    )
    from .async_table import AsyncDatabase, AsyncTableHandle


@dataclass(frozen=True)
class AsyncInsertMutation:
    """Lazy async insert operation that executes on collect()."""

    handle: "AsyncTableHandle"
    rows: Union[
        Sequence[Mapping[str, object]],
        "AsyncRecords",
        "pd.DataFrame",
        "pl.DataFrame",
        "pl.LazyFrame",
    ]

    async def collect(self) -> int:
        """Execute the insert operation and return number of rows affected.

        Returns:
            Number of rows inserted

        Raises:
            ValidationError: If rows are empty or have inconsistent schemas
            ExecutionError: If SQL execution fails
        """
        from .async_mutations import insert_rows_async

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return await insert_rows_async(self.handle, self.rows, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL INSERT statement as a string
        """
        from ..sql.builders import comma_separated, quote_identifier

        if not self.rows:
            return ""
        if isinstance(self.rows, (list, tuple)) and len(self.rows) > 0:
            first_row = self.rows[0]
        elif hasattr(self.rows, "__getitem__") and hasattr(self.rows, "__len__"):
            # For AsyncRecords or other sequence-like objects
            try:
                row_count = len(self.rows)
                if row_count > 0:
                    first_row = self.rows[0]
                else:
                    return ""
            except (TypeError, AttributeError):
                return ""
        else:
            return ""
        columns = list(first_row.keys())
        if not columns:
            return ""
        table_sql = quote_identifier(self.handle.name, self.handle.database.dialect.quote_char)
        column_sql = comma_separated(
            quote_identifier(col, self.handle.database.dialect.quote_char) for col in columns
        )
        placeholder_sql = comma_separated(f":{col}" for col in columns)
        return f"INSERT INTO {table_sql} ({column_sql}) VALUES ({placeholder_sql})"


@dataclass(frozen=True)
class AsyncUpdateMutation:
    """Lazy async update operation that executes on collect()."""

    handle: "AsyncTableHandle"
    where: "Column"
    values: Mapping[str, object]

    async def collect(self) -> int:
        """Execute the update operation and return number of rows affected.

        Returns:
            Number of rows updated

        Raises:
            ValidationError: If values dictionary is empty
            ExecutionError: If SQL execution fails
        """
        from .async_mutations import update_rows_async

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return await update_rows_async(
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
class AsyncDeleteMutation:
    """Lazy async delete operation that executes on collect()."""

    handle: "AsyncTableHandle"
    where: "Column"

    async def collect(self) -> int:
        """Execute the delete operation and return number of rows affected.

        Returns:
            Number of rows deleted

        Raises:
            ExecutionError: If SQL execution fails
        """
        from .async_mutations import delete_rows_async

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return await delete_rows_async(self.handle, where=self.where, transaction=transaction)

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
class AsyncMergeMutation:
    """Lazy async merge (upsert) operation that executes on collect()."""

    handle: "AsyncTableHandle"
    rows: Union[
        Sequence[Mapping[str, object]],
        "AsyncRecords",
        "pd.DataFrame",
        "pl.DataFrame",
        "pl.LazyFrame",
    ]
    on: Sequence[str]
    when_matched: Optional[Mapping[str, object]] = None
    when_not_matched: Optional[Mapping[str, object]] = None

    async def collect(self) -> int:
        """Execute the merge operation and return number of rows affected.

        Returns:
            Number of rows inserted or updated

        Raises:
            ValidationError: If rows are empty, on columns are invalid, etc.
            ExecutionError: If SQL execution fails
        """
        from .async_mutations import merge_rows_async

        # Check for active transaction
        transaction = self.handle.database.connection_manager.active_transaction
        return await merge_rows_async(
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
        # The actual SQL is generated in merge_rows_async function
        return "MERGE/UPSERT (SQL generation in merge_rows_async)"


@dataclass(frozen=True)
class AsyncCreateTableOperation:
    """Lazy async create table operation that executes on collect()."""

    database: "AsyncDatabase"
    name: str
    columns: Sequence["ColumnDef"]
    if_not_exists: bool = True
    temporary: bool = False
    constraints: Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]] = ()
    model: Optional[Type["DeclarativeBase"]] = None

    async def collect(self) -> "AsyncTableHandle":
        """Execute the create table operation and return AsyncTableHandle.

        Returns:
            AsyncTableHandle for the newly created table

        Raises:
            ExecutionError: If table creation fails
        """
        from .async_table import AsyncTableHandle
        from .schema import TableSchema

        schema = TableSchema(
            name=self.name,
            columns=self.columns,
            if_not_exists=self.if_not_exists,
            temporary=self.temporary,
            constraints=self.constraints,
        )
        from ..sql.ddl import compile_create_table

        # Pass engine to use SQLAlchemy Table API
        # SQLAlchemy Table API can compile with AsyncEngine (it just needs dialect info)
        engine = self.database.connection_manager.engine
        sql = compile_create_table(schema, self.database.dialect, engine=engine)
        # Check for active transaction
        transaction = self.database.connection_manager.active_transaction
        await self.database.executor.execute(sql, transaction=transaction)
        return AsyncTableHandle(name=self.name, database=self.database, model=self.model)

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
        )
        return compile_create_table(schema, self.database.dialect)


@dataclass(frozen=True)
class AsyncDropTableOperation:
    """Lazy async drop table operation that executes on collect()."""

    database: "AsyncDatabase"
    name: str
    if_exists: bool = True

    async def collect(self) -> None:
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
        await self.database.executor.execute(sql, transaction=transaction)

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
class AsyncCreateIndexOperation:
    """Lazy async create index operation that executes on collect()."""

    database: "AsyncDatabase"
    name: str
    table_name: str
    columns: Union[str, Sequence[str]]
    unique: bool = False
    if_not_exists: bool = True

    async def collect(self) -> None:
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
        await self.database.executor.execute(sql, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL CREATE INDEX statement as a string
        """
        from ..sql.ddl import compile_create_index

        from sqlalchemy.ext.asyncio import AsyncEngine

        engine = self.database.connection_manager.engine
        if isinstance(engine, AsyncEngine):
            sync_engine = engine.sync_engine
        else:
            sync_engine = engine
        return compile_create_index(
            self.name,
            self.table_name,
            self.columns,
            unique=self.unique,
            engine=sync_engine,
            if_not_exists=self.if_not_exists,
        )


@dataclass(frozen=True)
class AsyncDropIndexOperation:
    """Lazy async drop index operation that executes on collect()."""

    database: "AsyncDatabase"
    name: str
    table_name: Optional[str] = None
    if_exists: bool = True

    async def collect(self) -> None:
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
        await self.database.executor.execute(sql, transaction=transaction)

    def to_sql(self) -> str:
        """Return the SQL statement that will be executed.

        Returns:
            SQL DROP INDEX statement as a string
        """
        from ..sql.ddl import compile_drop_index

        from sqlalchemy.ext.asyncio import AsyncEngine

        engine = self.database.connection_manager.engine
        if isinstance(engine, AsyncEngine):
            sync_engine = engine.sync_engine
        else:
            sync_engine = engine
        return compile_drop_index(
            self.name,
            table_name=self.table_name,
            engine=sync_engine,
            if_exists=self.if_exists,
        )
