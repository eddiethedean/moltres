"""DDL (Data Definition Language) SQL generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    UniqueConstraint,
    CheckConstraint,
    ForeignKeyConstraint,
    Index,
    text,
)
from sqlalchemy.sql import Select
from sqlalchemy.sql.ddl import CreateTable, DropTable, CreateIndex, DropIndex

try:
    from sqlalchemy.ext.asyncio import AsyncEngine
except ImportError:
    AsyncEngine = None  # type: ignore[assignment, misc]

from ..engine.dialects import DialectSpec  # noqa: E402
from ..table.schema import (  # noqa: E402
    TableSchema,
    ColumnDef,
    UniqueConstraint as MoltresUniqueConstraint,
    CheckConstraint as MoltresCheckConstraint,
    ForeignKeyConstraint as MoltresForeignKeyConstraint,
)
from .builders import format_literal  # noqa: E402

if TYPE_CHECKING:
    pass


def compile_create_table(
    schema: TableSchema,
    dialect: DialectSpec,
    engine: Optional[Any] = None,
) -> str:
    """Compile a TableSchema into a CREATE TABLE statement using SQLAlchemy.

    Args:
        schema: TableSchema to compile
        dialect: Dialect specification
        engine: Optional SQLAlchemy Engine. If provided, uses the engine directly.
                If None, creates a temporary engine from the dialect for compilation.

    Returns:
        SQL CREATE TABLE statement as a string
    """
    # Always use SQLAlchemy Table API
    if engine is None:
        # Create a temporary engine from dialect for compilation
        from sqlalchemy import create_engine

        # Create a dummy DSN based on dialect name
        if dialect.name == "sqlite":
            dsn = "sqlite:///:memory:"
        elif dialect.name == "postgresql":
            dsn = "postgresql://localhost/dummy"
        elif dialect.name == "mysql":
            dsn = "mysql://localhost/dummy"
        elif dialect.name == "duckdb":
            dsn = "duckdb:///:memory:"
        else:
            dsn = "sqlite:///:memory:"  # Default fallback
        engine = create_engine(dsn, future=True)

    return _compile_create_table_sqlalchemy(schema, engine)


def _compile_create_table_sqlalchemy(schema: TableSchema, engine: Any) -> str:
    """Compile TableSchema to SQL using SQLAlchemy Table API."""
    metadata = MetaData()

    # First, collect foreign key constraints to apply to columns
    fk_constraints_by_column: dict[str, MoltresForeignKeyConstraint] = {}
    other_constraints: list[
        Union[MoltresUniqueConstraint, MoltresCheckConstraint, MoltresForeignKeyConstraint]
    ] = []

    for constraint in schema.constraints:
        if isinstance(constraint, MoltresForeignKeyConstraint):
            # Normalize columns to sequences
            cols = (
                constraint.columns
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else (constraint.columns,)
            )
            # Store FK constraint for each column (for single-column FKs)
            if len(cols) == 1:
                fk_constraints_by_column[cols[0]] = constraint
            else:
                # Multi-column FK - we'll handle as ForeignKeyConstraint
                other_constraints.append(constraint)
        else:
            other_constraints.append(constraint)

    # Convert columns to SQLAlchemy Column objects, adding ForeignKey if needed
    sa_columns = []
    for col_def in schema.columns:
        sa_col = _column_def_to_sqlalchemy(col_def, engine.dialect)
        # Add ForeignKey if this column has a single-column FK constraint
        if col_def.name in fk_constraints_by_column:
            fk_constraint = fk_constraints_by_column[col_def.name]
            ref_cols = (
                fk_constraint.references_columns
                if isinstance(fk_constraint.references_columns, Sequence)
                and not isinstance(fk_constraint.references_columns, str)
                else (fk_constraint.references_columns,)
            )
            from sqlalchemy import ForeignKey

            ref_str = f"{fk_constraint.references_table}.{ref_cols[0]}"
            # Create new column with ForeignKey
            # Use use_alter=True to defer constraint resolution (allows string-based FK)
            sa_col = Column(
                sa_col.name,
                sa_col.type,
                ForeignKey(
                    ref_str,
                    ondelete=fk_constraint.on_delete,
                    onupdate=fk_constraint.on_update,
                    use_alter=True,  # Defer constraint resolution
                ),
                nullable=sa_col.nullable,
                primary_key=sa_col.primary_key,
                server_default=sa_col.server_default,
            )
        sa_columns.append(sa_col)

    # Convert remaining constraints to SQLAlchemy constraint objects
    sa_constraints: list[Union[UniqueConstraint, CheckConstraint, ForeignKeyConstraint]] = []
    # Handle UniqueConstraint, CheckConstraint, and multi-column ForeignKeyConstraint
    for constraint in other_constraints:
        if isinstance(constraint, MoltresUniqueConstraint):
            # Normalize columns to tuple
            cols = (
                constraint.columns
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else (constraint.columns,)
            )
            sa_constraints.append(UniqueConstraint(*cols, name=constraint.name))
        elif isinstance(constraint, MoltresCheckConstraint):
            sa_constraints.append(
                CheckConstraint(text(constraint.expression), name=constraint.name)
            )
        elif isinstance(constraint, MoltresForeignKeyConstraint):
            # Multi-column foreign key constraint
            # Normalize columns to sequences
            cols = (
                constraint.columns
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else (constraint.columns,)
            )
            ref_cols = (
                constraint.references_columns
                if isinstance(constraint.references_columns, Sequence)
                and not isinstance(constraint.references_columns, str)
                else (constraint.references_columns,)
            )
            # Build reference strings for each column: "table.column"
            refs = [f"{constraint.references_table}.{ref_col}" for ref_col in ref_cols]
            # Use ForeignKeyConstraint for multi-column foreign keys
            # Note: This may fail if referenced table isn't in same MetaData, but we'll try
            sa_constraints.append(
                ForeignKeyConstraint(
                    list(cols),
                    refs,
                    name=constraint.name,
                    ondelete=constraint.on_delete,
                    onupdate=constraint.on_update,
                )
            )

    # Build SQLAlchemy Table
    # Note: SQLAlchemy Table doesn't directly support TEMPORARY flag in constructor
    # We'll handle it in the SQL generation
    table = Table(
        schema.name,
        metadata,
        *sa_columns,
        *sa_constraints,
    )

    # Generate CREATE TABLE SQL
    create_stmt = CreateTable(table)
    try:
        compiled = create_stmt.compile(engine)
        sql = str(compiled)
    except Exception as e:
        # If compilation fails (e.g., foreign key references table not in MetaData),
        # fall back to raw SQL generation for the entire table
        # This can happen when foreign keys reference tables created separately
        from sqlalchemy.exc import NoReferencedTableError

        if isinstance(e, NoReferencedTableError) or (
            hasattr(e, "__cause__") and isinstance(e.__cause__, NoReferencedTableError)
        ):
            # Manually construct SQL using SQLAlchemy but handle FKs as strings
            return _compile_create_table_with_string_fks(schema, engine)
        raise

    # Handle TEMPORARY - SQLAlchemy doesn't support this directly in Table constructor
    if schema.temporary:
        sql = sql.replace("CREATE TABLE", "CREATE TEMPORARY TABLE", 1)

    # Add IF NOT EXISTS if requested (SQLAlchemy doesn't support this natively)
    if schema.if_not_exists and "IF NOT EXISTS" not in sql.upper():
        # Insert IF NOT EXISTS after CREATE [TEMPORARY] TABLE
        if schema.temporary:
            sql = sql.replace("CREATE TEMPORARY TABLE", "CREATE TEMPORARY TABLE IF NOT EXISTS", 1)
        else:
            sql = sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)

    return sql


def _compile_create_table_with_string_fks(schema: TableSchema, engine: Any) -> str:
    """Compile TableSchema to SQL using SQLAlchemy but with string-based foreign keys.

    This is used when foreign keys reference tables not in the same MetaData.
    We use SQLAlchemy DDL objects but handle FKs manually as strings.
    """
    metadata = MetaData()

    # Convert columns to SQLAlchemy Column objects
    sa_columns = []
    for col_def in schema.columns:
        sa_col = _column_def_to_sqlalchemy(col_def, engine.dialect)
        sa_columns.append(sa_col)

    # Convert constraints to SQLAlchemy constraint objects (except FKs which we'll handle manually)
    sa_constraints: list[Union[UniqueConstraint, CheckConstraint]] = []
    fk_constraints: list[MoltresForeignKeyConstraint] = []

    for constraint in schema.constraints:
        if isinstance(constraint, MoltresUniqueConstraint):
            cols = (
                constraint.columns
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else (constraint.columns,)
            )
            sa_constraints.append(UniqueConstraint(*cols, name=constraint.name))
        elif isinstance(constraint, MoltresCheckConstraint):
            sa_constraints.append(
                CheckConstraint(text(constraint.expression), name=constraint.name)
            )
        elif isinstance(constraint, MoltresForeignKeyConstraint):
            fk_constraints.append(constraint)

    # Build SQLAlchemy Table (without FKs for now)
    table = Table(schema.name, metadata, *sa_columns, *sa_constraints)

    # Generate base CREATE TABLE SQL
    create_stmt = CreateTable(table)
    compiled = create_stmt.compile(engine)
    sql = str(compiled)

    # Manually add FOREIGN KEY constraints as SQL strings
    if fk_constraints:
        # Extract the column definitions and constraints from the SQL
        # We need to insert FK constraints before the closing parenthesis
        from .builders import quote_identifier

        # Get quote character from SQLAlchemy's identifier preparer
        # SQLAlchemy uses double quotes for SQLite, backticks for MySQL, etc.
        preparer = engine.dialect.identifier_preparer
        # Use SQLAlchemy's quote method or default to double quotes
        quote = '"'  # Default
        if hasattr(preparer, "_initial_quote"):
            quote = preparer._initial_quote
        elif hasattr(preparer, "quote"):
            # Test quote to determine quote character
            test_quoted = preparer.quote("test")
            if test_quoted.startswith('"'):
                quote = '"'
            elif test_quoted.startswith("`"):
                quote = "`"
            elif test_quoted.startswith("'"):
                quote = "'"

        fk_sql_parts = []
        for constraint in fk_constraints:
            cols = (
                constraint.columns
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else (constraint.columns,)
            )
            ref_cols = (
                constraint.references_columns
                if isinstance(constraint.references_columns, Sequence)
                and not isinstance(constraint.references_columns, str)
                else (constraint.references_columns,)
            )
            quoted_cols = [quote_identifier(col, quote) for col in cols]
            quoted_ref_cols = [quote_identifier(ref_col, quote) for ref_col in ref_cols]
            quoted_ref_table = quote_identifier(constraint.references_table, quote)

            fk_sql = f"FOREIGN KEY ({', '.join(quoted_cols)}) REFERENCES {quoted_ref_table} ({', '.join(quoted_ref_cols)})"
            if constraint.on_delete:
                fk_sql += f" ON DELETE {constraint.on_delete}"
            if constraint.on_update:
                fk_sql += f" ON UPDATE {constraint.on_update}"

            if constraint.name:
                fk_sql_parts.append(
                    f"CONSTRAINT {quote_identifier(constraint.name, quote)} {fk_sql}"
                )
            else:
                fk_sql_parts.append(fk_sql)

        # Insert FK constraints before the closing parenthesis
        if fk_sql_parts:
            # Find the last ')' before the final ')'
            # SQL format: CREATE TABLE name (col1, col2, CONSTRAINT ...)
            # We want to insert FKs before the final ')'
            last_paren = sql.rfind(")")
            if last_paren > 0:
                # Insert FKs with a comma separator
                fk_sql_str = ", " + ", ".join(fk_sql_parts)
                sql = sql[:last_paren] + fk_sql_str + sql[last_paren:]

    # Handle TEMPORARY
    if schema.temporary:
        sql = sql.replace("CREATE TABLE", "CREATE TEMPORARY TABLE", 1)

    # Add IF NOT EXISTS if requested
    if schema.if_not_exists and "IF NOT EXISTS" not in sql.upper():
        if schema.temporary:
            sql = sql.replace("CREATE TEMPORARY TABLE", "CREATE TEMPORARY TABLE IF NOT EXISTS", 1)
        else:
            sql = sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS", 1)

    return sql


def compile_drop_table(
    table_name: str,
    dialect: DialectSpec,
    if_exists: bool = True,
    engine: Optional[Any] = None,
) -> str:
    """Compile a DROP TABLE statement using SQLAlchemy.

    Args:
        table_name: Name of the table to drop
        dialect: Dialect specification
        if_exists: If True, add IF EXISTS clause
        engine: Optional SQLAlchemy Engine. If None, creates a temporary engine for compilation.

    Returns:
        SQL DROP TABLE statement as a string
    """
    # Always use SQLAlchemy DDL API
    if engine is None:
        from sqlalchemy import create_engine

        if dialect.name == "sqlite":
            dsn = "sqlite:///:memory:"
        elif dialect.name == "postgresql":
            dsn = "postgresql://localhost/dummy"
        elif dialect.name == "mysql":
            dsn = "mysql://localhost/dummy"
        elif dialect.name == "duckdb":
            dsn = "duckdb:///:memory:"
        else:
            dsn = "sqlite:///:memory:"
        engine = create_engine(dsn, future=True)

    metadata = MetaData()
    table = Table(table_name, metadata)
    drop_stmt = DropTable(table, if_exists=if_exists)
    sql = str(drop_stmt.compile(engine))
    return sql


def compile_create_index(
    name: str,
    table_name: str,
    columns: Union[str, Sequence[str]],
    unique: bool = False,
    dialect: Optional[DialectSpec] = None,
    engine: Optional[Any] = None,
    if_not_exists: bool = True,
) -> str:
    """Compile a CREATE INDEX statement using SQLAlchemy.

    Args:
        name: Index name
        table_name: Table name
        columns: :class:`Column` name(s) to index
        unique: If True, create a UNIQUE index
        dialect: Dialect specification (used if engine is None)
        engine: Optional SQLAlchemy Engine. If None, creates a temporary engine for compilation.
        if_not_exists: If True, add IF NOT EXISTS clause

    Returns:
        SQL CREATE INDEX statement as a string
    """
    # Always use SQLAlchemy Index API
    if engine is None:
        from sqlalchemy import create_engine
        from ..engine.dialects import get_dialect

        d = dialect or get_dialect("ansi")
        if d.name == "sqlite":
            dsn = "sqlite:///:memory:"
        elif d.name == "postgresql":
            dsn = "postgresql://localhost/dummy"
        elif d.name == "mysql":
            dsn = "mysql://localhost/dummy"
        else:
            dsn = "sqlite:///:memory:"
        engine = create_engine(dsn, future=True)

    return _compile_create_index_sqlalchemy(
        name, table_name, columns, unique, engine, if_not_exists
    )


def _compile_create_index_sqlalchemy(
    name: str,
    table_name: str,
    columns: Union[str, Sequence[str]],
    unique: bool,
    engine: Any,
    if_not_exists: bool,
) -> str:
    """Compile CREATE INDEX using SQLAlchemy Index API."""
    metadata = MetaData()

    # Normalize columns to list
    if isinstance(columns, str):
        col_list = [columns]
    elif isinstance(columns, Sequence):
        col_list = list(columns)
    else:
        col_list = [columns]

    # Create table with columns for index reference
    # We need actual Column objects for SQLAlchemy Index
    from sqlalchemy import types as sa_types

    table = Table(table_name, metadata)
    # Add columns to the table
    for col_name in col_list:
        table.append_column(Column(col_name, sa_types.String()))  # Type doesn't matter for index

    # Create index using column objects
    index = Index(name, *[table.c[col] for col in col_list], unique=unique)

    # Generate CREATE INDEX SQL
    create_stmt = CreateIndex(index)
    sql = str(create_stmt.compile(engine))

    # Add IF NOT EXISTS if requested
    if if_not_exists and "IF NOT EXISTS" not in sql.upper():
        if unique:
            # For UNIQUE INDEX, replace "CREATE UNIQUE INDEX" with "CREATE UNIQUE INDEX IF NOT EXISTS"
            sql = sql.replace("CREATE UNIQUE INDEX", "CREATE UNIQUE INDEX IF NOT EXISTS", 1)
        else:
            # For regular INDEX, replace "CREATE INDEX" with "CREATE INDEX IF NOT EXISTS"
            sql = sql.replace("CREATE INDEX", "CREATE INDEX IF NOT EXISTS", 1)

    return sql


def compile_drop_index(
    name: str,
    table_name: Optional[str] = None,
    dialect: Optional[DialectSpec] = None,
    engine: Optional[Any] = None,
    if_exists: bool = True,
) -> str:
    """Compile a DROP INDEX statement using SQLAlchemy.

    Args:
        name: Index name
        table_name: Optional table name (required for some dialects like MySQL)
        dialect: Dialect specification (used if engine is None)
        engine: Optional SQLAlchemy Engine. If None, creates a temporary engine for compilation.
        if_exists: If True, add IF EXISTS clause

    Returns:
        SQL DROP INDEX statement as a string
    """
    # Always use SQLAlchemy Index API
    if engine is None:
        from sqlalchemy import create_engine
        from ..engine.dialects import get_dialect

        d = dialect or get_dialect("ansi")
        if d.name == "sqlite":
            dsn = "sqlite:///:memory:"
        elif d.name == "postgresql":
            dsn = "postgresql://localhost/dummy"
        elif d.name == "mysql":
            dsn = "mysql://localhost/dummy"
        else:
            dsn = "sqlite:///:memory:"
        engine = create_engine(dsn, future=True)

    return _compile_drop_index_sqlalchemy(name, table_name, engine, if_exists)


def _compile_drop_index_sqlalchemy(
    name: str,
    table_name: Optional[str],
    engine: Any,
    if_exists: bool,
) -> str:
    """Compile DROP INDEX using SQLAlchemy Index API."""
    metadata = MetaData()
    # Create a minimal table and index for DROP INDEX
    # We need at least one column to create an index
    from sqlalchemy import types as sa_types

    table = Table(table_name or "dummy_table", metadata)
    dummy_col = Column("_dummy", sa_types.Integer())
    table.append_column(dummy_col)
    # Index constructor doesn't accept table=, columns are bound to table automatically
    index = Index(name, dummy_col)

    # Generate DROP INDEX SQL
    drop_stmt = DropIndex(index)
    sql = str(drop_stmt.compile(engine))

    # Add IF EXISTS if requested
    if if_exists and "IF EXISTS" not in sql.upper():
        sql = sql.replace("DROP INDEX", "DROP INDEX IF EXISTS", 1)

    return sql


def compile_insert_select(
    target_table: str,
    select_stmt: Select,
    dialect: DialectSpec,
    columns: Optional[Sequence[str]] = None,
    engine: Optional[Any] = None,
) -> tuple[str, dict[str, Any]]:
    """Compile an INSERT INTO ... SELECT statement using SQLAlchemy.

    Args:
        target_table: Name of target table
        select_stmt: SQLAlchemy Select statement for the SELECT part
        columns: Optional list of column names to insert into
        dialect: SQL dialect specification
        engine: Optional SQLAlchemy Engine. If None, creates a temporary engine for compilation.

    Returns:
        Tuple of (SQL string, parameters dict) for INSERT INTO ... SELECT statement
    """
    # Always use SQLAlchemy Insert API
    if engine is None:
        from sqlalchemy import create_engine

        if dialect.name == "sqlite":
            dsn = "sqlite:///:memory:"
        elif dialect.name == "postgresql":
            dsn = "postgresql://localhost/dummy"
        elif dialect.name == "mysql":
            dsn = "mysql://localhost/dummy"
        elif dialect.name == "duckdb":
            dsn = "duckdb:///:memory:"
        else:
            dsn = "sqlite:///:memory:"
        engine = create_engine(dsn, future=True)

    from sqlalchemy import insert, types as sa_types

    # Create Insert statement from Select
    metadata = MetaData()
    table = Table(target_table, metadata)

    # Add columns to the table if specified (needed for from_select)
    if columns:
        # Add columns to table so SQLAlchemy can reference them
        for col_name in columns:
            table.append_column(
                Column(col_name, sa_types.String())
            )  # Type doesn't matter for compilation
        # Insert with specific columns
        insert_stmt = insert(table).from_select([table.c[col] for col in columns], select_stmt)
    else:
        # Insert all columns (requires matching column count and order)
        # Extract column names from select statement
        col_names = [
            col.name if hasattr(col, "name") else str(col) for col in select_stmt.selected_columns
        ]
        # Add columns to table
        for col_name in col_names:
            table.append_column(Column(col_name, sa_types.String()))
        # Use column objects from table
        insert_stmt = insert(table).from_select([table.c[col] for col in col_names], select_stmt)

    # Compile to SQL - SQLAlchemy will handle parameters from the SELECT statement
    # We need to compile with the actual engine to get proper parameter handling
    compiled = insert_stmt.compile(engine)
    sql = str(compiled)

    # Extract parameters from the compiled SELECT statement
    # The SELECT statement may have bound parameters that need to be passed through
    # SQLAlchemy uses positional parameters for some dialects (like SQLite)
    # We need to handle this properly. For now, if there are parameters,
    # we'll need to execute the statement directly rather than as a string
    # But since we're returning a string, we'll need to inline parameters or handle them separately
    # For simplicity, return empty params - the caller should execute the SQLAlchemy statement directly
    # if it has parameters
    params: dict[str, Any] = {}

    return sql, params


def _column_def_to_sqlalchemy(col_def: ColumnDef, sa_dialect: Any) -> Column:
    """Convert a ColumnDef to a SQLAlchemy :class:`Column` object.

    Args:
        col_def: Moltres ColumnDef
        sa_dialect: SQLAlchemy dialect object

    Returns:
        SQLAlchemy :class:`Column` object
    """
    from sqlalchemy import types as sa_types

    # Map Moltres type names to SQLAlchemy types
    type_name = col_def.type_name.upper()
    sa_type: Any = None

    # Basic types
    if type_name == "INTEGER":
        sa_type = sa_types.Integer()
    elif type_name == "BIGINT":
        sa_type = sa_types.BigInteger()
    elif type_name == "SMALLINT":
        sa_type = sa_types.SmallInteger()
    elif type_name == "TEXT":
        sa_type = sa_types.Text()
    elif type_name == "VARCHAR":
        # MySQL requires length, default to 255
        length = 255 if sa_dialect.name == "mysql" else None
        sa_type = sa_types.String(length=length)
    elif type_name == "CHAR":
        sa_type = sa_types.CHAR(length=1)  # Default length, can be overridden
    elif type_name == "BOOLEAN":
        sa_type = sa_types.Boolean()
    elif type_name == "REAL":
        sa_type = sa_types.REAL()
    elif type_name == "FLOAT":
        sa_type = sa_types.Float()
    elif type_name == "DOUBLE":
        sa_type = sa_types.Double()
    elif type_name == "DECIMAL" or type_name == "NUMERIC":
        precision = col_def.precision or 10
        scale = col_def.scale or 0
        sa_type = sa_types.Numeric(precision=precision, scale=scale)
    elif type_name == "DATE":
        sa_type = sa_types.Date()
    elif type_name == "TIME":
        sa_type = sa_types.Time()
    elif type_name == "TIMESTAMP":
        sa_type = sa_types.TIMESTAMP()
    elif type_name == "DATETIME":
        sa_type = sa_types.DateTime()
    elif type_name == "UUID":
        if sa_dialect.name == "postgresql":
            sa_type = sa_types.UUID()
        elif sa_dialect.name == "mysql":
            sa_type = sa_types.CHAR(length=36)
        else:
            sa_type = sa_types.Text()
    elif type_name == "JSON" or type_name == "JSONB":
        if sa_dialect.name == "postgresql":
            if type_name == "JSONB":
                from sqlalchemy.dialects.postgresql import JSONB

                sa_type = JSONB()
            else:
                from sqlalchemy.dialects.postgresql import JSON

                sa_type = JSON()
        elif sa_dialect.name == "mysql":
            sa_type = sa_types.JSON()
        else:
            sa_type = sa_types.Text()
    elif type_name == "INTERVAL":
        if sa_dialect.name == "postgresql":
            sa_type = sa_types.Interval()
        elif sa_dialect.name == "mysql":
            sa_type = sa_types.Time()
        else:
            sa_type = sa_types.Text()
    else:
        # Fallback: use String for unknown types
        # MySQL requires length, default to 255
        length = 255 if sa_dialect.name == "mysql" else None
        sa_type = sa_types.String(length=length)

    # Build column with properties
    column_kwargs: dict[str, Any] = {
        "nullable": col_def.nullable,
        "primary_key": col_def.primary_key,
    }

    # DuckDB doesn't support SERIAL type - explicitly disable autoincrement
    # to prevent SQLAlchemy from converting INTEGER primary keys to SERIAL
    if sa_dialect.name == "duckdb" and col_def.primary_key and type_name == "INTEGER":
        column_kwargs["autoincrement"] = False

    if col_def.default is not None:
        # Format default value as SQL literal
        default_sql = format_literal(col_def.default)
        column_kwargs["server_default"] = text(default_sql)

    return Column(col_def.name, sa_type, **column_kwargs)
