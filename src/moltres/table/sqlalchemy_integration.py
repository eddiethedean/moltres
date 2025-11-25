"""SQLAlchemy ORM model integration for Moltres."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Type, Union

if TYPE_CHECKING:
    from sqlalchemy.orm import DeclarativeBase
    from sqlalchemy import MetaData, Table
    from sqlalchemy.sql.type_api import TypeEngine

from .schema import (
    ColumnDef,
    TableSchema,
    UniqueConstraint,
    CheckConstraint,
    ForeignKeyConstraint,
)


def is_sqlalchemy_model(obj: Any) -> bool:
    """Detect if an object is a SQLAlchemy ORM model class.

    Args:
        obj: Object to check

    Returns:
        True if obj is a SQLAlchemy model class, False otherwise
    """
    try:
        from sqlalchemy.orm import DeclarativeMeta
        from sqlalchemy.inspection import inspect

        # Check if it's a class
        if not isinstance(obj, type):
            return False

        # Check if it has __table__ attribute (SQLAlchemy ORM models have this)
        if hasattr(obj, "__table__"):
            # Try to inspect it - if it's a model, inspect will work
            try:
                inspect(obj)
                return True
            except Exception:
                return False

        # Also check for DeclarativeMeta (older SQLAlchemy versions)
        if isinstance(obj, DeclarativeMeta):
            return True

        return False
    except ImportError:
        return False


def get_model_table_name(model_class: Type) -> str:
    """Extract table name from a SQLAlchemy model class.

    Args:
        model_class: SQLAlchemy model class

    Returns:
        Table name

    Raises:
        ValueError: If model doesn't have a table name
    """
    if hasattr(model_class, "__tablename__"):
        tablename = getattr(model_class, "__tablename__")
        if isinstance(tablename, str):
            return tablename
    if hasattr(model_class, "__table__"):
        # Fallback to __table__.name
        table = getattr(model_class, "__table__")
        if table is not None and hasattr(table, "name"):
            return str(table.name)
    # Fallback to class name (lowercased)
    return model_class.__name__.lower()


def sqlalchemy_type_to_moltres_type(sa_type: "TypeEngine") -> str:
    """Convert SQLAlchemy type to Moltres type name.

    Args:
        sa_type: SQLAlchemy TypeEngine instance

    Returns:
        Moltres type name string
    """
    from sqlalchemy import types as sa_types

    # Get the actual type class (handle wrapped types)
    type_class = type(sa_type)
    type_name = type_class.__name__

    # Handle common SQLAlchemy types
    if type_class == sa_types.Integer:
        return "INTEGER"
    elif type_class == sa_types.BigInteger:
        return "BIGINT"
    elif type_class == sa_types.SmallInteger:
        return "SMALLINT"
    elif type_class == sa_types.String:
        # Check if it has a length
        if hasattr(sa_type, "length") and sa_type.length is not None:
            return f"VARCHAR({sa_type.length})"
        return "TEXT"
    elif type_class == sa_types.Text:
        return "TEXT"
    elif type_class == sa_types.CHAR:
        if hasattr(sa_type, "length") and sa_type.length is not None:
            return f"CHAR({sa_type.length})"
        return "CHAR(1)"
    elif type_class == sa_types.Boolean:
        return "BOOLEAN"
    elif type_class == sa_types.REAL:
        return "REAL"
    elif type_class == sa_types.Float:
        return "FLOAT"
    elif type_class == sa_types.Double:
        return "DOUBLE"
    elif type_class == sa_types.Numeric:
        # Extract precision and scale
        precision = getattr(sa_type, "precision", None) or 10
        scale = getattr(sa_type, "scale", None) or 0
        return f"DECIMAL({precision},{scale})"
    elif type_class == sa_types.Date:
        return "DATE"
    elif type_class == sa_types.Time:
        return "TIME"
    elif type_class == sa_types.DateTime:
        return "DATETIME"
    elif type_class == sa_types.TIMESTAMP:
        return "TIMESTAMP"
    elif type_class == sa_types.Interval:
        return "INTERVAL"
    # Handle dialect-specific types
    elif hasattr(sa_types, "JSON") and type_class == sa_types.JSON:
        return "JSON"
    elif type_name == "UUID":
        return "UUID"
    elif type_name == "JSONB":
        return "JSONB"
    else:
        # Try to get dialect-specific types
        try:
            from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID

            if type_class == JSONB:
                return "JSONB"
            elif type_class == PG_UUID:
                return "UUID"
        except ImportError:
            pass

        # Fallback to TEXT for unknown types
        return "TEXT"


def moltres_type_to_sqlalchemy_type(
    type_name: str,
    nullable: bool = True,
    precision: Optional[int] = None,
    scale: Optional[int] = None,
) -> "TypeEngine":
    """Convert Moltres type name to SQLAlchemy TypeEngine.

    Args:
        type_name: Moltres type name
        nullable: Whether the column is nullable
        precision: Optional precision for DECIMAL/NUMERIC
        scale: Optional scale for DECIMAL/NUMERIC

    Returns:
        SQLAlchemy TypeEngine instance
    """
    from sqlalchemy import types as sa_types

    type_name_upper = type_name.upper()

    # Handle types with parameters (e.g., VARCHAR(100), DECIMAL(10,2))
    if "(" in type_name_upper:
        base_type = type_name_upper.split("(")[0]
        params = type_name_upper.split("(")[1].rstrip(")").split(",")
        if base_type == "VARCHAR":
            length = int(params[0].strip()) if params else None
            return sa_types.String(length=length)
        elif base_type == "CHAR":
            length = int(params[0].strip()) if params else 1
            return sa_types.CHAR(length=length)
        elif base_type in ("DECIMAL", "NUMERIC"):
            prec = int(params[0].strip()) if params else (precision or 10)
            sc = int(params[1].strip()) if len(params) > 1 else (scale or 0)
            return sa_types.Numeric(precision=prec, scale=sc)
        else:
            # Fallback to String for unknown parameterized types
            return sa_types.String()

    # Handle simple types
    if type_name_upper == "INTEGER":
        return sa_types.Integer()
    elif type_name_upper == "BIGINT":
        return sa_types.BigInteger()
    elif type_name_upper == "SMALLINT":
        return sa_types.SmallInteger()
    elif type_name_upper == "TEXT":
        return sa_types.Text()
    elif type_name_upper == "VARCHAR":
        return sa_types.String()
    elif type_name_upper == "CHAR":
        return sa_types.CHAR(length=1)
    elif type_name_upper == "BOOLEAN":
        return sa_types.Boolean()
    elif type_name_upper == "REAL":
        return sa_types.REAL()
    elif type_name_upper == "FLOAT":
        return sa_types.Float()
    elif type_name_upper == "DOUBLE":
        return sa_types.Double()
    elif type_name_upper in ("DECIMAL", "NUMERIC"):
        prec = precision or 10
        sc = scale or 0
        return sa_types.Numeric(precision=prec, scale=scale)
    elif type_name_upper == "DATE":
        return sa_types.Date()
    elif type_name_upper == "TIME":
        return sa_types.Time()
    elif type_name_upper == "DATETIME":
        return sa_types.DateTime()
    elif type_name_upper == "TIMESTAMP":
        return sa_types.TIMESTAMP()
    elif type_name_upper == "INTERVAL":
        return sa_types.Interval()
    elif type_name_upper == "UUID":
        try:
            from sqlalchemy.dialects.postgresql import UUID

            return UUID()
        except ImportError:
            return sa_types.CHAR(length=36)
    elif type_name_upper == "JSON":
        try:
            from sqlalchemy.dialects.postgresql import JSON

            return JSON()
        except ImportError:
            return sa_types.JSON()
    elif type_name_upper == "JSONB":
        try:
            from sqlalchemy.dialects.postgresql import JSONB

            return JSONB()
        except ImportError:
            return sa_types.JSON()
    else:
        # Fallback to String for unknown types
        return sa_types.String()


def extract_foreign_keys(model_class: Type) -> list[ForeignKeyConstraint]:
    """Extract foreign key constraints from a SQLAlchemy model.

    Args:
        model_class: SQLAlchemy model class

    Returns:
        List of ForeignKeyConstraint objects
    """
    from sqlalchemy import inspect as sa_inspect

    fk_constraints: list[ForeignKeyConstraint] = []

    try:
        mapper = sa_inspect(model_class)
        table = mapper.tables[0] if mapper.tables else None

        if table is None:
            return fk_constraints

        # Extract foreign keys from columns
        for column in table.columns:
            for fk in column.foreign_keys:
                # Get referenced table and column
                ref_table = fk.column.table.name
                ref_column = fk.column.name

                # Get ondelete and onupdate actions
                on_delete = fk.ondelete if hasattr(fk, "ondelete") else None
                on_update = fk.onupdate if hasattr(fk, "onupdate") else None

                # Convert ondelete/onupdate to strings
                on_delete_str = str(on_delete).upper() if on_delete else None
                on_update_str = str(on_update).upper() if on_update else None

                fk_constraints.append(
                    ForeignKeyConstraint(
                        name=None,  # SQLAlchemy doesn't always expose FK constraint names
                        columns=column.name,
                        references_table=ref_table,
                        references_columns=ref_column,
                        on_delete=on_delete_str,
                        on_update=on_update_str,
                    )
                )
    except Exception:
        # If inspection fails, return empty list
        pass

    return fk_constraints


def model_to_schema(model_class: Type["DeclarativeBase"]) -> TableSchema:
    """Convert a SQLAlchemy ORM model to a Moltres TableSchema.

    Args:
        model_class: SQLAlchemy model class

    Returns:
        TableSchema object

    Raises:
        ValueError: If model_class is not a valid SQLAlchemy model
    """
    from sqlalchemy import inspect as sa_inspect

    if not is_sqlalchemy_model(model_class):
        raise ValueError(f"{model_class} is not a valid SQLAlchemy model")

    # Get table name
    table_name = get_model_table_name(model_class)

    # Inspect the model
    mapper = sa_inspect(model_class)
    table = mapper.tables[0] if mapper.tables else None

    if table is None:
        raise ValueError(f"Model {model_class} does not have a table definition")

    # Convert columns
    columns: list[ColumnDef] = []
    for column in table.columns:
        # Get type name
        type_name = sqlalchemy_type_to_moltres_type(column.type)

        # Extract precision and scale for DECIMAL types
        precision = None
        scale = None
        if "DECIMAL" in type_name.upper() or "NUMERIC" in type_name.upper():
            if hasattr(column.type, "precision"):
                precision = column.type.precision
            if hasattr(column.type, "scale"):
                scale = column.type.scale

        # Get default value
        default = None
        if column.server_default is not None:
            # Extract default value from server_default
            if hasattr(column.server_default, "arg"):
                default = column.server_default.arg
            else:
                default = str(column.server_default)

        columns.append(
            ColumnDef(
                name=column.name,
                type_name=type_name,
                nullable=column.nullable,
                default=default,
                primary_key=column.primary_key,
                precision=precision,
                scale=scale,
            )
        )

    # Extract constraints
    constraints: list[Union[UniqueConstraint, CheckConstraint, ForeignKeyConstraint]] = []

    # Extract unique constraints from __table_args__
    if hasattr(model_class, "__table_args__"):
        table_args = model_class.__table_args__
        if isinstance(table_args, tuple):
            for arg in table_args:
                if isinstance(arg, type) and hasattr(arg, "__name__"):
                    # Skip type annotations
                    continue
                if hasattr(arg, "columns"):
                    # UniqueConstraint
                    from sqlalchemy import UniqueConstraint as SAUniqueConstraint

                    if isinstance(arg, SAUniqueConstraint):
                        cols = list(arg.columns.keys()) if hasattr(arg.columns, "keys") else []
                        if not cols:
                            # Try to get column names from the constraint
                            cols = (
                                [col.name for col in arg.columns]
                                if hasattr(arg.columns, "__iter__")
                                else []
                            )
                        # Handle SQLAlchemy's _NoneName type for unnamed constraints
                        unique_constraint_name: Optional[str] = None
                        if arg.name is not None:
                            name_str = str(arg.name)
                            if name_str and not name_str.startswith("_"):
                                unique_constraint_name = name_str
                        constraints.append(
                            UniqueConstraint(name=unique_constraint_name, columns=cols)
                        )
                elif hasattr(arg, "sqltext") or hasattr(arg, "sqltext"):
                    # CheckConstraint
                    from sqlalchemy import CheckConstraint as SACheckConstraint

                    if isinstance(arg, SACheckConstraint):
                        expr = str(arg.sqltext) if hasattr(arg, "sqltext") else ""
                        # Handle SQLAlchemy's _NoneName type for unnamed constraints
                        check_constraint_name: Optional[str] = None
                        if arg.name is not None:
                            name_str = str(arg.name)
                            if name_str and not name_str.startswith("_"):
                                check_constraint_name = name_str
                        constraints.append(
                            CheckConstraint(name=check_constraint_name, expression=expr)
                        )

    # Extract foreign keys
    fk_constraints = extract_foreign_keys(model_class)
    constraints.extend(fk_constraints)

    return TableSchema(
        name=table_name,
        columns=columns,
        if_not_exists=True,
        temporary=False,
        constraints=constraints,
    )


def schema_to_table(schema: TableSchema, metadata: "MetaData") -> "Table":
    """Convert a Moltres TableSchema to a SQLAlchemy Table object.

    Args:
        schema: TableSchema to convert
        metadata: SQLAlchemy MetaData object to attach the table to

    Returns:
        SQLAlchemy Table object
    """
    from sqlalchemy import Table as SATable

    # Convert columns
    sa_columns = []
    for col_def in schema.columns:
        sa_type = moltres_type_to_sqlalchemy_type(
            col_def.type_name,
            nullable=col_def.nullable,
            precision=col_def.precision,
            scale=col_def.scale,
        )

        from sqlalchemy import Column

        column_kwargs: dict[str, Any] = {
            "nullable": col_def.nullable,
            "primary_key": col_def.primary_key,
        }

        if col_def.default is not None:
            from sqlalchemy import text

            column_kwargs["server_default"] = text(str(col_def.default))

        sa_columns.append(Column(col_def.name, sa_type, **column_kwargs))

    # Convert constraints
    sa_constraints: list[Any] = []
    for constraint in schema.constraints:
        if isinstance(constraint, UniqueConstraint):
            from sqlalchemy import UniqueConstraint as SAUniqueConstraint

            cols = (
                list(constraint.columns)
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else [constraint.columns]
            )
            sa_constraints.append(SAUniqueConstraint(*cols, name=constraint.name))
        elif isinstance(constraint, CheckConstraint):
            from sqlalchemy import CheckConstraint as SACheckConstraint
            from sqlalchemy import text

            sa_constraints.append(
                SACheckConstraint(text(constraint.expression), name=constraint.name)
            )
        elif isinstance(constraint, ForeignKeyConstraint):
            from sqlalchemy import ForeignKey

            cols = (
                list(constraint.columns)
                if isinstance(constraint.columns, Sequence)
                and not isinstance(constraint.columns, str)
                else [constraint.columns]
            )
            ref_cols = (
                list(constraint.references_columns)
                if isinstance(constraint.references_columns, Sequence)
                and not isinstance(constraint.references_columns, str)
                else [constraint.references_columns]
            )
            # Create ForeignKey objects for each column
            refs = [f"{constraint.references_table}.{ref_col}" for ref_col in ref_cols]
            for i, col in enumerate(cols):
                fk = ForeignKey(
                    refs[i] if i < len(refs) else refs[0],
                    ondelete=constraint.on_delete,
                    onupdate=constraint.on_update,
                )
                # Find the column and add FK to it
                for sa_col in sa_columns:
                    if sa_col.name == col:
                        sa_col.foreign_keys.add(fk)
                        break

    # Create table
    table = SATable(
        schema.name,
        metadata,
        *sa_columns,
        *sa_constraints,
    )

    return table
