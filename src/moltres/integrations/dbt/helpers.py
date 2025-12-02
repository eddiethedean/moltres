"""dbt helper functions for Moltres.

This module provides helper functions for referencing dbt models, sources,
and variables in Moltres DataFrames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from moltres.dataframe.core.dataframe import DataFrame
    from moltres.table.table import Database

try:
    import dbt

    DBT_AVAILABLE = True
except ImportError:
    DBT_AVAILABLE = False
    # Create stubs for type checking
    dbt: Any = None  # type: ignore[no-redef]


def moltres_ref(dbt: Any, model_name: str, db: Optional["Database"] = None) -> "DataFrame":
    """Reference a dbt model as a Moltres :class:`DataFrame`.

    Args:
        dbt: dbt context object (passed to model() function)
        model_name: Name of the dbt model to reference
        db: Optional :class:`Database` instance (will use get_moltres_connection if not provided)

    Returns:
        :class:`DataFrame` referencing the dbt model table

    Example:
        >>> def model(dbt, session):
        ...     from moltres.integrations.dbt import moltres_ref, get_moltres_connection
        ...     db = get_moltres_connection(dbt.config)
        ...     users = moltres_ref(dbt, "users", db)
        ...     orders = moltres_ref(dbt, "orders", db)
        ...     df = users.join(orders, on="user_id")
        ...     return df.collect()
    """
    if not DBT_AVAILABLE:
        raise ImportError(
            "dbt-core is required for dbt integration. Install with: pip install dbt-core"
        )

    from .adapter import get_moltres_connection

    if db is None:
        db = get_moltres_connection(dbt.config)

    # Get the model table name from dbt
    # dbt typically uses the model name as the table name, but can be overridden
    table_name = _get_model_table_name(dbt, model_name)

    return db.table(table_name).select()


def moltres_source(
    dbt: Any, source_name: str, table_name: str, db: Optional["Database"] = None
) -> "DataFrame":
    """Reference a dbt source as a Moltres :class:`DataFrame`.

    Args:
        dbt: dbt context object (passed to model() function)
        source_name: Name of the dbt source
        table_name: Name of the table in the source
        db: Optional :class:`Database` instance (will use get_moltres_connection if not provided)

    Returns:
        :class:`DataFrame` referencing the dbt source table

    Example:
        >>> def model(dbt, session):
        ...     from moltres.integrations.dbt import moltres_source, get_moltres_connection
        ...     db = get_moltres_connection(dbt.config)
        ...     raw_users = moltres_source(dbt, "raw", "users", db)
        ...     return raw_users.collect()
    """
    if not DBT_AVAILABLE:
        raise ImportError(
            "dbt-core is required for dbt integration. Install with: pip install dbt-core"
        )

    from .adapter import get_moltres_connection

    if db is None:
        db = get_moltres_connection(dbt.config)

    # Get the source table name from dbt
    # dbt sources are typically prefixed with the source schema
    source_table_name = _get_source_table_name(dbt, source_name, table_name)

    return db.table(source_table_name).select()


def moltres_var(dbt: Any, var_name: str, default: Optional[Any] = None) -> Any:
    """Get a dbt variable value.

    Args:
        dbt: dbt context object (passed to model() function)
        var_name: Name of the dbt variable
        default: Optional default value if variable is not set

    Returns:
        Variable value or default

    Example:
        >>> def model(dbt, session):
        ...     from moltres.integrations.dbt import moltres_var
        ...     min_age = moltres_var(dbt, "min_age", default=18)
        ...     # Use in query
    """
    if not DBT_AVAILABLE:
        raise ImportError(
            "dbt-core is required for dbt integration. Install with: pip install dbt-core"
        )

    # Access dbt variables through dbt.config
    # dbt variables are typically accessed via dbt.config.get("vars", {}).get(var_name)
    vars_dict = getattr(dbt.config, "vars", {}) or {}
    if isinstance(vars_dict, dict):
        return vars_dict.get(var_name, default)

    # Try alternative access method
    if hasattr(dbt.config, "get"):
        return dbt.config.get("vars", {}).get(var_name, default)

    return default


def _get_model_table_name(dbt: Any, model_name: str) -> str:
    """Get the actual table name for a dbt model.

    Args:
        dbt: dbt context object
        model_name: Model name

    Returns:
        Table name

    Note:
        This is a simplified implementation. In practice, dbt models
        might have custom table names or be in different schemas.
    """
    # Try to get from dbt's relation system
    try:
        relation = dbt.ref(model_name)
        if hasattr(relation, "identifier"):
            return str(relation.identifier)
        if hasattr(relation, "name"):
            return str(relation.name)
    except Exception:
        pass

    # Fallback: use model name directly
    # In dbt, models are typically materialized with their model name
    return model_name


def _get_source_table_name(dbt: Any, source_name: str, table_name: str) -> str:
    """Get the actual table name for a dbt source.

    Args:
        dbt: dbt context object
        source_name: Source name
        table_name: Table name in source

    Returns:
        Full table name

    Note:
        This is a simplified implementation. In practice, sources
        might have schema prefixes or custom naming.
    """
    # Try to get from dbt's source system
    try:
        relation = dbt.source(source_name, table_name)
        if hasattr(relation, "identifier"):
            return str(relation.identifier)
        if hasattr(relation, "name"):
            return str(relation.name)
    except Exception:
        pass

    # Fallback: use table name directly
    # Sources are typically in the same schema or prefixed
    return table_name
