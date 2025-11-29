"""dbt adapter functions for Moltres.

This module provides functions to connect Moltres with dbt's Python model execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from moltres.table.table import Database

try:
    import dbt

    DBT_AVAILABLE = True
except ImportError:
    DBT_AVAILABLE = False
    # Create stubs for type checking
    dbt: Any = None  # type: ignore[no-redef]


def get_moltres_connection(dbt_config: Any, profile_name: Optional[str] = None) -> "Database":
    """Get a Moltres :class:`Database` instance from dbt configuration.

    Args:
        dbt_config: dbt config object (from model() function)
        profile_name: Optional profile name (uses default if not provided)

    Returns:
        :class:`Database` instance configured from dbt connection

    Example:
        >>> def model(dbt, session):
        ...     from moltres.integrations.dbt import get_moltres_connection
        ...     db = get_moltres_connection(dbt.config)
        ...     df = db.table("source_table").select()
        ...     return df.collect()

    Raises:
        ImportError: If dbt is not installed
        ValueError: If connection configuration is invalid
    """
    if not DBT_AVAILABLE:
        raise ImportError(
            "dbt-core is required for dbt integration. Install with: pip install dbt-core"
        )

    from ... import connect  # connect is in moltres, not moltres.integrations

    # Extract connection details from dbt config
    # dbt.config contains profile and target information
    profile = getattr(dbt_config, "profile_name", None) or profile_name or "default"
    target = getattr(dbt_config, "target_name", None) or "default"

    # Get connection string from dbt profile
    # This is a simplified version - real implementation would use dbt's profile system
    try:
        # Try to get connection from dbt's adapter
        # This is a placeholder - actual implementation depends on dbt version
        connection_string = _extract_connection_string(dbt_config, profile, target)
    except Exception as e:
        raise ValueError(f"Could not extract connection string from dbt config: {e}") from e

    return connect(connection_string)


def _extract_connection_string(dbt_config: Any, profile_name: str, target_name: str) -> str:
    """Extract connection string from dbt configuration.

    Args:
        dbt_config: dbt config object
        profile_name: Profile name
        target_name: Target name

    Returns:
        Connection string for Moltres

    Raises:
        ValueError: If connection details cannot be extracted
    """
    # This is a simplified implementation
    # In practice, dbt stores connection info in profiles.yml
    # We need to access it through dbt's adapter system

    # Try to get connection info from config
    if hasattr(dbt_config, "credentials"):
        creds = dbt_config.credentials
        return _build_connection_string_from_credentials(creds)

    # Fallback: try environment variables
    dsn = _get_connection_string_from_env(profile_name, target_name)
    if dsn:
        return dsn

    raise ValueError(
        f"Could not extract connection string. "
        f"Please configure dbt profile '{profile_name}' target '{target_name}' or set "
        f"environment variables."
    )


def _build_connection_string_from_credentials(creds: Any) -> str:
    """Build connection string from dbt credentials.

    Args:
        creds: dbt credentials object

    Returns:
        Connection string
    """
    # Extract common fields
    db_type = getattr(creds, "type", None) or getattr(creds, "adapter", None)
    host = getattr(creds, "host", None)
    port = getattr(creds, "port", None)
    user = getattr(creds, "user", None) or getattr(creds, "username", None)
    password = getattr(creds, "password", None)
    database = getattr(creds, "database", None) or getattr(creds, "dbname", None)
    # schema = getattr(creds, "schema", None)  # Not currently used

    if not db_type:
        raise ValueError("Could not determine database type from credentials")

    # Build connection string based on database type
    if db_type in ("postgres", "postgresql"):
        dsn = "postgresql://"
        if user:
            dsn += user
            if password:
                dsn += f":{password}"
            dsn += "@"
        if host:
            dsn += host
            if port:
                dsn += f":{port}"
        if database:
            dsn += f"/{database}"
        return dsn
    elif db_type in ("mysql", "mariadb"):
        dsn = "mysql+pymysql://"
        if user:
            dsn += user
            if password:
                dsn += f":{password}"
            dsn += "@"
        if host:
            dsn += host
            if port:
                dsn += f":{port or 3306}"
        if database:
            dsn += f"/{database}"
        return dsn
    elif db_type == "sqlite":
        database_path = database or getattr(creds, "path", None)
        if database_path:
            return f"sqlite:///{database_path}"
        raise ValueError("SQLite requires a database path")

    raise ValueError(f"Unsupported database type: {db_type}")


def _get_connection_string_from_env(profile_name: str, target_name: str) -> Optional[str]:
    """Get connection string from environment variables.

    Args:
        profile_name: Profile name
        target_name: Target name

    Returns:
        Connection string or None
    """
    import os

    # Check for standard environment variable
    dsn = os.environ.get("DBT_CONNECTION_STRING")
    if dsn:
        return dsn

    # Check for profile/target specific variables
    key = f"DBT_{profile_name.upper()}_{target_name.upper()}_DSN"
    dsn = os.environ.get(key)
    if dsn:
        return dsn

    return None


def moltres_dbt_adapter(dbt: Any, session: Any = None) -> "Database":
    """Get Moltres :class:`Database` instance from dbt context.

    This is a convenience function that extracts the database connection
    from the dbt context provided to Python models.

    Args:
        dbt: dbt context object (passed to model() function)
        session: Optional session (not used, kept for compatibility)

    Returns:
        :class:`Database` instance

    Example:
        >>> def model(dbt, session):
        ...     from moltres.integrations.dbt import moltres_dbt_adapter
        ...     db = moltres_dbt_adapter(dbt, session)
        ...     df = db.table("source_table").select()
        ...     return df.collect()
    """
    return get_moltres_connection(dbt.config)
