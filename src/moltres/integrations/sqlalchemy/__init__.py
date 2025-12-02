"""SQLAlchemy integration helpers for Moltres.

This module provides helper functions for integrating Moltres with existing
SQLAlchemy projects, allowing you to use Moltres DataFrames with existing
SQLAlchemy connections, sessions, and infrastructure.
"""

from __future__ import annotations

from .sync_integration import (
    execute_with_connection,
    execute_with_connection_model,
    execute_with_session,
    execute_with_session_model,
    from_sqlalchemy_select,
    to_sqlalchemy_select,
    with_sqlmodel,
)

__all__ = [
    "execute_with_connection",
    "execute_with_session",
    "to_sqlalchemy_select",
    "from_sqlalchemy_select",
    "with_sqlmodel",
    "execute_with_connection_model",
    "execute_with_session_model",
]
