"""Typing helpers shared across the project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase
    from ..table.table import Database

# Fill values can be basic Python types (int, float, str, bool, None)
FillValue = Union[int, float, str, bool, None]

# Database union type for functions that accept both sync and async databases
DatabaseType = Union["Database", "AsyncDatabase"]


@runtime_checkable
class SupportsAlias(Protocol):
    alias: str
