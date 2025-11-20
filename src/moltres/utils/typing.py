"""Typing helpers shared across the project."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SupportsAlias(Protocol):
    alias: str
