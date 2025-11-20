"""Schema inspector stubs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ColumnInfo:
    name: str
    type_name: str
