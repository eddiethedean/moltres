"""Helper utilities for SQL generation."""

from __future__ import annotations

from typing import Iterable


def comma_separated(values: Iterable[str]) -> str:
    return ", ".join(values)


def quote_identifier(identifier: str, quote_char: str = '"') -> str:
    parts = identifier.split(".")
    quoted = [f"{quote_char}{part}{quote_char}" for part in parts if part]
    return ".".join(quoted) if quoted else identifier


def format_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    raise TypeError(f"Unsupported literal type: {type(value)!r}")
