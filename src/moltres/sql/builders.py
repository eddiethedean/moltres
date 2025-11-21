"""Helper utilities for SQL generation."""

from __future__ import annotations

import re
from collections.abc import Iterable

from ..utils.exceptions import ValidationError


def comma_separated(values: Iterable[str]) -> str:
    return ", ".join(values)


def quote_identifier(identifier: str, quote_char: str = '"') -> str:
    """Quote a SQL identifier, validating it for safety.

    Args:
        identifier: The identifier to quote (e.g., "table_name" or "schema.table")
        quote_char: The character to use for quoting (default: double quote)

    Returns:
        The quoted identifier

    Raises:
        ValidationError: If the identifier is empty or contains invalid characters
    """
    if not identifier or not identifier.strip():
        raise ValidationError("SQL identifier cannot be empty")

    # Validate identifier parts (alphanumeric, underscore, and dot for qualified names)
    # SQL identifiers can contain letters, digits, underscores, and dots for schema.table
    # We allow dots for qualified names like "schema.table"
    parts = identifier.split(".")
    for part in parts:
        if not part:
            raise ValidationError(f"SQL identifier contains empty part: {identifier!r}")
        # Check for SQL injection patterns (semicolons, comments, etc.)
        if re.search(r"[;\'\"\\]", part):
            raise ValidationError(
                f"SQL identifier contains invalid characters: {identifier!r}. "
                "Identifiers may only contain letters, digits, underscores, and dots."
            )

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
