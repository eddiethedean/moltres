"""Safe parsing of SQL interval and strptime format strings."""

from __future__ import annotations

import re

from ..utils.exceptions import CompilationError

ALLOWED_INTERVAL_UNITS = frozenset(
    {
        "DAY",
        "DAYS",
        "HOUR",
        "HOURS",
        "MINUTE",
        "MINUTES",
        "MONTH",
        "MONTHS",
        "YEAR",
        "YEARS",
        "WEEK",
        "WEEKS",
        "SECOND",
        "SECONDS",
    }
)

# Normalize plural units to singular for SQL emission
_UNIT_SINGULAR = {
    "DAYS": "DAY",
    "HOURS": "HOUR",
    "MINUTES": "MINUTE",
    "MONTHS": "MONTH",
    "YEARS": "YEAR",
    "WEEKS": "WEEK",
    "SECONDS": "SECOND",
}

_STRPTIME_FORMAT_RE = re.compile(r"^[A-Za-z0-9%\-_.: ]+$")


def parse_interval(interval_str: str) -> tuple[int, str]:
    """Parse and validate an interval string like '1 DAY'.

    Returns:
        (numeric_amount, normalized_unit) e.g. (1, 'DAY')

    Raises:
        CompilationError: If format or content is invalid or unsafe.
    """
    if not isinstance(interval_str, str):
        raise CompilationError(
            f"Expected string interval, got {type(interval_str).__name__}",
        )
    parts = interval_str.split()
    if len(parts) != 2:
        raise CompilationError(
            f"Invalid interval format: {interval_str!r}. Expected 'N UNIT' (e.g. '1 DAY')",
        )
    num_str, unit = parts
    if not num_str.lstrip("-").isdigit():
        raise CompilationError(
            f"Invalid interval magnitude: {num_str!r}. Must be an integer.",
        )
    unit_upper = unit.upper()
    if unit_upper not in ALLOWED_INTERVAL_UNITS:
        raise CompilationError(
            f"Invalid interval unit: {unit!r}. "
            f"Allowed: {', '.join(sorted(ALLOWED_INTERVAL_UNITS))}",
        )
    normalized = _UNIT_SINGULAR.get(unit_upper, unit_upper)
    return int(num_str), normalized


def safe_interval_literal(num: int, unit: str) -> str:
    """Build a safe INTERVAL literal fragment after validation."""
    return f"{num} {unit}"


def validate_and_convert_strptime_format(format_str: str) -> str:
    """Validate and convert PySpark-style format to DuckDB strptime format."""
    if not isinstance(format_str, str):
        raise CompilationError(
            f"Expected string format, got {type(format_str).__name__}",
        )
    if not _STRPTIME_FORMAT_RE.match(format_str):
        raise CompilationError(
            f"Invalid strptime format: {format_str!r}. "
            "Only alphanumeric, %, -, _, ., :, and space are allowed.",
        )
    return (
        format_str.replace("yyyy", "%Y")
        .replace("MM", "%m")
        .replace("dd", "%d")
        .replace("HH", "%H")
        .replace("mm", "%M")
        .replace("ss", "%S")
    )
