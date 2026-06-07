"""Safe, declarative query execution for Django integrations (no eval)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

from moltres import col

if TYPE_CHECKING:
    from moltres.table.table import Database

QueryResults = Union[List[Mapping[str, Any]], Any]

_ALLOWED_WHERE_OPS = frozenset({"eq", "ne", "gt", "gte", "lt", "lte"})
_TABLE_SELECT_RE = re.compile(
    r"""^db\.table\((['"])([^'"]+)\1\)\.select\(\)\s*$""",
    re.DOTALL,
)


def _apply_where(
    df: Any,
    *,
    where_column: Optional[str],
    where_op: Optional[str],
    where_value: Optional[str],
) -> Any:
    if not where_column:
        return df
    if where_op is None or where_value is None:
        raise ValueError("where_column requires both where_op and where_value")
    op = where_op.lower()
    if op not in _ALLOWED_WHERE_OPS:
        raise ValueError(
            f"Unsupported where_op {where_op!r}. Allowed: {', '.join(sorted(_ALLOWED_WHERE_OPS))}"
        )
    column = col(where_column)
    # Coerce numeric/boolean literals when unambiguous
    parsed_value: object = where_value
    if where_value.lower() == "true":
        parsed_value = True
    elif where_value.lower() == "false":
        parsed_value = False
    else:
        try:
            parsed_value = int(where_value)
        except ValueError:
            try:
                parsed_value = float(where_value)
            except ValueError:
                parsed_value = where_value

    if op == "eq":
        predicate = column == parsed_value
    elif op == "ne":
        predicate = column != parsed_value
    elif op == "gt":
        predicate = column > parsed_value
    elif op == "gte":
        predicate = column >= parsed_value
    elif op == "lt":
        predicate = column < parsed_value
    else:
        predicate = column <= parsed_value
    return df.where(predicate)


def execute_table_query(
    db: "Database",
    table_name: str,
    *,
    where_column: Optional[str] = None,
    where_op: Optional[str] = None,
    where_value: Optional[str] = None,
    limit: Optional[int] = None,
) -> QueryResults:
    """Run a safe table query without code execution."""
    if not table_name or not table_name.strip():
        raise ValueError("table_name is required")
    df = db.table(table_name).select()
    df = _apply_where(
        df,
        where_column=where_column,
        where_op=where_op,
        where_value=where_value,
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        df = df.limit(limit)
    return df.collect()


def execute_safe_query_string(db: "Database", query_str: str) -> QueryResults:
    """Execute a restricted query string (table select only, no eval)."""
    stripped = query_str.strip()
    match = _TABLE_SELECT_RE.match(stripped)
    if not match:
        raise ValueError(
            "Only simple table queries are supported: db.table('name').select(). "
            "Use --table with optional --where-* flags for filters."
        )
    table_name = match.group(2)
    return execute_table_query(db, table_name)
