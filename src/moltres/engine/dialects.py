"""Dialect registry and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DialectSpec:
    name: str
    supports_cte: bool = True
    quote_char: str = '"'
    supports_filter_clause: bool = True
    supports_savepoints: bool = True
    supports_isolation_levels: bool = True
    supports_read_only_transactions: bool = True
    supports_row_locking: bool = True
    supports_for_update_nowait: bool = True
    supports_for_update_skip_locked: bool = True


DIALECTS: dict[str, DialectSpec] = {
    "ansi": DialectSpec(
        name="ansi",
        supports_filter_clause=True,
        supports_isolation_levels=True,
        supports_read_only_transactions=True,
    ),
    "postgresql": DialectSpec(
        name="postgresql",
        quote_char='"',
        supports_filter_clause=True,
        supports_savepoints=True,
        supports_isolation_levels=True,
        supports_read_only_transactions=True,
        supports_row_locking=True,
        supports_for_update_nowait=True,
        supports_for_update_skip_locked=True,
    ),
    "sqlite": DialectSpec(
        name="sqlite",
        quote_char='"',
        supports_filter_clause=False,
        supports_savepoints=True,  # SQLite 3.6.8+
        supports_isolation_levels=False,  # SQLite only supports SERIALIZABLE and READ UNCOMMITTED via PRAGMA
        supports_read_only_transactions=False,  # Not directly supported
        supports_row_locking=True,  # SQLite 3.6.8+
        supports_for_update_nowait=False,
        supports_for_update_skip_locked=False,
    ),
    "mysql": DialectSpec(
        name="mysql",
        quote_char="`",
        supports_filter_clause=True,
        supports_savepoints=True,
        supports_isolation_levels=True,
        supports_read_only_transactions=False,  # MySQL doesn't support read-only transactions
        supports_row_locking=True,
        supports_for_update_nowait=True,
        supports_for_update_skip_locked=True,  # MySQL 8.0+
    ),
    "mysql+pymysql": DialectSpec(
        name="mysql",
        quote_char="`",
        supports_filter_clause=True,
        supports_savepoints=True,
        supports_isolation_levels=True,
        supports_read_only_transactions=False,
        supports_row_locking=True,
        supports_for_update_nowait=True,
        supports_for_update_skip_locked=True,
    ),
    "duckdb": DialectSpec(
        name="duckdb",
        quote_char='"',
        supports_filter_clause=True,
        supports_savepoints=True,
        supports_isolation_levels=False,  # DuckDB uses snapshot isolation
        supports_read_only_transactions=False,
        supports_row_locking=False,
        supports_for_update_nowait=False,
        supports_for_update_skip_locked=False,
    ),
}


def get_dialect(name: str) -> DialectSpec:
    try:
        return DIALECTS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown dialect '{name}'") from exc
