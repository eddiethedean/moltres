"""Dialect registry and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DialectSpec:
    name: str
    supports_cte: bool = True
    quote_char: str = '"'
    supports_filter_clause: bool = True


DIALECTS: dict[str, DialectSpec] = {
    "ansi": DialectSpec(name="ansi", supports_filter_clause=True),
    "postgresql": DialectSpec(name="postgresql", quote_char='"', supports_filter_clause=True),
    "sqlite": DialectSpec(name="sqlite", quote_char='"', supports_filter_clause=False),
    "mysql": DialectSpec(name="mysql", quote_char="`", supports_filter_clause=True),
    "mysql+pymysql": DialectSpec(name="mysql", quote_char="`", supports_filter_clause=True),
    "duckdb": DialectSpec(name="duckdb", quote_char='"', supports_filter_clause=True),
}


def get_dialect(name: str) -> DialectSpec:
    try:
        return DIALECTS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown dialect '{name}'") from exc
