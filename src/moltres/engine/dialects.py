"""Dialect registry and helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DialectSpec:
    name: str
    supports_cte: bool = True
    quote_char: str = '"'


DIALECTS: dict[str, DialectSpec] = {
    "ansi": DialectSpec(name="ansi"),
    "postgresql": DialectSpec(name="postgresql", quote_char='"'),
    "sqlite": DialectSpec(name="sqlite", quote_char='"'),
}


def get_dialect(name: str) -> DialectSpec:
    try:
        return DIALECTS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unknown dialect '{name}'") from exc
