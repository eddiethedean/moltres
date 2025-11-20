"""Dataset writers."""

from __future__ import annotations

from typing import Protocol


class SupportsToDicts(Protocol):  # pragma: no cover - typing aid
    def to_dicts(self) -> list[dict[str, object]]: ...


def insert_rows(table: str, rows: list[dict[str, object]]):  # pragma: no cover - placeholder
    raise NotImplementedError
