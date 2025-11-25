"""Base expression definitions used across Moltres."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .column import Column


@dataclass(frozen=True)
class Expression:
    """Immutable node in an expression tree."""

    op: str
    args: tuple[Any, ...]
    _alias: str | None = None
    _filter: Optional["Column"] = None

    def with_alias(self, alias: str) -> Expression:
        return replace(self, _alias=alias)

    @property
    def alias_name(self) -> str | None:
        return self._alias

    def children(self) -> Iterator[Any]:
        yield from self.args

    def walk(self) -> Iterator[Expression]:
        """Depth-first traversal generator."""

        yield self
        for arg in self.args:
            if isinstance(arg, Expression):
                yield from arg.walk()
            elif isinstance(arg, Iterable):
                for nested in arg:
                    if isinstance(nested, Expression):
                        yield from nested.walk()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        args = ", ".join(repr(arg) for arg in self.args)
        return f"Expression(op={self.op!r}, args=({args}), alias={self._alias!r})"
