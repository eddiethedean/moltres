"""Base expression definitions used across Moltres."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .column import Column, LiteralValue
    from ..logical.plan import LogicalPlan, WindowSpec
else:
    LiteralValue = Union[bool, int, float, str, None]
    LogicalPlan = Any
    WindowSpec = Any
    Column = Any

# Type alias for expression arguments
# Args can be Expressions (for nested expressions), LiteralValues, strings (for identifiers),
# LogicalPlan (for subqueries), WindowSpec (for window functions), or sequences of these
ExpressionArg = Union["Expression", LiteralValue, str, LogicalPlan, WindowSpec, Iterable[Any]]


@dataclass(frozen=True)
class Expression:
    """Immutable node in an expression tree."""

    op: str
    args: tuple[ExpressionArg, ...]
    _alias: str | None = None
    _filter: Optional[Column] = None

    def with_alias(self, alias: str) -> Expression:
        return replace(self, _alias=alias)

    @property
    def alias_name(self) -> str | None:
        return self._alias

    def children(self) -> Iterator[ExpressionArg]:
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
        """Return a readable string representation of the expression."""
        # Limit args display to avoid overly long representations
        if len(self.args) > 3:
            args_str = ", ".join(repr(arg) for arg in self.args[:3]) + ", ..."
        else:
            args_str = ", ".join(repr(arg) for arg in self.args)

        alias_str = f", alias={self._alias!r}" if self._alias else ""
        return f"Expression(op='{self.op}', args=({args_str}){alias_str})"
