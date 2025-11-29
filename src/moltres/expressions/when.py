"""Conditional expression builder for CASE WHEN statements."""

from __future__ import annotations

from .column import Column, ColumnLike, ensure_column


class WhenBuilder:
    """Builder for CASE WHEN expressions."""

    def __init__(self, condition: Column, value: ColumnLike):
        self._conditions: list[tuple[Column, Column]] = [(condition, ensure_column(value))]
        self._otherwise: Column | None = None

    def when(self, condition: Column, value: ColumnLike) -> WhenBuilder:
        """Add another WHEN clause.

        Args:
            condition: Condition expression
            value: Value to return if condition is true

        Returns:
            Self for chaining
        """
        self._conditions.append((condition, ensure_column(value)))
        return self

    def otherwise(self, value: ColumnLike) -> Column:
        """Complete the CASE expression with an ELSE clause.

        Args:
            value: Default value if no conditions match

        Returns:
            :class:`Column` expression representing the CASE WHEN statement
        """
        self._otherwise = ensure_column(value)
        # Build the CASE expression
        # CASE WHEN ... THEN ... WHEN ... THEN ... ELSE ... END
        args: list[Column | str] = []
        for cond, val in self._conditions:
            args.append(cond)
            args.append(val)
        if self._otherwise is not None:
            args.append(self._otherwise)
        return Column(op="case_when", args=tuple(args))


def when(condition: Column, value: ColumnLike) -> WhenBuilder:
    """Start building a CASE WHEN expression.

    Args:
        condition: Condition expression
        value: Value to return if condition is true

    Returns:
        WhenBuilder for chaining additional WHEN clauses and ELSE

    Example:
        >>> from moltres import col
        >>> from moltres.expressions import functions as F
        >>> expr = F.when(col("age") >= 18, "adult").otherwise("minor")
    """
    return WhenBuilder(condition, value)
