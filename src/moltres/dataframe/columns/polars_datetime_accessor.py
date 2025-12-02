"""DateTime accessor for Polars-style datetime operations on columns."""

from __future__ import annotations

from dataclasses import dataclass

from ...expressions.column import Column
from ...expressions.functions import (
    day,
    dayofweek,
    dayofyear,
    hour,
    minute,
    month,
    quarter,
    second,
    week,
    year,
)


@dataclass(frozen=True)
class _PolarsDateTimeAccessor:
    """Polars-style datetime accessor for :class:`Column` expressions.

    Provides datetime methods that compile to SQL date/time functions.
    All methods return :class:`Column` expressions for lazy evaluation.

    Example:
        >>> df['date'].dt.year()  # Returns :class:`Column` expression
        >>> df.filter(df['date'].dt.year() > 2020)
    """

    _column: Column

    def year(self) -> Column:
        """Extract year from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(YEAR FROM ...) SQL function

        Example:
            >>> df['date'].dt.year()
        """
        return year(self._column)

    def month(self) -> Column:
        """Extract month from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(MONTH FROM ...) SQL function

        Example:
            >>> df['date'].dt.month()
        """
        return month(self._column)

    def day(self) -> Column:
        """Extract day of month from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(DAY FROM ...) SQL function

        Example:
            >>> df['date'].dt.day()
        """
        return day(self._column)

    def hour(self) -> Column:
        """Extract hour from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(HOUR FROM ...) SQL function

        Example:
            >>> df['timestamp'].dt.hour()
        """
        return hour(self._column)

    def minute(self) -> Column:
        """Extract minute from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(MINUTE FROM ...) SQL function

        Example:
            >>> df['timestamp'].dt.minute()
        """
        return minute(self._column)

    def second(self) -> Column:
        """Extract second from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(SECOND FROM ...) SQL function

        Example:
            >>> df['timestamp'].dt.second()
        """
        return second(self._column)

    def quarter(self) -> Column:
        """Extract quarter from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(QUARTER FROM ...) SQL function

        Example:
            >>> df['date'].dt.quarter()
        """
        return quarter(self._column)

    def week(self) -> Column:
        """Extract week number from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(WEEK FROM ...) SQL function

        Example:
            >>> df['date'].dt.week()
        """
        return week(self._column)

    def day_of_week(self) -> Column:
        """Extract day of week from datetime (1=Monday, 7=Sunday).

        Returns:
            :class:`Column` expression for EXTRACT(DOW FROM ...) SQL function

        Example:
            >>> df['date'].dt.day_of_week()
        """
        return dayofweek(self._column)

    def day_of_year(self) -> Column:
        """Extract day of year from datetime.

        Returns:
            :class:`Column` expression for EXTRACT(DOY FROM ...) SQL function

        Example:
            >>> df['date'].dt.day_of_year()
        """
        return dayofyear(self._column)
