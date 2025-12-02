"""String accessor for pandas-style string operations on columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ...expressions.column import Column
from ...expressions.functions import (
    length,
    lower,
    ltrim,
    replace,
    rtrim,
    trim,
    upper,
)


@dataclass(frozen=True)
class _StringAccessor:
    """Pandas-style string accessor for :class:`Column` expressions.

    Provides string methods that compile to SQL string functions.
    All methods return :class:`Column` expressions for lazy evaluation.

    Example:
        >>> df['name'].str.upper()  # Returns :class:`Column` expression
        >>> df.query("df['name'].str.length() > 10")  # In query string
    """

    _column: Column

    def upper(self) -> Column:
        """Convert string to uppercase.

        Returns:
            :class:`Column` expression for UPPER() SQL function

        Example:
            >>> df['name'].str.upper()
        """
        return upper(self._column)

    def lower(self) -> Column:
        """Convert string to lowercase.

        Returns:
            :class:`Column` expression for LOWER() SQL function

        Example:
            >>> df['name'].str.lower()
        """
        return lower(self._column)

    def strip(self) -> Column:
        """Remove leading and trailing whitespace.

        Returns:
            :class:`Column` expression for TRIM() SQL function

        Example:
            >>> df['name'].str.strip()
        """
        return trim(self._column)

    def lstrip(self) -> Column:
        """Remove leading whitespace.

        Returns:
            :class:`Column` expression for LTRIM() SQL function

        Example:
            >>> df['name'].str.lstrip()
        """
        return ltrim(self._column)

    def rstrip(self) -> Column:
        """Remove trailing whitespace.

        Returns:
            :class:`Column` expression for RTRIM() SQL function

        Example:
            >>> df['name'].str.rstrip()
        """
        return rtrim(self._column)

    def contains(self, pat: str, case: bool = True, na: Optional[Any] = None) -> Column:
        """Test if pattern is contained in string.

        Args:
            pat: Pattern to search for
            case: If True, case-sensitive search
            na: Not used (for pandas compatibility)

        Returns:
            :class:`Column` expression for LIKE/ILIKE SQL pattern

        Example:
            >>> df['name'].str.contains('Alice')
        """
        # Use LIKE pattern matching
        pattern = f"%{pat}%"
        if case:
            # Case-sensitive: use LIKE
            return self._column.like(pattern)
        else:
            # Case-insensitive: use ILIKE if available, otherwise LOWER() LIKE
            # For now, use LOWER() LIKE for compatibility
            return lower(self._column).like(pattern.lower())

    def startswith(self, pat: str, na: Optional[Any] = None) -> Column:
        """Test if string starts with pattern.

        Args:
            pat: Pattern to check
            na: Not used (for pandas compatibility)

        Returns:
            :class:`Column` expression for LIKE SQL pattern

        Example:
            >>> df['name'].str.startswith('A')
        """
        pattern = f"{pat}%"
        return self._column.like(pattern)

    def endswith(self, pat: str, na: Optional[Any] = None) -> Column:
        """Test if string ends with pattern.

        Args:
            pat: Pattern to check
            na: Not used (for pandas compatibility)

        Returns:
            :class:`Column` expression for LIKE SQL pattern

        Example:
            >>> df['name'].str.endswith('e')
        """
        pattern = f"%{pat}"
        return self._column.like(pattern)

    def replace(self, pat: str, repl: str, n: int = -1, case: bool = True) -> Column:
        """Replace occurrences of pattern with replacement.

        Args:
            pat: Pattern to replace
            repl: Replacement string
            n: Number of replacements (-1 for all) - not fully supported
            case: If True, case-sensitive

        Returns:
            :class:`Column` expression for REPLACE() SQL function

        Example:
            >>> df['name'].str.replace('Alice', 'Alicia')
        """
        # SQL REPLACE replaces all occurrences, so n parameter is ignored
        return replace(self._column, pat, repl)

    def split(self, pat: str = " ", n: int = -1) -> Column:
        """Split string by delimiter.

        Args:
            pat: Delimiter to split on
            n: Maximum number of splits (-1 for all)

        Returns:
            :class:`Column` expression (note: full split array support varies by database)

        Note:
            Full split array functionality depends on database support.
            This returns a placeholder :class:`Column` expression.

        Example:
            >>> df['tags'].str.split(',')
        """
        # SQL split functions are database-specific
        # For now, return a Column that can be used in expressions
        # Full implementation would require database-specific functions
        from ...expressions.column import Column as Col

        # Return a placeholder - actual implementation would use database-specific split
        return Col(op="split", args=(self._column, pat))

    def len(self) -> Column:
        """Get string length.

        Returns:
            :class:`Column` expression for LENGTH() SQL function

        Example:
            >>> df['name'].str.len()
        """
        return length(self._column)

    def __len__(self) -> Column:
        """Get string length (alternative syntax).

        Returns:
            :class:`Column` expression for LENGTH() SQL function

        Example:
            >>> len(df['name'].str)
        """
        return self.len()
