"""SQL expression parser for converting SQL strings to :class:`Column` expressions."""

from __future__ import annotations

import re
from typing import Callable, Optional, Set

from .column import Column, col
from .functions import (
    lit,
    abs as abs_func,
    avg,
    ceil,
    coalesce,
    concat,
    count,
    count_distinct,
    floor,
    length,
    lower,
    max as max_func,
    min as min_func,
    round,
    sum as sum_func,
    upper,
)

# Map SQL function names to Moltres functions
FUNCTION_MAP: dict[str, Callable[..., Column]] = {
    "SUM": sum_func,
    "AVG": avg,
    "COUNT": count,
    "COUNT_DISTINCT": count_distinct,
    "MAX": max_func,
    "MIN": min_func,
    "UPPER": upper,
    "LOWER": lower,
    "LENGTH": length,
    "ROUND": round,
    "FLOOR": floor,
    "CEIL": ceil,
    "CEILING": ceil,
    "ABS": abs_func,
    "COALESCE": coalesce,
    "CONCAT": concat,
}


class SQLParser:
    """Parser for SQL expressions that converts them to :class:`Column` expressions."""

    def __init__(self, available_columns: Optional[Set[str]] = None):
        """Initialize the parser.

        Args:
            available_columns: Optional set of available column names for validation
        """
        self.available_columns = available_columns or set()
        self.pos = 0
        self.expr = ""

    def parse(self, expr_str: str) -> Column:
        """Parse a SQL expression string into a :class:`Column` expression.

        Args:
            expr_str: SQL expression string (e.g., "amount * 1.1 as with_tax")

        Returns:
            :class:`Column` expression

        Raises:
            ValueError: If the expression cannot be parsed
        """
        # Remove leading/trailing whitespace
        expr_str = expr_str.strip()
        if not expr_str:
            raise ValueError("Empty expression")

        # Handle alias (AS keyword) - need to be careful not to match "AS" in other contexts
        # Look for " AS " followed by an identifier at the end
        alias_match = re.search(r"\s+AS\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", expr_str, re.IGNORECASE)
        if alias_match:
            alias = alias_match.group(1)
            expr_str = expr_str[: alias_match.start()].strip()
        else:
            alias = None

        # Reset parser state
        self.expr = expr_str
        self.pos = 0

        # Parse the expression
        result = self._parse_expression()

        # Check if we consumed the entire expression
        self._skip_whitespace()
        if self.pos < len(self.expr):
            raise ValueError(
                f"Unexpected token at position {self.pos}: {self.expr[self.pos : min(self.pos + 20, len(self.expr))]}"
            )

        # Apply alias if present
        if alias:
            result = result.alias(alias)

        return result

    def _parse_expression(self) -> Column:
        """Parse an expression (handles operator precedence)."""
        return self._parse_logical_or()

    def _parse_logical_or(self) -> Column:
        """Parse logical OR expressions."""
        left = self._parse_logical_and()
        while self._match_token(r"\bOR\b\s*", case_insensitive=True):
            right = self._parse_logical_and()
            left = Column(op="or", args=(left, right))
        return left

    def _parse_logical_and(self) -> Column:
        """Parse logical AND expressions."""
        left = self._parse_comparison()
        while self._match_token(r"\bAND\b\s*", case_insensitive=True):
            right = self._parse_comparison()
            left = Column(op="and", args=(left, right))
        return left

    def _parse_comparison(self) -> Column:
        """Parse comparison expressions."""
        left = self._parse_additive()
        while True:
            self._skip_whitespace()
            # IS NULL / IS NOT NULL
            if self._match_token(r"IS\s+NOT\s+NULL", case_insensitive=True):
                left = Column(op="is_not_null", args=(left,))
                continue
            elif self._match_token(r"IS\s+NULL", case_insensitive=True):
                left = Column(op="is_null", args=(left,))
                continue
            # LIKE / ILIKE
            elif self._match_token(r"ILIKE\s+", case_insensitive=True):
                self._skip_whitespace()
                right = self._parse_additive()
                # Extract string value from literal Column if it's a literal
                pattern = right.args[0] if right.op == "literal" else right
                left = Column(op="ilike", args=(left, pattern))
                continue
            elif self._match_token(r"LIKE\s+", case_insensitive=True):
                self._skip_whitespace()
                right = self._parse_additive()
                # Extract string value from literal Column if it's a literal
                pattern = right.args[0] if right.op == "literal" else right
                left = Column(op="like", args=(left, pattern))
                continue
            # BETWEEN ... AND
            elif self._match_token(r"BETWEEN\s+", case_insensitive=True):
                self._skip_whitespace()
                lower = self._parse_additive()
                self._skip_whitespace()
                if not self._match_token(r"\bAND\b\s*", case_insensitive=True):
                    raise ValueError(f"Expected AND after BETWEEN at position {self.pos}")
                self._skip_whitespace()
                upper = self._parse_additive()
                left = Column(op="between", args=(left, lower, upper))
                continue
            # Support both = and == for equality (pandas-style)
            elif self._peek() == "=":
                self._advance()
                # Check if next character is also = (==)
                if self.pos < len(self.expr) and self.expr[self.pos] == "=":
                    self._advance()  # Skip second =
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="eq", args=(left, right))
            elif (
                self._peek() == "!"
                and self.pos + 1 < len(self.expr)
                and self.expr[self.pos + 1] == "="
            ):
                self._advance()  # Skip !
                self._advance()  # Skip =
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="ne", args=(left, right))
            elif (
                self._peek() == "<"
                and self.pos + 1 < len(self.expr)
                and self.expr[self.pos + 1] == ">"
            ):
                self._advance()  # Skip <
                self._advance()  # Skip >
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="ne", args=(left, right))
            elif (
                self._peek() == "<"
                and self.pos + 1 < len(self.expr)
                and self.expr[self.pos + 1] == "="
            ):
                self._advance()  # Skip <
                self._advance()  # Skip =
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="le", args=(left, right))
            elif self._peek() == "<":
                self._advance()
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="lt", args=(left, right))
            elif (
                self._peek() == ">"
                and self.pos + 1 < len(self.expr)
                and self.expr[self.pos + 1] == "="
            ):
                self._advance()  # Skip >
                self._advance()  # Skip =
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="ge", args=(left, right))
            elif self._peek() == ">":
                self._advance()
                self._skip_whitespace()
                right = self._parse_additive()
                left = Column(op="gt", args=(left, right))
            else:
                break
        return left

    def _parse_additive(self) -> Column:
        """Parse addition and subtraction."""
        left = self._parse_multiplicative()
        while True:
            self._skip_whitespace()
            if self._peek() == "+":
                self._advance()
                self._skip_whitespace()
                right = self._parse_multiplicative()
                left = Column(op="add", args=(left, right))
            elif self._peek() == "-" and (
                self.pos == 0
                or self.expr[self.pos - 1].isspace()
                or self.expr[self.pos - 1] in "()"
            ):
                # Only treat as subtraction if it's not part of a number
                self._advance()
                self._skip_whitespace()
                right = self._parse_multiplicative()
                left = Column(op="sub", args=(left, right))
            else:
                break
        return left

    def _parse_multiplicative(self) -> Column:
        """Parse multiplication, division, and modulo."""
        left = self._parse_unary()
        while True:
            self._skip_whitespace()
            if self._peek() == "*":
                self._advance()
                self._skip_whitespace()
                right = self._parse_unary()
                left = Column(op="mul", args=(left, right))
            elif self._peek() == "/":
                self._advance()
                self._skip_whitespace()
                right = self._parse_unary()
                left = Column(op="div", args=(left, right))
            elif self._peek() == "%":
                self._advance()
                self._skip_whitespace()
                right = self._parse_unary()
                left = Column(op="mod", args=(left, right))
            else:
                break
        return left

    def _parse_unary(self) -> Column:
        """Parse unary operators."""
        self._skip_whitespace()
        # NOT operator
        if self._match_token(r"NOT\s+", case_insensitive=True):
            expr = self._parse_unary()
            return Column(op="not", args=(expr,))
        elif self._peek() == "-" and (
            self.pos == 0 or self.expr[self.pos - 1].isspace() or self.expr[self.pos - 1] in "()"
        ):
            # Unary minus
            self._advance()
            self._skip_whitespace()
            expr = self._parse_unary()
            return Column(op="neg", args=(expr,))
        elif self._peek() == "+":
            self._advance()
            self._skip_whitespace()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> Column:
        """Parse primary expressions (literals, columns, functions, parentheses)."""
        self._skip_whitespace()

        # Parentheses
        if self._peek() == "(":
            self._advance()  # Skip (
            self._skip_whitespace()
            expr = self._parse_expression()
            self._skip_whitespace()
            if self._peek() != ")":
                raise ValueError(f"Unclosed parenthesis at position {self.pos}")
            self._advance()  # Skip )
            return expr

        # Function call
        func_match = self._match_token(r"([A-Z_][A-Z0-9_]*)\s*\(")
        if func_match:
            func_name = func_match.group(1).upper()
            args = self._parse_function_args()
            if not self._match_token(r"\)"):
                raise ValueError(f"Unclosed function call at position {self.pos}")
            return self._parse_function(func_name, args)

        # String literal
        if self._peek() == "'" or self._peek() == '"':
            quote = self._peek()
            self._advance()
            value = ""
            while self.pos < len(self.expr) and self._peek() != quote:
                if self._peek() == "\\" and self.pos + 1 < len(self.expr):
                    self._advance()
                    value += self._peek()
                else:
                    value += self._peek()
                self._advance()
            if self.pos >= len(self.expr):
                raise ValueError(f"Unclosed string literal at position {self.pos}")
            self._advance()  # Skip closing quote
            return lit(value)

        # Number literal (handle both positive and negative)
        # Check for negative number (but not if it's part of subtraction)
        is_negative = False
        if self._peek() == "-":
            # Check if this is a unary minus (not part of subtraction)
            if (
                self.pos == 0
                or self.expr[self.pos - 1].isspace()
                or self.expr[self.pos - 1] in "()=<>!"
            ):
                is_negative = True
                self._advance()
                self._skip_whitespace()

        num_match = self._match_token(r"(\d+\.?\d*)")
        if num_match:
            num_str = num_match.group(1)
            if "." in num_str:
                num_value: float = float(num_str)
            else:
                num_value = int(num_str)
            if is_negative:
                num_value = -num_value
            return lit(num_value)

        # Boolean or NULL
        if self._match_token(r"NULL", case_insensitive=True):
            return lit(None)
        if self._match_token(r"TRUE", case_insensitive=True):
            return lit(True)
        if self._match_token(r"FALSE", case_insensitive=True):
            return lit(False)

        # Column reference
        col_match = self._match_token(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)")
        if col_match:
            col_name = col_match.group(1)
            # Handle qualified column names (table.column)
            if "." in col_name:
                parts = col_name.split(".")
                # For now, just use the column name part
                col_name = parts[-1]
            return col(col_name)

        # Wildcard
        if self._peek() == "*":
            self._advance()
            return col("*")

        # If we get here and haven't consumed anything, it's an error
        if self.pos >= len(self.expr) or (
            self._peek().isspace() and self.pos == len(self.expr) - 1
        ):
            raise ValueError(f"Unexpected end of expression at position {self.pos}")

        raise ValueError(
            f"Unexpected token at position {self.pos}: {self.expr[self.pos : min(self.pos + 20, len(self.expr))]}"
        )

    def _parse_function_args(self) -> list[Column]:
        """Parse function arguments."""
        args: list[Column] = []
        self._skip_whitespace()
        if self._peek() == ")":
            return args

        while True:
            args.append(self._parse_expression())
            self._skip_whitespace()
            if self._peek() == ")":
                break
            if not self._match_token(r","):
                raise ValueError(f"Expected comma or closing parenthesis at position {self.pos}")
            self._skip_whitespace()

        return args

    def _parse_function(self, func_name: str, args: list[Column]) -> Column:
        """Parse a function call and convert to :class:`Column` expression."""
        if func_name in FUNCTION_MAP:
            func = FUNCTION_MAP[func_name]
            if len(args) == 1:
                return func(args[0])
            elif len(args) > 1:
                # For functions that take multiple args
                if func_name == "COALESCE":
                    return coalesce(*args)
                elif func_name == "CONCAT":
                    return concat(*args)
                else:
                    # Default: apply function to first arg, pass rest as additional args
                    return func(args[0], *args[1:])
            else:
                # Functions with no args (like COUNT(*))
                if func_name == "COUNT":
                    return count("*")
                raise ValueError(f"Function {func_name} requires at least one argument")

        # Unknown function - create a generic function call
        # This allows the SQL compiler to handle dialect-specific functions
        return Column(op="function", args=(func_name.lower(), *args))

    def _skip_whitespace(self) -> None:
        """Skip whitespace."""
        while self.pos < len(self.expr) and self.expr[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str:
        """Peek at the current character."""
        if self.pos >= len(self.expr):
            return ""
        return self.expr[self.pos]

    def _advance(self) -> None:
        """Advance the position."""
        if self.pos < len(self.expr):
            self.pos += 1

    def _match_token(self, pattern: str, case_insensitive: bool = False) -> Optional[re.Match[str]]:
        """Try to match a token at the current position."""
        self._skip_whitespace()
        flags = re.IGNORECASE if case_insensitive else 0
        match = re.match(pattern, self.expr[self.pos :], flags=flags)
        if match:
            self.pos += match.end()
            return match
        return None


def parse_sql_expr(expr_str: str, available_columns: Optional[Set[str]] = None) -> Column:
    """Parse a SQL expression string into a :class:`Column` expression.

    This is the main entry point for parsing SQL expressions.

    Args:
        expr_str: SQL expression string (e.g., "amount * 1.1 as with_tax")
        available_columns: Optional set of available column names for validation

    Returns:
        :class:`Column` expression

    Example:
        >>> parse_sql_expr("amount * 1.1 as with_tax")
        :class:`Column`(...)  # Equivalent to (col("amount") * 1.1).alias("with_tax")

    Raises:
        ValueError: If the expression cannot be parsed
    """
    parser = SQLParser(available_columns)
    return parser.parse(expr_str)
