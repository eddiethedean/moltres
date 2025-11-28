"""Compile logical plans into SQL using SQLAlchemy Core API.

This module provides backward compatibility by re-exporting the main compiler classes
and functions. The implementation has been split into:
- plan_compiler.py: SQLCompiler class for compiling logical plans
- expression_compiler.py: ExpressionCompiler class for compiling expressions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from sqlalchemy.sql import Select

from ..engine.dialects import DialectSpec, get_dialect
from ..expressions.column import Column
from ..logical.plan import LogicalPlan, SortOrder
from .plan_compiler import SQLCompiler
from .expression_compiler import ExpressionCompiler

if TYPE_CHECKING:
    pass


def compile_plan(plan: LogicalPlan, dialect: Union[str, DialectSpec] = "ansi") -> Select:
    """Compile a logical plan to a SQLAlchemy Select statement.

    Args:
        plan: Logical plan to compile
        dialect: SQL dialect specification

    Returns:
        SQLAlchemy Select statement
    """
    spec = get_dialect(dialect) if isinstance(dialect, str) else dialect
    compiler = SQLCompiler(spec)
    return compiler.compile(plan)


@dataclass(frozen=True)
class CompilationState:
    """State information for compilation context."""

    source: str
    alias: str
    select: Optional[tuple[Column, ...]] = None
    predicate: Optional[Column] = None
    group_by: tuple[Column, ...] = ()
    orders: tuple[SortOrder, ...] = ()
    limit: Optional[int] = None


# Re-export for backward compatibility
__all__ = [
    "compile_plan",
    "CompilationState",
    "SQLCompiler",
    "ExpressionCompiler",
]
