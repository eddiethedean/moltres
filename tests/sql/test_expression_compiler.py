"""Unit tests for ExpressionCompiler SQL generation."""

from __future__ import annotations

import pytest
from sqlalchemy.dialects import sqlite as sa_sqlite

from moltres.engine.dialects import get_dialect
from moltres.expressions import col, lit
from moltres.sql.expression_compiler import ExpressionCompiler
from moltres.utils.exceptions import CompilationError


def _compile_sql(compiler: ExpressionCompiler, expression: object) -> str:
    element = compiler.compile_expr(expression)  # type: ignore[arg-type]
    return str(
        element.compile(
            dialect=sa_sqlite.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )


@pytest.fixture
def sqlite_compiler() -> ExpressionCompiler:
    return ExpressionCompiler(get_dialect("sqlite"))


def test_null_aware_eq_compiles_to_is_null(sqlite_compiler: ExpressionCompiler) -> None:
    sql = _compile_sql(sqlite_compiler, col("x") == lit(None))
    assert sql == "x IS NULL"


def test_null_aware_ne_compiles_to_is_not_null(sqlite_compiler: ExpressionCompiler) -> None:
    sql = _compile_sql(sqlite_compiler, col("x") != lit(None))
    assert sql == "x IS NOT NULL"


def test_null_literal_on_left_compiles_to_is_null(sqlite_compiler: ExpressionCompiler) -> None:
    sql = _compile_sql(sqlite_compiler, lit(None) == col("active"))
    assert sql == "active IS NULL"


def test_non_null_equality_uses_equals(sqlite_compiler: ExpressionCompiler) -> None:
    sql = _compile_sql(sqlite_compiler, col("x") == lit(1))
    assert "IS NULL" not in sql
    assert "=" in sql


def test_randn_unsupported_on_sqlite(sqlite_compiler: ExpressionCompiler) -> None:
    from moltres.expressions.functions import randn

    with pytest.raises(CompilationError, match="randn"):
        _compile_sql(sqlite_compiler, randn())
