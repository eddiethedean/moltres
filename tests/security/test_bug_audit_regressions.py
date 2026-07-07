"""Regression tests for bug audit fixes."""

from __future__ import annotations

import pytest

from moltres import col, connect
from moltres.expressions import when
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.sql.builders import quote_identifier
from moltres.utils.exceptions import CompilationError, ValidationError


def test_savepoint_explicit_commit_does_not_commit_outer_transaction(tmp_path):
    from moltres.table.schema import column

    db_path = tmp_path / "savepoint.db"
    with connect(f"sqlite:///{db_path}") as db:
        db.create_table("t", [column("id", "INTEGER")]).collect()
        with db.transaction() as outer:
            db.insert("t", [{"id": 1}])
            with db.transaction(savepoint=True) as inner:
                db.insert("t", [{"id": 2}])
                inner.commit()
            outer.rollback()
        rows = db.table("t").select().collect()
    assert rows == []


def test_quote_identifier_rejects_backticks():
    with pytest.raises(ValidationError):
        quote_identifier("id` OR 1=1 --", "`")


def test_when_direct_import_compiles():
    from moltres.sql.expression_compiler import ExpressionCompiler
    from moltres.engine.dialects import get_dialect

    expr = when(col("x") > 1, "a").otherwise("b")
    assert expr.op == "case_when"
    assert len(expr.args) == 2
    compiler = ExpressionCompiler(get_dialect("sqlite"))
    compiler.compile_expr(expr)


def test_isin_rejects_string():
    with pytest.raises(TypeError, match="single string"):
        col("status").isin("active")


def test_isin_empty_list_compiles_false():
    from moltres.sql.compiler import ExpressionCompiler
    from moltres.engine.dialects import get_dialect

    expr = col("id").isin([])
    compiler = ExpressionCompiler(get_dialect("sqlite"))
    compiled = compiler.compile_expr(expr)
    assert "0 = 1" in str(compiled) or "false" in str(compiled).lower()


def test_date_trunc_sqlite_compiles():
    from moltres.sql.compiler import ExpressionCompiler
    from moltres.engine.dialects import get_dialect

    expr = F.date_trunc("month", col("created_at"))
    compiler = ExpressionCompiler(get_dialect("sqlite"))
    compiler.compile_expr(expr)


def test_array_position_sqlite_raises():
    from moltres.sql.compiler import ExpressionCompiler
    from moltres.engine.dialects import get_dialect

    expr = F.array_position(col("arr"), F.lit(3))
    compiler = ExpressionCompiler(get_dialect("sqlite"))
    with pytest.raises(CompilationError):
        compiler.compile_expr(expr)


def test_streaming_insert_rejects_schema_mismatch(tmp_path):
    db_path = tmp_path / "stream.db"
    with connect(f"sqlite:///{db_path}") as db:
        from moltres.table.schema import column

        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        def chunks():
            yield [{"id": 1, "name": "Alice"}]
            yield [{"id": 2, "status": "active"}]

        records = Records(_generator=chunks, _database=db)
        with pytest.raises(ValidationError, match="schema mismatch"):
            records.insert_into("users")


def test_merge_when_not_matched_applies_defaults(tmp_path):
    db_path = tmp_path / "merge.db"
    with connect(f"sqlite:///{db_path}") as db:
        from moltres.table.schema import column

        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("status", "TEXT"),
            ],
        ).collect()
        db.merge(
            "users",
            [{"id": 1, "name": "Alice"}],
            on=["id"],
            when_not_matched={"status": "pending"},
        )
        row = db.table("users").select().collect()[0]
        assert row["status"] == "pending"


def test_closed_database_raises(tmp_path):
    db_path = tmp_path / "closed.db"
    db = connect(f"sqlite:///{db_path}")
    db.close()
    with pytest.raises(RuntimeError, match="closed"):
        db.table("users")
