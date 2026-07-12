"""Tests for error handling — prefer real SQLite failures over mock theater."""

from __future__ import annotations

import pytest

from moltres import connect
from moltres.io.records import Records
from moltres.table.schema import column
from moltres.utils.exceptions import ExecutionError


class TestSelectExprErrorHandling:
    def test_selectexpr_valid_columns(self, tmp_path):
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()
        Records.from_list([{"id": 1, "name": "Alice"}], database=db).insert_into("users")

        results = db.table("users").select().selectExpr("id", "name").collect()
        assert results == [{"id": 1, "name": "Alice"}]
        db.close()

    def test_selectexpr_with_invalid_sql(self, tmp_path):
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()
        df = db.table("users").select()
        with pytest.raises((ValueError, SyntaxError, ExecutionError, Exception)):
            df.selectExpr("invalid sql expression !!!")
        db.close()


class TestRealExecutionErrors:
    def test_missing_table_raises_execution_error(self, tmp_path):
        db = connect(f"sqlite:///{tmp_path / 'missing.db'}")
        with pytest.raises(Exception, match="(?i)no such table|nonexistent|does not exist"):
            db.table("nonexistent").select().collect()
        db.close()

    def test_invalid_sql_via_execute_sql(self, tmp_path):
        db = connect(f"sqlite:///{tmp_path / 'bad_sql.db'}")
        db.create_table("t", [column("id", "INTEGER")]).collect()
        with pytest.raises(Exception, match="(?i)syntax|error|failed"):
            db.execute_sql("SELECT FROM WHERE")
        # Table still usable after failed statement
        assert db.table("t").select().collect() == []
        db.close()


class TestSQLModelFallbackSeam:
    """Document that full-session MagicMocks hide bugs; prefer real execute_sql paths above.

    SQLModel .exec() fallback is an internal seam. Behavioral confidence comes from
    TestRealExecutionErrors and selectExpr tests, not from mocking the entire session.
    """

    def test_invalid_model_fetch_still_safe(self, tmp_path):
        """Fetching with a model class against real SQLite returns typed or mapped rows."""
        from sqlalchemy import column as sa_column, table
        from sqlalchemy.sql import select

        db = connect(f"sqlite:///{tmp_path / 'model.db'}")
        db.create_table(
            "users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()
        Records.from_list([{"id": 1, "name": "Alice"}], database=db).insert_into("users")

        users_table = table("users", sa_column("id"), sa_column("name"))
        result = db.executor.fetch(select(users_table))
        assert len(result.rows) == 1
        row = result.rows[0]
        assert row["id"] == 1
        assert row["name"] == "Alice"
        db.close()
