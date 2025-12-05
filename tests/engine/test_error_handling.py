"""Tests for error handling and fallback scenarios."""

import logging
from unittest.mock import MagicMock

import pytest

from moltres import connect
from moltres.engine.execution import QueryExecutor
from moltres.table.schema import column


class TestSQLModelFallback:
    """Test SQLModel .exec() fallback behavior."""

    def test_sqlmodel_exec_fallback_with_attribute_error(self, tmp_path, caplog):
        """Test that AttributeError in SQLModel .exec() falls back gracefully."""
        from sqlalchemy.sql import select
        from moltres.engine.connection import ConnectionManager
        from moltres.config import EngineConfig

        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        # Create a mock SQLModel session with .exec() that raises AttributeError
        mock_session = MagicMock()
        mock_session.exec = MagicMock(side_effect=AttributeError("exec method error"))

        # Set up the connection manager with the mock session
        config = EngineConfig(dsn=f"sqlite:///{db_path}")
        conn_manager = ConnectionManager(config)
        conn_manager._session = mock_session

        executor = QueryExecutor(conn_manager, config)

        # Create a simple model class for testing
        class User:
            def __init__(self, id, name):
                self.id = id
                self.name = name

        # Create a Select statement
        from sqlalchemy import table, column as sa_column

        users_table = table("users", sa_column("id"), sa_column("name"))
        stmt = select(users_table)

        # Execute with model - should trigger SQLModel path and fallback
        with caplog.at_level(logging.DEBUG):
            result = executor.fetch(stmt, model=User)

        # Should have logged the fallback
        assert "falling back to regular execute" in caplog.text.lower()
        # Query should still work
        assert result.rows is not None

    def test_sqlmodel_exec_fallback_with_type_error(self, tmp_path, caplog):
        """Test that TypeError in SQLModel .exec() falls back gracefully."""
        from sqlalchemy.sql import select
        from moltres.engine.connection import ConnectionManager
        from moltres.config import EngineConfig

        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        # Create a mock SQLModel session with .exec() that raises TypeError
        mock_session = MagicMock()
        mock_session.exec = MagicMock(side_effect=TypeError("Incompatible types"))

        # Set up the connection manager with the mock session
        config = EngineConfig(dsn=f"sqlite:///{db_path}")
        conn_manager = ConnectionManager(config)
        conn_manager._session = mock_session

        executor = QueryExecutor(conn_manager, config)

        # Create a simple model class for testing
        class User:
            def __init__(self, id, name):
                self.id = id
                self.name = name

        # Create a Select statement
        from sqlalchemy import table, column as sa_column

        users_table = table("users", sa_column("id"), sa_column("name"))
        stmt = select(users_table)

        # Execute with model - should trigger SQLModel path and fallback
        with caplog.at_level(logging.DEBUG):
            result = executor.fetch(stmt, model=User)

        # Should have logged the fallback
        assert "falling back to regular execute" in caplog.text.lower()
        # Query should still work
        assert result.rows is not None

    def test_sqlmodel_exec_fallback_with_unexpected_exception(self, tmp_path, caplog):
        """Test that unexpected exceptions in SQLModel .exec() fall back gracefully."""
        from sqlalchemy.sql import select
        from moltres.engine.connection import ConnectionManager
        from moltres.config import EngineConfig

        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        # Create a mock SQLModel session with .exec() that raises an unexpected exception
        mock_session = MagicMock()
        mock_session.exec = MagicMock(side_effect=RuntimeError("Unexpected error"))

        # Set up the connection manager with the mock session
        config = EngineConfig(dsn=f"sqlite:///{db_path}")
        conn_manager = ConnectionManager(config)
        conn_manager._session = mock_session

        executor = QueryExecutor(conn_manager, config)

        # Create a simple model class for testing
        class User:
            def __init__(self, id, name):
                self.id = id
                self.name = name

        # Create a Select statement
        from sqlalchemy import table, column as sa_column

        users_table = table("users", sa_column("id"), sa_column("name"))
        stmt = select(users_table)

        # Execute with model - should trigger SQLModel path and fallback
        with caplog.at_level(logging.DEBUG):
            result = executor.fetch(stmt, model=User)

        # Should have logged the fallback with exception details
        assert "falling back to regular execute" in caplog.text.lower()
        assert "Unexpected error" in caplog.text
        # Query should still work
        assert result.rows is not None


class TestSelectExprErrorHandling:
    """Test selectExpr() error handling."""

    def test_selectexpr_column_extraction_failure(self, tmp_path, caplog):
        """Test that selectExpr() handles column extraction failures gracefully."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        # Create a DataFrame with a plan that might cause extraction issues
        df = db.table("users").select()

        # selectExpr should work even if column extraction fails
        with caplog.at_level(logging.DEBUG):
            result_df = df.selectExpr("id", "name")

        # Should be able to execute the query
        results = result_df.collect()
        assert isinstance(results, list)

    def test_selectexpr_with_invalid_sql(self, tmp_path):
        """Test that selectExpr() raises appropriate errors for invalid SQL."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = db.table("users").select()

        # Invalid SQL should raise an error
        with pytest.raises((ValueError, SyntaxError)):
            df.selectExpr("invalid sql expression !!!")


class TestErrorContextLogging:
    """Test that error context is properly logged."""

    def test_execution_error_includes_sql_context(self, tmp_path):
        """Test that execution errors include SQL context."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Try to execute a query on a non-existent table
        df = db.table("nonexistent").select()

        with pytest.raises(Exception) as exc_info:
            df.collect()

        # Error should provide context
        error_msg = str(exc_info.value)
        assert "nonexistent" in error_msg.lower() or "table" in error_msg.lower()

    def test_fallback_logging_includes_exception_details(self, tmp_path, caplog):
        """Test that fallback logging includes exception details."""
        from sqlalchemy.sql import select
        from moltres.engine.connection import ConnectionManager
        from moltres.config import EngineConfig

        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [column("id", "INTEGER")]).collect()

        # Create a mock SQLModel session that raises a specific error
        mock_session = MagicMock()
        mock_session.exec = MagicMock(side_effect=ValueError("Test error message"))

        # Set up the connection manager with the mock session
        config = EngineConfig(dsn=f"sqlite:///{db_path}")
        conn_manager = ConnectionManager(config)
        conn_manager._session = mock_session

        executor = QueryExecutor(conn_manager, config)

        # Create a simple model class for testing
        class User:
            def __init__(self, id):
                self.id = id

        # Create a Select statement
        from sqlalchemy import table, column as sa_column

        users_table = table("users", sa_column("id"))
        stmt = select(users_table)

        with caplog.at_level(logging.DEBUG):
            result = executor.fetch(stmt, model=User)

        # Should have logged the specific error
        assert "Test error message" in caplog.text
        # Query should still work
        assert result.rows is not None
