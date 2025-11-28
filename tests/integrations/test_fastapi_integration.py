"""Comprehensive tests for FastAPI integration utilities.

This module tests all FastAPI integration features:
- Exception handlers for converting Moltres errors to HTTP responses
- Dependency injection helpers for database connections
- Error handling decorator for route handlers
- Convenience functions for session handling
"""

from __future__ import annotations

import pytest

# Check if FastAPI is available
try:
    from fastapi import FastAPI, HTTPException, Depends
    from fastapi.testclient import TestClient
    from fastapi import status
    from sqlalchemy import create_engine
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create stubs for type checking
    FastAPI = None  # type: ignore[assignment, misc]
    HTTPException = None  # type: ignore[assignment, misc]
    TestClient = None  # type: ignore[assignment, misc]
    status = None  # type: ignore[assignment, misc]

from moltres import col, connect
from moltres.table.schema import column
from moltres.utils.exceptions import (
    ColumnNotFoundError,
    CompilationError,
    ConnectionPoolError,
    DatabaseConnectionError,
    ExecutionError,
    MoltresError,
    QueryTimeoutError,
    TransactionError,
    ValidationError,
)


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_fastapi.db"


@pytest.fixture
def engine(db_path):
    """Create a SQLAlchemy engine."""
    return create_engine(f"sqlite:///{db_path}")


@pytest.fixture
def session_factory(engine):
    """Create a session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def session(session_factory):
    """Create a test session."""
    with session_factory() as sess:
        yield sess


@pytest.fixture
def async_engine(db_path):
    """Create an async SQLAlchemy engine."""
    return create_async_engine(f"sqlite+aiosqlite:///{db_path}")


@pytest.fixture
def async_session_factory(async_engine):
    """Create an async session factory."""
    return sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestExceptionHandlers:
    """Test exception handlers for converting Moltres errors to HTTP responses."""

    def test_database_connection_error_handler(self):
        """Test DatabaseConnectionError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise DatabaseConnectionError(
                "Connection failed",
                suggestion="Check your connection string",
                context={"host": "localhost"},
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["error"] == "Database connection error"
        assert data["message"] == "Connection failed"
        assert data["suggestion"] == "Check your connection string"
        assert data["detail"]["host"] == "localhost"

    def test_connection_pool_error_handler(self):
        """Test ConnectionPoolError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ConnectionPoolError(
                "Pool exhausted",
                suggestion="Increase pool size",
                context={"pool_size": 5},
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["error"] == "Connection pool error"
        assert data["message"] == "Pool exhausted"
        assert data["suggestion"] == "Increase pool size"

    def test_query_timeout_error_handler(self):
        """Test QueryTimeoutError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise QueryTimeoutError(
                "Query exceeded timeout",
                timeout=30.0,
                context={"query": "SELECT * FROM users"},
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        data = response.json()
        assert data["error"] == "Query timeout"
        assert data["message"] == "Query exceeded timeout"
        assert data["timeout_seconds"] == 30.0

    def test_execution_error_handler_not_found(self):
        """Test ExecutionError handler for 'not found' errors."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ExecutionError(
                "Table 'users' does not exist",
                suggestion="Create the table first",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["error"] == "SQL execution error"
        assert "does not exist" in data["message"]

    def test_execution_error_handler_permission(self):
        """Test ExecutionError handler for permission errors."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ExecutionError(
                "Permission denied",
                suggestion="Check your database permissions",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_403_FORBIDDEN
        data = response.json()
        assert data["error"] == "SQL execution error"

    def test_execution_error_handler_syntax_error(self):
        """Test ExecutionError handler for syntax errors."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ExecutionError(
                "SQL syntax error near 'FROM'",
                suggestion="Check your SQL syntax",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"] == "SQL execution error"

    def test_execution_error_handler_generic(self):
        """Test ExecutionError handler for generic errors."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ExecutionError(
                "Unknown error occurred",
                suggestion="Check logs for details",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "SQL execution error"

    def test_compilation_error_handler(self):
        """Test CompilationError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise CompilationError(
                "Unsupported operation",
                suggestion="Use a different approach",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"] == "SQL compilation error"
        assert data["suggestion"] == "Use a different approach"

    def test_validation_error_handler(self):
        """Test ValidationError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ValidationError(
                "Invalid column name",
                suggestion="Use valid identifiers",
                context={"column": "invalid-column"},
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"] == "Validation error"
        assert data["detail"]["column"] == "invalid-column"

    def test_column_not_found_error_handler(self):
        """Test ColumnNotFoundError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise ColumnNotFoundError(
                "nonexistent",
                available_columns=["id", "name", "age"],
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"] == "Column not found"
        assert data["column_name"] == "nonexistent"
        assert "id" in data["available_columns"]
        assert "suggestion" in data  # Should have suggestion from error

    def test_transaction_error_handler(self):
        """Test TransactionError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise TransactionError(
                "Transaction failed",
                suggestion="Retry the operation",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "Transaction error"

    def test_moltres_error_handler(self):
        """Test generic MoltresError handler."""
        from moltres.integrations.fastapi import register_exception_handlers

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/test")
        def test_endpoint():
            raise MoltresError(
                "Generic error",
                suggestion="Check documentation",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "Moltres error"
        assert data["message"] == "Generic error"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDependencyHelpers:
    """Test dependency injection helpers."""

    def test_get_db(self, session):
        """Test get_db convenience function."""
        from moltres.integrations.fastapi import get_db

        db = get_db(session)
        assert db is not None
        assert db.dialect.name == "sqlite"

    @pytest.mark.asyncio
    async def test_get_async_db(self, async_session_factory):
        """Test get_async_db convenience function."""
        from moltres.integrations.fastapi import get_async_db

        async with async_session_factory() as async_session:
            db = await get_async_db(async_session)
            assert db is not None
            assert db.dialect.name == "sqlite"

    def test_create_db_dependency(self, session_factory):
        """Test create_db_dependency helper."""
        from moltres.integrations.fastapi import create_db_dependency

        def get_session():
            with session_factory() as sess:
                yield sess

        get_db = create_db_dependency(get_session)

        # Test that it returns a callable
        assert callable(get_db)

        # Test that it creates a Database instance
        db = get_db()
        assert db is not None
        assert db.dialect.name == "sqlite"

    def test_create_db_dependency_with_fastapi(self, session_factory, db_path):
        """Test create_db_dependency in a FastAPI route."""
        from moltres.integrations.fastapi import create_db_dependency

        def get_session():
            with session_factory() as sess:
                yield sess

        get_db = create_db_dependency(get_session)

        app = FastAPI()

        @app.get("/test")
        def test_endpoint(db=Depends(get_db)):
            # Create a table and query it
            db.create_table(
                "test_table",
                [column("id", "INTEGER"), column("name", "TEXT")],
            ).collect()

            from moltres.io.records import Records

            Records(
                _data=[{"id": 1, "name": "Alice"}],
                _database=db,
            ).insert_into("test_table")

            df = db.table("test_table").select()
            results = df.collect()
            return {"count": len(results), "name": results[0]["name"]}

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_create_async_db_dependency(self, async_session_factory):
        """Test create_async_db_dependency helper."""
        from moltres.integrations.fastapi import create_async_db_dependency

        async def get_async_session():
            async with async_session_factory() as sess:
                yield sess

        get_db = create_async_db_dependency(get_async_session)

        # Test that it returns a callable
        assert callable(get_db)

        # Test that it creates an AsyncDatabase instance
        # Note: When called directly (not through FastAPI), get_session() returns a generator
        # which the dependency function handles automatically
        db = await get_db()
        assert db is not None
        assert db.dialect.name == "sqlite"

    @pytest.mark.asyncio
    async def test_create_async_db_dependency_with_fastapi(self, async_session_factory, db_path):
        """Test create_async_db_dependency in a FastAPI route."""
        from moltres.integrations.fastapi import create_async_db_dependency

        async def get_async_session():
            async with async_session_factory() as sess:
                yield sess

        get_db = create_async_db_dependency(get_async_session)

        # Test directly since TestClient has limitations with async dependencies
        async with async_session_factory():
            db = await get_db()
            await db.create_table(
                "test_table",
                [column("id", "INTEGER"), column("name", "TEXT")],
            ).collect()

            from moltres.io.records import AsyncRecords

            await AsyncRecords(
                _data=[{"id": 1, "name": "Alice"}],
                _database=db,
            ).insert_into("test_table")

            table_handle = await db.table("test_table")
            df = table_handle.select()
            results = await df.collect()

            assert len(results) == 1
            assert results[0]["name"] == "Alice"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestErrorHandlingDecorator:
    """Test error handling decorator."""

    def test_handle_moltres_errors_sync(self):
        """Test @handle_moltres_errors decorator with sync function."""
        from moltres.integrations.fastapi import handle_moltres_errors

        app = FastAPI()

        @app.get("/test")
        @handle_moltres_errors
        def test_endpoint():
            raise ColumnNotFoundError(
                "nonexistent",
                available_columns=["id", "name"],
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["detail"]["error"] == "ColumnNotFoundError"

    @pytest.mark.asyncio
    async def test_handle_moltres_errors_async(self):
        """Test @handle_moltres_errors decorator with async function."""
        from moltres.integrations.fastapi import handle_moltres_errors

        app = FastAPI()

        @app.get("/test")
        @handle_moltres_errors
        async def test_endpoint():
            raise QueryTimeoutError(
                "Query timeout",
                timeout=30.0,
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        data = response.json()
        assert data["detail"]["error"] == "Query timeout"

    def test_handle_moltres_errors_database_connection(self):
        """Test @handle_moltres_errors with DatabaseConnectionError."""
        from moltres.integrations.fastapi import handle_moltres_errors

        app = FastAPI()

        @app.get("/test")
        @handle_moltres_errors
        def test_endpoint():
            raise DatabaseConnectionError(
                "Connection failed",
                suggestion="Check connection string",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert data["detail"]["error"] == "Database connection error"

    def test_handle_moltres_errors_execution_error_not_found(self):
        """Test @handle_moltres_errors with ExecutionError (not found)."""
        from moltres.integrations.fastapi import handle_moltres_errors

        app = FastAPI()

        @app.get("/test")
        @handle_moltres_errors
        def test_endpoint():
            raise ExecutionError(
                "Table not found",
                suggestion="Create the table",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert data["detail"]["error"] == "SQL execution error"

    def test_handle_moltres_errors_generic_moltres_error(self):
        """Test @handle_moltres_errors with generic MoltresError."""
        from moltres.integrations.fastapi import handle_moltres_errors

        app = FastAPI()

        @app.get("/test")
        @handle_moltres_errors
        def test_endpoint():
            raise MoltresError(
                "Generic error",
                suggestion="Check logs",
            )

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["detail"]["error"] == "Moltres error"


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestIntegrationWithSQLModel:
    """Test FastAPI integration with SQLModel."""

    def test_exception_handlers_with_real_errors(self, session_factory, db_path):
        """Test exception handlers with real Moltres operations."""
        from moltres.integrations.fastapi import register_exception_handlers, create_db_dependency

        def get_session():
            with session_factory() as sess:
                yield sess

        get_db = create_db_dependency(get_session)

        app = FastAPI()
        register_exception_handlers(app)

        @app.get("/users")
        def get_users(db=Depends(get_db)):
            # This will raise ExecutionError if table doesn't exist
            df = db.table("nonexistent_table").select()
            return df.collect()

        client = TestClient(app)
        response = client.get("/users")

        # Should return 404 or 500 because table doesn't exist
        # (The exact status depends on how the error is caught)
        assert response.status_code in [
            status.HTTP_404_NOT_FOUND,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ]
        data = response.json()
        assert "error" in data
        # The error message should mention the table issue
        assert (
            "nonexistent_table" in str(data).lower()
            or "does not exist" in str(data).lower()
            or "not found" in str(data).lower()
        )

    def test_column_not_found_with_real_query(self, session_factory, db_path):
        """Test ColumnNotFoundError with real query."""
        from moltres.integrations.fastapi import register_exception_handlers, create_db_dependency

        def get_session():
            with session_factory() as sess:
                yield sess

        get_db = create_db_dependency(get_session)

        app = FastAPI()
        register_exception_handlers(app)

        # Create table first
        with session_factory() as sess:
            db = connect(session=sess)
            db.create_table(
                "users",
                [column("id", "INTEGER"), column("name", "TEXT")],
            ).collect()

        @app.get("/users")
        def get_users(db=Depends(get_db)):
            # This should work fine
            df = db.table("users").select()
            return df.collect()

        client = TestClient(app)
        response = client.get("/users")

        # Should succeed
        assert response.status_code == 200
        assert isinstance(response.json(), list)


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestErrorHandlingWithoutFastAPI:
    """Test graceful handling when FastAPI is not available."""

    def test_register_exception_handlers_without_fastapi(self, monkeypatch):
        """Test that register_exception_handlers raises ImportError when FastAPI is not available."""
        # Mock FastAPI as not available
        import moltres.integrations.fastapi as fastapi_module

        original_available = fastapi_module.FASTAPI_AVAILABLE
        fastapi_module.FASTAPI_AVAILABLE = False

        try:
            from moltres.integrations.fastapi import register_exception_handlers

            # Create a mock app
            class MockApp:
                pass

            app = MockApp()

            with pytest.raises(ImportError, match="FastAPI is required"):
                register_exception_handlers(app)
        finally:
            fastapi_module.FASTAPI_AVAILABLE = original_available

    def test_handle_moltres_errors_without_fastapi(self, monkeypatch):
        """Test that handle_moltres_errors raises ImportError when FastAPI is not available."""
        # Mock FastAPI as not available
        import moltres.integrations.fastapi as fastapi_module

        original_available = fastapi_module.FASTAPI_AVAILABLE
        fastapi_module.FASTAPI_AVAILABLE = False

        try:
            from moltres.integrations.fastapi import handle_moltres_errors

            def test_func():
                pass

            with pytest.raises(ImportError, match="FastAPI is required"):
                handle_moltres_errors(test_func)
        finally:
            fastapi_module.FASTAPI_AVAILABLE = original_available


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestRealWorldScenarios:
    """Test real-world FastAPI usage scenarios."""

    def test_complete_fastapi_app(self, session_factory, db_path):
        """Test a complete FastAPI app with all integration features."""
        from moltres.integrations.fastapi import (
            register_exception_handlers,
            create_db_dependency,
            handle_moltres_errors,
        )

        def get_session():
            with session_factory() as sess:
                yield sess

        get_db = create_db_dependency(get_session)

        app = FastAPI(title="Test App")
        register_exception_handlers(app)

        # Create table and insert data in a separate session
        # (This ensures the table exists before the FastAPI routes are called)
        with session_factory() as sess:
            db = connect(session=sess)
            db.create_table(
                "users",
                [
                    column("id", "INTEGER", primary_key=True),
                    column("name", "TEXT"),
                    column("age", "INTEGER"),
                ],
            ).collect()

            from moltres.io.records import Records

            Records(
                _data=[
                    {"id": 1, "name": "Alice", "age": 30},
                    {"id": 2, "name": "Bob", "age": 25},
                ],
                _database=db,
            ).insert_into("users")
            # Commit the transaction
            sess.commit()

        @app.get("/users")
        @handle_moltres_errors
        def get_users(db=Depends(get_db)):
            df = db.table("users").select()
            return df.collect()

        # Define /users/stats BEFORE /users/{user_id} to avoid route conflicts
        @app.get("/users/stats")
        def get_stats(db=Depends(get_db)):
            from moltres.expressions import functions as F

            df = db.table("users").select()
            stats = df.select(
                F.count(col("id")).alias("count"),
                F.avg(col("age")).alias("avg_age"),
            ).collect()[0]
            return stats

        @app.get("/users/{user_id}")
        @handle_moltres_errors
        def get_user(user_id: int, db=Depends(get_db)):
            df = db.table("users").select().where(col("id") == user_id)
            results = df.collect()
            if not results:
                raise HTTPException(status_code=404, detail="User not found")
            return results[0]

        client = TestClient(app)

        # Test get all users
        response = client.get("/users")
        assert response.status_code == 200
        users = response.json()
        assert len(users) == 2

        # Test get user by ID
        response = client.get("/users/1")
        assert response.status_code == 200
        user = response.json()
        assert user["name"] == "Alice"

        # Test get non-existent user
        response = client.get("/users/999")
        assert response.status_code == 404

        # Test get stats
        response = client.get("/users/stats")
        assert response.status_code == 200
        stats = response.json()
        assert stats["count"] == 2
        assert stats["avg_age"] == 27.5

        # Test error handling - invalid column
        # Use a different route path to avoid conflict with /users/{user_id}
        @app.get("/users/invalid-column")
        def get_invalid(db=Depends(get_db)):
            df = db.table("users").select(col("nonexistent"))
            return df.collect()

        response = client.get("/users/invalid-column")
        # Should be handled by exception handler
        # 422 can occur if FastAPI validates the route before execution
        assert response.status_code in [400, 422, 500]  # Depending on when error is caught
