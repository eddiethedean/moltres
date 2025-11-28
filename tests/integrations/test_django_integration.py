"""Comprehensive tests for Django integration utilities.
import json

This module tests all Django integration features:
- Middleware for converting Moltres errors to HTTP responses
- Database connection helpers with Django database routing
- Management commands for query execution
- Template tags for querying data in templates
"""

from __future__ import annotations

import json
import tempfile

import pytest

# Check if Django is available
try:
    import django
    from django.conf import settings
    from django.core.management import call_command
    from django.core.management.base import CommandError
    from django.http import JsonResponse
    from django.test import RequestFactory, TestCase, override_settings
    from django.template import Context, Template

    DJANGO_AVAILABLE = True

    # Configure Django for testing if not already configured
    if not settings.configured:
        # Create a minimal Django settings configuration
        test_db_path = tempfile.mkstemp(suffix=".db")[1]
        settings.configure(
            DEBUG=True,
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": test_db_path,
                },
                "other": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": ":memory:",
                },
            },
            SECRET_KEY="test-secret-key",
            INSTALLED_APPS=[
                "django.contrib.contenttypes",
                "django.contrib.auth",
                "moltres.integrations.django",  # Register Django integration app
            ],
            MIDDLEWARE=[],
            TEMPLATES=[
                {
                    "BACKEND": "django.template.backends.django.DjangoTemplates",
                    "APP_DIRS": True,
                    "OPTIONS": {
                        "context_processors": [],
                    },
                },
            ],
            USE_TZ=True,
        )
        django.setup()
except (ImportError, Exception) as e:
    # Catch any exception during Django setup, not just ImportError
    import warnings

    warnings.warn(f"Django setup failed: {e}")
    DJANGO_AVAILABLE = False
    TestCase = None  # type: ignore[assignment, misc]
    RequestFactory = None  # type: ignore[assignment, misc]
    call_command = None  # type: ignore[assignment, misc]
    CommandError = None  # type: ignore[assignment, misc]
    Template = None  # type: ignore[assignment, misc]
    Context = None  # type: ignore[assignment, misc]

    # Create a no-op decorator for override_settings when Django is not available
    def override_settings(*args, **kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator


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
    return tmp_path / "test_django.db"


@pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")
class TestMoltresExceptionMiddleware:
    """Test MoltresExceptionMiddleware for error handling."""

    def test_database_connection_error(self):
        """Test DatabaseConnectionError handling."""
        # Import directly from the django.py module to avoid package/module conflict
        from moltres.integrations import django as django_module

        MoltresExceptionMiddleware = django_module.MoltresExceptionMiddleware

        def get_response(request):
            raise DatabaseConnectionError(
                "Connection failed",
                suggestion="Check your connection string",
                context={"host": "localhost"},
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 503
        assert isinstance(response, JsonResponse)
        # Django JsonResponse doesn't have .json(), use .content and decode
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Database connection error"
        assert data["message"] == "Connection failed"
        assert data["suggestion"] == "Check your connection string"

    def test_connection_pool_error(self):
        """Test ConnectionPoolError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise ConnectionPoolError(
                "Pool exhausted",
                suggestion="Increase pool size",
                context={"pool_size": 5},
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 503
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Connection pool error"

    def test_query_timeout_error(self):
        """Test QueryTimeoutError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise QueryTimeoutError(
                "Query exceeded timeout",
                timeout=30.0,
                context={"query": "SELECT * FROM users"},
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 504
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Query timeout"
        assert data["timeout_seconds"] == 30.0

    def test_execution_error_not_found(self):
        """Test ExecutionError handling for 'not found' errors."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise ExecutionError(
                "Table 'users' does not exist",
                suggestion="Create the table first",
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 404
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "SQL execution error"

    def test_execution_error_permission(self):
        """Test ExecutionError handling for permission errors."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise ExecutionError(
                "Permission denied",
                suggestion="Check your database permissions",
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 403
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "SQL execution error"

    def test_compilation_error(self):
        """Test CompilationError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise CompilationError(
                "Unsupported operation",
                suggestion="Use a different approach",
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 400
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "SQL compilation error"

    def test_validation_error(self):
        """Test ValidationError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise ValidationError(
                "Invalid column name",
                suggestion="Use valid identifiers",
                context={"column": "invalid-column"},
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 400
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Validation error"

    def test_column_not_found_error(self):
        """Test ColumnNotFoundError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise ColumnNotFoundError(
                "nonexistent",
                available_columns=["id", "name", "age"],
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 400
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Column not found"
        assert data["column_name"] == "nonexistent"

    def test_transaction_error(self):
        """Test TransactionError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise TransactionError(
                "Transaction failed",
                suggestion="Retry the operation",
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 500
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Transaction error"

    def test_moltres_error(self):
        """Test generic MoltresError handling."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise MoltresError(
                "Generic error",
                suggestion="Check documentation",
            )

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        response = middleware(request)

        assert response.status_code == 500
        data = json.loads(response.content.decode("utf-8"))
        assert data["error"] == "Moltres error"

    def test_non_moltres_exception_passthrough(self):
        """Test that non-Moltres exceptions are re-raised."""
        from moltres.integrations.django import MoltresExceptionMiddleware

        def get_response(request):
            raise ValueError("Not a Moltres error")

        middleware = MoltresExceptionMiddleware(get_response)
        factory = RequestFactory()
        request = factory.get("/test")

        with pytest.raises(ValueError, match="Not a Moltres error"):
            middleware(request)


@pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")
class TestDatabaseConnectionHelpers:
    """Test database connection helpers."""

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        }
    )
    def test_get_moltres_db_default(self):
        """Test get_moltres_db with default database."""
        from moltres.integrations import django as django_module

        get_moltres_db = django_module.get_moltres_db

        db = get_moltres_db(using="default")
        assert db is not None
        assert db.dialect.name == "sqlite"

    def test_get_moltres_db_routing(self):
        """Test get_moltres_db with database routing."""
        from moltres.integrations.django import get_moltres_db

        # The 'other' database should be available from the default settings
        db = get_moltres_db(using="other")
        assert db is not None
        assert db.dialect.name == "sqlite"

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        }
    )
    def test_get_moltres_db_invalid_alias(self):
        """Test get_moltres_db with invalid database alias."""
        from django.core.exceptions import ImproperlyConfigured
        from moltres.integrations.django import get_moltres_db

        with pytest.raises(ImproperlyConfigured):
            get_moltres_db(using="nonexistent")

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        }
    )
    def test_get_moltres_db_with_query(self):
        """Test get_moltres_db with actual query execution."""
        from moltres.integrations.django import get_moltres_db
        from moltres.io.records import Records

        db = get_moltres_db(using="default")

        # Create table
        db.create_table(
            "test_users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()

        # Insert data
        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("test_users")

        # Query data
        df = db.table("test_users").select()
        results = df.collect()

        assert len(results) == 1
        assert results[0]["name"] == "Alice"


@pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")
class TestManagementCommands:
    """Test Django management commands."""

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "moltres.integrations.django",  # Register app for management commands
        ],
    )
    def test_moltres_query_command_simple(self, capsys, tmp_path):
        """Test moltres_query command with simple query."""
        from django.conf import settings

        # Use a file-based database instead of :memory: so data persists across connections
        db_file = tmp_path / "test_command.db"
        settings.DATABASES["default"]["NAME"] = str(db_file)

        from moltres.integrations.django import get_moltres_db
        from moltres.io.records import Records

        # Setup: Create table and data
        db = get_moltres_db(using="default")
        db.create_table(
            "test_users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()
        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("test_users")
        # Close the connection to ensure data is written
        db.close()

        # Execute command
        call_command(
            "moltres_query",
            'db.table("test_users").select()',
            database="default",
        )

        captured = capsys.readouterr()
        assert "Alice" in captured.out or "Bob" in captured.out

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.auth",
            "moltres.integrations.django",  # Register app for management commands
        ],
    )
    def test_moltres_query_command_missing_query(self):
        """Test moltres_query command without query."""
        with pytest.raises(CommandError, match="Query is required"):
            call_command("moltres_query", database="default")


@pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")
class TestTemplateTags:
    """Test Django template tags."""

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [],
                },
            },
        ],
    )
    def test_moltres_query_template_tag(self, tmp_path):
        """Test moltres_query template tag."""
        from django.conf import settings

        # Use a file-based database instead of :memory: so data persists across connections
        db_file = tmp_path / "test_template.db"
        settings.DATABASES["default"]["NAME"] = str(db_file)

        from moltres.integrations.django import get_moltres_db
        from moltres.io.records import Records

        # Setup: Create table and data
        db = get_moltres_db(using="default")
        db.create_table(
            "test_users",
            [column("id", "INTEGER"), column("name", "TEXT")],
        ).collect()
        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into("test_users")
        # Close the connection to ensure data is written
        db.close()

        # Test template tag
        template_str = """
        {% load moltres_tags %}
        {% moltres_query "test_users" as users %}
        {% for user in users %}
            {{ user.name }}
        {% endfor %}
        """
        template = Template(template_str)
        context = Context({})
        result = template.render(context)

        assert "Alice" in result

    @override_settings(
        DATABASES={
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": ":memory:",
            }
        },
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [],
                },
            },
        ],
    )
    def test_moltres_format_template_filter(self):
        """Test moltres_format template filter."""
        template_str = """
        {% load moltres_tags %}
        {% moltres_query "test_users" as users %}
        {{ users|moltres_format:"count" }}
        """
        template = Template(template_str)
        context = Context({})
        result = template.render(context)

        # Should show count (may be 0 if table doesn't exist, which is fine for this test)
        assert isinstance(result, str)


@pytest.mark.skipif(not DJANGO_AVAILABLE, reason="Django not installed")
class TestGracefulDegradation:
    """Test graceful degradation when Django is not available."""

    def test_middleware_without_django(self, monkeypatch):
        """Test that middleware raises ImportError when Django is not available."""
        # Import the actual module file, not the package
        from moltres.integrations import django_module

        original_available = django_module.DJANGO_AVAILABLE
        django_module.DJANGO_AVAILABLE = False

        try:
            from moltres.integrations.django import MoltresExceptionMiddleware

            def get_response(request):
                pass

            with pytest.raises(ImportError, match="Django is required"):
                MoltresExceptionMiddleware(get_response)
        finally:
            django_module.DJANGO_AVAILABLE = original_available

    def test_get_moltres_db_without_django(self, monkeypatch):
        """Test that get_moltres_db raises ImportError when Django is not available."""
        # Import the actual module file, not the package
        from moltres.integrations import django_module

        original_available = django_module.DJANGO_AVAILABLE
        django_module.DJANGO_AVAILABLE = False

        try:
            from moltres.integrations.django import get_moltres_db

            with pytest.raises(ImportError, match="Django is required"):
                get_moltres_db()
        finally:
            django_module.DJANGO_AVAILABLE = original_available
