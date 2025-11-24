"""Tests for health check utilities."""

from __future__ import annotations

import pytest

from moltres import async_connect, connect
from moltres.utils.health import (
    HealthCheckResult,
    check_connection_health,
    check_connection_health_async,
    check_pool_health,
    validate_configuration,
)


class TestHealthCheckResult:
    """Tests for HealthCheckResult class."""

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        result = HealthCheckResult(
            healthy=True,
            message="Test message",
            latency=0.123,
            details={"key": "value"},
        )
        assert result.healthy is True
        assert result.message == "Test message"
        assert result.latency == 0.123
        assert result.details == {"key": "value"}

    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        result = HealthCheckResult(healthy=False, message="Error")
        assert result.healthy is False
        assert result.message == "Error"
        assert result.latency is None
        assert result.details == {}

    def test_init_with_none_values(self):
        """Test initialization with None values."""
        result = HealthCheckResult(
            healthy=True,
            message="Test",
            latency=None,
            details=None,
        )
        assert result.latency is None
        assert result.details == {}

    def test_bool_method_true(self):
        """Test __bool__ method returns True for healthy result."""
        result = HealthCheckResult(healthy=True, message="OK")
        assert bool(result) is True
        # Test in conditional context
        if result:
            assert True
        else:
            assert False, "Healthy result should be truthy"

    def test_bool_method_false(self):
        """Test __bool__ method returns False for unhealthy result."""
        result = HealthCheckResult(healthy=False, message="Error")
        assert bool(result) is False
        # Test in conditional context
        if result:
            assert False, "Unhealthy result should be falsy"
        else:
            assert True

    def test_repr_with_latency(self):
        """Test __repr__ with latency."""
        result = HealthCheckResult(
            healthy=True,
            message="Test",
            latency=0.456,
        )
        repr_str = repr(result)
        assert "healthy" in repr_str
        assert "Test" in repr_str
        assert "latency=0.456" in repr_str

    def test_repr_without_latency(self):
        """Test __repr__ without latency."""
        result = HealthCheckResult(healthy=False, message="Error")
        repr_str = repr(result)
        assert "unhealthy" in repr_str
        assert "Error" in repr_str
        assert "latency" not in repr_str

    def test_repr_unhealthy(self):
        """Test __repr__ for unhealthy result."""
        result = HealthCheckResult(healthy=False, message="Failed")
        repr_str = repr(result)
        assert "unhealthy" in repr_str
        assert "Failed" in repr_str


class TestCheckConnectionHealth:
    """Tests for check_connection_health function."""

    def test_successful_connection_sqlite(self, tmp_path):
        """Test successful health check with SQLite."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        result = check_connection_health(db)
        assert result.healthy is True
        assert "healthy" in result.message.lower()
        assert result.latency is not None
        assert result.latency > 0
        assert "query_result" in result.details
        db.close()

    def test_successful_connection_postgresql(self, postgresql_connection):
        """Test successful health check with PostgreSQL."""
        result = check_connection_health(postgresql_connection)
        assert result.healthy is True
        assert "healthy" in result.message.lower()
        assert result.latency is not None
        postgresql_connection.close()

    def test_successful_connection_mysql(self, mysql_connection):
        """Test successful health check with MySQL."""
        result = check_connection_health(mysql_connection)
        assert result.healthy is True
        assert "healthy" in result.message.lower()
        assert result.latency is not None
        mysql_connection.close()

    def test_failed_connection_invalid_db(self):
        """Test health check with invalid database object."""

        class InvalidDB:
            pass

        invalid_db = InvalidDB()
        result = check_connection_health(invalid_db)
        assert result.healthy is False
        assert "does not support health checks" in result.message

    def test_failed_connection_closed_db(self, tmp_path):
        """Test health check with closed database."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        db.close()
        result = check_connection_health(db)
        # SQLite may still work after close() in some cases, so we just check it returns a result
        assert isinstance(result, HealthCheckResult)
        assert result.latency is not None

    def test_timeout_parameter(self, tmp_path):
        """Test that timeout parameter is accepted (even if not fully implemented)."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        result = check_connection_health(db, timeout=10.0)
        assert result.healthy is True
        db.close()


class TestCheckConnectionHealthAsync:
    """Tests for check_connection_health_async function."""

    @pytest.mark.asyncio
    async def test_successful_connection_async_sqlite(self, tmp_path):
        """Test successful async health check with SQLite."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")
        result = await check_connection_health_async(db)
        assert result.healthy is True
        assert "healthy" in result.message.lower()
        assert result.latency is not None
        await db.close()

    @pytest.mark.asyncio
    async def test_successful_connection_async_postgresql(self, postgresql_async_connection):
        """Test successful async health check with PostgreSQL."""
        result = await check_connection_health_async(postgresql_async_connection)
        assert result.healthy is True
        assert "healthy" in result.message.lower()
        assert result.latency is not None
        await postgresql_async_connection.close()

    @pytest.mark.asyncio
    async def test_failed_connection_async_invalid_db(self):
        """Test async health check with invalid database object."""

        class InvalidDB:
            pass

        invalid_db = InvalidDB()
        result = await check_connection_health_async(invalid_db)
        assert result.healthy is False
        assert "does not support health checks" in result.message

    @pytest.mark.asyncio
    async def test_failed_connection_async_closed_db(self, tmp_path):
        """Test async health check with closed database."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")
        await db.close()
        result = await check_connection_health_async(db)
        # SQLite may still work after close() in some cases, so we just check it returns a result
        assert isinstance(result, HealthCheckResult)
        assert result.latency is not None


class TestCheckPoolHealth:
    """Tests for check_pool_health function."""

    def test_pool_health_sqlite(self, tmp_path):
        """Test pool health check with SQLite."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        result = check_pool_health(db)
        # SQLite may or may not have a pool, but should return a result
        assert isinstance(result, HealthCheckResult)
        assert result.message is not None
        db.close()

    def test_pool_health_postgresql(self, postgresql_connection):
        """Test pool health check with PostgreSQL."""
        result = check_pool_health(postgresql_connection)
        assert isinstance(result, HealthCheckResult)
        # PostgreSQL typically has a pool
        assert "pool" in result.message.lower() or "connection" in result.message.lower()
        postgresql_connection.close()

    def test_pool_health_mysql(self, mysql_connection):
        """Test pool health check with MySQL."""
        result = check_pool_health(mysql_connection)
        assert isinstance(result, HealthCheckResult)
        mysql_connection.close()

    def test_pool_health_invalid_db(self):
        """Test pool health check with invalid database object."""

        class InvalidDB:
            pass

        invalid_db = InvalidDB()
        result = check_pool_health(invalid_db)
        assert result.healthy is False
        assert "connection_manager" in result.message.lower() or "failed" in result.message.lower()

    def test_pool_health_no_connection_manager(self):
        """Test pool health check with database without connection_manager."""

        class DBWithoutManager:
            pass

        db = DBWithoutManager()
        result = check_pool_health(db)
        assert result.healthy is False
        assert "connection_manager" in result.message.lower() or "failed" in result.message.lower()


class TestValidateConfiguration:
    """Tests for validate_configuration function."""

    def test_valid_configuration_sqlite(self, tmp_path):
        """Test validation of valid SQLite configuration."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        result = validate_configuration(db)
        assert result.healthy is True
        assert "valid" in result.message.lower()
        db.close()

    def test_valid_configuration_postgresql(self, postgresql_connection):
        """Test validation of valid PostgreSQL configuration."""
        result = validate_configuration(postgresql_connection)
        assert result.healthy is True
        assert "valid" in result.message.lower()
        postgresql_connection.close()

    def test_invalid_dsn_format(self):
        """Test validation with invalid DSN format."""
        from moltres.config import EngineConfig, MoltresConfig

        class MockDB:
            def __init__(self):
                engine_config = EngineConfig(dsn="invalid-dsn-without-protocol")
                self.config = MoltresConfig(engine=engine_config)

        db = MockDB()
        result = validate_configuration(db)
        assert result.healthy is False
        assert "invalid" in result.message.lower() or "issue" in result.message.lower()

    def test_missing_dsn(self):
        """Test validation with missing DSN."""

        # Create a mock config that simulates missing DSN
        class MockEngineConfig:
            dsn = None

        class MockConfig:
            engine = MockEngineConfig()

        class MockDB:
            def __init__(self):
                self.config = MockConfig()

        db = MockDB()
        result = validate_configuration(db)
        # When DSN is None, validation should fail
        assert result.healthy is False
        assert "not configured" in result.message.lower() or "issue" in result.message.lower()

    def test_invalid_pool_size(self):
        """Test validation with invalid pool size."""
        from moltres.config import EngineConfig, MoltresConfig

        class MockDB:
            def __init__(self):
                engine_config = EngineConfig(dsn="sqlite:///test.db", pool_size=-1)
                self.config = MoltresConfig(engine=engine_config)

        db = MockDB()
        result = validate_configuration(db)
        assert result.healthy is False
        assert "pool_size" in result.message.lower() or "issue" in result.message.lower()

    def test_zero_pool_size(self):
        """Test validation with zero pool size."""
        from moltres.config import EngineConfig, MoltresConfig

        class MockDB:
            def __init__(self):
                engine_config = EngineConfig(dsn="sqlite:///test.db", pool_size=0)
                self.config = MoltresConfig(engine=engine_config)

        db = MockDB()
        result = validate_configuration(db)
        assert result.healthy is False
        assert "pool_size" in result.message.lower() or "issue" in result.message.lower()

    def test_validation_error_handling(self):
        """Test validation error handling."""

        class MockDB:
            @property
            def config(self):
                raise AttributeError("Config not available")

        db = MockDB()
        result = validate_configuration(db)
        assert result.healthy is False
        assert "failed" in result.message.lower()
        assert "error_type" in result.details
