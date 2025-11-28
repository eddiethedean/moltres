"""Comprehensive tests for async connection covering edge cases."""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres.engine.async_connection import (
    AsyncConnectionManager,
    _extract_postgres_server_settings,
)
from moltres.config import EngineConfig


class TestExtractPostgresServerSettings:
    """Test _extract_postgres_server_settings function."""

    def test_extract_postgres_server_settings_no_options(self):
        """Test with DSN that has no options."""
        dsn = "postgresql://user:pass@host/db"
        normalized, settings = _extract_postgres_server_settings(dsn)
        assert normalized == dsn
        assert settings == {}

    def test_extract_postgres_server_settings_with_options(self):
        """Test with DSN that has options parameter."""
        dsn = "postgresql://user:pass@host/db?options=-csearch_path=public"
        normalized, settings = _extract_postgres_server_settings(dsn)
        assert "options=" not in normalized
        assert "search_path" in settings
        assert settings["search_path"] == "public"

    def test_extract_postgres_server_settings_multiple_options(self):
        """Test with multiple options."""
        dsn = "postgresql://user:pass@host/db?options=-csearch_path=public -ctimezone=UTC"
        normalized, settings = _extract_postgres_server_settings(dsn)
        assert "search_path" in settings
        assert settings["search_path"] == "public"
        assert "timezone" in settings
        assert settings["timezone"] == "UTC"

    def test_extract_postgres_server_settings_non_postgres(self):
        """Test with non-PostgreSQL DSN."""
        dsn = "sqlite:///test.db"
        normalized, settings = _extract_postgres_server_settings(dsn)
        assert normalized == dsn
        assert settings == {}

    def test_extract_postgres_server_settings_empty_options(self):
        """Test with empty options parameter."""
        dsn = "postgresql://user:pass@host/db?options="
        normalized, settings = _extract_postgres_server_settings(dsn)
        assert normalized == dsn
        assert settings == {}

    def test_extract_postgres_server_settings_other_params(self):
        """Test with other query parameters."""
        dsn = "postgresql://user:pass@host/db?options=-csearch_path=public&ssl=true"
        normalized, settings = _extract_postgres_server_settings(dsn)
        assert "ssl=true" in normalized
        assert "search_path" in settings


@pytest.mark.asyncio
class TestAsyncConnectionManager:
    """Test AsyncConnectionManager class."""

    async def test_connect_basic(self, tmp_path):
        """Test basic connection."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        async with manager.connect() as conn:
            assert conn is not None

        await manager.close()

    async def test_connect_with_transaction(self, tmp_path):
        """Test connect with transaction parameter."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        # Begin a transaction
        txn_conn = await manager.begin_transaction()

        # Use transaction in connect
        async with manager.connect(transaction=txn_conn) as conn:
            assert conn is txn_conn

        await manager.close()

    async def test_begin_transaction(self, tmp_path):
        """Test begin_transaction."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        conn = await manager.begin_transaction()
        assert conn is not None
        assert manager.active_transaction == conn

        await manager.close()

    async def test_begin_transaction_nested_error(self, tmp_path):
        """Test that nested transactions raise error."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        await manager.begin_transaction()

        with pytest.raises(RuntimeError, match="Transaction already active"):
            await manager.begin_transaction()

        await manager.close()

    async def test_commit_transaction(self, tmp_path):
        """Test commit_transaction."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        conn = await manager.begin_transaction()
        await manager.commit_transaction(conn)

        assert manager.active_transaction is None

        await manager.close()

    async def test_commit_transaction_wrong_connection_error(self, tmp_path):
        """Test commit_transaction with wrong connection raises error."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        # Create a different connection
        async with manager.connect() as other_conn:
            with pytest.raises(RuntimeError, match="not the active transaction"):
                await manager.commit_transaction(other_conn)

        await manager.close()

    async def test_rollback_transaction(self, tmp_path):
        """Test rollback_transaction."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        conn = await manager.begin_transaction()
        await manager.rollback_transaction(conn)

        assert manager.active_transaction is None

        await manager.close()

    async def test_rollback_transaction_wrong_connection_error(self, tmp_path):
        """Test rollback_transaction with wrong connection raises error."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        # Create a different connection
        async with manager.connect() as other_conn:
            with pytest.raises(RuntimeError, match="not the active transaction"):
                await manager.rollback_transaction(other_conn)

        await manager.close()

    async def test_active_transaction_property(self, tmp_path):
        """Test active_transaction property."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        assert manager.active_transaction is None

        conn = await manager.begin_transaction()
        assert manager.active_transaction == conn

        await manager.rollback_transaction(conn)
        assert manager.active_transaction is None

        await manager.close()

    async def test_engine_property(self, tmp_path):
        """Test engine property."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        engine = manager.engine
        assert engine is not None

        # Should return same engine on subsequent calls
        engine2 = manager.engine
        assert engine is engine2

        await manager.close()

    async def test_close(self, tmp_path):
        """Test close method."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        # Create engine
        _ = manager.engine

        await manager.close()

        # Engine should be None after close
        assert manager._engine is None

    async def test_close_idempotent(self, tmp_path):
        """Test that close is idempotent."""
        db_path = tmp_path / "test.db"
        config = EngineConfig(dsn=f"sqlite+aiosqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        await manager.close()
        # Should not error on second close
        await manager.close()

    async def test_create_engine_with_explicit_engine(self, tmp_path):
        """Test _create_engine with explicit engine in config."""
        from sqlalchemy.ext.asyncio import create_async_engine

        db_path = tmp_path / "test.db"
        explicit_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
        config = EngineConfig(engine=explicit_engine)
        manager = AsyncConnectionManager(config)

        engine = manager._create_engine()
        assert engine is explicit_engine

        await manager.close()
        await explicit_engine.dispose()

    async def test_create_engine_with_explicit_sync_engine_error(self, tmp_path):
        """Test _create_engine with sync engine raises error."""
        from sqlalchemy import create_engine

        db_path = tmp_path / "test.db"
        sync_engine = create_engine(f"sqlite:///{db_path}")
        config = EngineConfig(engine=sync_engine)  # type: ignore[arg-type]
        manager = AsyncConnectionManager(config)

        with pytest.raises(TypeError, match="must be a SQLAlchemy AsyncEngine"):
            manager._create_engine()

    async def test_create_engine_no_dsn_no_engine_error(self):
        """Test _create_engine with no DSN and no engine raises error."""
        # Error is raised during EngineConfig creation, not in _create_engine
        with pytest.raises(ValueError, match="Either 'dsn', 'engine', or 'session' must be provided"):
            EngineConfig(dsn=None, engine=None)

    async def test_create_engine_auto_detect_async_driver(self, tmp_path):
        """Test auto-detection of async driver."""
        db_path = tmp_path / "test.db"
        # Use DSN without async driver - should auto-detect
        config = EngineConfig(dsn=f"sqlite:///{db_path}")
        manager = AsyncConnectionManager(config)

        # Should auto-convert to sqlite+aiosqlite
        engine = manager._create_engine()
        assert engine is not None

        await manager.close()

    async def test_create_engine_invalid_dsn_error(self, tmp_path):
        """Test _create_engine with invalid DSN raises error."""
        # DSN without :// separator
        config = EngineConfig(dsn="invalid")
        manager = AsyncConnectionManager(config)

        # Should raise ValueError for invalid DSN format
        with pytest.raises(ValueError):
            manager._create_engine()

    async def test_create_engine_unsupported_database_error(self):
        """Test _create_engine with unsupported database raises error."""
        config = EngineConfig(dsn="oracle://user:pass@host/db")
        manager = AsyncConnectionManager(config)

        with pytest.raises(ValueError, match="does not specify an async driver"):
            manager._create_engine()
