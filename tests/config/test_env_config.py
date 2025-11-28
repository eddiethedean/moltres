"""Tests for environment variable configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine

from moltres import connect
from moltres.config import create_config


def test_env_dsn():
    """Test that DSN can be loaded from environment variable."""
    with patch.dict(os.environ, {"MOLTRES_DSN": "sqlite:///test.db"}):
        config = create_config()
        assert config.engine.dsn == "sqlite:///test.db"


def test_env_dsn_missing():
    """Test that missing DSN raises ValueError."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError, match="Either 'dsn' or 'engine' must be provided as argument"
        ):
            create_config()


def test_env_dsn_override():
    """Test that explicit dsn overrides environment variable."""
    with patch.dict(os.environ, {"MOLTRES_DSN": "sqlite:///env.db"}):
        config = create_config("sqlite:///explicit.db")
        assert config.engine.dsn == "sqlite:///explicit.db"


def test_env_echo():
    """Test that echo can be set via environment variable."""
    with patch.dict(os.environ, {"MOLTRES_DSN": "sqlite:///test.db", "MOLTRES_ECHO": "true"}):
        config = create_config()
        assert config.engine.echo is True

    with patch.dict(os.environ, {"MOLTRES_DSN": "sqlite:///test.db", "MOLTRES_ECHO": "false"}):
        config = create_config()
        assert config.engine.echo is False


def test_env_fetch_format():
    """Test that fetch_format can be set via environment variable."""
    with patch.dict(
        os.environ, {"MOLTRES_DSN": "sqlite:///test.db", "MOLTRES_FETCH_FORMAT": "pandas"}
    ):
        config = create_config()
        assert config.engine.fetch_format == "pandas"


def test_env_pool_options():
    """Test that pool options can be set via environment variables."""
    with patch.dict(
        os.environ,
        {
            "MOLTRES_DSN": "sqlite:///test.db",
            "MOLTRES_POOL_SIZE": "10",
            "MOLTRES_MAX_OVERFLOW": "5",
            "MOLTRES_POOL_TIMEOUT": "30",
            "MOLTRES_POOL_RECYCLE": "3600",
            "MOLTRES_POOL_PRE_PING": "true",
        },
    ):
        config = create_config()
        assert config.engine.pool_size == 10
        assert config.engine.max_overflow == 5
        assert config.engine.pool_timeout == 30
        assert config.engine.pool_recycle == 3600
        assert config.engine.pool_pre_ping is True


def test_env_override_with_kwargs():
    """Test that kwargs override environment variables."""
    with patch.dict(
        os.environ,
        {"MOLTRES_DSN": "sqlite:///env.db", "MOLTRES_POOL_SIZE": "10"},
    ):
        config = create_config("sqlite:///kwarg.db", pool_size=20)
        assert config.engine.dsn == "sqlite:///kwarg.db"
        assert config.engine.pool_size == 20  # kwargs override env


def test_connect_with_env():
    """Test that connect() works with environment variables."""
    with patch.dict(os.environ, {"MOLTRES_DSN": "sqlite:///:memory:"}):
        db = connect()
        assert db.config.engine.dsn == "sqlite:///:memory:"


def test_engine_dialect_detection():
    """Ensure dialect is inferred from provided SQLAlchemy engine instances."""
    engine = create_engine("sqlite:///:memory:")
    db = connect(engine=engine)
    try:
        assert db.dialect.name == "sqlite"
    finally:
        db.close()
