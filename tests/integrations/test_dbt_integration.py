"""Tests for dbt integration utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from moltres.table.schema import column

try:
    import dbt  # noqa: F401

    DBT_AVAILABLE = True
except ImportError:
    DBT_AVAILABLE = False


@pytest.mark.skipif(not DBT_AVAILABLE, reason="dbt-core not installed")
class TestDbtAdapter:
    """Test dbt adapter functions."""

    def test_get_moltres_connection_from_config(self, tmp_path):
        """Test getting Moltres connection from dbt config."""
        from moltres.integrations.dbt import get_moltres_connection

        # Create mock dbt config
        mock_config = MagicMock()
        mock_config.profile_name = "test_profile"
        mock_config.target_name = "test_target"
        mock_config.credentials = MagicMock()
        mock_config.credentials.type = "postgres"
        mock_config.credentials.host = "localhost"
        mock_config.credentials.port = 5432
        mock_config.credentials.user = "test_user"
        mock_config.credentials.password = "test_pass"
        mock_config.credentials.database = "test_db"

        # Test connection string extraction
        # Note: This will try to connect, so we'll use environment variable fallback
        import os

        os.environ["DBT_CONNECTION_STRING"] = f"sqlite:///{tmp_path}/test.db"

        db = get_moltres_connection(mock_config)
        assert db is not None


@pytest.mark.skipif(not DBT_AVAILABLE, reason="dbt-core not installed")
class TestDbtHelpers:
    """Test dbt helper functions."""

    def test_moltres_ref(self, tmp_path):
        """Test moltres_ref helper."""
        from moltres import connect
        from moltres.integrations.dbt import moltres_ref

        db = connect(f"sqlite:///{tmp_path}/test.db")

        # Create a table
        db.create_table(
            "test_model",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        # Create mock dbt context
        mock_dbt = MagicMock()
        mock_dbt.config = MagicMock()
        
        # Mock dbt.ref() to return an object with a proper identifier attribute
        mock_relation = MagicMock()
        mock_relation.identifier = "test_model"  # Return actual string identifier
        mock_dbt.ref.return_value = mock_relation

        # Test referencing
        # Note: In real usage, dbt would handle the model name resolution
        df = moltres_ref(mock_dbt, "test_model", db)
        assert df is not None

    def test_moltres_source(self, tmp_path):
        """Test moltres_source helper."""
        from moltres import connect
        from moltres.integrations.dbt import moltres_source

        db = connect(f"sqlite:///{tmp_path}/test.db")

        # Create a table
        db.create_table(
            "source_table",
            [
                column("id", "INTEGER", primary_key=True),
            ],
        ).collect()

        # Create mock dbt context
        mock_dbt = MagicMock()
        mock_dbt.config = MagicMock()
        
        # Mock dbt.source() to return an object with a proper identifier attribute
        mock_relation = MagicMock()
        mock_relation.identifier = "source_table"  # Return actual string identifier
        mock_dbt.source.return_value = mock_relation

        df = moltres_source(mock_dbt, "raw", "source_table", db)
        assert df is not None

    def test_moltres_var(self):
        """Test moltres_var helper."""
        from moltres.integrations.dbt import moltres_var

        # Create mock dbt context
        mock_dbt = MagicMock()
        mock_dbt.config = MagicMock()
        mock_dbt.config.vars = {"min_age": 25}

        value = moltres_var(mock_dbt, "min_age", default=18)
        assert value == 25

        # Test with default
        value = moltres_var(mock_dbt, "missing_var", default=18)
        assert value == 18
