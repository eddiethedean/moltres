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

pytestmark = pytest.mark.tier3_integration


@pytest.mark.skipif(not DBT_AVAILABLE, reason="dbt-core not installed")
class TestDbtAdapter:
    """Test dbt adapter functions."""

    def test_get_moltres_connection_from_config(self, tmp_path):
        """Test getting Moltres connection from dbt config (sqlite credentials)."""
        from moltres.integrations.dbt import get_moltres_connection
        from moltres.table.table import Database

        mock_config = MagicMock()
        mock_config.profile_name = "test_profile"
        mock_config.target_name = "test_target"
        mock_config.credentials = MagicMock()
        mock_config.credentials.type = "sqlite"
        mock_config.credentials.database = str(tmp_path / "from_creds.db")
        # Clear host-style fields so builder uses sqlite path
        mock_config.credentials.host = None
        mock_config.credentials.port = None
        mock_config.credentials.user = None
        mock_config.credentials.password = None

        db = get_moltres_connection(mock_config)
        assert isinstance(db, Database)
        db.create_table(
            "probe",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()
        assert db.table("probe").select().collect() == []
        db.close()

    def test_get_moltres_connection_from_env_override(self, tmp_path, monkeypatch):
        """Env DBT_CONNECTION_STRING is used when credentials are absent."""
        from moltres.integrations.dbt import get_moltres_connection
        from moltres.table.table import Database

        mock_config = MagicMock(spec=["profile_name", "target_name"])
        mock_config.profile_name = "test_profile"
        mock_config.target_name = "test_target"
        # No credentials attribute → env fallback
        monkeypatch.setenv("DBT_CONNECTION_STRING", f"sqlite:///{tmp_path}/env.db")

        db = get_moltres_connection(mock_config)
        assert isinstance(db, Database)
        assert db.table  # callable handle factory
        db.close()


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

        # Insert a row so ref() yields real queryable data
        from moltres.io.records import Records

        Records.from_list([{"id": 1, "name": "Ada"}], database=db).insert_into("test_model")

        # Create mock dbt context
        mock_dbt = MagicMock()
        mock_dbt.config = MagicMock()

        # Mock dbt.ref() to return an object with a proper identifier attribute
        mock_relation = MagicMock()
        mock_relation.identifier = "test_model"  # Return actual string identifier
        mock_dbt.ref.return_value = mock_relation

        df = moltres_ref(mock_dbt, "test_model", db)
        rows = df.collect()
        assert rows == [{"id": 1, "name": "Ada"}]
        db.close()

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
        rows = df.collect()
        assert rows == []  # empty table, but query must succeed
        db.close()

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
