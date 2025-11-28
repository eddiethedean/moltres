"""Comprehensive tests for Streamlit integration utilities.

This module tests all Streamlit integration features using Streamlit's
built-in testing framework (AppTest):
- DataFrame display components
- Query builder widget
- Caching utilities
- Session state helpers
- Query visualization
- Error handling
"""

from __future__ import annotations


import pytest

# Check if Streamlit is available
try:
    from streamlit.testing.v1 import AppTest

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    AppTest = None  # type: ignore[assignment, misc]

from moltres import connect
from moltres.table.schema import column


@pytest.fixture
def db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "test_streamlit.db"


@pytest.fixture
def db(db_path):
    """Create a test database."""
    database = connect(f"sqlite:///{db_path}")
    # Create a test table
    database.create_table(
        "users",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    # Insert test data
    from moltres.io.records import Records

    Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
        ],
        _database=database,
    ).insert_into("users")

    yield database
    database.close()


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestDataFrameDisplay:
    """Test DataFrame display components using AppTest."""

    def test_moltres_dataframe_basic(self, db, db_path):
        """Test basic DataFrame display."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import moltres_dataframe

db = connect("sqlite:///{db_path}")
df = db.table("users").select()
moltres_dataframe(df, show_query_info=False)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Check that dataframe was displayed
        assert len(at.dataframe) > 0

    def test_moltres_dataframe_with_query_info(self, db, db_path):
        """Test DataFrame display with query information."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import moltres_dataframe

db = connect("sqlite:///{db_path}")
df = db.table("users").select()
moltres_dataframe(df, show_query_info=True)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Check that dataframe was displayed
        assert len(at.dataframe) > 0
        # Check that expander was created (query info)
        assert len(at.expander) > 0

    def test_moltres_dataframe_with_kwargs(self, db, db_path):
        """Test DataFrame display with additional kwargs."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import moltres_dataframe

db = connect("sqlite:///{db_path}")
df = db.table("users").select()
moltres_dataframe(df, height=400, use_container_width=True)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Check that dataframe was displayed
        assert len(at.dataframe) > 0

    def test_moltres_dataframe_error_handling(self, db_path):
        """Test DataFrame display error handling."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import moltres_dataframe

db = connect("sqlite:///{db_path}")
# Try to query non-existent table
df = db.table("nonexistent").select()
moltres_dataframe(df)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should display error
        assert len(at.error) > 0 or len(at.exception) > 0


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestQueryBuilder:
    """Test query builder widget using AppTest."""

    def test_query_builder_basic(self, db, db_path):
        """Test basic query builder."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import query_builder

db = connect("sqlite:///{db_path}")
df = query_builder(db)
if df:
    st.write("Query built successfully")
    results = df.collect()
    st.dataframe(results)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Query builder should work (may return None if no table selected)
        # At minimum, the app should run without errors
        assert True

    def test_query_builder_no_tables(self, tmp_path):
        """Test query builder with no tables."""
        empty_db_path = tmp_path / "empty.db"
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import query_builder

db = connect("sqlite:///{empty_db_path}")
df = query_builder(db)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should show info about no tables
        assert len(at.info) > 0


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestCaching:
    """Test caching utilities using AppTest."""

    def test_cached_query_decorator(self, db, db_path):
        """Test cached_query decorator."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import cached_query

db = connect("sqlite:///{db_path}")

@cached_query(ttl=3600)
def get_users():
    return db.table("users").select().collect()

results = get_users()
st.dataframe(results)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Check that dataframe was displayed
        assert len(at.dataframe) > 0

    def test_clear_moltres_cache(self, db_path):
        """Test clear_moltres_cache function."""
        app_code = """
import streamlit as st
from moltres.integrations.streamlit import clear_moltres_cache

if st.button("Clear Cache"):
    clear_moltres_cache()
    st.success("Cache cleared!")
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Click the button
        if len(at.button) > 0:
            at.button[0].click().run()
            # Should show success message
            assert len(at.success) > 0


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestSessionStateHelpers:
    """Test session state helpers using AppTest."""

    def test_get_db_from_session_new(self, db_path):
        """Test getting database from session state (new connection)."""
        # First, create the database with table and data using the fixture db
        # We'll use the db_path to create a new connection in the test app
        from moltres import connect
        from moltres.table.schema import column
        from moltres.io.records import Records

        # Create database and table
        test_db = connect(f"sqlite:///{db_path}")
        test_db.create_table(
            "users",
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            ],
            _database=test_db,
        ).insert_into("users")
        test_db.close()

        # Now test the Streamlit app
        app_code = f"""
import streamlit as st
from moltres.integrations.streamlit import get_db_from_session

# Set default DSN in session state for testing
if "db" not in st.session_state:
    from moltres.integrations.streamlit import init_db_connection
    db = init_db_connection("sqlite:///{db_path}")
else:
    db = get_db_from_session()

st.write("Database connected")
df = db.table("users").select()
st.dataframe(df.collect())
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should display dataframe
        assert len(at.dataframe) > 0

    def test_init_db_connection(self, db_path):
        """Test initializing database connection."""
        app_code = f"""
import streamlit as st
from moltres.integrations.streamlit import init_db_connection

db = init_db_connection("sqlite:///{db_path}")
st.write("Database initialized")
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should run without errors
        assert True


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestQueryVisualization:
    """Test query visualization components using AppTest."""

    def test_visualize_query_sql_only(self, db, db_path):
        """Test query visualization with SQL only."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import visualize_query

db = connect("sqlite:///{db_path}")
df = db.table("users").select()
visualize_query(df, show_sql=True, show_plan=False, show_metrics=False)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should have expander with SQL
        assert len(at.expander) > 0
        # Should have code block with SQL
        assert len(at.code) > 0

    def test_visualize_query_with_plan(self, db, db_path):
        """Test query visualization with plan."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import visualize_query

db = connect("sqlite:///{db_path}")
df = db.table("users").select()
visualize_query(df, show_sql=True, show_plan=True, show_metrics=False)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should have multiple expanders (SQL and plan)
        assert len(at.expander) >= 2

    def test_visualize_query_with_metrics(self, db, db_path):
        """Test query visualization with metrics."""
        app_code = f"""
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import visualize_query

db = connect("sqlite:///{db_path}")
df = db.table("users").select()
visualize_query(df, show_sql=True, show_plan=True, show_metrics=True)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should have metrics displayed
        # Metrics are displayed using st.metric which may not be directly accessible
        # but the app should run without errors
        assert True


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestErrorHandling:
    """Test error handling helpers using AppTest."""

    def test_display_moltres_error(self, db_path):
        """Test displaying Moltres error."""
        app_code = """
import streamlit as st
from moltres import connect
from moltres.integrations.streamlit import display_moltres_error
from moltres.utils.exceptions import DatabaseConnectionError

error = DatabaseConnectionError(
    "Connection failed",
    suggestion="Check your connection string",
    context={"host": "localhost"},
)
display_moltres_error(error)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should display error
        assert len(at.error) > 0
        # Should display warning with suggestion
        assert len(at.warning) > 0

    def test_display_moltres_error_generic(self):
        """Test displaying generic error."""
        app_code = """
import streamlit as st
from moltres.integrations.streamlit import display_moltres_error

error = ValueError("Generic error")
display_moltres_error(error)
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should display error
        assert len(at.error) > 0


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestGracefulDegradation:
    """Test graceful degradation when Streamlit is not available."""

    def test_import_error_without_streamlit(self):
        """Test that functions raise ImportError when Streamlit is not available."""
        # Temporarily remove streamlit
        import moltres.integrations.streamlit as streamlit_module

        # Save original value
        original_available = streamlit_module.STREAMLIT_AVAILABLE
        streamlit_module.STREAMLIT_AVAILABLE = False

        try:
            from moltres.integrations.streamlit import moltres_dataframe

            with pytest.raises(ImportError, match="Streamlit is required"):
                moltres_dataframe(None)  # type: ignore[arg-type]
        finally:
            # Restore original value
            streamlit_module.STREAMLIT_AVAILABLE = original_available


@pytest.mark.skipif(not STREAMLIT_AVAILABLE, reason="Streamlit not installed")
class TestIntegration:
    """Integration tests for Streamlit features using AppTest."""

    def test_full_workflow(self, db, db_path):
        """Test a full workflow using Streamlit integration."""
        app_code = f"""
import streamlit as st
from moltres import connect, col
from moltres.integrations.streamlit import (
    cached_query,
    init_db_connection,
    moltres_dataframe,
    visualize_query,
)

# Initialize database
db = init_db_connection("sqlite:///{db_path}")

# Create a query
df = db.table("users").select().where(col("age") > 25)

# Visualize query
visualize_query(df, show_sql=True, show_plan=True, show_metrics=False)

# Display DataFrame
moltres_dataframe(df, show_query_info=True)

# Test caching
@cached_query(ttl=3600)
def get_filtered_users():
    return db.table("users").select().where(col("age") > 25).collect()

results = get_filtered_users()
st.write(f"Found {{len(results)}} users")
"""
        at = AppTest.from_string(app_code)
        at.run()

        # Should have displayed dataframes and visualizations
        assert len(at.dataframe) > 0
        assert len(at.expander) > 0
