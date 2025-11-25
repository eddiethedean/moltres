"""Tests for schema reflection functionality."""

from __future__ import annotations

import pytest

from moltres.table.schema import column


class TestDatabaseReflection:
    """Tests for Database reflection methods."""

    def test_get_table_names_empty_database(self, sqlite_db):
        """Test getting table names from empty database."""
        tables = sqlite_db.get_table_names()
        assert isinstance(tables, list)
        assert len(tables) == 0

    def test_get_table_names_with_tables(self, sqlite_db):
        """Test getting table names from database with tables."""
        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()
        sqlite_db.create_table(
            "orders",
            [column("id", "INTEGER", primary_key=True), column("user_id", "INTEGER")],
        ).collect()

        tables = sqlite_db.get_table_names()
        assert isinstance(tables, list)
        assert len(tables) == 2
        assert "users" in tables
        assert "orders" in tables

    def test_get_view_names_no_views(self, sqlite_db):
        """Test getting view names when no views exist."""
        views = sqlite_db.get_view_names()
        assert isinstance(views, list)
        assert len(views) == 0

    def test_get_columns(self, sqlite_db):
        """Test getting column information."""
        sqlite_db.create_table(
            "users",
            [
                column("id", "INTEGER", nullable=False, primary_key=True),
                column("name", "TEXT", nullable=False),
                column("email", "TEXT", nullable=True),
            ],
        ).collect()

        columns = sqlite_db.get_columns("users")
        assert len(columns) == 3

        # Check column names
        col_names = [col.name for col in columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names

        # Check primary key
        id_col = next(col for col in columns if col.name == "id")
        assert id_col.primary_key is True
        assert id_col.nullable is False

    def test_get_columns_with_metadata(self, sqlite_db):
        """Test getting columns with full metadata."""
        sqlite_db.create_table(
            "products",
            [
                column("id", "INTEGER", nullable=False, primary_key=True),
                column("price", "REAL", nullable=False, default=0.0),
                column("description", "TEXT", nullable=True),
            ],
        ).collect()

        columns = sqlite_db.get_columns("products")
        assert len(columns) == 3

        # Check that metadata is populated
        for col in columns:
            assert col.name is not None
            assert col.type_name is not None
            assert isinstance(col.nullable, bool)
            # primary_key may be True or False
            assert isinstance(col.primary_key, bool)

    def test_get_columns_nonexistent_table(self, sqlite_db):
        """Test getting columns from non-existent table."""
        with pytest.raises(RuntimeError, match="Failed to inspect table"):
            sqlite_db.get_columns("nonexistent")

    def test_reflect_table(self, sqlite_db):
        """Test reflecting a single table."""
        sqlite_db.create_table(
            "users",
            [
                column("id", "INTEGER", nullable=False, primary_key=True),
                column("name", "TEXT", nullable=False),
                column("email", "TEXT", nullable=True),
            ],
        ).collect()

        schema = sqlite_db.reflect_table("users")
        assert schema.name == "users"
        assert len(schema.columns) == 3

        # Check column names
        col_names = [col.name for col in schema.columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "email" in col_names

        # Check primary key
        id_col = next(col for col in schema.columns if col.name == "id")
        assert id_col.primary_key is True
        assert id_col.nullable is False

    def test_reflect_table_nonexistent(self, sqlite_db):
        """Test reflecting a non-existent table."""
        with pytest.raises(RuntimeError, match="Failed to inspect table"):
            sqlite_db.reflect_table("nonexistent")

    def test_reflect_database(self, sqlite_db):
        """Test reflecting entire database."""
        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()
        sqlite_db.create_table(
            "orders",
            [column("id", "INTEGER", primary_key=True), column("user_id", "INTEGER")],
        ).collect()

        schemas = sqlite_db.reflect()
        assert isinstance(schemas, dict)
        assert len(schemas) == 2
        assert "users" in schemas
        assert "orders" in schemas

        # Check that schemas are TableSchema objects
        users_schema = schemas["users"]
        assert users_schema.name == "users"
        assert len(users_schema.columns) == 2

        orders_schema = schemas["orders"]
        assert orders_schema.name == "orders"
        assert len(orders_schema.columns) == 2

    def test_reflect_empty_database(self, sqlite_db):
        """Test reflecting empty database."""
        schemas = sqlite_db.reflect()
        assert isinstance(schemas, dict)
        assert len(schemas) == 0

    def test_reflect_with_views(self, sqlite_db):
        """Test reflecting database with views (if supported)."""
        # Create a table first
        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()

        # SQLite supports views, but we'll test reflect() with views=False first
        schemas = sqlite_db.reflect(views=False)
        assert "users" in schemas

        # Test with views=True (should still work even if no views exist)
        schemas_with_views = sqlite_db.reflect(views=True)
        assert "users" in schemas_with_views

    def test_get_columns_invalid_table_name(self, sqlite_db):
        """Test getting columns with invalid table name."""
        with pytest.raises(Exception):  # Should raise ValidationError or similar
            sqlite_db.get_columns("")

    def test_reflect_table_invalid_name(self, sqlite_db):
        """Test reflecting table with invalid name."""
        with pytest.raises(Exception):  # Should raise ValidationError or similar
            sqlite_db.reflect_table("")

    def test_get_table_names_with_schema(self, sqlite_db):
        """Test get_table_names with schema parameter."""
        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        # SQLite doesn't support schemas, so this will raise an error
        # This is expected behavior - schema support is database-specific
        with pytest.raises(RuntimeError):
            sqlite_db.get_table_names(schema="public")

    def test_get_view_names_with_schema(self, sqlite_db):
        """Test get_view_names with schema parameter."""
        # SQLite doesn't support schemas, so this will raise an error
        with pytest.raises(RuntimeError):
            sqlite_db.get_view_names(schema="public")

    def test_reflect_with_schema(self, sqlite_db):
        """Test reflect with schema parameter."""
        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        # SQLite doesn't support schemas, so this will raise an error
        with pytest.raises(RuntimeError):
            sqlite_db.reflect(schema="public")

    def test_reflect_table_with_schema(self, sqlite_db):
        """Test reflect_table with schema parameter."""
        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        # Note: schema parameter is currently accepted but not used in reflect_table
        # (get_table_columns doesn't support schema yet)
        schema = sqlite_db.reflect_table("users", schema="public")
        assert schema.name == "users"
