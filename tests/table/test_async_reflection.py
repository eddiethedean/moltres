"""Tests for async schema reflection functionality."""

from __future__ import annotations

import pytest

try:
    from moltres import async_connect
except ImportError:
    pytest.skip("Async dependencies not installed", allow_module_level=True)

from moltres.table.schema import column


@pytest.mark.asyncio
class TestAsyncDatabaseReflection:
    """Tests for AsyncDatabase reflection methods."""

    async def test_get_table_names_empty_database(self, tmp_path):
        """Test getting table names from empty async database."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            tables = await db.get_table_names()
            assert isinstance(tables, list)
            assert len(tables) == 0
        finally:
            await db.close()

    async def test_get_table_names_with_tables(self, tmp_path):
        """Test getting table names from async database with tables."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
            ).collect()
            await db.create_table(
                "orders",
                [column("id", "INTEGER", primary_key=True), column("user_id", "INTEGER")],
            ).collect()

            tables = await db.get_table_names()
            assert isinstance(tables, list)
            assert len(tables) == 2
            assert "users" in tables
            assert "orders" in tables
        finally:
            await db.close()

    async def test_get_view_names_no_views(self, tmp_path):
        """Test getting view names when no views exist."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            views = await db.get_view_names()
            assert isinstance(views, list)
            assert len(views) == 0
        finally:
            await db.close()

    async def test_get_columns(self, tmp_path):
        """Test getting column information from async database."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [
                    column("id", "INTEGER", nullable=False, primary_key=True),
                    column("name", "TEXT", nullable=False),
                    column("email", "TEXT", nullable=True),
                ],
            ).collect()

            columns = await db.get_columns("users")
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
        finally:
            await db.close()

    async def test_get_columns_with_metadata(self, tmp_path):
        """Test getting columns with full metadata from async database."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "products",
                [
                    column("id", "INTEGER", nullable=False, primary_key=True),
                    column("price", "REAL", nullable=False, default=0.0),
                    column("description", "TEXT", nullable=True),
                ],
            ).collect()

            columns = await db.get_columns("products")
            assert len(columns) == 3

            # Check that metadata is populated
            for col in columns:
                assert col.name is not None
                assert col.type_name is not None
                assert isinstance(col.nullable, bool)
                assert isinstance(col.primary_key, bool)
        finally:
            await db.close()

    async def test_get_columns_nonexistent_table(self, tmp_path):
        """Test getting columns from non-existent table."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            with pytest.raises(RuntimeError, match="Failed to inspect table"):
                await db.get_columns("nonexistent")
        finally:
            await db.close()

    async def test_reflect_table(self, tmp_path):
        """Test reflecting a single table from async database."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [
                    column("id", "INTEGER", nullable=False, primary_key=True),
                    column("name", "TEXT", nullable=False),
                    column("email", "TEXT", nullable=True),
                ],
            ).collect()

            schema = await db.reflect_table("users")
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
        finally:
            await db.close()

    async def test_reflect_table_nonexistent(self, tmp_path):
        """Test reflecting a non-existent table."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            with pytest.raises(RuntimeError, match="Failed to inspect table"):
                await db.reflect_table("nonexistent")
        finally:
            await db.close()

    async def test_reflect_database(self, tmp_path):
        """Test reflecting entire async database."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
            ).collect()
            await db.create_table(
                "orders",
                [column("id", "INTEGER", primary_key=True), column("user_id", "INTEGER")],
            ).collect()

            schemas = await db.reflect()
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
        finally:
            await db.close()

    async def test_reflect_empty_database(self, tmp_path):
        """Test reflecting empty async database."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            schemas = await db.reflect()
            assert isinstance(schemas, dict)
            assert len(schemas) == 0
        finally:
            await db.close()

    async def test_reflect_with_views(self, tmp_path):
        """Test reflecting async database with views (if supported)."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            # Create a table first
            await db.create_table(
                "users",
                [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
            ).collect()

            # Test reflect() with views=False first
            schemas = await db.reflect(views=False)
            assert "users" in schemas

            # Test with views=True (should still work even if no views exist)
            schemas_with_views = await db.reflect(views=True)
            assert "users" in schemas_with_views
        finally:
            await db.close()

    async def test_get_columns_invalid_table_name(self, tmp_path):
        """Test getting columns with invalid table name."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            with pytest.raises(Exception):  # Should raise ValidationError or similar
                await db.get_columns("")
        finally:
            await db.close()

    async def test_reflect_table_invalid_name(self, tmp_path):
        """Test reflecting table with invalid name."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            with pytest.raises(Exception):  # Should raise ValidationError or similar
                await db.reflect_table("")
        finally:
            await db.close()

    async def test_get_table_names_with_schema(self, tmp_path):
        """Test get_table_names with schema parameter."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [column("id", "INTEGER", primary_key=True)],
            ).collect()

            # SQLite doesn't support schemas, so this will raise an error
            with pytest.raises(RuntimeError):
                await db.get_table_names(schema="public")
        finally:
            await db.close()

    async def test_get_view_names_with_schema(self, tmp_path):
        """Test get_view_names with schema parameter."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            # SQLite doesn't support schemas, so this will raise an error
            with pytest.raises(RuntimeError):
                await db.get_view_names(schema="public")
        finally:
            await db.close()

    async def test_reflect_with_schema(self, tmp_path):
        """Test reflect with schema parameter."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [column("id", "INTEGER", primary_key=True)],
            ).collect()

            # SQLite doesn't support schemas, so this will raise an error
            with pytest.raises(RuntimeError):
                await db.reflect(schema="public")
        finally:
            await db.close()

    async def test_reflect_table_with_schema(self, tmp_path):
        """Test reflect_table with schema parameter."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        try:
            await db.create_table(
                "users",
                [column("id", "INTEGER", primary_key=True)],
            ).collect()

            # Note: schema parameter is currently accepted but not used in reflect_table
            # (get_table_columns doesn't support schema yet)
            schema = await db.reflect_table("users", schema="public")
            assert schema.name == "users"
        finally:
            await db.close()
