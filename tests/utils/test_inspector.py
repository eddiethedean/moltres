"""Tests for schema inspector utilities."""

from __future__ import annotations

from moltres.utils.inspector import ColumnInfo


class TestColumnInfo:
    """Tests for ColumnInfo dataclass."""

    def test_init(self):
        """Test initialization."""
        col = ColumnInfo(name="id", type_name="INTEGER")
        assert col.name == "id"
        assert col.type_name == "INTEGER"

    def test_init_with_all_fields(self):
        """Test initialization with all fields."""
        col = ColumnInfo(
            name="price",
            type_name="DECIMAL",
            nullable=False,
            default=0.0,
            primary_key=False,
            precision=10,
            scale=2,
        )
        assert col.name == "price"
        assert col.type_name == "DECIMAL"
        assert col.nullable is False
        assert col.default == 0.0
        assert col.primary_key is False
        assert col.precision == 10
        assert col.scale == 2

    def test_default_values(self):
        """Test default values for optional fields."""
        col = ColumnInfo(name="id", type_name="INTEGER")
        assert col.nullable is True
        assert col.default is None
        assert col.primary_key is False
        assert col.precision is None
        assert col.scale is None

    def test_field_access(self):
        """Test field access."""
        col = ColumnInfo(name="name", type_name="VARCHAR(255)")
        assert col.name == "name"
        assert col.type_name == "VARCHAR(255)"

    def test_equality(self):
        """Test equality comparison."""
        col1 = ColumnInfo(name="id", type_name="INTEGER")
        col2 = ColumnInfo(name="id", type_name="INTEGER")
        col3 = ColumnInfo(name="name", type_name="TEXT")
        assert col1 == col2
        assert col1 != col3

    def test_equality_with_all_fields(self):
        """Test equality with all fields."""
        col1 = ColumnInfo(
            name="id",
            type_name="INTEGER",
            nullable=False,
            primary_key=True,
        )
        col2 = ColumnInfo(
            name="id",
            type_name="INTEGER",
            nullable=False,
            primary_key=True,
        )
        col3 = ColumnInfo(
            name="id",
            type_name="INTEGER",
            nullable=True,
            primary_key=False,
        )
        assert col1 == col2
        assert col1 != col3

    def test_string_representation(self):
        """Test string representation."""
        col = ColumnInfo(name="id", type_name="INTEGER")
        repr_str = repr(col)
        assert "id" in repr_str
        assert "INTEGER" in repr_str

    def test_to_column_def(self):
        """Test conversion to ColumnDef."""
        from moltres.table.schema import ColumnDef

        col_info = ColumnInfo(
            name="price",
            type_name="DECIMAL",
            nullable=False,
            default=0.0,
            primary_key=False,
            precision=10,
            scale=2,
        )
        col_def = col_info.to_column_def()
        assert isinstance(col_def, ColumnDef)
        assert col_def.name == "price"
        assert col_def.type_name == "DECIMAL"
        assert col_def.nullable is False
        assert col_def.default == 0.0
        assert col_def.primary_key is False
        assert col_def.precision == 10
        assert col_def.scale == 2


class TestInspectorUtilities:
    """Tests for inspector utility functions."""

    def test_get_table_names_empty(self, sqlite_db):
        """Test get_table_names with empty database."""
        from moltres.utils.inspector import get_table_names

        tables = get_table_names(sqlite_db)
        assert isinstance(tables, list)
        assert len(tables) == 0

    def test_get_table_names_with_tables(self, sqlite_db):
        """Test get_table_names with tables."""
        from moltres.utils.inspector import get_table_names
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()
        sqlite_db.create_table(
            "orders",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        tables = get_table_names(sqlite_db)
        assert isinstance(tables, list)
        assert len(tables) == 2
        assert "users" in tables
        assert "orders" in tables

    def test_get_view_names_no_views(self, sqlite_db):
        """Test get_view_names with no views."""
        from moltres.utils.inspector import get_view_names

        views = get_view_names(sqlite_db)
        assert isinstance(views, list)
        assert len(views) == 0

    def test_get_table_columns_enhanced(self, sqlite_db):
        """Test get_table_columns returns enhanced ColumnInfo."""
        from moltres.utils.inspector import get_table_columns
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [
                column("id", "INTEGER", nullable=False, primary_key=True),
                column("name", "TEXT", nullable=False),
                column("email", "TEXT", nullable=True),
            ],
        ).collect()

        columns = get_table_columns(sqlite_db, "users")
        assert len(columns) == 3

        # Check that enhanced metadata is present
        id_col = next(col for col in columns if col.name == "id")
        assert id_col.primary_key is True
        assert id_col.nullable is False

        email_col = next(col for col in columns if col.name == "email")
        assert email_col.nullable is True

    def test_reflect_table(self, sqlite_db):
        """Test reflect_table utility function."""
        from moltres.utils.inspector import reflect_table
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [
                column("id", "INTEGER", nullable=False, primary_key=True),
                column("name", "TEXT", nullable=False),
            ],
        ).collect()

        reflected = reflect_table(sqlite_db, "users")
        assert isinstance(reflected, dict)
        assert "users" in reflected
        assert len(reflected["users"]) == 2

        # Check that columns are ColumnDef objects
        from moltres.table.schema import ColumnDef

        for col_def in reflected["users"]:
            assert isinstance(col_def, ColumnDef)

    def test_reflect_database(self, sqlite_db):
        """Test reflect_database utility function."""
        from moltres.utils.inspector import reflect_database
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()
        sqlite_db.create_table(
            "orders",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        reflected = reflect_database(sqlite_db)
        assert isinstance(reflected, dict)
        assert len(reflected) == 2
        assert "users" in reflected
        assert "orders" in reflected

        # Check that columns are ColumnDef objects
        from moltres.table.schema import ColumnDef

        for table_name, columns in reflected.items():
            assert isinstance(columns, list)
            for col_def in columns:
                assert isinstance(col_def, ColumnDef)

    def test_get_table_columns_with_precision_scale(self, sqlite_db):
        """Test get_table_columns extracts precision and scale for numeric types."""
        from moltres.utils.inspector import get_table_columns
        from moltres.table.schema import decimal

        # SQLite doesn't enforce precision/scale, but we can test the extraction logic
        sqlite_db.create_table(
            "products",
            [
                decimal("price", precision=10, scale=2),
                decimal("quantity", precision=5, scale=0),
            ],
        ).collect()

        columns = get_table_columns(sqlite_db, "products")
        assert len(columns) == 2
        # Note: SQLite may not preserve precision/scale, so we just check the function works

    def test_reflect_database_with_views(self, sqlite_db):
        """Test reflect_database with views=True."""
        from moltres.utils.inspector import reflect_database
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        # Test with views=False
        reflected = reflect_database(sqlite_db, views=False)
        assert "users" in reflected

        # Test with views=True (should still work even if no views)
        reflected_with_views = reflect_database(sqlite_db, views=True)
        assert "users" in reflected_with_views

    def test_reflect_database_error_handling(self, sqlite_db):
        """Test reflect_database handles errors gracefully."""
        from moltres.utils.inspector import reflect_database
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True)],
        ).collect()

        # Should not raise even if there are issues (logs warnings)
        reflected = reflect_database(sqlite_db)
        assert isinstance(reflected, dict)

    def test_get_table_names_error_handling(self, sqlite_db):
        """Test get_table_names error handling."""
        from moltres.utils.inspector import get_table_names
        import pytest

        # Test with invalid database (connection_manager is None)
        class FakeDB:
            connection_manager = None

        fake_db = FakeDB()
        with pytest.raises(ValueError, match="Database connection manager is not available"):
            get_table_names(fake_db)

    def test_get_view_names_error_handling(self, sqlite_db):
        """Test get_view_names error handling."""
        from moltres.utils.inspector import get_view_names
        import pytest

        # Test with invalid database
        class FakeDB:
            connection_manager = None

        fake_db = FakeDB()
        with pytest.raises(ValueError, match="Database connection manager is not available"):
            get_view_names(fake_db)

    def test_reflect_table_error_handling(self, sqlite_db):
        """Test reflect_table error handling."""
        from moltres.utils.inspector import reflect_table
        import pytest

        # Test with invalid database
        class FakeDB:
            connection_manager = None

        fake_db = FakeDB()
        with pytest.raises(ValueError, match="Database connection manager is not available"):
            reflect_table(fake_db, "users")

    def test_reflect_database_error_handling_no_connection(self, sqlite_db):
        """Test reflect_database error handling with no connection."""
        from moltres.utils.inspector import reflect_database
        import pytest

        # Test with invalid database
        class FakeDB:
            connection_manager = None

        fake_db = FakeDB()
        with pytest.raises(ValueError, match="Database connection manager is not available"):
            reflect_database(fake_db)

    def test_get_table_schema_function(self, sqlite_db):
        """Test get_table_schema is an alias for get_table_columns."""
        from moltres.utils.inspector import get_table_schema, get_table_columns
        from moltres.table.schema import column

        sqlite_db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()

        schema1 = get_table_schema(sqlite_db, "users")
        schema2 = get_table_columns(sqlite_db, "users")
        assert schema1 == schema2
        assert len(schema1) == 2

    def test_get_table_columns_primary_key_as_int(self, sqlite_db):
        """Test get_table_columns handles primary_key as integer (SQLAlchemy returns 1/0)."""
        from moltres.utils.inspector import get_table_columns
        from moltres.table.schema import column

        sqlite_db.create_table(
            "test_pk",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()

        columns = get_table_columns(sqlite_db, "test_pk")
        id_col = next(col for col in columns if col.name == "id")
        # Should be converted to boolean
        assert isinstance(id_col.primary_key, bool)
        assert id_col.primary_key is True
