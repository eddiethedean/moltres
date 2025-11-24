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

    def test_string_representation(self):
        """Test string representation."""
        col = ColumnInfo(name="id", type_name="INTEGER")
        repr_str = repr(col)
        assert "id" in repr_str
        assert "INTEGER" in repr_str
