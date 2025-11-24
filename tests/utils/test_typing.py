"""Tests for typing helpers."""

from __future__ import annotations


from moltres.utils.typing import SupportsAlias


class TestSupportsAlias:
    """Tests for SupportsAlias protocol."""

    def test_runtime_check_with_alias(self):
        """Test runtime checking with object that has alias attribute."""

        class HasAlias:
            def __init__(self):
                self.alias = "test_alias"

        obj = HasAlias()
        assert isinstance(obj, SupportsAlias)

    def test_runtime_check_without_alias(self):
        """Test runtime checking with object that doesn't have alias attribute."""

        class NoAlias:
            def __init__(self):
                self.name = "test"

        obj = NoAlias()
        assert not isinstance(obj, SupportsAlias)

    def test_runtime_check_with_none_alias(self):
        """Test runtime checking with object that has None alias."""

        class HasNoneAlias:
            def __init__(self):
                self.alias = None

        obj = HasNoneAlias()
        # Protocol checks for attribute existence, not value
        assert isinstance(obj, SupportsAlias)

    def test_integration_with_moltres_objects(self):
        """Test integration with actual Moltres objects that have alias."""
        from moltres.expressions.column import Column

        # Column objects have alias attribute (via property)
        col = Column(op="literal", args=(1,), _alias="test")
        assert isinstance(col, SupportsAlias)
        # Column.alias is a property that returns _alias
        assert hasattr(col, "alias")
