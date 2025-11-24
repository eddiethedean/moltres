"""Tests for base Expression class."""

from __future__ import annotations

import pytest

from moltres.expressions.expr import Expression


class TestExpression:
    """Test Expression base class."""

    def test_expression_initialization(self):
        """Test Expression initialization."""
        expr = Expression(op="add", args=(1, 2))
        assert expr.op == "add"
        assert expr.args == (1, 2)
        assert expr._alias is None

    def test_expression_with_alias(self):
        """Test Expression with alias."""
        expr = Expression(op="column", args=("name",), _alias="user_name")
        assert expr._alias == "user_name"
        assert expr.alias_name == "user_name"

    def test_expression_with_alias_method(self):
        """Test with_alias method."""
        expr = Expression(op="column", args=("name",))
        aliased = expr.with_alias("user_name")
        assert aliased._alias == "user_name"
        # Original should be unchanged (immutability)
        assert expr._alias is None

    def test_expression_children(self):
        """Test children() method."""
        expr = Expression(op="add", args=(1, 2))
        children = list(expr.children())
        assert children == [1, 2]

    def test_expression_children_nested(self):
        """Test children() with nested expressions."""
        inner = Expression(op="column", args=("x",))
        outer = Expression(op="add", args=(inner, 2))
        children = list(outer.children())
        assert len(children) == 2
        assert isinstance(children[0], Expression)
        assert children[1] == 2

    def test_expression_walk(self):
        """Test walk() method for depth-first traversal."""
        inner1 = Expression(op="column", args=("x",))
        inner2 = Expression(op="column", args=("y",))
        outer = Expression(op="add", args=(inner1, inner2))

        nodes = list(outer.walk())
        assert len(nodes) == 3  # outer + 2 inner
        assert nodes[0] == outer
        assert inner1 in nodes
        assert inner2 in nodes

    def test_expression_walk_nested_iterable(self):
        """Test walk() with nested iterable expressions."""
        inner1 = Expression(op="column", args=("x",))
        inner2 = Expression(op="column", args=("y",))
        # Create expression with iterable args
        outer = Expression(op="func", args=([inner1, inner2],))

        nodes = list(outer.walk())
        assert len(nodes) == 3  # outer + 2 inner
        assert inner1 in nodes
        assert inner2 in nodes

    def test_expression_walk_single_node(self):
        """Test walk() on single expression."""
        expr = Expression(op="column", args=("name",))
        nodes = list(expr.walk())
        assert len(nodes) == 1
        assert nodes[0] == expr

    def test_expression_repr(self):
        """Test __repr__ method."""
        expr = Expression(op="add", args=(1, 2), _alias="sum")
        repr_str = repr(expr)
        assert "add" in repr_str
        assert "sum" in repr_str or "alias" in repr_str

    def test_expression_immutability(self):
        """Test that Expression is immutable."""
        expr = Expression(op="column", args=("name",))
        # Try to modify (should not work if frozen)
        # Since it's a dataclass with frozen=True, this should raise
        with pytest.raises(Exception):  # FrozenInstanceError or similar
            expr._alias = "test"  # type: ignore[misc]
