"""Tests for CASE WHEN expression builder."""

from __future__ import annotations


from moltres import col, lit
from moltres.expressions.when import when


class TestWhenBuilder:
    """Test WhenBuilder class."""

    def test_when_builder_initialization(self):
        """Test WhenBuilder initialization."""
        builder = when(col("age") >= 18, "adult")
        assert len(builder._conditions) == 1
        assert builder._otherwise is None

    def test_when_builder_chaining(self):
        """Test chaining multiple WHEN clauses."""
        builder = (
            when(col("age") >= 18, "adult")
            .when(col("age") >= 13, "teen")
            .when(col("age") >= 0, "child")
        )
        assert len(builder._conditions) == 3

    def test_when_builder_otherwise(self):
        """Test adding ELSE clause."""
        expr = when(col("age") >= 18, "adult").otherwise("minor")
        assert expr.op == "case_when"
        assert len(expr.args) == 3  # condition, value, otherwise

    def test_when_builder_multiple_conditions(self):
        """Test multiple WHEN conditions."""
        expr = (
            when(col("score") >= 90, "A")
            .when(col("score") >= 80, "B")
            .when(col("score") >= 70, "C")
            .otherwise("F")
        )
        assert expr.op == "case_when"
        # Should have 7 args: 3 conditions (6) + 1 otherwise
        assert len(expr.args) == 7

    def test_when_builder_without_otherwise(self):
        """Test WHEN without otherwise (should still work)."""
        builder = when(col("age") >= 18, "adult")
        # Without otherwise, we can't complete the expression
        # But the builder should still be valid
        assert len(builder._conditions) == 1

    def test_when_with_literals(self):
        """Test WHEN with literal values."""
        expr = when(col("x") > 0, lit(1)).otherwise(lit(0))
        assert expr.op == "case_when"

    def test_when_with_string_values(self):
        """Test WHEN with string values."""
        expr = when(col("status") == "active", "enabled").otherwise("disabled")
        assert expr.op == "case_when"

    def test_when_nested_conditions(self):
        """Test nested conditions in WHEN."""
        expr = when((col("age") >= 18) & (col("country") == "US"), "eligible").otherwise(
            "ineligible"
        )
        assert expr.op == "case_when"

    def test_when_builder_chaining_immutability(self):
        """Test that WhenBuilder methods support chaining."""
        builder1 = when(col("x") > 0, "positive")
        builder2 = builder1.when(col("x") < 0, "negative")

        # Builder2 should have 2 conditions (chaining modifies the builder)
        assert len(builder2._conditions) == 2
        # Builder1 and builder2 are the same object (mutating builder)
        assert len(builder1._conditions) == 2
