"""Tests for AsyncGroupedDataFrame helper methods."""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import col
from moltres.dataframe.async_groupby import AsyncPivotedGroupedDataFrame
from moltres.expressions.functions import (
    avg,
    count,
    max as max_func,
    min as min_func,
    sum as sum_func,
)


class TestAsyncGroupedDataFrameHelpers:
    """Test helper methods for AsyncGroupedDataFrame and AsyncPivotedGroupedDataFrame."""

    def test_extract_value_column(self):
        """Test _extract_value_column() method."""
        # Test with sum aggregation
        agg_col = sum_func(col("amount"))
        value_col = AsyncPivotedGroupedDataFrame._extract_value_column(agg_col)
        assert value_col == "amount"

        # Test with avg aggregation
        agg_col = avg(col("price"))
        value_col = AsyncPivotedGroupedDataFrame._extract_value_column(agg_col)
        assert value_col == "price"

    def test_extract_value_column_invalid(self):
        """Test _extract_value_column() with invalid expression raises ValueError."""
        # Non-aggregation expression
        with pytest.raises(ValueError, match="Expected an aggregation expression"):
            AsyncPivotedGroupedDataFrame._extract_value_column(col("amount"))

        # Aggregation without args
        from moltres.expressions.column import Column

        invalid_agg = Column(op="agg_sum", args=())
        with pytest.raises(ValueError, match="must have arguments"):
            AsyncPivotedGroupedDataFrame._extract_value_column(invalid_agg)

    def test_extract_agg_func(self):
        """Test _extract_agg_func() method."""
        # Test various aggregation functions
        assert AsyncPivotedGroupedDataFrame._extract_agg_func(sum_func(col("amount"))) == "sum"
        assert AsyncPivotedGroupedDataFrame._extract_agg_func(avg(col("amount"))) == "avg"
        assert AsyncPivotedGroupedDataFrame._extract_agg_func(min_func(col("amount"))) == "min"
        assert AsyncPivotedGroupedDataFrame._extract_agg_func(max_func(col("amount"))) == "max"
        assert AsyncPivotedGroupedDataFrame._extract_agg_func(count(col("amount"))) == "count"

    def test_create_aggregation_from_string_pivoted(self):
        """Test _create_aggregation_from_string() for pivoted grouped DataFrame."""
        # Test various aggregation functions
        agg_col = AsyncPivotedGroupedDataFrame._create_aggregation_from_string("amount", "sum")
        assert agg_col is not None

        agg_col = AsyncPivotedGroupedDataFrame._create_aggregation_from_string("amount", "avg")
        assert agg_col is not None

        agg_col = AsyncPivotedGroupedDataFrame._create_aggregation_from_string("amount", "min")
        assert agg_col is not None

        agg_col = AsyncPivotedGroupedDataFrame._create_aggregation_from_string("amount", "max")
        assert agg_col is not None

        agg_col = AsyncPivotedGroupedDataFrame._create_aggregation_from_string("amount", "count")
        assert agg_col is not None

    def test_create_aggregation_from_string_invalid(self):
        """Test _create_aggregation_from_string() with invalid function raises ValueError."""
        with pytest.raises(ValueError, match="Unknown aggregation function"):
            AsyncPivotedGroupedDataFrame._create_aggregation_from_string("amount", "invalid_func")

    def test_validate_aggregation(self):
        """Test _validate_aggregation() method."""
        # Valid aggregation
        agg_col = sum_func(col("amount"))
        validated = AsyncPivotedGroupedDataFrame._validate_aggregation(agg_col)
        # Check that it returns the same object (identity check, not equality)
        assert validated is agg_col
        assert validated.op.startswith("agg_")

        # Invalid aggregation (not starting with agg_)
        with pytest.raises(ValueError, match="Aggregation expressions must be created"):
            AsyncPivotedGroupedDataFrame._validate_aggregation(col("amount"))
