"""Tests for FILTER clause in conditional aggregation."""

from __future__ import annotations

import pytest
from moltres import col, connect, lit
from moltres.expressions import functions as F


class TestFilterClause:
    """Test FILTER clause for conditional aggregation."""

    def test_sum_with_filter(self, tmp_path):
        """Test sum() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "active": True},
                {"id": 2, "category": "A", "amount": 50, "active": False},
                {"id": 3, "category": "B", "amount": 200, "active": True},
                {"id": 4, "category": "B", "amount": 150, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.sum(col("amount")).filter(col("active")).alias("active_sum"))
            .collect()
        )
        assert len(result) == 2
        row_a = next(r for r in result if r["category"] == "A")
        assert row_a["active_sum"] == 100
        row_b = next(r for r in result if r["category"] == "B")
        assert row_b["active_sum"] == 200

    def test_avg_with_filter(self, tmp_path):
        """Test avg() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "price": 10.0, "active": True},
                {"id": 2, "category": "A", "price": 20.0, "active": True},
                {"id": 3, "category": "A", "price": 30.0, "active": False},
                {"id": 4, "category": "B", "price": 40.0, "active": True},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.avg(col("price")).filter(col("active")).alias("avg_price"))
            .collect()
        )
        assert len(result) == 2
        row_a = next(r for r in result if r["category"] == "A")
        assert row_a["avg_price"] == 15.0
        row_b = next(r for r in result if r["category"] == "B")
        assert row_b["avg_price"] == 40.0

    def test_count_with_filter(self, tmp_path):
        """Test count() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "active": True},
                {"id": 2, "category": "A", "active": True},
                {"id": 3, "category": "A", "active": False},
                {"id": 4, "category": "B", "active": True},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.count("*").filter(col("active")).alias("active_count"))
            .collect()
        )
        assert len(result) == 2
        row_a = next(r for r in result if r["category"] == "A")
        assert row_a["active_count"] == 2
        row_b = next(r for r in result if r["category"] == "B")
        assert row_b["active_count"] == 1

    def test_min_with_filter(self, tmp_path):
        """Test min() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "value": 10, "active": True},
                {"id": 2, "category": "A", "value": 5, "active": True},
                {"id": 3, "category": "A", "value": 1, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.min(col("value")).filter(col("active")).alias("min_value"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["min_value"] == 5

    def test_max_with_filter(self, tmp_path):
        """Test max() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "value": 10, "active": True},
                {"id": 2, "category": "A", "value": 5, "active": True},
                {"id": 3, "category": "A", "value": 100, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.max(col("value")).filter(col("active")).alias("max_value"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["max_value"] == 10

    def test_count_distinct_with_filter(self, tmp_path):
        """Test count_distinct() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "user_id": 1, "active": True},
                {"id": 2, "category": "A", "user_id": 1, "active": True},
                {"id": 3, "category": "A", "user_id": 2, "active": True},
                {"id": 4, "category": "A", "user_id": 3, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.count_distinct(col("user_id")).filter(col("active")).alias("active_users"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["active_users"] == 2

    def test_multiple_aggregations_with_different_filters(self, tmp_path):
        """Test multiple aggregations with different FILTER clauses."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "status": "active"},
                {"id": 2, "category": "A", "amount": 50, "status": "inactive"},
                {"id": 3, "category": "A", "amount": 75, "status": "pending"},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(
                F.sum(col("amount")).filter(col("status") == "active").alias("active_sum"),
                F.sum(col("amount")).filter(col("status") == "inactive").alias("inactive_sum"),
            )
            .collect()
        )
        assert len(result) == 1
        assert result[0]["active_sum"] == 100
        assert result[0]["inactive_sum"] == 50

    def test_filter_with_complex_condition(self, tmp_path):
        """Test FILTER clause with complex condition."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "price": 10},
                {"id": 2, "category": "A", "amount": 50, "price": 5},
                {"id": 3, "category": "A", "amount": 200, "price": 20},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(
                F.sum(col("amount"))
                .filter((col("price") > 5) & (col("price") < 20))
                .alias("filtered_sum")
            )
            .collect()
        )
        assert len(result) == 1
        # Only amount=100 with price=10 matches the condition (5 < price < 20)
        assert result[0]["filtered_sum"] == 100

    def test_filter_with_alias(self, tmp_path):
        """Test FILTER clause with alias on aggregation."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "active": True},
                {"id": 2, "category": "A", "amount": 50, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.sum(col("amount")).filter(col("active")).alias("total_active"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["total_active"] == 100

    def test_filter_without_group_by(self, tmp_path):
        """Test FILTER clause in aggregation without GROUP BY."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "amount": 100, "active": True},
                {"id": 2, "amount": 50, "active": False},
                {"id": 3, "amount": 200, "active": True},
            ],
            pk="id",
        )
        result = (
            df.group_by(lit(1))
            .agg(F.sum(col("amount")).filter(col("active")).alias("active_total"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["active_total"] == 300

    @pytest.mark.skipif(True, reason="PostgreSQL test requires PostgreSQL connection")
    def test_filter_clause_postgresql(self, postgresql_connection):
        """Test FILTER clause with PostgreSQL (native support)."""
        db = postgresql_connection
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "active": True},
                {"id": 2, "category": "A", "amount": 50, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.sum(col("amount")).filter(col("active")).alias("active_sum"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["active_sum"] == 100

    def test_filter_clause_sqlite_fallback(self, tmp_path):
        """Test FILTER clause with SQLite (CASE WHEN fallback)."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "active": True},
                {"id": 2, "category": "A", "amount": 50, "active": False},
            ],
            pk="id",
        )
        # SQLite doesn't support FILTER clause natively, should use CASE WHEN
        result = (
            df.group_by("category")
            .agg(F.sum(col("amount")).filter(col("active")).alias("active_sum"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["active_sum"] == 100

    @pytest.mark.skip(reason="SQLite doesn't support stddev() function natively")
    def test_stddev_with_filter(self, tmp_path):
        """Test stddev() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "value": 10, "active": True},
                {"id": 2, "category": "A", "value": 20, "active": True},
                {"id": 3, "category": "A", "value": 5, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.stddev(col("value")).filter(col("active")).alias("active_stddev"))
            .collect()
        )
        assert len(result) == 1
        # Should calculate stddev of [10, 20] = 5.0 (approximately)
        assert abs(result[0]["active_stddev"] - 5.0) < 0.1

    @pytest.mark.skip(reason="SQLite doesn't support variance() function natively")
    def test_variance_with_filter(self, tmp_path):
        """Test variance() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "value": 10, "active": True},
                {"id": 2, "category": "A", "value": 20, "active": True},
                {"id": 3, "category": "A", "value": 5, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.variance(col("value")).filter(col("active")).alias("active_variance"))
            .collect()
        )
        assert len(result) == 1
        # Should calculate variance of [10, 20] = 25.0
        assert abs(result[0]["active_variance"] - 25.0) < 0.1

    @pytest.mark.skip(reason="SQLite doesn't support corr() function natively")
    def test_corr_with_filter(self, tmp_path):
        """Test corr() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "x": 1, "y": 2, "active": True},
                {"id": 2, "category": "A", "x": 2, "y": 4, "active": True},
                {"id": 3, "category": "A", "x": 3, "y": 6, "active": True},
                {"id": 4, "category": "A", "x": 10, "y": 20, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.corr(col("x"), col("y")).filter(col("active")).alias("active_corr"))
            .collect()
        )
        assert len(result) == 1
        # Perfect positive correlation for active rows (x=1,2,3 and y=2,4,6)
        assert abs(result[0]["active_corr"] - 1.0) < 0.01

    @pytest.mark.skip(reason="SQLite doesn't support covar() function natively")
    def test_covar_with_filter(self, tmp_path):
        """Test covar() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "x": 1, "y": 2, "active": True},
                {"id": 2, "category": "A", "x": 2, "y": 4, "active": True},
                {"id": 3, "category": "A", "x": 3, "y": 6, "active": True},
                {"id": 4, "category": "A", "x": 10, "y": 20, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.covar(col("x"), col("y")).filter(col("active")).alias("active_covar"))
            .collect()
        )
        assert len(result) == 1
        # Covariance should be positive for active rows
        assert result[0]["active_covar"] is not None
        assert result[0]["active_covar"] > 0

    def test_collect_list_with_filter(self, tmp_path):
        """Test collect_list() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "value": 1, "active": True},
                {"id": 2, "category": "A", "value": 2, "active": True},
                {"id": 3, "category": "A", "value": 3, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.collect_list(col("value")).filter(col("active")).alias("active_list"))
            .collect()
        )
        assert len(result) == 1
        # Should collect only active values [1, 2]
        active_list = result[0]["active_list"]
        assert active_list is not None
        # SQLite returns JSON array as string, parse it
        if isinstance(active_list, str):
            import json

            active_list = json.loads(active_list)
        # Filter out None values (from CASE WHEN fallback)
        active_list = [x for x in active_list if x is not None]
        assert len(active_list) == 2
        assert 1 in active_list
        assert 2 in active_list

    def test_collect_set_with_filter(self, tmp_path):
        """Test collect_set() with FILTER clause."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "value": 1, "active": True},
                {"id": 2, "category": "A", "value": 1, "active": True},
                {"id": 3, "category": "A", "value": 2, "active": True},
                {"id": 4, "category": "A", "value": 3, "active": False},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.collect_set(col("value")).filter(col("active")).alias("active_set"))
            .collect()
        )
        assert len(result) == 1
        # Should collect distinct active values [1, 2]
        active_set = result[0]["active_set"]
        assert active_set is not None
        # SQLite returns JSON array as string, parse it
        if isinstance(active_set, str):
            import json

            active_set = json.loads(active_set)
        # Filter out None values (from CASE WHEN fallback)
        active_set = [x for x in active_set if x is not None]
        # Convert to set to check distinctness
        active_set = list(set(active_set))
        assert len(active_set) == 2
        assert 1 in active_set
        assert 2 in active_set

    def test_filter_method_on_non_aggregate(self):
        """Test that filter() raises error on non-aggregate expressions."""
        # Create a non-aggregate column
        non_agg = col("amount")
        filter_cond = col("active") == True  # noqa: E712

        # Should raise ValueError
        with pytest.raises(
            ValueError, match="Filter clause can only be applied to aggregate functions"
        ):
            non_agg.filter(filter_cond)

    def test_filter_preserves_expression_structure(self):
        """Test that filter() preserves the expression structure."""
        from moltres.expressions import functions as F

        # Create an aggregate with filter
        agg_expr = F.sum(col("amount")).filter(col("active"))

        # Check that _filter is set (access via private attribute)
        assert agg_expr._filter is not None
        # The filter predicate is a Column (when using col("active") directly)
        filter_pred = agg_expr._filter
        assert isinstance(filter_pred, type(col("amount")))  # Column type
        assert filter_pred.op == "column"  # Direct column reference
        assert agg_expr.op == "agg_sum"

    def test_filter_with_alias_chain(self, tmp_path):
        """Test that filter() and alias() can be chained in any order."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "active": True},
                {"id": 2, "category": "A", "amount": 50, "active": False},
            ],
            pk="id",
        )

        # Test filter().alias()
        result1 = (
            df.group_by("category")
            .agg(F.sum(col("amount")).filter(col("active")).alias("total"))
            .collect()
        )

        # Test alias().filter() - should also work
        result2 = (
            df.group_by("category")
            .agg(F.sum(col("amount")).alias("total").filter(col("active")))
            .collect()
        )

        assert result1[0]["total"] == 100
        assert result2[0]["total"] == 100

    def test_filter_with_null_values(self, tmp_path):
        """Test FILTER clause with NULL values in filter condition."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "status": "active"},
                {"id": 2, "category": "A", "amount": 50, "status": None},
                {"id": 3, "category": "A", "amount": 200, "status": "inactive"},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(F.sum(col("amount")).filter(col("status") == "active").alias("active_sum"))
            .collect()
        )
        assert len(result) == 1
        assert result[0]["active_sum"] == 100

    def test_filter_with_multiple_conditions(self, tmp_path):
        """Test FILTER clause with multiple conditions using AND/OR."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")
        df = db.createDataFrame(
            [
                {"id": 1, "category": "A", "amount": 100, "active": True, "verified": True},
                {"id": 2, "category": "A", "amount": 50, "active": True, "verified": False},
                {"id": 3, "category": "A", "amount": 200, "active": False, "verified": True},
            ],
            pk="id",
        )
        result = (
            df.group_by("category")
            .agg(
                F.sum(col("amount"))
                .filter((col("active")) & (col("verified")))
                .alias("active_verified_sum")
            )
            .collect()
        )
        assert len(result) == 1
        # Only row 1 matches both conditions
        assert result[0]["active_verified_sum"] == 100
