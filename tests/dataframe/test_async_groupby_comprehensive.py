"""Comprehensive tests for AsyncGroupedDataFrame covering all operations."""

from __future__ import annotations

import pytest

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col
from moltres.expressions.functions import (
    sum as sum_func,
)


@pytest.mark.asyncio
class TestAsyncGroupedDataFrame:
    """Test AsyncGroupedDataFrame operations."""

    async def test_agg_with_column_expressions(self, tmp_path):
        """Test agg() with Column expressions."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [
                {"category": "A", "amount": 10.0},
                {"category": "A", "amount": 20.0},
                {"category": "B", "amount": 15.0},
            ],
            auto_pk="id",
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        grouped = df2.group_by("category")
        result = grouped.agg(sum_func(col("amount")).alias("total"))
        rows = await result.collect()

        assert len(rows) == 2

        await db.close()

    async def test_agg_with_string_columns(self, tmp_path):
        """Test agg() with string column names."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame([{"category": "A", "amount": 10.0}], auto_pk="id")
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        grouped = df2.group_by("category")
        result = grouped.agg("amount")
        rows = await result.collect()

        assert len(rows) == 1

        await db.close()

    async def test_agg_with_dict_syntax(self, tmp_path):
        """Test agg() with dictionary syntax."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("amount", "REAL"),
                column("price", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [{"category": "A", "amount": 10.0, "price": 5.0}], auto_pk="id"
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        grouped = df2.group_by("category")
        result = grouped.agg({"amount": "sum", "price": "avg"})
        rows = await result.collect()

        assert len(rows) == 1

        await db.close()

    async def test_agg_invalid_type(self, tmp_path):
        """Test agg() with invalid aggregation type raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales", [column("id", "INTEGER", primary_key=True), column("category", "TEXT")]
        ).collect()

        table = await db.table("sales")
        df = table.select()
        grouped = df.group_by("category")

        with pytest.raises(ValueError, match="Invalid aggregation type"):
            grouped.agg(123)  # type: ignore

        await db.close()

    async def test_create_aggregation_from_string(self, tmp_path):
        """_create_aggregation_from_string maps names to aggregation Column ops."""
        from moltres.dataframe.groupby.async_groupby import AsyncGroupedDataFrame

        expected_ops = {
            "sum": "agg_sum",
            "avg": "agg_avg",
            "average": "agg_avg",
            "min": "agg_min",
            "max": "agg_max",
            "count": "agg_count",
        }
        for func_name, expected_op in expected_ops.items():
            agg_col = AsyncGroupedDataFrame._create_aggregation_from_string("amount", func_name)
            assert agg_col.op == expected_op, f"{func_name} should produce op={expected_op}"

        # Behavioral: string agg path yields correct totals
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")
        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()
        df = await db.createDataFrame(
            [
                {"category": "A", "amount": 10.0},
                {"category": "A", "amount": 5.0},
                {"category": "B", "amount": 7.0},
            ],
            auto_pk="id",
        )
        await df.write.insertInto("sales")

        rows = (
            await (await db.table("sales"))
            .select()
            .group_by("category")
            .agg({"amount": "sum"})
            .order_by(col("category"))
            .collect()
        )
        by_cat = {r["category"]: float(r["amount"]) for r in rows}
        assert by_cat == {"A": 15.0, "B": 7.0}

        await db.close()

    async def test_create_aggregation_invalid_function(self, tmp_path):
        """Test _create_aggregation_from_string() with invalid function raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.dataframe.groupby.async_groupby import AsyncGroupedDataFrame

        with pytest.raises(ValueError, match="Unknown aggregation function"):
            AsyncGroupedDataFrame._create_aggregation_from_string("amount", "invalid_func")

        await db.close()

    async def test_pivot(self, tmp_path):
        """Test pivot() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("status", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [{"category": "A", "status": "active", "amount": 10.0}], auto_pk="id"
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        grouped = df2.group_by("category")
        pivoted = grouped.pivot("status")

        assert pivoted is not None

        await db.close()

    async def test_pivot_with_values(self, tmp_path):
        """Test pivot() with explicit values."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("status", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [{"category": "A", "status": "active", "amount": 10.0}], auto_pk="id"
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        grouped = df2.group_by("category")
        # pivot() returns AsyncPivotedGroupedDataFrame
        pivoted = grouped.pivot("status", values=["active", "inactive"])

        assert pivoted is not None
        assert pivoted.pivot_column == "status"
        assert pivoted.pivot_values == ("active", "inactive")

        await db.close()

    async def test_agg_without_grouping_error(self, tmp_path):
        """Test agg() without grouping raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.dataframe.groupby.async_groupby import AsyncGroupedDataFrame
        from moltres.logical import operators

        # Create a grouped DataFrame without proper grouping
        plan = operators.scan("test")
        grouped = AsyncGroupedDataFrame(plan=plan, database=db)

        with pytest.raises(ValueError, match="must have grouping columns"):
            grouped.agg(sum_func(col("amount")))

        await db.close()


@pytest.mark.asyncio
class TestAsyncPivotedGroupedDataFrame:
    """Test AsyncPivotedGroupedDataFrame operations."""

    async def test_pivoted_agg(self, tmp_path):
        """Test pivoted agg() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("status", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [
                {"category": "A", "status": "active", "amount": 10.0},
                {"category": "A", "status": "inactive", "amount": 5.0},
            ],
            auto_pk="id",
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        # Provide explicit pivot values to avoid inference issues
        pivoted = df2.group_by("category").pivot("status", values=["active", "inactive"])
        result = await pivoted.agg("amount")
        rows = await result.collect()

        assert len(rows) == 1
        row = rows[0]
        assert row["category"] == "A"
        # Pivot columns for active/inactive amounts
        active_key = next(
            k for k in row if "active" in str(k).lower() and "inactive" not in str(k).lower()
        )
        inactive_key = next(k for k in row if "inactive" in str(k).lower())
        assert float(row[active_key]) == 10.0
        assert float(row[inactive_key]) == 5.0

        await db.close()

    async def test_pivoted_agg_with_column(self, tmp_path):
        """Test pivoted agg() with Column expression."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("status", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [
                {"category": "A", "status": "active", "amount": 10.0},
                {"category": "A", "status": "inactive", "amount": 5.0},
            ],
            auto_pk="id",
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        pivoted = df2.group_by("category").pivot("status", values=["active", "inactive"])
        result = await pivoted.agg(sum_func(col("amount")))
        rows = await result.collect()
        assert len(rows) == 1
        assert rows[0]["category"] == "A"

        await db.close()

    async def test_pivoted_agg_with_dict(self, tmp_path):
        """Test pivoted agg() with dictionary syntax."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "TEXT"),
                column("status", "TEXT"),
                column("amount", "REAL"),
            ],
        ).collect()

        df = await db.createDataFrame(
            [
                {"category": "A", "status": "active", "amount": 10.0},
                {"category": "A", "status": "inactive", "amount": 5.0},
            ],
            auto_pk="id",
        )
        await df.write.insertInto("sales")

        table = await db.table("sales")
        df2 = table.select()
        pivoted = df2.group_by("category").pivot("status", values=["active", "inactive"])
        result = await pivoted.agg({"amount": "sum"})
        rows = await result.collect()
        assert len(rows) == 1
        assert rows[0]["category"] == "A"

        await db.close()

    async def test_pivoted_agg_invalid_type(self, tmp_path):
        """Test pivoted agg() with invalid type raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "sales", [column("category", "TEXT"), column("status", "TEXT")]
        ).collect()

        table = await db.table("sales")
        df = table.select()
        pivoted = df.group_by("category").pivot("status")

        with pytest.raises(ValueError, match="Invalid aggregation type"):
            await pivoted.agg(123)  # type: ignore

        await db.close()
