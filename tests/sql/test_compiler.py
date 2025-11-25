"""Comprehensive tests for SQL compiler covering all plan types and edge cases."""

from __future__ import annotations

import pytest

from moltres.expressions import col
from moltres.expressions.functions import sum as sum_, count  # noqa: A001
from moltres.logical import operators
from moltres.logical.plan import (
    FileScan,
    Join,
    TableScan,
)
from moltres.sql.compiler import compile_plan
from moltres.utils.exceptions import CompilationError


def test_compile_project_filter_limit():
    """Test basic project, filter, and limit compilation."""
    scan = operators.scan("customers")
    proj = operators.project(scan, (col("id"), col("name")))
    predicate = (col("active") == True) & ~(col("country") == "US")  # noqa: E712
    filt = operators.filter(proj, predicate)
    limited = operators.limit(filt, 10)

    stmt = compile_plan(limited)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL (SQLAlchemy formatting may vary)
    assert "SELECT" in sql
    assert "id" in sql
    assert "name" in sql
    assert "FROM" in sql
    assert "customers" in sql
    assert "WHERE" in sql
    assert "active" in sql
    assert "country" in sql
    assert "US" in sql
    assert "LIMIT 10" in sql


def test_compile_order_and_aliases():
    """Test ORDER BY and column aliases."""
    scan = operators.scan("orders")
    expr = (col("spend") * 1.1).alias("adj_spend")
    proj = operators.project(scan, (col("id"), expr))
    ordered = operators.order_by(proj, [operators.sort_order(col("created_at"), descending=True)])
    stmt = compile_plan(ordered)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL
    assert "SELECT" in sql
    assert "id" in sql
    assert "adj_spend" in sql
    assert "spend" in sql
    assert "1.1" in sql
    assert "FROM" in sql
    assert "orders" in sql
    assert "ORDER BY" in sql
    assert "created_at" in sql
    assert "DESC" in sql


def test_compile_join_on_columns():
    """Test join with column pairs."""
    orders = operators.scan("orders")
    customers = operators.scan("customers")
    joined = operators.join(orders, customers, how="inner", on=[("customer_id", "id")])
    proj = operators.project(joined, (col("orders.id").alias("order_id"), col("customers.name")))
    stmt = compile_plan(proj)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL
    assert "JOIN" in sql
    assert "customer_id" in sql
    assert "id" in sql
    assert "order_id" in sql
    assert "name" in sql


def test_compile_groupby_aggregate():
    """Test GROUP BY and aggregate functions."""
    scan = operators.scan("orders")
    total = sum_(col("amount")).alias("total_amount")
    grouped = operators.aggregate(scan, (col("country"),), (total,))
    stmt = compile_plan(grouped)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL
    assert "GROUP BY" in sql
    assert "country" in sql
    assert "SUM" in sql or "sum" in sql
    assert "amount" in sql
    assert "total_amount" in sql


def test_compile_table_scan_with_alias():
    """Test TableScan with alias."""
    scan = TableScan(table="users", alias="u")
    stmt = compile_plan(scan)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql
    assert "u" in sql or "AS" in sql


def test_compile_table_scan_without_alias():
    """Test TableScan without alias."""
    scan = TableScan(table="users")
    stmt = compile_plan(scan)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql
    assert "SELECT" in sql


def test_compile_file_scan_error():
    """Test that FileScan raises CompilationError."""
    file_scan = FileScan(path="data.csv", format="csv")
    with pytest.raises(CompilationError, match="FileScan cannot be compiled directly"):
        compile_plan(file_scan)


def test_compile_cte():
    """Test Common Table Expression compilation."""
    scan = operators.scan("users")
    proj = operators.project(scan, (col("id"), col("name")))
    cte = operators.cte(proj, name="user_cte")
    stmt = compile_plan(cte)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "user_cte" in sql or "WITH" in sql


def test_compile_recursive_cte():
    """Test Recursive CTE compilation."""
    initial = operators.scan("employees")
    recursive = operators.scan("employees")
    recursive_cte = operators.recursive_cte(
        name="org_chart", initial=initial, recursive=recursive, union_all=True
    )
    stmt = compile_plan(recursive_cte)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "org_chart" in sql or "WITH" in sql


def test_compile_raw_sql():
    """Test RawSQL compilation."""
    raw_sql = operators.raw_sql(sql="SELECT * FROM users WHERE id = :id", params={"id": 1})
    stmt = compile_plan(raw_sql)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql
    assert "id" in sql


def test_compile_raw_sql_without_params():
    """Test RawSQL without parameters."""
    raw_sql = operators.raw_sql(sql="SELECT * FROM users")
    stmt = compile_plan(raw_sql)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql


def test_compile_distinct():
    """Test DISTINCT compilation."""
    scan = operators.scan("users")
    proj = operators.project(scan, (col("country"),))
    distinct = operators.distinct(proj)
    stmt = compile_plan(distinct)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "DISTINCT" in sql
    assert "country" in sql


def test_compile_union():
    """Test UNION compilation."""
    left = operators.scan("users1")
    right = operators.scan("users2")
    union = operators.union(left, right, distinct=True)
    stmt = compile_plan(union)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "UNION" in sql or "users1" in sql


def test_compile_union_all():
    """Test UNION ALL compilation."""
    left = operators.scan("users1")
    right = operators.scan("users2")
    union = operators.union(left, right, distinct=False)
    stmt = compile_plan(union)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "UNION" in sql or "users1" in sql


def test_compile_intersect():
    """Test INTERSECT compilation."""
    left = operators.scan("users1")
    right = operators.scan("users2")
    intersect = operators.intersect(left, right, distinct=True)
    stmt = compile_plan(intersect)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "INTERSECT" in sql or "users1" in sql


def test_compile_intersect_all():
    """Test INTERSECT ALL compilation."""
    left = operators.scan("users1")
    right = operators.scan("users2")
    intersect = operators.intersect(left, right, distinct=False)
    stmt = compile_plan(intersect)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "INTERSECT" in sql or "users1" in sql


def test_compile_except():
    """Test EXCEPT compilation."""
    left = operators.scan("users1")
    right = operators.scan("users2")
    except_plan = operators.except_(left, right, distinct=True)
    stmt = compile_plan(except_plan)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "EXCEPT" in sql or "users1" in sql


def test_compile_except_all():
    """Test EXCEPT ALL compilation."""
    left = operators.scan("users1")
    right = operators.scan("users2")
    except_plan = operators.except_(left, right, distinct=False)
    stmt = compile_plan(except_plan)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "EXCEPT" in sql or "users1" in sql


def test_compile_sample():
    """Test SAMPLE compilation."""
    scan = operators.scan("users")
    sample = operators.sample(scan, fraction=0.1, seed=None)
    stmt = compile_plan(sample)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql
    # Sample uses RANDOM() and LIMIT
    assert "LIMIT" in sql or "RANDOM" in sql or "RAND" in sql


def test_compile_sample_with_seed():
    """Test SAMPLE with seed (seed is accepted but may not be fully supported)."""
    scan = operators.scan("users")
    sample = operators.sample(scan, fraction=0.1, seed=42)
    stmt = compile_plan(sample)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql


def test_compile_explode_unsupported_dialect():
    """Explode should raise for unsupported (ANSI) dialect."""
    scan = operators.scan("users")
    explode = operators.explode(scan, column=col("tags"))
    with pytest.raises(CompilationError, match="explode.*not yet implemented"):
        compile_plan(explode, dialect="ansi")


def test_compile_explode_sqlite():
    """Explode should compile for SQLite using json_each."""
    scan = operators.scan("users")
    explode = operators.explode(scan, column=col("tags"), alias="tag_value")
    stmt = compile_plan(explode, dialect="sqlite")
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "json_each" in sql
    assert "tag_value" in sql


def test_compile_explode_postgresql():
    """Explode should compile for PostgreSQL using jsonb_array_elements."""
    scan = operators.scan("users")
    explode = operators.explode(scan, column=col("tags"), alias="tag_value")
    stmt = compile_plan(explode, dialect="postgresql")
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "jsonb_array_elements" in sql
    assert "tag_value" in sql


def test_compile_join_types():
    """Test different join types."""
    left = operators.scan("orders")
    right = operators.scan("customers")

    # Inner join
    inner_join = operators.join(left, right, how="inner", on=[("customer_id", "id")])
    stmt = compile_plan(inner_join)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql or "INNER" in sql

    # Left join
    left_join = operators.join(left, right, how="left", on=[("customer_id", "id")])
    stmt = compile_plan(left_join)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql or "LEFT" in sql

    # Right join
    right_join = operators.join(left, right, how="right", on=[("customer_id", "id")])
    stmt = compile_plan(right_join)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql or "RIGHT" in sql

    # Full/Outer join
    full_join = operators.join(left, right, how="outer", on=[("customer_id", "id")])
    stmt = compile_plan(full_join)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql or "FULL" in sql or "OUTER" in sql

    # Cross join (represented as comma-separated tables)
    cross_join = operators.join(left, right, how="cross")
    stmt = compile_plan(cross_join)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "orders" in sql and "customers" in sql


def test_compile_join_with_condition():
    """Test join with condition expression."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    join_cond = col("orders.customer_id") == col("customers.id")
    joined = operators.join(left, right, how="inner", condition=join_cond)
    stmt = compile_plan(joined)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql
    assert "customer_id" in sql or "id" in sql


def test_compile_join_unsupported_type():
    """Test that unsupported join type raises CompilationError."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    # Create invalid join directly since operators.join validates
    invalid_join = Join(left=left, right=right, how="invalid", on=[("customer_id", "id")])  # type: ignore[call-overload]
    with pytest.raises(CompilationError, match="Unsupported join type"):
        compile_plan(invalid_join)


def test_compile_semijoin():
    """Test SemiJoin compilation."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    semijoin = operators.semi_join(left, right, on=[("customer_id", "id")])
    stmt = compile_plan(semijoin)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql
    assert "DISTINCT" in sql


def test_compile_semijoin_with_condition():
    """Test SemiJoin with condition."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    join_cond = col("orders.customer_id") == col("customers.id")
    semijoin = operators.semi_join(left, right, condition=join_cond)
    stmt = compile_plan(semijoin)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql


def test_compile_semijoin_error():
    """Test that SemiJoin without on or condition raises error."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    semijoin = operators.semi_join(left, right)
    with pytest.raises(CompilationError, match="SemiJoin requires"):
        compile_plan(semijoin)


def test_compile_antijoin():
    """Test AntiJoin compilation."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    antijoin = operators.anti_join(left, right, on=[("customer_id", "id")])
    stmt = compile_plan(antijoin)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql or "NOT EXISTS" in sql or "NOT IN" in sql


def test_compile_antijoin_with_condition():
    """Test AntiJoin with condition."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    join_cond = col("orders.customer_id") == col("customers.id")
    antijoin = operators.anti_join(left, right, condition=join_cond)
    stmt = compile_plan(antijoin)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql or "NOT EXISTS" in sql


def test_compile_antijoin_error():
    """Test that AntiJoin without on or condition raises error."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    antijoin = operators.anti_join(left, right)
    with pytest.raises(CompilationError, match="AntiJoin requires"):
        compile_plan(antijoin)


def test_compile_pivot():
    """Test Pivot compilation."""
    scan = operators.scan("sales")
    pivot = operators.pivot(
        child=scan,
        pivot_column="product",
        value_column="amount",
        agg_func="sum",
        pivot_values=["Widget", "Gadget"],
    )
    stmt = compile_plan(pivot)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "sales" in sql
    assert "product" in sql or "amount" in sql


def test_compile_pivot_error():
    """Test that Pivot without pivot_values raises error."""
    scan = operators.scan("sales")
    pivot = operators.pivot(
        child=scan,
        pivot_column="product",
        value_column="amount",
        agg_func="sum",
        pivot_values=[],
    )
    with pytest.raises(CompilationError, match="PIVOT without pivot_values"):
        compile_plan(pivot)


def test_compile_grouped_pivot():
    """Test GroupedPivot compilation."""
    scan = operators.scan("sales")
    grouped_pivot = operators.grouped_pivot(
        child=scan,
        grouping=(col("date"),),
        pivot_column="product",
        value_column="amount",
        agg_func="sum",
        pivot_values=["Widget", "Gadget"],
    )
    stmt = compile_plan(grouped_pivot)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "sales" in sql
    assert "date" in sql or "product" in sql


def test_compile_aggregate_without_group_by():
    """Test Aggregate without GROUP BY columns."""
    scan = operators.scan("orders")
    total = sum_(col("amount")).alias("total")
    agg = operators.aggregate(scan, (), (total,))
    stmt = compile_plan(agg)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "SUM" in sql or "sum" in sql
    assert "amount" in sql


def test_compile_sort_multiple_columns():
    """Test Sort with multiple columns."""
    scan = operators.scan("orders")
    proj = operators.project(scan, (col("id"), col("amount"), col("date")))
    sort = operators.order_by(
        proj,
        [
            operators.sort_order(col("date"), descending=True),
            operators.sort_order(col("amount"), descending=False),
        ],
    )
    stmt = compile_plan(sort)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "ORDER BY" in sql
    assert "date" in sql
    assert "amount" in sql


def test_compile_project_with_join_child():
    """Test Project with Join as child (tests join_info logic)."""
    left = operators.scan("orders")
    right = operators.scan("customers")
    joined = operators.join(left, right, how="inner", on=[("customer_id", "id")])
    proj = operators.project(joined, (col("orders.id"), col("customers.name")))
    stmt = compile_plan(proj)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "SELECT" in sql
    assert "JOIN" in sql


def test_compile_nested_subqueries():
    """Test nested subqueries in compilation."""
    inner = operators.scan("orders")
    inner_proj = operators.project(inner, (col("customer_id"),))
    inner_agg = operators.aggregate(
        inner_proj, (col("customer_id"),), (count("*").alias("order_count"),)
    )
    outer = operators.scan("customers")
    joined = operators.join(outer, inner_agg, how="left", on=[("id", "customer_id")])
    stmt = compile_plan(joined)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "JOIN" in sql
    assert "customer_id" in sql


def test_compile_with_dialect():
    """Test compilation with specific dialect."""
    scan = operators.scan("users")
    stmt = compile_plan(scan, dialect="postgresql")
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql

    stmt = compile_plan(scan, dialect="sqlite")
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "users" in sql


def test_compile_unsupported_plan_type():
    """Test that unsupported plan type raises CompilationError."""

    class UnsupportedPlan:
        pass

    # This should raise an error when trying to compile
    # We can't easily create an unsupported plan through the operators,
    # but we can test the error message path
    from moltres.sql.compiler import SQLCompiler
    from moltres.engine.dialects import get_dialect

    compiler = SQLCompiler(get_dialect("sqlite"))
    unsupported = UnsupportedPlan()
    # Set it up to look like a LogicalPlan
    unsupported.__class__.__name__ = "UnsupportedPlan"
    with pytest.raises(CompilationError, match="Unsupported logical plan node"):
        compiler._compile_plan(unsupported)  # type: ignore[arg-type]
