import moltres.expressions.functions as F
from moltres.expressions import col, lit


def test_column_arithmetic_chain():
    expr = col("spend") * 1.1 - col("discount")
    assert expr.op == "sub"
    left = expr.args[0]
    assert left.op == "mul"
    assert left.args[0].source == "spend"
    assert left.args[1].op == "literal"


def test_column_comparisons_and_boolean_ops():
    expr = (col("active") == True) & ~(col("country") == "US")  # noqa: E712
    assert expr.op == "and"
    left = expr.args[0]
    assert left.op == "eq"
    right = expr.args[1]
    assert right.op == "not"


def test_literal_helper_creates_literal_expression():
    literal_expr = lit("value")
    assert literal_expr.op == "literal"
    assert literal_expr.args == ("value",)


def test_aggregate_helpers():
    assert callable(F.sum)
    agg = F.sum(col("amount")).alias("total")
    assert agg.op == "agg_sum"
    assert agg.alias_name == "total"
    assert F.avg("amount").op == "agg_avg"


def test_count_helpers():
    assert F.count().op == "agg_count_star"
    single = F.count(col("id"))
    assert single.op == "agg_count"
    distinct = F.count_distinct(col("country"), col("state"))
    assert distinct.op == "agg_count_distinct"
    assert len(distinct.args) == 2


def test_variadic_string_helpers():
    expr = F.concat(col("first"), " ", col("last"))
    assert expr.op == "concat"
    assert len(expr.args) == 3
    nz = F.coalesce(col("nickname"), col("first"))
    assert nz.op == "coalesce"
