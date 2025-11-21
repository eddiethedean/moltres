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


def test_isnull_isnotnull_aliases():
    """Test isnull() and isnotnull() aliases for is_null() and is_not_null()."""
    from moltres.expressions.functions import isnull, isnotnull

    # Test isnull() creates is_null operation
    isnull_expr = isnull(col("email"))
    assert isnull_expr.op == "is_null"
    assert isnull_expr.args[0].source == "email"

    # Test isnotnull() creates is_not_null operation
    isnotnull_expr = isnotnull(col("email"))
    assert isnotnull_expr.op == "is_not_null"
    assert isnotnull_expr.args[0].source == "email"

    # Verify they work the same as is_null() and is_not_null() methods
    assert isnull_expr.op == col("email").is_null().op
    assert isnotnull_expr.op == col("email").is_not_null().op


def test_cast_decimal_with_precision_scale_execution(tmp_path):
    """Test cast() to DECIMAL with precision and scale in actual query."""
    from moltres import connect, col

    db_path = tmp_path / "cast_decimal.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE prices (id INTEGER PRIMARY KEY, amount REAL)")
        conn.exec_driver_sql("INSERT INTO prices (id, amount) VALUES (1, 99.999), (2, 149.50)")

    # Test casting to DECIMAL with precision and scale
    df = db.table("prices").select(
        col("id"), col("amount").cast("DECIMAL", precision=10, scale=2).alias("amount_decimal")
    )

    result = df.collect()
    assert len(result) == 2
    # The cast should work (SQLite will handle it)
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
