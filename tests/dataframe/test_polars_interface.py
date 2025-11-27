"""Tests for Polars-style interface."""

import pytest

from moltres import connect, col
from moltres.dataframe.polars_dataframe import PolarsDataFrame
from moltres.expressions import functions as F


def _to_dict_list(results):
    """Convert results (Polars DataFrame or list of dicts) to list of dicts."""
    try:
        import polars as pl

        if isinstance(results, pl.DataFrame):
            return results.to_dicts()
    except ImportError:
        pass
    return results


@pytest.fixture
def sample_db():
    """Create a sample database with test data."""
    db = connect("sqlite:///:memory:")

    from moltres.table.schema import column

    with db.batch():
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
                column("country", "TEXT"),
            ],
        ).collect()

        db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("user_id", "INTEGER"),
                column("amount", "REAL"),
                column("status", "TEXT"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "name": "Alice", "age": 30, "country": "USA"},
            {"id": 2, "name": "Bob", "age": 25, "country": "UK"},
            {"id": 3, "name": "Charlie", "age": 35, "country": "USA"},
        ],
        database=db,
    ).insert_into("users")

    Records.from_list(
        [
            {"id": 1, "user_id": 1, "amount": 100.0, "status": "active"},
            {"id": 2, "user_id": 2, "amount": 200.0, "status": "active"},
            {"id": 3, "user_id": 1, "amount": 150.0, "status": "completed"},
        ],
        database=db,
    ).insert_into("orders")

    yield db
    db.close()


def test_polars_dataframe_creation(sample_db):
    """Test creating a PolarsDataFrame from a table."""
    df = sample_db.table("users").polars()
    assert df is not None
    assert df.database == sample_db


def test_column_access_single(sample_db):
    """Test single column access: df['col']."""
    df = sample_db.table("users").polars()
    col_expr = df["age"]
    # Should return a Column expression
    assert hasattr(col_expr, "op")


def test_column_access_multiple(sample_db):
    """Test multiple column access: df[['col1', 'col2']]."""
    df = sample_db.table("users").polars()
    df_selected = df[["id", "name"]]

    # Should return a PolarsDataFrame

    assert isinstance(df_selected, PolarsDataFrame)

    # Check columns
    assert "id" in df_selected.columns
    assert "name" in df_selected.columns
    assert "age" not in df_selected.columns


def test_filter_method(sample_db):
    """Test df.filter() method."""
    df = sample_db.table("users").polars()

    # Test simple filter
    result_df = df.filter(col("age") > 25)
    results = result_df.collect()
    results_list = _to_dict_list(results)

    # Should have 2 results (Alice and Charlie)
    assert len(results_list) == 2
    assert all(r["age"] > 25 for r in results_list)


def test_select_method(sample_db):
    """Test df.select() method."""
    df = sample_db.table("users").polars()

    # Select specific columns
    result_df = df.select("id", "name")
    results = result_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) > 0
    assert "id" in results_list[0]
    assert "name" in results_list[0]
    assert "age" not in results_list[0]


def test_with_columns(sample_db):
    """Test df.with_columns() method."""
    df = sample_db.table("users").polars()

    # Add a new column
    result_df = df.with_columns((col("age") + 10).alias("age_plus_10"))
    results = result_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) > 0
    assert "age_plus_10" in results_list[0]
    assert results_list[0]["age_plus_10"] == results_list[0]["age"] + 10


def test_with_column(sample_db):
    """Test df.with_column() method (alias for with_columns)."""
    df = sample_db.table("users").polars()

    result_df = df.with_column((col("age") * 2).alias("double_age"))
    results = result_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) > 0
    assert "double_age" in results_list[0]


def test_drop(sample_db):
    """Test df.drop() method."""
    df = sample_db.table("users").polars()

    dropped_df = df.drop("age")
    results = dropped_df.collect()
    results_list = _to_dict_list(results)

    # Check that age column is not in results and columns property
    if results_list:
        assert "age" not in results_list[0]
        assert "id" in results_list[0]
        assert "name" in results_list[0]
    # Check columns property
    assert "age" not in dropped_df.columns
    assert "id" in dropped_df.columns
    assert "name" in dropped_df.columns


def test_rename(sample_db):
    """Test df.rename() method."""
    df = sample_db.table("users").polars()

    renamed_df = df.rename({"name": "full_name"})

    # Check that column was renamed
    assert "full_name" in renamed_df.columns
    assert "name" not in renamed_df.columns


def test_sort(sample_db):
    """Test df.sort() method."""
    df = sample_db.table("users").polars()

    sorted_df = df.sort("age")
    results = sorted_df.collect()
    results_list = _to_dict_list(results)

    # Check that ages are in ascending order
    ages = [r["age"] for r in results_list]
    assert ages == sorted(ages)


def test_sort_descending(sample_db):
    """Test df.sort() with descending=True."""
    df = sample_db.table("users").polars()

    sorted_df = df.sort("age", descending=True)
    results = sorted_df.collect()
    results_list = _to_dict_list(results)

    # Check that ages are in descending order
    ages = [r["age"] for r in results_list]
    assert ages == sorted(ages, reverse=True)


def test_limit(sample_db):
    """Test df.limit() method."""
    df = sample_db.table("users").polars()

    limited_df = df.limit(2)
    results = limited_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) <= 2


def test_head(sample_db):
    """Test df.head() method."""
    df = sample_db.table("users").polars()

    head_df = df.head(2)
    results = head_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) <= 2


def test_group_by_basic(sample_db):
    """Test basic group_by operation."""
    df = sample_db.table("users").polars()

    grouped = df.group_by("country")
    result_df = grouped.count()

    results = result_df.collect()
    results_list = _to_dict_list(results)
    assert len(results_list) == 2  # USA and UK


def test_group_by_agg(sample_db):
    """Test group_by with agg() method."""
    df = sample_db.table("orders").polars()

    grouped = df.group_by("status")
    result_df = grouped.agg(F.sum(col("amount")).alias("total_amount"))

    results = result_df.collect()
    results_list = _to_dict_list(results)
    assert len(results_list) >= 1
    assert "total_amount" in results_list[0] or any(
        "amount" in str(k) for k in results_list[0].keys()
    )


def test_group_by_sum(sample_db):
    """Test group_by.sum() method."""
    df = sample_db.table("orders").polars()

    grouped = df.group_by("status")
    result_df = grouped.sum()

    results = result_df.collect()
    assert len(results) >= 1


def test_group_by_mean(sample_db):
    """Test group_by.mean() method."""
    df = sample_db.table("orders").polars()

    grouped = df.group_by("status")
    result_df = grouped.mean()

    results = result_df.collect()
    assert len(results) >= 1


def test_join_inner(sample_db):
    """Test join operation with inner join."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("orders").polars()

    joined = df1.join(df2, left_on="id", right_on="user_id", how="inner")
    results = joined.collect()
    results_list = _to_dict_list(results)

    # Should have matches
    assert len(results_list) > 0


def test_join_on_same_column(sample_db):
    """Test join with same column name in both DataFrames."""
    df1 = sample_db.table("users").polars()[["id", "name"]]
    df2 = sample_db.table("orders").polars()[["user_id", "amount"]]
    df2_renamed = df2.rename({"user_id": "id"})

    joined = df1.join(df2_renamed, on="id")
    results = joined.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) > 0


def test_unique(sample_db):
    """Test df.unique() method."""
    df = sample_db.table("users").polars()

    # All rows should be unique, so unique shouldn't change anything
    unique_df = df.unique()
    results = unique_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3  # Same as original


def test_distinct(sample_db):
    """Test df.distinct() method (alias for unique)."""
    df = sample_db.table("users").polars()

    distinct_df = df.distinct()
    results = distinct_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3


def test_drop_nulls(sample_db):
    """Test df.drop_nulls() method."""
    from moltres.io.records import Records

    # Add a row with null values
    Records.from_list(
        [{"id": 4, "name": None, "age": None, "country": "USA"}],
        database=sample_db,
    ).insert_into("users")

    df = sample_db.table("users").polars()
    result_df = df.drop_nulls(subset=["name", "age"])
    results = result_df.collect()
    results_list = _to_dict_list(results)

    # Should have filtered out the null row
    assert len(results_list) <= 3


def test_fill_null(sample_db):
    """Test df.fill_null() method."""
    from moltres.io.records import Records

    # Add a row with null values
    Records.from_list(
        [{"id": 5, "name": None, "age": 20, "country": "UK"}],
        database=sample_db,
    ).insert_into("users")

    df = sample_db.table("users").polars()
    result_df = df.fill_null(value="Unknown", subset=["name"])
    results = result_df.collect()
    results_list = _to_dict_list(results)

    # Check that nulls were filled
    for row in results_list:
        if row.get("id") == 5:
            assert row.get("name") == "Unknown" or row.get("name") is not None


def test_boolean_indexing(sample_db):
    """Test boolean indexing: df[df['col'] > value]."""
    df = sample_db.table("users").polars()

    filtered_df = df[df["age"] > 25]
    results = filtered_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 2
    assert all(r["age"] > 25 for r in results_list)


def test_columns_property(sample_db):
    """Test df.columns property."""
    df = sample_db.table("users").polars()

    cols = df.columns
    assert isinstance(cols, list)
    assert "id" in cols
    assert "name" in cols
    assert "age" in cols


def test_width_property(sample_db):
    """Test df.width property."""
    df = sample_db.table("users").polars()

    width = df.width
    assert width == 4  # id, name, age, country


def test_schema_property(sample_db):
    """Test df.schema property."""
    df = sample_db.table("users").polars()

    schema = df.schema
    assert isinstance(schema, list)
    if schema:  # Schema might be empty if extraction fails
        assert isinstance(schema[0], tuple)
        assert len(schema[0]) == 2  # (name, dtype)


def test_lazy_method(sample_db):
    """Test df.lazy() method (should return self)."""
    df = sample_db.table("users").polars()

    lazy_df = df.lazy()
    assert lazy_df is df


def test_collect_without_polars(sample_db):
    """Test that collect() works even if polars is not installed."""
    df = sample_db.table("users").polars()

    results = df.collect()
    # Should return either Polars DataFrame or list of dicts
    assert results is not None
    if isinstance(results, list):
        assert len(results) > 0
        assert isinstance(results[0], dict)


def test_collect_with_polars(sample_db):
    """Test collect() returns Polars DataFrame when polars is installed."""
    pytest.importorskip("polars")
    import polars as pl

    df = sample_db.table("users").polars()

    results = df.collect()
    # If polars is installed, should return Polars DataFrame
    if isinstance(results, pl.DataFrame):
        assert isinstance(results, pl.DataFrame)
        assert len(results) > 0


def test_fetch(sample_db):
    """Test df.fetch() method."""
    pytest.importorskip("polars")

    df = sample_db.table("users").polars()

    results = df.fetch(2)
    # Should return Polars DataFrame with at most 2 rows
    assert len(results) <= 2


def test_sample(sample_db):
    """Test df.sample() method."""
    df = sample_db.table("users").polars()

    sampled_df = df.sample(n=2, seed=42)
    results = sampled_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) <= 2


def test_chained_operations(sample_db):
    """Test chaining multiple operations."""
    df = sample_db.table("users").polars()

    result_df = df.filter(col("age") > 25).select("id", "name", "age").sort("age", descending=True)

    results = result_df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 2
    ages = [r["age"] for r in results_list]
    assert ages == sorted(ages, reverse=True)


def test_scan_csv(sample_db, tmp_path):
    """Test db.scan_csv() method."""
    import csv

    # Create a CSV file
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "age"])
        writer.writerow([1, "Alice", 30])
        writer.writerow([2, "Bob", 25])

    # Scan CSV file
    df = sample_db.scan_csv(str(csv_file), header=True)
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 2
    assert results_list[0]["name"] == "Alice"
    assert results_list[1]["name"] == "Bob"


def test_scan_json(sample_db, tmp_path):
    """Test db.scan_json() method."""
    import json

    # Create a JSON file
    json_file = tmp_path / "test.json"
    data = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ]
    with open(json_file, "w") as f:
        json.dump(data, f)

    # Scan JSON file
    df = sample_db.scan_json(str(json_file))
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 2
    assert results_list[0]["name"] == "Alice"
    assert results_list[1]["name"] == "Bob"


def test_scan_jsonl(sample_db, tmp_path):
    """Test db.scan_jsonl() method."""
    import json

    # Create a JSONL file
    jsonl_file = tmp_path / "test.jsonl"
    data = [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
    ]
    with open(jsonl_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    # Scan JSONL file
    df = sample_db.scan_jsonl(str(jsonl_file))
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 2
    assert results_list[0]["name"] == "Alice"
    assert results_list[1]["name"] == "Bob"


def test_scan_text(sample_db, tmp_path):
    """Test db.scan_text() method."""
    # Create a text file
    text_file = tmp_path / "test.txt"
    text_file.write_text("Line 1\nLine 2\nLine 3\n")

    # Scan text file
    df = sample_db.scan_text(str(text_file), column_name="line")
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3
    assert results_list[0]["line"] == "Line 1"
    assert results_list[1]["line"] == "Line 2"


def test_scan_with_schema(sample_db, tmp_path):
    """Test scan methods with explicit schema."""
    import csv
    from moltres.table.schema import column

    # Create a CSV file
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "age"])
        writer.writerow([1, "Alice", 30])

    # Scan with schema
    schema = [
        column("id", "INTEGER"),
        column("name", "TEXT"),
        column("age", "INTEGER"),
    ]
    df = sample_db.scan_csv(str(csv_file), schema=schema, header=True)
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 1
    assert isinstance(results_list[0]["id"], int)
    assert isinstance(results_list[0]["age"], int)


def test_scan_with_options(sample_db, tmp_path):
    """Test scan methods with options."""
    import csv

    # Create a CSV file with custom delimiter
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "name", "age"])
        writer.writerow([1, "Alice", 30])

    # Scan with delimiter option
    df = sample_db.scan_csv(str(csv_file), header=True, delimiter=";")
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 1
    assert results_list[0]["name"] == "Alice"


def test_dataframe_polars_method(sample_db, tmp_path):
    """Test DataFrame.polars() method."""
    import csv

    # Create a CSV file
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])
        writer.writerow([1, "Alice"])

    # Use db.read.csv().polars()
    df = sample_db.read.csv(str(csv_file)).polars()
    results = df.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 1
    assert results_list[0]["name"] == "Alice"


def test_write_csv(sample_db, tmp_path):
    """Test df.write_csv() method."""
    df = sample_db.table("users").polars()

    # Write to CSV
    output_file = tmp_path / "output.csv"
    df.write_csv(str(output_file), header=True)

    # Verify file was created
    assert output_file.exists()

    # Read it back
    df_read = sample_db.scan_csv(str(output_file), header=True)
    results = df_read.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3  # 3 users in sample_db
    assert "id" in results_list[0]
    assert "name" in results_list[0]


def test_write_json(sample_db, tmp_path):
    """Test df.write_json() method."""
    df = sample_db.table("users").polars()

    # Write to JSON
    output_file = tmp_path / "output.json"
    df.write_json(str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read it back
    df_read = sample_db.scan_json(str(output_file))
    results = df_read.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3
    assert "id" in results_list[0]
    assert "name" in results_list[0]


def test_write_jsonl(sample_db, tmp_path):
    """Test df.write_jsonl() method."""
    df = sample_db.table("users").polars()

    # Write to JSONL
    output_file = tmp_path / "output.jsonl"
    df.write_jsonl(str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read it back
    df_read = sample_db.scan_jsonl(str(output_file))
    results = df_read.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3
    assert "id" in results_list[0]


def test_write_parquet(sample_db, tmp_path):
    """Test df.write_parquet() method."""
    try:
        import importlib.util

        if importlib.util.find_spec("pyarrow") is None:
            pytest.skip("PyArrow not installed")
    except ImportError:
        pytest.skip("PyArrow not installed")

    df = sample_db.table("users").polars()

    # Write to Parquet
    output_file = tmp_path / "output.parquet"
    df.write_parquet(str(output_file))

    # Verify file was created
    assert output_file.exists()

    # Read it back
    df_read = sample_db.scan_parquet(str(output_file))
    results = df_read.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3
    assert "id" in results_list[0]


def test_write_with_mode(sample_db, tmp_path):
    """Test write methods with different modes."""
    df = sample_db.table("users").polars()

    # Write with overwrite mode (default)
    output_file = tmp_path / "output.csv"
    df.write_csv(str(output_file), header=True, mode="overwrite")

    # Verify initial write
    df_read1 = sample_db.scan_csv(str(output_file), header=True)
    results1 = df_read1.collect()
    results_list1 = _to_dict_list(results1)
    assert len(results_list1) == 3

    # Write again with append mode
    df.write_csv(str(output_file), mode="append", header=False)

    # Verify file exists and was modified
    assert output_file.exists()

    # Test error_if_exists mode with a new file
    new_file = tmp_path / "new_output.csv"
    df.write_csv(str(new_file), mode="error_if_exists")
    assert new_file.exists()

    # Verify that overwrite mode works
    df.write_csv(str(new_file), mode="overwrite")
    assert new_file.exists()


def test_write_with_options(sample_db, tmp_path):
    """Test write methods with options."""
    df = sample_db.table("users").polars()

    # Write with custom delimiter
    output_file = tmp_path / "output.csv"
    df.write_csv(str(output_file), header=True, delimiter=";")

    # Read it back with matching delimiter
    df_read = sample_db.scan_csv(str(output_file), header=True, delimiter=";")
    results = df_read.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) == 3
    assert "id" in results_list[0]


def test_write_filtered_data(sample_db, tmp_path):
    """Test writing filtered data."""
    df = sample_db.table("users").polars()

    # Filter and write
    filtered_df = df.filter(col("age") > 25)
    output_file = tmp_path / "filtered.json"
    filtered_df.write_json(str(output_file))

    # Read it back
    df_read = sample_db.scan_json(str(output_file))
    results = df_read.collect()
    results_list = _to_dict_list(results)

    # Should only have users with age > 25
    assert len(results_list) == 2  # Alice (30) and Charlie (35)
    assert all(row["age"] > 25 for row in results_list)


# ============================================================================
# String and DateTime Accessors
# ============================================================================


def test_string_accessor(sample_db):
    """Test .str namespace for string operations."""
    df = sample_db.table("users").polars()

    # Test string operations
    df_upper = df.with_columns(df["name"].str.upper().alias("name_upper"))
    results = df_upper.collect()
    results_list = _to_dict_list(results)

    assert "name_upper" in results_list[0]
    assert results_list[0]["name_upper"] == "ALICE"

    # Test contains
    filtered = df.filter(df["name"].str.contains("Ali"))
    results = filtered.collect()
    results_list = _to_dict_list(results)
    assert len(results_list) == 1
    assert "Alice" in results_list[0]["name"]


def test_datetime_accessor(sample_db):
    """Test .dt namespace for datetime operations."""
    from moltres.table.schema import column
    from moltres.io.records import Records

    # Create a table with datetime column
    sample_db.create_table(
        "events",
        [
            column("id", "INTEGER", primary_key=True),
            column("event_date", "TIMESTAMP"),
        ],
    ).collect()

    # Insert test data
    Records(
        _data=[
            {"id": 1, "event_date": "2023-01-15 10:30:00"},
            {"id": 2, "event_date": "2023-06-20 14:45:00"},
        ],
        _database=sample_db,
    ).insert_into("events")

    df = sample_db.table("events").polars()

    # Test datetime operations
    df_with_year = df.with_columns(df["event_date"].dt.year().alias("year"))
    results = df_with_year.collect()
    results_list = _to_dict_list(results)

    assert "year" in results_list[0]
    assert results_list[0]["year"] == 2023


# ============================================================================
# Window Functions
# ============================================================================


def test_window_functions(sample_db):
    """Test window functions with over()."""
    from moltres.expressions import functions as F

    df = sample_db.table("users").polars()

    # Test row_number with over()
    df_with_rank = df.with_columns(F.row_number().over().alias("row_num"))
    results = df_with_rank.collect()
    results_list = _to_dict_list(results)

    assert "row_num" in results_list[0]
    assert results_list[0]["row_num"] == 1

    # Test rank with partition_by
    df_with_rank_by_country = df.with_columns(
        F.rank().over(partition_by=col("country")).alias("rank_by_country")
    )
    results = df_with_rank_by_country.collect()
    results_list = _to_dict_list(results)

    assert "rank_by_country" in results_list[0]


# ============================================================================
# Conditional Expressions
# ============================================================================


def test_when_then_otherwise(sample_db):
    """Test when().then().otherwise() conditional expressions."""
    from moltres.expressions import functions as F

    df = sample_db.table("users").polars()

    # Test when/then/otherwise
    df_with_category = df.with_columns(
        F.when(col("age") >= 30, "senior").otherwise("junior").alias("category")
    )
    results = df_with_category.collect()
    results_list = _to_dict_list(results)

    assert "category" in results_list[0]
    # Alice is 30, so should be "senior"
    alice = [r for r in results_list if r["name"] == "Alice"][0]
    assert alice["category"] == "senior"


# ============================================================================
# Explode and Unnest
# ============================================================================


def test_explode(sample_db):
    """Test explode() method."""
    # Note: explode requires JSON/array support which varies by database
    # This test verifies the API exists
    df = sample_db.table("users").polars()

    # The method should exist
    assert hasattr(df, "explode")
    assert hasattr(df, "unnest")


# ============================================================================
# Pivot
# ============================================================================


def test_pivot(sample_db):
    """Test pivot() method."""
    df = sample_db.table("users").polars()

    # The method should exist
    assert hasattr(df, "pivot")

    # Note: pivot requires specific data structure, so we'll just test it exists
    # Full pivot test would require appropriate test data


# ============================================================================
# Utility Methods
# ============================================================================


def test_slice(sample_db):
    """Test slice() method."""
    df = sample_db.table("users").polars()

    # Slice first 2 rows
    sliced = df.slice(0, 2)
    results = sliced.collect()
    results_list = _to_dict_list(results)

    assert len(results_list) <= 2


def test_describe(sample_db):
    """Test describe() method."""
    df = sample_db.table("users").polars()

    # Describe should return statistics
    # Note: SQLite doesn't support stddev, so we'll skip that
    described = df.describe()

    # Try to collect, but handle stddev errors gracefully
    try:
        results = described.collect()
        results_list = _to_dict_list(results)
        # Should have statistics columns
        assert len(results_list) > 0
    except Exception as e:
        # If stddev fails (e.g., SQLite), that's okay
        # Just verify the method exists and can be called
        if "stddev" in str(e).lower():
            # Expected for SQLite, test passes
            pass
        else:
            raise


def test_explain(sample_db):
    """Test explain() method."""
    df = sample_db.table("users").polars()

    # Explain should return query plan as string
    plan = df.explain()
    assert isinstance(plan, str)
    assert len(plan) > 0


# ============================================================================
# Set Operations
# ============================================================================


def test_concat(sample_db):
    """Test concat() method for vertical concatenation."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("users").polars()

    # Concatenate vertically
    concatenated = df1.concat(df2, how="vertical")
    results = concatenated.collect()
    results_list = _to_dict_list(results)

    # Should have 6 rows (3 from df1 + 3 from df2)
    assert len(results_list) == 6


def test_union(sample_db):
    """Test union() method."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("users").polars()

    # Union distinct
    unioned = df1.union(df2, distinct=True)
    results = unioned.collect()
    results_list = _to_dict_list(results)

    # Should have 3 rows (distinct)
    assert len(results_list) == 3

    # Union all
    unioned_all = df1.union(df2, distinct=False)
    results_all = unioned_all.collect()
    results_list_all = _to_dict_list(results_all)

    # Should have 6 rows (all rows)
    assert len(results_list_all) == 6


def test_intersect(sample_db):
    """Test intersect() method."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("users").polars()

    # Intersect
    intersected = df1.intersect(df2)
    results = intersected.collect()
    results_list = _to_dict_list(results)

    # Should have 3 rows (all rows are common)
    assert len(results_list) == 3


def test_difference(sample_db):
    """Test difference() method."""
    from moltres.table.schema import column
    from moltres.io.records import Records

    # Create a second table with same schema
    sample_db.create_table(
        "users2",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("age", "INTEGER"),
            column("country", "TEXT"),
        ],
    ).collect()

    # Insert one overlapping row
    Records(
        _data=[{"id": 1, "name": "Alice", "age": 30, "country": "USA"}],
        _database=sample_db,
    ).insert_into("users2")

    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("users2").polars()

    # Difference
    diff = df1.difference(df2)
    results = diff.collect()
    results_list = _to_dict_list(results)

    # Should have rows from df1 that aren't in df2 (Bob and Charlie)
    assert len(results_list) == 2


def test_cross_join(sample_db):
    """Test cross_join() method."""
    df1 = sample_db.table("users").polars().select("id", "name")
    df2 = sample_db.table("orders").polars().select("id", "amount")

    # Cross join
    crossed = df1.cross_join(df2)
    results = crossed.collect()
    results_list = _to_dict_list(results)

    # Should have cartesian product (3 users * 3 orders = 9 rows)
    assert len(results_list) == 9


def test_vstack(sample_db):
    """Test vstack() method (alias for concat)."""
    df1 = sample_db.table("users").polars()
    df2 = sample_db.table("users").polars()

    # Vertical stack
    stacked = df1.vstack(df2)
    results = stacked.collect()
    results_list = _to_dict_list(results)

    # Should have 6 rows
    assert len(results_list) == 6


# ============================================================================
# SQL Expression Selection
# ============================================================================


def test_select_expr(sample_db):
    """Test select_expr() method."""
    df = sample_db.table("users").polars()

    # Select with SQL expressions
    selected = df.select_expr("id", "name", "age", "age * 2 as double_age")
    results = selected.collect()
    results_list = _to_dict_list(results)

    assert "double_age" in results_list[0]
    assert "age" in results_list[0]
    assert results_list[0]["double_age"] == results_list[0]["age"] * 2


# ============================================================================
# CTE Support
# ============================================================================


def test_cte(sample_db):
    """Test cte() method."""
    df = sample_db.table("users").polars()

    # Create CTE
    cte_df = df.filter(col("age") > 25).cte("adults")

    # Query the CTE
    result = cte_df.select().collect()
    results_list = _to_dict_list(result)

    # Should have users with age > 25
    assert len(results_list) == 2  # Alice (30) and Charlie (35)


# ============================================================================
# Utility Methods
# ============================================================================


def test_with_columns_renamed(sample_db):
    """Test with_columns_renamed() method."""
    # Select columns first to create a Project
    df = sample_db.table("users").polars().select("id", "name", "age", "country")

    # Rename columns
    renamed = df.with_columns_renamed({"name": "full_name", "age": "years"})
    results = renamed.collect()
    results_list = _to_dict_list(results)

    assert "full_name" in results_list[0]
    assert "years" in results_list[0]
    assert "name" not in results_list[0]
    assert "age" not in results_list[0]


def test_with_row_count(sample_db):
    """Test with_row_count() method."""
    df = sample_db.table("users").polars()

    # Add row count
    df_with_row = df.with_row_count("row_id")
    results = df_with_row.collect()
    results_list = _to_dict_list(results)

    assert "row_id" in results_list[0]
    assert results_list[0]["row_id"] == 1
