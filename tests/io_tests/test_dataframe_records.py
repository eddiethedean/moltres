"""Tests for pandas and polars DataFrame support as Records.

These tests require pandas and polars to be installed.
Install with: pip install -e ".[dev]" or pip install pandas polars
"""

from __future__ import annotations

import os

import pytest

if os.environ.get("MOLTRES_SKIP_PANDAS_TESTS") == "1":
    pytest.skip(
        "Skipping pandas-dependent tests (MOLTRES_SKIP_PANDAS_TESTS=1)",
        allow_module_level=True,
    )

# Require pandas and polars - skip entire test module if not available
pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

from moltres import connect  # noqa: E402
from moltres.io.records import Records  # noqa: E402
from moltres.table.schema import column  # noqa: E402


class TestPandasDataFrameRecords:
    """Tests for pandas DataFrame support in Records."""

    def test_pandas_dataframe_to_records(self, tmp_path):
        """Test converting pandas DataFrame to Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        assert records._dataframe is not None
        assert records._data is None  # Not converted yet (lazy)

    def test_pandas_dataframe_lazy_conversion(self, tmp_path):
        """Test that pandas DataFrame is converted lazily."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame([{"id": 1, "name": "Alice"}])
        records = Records.from_dataframe(df, database=db)
        # DataFrame should still be stored, not converted
        assert records._dataframe is not None
        assert records._data is None

        # Access rows to trigger conversion
        rows = records.rows()
        assert len(rows) == 1
        assert rows[0] == {"id": 1, "name": "Alice"}
        # After conversion, DataFrame should be cleared and data cached
        assert records._dataframe is None
        assert records._data is not None

    def test_pandas_dataframe_iteration(self, tmp_path):
        """Test iterating over Records from pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        rows = list(records)
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "Alice"}
        assert rows[1] == {"id": 2, "name": "Bob"}

    def test_pandas_dataframe_length(self, tmp_path):
        """Test getting length of Records from pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame([{"id": 1}, {"id": 2}, {"id": 3}])
        records = Records.from_dataframe(df, database=db)
        assert len(records) == 3

    def test_pandas_dataframe_indexing(self, tmp_path):
        """Test indexing Records from pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        assert records[0] == {"id": 1, "name": "Alice"}
        assert records[1] == {"id": 2, "name": "Bob"}

    def test_pandas_dataframe_schema_preservation(self, tmp_path):
        """Test that schema is extracted from pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        # Create DataFrame and let pandas infer types, then check schema extraction
        df = pd.DataFrame([{"id": 1, "name": "Alice", "age": 30}])
        records = Records.from_dataframe(df, database=db)
        assert records._schema is not None
        assert len(records._schema) == 3
        schema_names = {col.name for col in records._schema}
        assert schema_names == {"id", "name", "age"}

    def test_pandas_dataframe_insert_into(self, tmp_path):
        """Test inserting pandas DataFrame into table."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        inserted = records.insert_into("users")
        assert inserted == 2

        # Verify data was inserted
        table = db.table("users")
        df_result = table.select()
        rows = df_result.collect()
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"

    def test_pandas_dataframe_create_dataframe(self, tmp_path):
        """Test creating DataFrame from pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df_pandas = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        df_moltres = db.createDataFrame(df_pandas, pk="id")
        rows = df_moltres.collect()
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_pandas_dataframe_empty(self, tmp_path):
        """Test handling empty pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame()
        records = Records.from_dataframe(df, database=db)
        assert len(records) == 0
        assert list(records) == []

    def test_pandas_dataframe_with_nulls(self, tmp_path):
        """Test pandas DataFrame with null values."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pd.DataFrame([{"id": 1, "name": "Alice", "age": None}])
        records = Records.from_dataframe(df, database=db)
        rows = records.rows()
        assert rows[0]["age"] is None
        # Schema should indicate nullable
        assert records._schema is not None
        age_col = next(col for col in records._schema if col.name == "age")
        assert age_col.nullable is True


class TestPolarsDataFrameRecords:
    """Tests for polars DataFrame support in Records."""

    def test_polars_dataframe_to_records(self, tmp_path):
        """Test converting polars DataFrame to Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        assert records._dataframe is not None
        assert records._data is None  # Not converted yet (lazy)

    def test_polars_dataframe_lazy_conversion(self, tmp_path):
        """Test that polars DataFrame is converted lazily."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        records = Records.from_dataframe(df, database=db)
        # DataFrame should still be stored, not converted
        assert records._dataframe is not None
        assert records._data is None

        # Access rows to trigger conversion
        rows = records.rows()
        assert len(rows) == 1
        assert rows[0] == {"id": 1, "name": "Alice"}
        # After conversion, DataFrame should be cleared and data cached
        assert records._dataframe is None
        assert records._data is not None

    def test_polars_dataframe_iteration(self, tmp_path):
        """Test iterating over Records from polars DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        rows = list(records)
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "Alice"}
        assert rows[1] == {"id": 2, "name": "Bob"}

    def test_polars_dataframe_length(self, tmp_path):
        """Test getting length of Records from polars DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1}, {"id": 2}, {"id": 3}])
        records = Records.from_dataframe(df, database=db)
        assert len(records) == 3

    def test_polars_dataframe_schema_preservation(self, tmp_path):
        """Test that schema is extracted from polars DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame(
            [{"id": 1, "name": "Alice", "age": 30}],
            schema={"id": pl.Int64, "name": pl.Utf8, "age": pl.Int64},
        )
        records = Records.from_dataframe(df, database=db)
        assert records._schema is not None
        assert len(records._schema) == 3
        schema_names = {col.name for col in records._schema}
        assert schema_names == {"id", "name", "age"}

    def test_polars_dataframe_insert_into(self, tmp_path):
        """Test inserting polars DataFrame into table."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        records = Records.from_dataframe(df, database=db)
        inserted = records.insert_into("users")
        assert inserted == 2

        # Verify data was inserted
        table = db.table("users")
        df_result = table.select()
        rows = df_result.collect()
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"
        assert rows[1]["name"] == "Bob"

    def test_polars_dataframe_create_dataframe(self, tmp_path):
        """Test creating DataFrame from polars DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df_polars = pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        df_moltres = db.createDataFrame(df_polars, pk="id")
        rows = df_moltres.collect()
        assert len(rows) == 2
        assert rows[0]["name"] == "Alice"

    def test_polars_dataframe_empty(self, tmp_path):
        """Test handling empty polars DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame()
        records = Records.from_dataframe(df, database=db)
        assert len(records) == 0
        assert list(records) == []


class TestPolarsLazyFrameRecords:
    """Tests for polars LazyFrame support in Records."""

    def test_polars_lazyframe_to_records(self, tmp_path):
        """Test converting polars LazyFrame to Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        lf = df.lazy()
        records = Records.from_dataframe(lf, database=db)
        assert records._dataframe is not None
        assert records._data is None  # Not converted yet (lazy)

    def test_polars_lazyframe_materialization(self, tmp_path):
        """Test that polars LazyFrame is materialized on access."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        lf = df.lazy()
        records = Records.from_dataframe(lf, database=db)
        # LazyFrame should still be stored
        assert records._dataframe is not None

        # Access rows to trigger materialization and conversion
        rows = records.rows()
        assert len(rows) == 1
        assert rows[0] == {"id": 1, "name": "Alice"}

    def test_polars_lazyframe_length(self, tmp_path):
        """Test getting length of Records from polars LazyFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1}, {"id": 2}, {"id": 3}])
        lf = df.lazy()
        records = Records.from_dataframe(lf, database=db)
        # Length requires materialization
        assert len(records) == 3

    def test_polars_lazyframe_schema_preservation(self, tmp_path):
        """Test that schema is extracted from polars LazyFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame(
            [{"id": 1, "name": "Alice"}],
            schema={"id": pl.Int64, "name": pl.Utf8},
        )
        lf = df.lazy()
        records = Records.from_dataframe(lf, database=db)
        assert records._schema is not None
        assert len(records._schema) == 2
        schema_names = {col.name for col in records._schema}
        assert schema_names == {"id", "name"}

    def test_polars_lazyframe_insert_into(self, tmp_path):
        """Test inserting polars LazyFrame into table."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        lf = df.lazy()
        records = Records.from_dataframe(lf, database=db)
        inserted = records.insert_into("users")
        assert inserted == 1

    def test_polars_lazyframe_create_dataframe(self, tmp_path):
        """Test creating DataFrame from polars LazyFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        lf = df.lazy()
        df_moltres = db.createDataFrame(lf, pk="id")
        rows = df_moltres.collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"


class TestDataFrameMutations:
    """Tests for DataFrame support in mutation operations."""

    def test_insert_pandas_dataframe(self, tmp_path):
        """Test inserting pandas DataFrame directly."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pd.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        handle = db.table("users")
        from moltres.table.mutations import insert_rows

        inserted = insert_rows(handle, df)
        assert inserted == 2

    def test_insert_polars_dataframe(self, tmp_path):
        """Test inserting polars DataFrame directly."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pl.DataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
        handle = db.table("users")
        from moltres.table.mutations import insert_rows

        inserted = insert_rows(handle, df)
        assert inserted == 2

    def test_insert_polars_lazyframe(self, tmp_path):
        """Test inserting polars LazyFrame directly."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        lf = df.lazy()
        handle = db.table("users")
        from moltres.table.mutations import insert_rows

        inserted = insert_rows(handle, lf)
        assert inserted == 1

    def test_merge_pandas_dataframe(self, tmp_path):
        """Test merging pandas DataFrame directly."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()

        # Insert initial row
        handle = db.table("users")
        from moltres.table.mutations import insert_rows, merge_rows

        insert_rows(handle, [{"id": 1, "name": "Alice"}])

        # Merge with pandas DataFrame
        df = pd.DataFrame([{"id": 1, "name": "Bob"}])
        merged = merge_rows(handle, df, on=["id"], when_matched={"name": "Updated"})
        assert merged >= 0

    def test_merge_polars_dataframe(self, tmp_path):
        """Test merging polars DataFrame directly."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table(
            "users",
            [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
        ).collect()

        # Insert initial row
        handle = db.table("users")
        from moltres.table.mutations import insert_rows, merge_rows

        insert_rows(handle, [{"id": 1, "name": "Alice"}])

        # Merge with polars DataFrame
        df = pl.DataFrame([{"id": 1, "name": "Bob"}])
        merged = merge_rows(handle, df, on=["id"], when_matched={"name": "Updated"})
        assert merged >= 0


class TestDataFrameActions:
    """Tests for DataFrame support in action classes."""

    def test_insert_mutation_with_pandas(self, tmp_path):
        """Test InsertMutation with pandas DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pd.DataFrame([{"id": 1, "name": "Alice"}])
        handle = db.table("users")
        from moltres.table.actions import InsertMutation

        mutation = InsertMutation(handle=handle, rows=df)
        inserted = mutation.collect()
        assert inserted == 1

    def test_insert_mutation_with_polars(self, tmp_path):
        """Test InsertMutation with polars DataFrame."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()

        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        handle = db.table("users")
        from moltres.table.actions import InsertMutation

        mutation = InsertMutation(handle=handle, rows=df)
        inserted = mutation.collect()
        assert inserted == 1


class TestDataFrameNormalize:
    """Tests for DataFrame support in normalize_data_to_rows."""

    def test_normalize_pandas_dataframe(self, tmp_path):
        """Test normalize_data_to_rows with pandas DataFrame."""
        from moltres.dataframe.core.create_dataframe import normalize_data_to_rows

        df = pd.DataFrame([{"id": 1, "name": "Alice"}])
        rows = normalize_data_to_rows(df)
        assert len(rows) == 1
        assert rows[0] == {"id": 1, "name": "Alice"}

    def test_normalize_polars_dataframe(self, tmp_path):
        """Test normalize_data_to_rows with polars DataFrame."""
        from moltres.dataframe.core.create_dataframe import normalize_data_to_rows

        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        rows = normalize_data_to_rows(df)
        assert len(rows) == 1
        assert rows[0] == {"id": 1, "name": "Alice"}

    def test_normalize_polars_lazyframe(self, tmp_path):
        """Test normalize_data_to_rows with polars LazyFrame."""
        from moltres.dataframe.core.create_dataframe import normalize_data_to_rows

        df = pl.DataFrame([{"id": 1, "name": "Alice"}])
        lf = df.lazy()
        rows = normalize_data_to_rows(lf)
        assert len(rows) == 1
        assert rows[0] == {"id": 1, "name": "Alice"}
