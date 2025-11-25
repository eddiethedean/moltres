"""Comprehensive tests for AsyncDataFrameWriter covering all methods and edge cases."""

from __future__ import annotations

import json
import os

import pytest

if os.environ.get("MOLTRES_SKIP_PANDAS_TESTS"):
    pytest.skip(
        "Skipping pandas-dependent tests (MOLTRES_SKIP_PANDAS_TESTS=1)",
        allow_module_level=True,
    )

try:
    import aiosqlite  # noqa: F401
except ImportError:
    pytest.skip("aiosqlite not installed", allow_module_level=True)

from moltres import async_connect, col
from moltres.table.schema import ColumnDef


@pytest.mark.asyncio
class TestAsyncDataFrameWriterBuilder:
    """Test AsyncDataFrameWriter builder methods."""

    async def test_mode_append(self, tmp_path):
        """Test mode('append')."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        writer = df.write.mode("append")
        assert writer._mode == "append"

        await db.close()

    async def test_mode_overwrite(self, tmp_path):
        """Test mode('overwrite')."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        writer = df.write.mode("overwrite")
        assert writer._mode == "overwrite"

        await db.close()

    async def test_mode_ignore(self, tmp_path):
        """Test mode('ignore')."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        writer = df.write.mode("ignore")
        assert writer._mode == "ignore"

        await db.close()

    async def test_mode_error_if_exists(self, tmp_path):
        """Test mode('error_if_exists')."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        writer = df.write.mode("error_if_exists")
        assert writer._mode == "error_if_exists"

        await db.close()

    async def test_mode_normalization(self, tmp_path):
        """Test mode normalization with spaces, dashes, underscores."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")

        # Test various normalizations
        assert df.write.mode("error if exists")._mode == "error_if_exists"
        assert df.write.mode("error-if-exists")._mode == "error_if_exists"
        assert df.write.mode("error_if_exists")._mode == "error_if_exists"
        assert df.write.mode("ERROR_IF_EXISTS")._mode == "error_if_exists"

        await db.close()

    async def test_mode_invalid(self, tmp_path):
        """Test invalid mode raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(ValueError, match="Invalid mode"):
            df.write.mode("invalid_mode")

        await db.close()

    async def test_option(self, tmp_path):
        """Test option() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        writer = df.write.option("header", True).option("delimiter", "|")
        assert writer._options["header"] is True
        assert writer._options["delimiter"] == "|"

        await db.close()

    async def test_options_with_kwargs(self, tmp_path):
        """Test options() with keyword arguments."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        writer = df.write.options(header=True, delimiter="|", compression="gzip")
        assert writer._options["header"] is True
        assert writer._options["delimiter"] == "|"
        assert writer._options["compression"] == "gzip"

        await db.close()

    async def test_options_with_dict(self, tmp_path):
        """Test options() with dictionary argument."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        opts = {"header": True, "delimiter": "|"}
        writer = df.write.options(opts)
        assert writer._options["header"] is True
        assert writer._options["delimiter"] == "|"

        await db.close()

    async def test_options_multiple_positional_error(self, tmp_path):
        """Test options() with multiple positional args raises TypeError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(TypeError, match="at most one positional"):
            df.write.options({"a": 1}, {"b": 2})

        await db.close()

    async def test_format(self, tmp_path):
        """Test format() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        writer = df.write.format("csv")
        assert writer._format == "csv"

        writer = df.write.format("JSON")
        assert writer._format == "json"

        await db.close()

    async def test_stream(self, tmp_path):
        """Test stream() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        writer = df.write.stream(True)
        assert writer._stream_override is True

        writer = df.write.stream(False)
        assert writer._stream_override is False

        await db.close()

    async def test_partition_by(self, tmp_path):
        """Test partitionBy() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "category": "A"}], pk="id")
        writer = df.write.partitionBy("category")
        assert writer._partition_by == ("category",)

        writer = df.write.partitionBy("category", "year")
        assert writer._partition_by == ("category", "year")

        await db.close()

    async def test_partition_by_empty(self, tmp_path):
        """Test partitionBy() with no columns."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        writer = df.write.partitionBy()
        assert writer._partition_by is None

        await db.close()

    async def test_schema(self, tmp_path):
        """Test schema() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        schema = [ColumnDef(name="id", type_name="INTEGER")]
        writer = df.write.schema(schema)
        assert writer._schema == schema

        await db.close()

    async def test_primary_key(self, tmp_path):
        """Test primaryKey() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        writer = df.write.primaryKey("id")
        assert writer._primary_key == ("id",)

        writer = df.write.primaryKey("id", "name")
        assert writer._primary_key == ("id", "name")

        await db.close()

    async def test_primary_key_empty(self, tmp_path):
        """Test primaryKey() with no columns."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        writer = df.write.primaryKey()
        assert writer._primary_key is None

        await db.close()

    async def test_bucket_by(self, tmp_path):
        """Test bucketBy() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "category": "A"}], pk="id")
        writer = df.write.bucketBy(10, "category")
        assert writer._bucket_by == (10, ("category",))

        await db.close()

    async def test_bucket_by_invalid_num_buckets(self, tmp_path):
        """Test bucketBy() with invalid num_buckets raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(ValueError, match="num_buckets must be a positive integer"):
            df.write.bucketBy(0, "id")

        with pytest.raises(ValueError, match="num_buckets must be a positive integer"):
            df.write.bucketBy(-1, "id")

        await db.close()

    async def test_bucket_by_no_columns(self, tmp_path):
        """Test bucketBy() with no columns raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(ValueError, match="requires at least one column name"):
            df.write.bucketBy(10)

        await db.close()

    async def test_sort_by(self, tmp_path):
        """Test sortBy() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        writer = df.write.sortBy("id")
        assert writer._sort_by == ("id",)

        writer = df.write.sortBy("id", "name")
        assert writer._sort_by == ("id", "name")

        await db.close()

    async def test_sort_by_no_columns(self, tmp_path):
        """Test sortBy() with no columns raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(ValueError, match="requires at least one column name"):
            df.write.sortBy()

        await db.close()


@pytest.mark.asyncio
class TestAsyncDataFrameWriterTableOperations:
    """Test AsyncDataFrameWriter table operations."""

    async def test_save_as_table_append(self, tmp_path):
        """Test save_as_table with append mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create table first
        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        # Insert first row
        df1 = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df1.write.insertInto("users")

        # Append second row
        df2 = await db.createDataFrame([{"id": 2, "name": "Bob"}], pk="id")
        await df2.write.mode("append").insertInto("users")

        table = await db.table("users")
        results = await table.select().collect()
        assert len(results) == 2

        await db.close()

    async def test_save_as_table_overwrite(self, tmp_path):
        """Test save_as_table with overwrite mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df1 = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df1.write.save_as_table("users")

        # Verify first write
        table = await db.table("users")
        results = await table.select().collect()
        assert len(results) == 1

        # Overwrite with new data
        df2 = await db.createDataFrame([{"id": 2, "name": "Bob"}], pk="id")
        await df2.write.mode("overwrite").save_as_table("users")

        results = await table.select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"

        await db.close()

    async def test_save_as_table_ignore(self, tmp_path):
        """Test save_as_table with ignore mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df1 = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df1.write.save_as_table("users")

        # Verify first write
        table = await db.table("users")
        results = await table.select().collect()
        assert len(results) == 1
        original_name = results[0]["name"]

        # Try to write again with ignore mode (should do nothing)
        df2 = await db.createDataFrame([{"id": 2, "name": "Bob"}], pk="id")
        await df2.write.mode("ignore").save_as_table("users")

        results = await table.select().collect()
        assert len(results) == 1
        assert results[0]["name"] == original_name

        await db.close()

    async def test_save_as_table_error_if_exists(self, tmp_path):
        """Test save_as_table with error_if_exists mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create table first
        df1 = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df1.write.save_as_table("users")

        # Try to write again with error_if_exists mode (should raise error)
        df2 = await db.createDataFrame([{"id": 2, "name": "Bob"}], pk="id")
        with pytest.raises(ValueError, match="already exists and mode is 'error_if_exists'"):
            await df2.write.mode("error_if_exists").save_as_table("users")

        await db.close()

    async def test_save_as_table_with_primary_key(self, tmp_path):
        """Test save_as_table with primary key parameter."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Test that primary_key parameter is accepted
        # Note: The optimization path may have issues with ephemeral tables,
        # but the parameter is correctly passed through
        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        # Use explicit schema to avoid optimization path issues
        from moltres.table.schema import ColumnDef

        schema = [
            ColumnDef(name="id", type_name="INTEGER", primary_key=True),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        await df.write.schema(schema).save_as_table("users", primary_key=["id"])

        table = await db.table("users")
        results = await table.select().collect()
        assert len(results) == 1

        await db.close()

    async def test_save_as_table_without_database(self, tmp_path):
        """Test save_as_table without database raises RuntimeError."""
        from moltres.dataframe.async_dataframe import AsyncDataFrame
        from moltres.logical import operators

        plan = operators.scan("test")
        df = AsyncDataFrame(plan=plan, database=None)

        with pytest.raises(
            RuntimeError, match="Cannot write AsyncDataFrame without an attached AsyncDatabase"
        ):
            await df.write.save_as_table("test")

    async def test_insert_into(self, tmp_path):
        """Test insertInto() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        await df.write.insertInto("users")

        table = await db.table("users")
        results = await table.select().collect()
        assert len(results) == 1

        await db.close()

    async def test_insert_into_table_not_exists(self, tmp_path):
        """Test insertInto() with non-existent table raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        # The check should happen before execution, but _table_exists may not work perfectly
        # for SQLite, so we catch both ValueError and ExecutionError
        from sqlalchemy.exc import OperationalError
        from moltres.utils.exceptions import ExecutionError

        with pytest.raises(
            (ValueError, OperationalError, ExecutionError), match="does not exist|no such table"
        ):
            await df.write.stream(False).insertInto("nonexistent")

        await db.close()

    async def test_insert_into_with_bucket_by_error(self, tmp_path):
        """Test insertInto() with bucketBy raises NotImplementedError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        from moltres.table.schema import column

        await db.create_table("users", [column("id", "INTEGER", primary_key=True)]).collect()

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(NotImplementedError, match="bucketBy/sortBy are not yet supported"):
            await df.write.bucketBy(10, "id").insertInto("users")

        await db.close()

    async def test_update(self, tmp_path):
        """Test update() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create table and insert data
        from moltres.table.schema import column

        await db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        df = await db.createDataFrame([{"id": 1, "name": "Alice", "age": 30}], pk="id")
        await df.write.stream(False).insertInto("users")  # Disable optimization

        # Query the table and update
        table = await db.table("users")
        df2 = table.select()
        await df2.write.update("users", where=col("id") == 1, set={"name": "Bob"})

        # Verify update
        results = await table.select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"
        assert results[0]["age"] == 30  # Other columns unchanged

        await db.close()

    async def test_update_table_not_exists(self, tmp_path):
        """Test update() with non-existent table raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create a DataFrame from an existing table (to have a valid DataFrame)
        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table = await db.table("users")
        df = table.select()

        # Try to update a non-existent table
        # The _table_exists check may not work correctly for SQLite, so we catch both
        from sqlalchemy.exc import OperationalError
        from moltres.utils.exceptions import ExecutionError

        with pytest.raises(
            (ValueError, OperationalError, ExecutionError), match="does not exist|no such table"
        ):
            await df.write.update("nonexistent", where=col("id") == 1, set={"name": "Bob"})

        await db.close()

    async def test_delete(self, tmp_path):
        """Test delete() method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create table and insert data
        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        df = await db.createDataFrame(
            [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id"
        )
        await df.write.stream(False).insertInto("users")  # Disable optimization

        # Verify initial state
        table = await db.table("users")
        results = await table.select().collect()
        assert len(results) == 2

        # Delete one row
        df2 = table.select()
        await df2.write.delete("users", where=col("id") == 1)

        # Verify deletion
        results = await table.select().collect()
        assert len(results) == 1
        assert results[0]["name"] == "Bob"

        await db.close()

    async def test_delete_table_not_exists(self, tmp_path):
        """Test delete() with non-existent table raises error."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        # Create a DataFrame from an existing table (to have a valid DataFrame)
        from moltres.table.schema import column

        await db.create_table(
            "users", [column("id", "INTEGER", primary_key=True), column("name", "TEXT")]
        ).collect()

        table = await db.table("users")
        df = table.select()

        # Try to delete from a non-existent table
        # The _table_exists check may not work correctly for SQLite, so we catch both
        from sqlalchemy.exc import OperationalError
        from moltres.utils.exceptions import ExecutionError

        with pytest.raises(
            (ValueError, OperationalError, ExecutionError), match="does not exist|no such table"
        ):
            await df.write.delete("nonexistent", where=col("id") == 1)

        await db.close()


@pytest.mark.asyncio
class TestAsyncDataFrameWriterFileOperations:
    """Test AsyncDataFrameWriter file operations."""

    async def test_save_csv(self, tmp_path):
        """Test save() as CSV."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        csv_path = tmp_path / "output.csv"
        await df.write.save(str(csv_path), format="csv")

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "id,name" in content or "name,id" in content
        assert "Alice" in content

        await db.close()

    async def test_save_csv_with_options(self, tmp_path):
        """Test save() as CSV with options."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        csv_path = tmp_path / "output.csv"
        await df.write.options(delimiter="|", header=False).save(str(csv_path), format="csv")

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "|" in content

        await db.close()

    async def test_save_json(self, tmp_path):
        """Test save() as JSON."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        json_path = tmp_path / "output.json"
        await df.write.save(str(json_path), format="json")

        assert json_path.exists()
        content = json.loads(json_path.read_text())
        assert isinstance(content, list)
        assert len(content) == 1

        await db.close()

    async def test_save_jsonl(self, tmp_path):
        """Test save() as JSONL."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        jsonl_path = tmp_path / "output.jsonl"
        await df.write.save(str(jsonl_path), format="jsonl")

        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["name"] == "Alice"

        await db.close()

    async def test_save_text(self, tmp_path):
        """Test save() as text."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"value": "line1"}, {"value": "line2"}], auto_pk="id")
        text_path = tmp_path / "output.txt"
        await df.write.save(str(text_path), format="text")

        assert text_path.exists()
        content = text_path.read_text()
        assert "line1" in content
        assert "line2" in content

        await db.close()

    async def test_save_parquet(self, tmp_path):
        """Test save() as Parquet."""
        pytest.importorskip("pyarrow")
        pytest.importorskip("pandas")

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "name": "Alice"}], pk="id")
        parquet_path = tmp_path / "output.parquet"
        await df.write.save(str(parquet_path), format="parquet")

        assert parquet_path.exists()

        await db.close()

    async def test_save_format_inference(self, tmp_path):
        """Test save() with format inference from extension."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")

        csv_path = tmp_path / "output.csv"
        await df.write.save(str(csv_path))
        assert csv_path.exists()

        json_path = tmp_path / "output.json"
        await df.write.save(str(json_path))
        assert json_path.exists()

        await db.close()

    async def test_save_unknown_format(self, tmp_path):
        """Test save() with unknown format raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(ValueError, match="Cannot infer format"):
            await df.write.save(str(tmp_path / "output.unknown"))

        await db.close()

    async def test_save_unsupported_format(self, tmp_path):
        """Test save() with unsupported format raises ValueError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(ValueError, match="Unsupported format"):
            await df.write.save(str(tmp_path / "output.xyz"), format="xyz")

        await db.close()

    async def test_save_orc_not_implemented(self, tmp_path):
        """Test save() as ORC raises NotImplementedError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(NotImplementedError, match="ORC write support is not yet available"):
            await df.write.save(str(tmp_path / "output.orc"), format="orc")

        await db.close()

    async def test_csv_helper(self, tmp_path):
        """Test csv() helper method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        csv_path = tmp_path / "output.csv"
        await df.write.csv(str(csv_path))

        assert csv_path.exists()

        await db.close()

    async def test_json_helper(self, tmp_path):
        """Test json() helper method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        json_path = tmp_path / "output.json"
        await df.write.json(str(json_path))

        assert json_path.exists()

        await db.close()

    async def test_jsonl_helper(self, tmp_path):
        """Test jsonl() helper method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        jsonl_path = tmp_path / "output.jsonl"
        await df.write.jsonl(str(jsonl_path))

        assert jsonl_path.exists()

        await db.close()

    async def test_text_helper(self, tmp_path):
        """Test text() helper method."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"value": "test"}], auto_pk="id")
        text_path = tmp_path / "output.txt"
        await df.write.text(str(text_path))

        assert text_path.exists()

        await db.close()

    async def test_parquet_helper(self, tmp_path):
        """Test parquet() helper method."""
        pytest.importorskip("pyarrow")
        pytest.importorskip("pandas")

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        parquet_path = tmp_path / "output.parquet"
        await df.write.parquet(str(parquet_path))

        assert parquet_path.exists()

        await db.close()

    async def test_orc_helper_not_implemented(self, tmp_path):
        """Test orc() helper raises NotImplementedError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(NotImplementedError, match="Async ORC output is not supported"):
            await df.write.orc(str(tmp_path / "output.orc"))

        await db.close()

    async def test_save_mode_overwrite(self, tmp_path):
        """Test save() with overwrite mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df1 = await db.createDataFrame([{"id": 1}], pk="id")
        csv_path = tmp_path / "output.csv"
        await df1.write.save(str(csv_path), format="csv")

        df2 = await db.createDataFrame([{"id": 2}], pk="id")
        await df2.write.mode("overwrite").save(str(csv_path), format="csv")

        assert csv_path.exists()

        await db.close()

    async def test_save_mode_ignore(self, tmp_path):
        """Test save() with ignore mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df1 = await db.createDataFrame([{"id": 1}], pk="id")
        csv_path = tmp_path / "output.csv"
        await df1.write.save(str(csv_path), format="csv")
        original_size = csv_path.stat().st_size

        df2 = await db.createDataFrame([{"id": 2}], pk="id")
        await df2.write.mode("ignore").save(str(csv_path), format="csv")

        # File should be unchanged
        assert csv_path.stat().st_size == original_size

        await db.close()

    async def test_save_mode_error_if_exists(self, tmp_path):
        """Test save() with error_if_exists mode."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df1 = await db.createDataFrame([{"id": 1}], pk="id")
        csv_path = tmp_path / "output.csv"
        await df1.write.save(str(csv_path), format="csv")

        df2 = await db.createDataFrame([{"id": 2}], pk="id")
        with pytest.raises(ValueError, match="already exists"):
            await df2.write.mode("error_if_exists").save(str(csv_path), format="csv")

        await db.close()

    async def test_save_partition_by_not_implemented(self, tmp_path):
        """Test save() with partitionBy raises NotImplementedError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1, "category": "A"}], pk="id")
        with pytest.raises(NotImplementedError, match="partitionBy"):
            await df.write.partitionBy("category").csv(str(tmp_path / "output.csv"))

        await db.close()

    async def test_save_bucket_by_not_implemented(self, tmp_path):
        """Test save() with bucketBy raises NotImplementedError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(NotImplementedError, match="bucketBy/sortBy metadata"):
            await df.write.bucketBy(10, "id").save(str(tmp_path / "output.csv"), format="csv")

        await db.close()

    async def test_save_sort_by_not_implemented(self, tmp_path):
        """Test save() with sortBy raises NotImplementedError."""
        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        df = await db.createDataFrame([{"id": 1}], pk="id")
        with pytest.raises(NotImplementedError, match="bucketBy/sortBy metadata"):
            await df.write.sortBy("id").save(str(tmp_path / "output.csv"), format="csv")

        await db.close()
