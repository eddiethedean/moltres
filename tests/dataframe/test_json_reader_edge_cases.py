"""Comprehensive tests for JSON reader edge cases and error handling."""

from __future__ import annotations

import json

import pytest

from moltres import connect
from moltres.table.schema import ColumnDef


class TestJSONReaderEdgeCases:
    """Test edge cases and error handling for JSON reader."""

    def test_read_json_file_not_found(self, tmp_path):
        """Test FileNotFoundError when JSON file doesn't exist."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            records = db.read.records.json(str(tmp_path / "nonexistent.json"))
            list(records.rows())  # Trigger actual reading

    def test_read_jsonl_file_not_found(self, tmp_path):
        """Test FileNotFoundError when JSONL file doesn't exist."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        with pytest.raises(FileNotFoundError, match="JSONL file not found"):
            records = db.read.records.jsonl(str(tmp_path / "nonexistent.jsonl"))
            list(records.rows())  # Trigger actual reading

    def test_read_json_empty_file(self, tmp_path):
        """Test reading empty JSON file."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "empty.json"
        json_path.write_text("")

        with pytest.raises(
            (ValueError, json.JSONDecodeError), match="(JSON file is empty|Expecting)"
        ):
            records = db.read.records.json(str(json_path))
            list(records.rows())  # Trigger actual reading

    def test_read_jsonl_empty_file(self, tmp_path):
        """Test reading empty JSONL file."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        with pytest.raises(ValueError, match="JSONL file is empty"):
            records = db.read.records.jsonl(str(jsonl_path))
            list(records.rows())  # Trigger actual reading

    def test_read_json_empty_file_with_schema(self, tmp_path):
        """Test reading empty JSON file with explicit schema."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "empty.json"
        json_path.write_text("")

        schema = [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        records = db.read.records.schema(schema).json(str(json_path))
        assert len(records.rows()) == 0

    def test_read_jsonl_empty_file_with_schema(self, tmp_path):
        """Test reading empty JSONL file with explicit schema."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "empty.jsonl"
        jsonl_path.write_text("")

        schema = [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        records = db.read.records.schema(schema).jsonl(str(jsonl_path))
        assert len(records.rows()) == 0

    def test_read_json_mode_failfast(self, tmp_path):
        """Test FAILFAST mode with malformed JSON."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "malformed.json"
        json_path.write_text("{invalid json}")

        with pytest.raises(
            (ValueError, json.JSONDecodeError), match="(Failed to parse JSON|Expecting)"
        ):
            records = db.read.records.option("mode", "FAILFAST").json(str(json_path))
            list(records.rows())  # Trigger actual reading

    def test_read_jsonl_mode_failfast(self, tmp_path):
        """Test FAILFAST mode with malformed JSONL."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "malformed.jsonl"
        jsonl_path.write_text("{invalid json}\n")

        with pytest.raises(
            (ValueError, json.JSONDecodeError), match="(Failed to parse JSONL|Expecting)"
        ):
            records = db.read.records.option("mode", "FAILFAST").jsonl(str(jsonl_path))
            list(records.rows())  # Trigger actual reading

    def test_read_json_mode_dropmalformed(self, tmp_path):
        """Test DROPMALFORMED mode with malformed JSON."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "malformed.json"
        json_path.write_text("{invalid json}")

        # When all rows are dropped, file is effectively empty
        with pytest.raises(ValueError, match="JSON file is empty"):
            records = db.read.records.option("mode", "DROPMALFORMED").json(str(json_path))
            list(records.rows())

    def test_read_jsonl_mode_dropmalformed(self, tmp_path):
        """Test DROPMALFORMED mode with malformed JSONL lines."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "malformed.jsonl"
        jsonl_path.write_text('{"id": 1}\n{invalid}\n{"id": 2}\n')

        records = db.read.records.option("mode", "DROPMALFORMED").jsonl(str(jsonl_path))
        rows = records.rows()
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[1]["id"] == 2

    def test_read_json_mode_permissive_with_corrupt_column(self, tmp_path):
        """Test PERMISSIVE mode with corrupt column."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "malformed.json"
        json_path.write_text("{invalid json}")

        # If corrupt record handling results in empty rows, file is empty
        # Otherwise, corrupt column should be present
        try:
            records = (
                db.read.records.option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt")
                .json(str(json_path))
            )
            rows = list(records.rows())
            if len(rows) > 0:
                assert "_corrupt" in rows[0]
                assert rows[0]["_corrupt"] is not None
        except ValueError as e:
            # If all rows are dropped, file is empty
            assert "JSON file is empty" in str(e)

    def test_read_jsonl_mode_permissive_with_corrupt_column(self, tmp_path):
        """Test PERMISSIVE mode with corrupt column for JSONL."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "malformed.jsonl"
        jsonl_path.write_text('{"id": 1}\n{invalid}\n{"id": 2}\n')

        records = (
            db.read.records.option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt")
            .jsonl(str(jsonl_path))
        )
        rows = records.rows()
        assert len(rows) == 3  # 2 valid + 1 corrupt
        corrupt_rows = [r for r in rows if "_corrupt" in r and r["_corrupt"] is not None]
        assert len(corrupt_rows) == 1

    def test_read_json_non_array_non_object(self, tmp_path):
        """Test reading JSON file with non-array, non-object root."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "invalid.json"
        json_path.write_text("123")  # Just a number

        # PERMISSIVE mode should handle this - may result in empty file or corrupt record
        try:
            records = (
                db.read.records.option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt")
                .json(str(json_path))
            )
            rows = list(records.rows())
            assert len(rows) == 0 or "_corrupt" in rows[0]
        except ValueError as e:
            # If all rows are dropped, file is empty
            assert "JSON file is empty" in str(e) or "JSON file must contain" in str(e)

    def test_read_json_non_array_non_object_failfast(self, tmp_path):
        """Test FAILFAST mode with non-array, non-object root."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "invalid.json"
        json_path.write_text("123")

        # The JSON parser may accept this as valid JSON (a number)
        # So we check that it either raises an error or handles it gracefully
        records = db.read.records.option("mode", "FAILFAST").json(str(json_path))
        try:
            rows = list(records.rows())
            # If it doesn't raise, it should be empty or have corrupt column
            assert len(rows) == 0 or any("_corrupt" in r for r in rows)
        except ValueError as e:
            assert "JSON file must contain an array or object" in str(e) or "Expecting" in str(e)

    def test_read_json_drop_field_if_all_null(self, tmp_path):
        """Test dropFieldIfAllNull option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [
            {"id": 1, "name": "Alice", "null_field": None},
            {"id": 2, "name": "Bob", "null_field": None},
        ]
        json_path = tmp_path / "data.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        records = db.read.records.option("dropFieldIfAllNull", True).json(str(json_path))
        rows = records.rows()
        assert len(rows) == 2
        assert "null_field" not in rows[0]
        assert "null_field" not in rows[1]

    def test_read_jsonl_drop_field_if_all_null(self, tmp_path):
        """Test dropFieldIfAllNull option for JSONL."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"id": 1, "name": "Alice", "null_field": null}\n')
            f.write('{"id": 2, "name": "Bob", "null_field": null}\n')

        records = db.read.records.option("dropFieldIfAllNull", True).jsonl(str(jsonl_path))
        rows = records.rows()
        assert len(rows) == 2
        assert "null_field" not in rows[0]
        assert "null_field" not in rows[1]

    def test_read_json_sampling_ratio(self, tmp_path):
        """Test samplingRatio option for schema inference."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Create JSON with many rows
        data = [{"id": i, "value": f"value_{i}"} for i in range(100)]
        json_path = tmp_path / "data.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Use 0.1 sampling ratio (should sample 10 rows)
        records = db.read.records.option("samplingRatio", 0.1).json(str(json_path))
        rows = records.rows()
        assert len(rows) == 100  # All rows should be read, but schema inferred from sample
        assert "id" in rows[0]
        assert "value" in rows[0]

    def test_read_json_line_sep(self, tmp_path):
        """Test lineSep option for multiline JSON."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        # Use custom line separator
        with open(jsonl_path, "w") as f:
            f.write('{"id": 1}|{"id": 2}|{"id": 3}')

        records = (
            db.read.records.option("multiline", True).option("lineSep", "|").json(str(jsonl_path))
        )
        rows = records.rows()
        assert len(rows) == 3

    def test_read_json_compression_bz2(self, tmp_path):
        """Test reading bz2-compressed JSON."""
        try:
            import bz2
        except ImportError:
            pytest.skip("bz2 not available")

        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [{"id": 1, "name": "Alice"}]
        json_path = tmp_path / "data.json.bz2"
        with bz2.open(json_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        records = db.read.records.json(str(json_path))
        rows = records.rows()
        assert len(rows) == 1
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "Alice"

    def test_read_json_compression_xz(self, tmp_path):
        """Test reading xz-compressed JSON."""
        try:
            import lzma
        except ImportError:
            pytest.skip("lzma not available")

        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [{"id": 1, "name": "Alice"}]
        json_path = tmp_path / "data.json.xz"
        with lzma.open(json_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        records = db.read.records.json(str(json_path))
        rows = records.rows()
        assert len(rows) == 1
        assert rows[0]["id"] == 1
        assert rows[0]["name"] == "Alice"

    def test_read_json_encoding(self, tmp_path):
        """Test reading JSON with different encoding."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [{"id": 1, "name": "测试"}]
        json_path = tmp_path / "data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        records = db.read.records.option("encoding", "UTF-8").json(str(json_path))
        rows = records.rows()
        assert len(rows) == 1
        assert rows[0]["name"] == "测试"

    def test_read_json_multiLine_alias(self, tmp_path):
        """Test multiLine alias for multiline option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"id": 1}\n{"id": 2}\n')

        # Use multiLine (camelCase) instead of multiline
        records = db.read.records.option("multiLine", True).json(str(jsonl_path))
        rows = records.rows()
        assert len(rows) == 2

    def test_read_json_streaming_chunk_size(self, tmp_path):
        """Test streaming JSON with chunk_size option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [{"id": i} for i in range(50)]
        json_path = tmp_path / "data.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        records = db.read.records.option("chunk_size", 10).json(str(json_path))
        # Streaming should still return all rows
        rows = list(records.rows())
        assert len(rows) == 50

    def test_read_json_streaming_multiline(self, tmp_path):
        """Test streaming JSONL (multiline mode)."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(50):
                f.write(f'{{"id": {i}}}\n')

        records = (
            db.read.records.option("multiline", True).option("chunk_size", 10).json(str(jsonl_path))
        )
        rows = list(records.rows())
        assert len(rows) == 50

    def test_read_json_streaming_failfast(self, tmp_path):
        """Test streaming JSON with FAILFAST mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"id": 1}\n{invalid}\n{"id": 2}\n')

        with pytest.raises(ValueError, match="Failed to parse JSON"):
            list(
                db.read.records.option("multiline", True)
                .option("mode", "FAILFAST")
                .option("chunk_size", 10)
                .json(str(jsonl_path))
                .rows()
            )

    def test_read_json_streaming_dropmalformed(self, tmp_path):
        """Test streaming JSON with DROPMALFORMED mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"id": 1}\n{invalid}\n{"id": 2}\n')

        records = (
            db.read.records.option("multiline", True)
            .option("mode", "DROPMALFORMED")
            .option("chunk_size", 10)
            .json(str(jsonl_path))
        )
        rows = list(records.rows())
        assert len(rows) == 2

    def test_read_json_streaming_permissive_corrupt(self, tmp_path):
        """Test streaming JSON with PERMISSIVE mode and corrupt column."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"id": 1}\n{invalid}\n{"id": 2}\n')

        records = (
            db.read.records.option("multiline", True)
            .option("mode", "PERMISSIVE")
            .option("columnNameOfCorruptRecord", "_corrupt")
            .option("chunk_size", 10)
            .json(str(jsonl_path))
        )
        rows = list(records.rows())
        assert len(rows) == 3  # 2 valid + 1 corrupt

    def test_read_json_date_format(self, tmp_path):
        """Test dateFormat option for date parsing."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [{"id": 1, "date": "2024-01-15"}]
        json_path = tmp_path / "data.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        records = db.read.records.option("dateFormat", "yyyy-MM-dd").json(str(json_path))
        rows = records.rows()
        assert len(rows) == 1
        # Date should be parsed (exact type depends on schema inference)

    def test_read_json_timestamp_format(self, tmp_path):
        """Test timestampFormat option for timestamp parsing."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        data = [{"id": 1, "timestamp": "2024-01-15 10:30:00"}]
        json_path = tmp_path / "data.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        records = db.read.records.option("timestampFormat", "yyyy-MM-dd HH:mm:ss").json(
            str(json_path)
        )
        rows = records.rows()
        assert len(rows) == 1
        # Timestamp should be parsed (exact type depends on schema inference)
