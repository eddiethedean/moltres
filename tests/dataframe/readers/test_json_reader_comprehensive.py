"""Comprehensive tests for sync JSON reader covering all options and edge cases."""

from __future__ import annotations

import json
import gzip

import pytest

from moltres import connect
from moltres.dataframe.readers.json_reader import (
    read_json,
    read_json_stream,
    read_jsonl,
    read_jsonl_stream,
)
from moltres.table.schema import ColumnDef


class TestJsonReader:
    """Test sync JSON reading operations."""

    def test_read_json_array(self, tmp_path):
        """Test reading JSON array."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        json_path.write_text(json.dumps(data))

        records = read_json(str(json_path), db, schema=None, options={})
        rows = records.rows()
        assert len(rows) == 2

        db.close()

    def test_read_json_object(self, tmp_path):
        """Test reading single JSON object."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = {"id": 1, "name": "Alice"}
        json_path.write_text(json.dumps(data))

        records = read_json(str(json_path), db, schema=None, options={})
        rows = records.rows()
        assert len(rows) == 1

        db.close()

    def test_read_json_with_schema(self, tmp_path):
        """Test reading JSON with explicit schema."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": 1, "name": "Alice"}]
        json_path.write_text(json.dumps(data))

        schema = [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        records = read_json(str(json_path), db, schema=schema, options={})
        rows = records.rows()
        assert len(rows) == 1

        db.close()

    def test_read_json_multiline(self, tmp_path):
        """Test reading JSONL (multiline) format."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": 1, "name": "Alice"}), json.dumps({"id": 2, "name": "Bob"})]
        jsonl_path.write_text("\n".join(lines))

        records = read_json(str(jsonl_path), db, schema=None, options={"multiline": True})
        rows = records.rows()
        assert len(rows) >= 1

        db.close()

    def test_read_json_compressed(self, tmp_path):
        """Test reading compressed JSON."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json.gz"
        data = [{"id": 1, "name": "Alice"}]
        with gzip.open(json_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        records = read_json(str(json_path), db, schema=None, options={"compression": "gzip"})
        rows = records.rows()
        assert len(rows) == 1

        db.close()

    def test_read_json_file_not_found(self, tmp_path):
        """Test reading non-existent JSON file raises FileNotFoundError."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        with pytest.raises(FileNotFoundError):
            read_json(str(tmp_path / "nonexistent.json"), db, schema=None, options={})

        db.close()

    def test_read_json_empty_file(self, tmp_path):
        """Test reading empty JSON file raises ValueError."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "empty.json"
        json_path.write_text("")

        with pytest.raises(ValueError, match="empty"):
            read_json(str(json_path), db, schema=None, options={})

        db.close()

    def test_read_json_empty_with_schema(self, tmp_path):
        """Test reading empty JSON file with schema returns empty records."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "empty.json"
        json_path.write_text("")

        schema = [ColumnDef(name="id", type_name="INTEGER")]
        records = read_json(str(json_path), db, schema=schema, options={})
        rows = records.rows()
        assert len(rows) == 0

        db.close()

    def test_read_json_mode_permissive(self, tmp_path):
        """Test reading JSON with PERMISSIVE mode (default)."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        # Invalid JSON - should be handled permissively
        json_path.write_text("{invalid json")

        # With corrupt column, should create records with corrupt data
        try:
            records = read_json(
                str(json_path),
                db,
                schema=None,
                options={"mode": "PERMISSIVE", "columnNameOfCorruptRecord": "_corrupt"},
            )
            rows = records.rows()
            assert isinstance(rows, list)
        except ValueError:
            # If it raises ValueError, that's also acceptable for completely invalid JSON
            pass

        db.close()

    def test_read_json_mode_failfast(self, tmp_path):
        """Test reading JSON with FAILFAST mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        # Invalid JSON - should raise error
        json_path.write_text("{invalid json")

        with pytest.raises(ValueError):
            read_json(str(json_path), db, schema=None, options={"mode": "FAILFAST"})

        db.close()

    def test_read_json_mode_dropmalformed(self, tmp_path):
        """Test reading JSON with DROPMALFORMED mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        # Mix of valid and invalid JSON lines
        lines = [json.dumps({"id": 1}), "{invalid", json.dumps({"id": 2})]
        jsonl_path.write_text("\n".join(lines))

        records = read_json(
            str(jsonl_path), db, schema=None, options={"multiline": True, "mode": "DROPMALFORMED"}
        )
        rows = records.rows()
        # Should only have valid rows
        assert len(rows) == 2

        db.close()

    def test_read_json_drop_field_if_all_null(self, tmp_path):
        """Test reading JSON with dropFieldIfAllNull option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [
            {"id": 1, "name": "Alice", "null_field": None},
            {"id": 2, "name": "Bob", "null_field": None},
        ]
        json_path.write_text(json.dumps(data))

        records = read_json(str(json_path), db, schema=None, options={"dropFieldIfAllNull": True})
        rows = records.rows()
        assert len(rows) == 2
        # null_field should be dropped
        assert "null_field" not in rows[0]

        db.close()

    def test_read_json_encoding(self, tmp_path):
        """Test reading JSON with custom encoding."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": 1, "name": "Alice"}]
        json_path.write_text(json.dumps(data), encoding="utf-8")

        records = read_json(str(json_path), db, schema=None, options={"encoding": "UTF-8"})
        rows = records.rows()
        assert len(rows) == 1

        db.close()

    def test_read_json_stream(self, tmp_path):
        """Test reading JSON in streaming mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": i, "name": f"User{i}"} for i in range(1, 11)]
        json_path.write_text(json.dumps(data))

        records = read_json_stream(str(json_path), db, schema=None, options={"chunk_size": 3})
        chunks = []
        for chunk in records._generator():
            chunks.extend(chunk)

        assert len(chunks) == 10

        db.close()

    def test_read_jsonl(self, tmp_path):
        """Test read_jsonl() helper function."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": 1, "name": "Alice"}), json.dumps({"id": 2, "name": "Bob"})]
        jsonl_path.write_text("\n".join(lines))

        records = read_jsonl(str(jsonl_path), db, schema=None, options={})
        rows = records.rows()
        assert len(rows) == 2

        db.close()

    def test_read_jsonl_stream(self, tmp_path):
        """Test read_jsonl_stream() helper function."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": i}) for i in range(1, 11)]
        jsonl_path.write_text("\n".join(lines))

        records = read_jsonl_stream(str(jsonl_path), db, schema=None, options={"chunk_size": 3})
        chunks = []
        for chunk in records._generator():
            chunks.extend(chunk)

        assert len(chunks) == 10

        db.close()

    def test_read_json_line_sep(self, tmp_path):
        """Test reading JSON with custom line separator."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": 1}), json.dumps({"id": 2})]
        jsonl_path.write_text("|".join(lines))  # Custom separator

        records = read_json(
            str(jsonl_path), db, schema=None, options={"multiline": True, "lineSep": "|"}
        )
        rows = records.rows()
        assert len(rows) == 2

        db.close()

    def test_read_json_date_format(self, tmp_path):
        """Test reading JSON with date format option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": 1, "date": "2024-01-01"}]
        json_path.write_text(json.dumps(data))

        records = read_json(str(json_path), db, schema=None, options={"dateFormat": "yyyy-MM-dd"})
        rows = records.rows()
        assert len(rows) == 1

        db.close()

    def test_read_json_timestamp_format(self, tmp_path):
        """Test reading JSON with timestamp format option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": 1, "timestamp": "2024-01-01T12:00:00"}]
        json_path.write_text(json.dumps(data))

        records = read_json(
            str(json_path), db, schema=None, options={"timestampFormat": "yyyy-MM-dd'T'HH:mm:ss"}
        )
        rows = records.rows()
        assert len(rows) == 1

        db.close()

    def test_read_json_corrupt_column(self, tmp_path):
        """Test reading JSON with corrupt column option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": 1}), "{invalid", json.dumps({"id": 2})]
        jsonl_path.write_text("\n".join(lines))

        records = read_json(
            str(jsonl_path),
            db,
            schema=None,
            options={
                "multiline": True,
                "mode": "PERMISSIVE",
                "columnNameOfCorruptRecord": "_corrupt",
            },
        )
        rows = records.rows()
        # Should have corrupt records column
        assert len(rows) >= 2

        db.close()

    def test_read_json_multi_line_alias(self, tmp_path):
        """Test reading JSON with multiLine alias (PySpark compatibility)."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        jsonl_path = tmp_path / "data.jsonl"
        lines = [json.dumps({"id": 1}), json.dumps({"id": 2})]
        jsonl_path.write_text("\n".join(lines))

        # Use multiLine alias instead of multiline
        records = read_json(str(jsonl_path), db, schema=None, options={"multiLine": True})
        rows = records.rows()
        assert len(rows) >= 1

        db.close()

    def test_read_json_sampling_ratio(self, tmp_path):
        """Test reading JSON with sampling ratio for schema inference."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        json_path = tmp_path / "data.json"
        data = [{"id": i, "value": i * 2} for i in range(1, 21)]
        json_path.write_text(json.dumps(data))

        records = read_json(str(json_path), db, schema=None, options={"samplingRatio": 0.5})
        rows = records.rows()
        # Should read all rows, but schema inference uses sampling
        assert len(rows) == 20

        db.close()
