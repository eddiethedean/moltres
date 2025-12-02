"""Comprehensive tests for text reader covering all options and edge cases."""

from __future__ import annotations

import gzip

import pytest

from moltres import connect
from moltres.dataframe.io.readers.text_reader import read_text, read_text_stream
from moltres.table.schema import ColumnDef


class TestTextReader:
    """Test sync text reading operations."""

    def test_read_text_basic(self, tmp_path):
        """Test basic text reading."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2\nline3")

        records = read_text(str(text_path), db, schema=None, options={})
        rows = records.rows()
        assert len(rows) == 3
        assert rows[0]["value"] == "line1"

        db.close()

    def test_read_text_custom_column_name(self, tmp_path):
        """Test reading text with custom column name."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2")

        records = read_text(str(text_path), db, schema=None, options={}, column_name="text")
        rows = records.rows()
        assert len(rows) == 2
        assert "text" in rows[0]
        assert "value" not in rows[0]

        db.close()

    def test_read_text_wholetext(self, tmp_path):
        """Test reading text as whole file."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2\nline3")

        records = read_text(str(text_path), db, schema=None, options={"wholetext": True})
        rows = records.rows()
        assert len(rows) == 1
        assert "line1\nline2\nline3" in rows[0]["value"]

        db.close()

    def test_read_text_custom_line_sep(self, tmp_path):
        """Test reading text with custom line separator."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1|line2|line3")

        records = read_text(str(text_path), db, schema=None, options={"lineSep": "|"})
        rows = records.rows()
        assert len(rows) == 3

        db.close()

    def test_read_text_compressed(self, tmp_path):
        """Test reading compressed text."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt.gz"
        with gzip.open(text_path, "wt", encoding="utf-8") as f:
            f.write("line1\nline2")

        records = read_text(str(text_path), db, schema=None, options={"compression": "gzip"})
        rows = records.rows()
        assert len(rows) == 2

        db.close()

    def test_read_text_encoding(self, tmp_path):
        """Test reading text with custom encoding."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2", encoding="utf-8")

        records = read_text(str(text_path), db, schema=None, options={"encoding": "UTF-8"})
        rows = records.rows()
        assert len(rows) == 2

        db.close()

    def test_read_text_file_not_found(self, tmp_path):
        """Test reading non-existent text file raises FileNotFoundError."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        with pytest.raises(FileNotFoundError):
            read_text(str(tmp_path / "nonexistent.txt"), db, schema=None, options={})

        db.close()

    def test_read_text_empty_file(self, tmp_path):
        """Test reading empty text file returns empty records."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "empty.txt"
        text_path.write_text("")

        records = read_text(str(text_path), db, schema=None, options={})
        rows = records.rows()
        assert len(rows) == 0

        db.close()

    def test_read_text_stream(self, tmp_path):
        """Test reading text in streaming mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        lines = [f"line{i}" for i in range(1, 11)]
        text_path.write_text("\n".join(lines))

        records = read_text_stream(str(text_path), db, schema=None, options={"chunk_size": 3})
        chunks = []
        for chunk in records._generator():
            chunks.extend(chunk)

        assert len(chunks) == 10

        db.close()

    def test_read_text_stream_wholetext(self, tmp_path):
        """Test reading text stream with wholetext option."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2\nline3")

        records = read_text_stream(str(text_path), db, schema=None, options={"wholetext": True})
        chunks = []
        for chunk in records._generator():
            chunks.extend(chunk)

        assert len(chunks) == 1

        db.close()

    def test_read_text_stream_empty(self, tmp_path):
        """Test reading empty text file in streaming mode."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "empty.txt"
        text_path.write_text("")

        records = read_text_stream(str(text_path), db, schema=None, options={})
        # Empty file returns records with no generator (materialized empty data)
        if records._generator is None:
            # Materialized mode - check data directly
            rows = records.rows()
            assert len(rows) == 0
        else:
            chunks = []
            for chunk in records._generator():
                chunks.extend(chunk)
            assert len(chunks) == 0

        db.close()

    def test_read_text_with_schema(self, tmp_path):
        """Test reading text with explicit schema (schema is ignored for text)."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2")

        schema = [ColumnDef(name="value", type_name="TEXT")]
        records = read_text(str(text_path), db, schema=schema, options={})
        rows = records.rows()
        assert len(rows) == 2

        db.close()


@pytest.mark.asyncio
class TestAsyncTextReader:
    """Test async text reading operations."""

    async def test_read_text_basic(self, tmp_path):
        """Test basic async text reading."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2\nline3")

        records = await read_text(str(text_path), db, schema=None, options={})
        rows = await records.rows()
        assert len(rows) == 3
        assert rows[0]["value"] == "line1"

        await db.close()

    async def test_read_text_custom_column_name(self, tmp_path):
        """Test reading async text with custom column name."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2")

        records = await read_text(str(text_path), db, schema=None, options={}, column_name="text")
        rows = await records.rows()
        assert len(rows) == 2
        assert "text" in rows[0]

        await db.close()

    async def test_read_text_wholetext(self, tmp_path):
        """Test reading async text as whole file."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2\nline3")

        records = await read_text(str(text_path), db, schema=None, options={"wholetext": True})
        rows = await records.rows()
        assert len(rows) == 1
        assert "line1\nline2\nline3" in rows[0]["value"]

        await db.close()

    async def test_read_text_custom_line_sep(self, tmp_path):
        """Test reading async text with custom line separator."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1|line2|line3")

        records = await read_text(str(text_path), db, schema=None, options={"lineSep": "|"})
        rows = await records.rows()
        assert len(rows) == 3

        await db.close()

    async def test_read_text_compressed(self, tmp_path):
        """Test reading async compressed text."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt.gz"
        with gzip.open(text_path, "wt", encoding="utf-8") as f:
            f.write("line1\nline2")

        records = await read_text(str(text_path), db, schema=None, options={"compression": "gzip"})
        rows = await records.rows()
        assert len(rows) == 2

        await db.close()

    async def test_read_text_file_not_found(self, tmp_path):
        """Test reading non-existent async text file raises FileNotFoundError."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        with pytest.raises(FileNotFoundError):
            await read_text(str(tmp_path / "nonexistent.txt"), db, schema=None, options={})

        await db.close()

    async def test_read_text_empty_file(self, tmp_path):
        """Test reading empty async text file returns empty records."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "empty.txt"
        text_path.write_text("")

        records = await read_text(str(text_path), db, schema=None, options={})
        rows = await records.rows()
        assert len(rows) == 0

        await db.close()

    async def test_read_text_stream(self, tmp_path):
        """Test reading async text in streaming mode."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text_stream

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        lines = [f"line{i}" for i in range(1, 11)]
        text_path.write_text("\n".join(lines))

        records = await read_text_stream(str(text_path), db, schema=None, options={"chunk_size": 3})
        chunks = []
        async for chunk in records._generator():
            chunks.extend(chunk)

        assert len(chunks) == 10

        await db.close()

    async def test_read_text_stream_wholetext(self, tmp_path):
        """Test reading async text stream with wholetext option."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text_stream

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "data.txt"
        text_path.write_text("line1\nline2\nline3")

        records = await read_text_stream(
            str(text_path), db, schema=None, options={"wholetext": True}
        )
        chunks = []
        async for chunk in records._generator():
            chunks.extend(chunk)

        assert len(chunks) == 1

        await db.close()

    async def test_read_text_stream_empty(self, tmp_path):
        """Test reading empty async text file in streaming mode."""
        try:
            import aiosqlite  # noqa: F401
        except ImportError:
            pytest.skip("aiosqlite not installed")

        from moltres import async_connect
        from moltres.dataframe.io.readers.async_text_reader import read_text_stream

        db_path = tmp_path / "test.db"
        db = async_connect(f"sqlite+aiosqlite:///{db_path}")

        text_path = tmp_path / "empty.txt"
        text_path.write_text("")

        records = await read_text_stream(str(text_path), db, schema=None, options={})
        # Empty file returns records with no generator (materialized empty data)
        if records._generator is None:
            # Materialized mode - check data directly
            rows = await records.rows()
            assert len(rows) == 0
        else:
            chunks = []
            try:
                async for chunk in records._generator():
                    chunks.extend(chunk)
            except StopAsyncIteration:
                pass
            # Should have no chunks or empty chunks
            assert len(chunks) == 0 or all(len(c) == 0 for c in chunks)

        await db.close()
