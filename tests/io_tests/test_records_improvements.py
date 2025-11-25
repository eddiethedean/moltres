"""Tests for Records class improvements: select, rename, and convenience methods."""

from __future__ import annotations

import pytest

from moltres import connect, async_connect
from moltres.io.records import AsyncLazyRecords, AsyncRecords, LazyRecords, Records
from moltres.table.schema import column


class TestRecordsSelect:
    """Tests for Records.select() method."""

    def test_select_valid_columns(self, tmp_path):
        """Test select() with valid columns."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(
            _data=[{"id": 1, "name": "Alice", "age": 30}, {"id": 2, "name": "Bob", "age": 25}],
            _database=db,
        )
        selected = records.select("id", "name")
        rows = list(selected)
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "Alice"}
        assert rows[1] == {"id": 2, "name": "Bob"}

    def test_select_single_column(self, tmp_path):
        """Test select() with single column."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
        selected = records.select("id")
        rows = list(selected)
        assert rows[0] == {"id": 1}

    def test_select_invalid_column(self, tmp_path):
        """Test select() with invalid column raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
        with pytest.raises(ValueError, match="Column\\(s\\) not found: invalid_col"):
            records.select("invalid_col")

    def test_select_no_columns(self, tmp_path):
        """Test select() with no columns raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="select\\(\\) requires at least one column name"):
            records.select()

    def test_select_empty_records(self, tmp_path):
        """Test select() on empty Records raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[], _database=db)
        with pytest.raises(RuntimeError, match="Cannot select columns from empty Records"):
            records.select("id")

    def test_select_with_generator(self, tmp_path):
        """Test select() materializes generator-based Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def gen():
            yield [{"id": 1, "name": "Alice"}]
            yield [{"id": 2, "name": "Bob"}]

        records = Records(_generator=gen, _database=db)
        selected = records.select("id")
        rows = list(selected)
        assert len(rows) == 2
        assert rows[0] == {"id": 1}
        assert rows[1] == {"id": 2}

    def test_select_preserves_database(self, tmp_path):
        """Test select() preserves database reference."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        selected = records.select("id")
        assert selected._database is db

    def test_select_with_schema(self, tmp_path):
        """Test select() filters schema when available."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        schema = [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]
        records = Records(
            _data=[{"id": 1, "name": "Alice", "age": 30}],
            _schema=schema,
            _database=db,
        )
        selected = records.select("id", "name")
        assert selected._schema is not None
        assert len(selected._schema) == 2
        assert selected._schema[0].name == "id"
        assert selected._schema[1].name == "name"

    def test_select_with_schema_partial_match(self, tmp_path):
        """Test select() handles schema when some columns not in schema."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        schema = [column("id", "INTEGER")]  # Only id in schema, not name
        records = Records(
            _data=[{"id": 1, "name": "Alice"}],
            _schema=schema,
            _database=db,
        )
        selected = records.select("id", "name")
        assert selected._schema is not None
        assert len(selected._schema) == 1  # Only id in schema
        assert selected._schema[0].name == "id"


class TestRecordsRename:
    """Tests for Records.rename() method."""

    def test_rename_with_dict(self, tmp_path):
        """Test rename() with dict mapping."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
        renamed = records.rename({"id": "user_id", "name": "user_name"})
        rows = list(renamed)
        assert rows[0] == {"user_id": 1, "user_name": "Alice"}

    def test_rename_single_column(self, tmp_path):
        """Test rename() with single column."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
        renamed = records.rename("id", "user_id")
        rows = list(renamed)
        assert rows[0] == {"user_id": 1, "name": "Alice"}

    def test_rename_invalid_column(self, tmp_path):
        """Test rename() with invalid column raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="Column\\(s\\) not found: invalid_col"):
            records.rename("invalid_col", "new_name")

    def test_rename_name_conflict(self, tmp_path):
        """Test rename() with name conflict raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
        with pytest.raises(
            ValueError, match="New column name\\(s\\) conflict with existing columns"
        ):
            records.rename("id", "name")

    def test_rename_empty_dict(self, tmp_path):
        """Test rename() with empty dict raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="rename\\(\\) requires at least one column to rename"):
            records.rename({})

    def test_rename_missing_new_name(self, tmp_path):
        """Test rename() with string but no new_name raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="new_name is required when columns is a string"):
            records.rename("id")

    def test_rename_empty_records(self, tmp_path):
        """Test rename() on empty Records raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[], _database=db)
        with pytest.raises(RuntimeError, match="Cannot rename columns in empty Records"):
            records.rename("id", "user_id")

    def test_rename_with_generator(self, tmp_path):
        """Test rename() materializes generator-based Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def gen():
            yield [{"id": 1, "name": "Alice"}]

        records = Records(_generator=gen, _database=db)
        renamed = records.rename("id", "user_id")
        rows = list(renamed)
        assert rows[0] == {"user_id": 1, "name": "Alice"}

    def test_rename_with_schema(self, tmp_path):
        """Test rename() updates schema when available."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        schema = [column("id", "INTEGER"), column("name", "TEXT")]
        records = Records(
            _data=[{"id": 1, "name": "Alice"}],
            _schema=schema,
            _database=db,
        )
        renamed = records.rename("id", "user_id")
        assert renamed._schema is not None
        assert len(renamed._schema) == 2
        assert renamed._schema[0].name == "user_id"
        assert renamed._schema[0].type_name == "INTEGER"
        assert renamed._schema[1].name == "name"
        assert renamed._schema[1].type_name == "TEXT"

    def test_rename_with_schema_partial(self, tmp_path):
        """Test rename() updates schema when only some columns renamed."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        schema = [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]
        records = Records(
            _data=[{"id": 1, "name": "Alice", "age": 30}],
            _schema=schema,
            _database=db,
        )
        renamed = records.rename("id", "user_id")
        assert renamed._schema is not None
        assert len(renamed._schema) == 3
        assert renamed._schema[0].name == "user_id"
        assert renamed._schema[1].name == "name"
        assert renamed._schema[2].name == "age"


class TestRecordsConvenienceMethods:
    """Tests for Records convenience methods: head, tail, first, last."""

    def test_head_default(self, tmp_path):
        """Test head() with default n=5."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": i} for i in range(10)], _database=db)
        result = records.head()
        assert len(result) == 5
        assert result[0]["id"] == 0
        assert result[4]["id"] == 4

    def test_head_custom_n(self, tmp_path):
        """Test head() with custom n."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": i} for i in range(10)], _database=db)
        result = records.head(3)
        assert len(result) == 3

    def test_head_negative_n(self, tmp_path):
        """Test head() with negative n raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="n must be non-negative"):
            records.head(-1)

    def test_tail_default(self, tmp_path):
        """Test tail() with default n=5."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": i} for i in range(10)], _database=db)
        result = records.tail()
        assert len(result) == 5
        assert result[0]["id"] == 5
        assert result[4]["id"] == 9

    def test_tail_custom_n(self, tmp_path):
        """Test tail() with custom n."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": i} for i in range(10)], _database=db)
        result = records.tail(3)
        assert len(result) == 3
        assert result[0]["id"] == 7

    def test_tail_negative_n(self, tmp_path):
        """Test tail() with negative n raises error."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="n must be non-negative"):
            records.tail(-1)

    def test_first(self, tmp_path):
        """Test first() returns first row."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}, {"id": 2}], _database=db)
        result = records.first()
        assert result == {"id": 1}

    def test_first_empty(self, tmp_path):
        """Test first() returns None for empty Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[], _database=db)
        result = records.first()
        assert result is None

    def test_last(self, tmp_path):
        """Test last() returns last row."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[{"id": 1}, {"id": 2}], _database=db)
        result = records.last()
        assert result == {"id": 2}

    def test_last_empty(self, tmp_path):
        """Test last() returns None for empty Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")
        records = Records(_data=[], _database=db)
        result = records.last()
        assert result is None


class TestAsyncRecordsSelect:
    """Tests for AsyncRecords.select() method."""

    @pytest.mark.asyncio
    async def test_async_select_valid_columns(self, tmp_path):
        """Test async select() with valid columns."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        )
        selected = await records.select("id", "name")
        rows = await selected.rows()
        assert len(rows) == 2
        assert rows[0] == {"id": 1, "name": "Alice"}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_select_invalid_column(self, tmp_path):
        """Test async select() with invalid column raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            await records.select("invalid_col")
        await db.close()

    @pytest.mark.asyncio
    async def test_async_select_no_columns(self, tmp_path):
        """Test async select() with no columns raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="select\\(\\) requires at least one column name"):
            await records.select()
        await db.close()

    @pytest.mark.asyncio
    async def test_async_select_empty_records(self, tmp_path):
        """Test async select() on empty AsyncRecords raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[], _database=db)
        with pytest.raises(RuntimeError, match="Cannot select columns from empty AsyncRecords"):
            await records.select("id")
        await db.close()

    @pytest.mark.asyncio
    async def test_async_select_with_schema(self, tmp_path):
        """Test async select() filters schema when available."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        schema = [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]
        records = AsyncRecords(
            _data=[{"id": 1, "name": "Alice", "age": 30}],
            _schema=schema,
            _database=db,
        )
        selected = await records.select("id", "name")
        assert selected._schema is not None
        assert len(selected._schema) == 2
        await db.close()


class TestAsyncRecordsRename:
    """Tests for AsyncRecords.rename() method."""

    @pytest.mark.asyncio
    async def test_async_rename_with_dict(self, tmp_path):
        """Test async rename() with dict mapping."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
        renamed = await records.rename({"id": "user_id"})
        rows = await renamed.rows()
        assert rows[0] == {"user_id": 1, "name": "Alice"}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_rename_with_schema(self, tmp_path):
        """Test async rename() updates schema when available."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        schema = [column("id", "INTEGER"), column("name", "TEXT")]
        records = AsyncRecords(
            _data=[{"id": 1, "name": "Alice"}],
            _schema=schema,
            _database=db,
        )
        renamed = await records.rename("id", "user_id")
        assert renamed._schema is not None
        assert len(renamed._schema) == 2
        assert renamed._schema[0].name == "user_id"
        await db.close()

    @pytest.mark.asyncio
    async def test_async_rename_empty_records(self, tmp_path):
        """Test async rename() on empty AsyncRecords raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[], _database=db)
        with pytest.raises(RuntimeError, match="Cannot rename columns in empty AsyncRecords"):
            await records.rename("id", "user_id")
        await db.close()

    @pytest.mark.asyncio
    async def test_async_rename_missing_new_name(self, tmp_path):
        """Test async rename() with string but no new_name raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="new_name is required when columns is a string"):
            await records.rename("id")
        await db.close()

    @pytest.mark.asyncio
    async def test_async_rename_empty_dict(self, tmp_path):
        """Test async rename() with empty dict raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="rename\\(\\) requires at least one column to rename"):
            await records.rename({})
        await db.close()

    @pytest.mark.asyncio
    async def test_async_rename_invalid_column(self, tmp_path):
        """Test async rename() with invalid column raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            await records.rename("invalid_col", "new_name")
        await db.close()

    @pytest.mark.asyncio
    async def test_async_rename_name_conflict(self, tmp_path):
        """Test async rename() with name conflict raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
        with pytest.raises(
            ValueError, match="New column name\\(s\\) conflict with existing columns"
        ):
            await records.rename("id", "name")
        await db.close()


class TestAsyncRecordsConvenienceMethods:
    """Tests for AsyncRecords convenience methods."""

    @pytest.mark.asyncio
    async def test_async_head(self, tmp_path):
        """Test async head()."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": i} for i in range(10)], _database=db)
        result = await records.head(3)
        assert len(result) == 3
        await db.close()

    @pytest.mark.asyncio
    async def test_async_head_negative_n(self, tmp_path):
        """Test async head() with negative n raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="n must be non-negative"):
            await records.head(-1)
        await db.close()

    @pytest.mark.asyncio
    async def test_async_first(self, tmp_path):
        """Test async first()."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        result = await records.first()
        assert result == {"id": 1}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_tail(self, tmp_path):
        """Test async tail()."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": i} for i in range(10)], _database=db)
        result = await records.tail(3)
        assert len(result) == 3
        assert result[0]["id"] == 7
        await db.close()

    @pytest.mark.asyncio
    async def test_async_tail_negative_n(self, tmp_path):
        """Test async tail() with negative n raises error."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}], _database=db)
        with pytest.raises(ValueError, match="n must be non-negative"):
            await records.tail(-1)
        await db.close()

    @pytest.mark.asyncio
    async def test_async_last(self, tmp_path):
        """Test async last()."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[{"id": 1}, {"id": 2}], _database=db)
        result = await records.last()
        assert result == {"id": 2}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_last_empty(self, tmp_path):
        """Test async last() returns None for empty AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
        records = AsyncRecords(_data=[], _database=db)
        result = await records.last()
        assert result is None
        await db.close()


class TestLazyRecordsDelegation:
    """Tests for LazyRecords delegation to materialized Records."""

    def test_lazy_select(self, tmp_path):
        """Test LazyRecords.select() delegates to materialized Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def read_func():
            return Records(_data=[{"id": 1, "name": "Alice"}], _database=db)

        lazy_records = LazyRecords(_read_func=read_func, _database=db)
        selected = lazy_records.select("id")
        # Should return Records (not LazyRecords) since materialized
        assert isinstance(selected, Records)
        rows = list(selected)
        assert rows[0] == {"id": 1}

    def test_lazy_rename(self, tmp_path):
        """Test LazyRecords.rename() delegates to materialized Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def read_func():
            return Records(_data=[{"id": 1}], _database=db)

        lazy_records = LazyRecords(_read_func=read_func, _database=db)
        renamed = lazy_records.rename("id", "user_id")
        assert isinstance(renamed, Records)
        rows = list(renamed)
        assert rows[0] == {"user_id": 1}

    def test_lazy_head(self, tmp_path):
        """Test LazyRecords.head() delegates to materialized Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def read_func():
            return Records(_data=[{"id": i} for i in range(10)], _database=db)

        lazy_records = LazyRecords(_read_func=read_func, _database=db)
        result = lazy_records.head(3)
        assert len(result) == 3

    def test_lazy_tail(self, tmp_path):
        """Test LazyRecords.tail() delegates to materialized Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def read_func():
            return Records(_data=[{"id": i} for i in range(10)], _database=db)

        lazy_records = LazyRecords(_read_func=read_func, _database=db)
        result = lazy_records.tail(3)
        assert len(result) == 3
        assert result[0]["id"] == 7

    def test_lazy_first(self, tmp_path):
        """Test LazyRecords.first() delegates to materialized Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def read_func():
            return Records(_data=[{"id": 1}, {"id": 2}], _database=db)

        lazy_records = LazyRecords(_read_func=read_func, _database=db)
        result = lazy_records.first()
        assert result == {"id": 1}

    def test_lazy_last(self, tmp_path):
        """Test LazyRecords.last() delegates to materialized Records."""
        db = connect(f"sqlite:///{tmp_path / 'test.db'}")

        def read_func():
            return Records(_data=[{"id": 1}, {"id": 2}], _database=db)

        lazy_records = LazyRecords(_read_func=read_func, _database=db)
        result = lazy_records.last()
        assert result == {"id": 2}


class TestAsyncLazyRecordsDelegation:
    """Tests for AsyncLazyRecords delegation to materialized AsyncRecords."""

    @pytest.mark.asyncio
    async def test_async_lazy_select(self, tmp_path):
        """Test AsyncLazyRecords.select() delegates to materialized AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

        async def read_func():
            return AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)

        async_lazy_records = AsyncLazyRecords(_read_func=read_func, _database=db)
        selected = await async_lazy_records.select("id")
        # Should return AsyncRecords (not AsyncLazyRecords) since materialized
        assert isinstance(selected, AsyncRecords)
        rows = await selected.rows()
        assert rows[0] == {"id": 1}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_lazy_rename(self, tmp_path):
        """Test AsyncLazyRecords.rename() delegates to materialized AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

        async def read_func():
            return AsyncRecords(_data=[{"id": 1}], _database=db)

        async_lazy_records = AsyncLazyRecords(_read_func=read_func, _database=db)
        renamed = await async_lazy_records.rename("id", "user_id")
        assert isinstance(renamed, AsyncRecords)
        rows = await renamed.rows()
        assert rows[0] == {"user_id": 1}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_lazy_head(self, tmp_path):
        """Test AsyncLazyRecords.head() delegates to materialized AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

        async def read_func():
            return AsyncRecords(_data=[{"id": i} for i in range(10)], _database=db)

        async_lazy_records = AsyncLazyRecords(_read_func=read_func, _database=db)
        result = await async_lazy_records.head(3)
        assert len(result) == 3
        await db.close()

    @pytest.mark.asyncio
    async def test_async_lazy_tail(self, tmp_path):
        """Test AsyncLazyRecords.tail() delegates to materialized AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

        async def read_func():
            return AsyncRecords(_data=[{"id": i} for i in range(10)], _database=db)

        async_lazy_records = AsyncLazyRecords(_read_func=read_func, _database=db)
        result = await async_lazy_records.tail(3)
        assert len(result) == 3
        assert result[0]["id"] == 7
        await db.close()

    @pytest.mark.asyncio
    async def test_async_lazy_first(self, tmp_path):
        """Test AsyncLazyRecords.first() delegates to materialized AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

        async def read_func():
            return AsyncRecords(_data=[{"id": 1}, {"id": 2}], _database=db)

        async_lazy_records = AsyncLazyRecords(_read_func=read_func, _database=db)
        result = await async_lazy_records.first()
        assert result == {"id": 1}
        await db.close()

    @pytest.mark.asyncio
    async def test_async_lazy_last(self, tmp_path):
        """Test AsyncLazyRecords.last() delegates to materialized AsyncRecords."""
        db = async_connect(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

        async def read_func():
            return AsyncRecords(_data=[{"id": 1}, {"id": 2}], _database=db)

        async_lazy_records = AsyncLazyRecords(_read_func=read_func, _database=db)
        result = await async_lazy_records.last()
        assert result == {"id": 2}
        await db.close()
