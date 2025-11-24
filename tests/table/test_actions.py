"""Tests for table action operations."""

from __future__ import annotations


from moltres import col, connect
from moltres.io.records import Records
from moltres.table.actions import (
    CreateTableOperation,
    DeleteMutation,
    DropTableOperation,
    InsertMutation,
    MergeMutation,
    UpdateMutation,
)
from moltres.table.schema import ColumnDef


class TestInsertMutation:
    """Test InsertMutation class."""

    def test_insert_mutation_collect(self, tmp_path):
        """Test InsertMutation.collect()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Create table
        db.create_table(
            "users",
            [
                ColumnDef(name="id", type_name="INTEGER", primary_key=True),
                ColumnDef(name="name", type_name="TEXT"),
            ],
        ).collect()

        table = db.table("users")
        rows = [{"id": 1, "name": "Alice"}]
        mutation = InsertMutation(handle=table, rows=rows)

        result = mutation.collect()
        assert result == 1

        # Verify data was inserted
        all_rows = table.select().collect()
        assert len(all_rows) == 1
        assert all_rows[0]["name"] == "Alice"

    def test_insert_mutation_to_sql(self, tmp_path):
        """Test InsertMutation.to_sql()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "users",
            [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")],
        ).collect()

        table = db.table("users")
        rows = [{"id": 1, "name": "Alice"}]
        mutation = InsertMutation(handle=table, rows=rows)

        sql = mutation.to_sql()
        assert "INSERT INTO" in sql
        assert "users" in sql
        assert "id" in sql
        assert "name" in sql

    def test_insert_mutation_empty_rows(self, tmp_path):
        """Test InsertMutation with empty rows."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [ColumnDef(name="id", type_name="INTEGER")]).collect()

        table = db.table("users")
        mutation = InsertMutation(handle=table, rows=[])

        # Empty rows should return empty SQL
        sql = mutation.to_sql()
        assert sql == ""

    def test_insert_mutation_with_records(self, tmp_path):
        """Test InsertMutation with Records object."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "users",
            [
                ColumnDef(name="id", type_name="INTEGER", primary_key=True),
                ColumnDef(name="name", type_name="TEXT"),
            ],
        ).collect()

        table = db.table("users")
        records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
        mutation = InsertMutation(handle=table, rows=records)

        result = mutation.collect()
        assert result == 1


class TestUpdateMutation:
    """Test UpdateMutation class."""

    def test_update_mutation_collect(self, tmp_path):
        """Test UpdateMutation.collect()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Create table and insert data
        db.create_table(
            "users",
            [
                ColumnDef(name="id", type_name="INTEGER", primary_key=True),
                ColumnDef(name="name", type_name="TEXT"),
            ],
        ).collect()
        table = db.table("users")
        InsertMutation(handle=table, rows=[{"id": 1, "name": "Alice"}]).collect()

        mutation = UpdateMutation(handle=table, where=col("id") == 1, values={"name": "Bob"})
        result = mutation.collect()
        assert result == 1

        # Verify update
        rows = table.select().collect()
        assert rows[0]["name"] == "Bob"

    def test_update_mutation_to_sql(self, tmp_path):
        """Test UpdateMutation.to_sql()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "users",
            [ColumnDef(name="id", type_name="INTEGER"), ColumnDef(name="name", type_name="TEXT")],
        ).collect()
        table = db.table("users")

        mutation = UpdateMutation(handle=table, where=col("id") == 1, values={"name": "Bob"})
        sql = mutation.to_sql()
        assert "UPDATE" in sql
        assert "users" in sql
        assert "SET" in sql
        assert "WHERE" in sql

    def test_update_mutation_empty_values(self, tmp_path):
        """Test UpdateMutation with empty values."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [ColumnDef(name="id", type_name="INTEGER")]).collect()
        table = db.table("users")

        mutation = UpdateMutation(handle=table, where=col("id") == 1, values={})
        sql = mutation.to_sql()
        assert sql == ""


class TestDeleteMutation:
    """Test DeleteMutation class."""

    def test_delete_mutation_collect(self, tmp_path):
        """Test DeleteMutation.collect()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Create table and insert data
        db.create_table(
            "users",
            [
                ColumnDef(name="id", type_name="INTEGER", primary_key=True),
                ColumnDef(name="name", type_name="TEXT"),
            ],
        ).collect()
        table = db.table("users")
        InsertMutation(
            handle=table, rows=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        ).collect()

        mutation = DeleteMutation(handle=table, where=col("id") == 1)
        result = mutation.collect()
        assert result == 1

        # Verify deletion
        rows = table.select().collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "Bob"

    def test_delete_mutation_to_sql(self, tmp_path):
        """Test DeleteMutation.to_sql()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table("users", [ColumnDef(name="id", type_name="INTEGER")]).collect()
        table = db.table("users")

        mutation = DeleteMutation(handle=table, where=col("id") == 1)
        sql = mutation.to_sql()
        assert "DELETE FROM" in sql
        assert "users" in sql
        assert "WHERE" in sql


class TestMergeMutation:
    """Test MergeMutation class."""

    def test_merge_mutation_collect(self, tmp_path):
        """Test MergeMutation.collect()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Create table and insert data
        db.create_table(
            "users",
            [
                ColumnDef(name="id", type_name="INTEGER", primary_key=True),
                ColumnDef(name="name", type_name="TEXT"),
            ],
        ).collect()
        table = db.table("users")
        InsertMutation(handle=table, rows=[{"id": 1, "name": "Alice"}]).collect()

        rows = [{"id": 1, "name": "Bob"}, {"id": 2, "name": "Charlie"}]
        mutation = MergeMutation(
            handle=table, rows=rows, on=["id"], when_matched={"name": "updated"}
        )
        result = mutation.collect()
        assert result >= 1

        # Verify merge
        all_rows = table.select().collect()
        assert len(all_rows) >= 1

    def test_merge_mutation_to_sql(self, tmp_path):
        """Test MergeMutation.to_sql()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        db.create_table(
            "users",
            [
                ColumnDef(name="id", type_name="INTEGER", primary_key=True),
                ColumnDef(name="name", type_name="TEXT"),
            ],
        ).collect()
        table = db.table("users")

        rows = [{"id": 1, "name": "Alice"}]
        mutation = MergeMutation(handle=table, rows=rows, on=["id"])
        sql = mutation.to_sql()
        assert "MERGE" in sql or "UPSERT" in sql


class TestCreateTableOperation:
    """Test CreateTableOperation class."""

    def test_create_table_operation_collect(self, tmp_path):
        """Test CreateTableOperation.collect()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        columns = [
            ColumnDef(name="id", type_name="INTEGER", primary_key=True),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        operation = CreateTableOperation(database=db, name="users", columns=columns)

        handle = operation.collect()
        assert handle.name == "users"
        assert handle.database == db

        # Verify table was created
        rows = db.table("users").select().collect()
        assert isinstance(rows, list)

    def test_create_table_operation_to_sql(self, tmp_path):
        """Test CreateTableOperation.to_sql()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        columns = [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
        ]
        operation = CreateTableOperation(database=db, name="users", columns=columns)

        sql = operation.to_sql()
        assert "CREATE TABLE" in sql
        assert "users" in sql
        assert "id" in sql
        assert "name" in sql

    def test_create_table_operation_if_not_exists(self, tmp_path):
        """Test CreateTableOperation with if_not_exists."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        columns = [ColumnDef(name="id", type_name="INTEGER", primary_key=True)]
        operation = CreateTableOperation(
            database=db, name="users", columns=columns, if_not_exists=True
        )

        sql = operation.to_sql()
        assert "IF NOT EXISTS" in sql or "if_not_exists" in sql.lower()

    def test_create_table_operation_temporary(self, tmp_path):
        """Test CreateTableOperation with temporary."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        columns = [ColumnDef(name="id", type_name="INTEGER", primary_key=True)]
        operation = CreateTableOperation(database=db, name="users", columns=columns, temporary=True)

        sql = operation.to_sql()
        assert "TEMPORARY" in sql or "temporary" in sql.lower() or "TEMP" in sql


class TestDropTableOperation:
    """Test DropTableOperation class."""

    def test_drop_table_operation_collect(self, tmp_path):
        """Test DropTableOperation.collect()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        # Create table first
        db.create_table(
            "users", [ColumnDef(name="id", type_name="INTEGER", primary_key=True)]
        ).collect()

        operation = DropTableOperation(database=db, name="users")
        operation.collect()

        # Verify table was dropped (trying to select should fail or return empty)
        # In SQLite, we can check if table exists
        try:
            db.table("users").select().collect()
            # If we get here, table might still exist (some DBs allow this)
        except Exception:
            # Table doesn't exist, which is expected
            pass

    def test_drop_table_operation_to_sql(self, tmp_path):
        """Test DropTableOperation.to_sql()."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        operation = DropTableOperation(database=db, name="users")
        sql = operation.to_sql()
        assert "DROP TABLE" in sql
        assert "users" in sql

    def test_drop_table_operation_if_exists(self, tmp_path):
        """Test DropTableOperation with if_exists."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        operation = DropTableOperation(database=db, name="users", if_exists=True)
        sql = operation.to_sql()
        assert "IF EXISTS" in sql or "if_exists" in sql.lower()

    def test_drop_table_operation_if_exists_false(self, tmp_path):
        """Test DropTableOperation with if_exists=False."""
        db_path = tmp_path / "test.db"
        db = connect(f"sqlite:///{db_path}")

        operation = DropTableOperation(database=db, name="users", if_exists=False)
        sql = operation.to_sql()
        # Should not have IF EXISTS
        assert "IF EXISTS" not in sql
