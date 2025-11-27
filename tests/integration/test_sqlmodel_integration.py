"""Tests for SQLModel integration with Moltres."""

import pytest
import uuid

try:
    from sqlmodel import SQLModel, Field

    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False

from moltres import connect
from moltres.table.schema import column
from moltres.io.records import Records


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not installed")
class TestSQLModelIntegration:
    """Test SQLModel integration features."""

    def test_sqlmodel_detection(self):
        """Test that SQLModel models are detected correctly."""
        from moltres.utils.sqlmodel_integration import is_sqlmodel_model

        class User(SQLModel, table=True):
            __tablename__ = "users"
            id: int = Field(primary_key=True)
            name: str

        assert is_sqlmodel_model(User) is True
        assert is_sqlmodel_model(str) is False
        assert is_sqlmodel_model(None) is False

    def test_get_sqlmodel_table_name(self):
        """Test extracting table name from SQLModel."""
        from moltres.utils.sqlmodel_integration import get_sqlmodel_table_name

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str

        assert get_sqlmodel_table_name(User) == table_name

    def test_table_with_sqlmodel(self):
        """Test creating table handle with SQLModel."""
        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str
            email: str

        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()

        # Create table handle using SQLModel
        users_table = db.table(User)
        assert users_table.name == table_name
        assert users_table.model == User
        db.close()

    def test_collect_returns_sqlmodel_instances(self):
        """Test that collect() returns SQLModel instances when model is attached."""
        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str
            email: str

        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()

        # Insert data
        Records(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            _database=db,
        ).insert_into(table_name)

        # Get DataFrame with model attached
        df = db.table(User).select()
        results = df.collect()

        # Check that results are SQLModel instances
        assert len(results) == 2
        assert isinstance(results[0], User)
        assert results[0].name == "Alice"
        assert results[0].email == "alice@example.com"
        assert results[1].name == "Bob"

        db.close()

    def test_with_model_method(self):
        """Test the with_model() method."""
        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str

        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into(table_name)

        # Create DataFrame without model
        df = db.table(table_name).select()
        results1 = df.collect()
        assert isinstance(results1[0], dict)

        # Attach model
        df_with_model = df.with_model(User)
        results2 = df_with_model.collect()
        assert isinstance(results2[0], User)
        assert results2[0].name == "Alice"

        db.close()

    def test_with_model_chaining(self):
        """Test that with_model() works with chained operations."""
        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str
            age: int

        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
            _database=db,
        ).insert_into(table_name)

        from moltres import col

        # Chain operations with model attached
        df = db.table(User).select().where(col("age") > 25).order_by("age")
        results = df.collect()

        assert len(results) == 2
        assert all(isinstance(r, User) for r in results)
        # Results filtered for age > 25: Alice (30) and Charlie (35), sorted by age
        assert results[0].name == "Alice"  # Age 30, sorted
        assert results[1].name == "Charlie"  # Age 35

        db.close()

    def test_integration_helper_with_sqlmodel(self):
        """Test integration helper function."""
        from moltres.integration import with_sqlmodel

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str

        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        ).insert_into(table_name)

        df = db.table(table_name).select()
        df_with_model = with_sqlmodel(df, User)
        results = df_with_model.collect()

        assert isinstance(results[0], User)
        assert results[0].name == "Alice"

        db.close()

    def test_streaming_with_sqlmodel(self):
        """Test streaming with SQLModel."""
        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str

        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ],
            _database=db,
        ).insert_into(table_name)

        df = db.table(User).select()
        stream_results = df.collect(stream=True)

        chunk = next(stream_results)
        assert len(chunk) == 2
        assert all(isinstance(r, User) for r in chunk)
        assert chunk[0].name == "Alice"

        db.close()

    def test_with_model_type_error(self):
        """Test that with_model() raises TypeError for non-SQLModel classes."""
        table_name = f"users_{uuid.uuid4().hex[:8]}"
        db = connect("sqlite:///:memory:")
        db.create_table(
            table_name,
            [column("id", "INTEGER")],
        ).collect()

        df = db.table(table_name).select()

        # Should raise TypeError for non-SQLModel class
        with pytest.raises(TypeError):
            df.with_model(str)

        db.close()

    def test_sqlmodel_to_dict(self):
        """Test converting SQLModel instance to dict."""
        from moltres.utils.sqlmodel_integration import sqlmodel_to_dict

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str
            email: str

        user = User(id=1, name="Alice", email="alice@example.com")
        user_dict = sqlmodel_to_dict(user)

        assert user_dict == {"id": 1, "name": "Alice", "email": "alice@example.com"}

    def test_dict_to_sqlmodel(self):
        """Test converting dict to SQLModel instance."""
        from moltres.utils.sqlmodel_integration import dict_to_sqlmodel

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str
            email: str

        user_dict = {"id": 1, "name": "Alice", "email": "alice@example.com"}
        user = dict_to_sqlmodel(user_dict, User)

        assert isinstance(user, User)
        assert user.id == 1
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_rows_to_sqlmodels(self):
        """Test converting list of dicts to SQLModel instances."""
        from moltres.utils.sqlmodel_integration import rows_to_sqlmodels

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str

        rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        users = rows_to_sqlmodels(rows, User)

        assert len(users) == 2
        assert all(isinstance(u, User) for u in users)
        assert users[0].name == "Alice"
        assert users[1].name == "Bob"


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not installed")
class TestSQLModelAsyncIntegration:
    """Test async SQLModel integration features."""

    @pytest.mark.asyncio
    async def test_async_collect_returns_sqlmodel_instances(self):
        """Test that async collect() returns SQLModel instances."""
        from moltres import async_connect
        from moltres.io.records import AsyncRecords

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str
            email: str

        db = async_connect("sqlite+aiosqlite:///:memory:")
        await db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()

        # Insert data
        records = AsyncRecords(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
            ],
            _database=db,
        )
        await records.insert_into(table_name)

        # Get DataFrame with model attached
        table_handle = await db.table(User)
        df = table_handle.select()
        results = await df.collect()

        # Check that results are SQLModel instances
        assert len(results) == 1
        assert isinstance(results[0], User)
        assert results[0].name == "Alice"

        await db.close()

    @pytest.mark.asyncio
    async def test_async_with_model(self):
        """Test async with_model() method."""
        from moltres import async_connect
        from moltres.io.records import AsyncRecords

        table_name = f"users_{uuid.uuid4().hex[:8]}"

        class User(SQLModel, table=True):
            __tablename__ = table_name
            id: int = Field(primary_key=True)
            name: str

        db = async_connect("sqlite+aiosqlite:///:memory:")
        await db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
            ],
        ).collect()

        records = AsyncRecords(
            _data=[{"id": 1, "name": "Alice"}],
            _database=db,
        )
        await records.insert_into(table_name)

        table_handle = await db.table(table_name)
        df = table_handle.select()
        df_with_model = df.with_model(User)
        results = await df_with_model.collect()

        assert isinstance(results[0], User)
        assert results[0].name == "Alice"

        await db.close()


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="Pydantic/SQLModel not installed")
class TestPydanticIntegration:
    """Test pure Pydantic model integration features."""

    def test_pydantic_detection(self):
        """Test that Pydantic models are detected correctly."""
        from pydantic import BaseModel
        from moltres.utils.sqlmodel_integration import is_pydantic_model, is_model_class

        class UserData(BaseModel):
            id: int
            name: str
            email: str

        assert is_pydantic_model(UserData) is True
        assert is_model_class(UserData) is True
        assert is_pydantic_model(str) is False
        assert is_pydantic_model(None) is False

    def test_collect_returns_pydantic_instances(self):
        """Test that collect() returns Pydantic instances when model is attached."""
        from pydantic import BaseModel

        class UserData(BaseModel):
            id: int
            name: str
            email: str

        db = connect("sqlite:///:memory:")
        table_name = f"users_{uuid.uuid4().hex[:8]}"
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()

        # Insert data
        Records(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            _database=db,
        ).insert_into(table_name)

        # Get DataFrame and attach Pydantic model
        df = db.table(table_name).select()
        df_with_model = df.with_model(UserData)
        results = df_with_model.collect()

        # Check that results are Pydantic instances
        assert len(results) == 2
        assert isinstance(results[0], UserData)
        assert results[0].name == "Alice"
        assert results[0].email == "alice@example.com"
        assert results[1].name == "Bob"

        db.close()

    def test_pydantic_validation(self):
        """Test that Pydantic validation works with DataFrame results."""
        from pydantic import BaseModel

        class UserData(BaseModel):
            id: int
            name: str
            age: int

        db = connect("sqlite:///:memory:")
        table_name = f"users_{uuid.uuid4().hex[:8]}"
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 30},
            ],
            _database=db,
        ).insert_into(table_name)

        df = db.table(table_name).select()
        df_with_model = df.with_model(UserData)
        results = df_with_model.collect()

        # Pydantic validation should work
        user = results[0]
        assert user.id == 1
        assert user.name == "Alice"
        assert user.age == 30

        # Pydantic models provide type safety
        assert isinstance(user.id, int)
        assert isinstance(user.name, str)
        assert isinstance(user.age, int)

        db.close()

    def test_pydantic_model_to_dict(self):
        """Test converting Pydantic instance to dict."""
        from pydantic import BaseModel
        from moltres.utils.sqlmodel_integration import model_to_dict

        class UserData(BaseModel):
            id: int
            name: str
            email: str

        user = UserData(id=1, name="Alice", email="alice@example.com")
        user_dict = model_to_dict(user)

        assert user_dict == {"id": 1, "name": "Alice", "email": "alice@example.com"}

    def test_dict_to_pydantic_model(self):
        """Test converting dict to Pydantic instance."""
        from pydantic import BaseModel
        from moltres.utils.sqlmodel_integration import dict_to_model

        class UserData(BaseModel):
            id: int
            name: str
            email: str

        user_dict = {"id": 1, "name": "Alice", "email": "alice@example.com"}
        user = dict_to_model(user_dict, UserData)

        assert isinstance(user, UserData)
        assert user.id == 1
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_rows_to_pydantic_models(self):
        """Test converting list of dicts to Pydantic instances."""
        from pydantic import BaseModel
        from moltres.utils.sqlmodel_integration import rows_to_models

        class UserData(BaseModel):
            id: int
            name: str

        rows = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        users = rows_to_models(rows, UserData)

        assert len(users) == 2
        assert all(isinstance(u, UserData) for u in users)
        assert users[0].name == "Alice"
        assert users[1].name == "Bob"

    def test_pydantic_with_chaining(self):
        """Test that Pydantic models work with chained operations."""
        from pydantic import BaseModel
        from moltres import col

        class UserData(BaseModel):
            id: int
            name: str
            age: int

        db = connect("sqlite:///:memory:")
        table_name = f"users_{uuid.uuid4().hex[:8]}"
        db.create_table(
            table_name,
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
                {"id": 3, "name": "Charlie", "age": 35},
            ],
            _database=db,
        ).insert_into(table_name)

        # Chain operations with Pydantic model attached
        df = (
            db.table(table_name)
            .select()
            .where(col("age") > 25)
            .order_by("age")
            .with_model(UserData)
        )
        results = df.collect()

        assert len(results) == 2
        assert all(isinstance(r, UserData) for r in results)
        assert results[0].name == "Alice"  # Age 30, sorted
        assert results[1].name == "Charlie"  # Age 35

        db.close()
