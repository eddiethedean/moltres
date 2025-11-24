"""Example: Connecting to databases (sync and async).

This example demonstrates how to connect to different database types
using both synchronous and asynchronous connections.
"""

# Synchronous connection
from moltres import connect

# SQLite
db = connect("sqlite:///example.db")

# PostgreSQL
# db = connect("postgresql://user:password@localhost:5432/mydb")

# MySQL
# db = connect("mysql://user:password@localhost:3306/mydb")

# Using SQLAlchemy Engine
from sqlalchemy import create_engine

engine = create_engine("sqlite:///example.db")
db = connect(engine=engine)

# Close when done
db.close()

# Async connection
try:
    from moltres import async_connect

    async def main() -> None:
        # SQLite async
        db = async_connect("sqlite+aiosqlite:///example.db")

        # PostgreSQL async
        # db = async_connect("postgresql+asyncpg://user:password@localhost:5432/mydb")

        # MySQL async
        # db = async_connect("mysql+aiomysql://user:password@localhost:3306/mydb")

        # Close when done
        await db.close()

    # asyncio.run(main())
except ImportError:
    print("Async dependencies not installed. Install with: pip install moltres[async]")
