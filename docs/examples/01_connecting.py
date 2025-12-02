"""Example: Connecting to databases (sync and async).

This example demonstrates how to connect to different database types
using both synchronous and asynchronous connections.

Required dependencies:
- moltres (required)
- For async: moltres[async] or moltres[async-sqlite] (optional)
"""

# Synchronous connection
from moltres import connect

# SQLite (using in-memory database for this example)
db = connect("sqlite:///:memory:")

# PostgreSQL
# db = connect("postgresql://user:password@localhost:5432/mydb")

# MySQL
# db = connect("mysql://user:password@localhost:3306/mydb")

# Using SQLAlchemy Engine
from sqlalchemy import create_engine

engine = create_engine("sqlite:///:memory:")
db = connect(engine=engine)

# Close when done
db.close()

# Async connection
import sys

try:
    from moltres import async_connect
    import asyncio

    async def main() -> None:
        # SQLite async (using in-memory database)
        db = async_connect("sqlite+aiosqlite:///:memory:")

        # PostgreSQL async
        # db = async_connect("postgresql+asyncpg://user:password@localhost:5432/mydb")

        # MySQL async
        # db = async_connect("mysql+aiomysql://user:password@localhost:3306/mydb")

        # Close when done
        await db.close()

    if __name__ == "__main__":
        asyncio.run(main())
except ImportError:
    print("Async dependencies not installed.")
    print("Install with: pip install moltres[async]")
    print("Or: pip install moltres[async-sqlite]")
    # Output (if async dependencies not installed):
    #   Async dependencies not installed.
    #   Install with: pip install moltres[async]
    #   Or: pip install moltres[async-sqlite]
    if __name__ == "__main__":
        sys.exit(1)
