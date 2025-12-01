"""Example: Async DataFrame operations.

This example demonstrates asynchronous DataFrame operations.

Required dependencies:
- moltres (required)
- For async: moltres[async] or moltres[async-sqlite] (optional)
"""

import sys

try:
    from moltres import async_connect, col
    import asyncio

    async def main() -> None:
        # Use in-memory database
        db = async_connect("sqlite+aiosqlite:///:memory:")

        # Create a table
        from moltres.table.schema import column

        await db.create_table(
            "products",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("price", "REAL"),
                column("category", "TEXT"),
            ],
        ).collect()

        # Insert data
        products_data = [
            {"id": 1, "name": "Laptop", "price": 999.99, "category": "Electronics"},
            {"id": 2, "name": "Mouse", "price": 29.99, "category": "Electronics"},
            {"id": 3, "name": "Desk", "price": 199.99, "category": "Furniture"},
        ]

        from moltres.io.records import AsyncRecords

        records = AsyncRecords(_data=products_data, _database=db)
        await records.insert_into("products")

        # Async DataFrame operations
        df = (await db.table("products")).select()

        # Filter
        expensive = df.where(col("price") > 100)
        results = await expensive.collect()
        print(f"Expensive products: {results}")
        # Expected output: Expensive products: [{'id': 1, 'name': 'Laptop', 'price': 999.99, 'category': 'Electronics'}, {'id': 3, 'name': 'Desk', 'price': 199.99, 'category': 'Furniture'}]

        # Select with expressions
        from moltres.expressions import functions as F

        total = df.select(F.sum(col("price")).alias("total_price"))
        results = await total.collect()
        print(f"Total price: {results}")
        # Expected output: Total price: [{'total_price': 1229.97}]

        # Streaming results
        async for chunk in await df.collect(stream=True):
            print(f"Chunk: {chunk}")
            # Output: Chunk: [{'id': 1, 'name': 'Laptop', 'price': 999.99, 'category': 'Electronics'}, ...]

        await db.close()

    if __name__ == "__main__":
        asyncio.run(main())
except ImportError:
    print("Async dependencies not installed.")
    print("Install with: pip install moltres[async]")
    print("Or: pip install moltres[async-sqlite]")
    if __name__ == "__main__":
        sys.exit(1)
