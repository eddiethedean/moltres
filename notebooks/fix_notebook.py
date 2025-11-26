#!/usr/bin/env python3
"""Fix all notebook cells to use correct API syntax."""

import json
from pathlib import Path


def fix_notebook(notebook_path: str):
    """Fix all cells in the notebook."""
    nb_path = Path(notebook_path)
    with open(nb_path, "r") as f:
        nb = json.load(f)

    fixes_applied = 0

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell.get("source", []))
        original_source = source

        # Fix 1: count().collect() -> count() (count returns int directly)
        if ".count().collect()" in source:
            source = source.replace('.count().collect()[0]["count(*)"]', ".count()")
            source = source.replace(".count().collect()[0]['count(*)']", ".count()")

        # Fix 2: join() with Column expressions -> join() with on parameter
        # Pattern: .join(\n        db.table(...).select(),\n        col("left") == col("right")\n    )
        import re

        # Fix join patterns
        # Pattern 1: join with Column equality
        pattern1 = r'\.join\(\s*db\.table\([^)]+\)\.select\(\)\s*,\s*col\(["\']([^"\']+)["\']\)\s*==\s*col\(["\']([^"\']+)["\']\)\s*\)'

        def replace_join1(match):
            left_col = match.group(1)
            right_col = match.group(2)
            # Extract table names from context if possible
            return f'.join(\n        db.table("{right_col.split(".")[0] if "." in right_col else "orders"}").select(),\n        on=[("{left_col}", "{right_col}")],\n        how="inner"\n    )'

        source = re.sub(pattern1, replace_join1, source)

        # Pattern 2: More complex join with table prefixes
        # .join(\n        db.table("orders").select(),\n        col("order_items.order_id") == col("orders.order_id")\n    )
        pattern2 = r'\.join\(\s*db\.table\(["\']([^"\']+)["\']\)\.select\(\)\s*,\s*col\(["\']([^"\']+)["\']\)\s*==\s*col\(["\']([^"\']+)["\']\)\s*\)'

        def replace_join2(match):
            table = match.group(1)
            left_col = match.group(2)
            right_col = match.group(3)
            return f'.join(\n        db.table("{table}").select(),\n        on=[("{left_col}", "{right_col}")],\n        how="inner"\n    )'

        source = re.sub(pattern2, replace_join2, source)

        # Fix 3: Price formatting - handle Decimal/string types
        if "row['price']:.2f" in source or 'row["price"]:.2f' in source:
            # Replace direct formatting with float conversion
            source = source.replace(
                "f\"  {row['product_name']} - ${row['price']:.2f} (Cost: ${row['cost']:.2f})\"",
                "f\"  {row['product_name']} - ${float(row['price']):.2f} (Cost: ${float(row['cost']):.2f})\"",
            )
            source = source.replace(
                'f"  {row["product_name"]} - ${row["price"]:.2f} (Cost: ${row["cost"]:.2f})"',
                'f"  {row["product_name"]} - ${float(row["price"]):.2f} (Cost: ${float(row["cost"]):.2f})"',
            )

        if source != original_source:
            cell["source"] = source.split("\n")
            fixes_applied += 1
            print(f"Fixed cell {i + 1}")

    # Write back
    with open(nb_path, "w") as f:
        json.dump(nb, f, indent=1)

    print(f"\n✅ Applied {fixes_applied} fixes to notebook")


if __name__ == "__main__":
    fix_notebook("notebooks/ecommerce_analytics_demo.ipynb")
