#!/usr/bin/env python3
"""Fix all query issues in the notebook."""

import json
import re


def fix_notebook():
    with open("notebooks/ecommerce_analytics_demo.ipynb", "r") as f:
        nb = json.load(f)

    fixes = 0

    # Fix pattern: Filter orders BEFORE join, not after
    # Pattern: .join(orders_df, ...).where(col("orders.status") == "completed")
    # Should be: .join(orders_df.where(col("status") == "completed"), ...)

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell.get("source", []))
        original_source = source

        # Fix 1: Move WHERE clauses on orders.status to filter BEFORE join
        # Pattern: .join(orders_df, on=[...]).where(col("orders.status") == "completed")
        pattern1 = r'\.join\(orders_df(?:\.select\(\))?,\s*on=\[\([^)]+\)\],\s*how="inner"\)\s*\.where\(col\("orders\.status"\)\s*==\s*"completed"\)'

        def replacement1(m):
            return (
                m.group(0)
                .replace(
                    ".join(orders_df.select(), on=",
                    '.join(orders_df.select().where(col("status") == "completed"), on=',
                )
                .replace(
                    ".join(orders_df, on=",
                    '.join(orders_df.where(col("status") == "completed"), on=',
                )
                .replace('.where(col("orders.status") == "completed")', "")
            )

        source = re.sub(pattern1, replacement1, source)

        # Fix 2: Similar pattern but with different join order
        # Pattern: orders_df.join(...).where(col("orders.status") == "completed")
        pattern2 = r'orders_df(?:\.select\(\))?\s*\.join\([^)]+\)\s*\.where\(col\("orders\.status"\)\s*==\s*"completed"\)'

        def replacement2(m):
            match = m.group(0)
            if "orders_df.select()" in match:
                return match.replace(
                    "orders_df.select()", 'orders_df.select().where(col("status") == "completed")'
                ).replace('.where(col("orders.status") == "completed")', "")
            else:
                return match.replace(
                    "orders_df", 'orders_df.where(col("status") == "completed")'
                ).replace('.where(col("orders.status") == "completed")', "")

        source = re.sub(pattern2, replacement2, source)

        # Fix 3: Fix date_trunc - ensure it uses col() properly
        # Pattern: F.date_trunc("month", col("orders.order_date"))
        # After join, should be: F.date_trunc("month", col("order_date"))
        # But actually, we need to select the column first or reference it correctly
        # Let's fix the monthly revenue query specifically

        # Fix 4: Fix column references after joins - remove table prefixes
        # Pattern: col("orders.status") after join -> col("status")
        # But only if it's in a WHERE clause after a join
        if ".join(" in source and 'col("orders.' in source:
            # This is more complex - need to be careful
            # For now, let's handle specific cases

            # Fix date_trunc in monthly revenue
            if 'F.date_trunc("month", col("orders.order_date"))' in source:
                source = source.replace(
                    'F.date_trunc("month", col("orders.order_date"))',
                    'F.date_trunc("month", col("order_date"))',
                )

            # Fix other column references after joins
            # But be careful - only fix in WHERE clauses after joins
            if '.where(col("orders.' in source:
                # Replace col("orders.X") with col("X") in WHERE clauses after joins
                source = re.sub(r'\.where\(col\("orders\.([^"]+)"\)', r'.where(col("\1")', source)

        # Fix 5: Fix column references in SELECT after joins
        # Pattern: col("orders.order_date") -> col("order_date") after join
        if ".join(" in source:
            # Fix in select clauses
            source = re.sub(r'col\("orders\.([^"]+)"\)', r'col("\1")', source)
            source = re.sub(r'col\("order_items\.([^"]+)"\)', r'col("\1")', source)
            source = re.sub(r'col\("products\.([^"]+)"\)', r'col("\1")', source)
            source = re.sub(r'col\("customers\.([^"]+)"\)', r'col("\1")', source)

        if source != original_source:
            # Split back into lines
            nb["cells"][i]["source"] = source.split("\n")
            fixes += 1
            print(f"Fixed cell {i + 1}")

    if fixes > 0:
        with open("notebooks/ecommerce_analytics_demo.ipynb", "w") as f:
            json.dump(nb, f, indent=1)
        print(f"\n✅ Fixed {fixes} cell(s)")
    else:
        print("No fixes needed")


if __name__ == "__main__":
    fix_notebook()
