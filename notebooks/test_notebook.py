#!/usr/bin/env python3
"""Execute all cells in the ecommerce_analytics_demo notebook to verify they work."""

import json
import sys
from pathlib import Path

# Add parent directory to path to import moltres
sys.path.insert(0, str(Path(__file__).parent.parent))

# Disable matplotlib interactive backend to prevent hangs
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend


def execute_notebook_cells(notebook_path: str) -> bool:
    """Execute all code cells in a notebook and report results."""
    notebook_path = Path(notebook_path)
    if not notebook_path.exists():
        print(f"❌ Notebook not found: {notebook_path}")
        return False

    print(f"📓 Loading notebook: {notebook_path}")
    with open(notebook_path, "r") as f:
        nb = json.load(f)

    print(f"📊 Found {len(nb['cells'])} cells")

    # Extract and execute code cells
    code_cells = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if source.strip():  # Skip empty cells
                code_cells.append((i + 1, source))

    print(f"🔢 Found {len(code_cells)} non-empty code cells to execute\n")

    # Execute cells one by one
    success_count = 0
    error_count = 0

    # Create a namespace for execution
    exec_globals = {}
    exec_globals["__name__"] = "__main__"
    exec_globals["__builtins__"] = __builtins__

    # Suppress matplotlib show() calls that might hang
    exec_globals["plt"] = None  # Will be set if matplotlib is imported

    for cell_num, code in code_cells:
        print(f"{'=' * 70}")
        print(f"Executing cell {cell_num}...")
        print(f"{'=' * 70}")

        try:
            # Skip cells that might hang (visualization cells)
            # These will be handled separately or skipped in test mode
            if "plt.show()" in code or "plt.savefig" in code:
                # Mock plt.show() to prevent hangs
                if "import matplotlib" in code or "import plt" in code or "from matplotlib" in code:
                    # Replace plt.show() with a no-op
                    code = code.replace("plt.show()", "pass  # plt.show() disabled in test mode")
                    code = code.replace("plt.savefig", "pass  # plt.savefig disabled in test mode")

            # Execute the cell with proper error handling
            # Use exec with the globals dict to maintain state between cells
            compiled = compile(code, f"<cell {cell_num}>", "exec")
            exec(compiled, exec_globals)

            # After execution, if matplotlib was imported, mock plt.show()
            if "plt" in exec_globals and exec_globals["plt"] is not None:
                exec_globals["plt"].show = lambda: None  # Disable show() to prevent hangs

            print(f"✅ Cell {cell_num} executed successfully\n")
            success_count += 1
        except KeyboardInterrupt:
            print(f"\n⚠️  Execution interrupted at cell {cell_num}")
            break
        except Exception as e:
            print(f"❌ Cell {cell_num} failed with error:")
            print(f"   {type(e).__name__}: {e}\n")

            # Print full traceback for debugging
            import traceback

            tb_lines = traceback.format_exc().split("\n")
            # Show relevant parts of traceback (skip the exec line itself)
            relevant_tb = [line for line in tb_lines if "exec(" not in line and line.strip()][:10]
            if relevant_tb:
                print("   Traceback (most relevant lines):")
                for line in relevant_tb:
                    if line.strip():
                        print(f"   {line}")
                print()

            error_count += 1
            # Print a snippet of the code that failed
            lines = code.split("\n")
            print("   Code snippet (first 5 lines):")
            for line in lines[:5]:
                if line.strip():
                    print(f"   {line}")
            print()

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {error_count}")
    print(f"📝 Total: {len(code_cells)}")

    if error_count == 0:
        print("\n🎉 All cells executed successfully!")
        return True
    else:
        print(f"\n⚠️  {error_count} cell(s) failed. Please review the errors above.")
        return False


if __name__ == "__main__":
    notebook_path = Path(__file__).parent / "ecommerce_analytics_demo.ipynb"
    success = execute_notebook_cells(notebook_path)
    sys.exit(0 if success else 1)
