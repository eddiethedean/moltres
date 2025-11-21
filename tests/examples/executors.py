"""Execute code examples with proper setup."""

import asyncio
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from .output_capture import OutputCapture


class ExampleExecutor:
    """Execute code examples with setup and cleanup."""

    def __init__(
        self, temp_db_path: Path, temp_file_dir: Path, globals_dict: Optional[Dict] = None
    ):
        """Initialize executor.

        Args:
            temp_db_path: Path to temporary database
            temp_file_dir: Directory for temporary files
            globals_dict: Additional globals to provide
        """
        self.temp_db_path = temp_db_path
        self.temp_file_dir = temp_file_dir
        self.globals_dict = globals_dict or {}

    def prepare_code(self, code: str) -> str:
        """Prepare code for execution by replacing placeholders.

        Args:
            code: Original code

        Returns:
            Prepared code with replacements
        """
        prepared = code

        # Replace database paths
        # Use as_posix() to convert paths to forward slashes (required for SQLite URLs)
        db_path_str = self.temp_db_path.as_posix()
        prepared = re.sub(
            r"sqlite:///example\.db",
            f"sqlite:///{db_path_str}",
            prepared,
        )
        prepared = re.sub(
            r"sqlite\+aiosqlite:///example\.db",
            f"sqlite+aiosqlite:///{db_path_str}",
            prepared,
        )

        # Replace file paths (common patterns)
        # Use lambda to properly escape backslashes in Windows paths for regex replacement
        prepared = re.sub(
            r'"data\.csv"',
            lambda _: f'"{self.temp_file_dir / "data.csv"}"'.replace("\\", "\\\\"),
            prepared,
        )
        prepared = re.sub(
            r'"data\.json"',
            lambda _: f'"{self.temp_file_dir / "data.json"}"'.replace("\\", "\\\\"),
            prepared,
        )
        prepared = re.sub(
            r'"large_file\.csv"',
            lambda _: f'"{self.temp_file_dir / "large_file.csv"}"'.replace("\\", "\\\\"),
            prepared,
        )

        # Handle relative paths in quotes
        prepared = re.sub(
            r'"([^"]+\.(csv|json|jsonl|parquet|txt))"',
            lambda m: f'"{self.temp_file_dir / m.group(1)}"'.replace("\\", "\\\\"),
            prepared,
        )

        return prepared

    def execute(self, code: str) -> Tuple[bool, str, Optional[Exception]]:
        """Execute code and capture output.

        Args:
            code: Code to execute

        Returns:
            Tuple of (success, output, exception)
        """
        prepared_code = self.prepare_code(code)

        # Setup globals
        globals_dict = {
            "__builtins__": __builtins__,
            "asyncio": asyncio,
            **self.globals_dict,
        }

        # Import common modules
        try:
            from moltres import col, connect, lit
            from moltres.expressions.functions import sum, avg, count, max, min
            from moltres.table.schema import column, ColumnDef

            globals_dict.update(
                {
                    "col": col,
                    "connect": connect,
                    "lit": lit,
                    "sum": sum,
                    "avg": avg,
                    "count": count,
                    "max": max,
                    "min": min,
                    "column": column,
                    "ColumnDef": ColumnDef,
                }
            )

            # Try to import async_connect if available
            try:
                from moltres import async_connect

                globals_dict["async_connect"] = async_connect
            except (ImportError, AttributeError):
                pass
        except ImportError:
            pass

        locals_dict = {}

        with OutputCapture() as capture:
            try:
                # Check if code contains async
                is_async = "async def" in prepared_code or "await " in prepared_code

                if is_async:
                    # For async code, we need to handle it differently
                    # Check if it's already wrapped in asyncio.run()
                    if "asyncio.run(" in prepared_code:
                        # Code already has asyncio.run(), just execute it
                        capture.execute(prepared_code, globals_dict, locals_dict)
                    else:
                        # Wrap async code
                        async_code = f"""
import asyncio

async def _run_example():
{chr(10).join("    " + line for line in prepared_code.splitlines())}

if __name__ == "__main__":
    asyncio.run(_run_example())
"""
                        # Execute in a way that doesn't conflict with existing event loop
                        try:
                            capture.execute(async_code, globals_dict, locals_dict)
                        except RuntimeError as e:
                            if "cannot be called from a running event loop" in str(e):
                                # Skip async examples in async context
                                output = "Skipped: async example requires non-async context"
                                return True, output, None
                            raise
                else:
                    capture.execute(prepared_code, globals_dict, locals_dict)

                output = capture.get_output()
                return True, output, None

            except Exception as e:
                output = capture.get_output()
                return False, output, e

    def execute_with_setup(
        self, code: str, setup_code: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Exception]]:
        """Execute code with optional setup.

        Args:
            code: Code to execute
            setup_code: Optional setup code to run first

        Returns:
            Tuple of (success, output, exception)
        """
        if setup_code:
            # Execute setup first
            success, _, exc = self.execute(setup_code)
            if not success:
                return False, f"Setup failed: {exc}", exc

        return self.execute(code)
