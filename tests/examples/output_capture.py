"""Capture and format outputs from code execution."""

import io
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional


class OutputCapture:
    """Context manager to capture stdout, stderr, and return values."""

    def __init__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.captured_stdout = ""
        self.captured_stderr = ""
        self.return_value: Any = None
        self.exception: Optional[Exception] = None

    def __enter__(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.captured_stdout = self.stdout.getvalue()
        self.captured_stderr = self.stderr.getvalue()
        if exc_val:
            self.exception = exc_val
        return False  # Don't suppress exceptions

    def execute(self, code: str, globals_dict: dict, locals_dict: dict) -> str:
        """Execute code and capture output.

        Args:
            code: Python code to execute
            globals_dict: Global namespace
            locals_dict: Local namespace (will be updated)

        Returns:
            Combined output string
        """
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            try:
                exec(compile(code, "<string>", "exec"), globals_dict, locals_dict)
                self.return_value = locals_dict.get("result")
            except Exception as e:
                self.exception = e
                raise

        return self.captured_stdout

    def get_output(self) -> str:
        """Get formatted output string."""
        output_parts = []
        if self.captured_stdout:
            output_parts.append(self.captured_stdout.rstrip())
        if self.captured_stderr:
            output_parts.append(f"STDERR: {self.captured_stderr.rstrip()}")
        if self.exception:
            output_parts.append(f"ERROR: {self.exception}")
        return "\n".join(output_parts).rstrip()

    def normalize_output(self, output: str) -> str:
        """Normalize output for comparison (handle dict/list ordering, whitespace)."""
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in output.split("\n")]
        # Remove empty lines at end
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)
