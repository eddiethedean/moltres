"""Django management command for executing Moltres :class:`DataFrame` operations.

This command allows executing Moltres queries from the Django command line,
useful for data exploration, debugging, and one-off operations.
"""

from __future__ import annotations

import ast
import json
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any, Dict, List, Union, cast

if TYPE_CHECKING:
    from moltres.table.table import Database

try:
    from django.core.management.base import BaseCommand, CommandError  # type: ignore[import-untyped]
    from django.db import connections  # type: ignore[import-untyped]

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    BaseCommand = cast(type[Any], None)
    CommandError = cast(type[Any], None)
    connections = None

# Type alias for query results
QueryResults = Union[List[Dict[str, Any]], Any]


class Command(BaseCommand):
    """Django management command for executing Moltres queries.

    Usage:
        python manage.py moltres_query "db.table('users').select()"
        python manage.py moltres_query "db.table('users').select()" --database=other
        python manage.py moltres_query --interactive
    """

    help = "Execute Moltres DataFrame operations from the command line"

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add command arguments."""
        parser.add_argument(
            "query",
            nargs="?",
            type=str,
            help="Moltres query expression (e.g., \"db.table('users').select()\")",
        )
        parser.add_argument(
            "--database",
            type=str,
            default="default",
            help="Django database alias to use (default: 'default')",
        )
        parser.add_argument(
            "--interactive",
            action="store_true",
            help="Start interactive query mode",
        )
        parser.add_argument(
            "--format",
            type=str,
            choices=["json", "table", "csv"],
            default="table",
            help="Output format (default: table)",
        )
        parser.add_argument(
            "--file",
            type=str,
            help="Read query from file instead of command line",
        )

    def handle(self, *args: str, **options: Any) -> None:
        """Execute the command."""
        if not DJANGO_AVAILABLE:
            raise CommandError(
                "Django is required for this command. Install with: pip install django"
            )

        database = options["database"]
        interactive = options["interactive"]
        query_str = options.get("query")
        file_path = options.get("file")
        output_format = options["format"]

        # Validate database alias
        from django.conf import settings  # type: ignore[import-untyped]

        if database not in settings.DATABASES:
            raise CommandError(
                f"Database alias '{database}' is not configured in Django settings.DATABASES"
            )

        # Get Moltres database connection
        try:
            from moltres.integrations.django import get_moltres_db

            db = get_moltres_db(using=database)
        except ImportError as e:
            raise CommandError(f"Failed to import Moltres Django integration: {e}") from e
        except Exception as e:
            raise CommandError(f"Failed to create Moltres database connection: {e}") from e

        # Interactive mode
        if interactive:
            self._interactive_mode(db, output_format)
            return

        # Read query from file or command line
        if file_path:
            try:
                with open(file_path, "r") as f:
                    query_str = f.read().strip()
            except FileNotFoundError:
                raise CommandError(f"Query file not found: {file_path}")
            except Exception as e:
                raise CommandError(f"Failed to read query file: {e}") from e

        if not query_str:
            raise CommandError(
                "Query is required. Provide a query string, use --file, or use --interactive mode."
            )

        # Execute query
        try:
            results = self._execute_query(db, query_str)
            self._print_results(results, output_format)
        except Exception as e:
            raise CommandError(f"Query execution failed: {e}") from e

    def _execute_query(self, db: Database, query_str: str) -> QueryResults:
        """Execute a Moltres query string safely using AST parsing.

        Args:
            db: Moltres :class:`Database` instance
            query_str: Query string to execute

        Returns:
            Query results

        Raises:
            CommandError: If the query contains unsafe operations or fails to execute
        """
        # Import Moltres components
        from moltres import col
        from moltres.expressions import functions as F

        # Create namespace for query execution (restricted builtins)
        safe_builtins = {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "float": float,
            "int": int,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        }

        namespace = {
            "db": db,
            "col": col,
            "F": F,
            "__builtins__": safe_builtins,
        }

        # Parse and validate AST
        try:
            tree = ast.parse(query_str, mode="eval")
        except SyntaxError as e:
            # Check if it's an import statement or other statement-level syntax
            # that can't be parsed in eval mode
            error_str = str(e).lower()
            if "import" in error_str or (
                "invalid syntax" in error_str and "import" in query_str.lower()
            ):
                raise CommandError(
                    "Unsafe operation detected: Import statements are not allowed. "
                    "Only query expressions are allowed."
                ) from e
            raise CommandError(f"Invalid query syntax: {e}") from e

        # Validate AST for safety
        self._validate_ast(tree)

        # Execute query safely
        try:
            code = compile(tree, "<string>", "eval")
            result = eval(code, namespace)  # noqa: S307 - Safe after AST validation

            # If result is a DataFrame, collect it
            if hasattr(result, "collect"):
                return result.collect()
            return result
        except Exception as e:
            raise CommandError(f"Query evaluation failed: {e}") from e

    def _validate_ast(self, node: ast.AST) -> None:
        """Validate AST to ensure only safe operations are allowed.

        Args:
            node: AST node to validate

        Raises:
            CommandError: If unsafe operations are detected
        """
        # Forbidden node types
        forbidden_nodes = (
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.Delete,
            ast.AugAssign,
            ast.AnnAssign,
            ast.With,
            ast.AsyncWith,
            ast.Raise,
            ast.Try,
            ast.Assert,
            ast.For,
            ast.AsyncFor,
            ast.While,
            ast.If,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
            ast.Return,
            ast.Yield,
            ast.YieldFrom,
            ast.Await,
        )

        for child in ast.walk(node):
            # Check for forbidden node types
            if isinstance(child, forbidden_nodes):
                raise CommandError(
                    f"Unsafe operation detected: {type(child).__name__}. "
                    "Only query expressions are allowed."
                )

            # Check for list comprehensions and generator expressions (they can contain loops)
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                raise CommandError(
                    f"Unsafe operation detected: {type(child).__name__}. "
                    "Comprehensions are not allowed for security reasons. "
                    "Only query expressions are allowed."
                )

            # Check for dangerous function calls
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                    # Block dangerous builtins
                    dangerous_builtins = {
                        "open",
                        "exec",
                        "eval",
                        "__import__",
                        "compile",
                        "reload",
                        "input",
                        "raw_input",
                        "file",
                        "exit",
                        "quit",
                        "help",
                    }
                    if func_name in dangerous_builtins:
                        raise CommandError(
                            f"Dangerous function call detected: {func_name}. "
                            "This operation is not allowed for security reasons."
                        )

    def _print_results(self, results: QueryResults, output_format: str) -> None:
        """Print query results in the specified format.

        Args:
            results: Query results
            output_format: Output format (json, table, csv)
        """
        if output_format == "json":
            self.stdout.write(json.dumps(results, indent=2, default=str))
        elif output_format == "csv":
            if not results:
                self.stdout.write("")
                return
            if isinstance(results, list) and results and isinstance(results[0], dict):
                # Write CSV header
                headers = list(results[0].keys())
                self.stdout.write(",".join(headers))
                # Write rows
                for row in results:
                    values = [str(row.get(h, "")) for h in headers]
                    self.stdout.write(",".join(values))
            else:
                self.stdout.write(str(results))
        else:  # table format
            if not results:
                self.stdout.write("No results")
                return
            if isinstance(results, list) and results and isinstance(results[0], dict):
                # Pretty print as table
                self._print_table(results)
            else:
                self.stdout.write(str(results))

    def _print_table(self, results: list[dict]) -> None:
        """Print results as a formatted table.

        Args:
            results: List of dictionaries to print
        """
        if not results:
            self.stdout.write("No results")
            return

        # Get column widths
        headers = list(results[0].keys())
        widths = {h: len(str(h)) for h in headers}

        for row in results:
            for header in headers:
                value = str(row.get(header, ""))
                widths[header] = max(widths[header], len(value))

        # Print header
        header_row = " | ".join(h.ljust(widths[h]) for h in headers)
        self.stdout.write(header_row)
        self.stdout.write("-" * len(header_row))

        # Print rows
        for row in results:
            row_str = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
            self.stdout.write(row_str)

    def _interactive_mode(self, db: Database, output_format: str) -> None:
        """Start interactive query mode.

        Args:
            db: Moltres :class:`Database` instance
            output_format: Output format for results
        """
        self.stdout.write(self.style.SUCCESS("Moltres Interactive Query Mode"))
        self.stdout.write("Type 'exit' or 'quit' to exit")
        self.stdout.write("Type 'help' for help")
        self.stdout.write("")

        while True:
            try:
                query_str = input("moltres> ").strip()
                if not query_str:
                    continue

                if query_str.lower() in ("exit", "quit", "q"):
                    break

                if query_str.lower() == "help":
                    self.stdout.write("Available commands:")
                    self.stdout.write("  exit, quit, q - Exit interactive mode")
                    self.stdout.write("  help - Show this help message")
                    self.stdout.write("")
                    self.stdout.write("Example queries:")
                    self.stdout.write('  db.table("users").select()')
                    self.stdout.write('  db.table("users").select().where(col("age") > 25)')
                    self.stdout.write("")
                    continue

                # Execute query
                try:
                    results = self._execute_query(db, query_str)
                    self._print_results(results, output_format)
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error: {e}"))

                self.stdout.write("")
            except (EOFError, KeyboardInterrupt):
                self.stdout.write("")
                break

        self.stdout.write(self.style.SUCCESS("Exiting interactive mode"))
