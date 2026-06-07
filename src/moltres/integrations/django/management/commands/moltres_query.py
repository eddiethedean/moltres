"""Django management command for executing Moltres table queries safely."""

from __future__ import annotations

import json
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Any, Dict, List, Union, cast

if TYPE_CHECKING:
    from moltres.table.table import Database

try:
    from django.core.management.base import BaseCommand, CommandError  # type: ignore[import-untyped]

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    BaseCommand = cast(type[Any], None)
    CommandError = cast(type[Any], None)

QueryResults = Union[List[Dict[str, Any]], Any]


class Command(BaseCommand):
    """Execute safe Moltres table queries from the Django command line.

    Usage:
        python manage.py moltres_query --table users
        python manage.py moltres_query --table users --where-column age --where-op gt --where-value 25
        python manage.py moltres_query "db.table('users').select()"  # legacy simple form
    """

    help = "Execute Moltres table queries from the command line (no code execution)"

    def add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "query",
            nargs="?",
            type=str,
            help="Legacy simple query: db.table('name').select() only",
        )
        parser.add_argument(
            "--table",
            type=str,
            help="Table name to query",
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
            help="Read legacy simple query from file",
        )
        parser.add_argument(
            "--where-column",
            type=str,
            help="Column name for optional WHERE filter",
        )
        parser.add_argument(
            "--where-op",
            type=str,
            choices=["eq", "ne", "gt", "gte", "lt", "lte"],
            help="Comparison operator for WHERE filter",
        )
        parser.add_argument(
            "--where-value",
            type=str,
            help="Value for WHERE filter",
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Optional row limit",
        )

    def handle(self, *args: str, **options: Any) -> None:
        if not DJANGO_AVAILABLE:
            raise CommandError(
                "Django is required for this command. Install with: pip install django"
            )

        database = options["database"]
        interactive = options["interactive"]
        query_str = options.get("query")
        file_path = options.get("file")
        output_format = options["format"]
        table_name = options.get("table")

        from django.conf import settings  # type: ignore[import-untyped]

        if database not in settings.DATABASES:
            raise CommandError(
                f"Database alias '{database}' is not configured in Django settings.DATABASES"
            )

        try:
            from moltres.integrations.django import get_moltres_db

            db = get_moltres_db(using=database)
        except ImportError as e:
            raise CommandError(f"Failed to import Moltres Django integration: {e}") from e
        except Exception as e:
            raise CommandError(f"Failed to create Moltres database connection: {e}") from e

        if interactive:
            self._interactive_mode(db, output_format)
            return

        if file_path:
            try:
                with open(file_path, "r") as f:
                    query_str = f.read().strip()
            except FileNotFoundError:
                raise CommandError(f"Query file not found: {file_path}")
            except Exception as e:
                raise CommandError(f"Failed to read query file: {e}") from e

        if not table_name and not query_str:
            raise CommandError(
                "Provide --table, a legacy query string db.table('name').select(), "
                "use --file, or use --interactive mode."
            )

        try:
            results = self._execute_query(
                db,
                table_name=table_name,
                query_str=query_str,
                where_column=options.get("where_column"),
                where_op=options.get("where_op"),
                where_value=options.get("where_value"),
                limit=options.get("limit"),
            )
            self._print_results(results, output_format)
        except Exception as e:
            raise CommandError(f"Query execution failed: {e}") from e

    def _execute_query(
        self,
        db: "Database",
        *,
        table_name: str | None,
        query_str: str | None,
        where_column: str | None,
        where_op: str | None,
        where_value: str | None,
        limit: int | None,
    ) -> QueryResults:
        from moltres.integrations.django.safe_query import (
            execute_safe_query_string,
            execute_table_query,
        )

        if table_name:
            return execute_table_query(
                db,
                table_name,
                where_column=where_column,
                where_op=where_op,
                where_value=where_value,
                limit=limit,
            )
        if query_str:
            if where_column or where_op or where_value or limit is not None:
                raise CommandError(
                    "Legacy query strings cannot be combined with --where-* or --limit. "
                    "Use --table instead."
                )
            return execute_safe_query_string(db, query_str)
        raise CommandError("No query specified")

    def _print_results(self, results: QueryResults, output_format: str) -> None:
        if output_format == "json":
            self.stdout.write(json.dumps(results, indent=2, default=str))
        elif output_format == "csv":
            if not results:
                self.stdout.write("")
                return
            if isinstance(results, list) and results and isinstance(results[0], dict):
                headers = list(results[0].keys())
                self.stdout.write(",".join(headers))
                for row in results:
                    values = [str(row.get(h, "")) for h in headers]
                    self.stdout.write(",".join(values))
            else:
                self.stdout.write(str(results))
        else:
            if not results:
                self.stdout.write("No results")
                return
            if isinstance(results, list) and results and isinstance(results[0], dict):
                self._print_table(results)
            else:
                self.stdout.write(str(results))

    def _print_table(self, results: list[dict]) -> None:
        if not results:
            self.stdout.write("No results")
            return
        headers = list(results[0].keys())
        widths = {h: len(str(h)) for h in headers}
        for row in results:
            for header in headers:
                value = str(row.get(header, ""))
                widths[header] = max(widths[header], len(value))
        header_row = " | ".join(h.ljust(widths[h]) for h in headers)
        self.stdout.write(header_row)
        self.stdout.write("-" * len(header_row))
        for row in results:
            row_str = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
            self.stdout.write(row_str)

    def _interactive_mode(self, db: "Database", output_format: str) -> None:
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
                    self.stdout.write("Commands: exit, quit, help")
                    self.stdout.write("Examples:")
                    self.stdout.write("  --table users  (use: table users)")
                    self.stdout.write("  table users")
                    self.stdout.write('  db.table("users").select()')
                    self.stdout.write("")
                    continue

                try:
                    if query_str.lower().startswith("table "):
                        table = query_str.split(None, 1)[1].strip().strip("'\"")
                        results = self._execute_query(
                            db,
                            table_name=table,
                            query_str=None,
                            where_column=None,
                            where_op=None,
                            where_value=None,
                            limit=None,
                        )
                    else:
                        results = self._execute_query(
                            db,
                            table_name=None,
                            query_str=query_str,
                            where_column=None,
                            where_op=None,
                            where_value=None,
                            limit=None,
                        )
                    self._print_results(results, output_format)
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error: {e}"))
                self.stdout.write("")
            except (EOFError, KeyboardInterrupt):
                self.stdout.write("")
                break

        self.stdout.write(self.style.SUCCESS("Exiting interactive mode"))
