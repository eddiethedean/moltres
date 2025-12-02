"""Database query execution manager.

This module handles query execution operations including plan execution, SQL execution, and explain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type

if TYPE_CHECKING:
    from ..engine.execution import QueryResult
    from ..logical.plan import LogicalPlan
    from .table import Database


class DatabaseQueryExecutor:
    """Handles query execution operations for Database."""

    def __init__(self, database: "Database"):
        """Initialize query executor with a Database.

        Args:
            database: The Database instance to execute queries on
        """
        self._db = database

    def execute_plan(self, plan: "LogicalPlan", model: Optional[Type[Any]] = None) -> "QueryResult":
        """Execute a logical plan and return results.

        Args:
            plan: Logical plan to execute
            model: Optional SQLModel or Pydantic model class for result conversion

        Returns:
            QueryResult containing the execution results
        """
        stmt = self._db.compile_plan(plan)
        return self._db._executor.fetch(stmt, model=model)

    def execute_plan_stream(self, plan: "LogicalPlan") -> Iterator[List[Dict[str, object]]]:
        """Execute a plan and return an iterator of row chunks.

        Args:
            plan: Logical plan to execute

        Returns:
            Iterator of row chunks (each chunk is a list of dictionaries)
        """
        stmt = self._db.compile_plan(plan)
        return self._db._executor.fetch_stream(stmt)

    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> "QueryResult":
        """Execute a raw SQL query.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            QueryResult containing the execution results
        """
        return self._db._executor.fetch(sql, params=params)

    def explain(self, sql: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Get the execution plan for a SQL query.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            Execution plan as a string (dialect-specific)
        """
        dialect_name = self._db._dialect.name
        if dialect_name == "postgresql" or dialect_name == "duckdb":
            explain_sql = f"EXPLAIN ANALYZE {sql}"
        elif dialect_name == "mysql":
            explain_sql = f"EXPLAIN {sql}"
        elif dialect_name == "sqlite":
            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        else:
            explain_sql = f"EXPLAIN {sql}"

        result = self._db._executor.fetch(explain_sql, params=params)
        if result.rows is None:
            return ""
        # Handle different row types
        rows: List[Any] = []
        if isinstance(result.rows, list):
            rows = result.rows
        elif hasattr(result.rows, "to_dict"):
            # pandas DataFrame
            rows = result.rows.to_dict("records")  # type: ignore[call-overload]
        elif hasattr(result.rows, "to_dicts"):
            # polars DataFrame
            rows = list(result.rows.to_dicts())  # type: ignore[operator]
        # Format rows - handle dict or other types
        if not rows:
            return ""
        # Format as string
        lines = []
        for row in rows:
            if isinstance(row, dict):
                # Format dict row
                formatted = " | ".join(f"{k}: {v}" for k, v in row.items())
                lines.append(formatted)
            else:
                lines.append(str(row))
        return "\n".join(lines)
