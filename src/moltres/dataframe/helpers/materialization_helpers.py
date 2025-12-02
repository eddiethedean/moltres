"""Common helper functions for :class:`DataFrame` materialization.

This module contains shared logic used by both :class:`DataFrame` and AsyncDataFrame
for materializing FileScan nodes and converting results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

if TYPE_CHECKING:
    from ...logical.plan import LogicalPlan


def convert_rows_to_models(
    rows: List[Dict[str, object]], model: Optional[Type[Any]]
) -> Union[List[Dict[str, object]], List[Any]]:
    """Convert rows to model instances if a model is attached.

    Args:
        rows: List of row dictionaries
        model: Optional model class (SQLModel or Pydantic)

    Returns:
        List of model instances if model is provided, otherwise original rows

    Raises:
        ImportError: If Pydantic or SQLModel is required but not installed
    """
    if model is not None:
        try:
            from ...utils.sqlmodel_integration import rows_to_models

            return rows_to_models(rows, model)
        except ImportError:
            # Check if it's a Pydantic or SQLModel import error
            try:
                import pydantic  # noqa: F401
                import sqlmodel  # noqa: F401
            except ImportError:
                raise ImportError(
                    "Pydantic or SQLModel is required when a model is attached to DataFrame. "
                    "Install with: pip install pydantic or pip install sqlmodel"
                ) from None
            raise
    return rows


def convert_result_rows(result_rows: Any) -> List[Dict[str, object]]:
    """Convert query result rows to a list of dictionaries.

    Handles different result row formats (pandas :class:`DataFrame`, polars :class:`DataFrame`, etc.)

    Args:
        result_rows: Query result rows (can be various formats)

    Returns:
        List of row dictionaries with string keys
    """
    if result_rows is None:
        return []

    # Convert to list if it's a DataFrame
    if hasattr(result_rows, "to_dict"):
        records = result_rows.to_dict("records")
        # Convert Hashable keys to str keys
        return [{str(k): v for k, v in row.items()} for row in records]
    if hasattr(result_rows, "to_dicts"):
        records = list(result_rows.to_dicts())
        # Convert Hashable keys to str keys
        return [{str(k): v for k, v in row.items()} for row in records]

    # Assume it's already a list of dicts
    if isinstance(result_rows, list):
        return result_rows

    # Fallback: try to convert to list
    return list(result_rows)


def should_materialize_plan_node(plan: "LogicalPlan") -> bool:
    """Check if a plan node type requires FileScan materialization.

    Args:
        plan: Logical plan node

    Returns:
        True if this plan type can contain FileScan nodes that need materialization
    """
    from ...logical.plan import (
        Aggregate,
        AntiJoin,
        CTE,
        Distinct,
        Except,
        Explode,
        Filter,
        Intersect,
        Join,
        Limit,
        Pivot,
        Project,
        RecursiveCTE,
        Sample,
        SemiJoin,
        Sort,
        Union,
    )

    return isinstance(
        plan,
        (
            Project,
            Filter,
            Limit,
            Sample,
            Sort,
            Distinct,
            Aggregate,
            Explode,
            Pivot,
            Join,
            Union,
            Intersect,
            Except,
            SemiJoin,
            AntiJoin,
            CTE,
            RecursiveCTE,
        ),
    )


def get_plan_children(plan: "LogicalPlan") -> tuple[Optional["LogicalPlan"], ...]:
    """Get child plan nodes that may need materialization.

    Args:
        plan: Logical plan node

    Returns:
        Tuple of child plan nodes (may contain None for plans without children)
    """
    from ...logical.plan import (
        Aggregate,
        AntiJoin,
        CTE,
        Distinct,
        Except,
        Explode,
        Filter,
        Intersect,
        Join,
        Limit,
        Pivot,
        Project,
        RecursiveCTE,
        Sample,
        SemiJoin,
        Sort,
        Union,
    )

    if isinstance(
        plan, (Project, Filter, Limit, Sample, Sort, Distinct, Aggregate, Explode, Pivot)
    ):
        return (plan.child,)
    elif isinstance(plan, (Join, Union, Intersect, Except, SemiJoin, AntiJoin)):
        return (plan.left, plan.right)
    elif isinstance(plan, CTE):
        return (plan.child,)
    elif isinstance(plan, RecursiveCTE):
        return (plan.initial, plan.recursive)

    return ()
