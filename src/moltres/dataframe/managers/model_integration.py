"""DataFrame model integration operations.

This module handles model integration for DataFrames, including SQLModel and Pydantic model attachment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Type

if TYPE_CHECKING:
    from ..core.dataframe import DataFrame


class ModelIntegrator:
    """Handles model integration for DataFrames."""

    def __init__(self, df: "DataFrame"):
        """Initialize model integrator with a DataFrame.

        Args:
            df: The DataFrame to integrate models with
        """
        self._df = df

    def with_model(self, model: Type[Any]) -> "DataFrame":
        """Attach a SQLModel or Pydantic model to this DataFrame.

        When a model is attached, `collect()` will return model instances
        instead of dictionaries. This provides type safety and validation.

        Args:
            model: SQLModel or Pydantic model class to attach

        Returns:
            New DataFrame with the model attached

        Raises:
            TypeError: If model is not a SQLModel or Pydantic class
            ImportError: If required dependencies are not installed
        """
        from ...utils.sqlmodel_integration import is_model_class

        if not is_model_class(model):
            raise TypeError(f"Expected SQLModel or Pydantic class, got {type(model)}")

        return self._df._with_model(model)

    def _with_model(self, model: Optional[Type[Any]]) -> "DataFrame":
        """Internal method to attach or remove a model from this DataFrame.

        Args:
            model: SQLModel model class to attach, or None to remove model

        Returns:
            New DataFrame with the model attached
        """
        from ..core.dataframe import DataFrame

        return DataFrame(
            plan=self._df.plan,
            database=self._df.database,
            model=model,
        )
