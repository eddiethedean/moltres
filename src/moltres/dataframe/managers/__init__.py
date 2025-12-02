"""Manager classes for DataFrame operations (SRP refactoring)."""

from .execution import DataFrameExecutor
from .schema import SchemaInspector
from .statistics import StatisticsCalculator
from .materialization import MaterializationHandler
from .model_integration import ModelIntegrator

__all__ = [
    "DataFrameExecutor",
    "SchemaInspector",
    "StatisticsCalculator",
    "MaterializationHandler",
    "ModelIntegrator",
]
