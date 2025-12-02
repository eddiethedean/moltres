"""Data quality check framework for Moltres.

This module provides a reusable data quality check framework that can be used
in Airflow operators, Prefect tasks, and standalone data validation workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

if TYPE_CHECKING:
    from ..dataframe.core.async_dataframe import AsyncDataFrame
    from ..dataframe.core.dataframe import DataFrame

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single data quality check."""

    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"CheckResult({self.check_name}, {status}, {self.message})"


@dataclass
class QualityReport:
    """Report containing results of multiple data quality checks."""

    checks: List[Dict[str, Any]]
    overall_status: str  # "passed" or "failed"
    results: List[CheckResult] = field(default_factory=list)
    total_rows: Optional[int] = None
    execution_time_seconds: Optional[float] = None

    @property
    def passed(self) -> bool:
        """Check if all checks passed."""
        return self.overall_status == "passed"

    @property
    def failed_checks(self) -> List[CheckResult]:
        """Get list of failed checks."""
        return [r for r in self.results if not r.passed]

    @property
    def passed_checks(self) -> List[CheckResult]:
        """Get list of passed checks."""
        return [r for r in self.results if r.passed]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "overall_status": self.overall_status,
            "total_checks": len(self.results),
            "passed_checks": len(self.passed_checks),
            "failed_checks": len(self.failed_checks),
            "total_rows": self.total_rows,
            "execution_time_seconds": self.execution_time_seconds,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def __repr__(self) -> str:
        return (
            f"QualityReport({self.overall_status}, "
            f"{len(self.passed_checks)}/{len(self.results)} passed)"
        )


class DataQualityCheck:
    """Factory for creating data quality check configurations."""

    @staticmethod
    def column_not_null(column_name: str, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a check to ensure a column has no null values.

        Args:
            column_name: Name of the column to check
            check_name: Optional name for this check (defaults to f"not_null_{column_name}")

        Returns:
            Dictionary configuration for the check
        """
        return {
            "type": "not_null",
            "column": column_name,
            "name": check_name or f"not_null_{column_name}",
        }

    @staticmethod
    def column_range(
        column_name: str,
        min: Optional[Union[int, float]] = None,  # noqa: A002
        max: Optional[Union[int, float]] = None,  # noqa: A002
        check_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a check to ensure column values are within a range.

        Args:
            column_name: Name of the column to check
            min: Minimum allowed value (inclusive)
            max: Maximum allowed value (inclusive)
            check_name: Optional name for this check

        Returns:
            Dictionary configuration for the check
        """
        name = check_name or f"range_{column_name}"
        if min is not None and max is not None:
            name = f"{name}_{min}_{max}"
        elif min is not None:
            name = f"{name}_min_{min}"
        elif max is not None:
            name = f"{name}_max_{max}"

        return {
            "type": "range",
            "column": column_name,
            "min": min,
            "max": max,
            "name": name,
        }

    @staticmethod
    def column_unique(column_name: str, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Create a check to ensure column values are unique.

        Args:
            column_name: Name of the column to check
            check_name: Optional name for this check

        Returns:
            Dictionary configuration for the check
        """
        return {
            "type": "unique",
            "column": column_name,
            "name": check_name or f"unique_{column_name}",
        }

    @staticmethod
    def column_type(
        column_name: str, expected_type: type, check_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a check to ensure column has expected type.

        Args:
            column_name: Name of the column to check
            expected_type: Expected Python type (e.g., int, float, str, bool)
            check_name: Optional name for this check

        Returns:
            Dictionary configuration for the check
        """
        return {
            "type": "column_type",
            "column": column_name,
            "expected_type": expected_type.__name__,
            "name": check_name or f"type_{column_name}_{expected_type.__name__}",
        }

    @staticmethod
    def row_count(
        min: Optional[int] = None,  # noqa: A002
        max: Optional[int] = None,  # noqa: A002
        check_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a check to ensure row count is within a range.

        Args:
            min: Minimum number of rows (inclusive)
            max: Maximum number of rows (inclusive)
            check_name: Optional name for this check

        Returns:
            Dictionary configuration for the check
        """
        name = check_name or "row_count"
        if min is not None and max is not None:
            name = f"{name}_{min}_{max}"
        elif min is not None:
            name = f"{name}_min_{min}"
        elif max is not None:
            name = f"{name}_max_{max}"

        return {
            "type": "row_count",
            "min": min,
            "max": max,
            "name": name,
        }

    @staticmethod
    def column_completeness(
        column_name: str, threshold: float, check_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a check to ensure column completeness meets threshold.

        Args:
            column_name: Name of the column to check
            threshold: Minimum completeness percentage (0.0 to 1.0)
            check_name: Optional name for this check

        Returns:
            Dictionary configuration for the check
        """
        return {
            "type": "completeness",
            "column": column_name,
            "threshold": threshold,
            "name": check_name or f"completeness_{column_name}_{threshold}",
        }

    @staticmethod
    def custom(
        check_function: Callable[[List[Dict[str, Any]]], bool],
        check_name: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a custom check function.

        Args:
            check_function: Function that takes a list of dicts (rows) and returns bool
            check_name: Optional name for this check
            error_message: Optional custom error message if check fails

        Returns:
            Dictionary configuration for the check
        """
        return {
            "type": "custom",
            "function": check_function,
            "name": check_name or "custom_check",
            "error_message": error_message or "Custom check failed",
        }


class QualityChecker:
    """Execute data quality checks on DataFrames."""

    def __init__(self, fail_fast: bool = False):
        """Initialize quality checker.

        Args:
            fail_fast: If True, stop checking after first failure
        """
        self.fail_fast = fail_fast

    async def check_async(
        self, df: "AsyncDataFrame", checks: Sequence[Dict[str, Any]]
    ) -> QualityReport:
        """Execute quality checks on an async :class:`DataFrame`.

        Args:
            df: AsyncDataFrame to check
            checks: List of check configurations

        Returns:
            QualityReport with check results
        """
        import time

        start_time = time.time()

        # Collect DataFrame
        data = await df.collect()
        total_rows = len(data)

        # Execute checks
        results = []
        for check_config in checks:
            result = await self._execute_check_async(check_config, data)
            results.append(result)

            if self.fail_fast and not result.passed:
                break

        execution_time = time.time() - start_time

        # Determine overall status
        overall_status = "passed" if all(r.passed for r in results) else "failed"

        return QualityReport(
            checks=list(checks),
            overall_status=overall_status,
            results=results,
            total_rows=total_rows,
            execution_time_seconds=execution_time,
        )

    def check(self, df: "DataFrame", checks: Sequence[Dict[str, Any]]) -> QualityReport:
        """Execute quality checks on a sync :class:`DataFrame`.

        Args:
            df: :class:`DataFrame` to check
            checks: List of check configurations

        Returns:
            QualityReport with check results
        """
        import time

        start_time = time.time()

        # Collect DataFrame
        data = df.collect()
        total_rows = len(data)

        # Execute checks
        results = []
        for check_config in checks:
            result = self._execute_check(check_config, data)
            results.append(result)

            if self.fail_fast and not result.passed:
                break

        execution_time = time.time() - start_time

        # Determine overall status
        overall_status = "passed" if all(r.passed for r in results) else "failed"

        return QualityReport(
            checks=list(checks),
            overall_status=overall_status,
            results=results,
            total_rows=total_rows,
            execution_time_seconds=execution_time,
        )

    async def _execute_check_async(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> CheckResult:
        """Execute a single check on collected data (async wrapper)."""
        return self._execute_check(check_config, data)

    def _execute_check(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]]
    ) -> CheckResult:
        """Execute a single check on collected data."""
        check_type = check_config.get("type")
        check_name = check_config.get("name", "unknown_check")

        try:
            if check_type == "not_null":
                return self._check_not_null(check_config, data, check_name)
            elif check_type == "range":
                return self._check_range(check_config, data, check_name)
            elif check_type == "unique":
                return self._check_unique(check_config, data, check_name)
            elif check_type == "column_type":
                return self._check_column_type(check_config, data, check_name)
            elif check_type == "row_count":
                return self._check_row_count(check_config, data, check_name)
            elif check_type == "completeness":
                return self._check_completeness(check_config, data, check_name)
            elif check_type == "custom":
                return self._check_custom(check_config, data, check_name)
            else:
                return CheckResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Unknown check type: {check_type}",
                    details={"check_type": check_type},
                )
        except Exception as e:
            logger.exception(f"Error executing check {check_name}")
            return CheckResult(
                check_name=check_name,
                passed=False,
                message=f"Error executing check: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
            )

    def _check_not_null(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Check that column has no null values."""
        column_name = check_config["column"]
        null_count = sum(1 for row in data if row.get(column_name) is None)
        passed = null_count == 0

        message = (
            f"Column '{column_name}' has no null values"
            if passed
            else f"Column '{column_name}' has {null_count} null value(s)"
        )

        return CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details={"column": column_name, "null_count": null_count, "total_rows": len(data)},
        )

    def _check_range(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Check that column values are within range."""
        column_name = check_config["column"]
        min_val = check_config.get("min")
        max_val = check_config.get("max")

        violations = []
        for i, row in enumerate(data):
            value = row.get(column_name)
            if value is None:
                continue  # Skip nulls in range checks
            if min_val is not None and value < min_val:
                violations.append((i, value, "below_min"))
            elif max_val is not None and value > max_val:
                violations.append((i, value, "above_max"))

        passed = len(violations) == 0

        if passed:
            message = f"Column '{column_name}' values are within range [{min_val}, {max_val}]"
        else:
            message = f"Column '{column_name}' has {len(violations)} value(s) outside range"
            if len(violations) <= 5:
                message += f": {violations[:5]}"

        return CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details={
                "column": column_name,
                "min": min_val,
                "max": max_val,
                "violations": len(violations),
                "violation_details": violations[:10],  # Limit to first 10
            },
        )

    def _check_unique(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Check that column values are unique."""
        column_name = check_config["column"]
        values = [row.get(column_name) for row in data]
        unique_values = set(values)
        duplicates = len(values) - len(unique_values)

        passed = duplicates == 0

        message = (
            f"Column '{column_name}' values are unique"
            if passed
            else f"Column '{column_name}' has {duplicates} duplicate value(s)"
        )

        return CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details={"column": column_name, "duplicates": duplicates, "total_rows": len(data)},
        )

    def _check_column_type(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Check that column has expected type."""
        column_name = check_config["column"]
        expected_type_name = check_config["expected_type"]

        # Map type names to actual types
        type_map = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
        }
        expected_type = type_map.get(expected_type_name)

        if expected_type is None:
            return CheckResult(
                check_name=check_name,
                passed=False,
                message=f"Unknown expected type: {expected_type_name}",
                details={"column": column_name, "expected_type": expected_type_name},
            )

        violations = []
        for i, row in enumerate(data):
            value = row.get(column_name)
            if value is not None and not isinstance(value, expected_type):
                violations.append((i, value, type(value).__name__))

        passed = len(violations) == 0

        message = (
            f"Column '{column_name}' has correct type {expected_type_name}"
            if passed
            else f"Column '{column_name}' has {len(violations)} value(s) with wrong type"
        )

        return CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details={
                "column": column_name,
                "expected_type": expected_type_name,
                "violations": len(violations),
                "violation_details": violations[:10],
            },
        )

    def _check_row_count(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Check that row count is within range."""
        min_count = check_config.get("min")
        max_count = check_config.get("max")
        row_count = len(data)

        passed = True
        if min_count is not None and row_count < min_count:
            passed = False
        if max_count is not None and row_count > max_count:
            passed = False

        message = (
            f"Row count {row_count} is within range"
            if passed
            else f"Row count {row_count} is outside range [{min_count}, {max_count}]"
        )

        return CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details={
                "row_count": row_count,
                "min": min_count,
                "max": max_count,
            },
        )

    def _check_completeness(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Check that column completeness meets threshold."""
        column_name = check_config["column"]
        threshold = check_config["threshold"]

        total_rows = len(data)
        if total_rows == 0:
            completeness = 0.0
        else:
            non_null_count = sum(1 for row in data if row.get(column_name) is not None)
            completeness = non_null_count / total_rows

        passed = completeness >= threshold

        message = (
            f"Column '{column_name}' completeness {completeness:.2%} meets threshold {threshold:.2%}"
            if passed
            else f"Column '{column_name}' completeness {completeness:.2%} below threshold {threshold:.2%}"
        )

        return CheckResult(
            check_name=check_name,
            passed=passed,
            message=message,
            details={
                "column": column_name,
                "completeness": completeness,
                "threshold": threshold,
                "non_null_count": sum(1 for row in data if row.get(column_name) is not None),
                "total_rows": total_rows,
            },
        )

    def _check_custom(
        self, check_config: Dict[str, Any], data: List[Dict[str, Any]], check_name: str
    ) -> CheckResult:
        """Execute custom check function."""
        check_function = check_config.get("function")
        error_message = check_config.get("error_message", "Custom check failed")

        if check_function is None:
            return CheckResult(
                check_name=check_name,
                passed=False,
                message="Custom check function is None",
                details={},
            )

        try:
            passed = check_function(data)
            message = "Custom check passed" if passed else error_message
            return CheckResult(
                check_name=check_name,
                passed=passed,
                message=message,
                details={},
            )
        except Exception as e:
            return CheckResult(
                check_name=check_name,
                passed=False,
                message=f"Custom check raised exception: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
            )
