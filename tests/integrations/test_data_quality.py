"""Comprehensive tests for data quality framework.

This module tests all data quality check features:
- DataQualityCheck factory methods
- QualityChecker execution
- QualityReport generation
- All check types (not_null, range, unique, type, row_count, completeness, custom)
"""

from __future__ import annotations

import pytest

from moltres import column, connect
from moltres.integrations.data_quality import (
    DataQualityCheck,
    QualityChecker,
)
from moltres.io.records import Records


@pytest.fixture
def db(tmp_path):
    """Create a test database."""
    db_path = tmp_path / "test_quality.db"
    db = connect(f"sqlite:///{db_path}")
    yield db
    db.close()


@pytest.fixture
def users_table(db):
    """Create a users table with test data."""
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("age", "INTEGER"),
            column("score", "REAL"),
        ],
    ).collect()

    Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 25, "score": 85.5},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 30, "score": 90.0},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35, "score": 75.5},
        ],
        _database=db,
    ).insert_into("users")

    return db.table("users").select()


class TestDataQualityCheck:
    """Test DataQualityCheck factory methods."""

    def test_column_not_null(self):
        """Test column_not_null check creation."""
        check = DataQualityCheck.column_not_null("email")
        assert check["type"] == "not_null"
        assert check["column"] == "email"
        assert check["name"] == "not_null_email"

    def test_column_not_null_with_name(self):
        """Test column_not_null with custom name."""
        check = DataQualityCheck.column_not_null("email", check_name="custom_check")
        assert check["name"] == "custom_check"

    def test_column_range(self):
        """Test column_range check creation."""
        check = DataQualityCheck.column_range("age", min=0, max=150)
        assert check["type"] == "range"
        assert check["column"] == "age"
        assert check["min"] == 0
        assert check["max"] == 150

    def test_column_range_min_only(self):
        """Test column_range with only min value."""
        check = DataQualityCheck.column_range("age", min=0)
        assert check["min"] == 0
        assert check["max"] is None

    def test_column_range_max_only(self):
        """Test column_range with only max value."""
        check = DataQualityCheck.column_range("age", max=150)
        assert check["min"] is None
        assert check["max"] == 150

    def test_column_unique(self):
        """Test column_unique check creation."""
        check = DataQualityCheck.column_unique("email")
        assert check["type"] == "unique"
        assert check["column"] == "email"

    def test_column_type(self):
        """Test column_type check creation."""
        check = DataQualityCheck.column_type("age", int)
        assert check["type"] == "column_type"
        assert check["column"] == "age"
        assert check["expected_type"] == "int"

    def test_row_count(self):
        """Test row_count check creation."""
        check = DataQualityCheck.row_count(min=1, max=10)
        assert check["type"] == "row_count"
        assert check["min"] == 1
        assert check["max"] == 10

    def test_column_completeness(self):
        """Test column_completeness check creation."""
        check = DataQualityCheck.column_completeness("email", threshold=0.9)
        assert check["type"] == "completeness"
        assert check["column"] == "email"
        assert check["threshold"] == 0.9

    def test_custom_check(self):
        """Test custom check creation."""

        def custom_func(data):
            return len(data) > 0

        check = DataQualityCheck.custom(custom_func, check_name="custom_test")
        assert check["type"] == "custom"
        assert check["name"] == "custom_test"
        assert callable(check["function"])


class TestQualityChecker:
    """Test QualityChecker execution."""

    def test_check_not_null_passed(self, users_table):
        """Test not_null check that passes."""
        checker = QualityChecker()
        checks = [DataQualityCheck.column_not_null("email")]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert len(report.results) == 1
        assert report.results[0].passed
        assert report.results[0].check_name == "not_null_email"
        assert report.total_rows == 3

    def test_check_not_null_failed(self, db):
        """Test not_null check that fails."""
        db.create_table(
            "users_with_nulls",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": None},
            ],
            _database=db,
        ).insert_into("users_with_nulls")

        df = db.table("users_with_nulls").select()
        checker = QualityChecker()
        checks = [DataQualityCheck.column_not_null("email")]
        report = checker.check(df, checks)

        assert report.overall_status == "failed"
        assert len(report.results) == 1
        assert not report.results[0].passed
        assert "null value" in report.results[0].message.lower()
        assert report.results[0].details["null_count"] == 1

    def test_check_range_passed(self, users_table):
        """Test range check that passes."""
        checker = QualityChecker()
        checks = [DataQualityCheck.column_range("age", min=0, max=150)]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert report.results[0].passed

    def test_check_range_failed(self, db):
        """Test range check that fails."""
        db.create_table(
            "users_invalid_age",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 25},
                {"id": 2, "name": "Bob", "age": 200},  # Outside range
            ],
            _database=db,
        ).insert_into("users_invalid_age")

        df = db.table("users_invalid_age").select()
        checker = QualityChecker()
        checks = [DataQualityCheck.column_range("age", min=0, max=150)]
        report = checker.check(df, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed
        assert report.results[0].details["violations"] == 1

    def test_check_unique_passed(self, users_table):
        """Test unique check that passes."""
        checker = QualityChecker()
        checks = [DataQualityCheck.column_unique("email")]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert report.results[0].passed

    def test_check_unique_failed(self, db):
        """Test unique check that fails."""
        db.create_table(
            "users_duplicates",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": "alice@example.com"},
                {"id": 2, "email": "alice@example.com"},  # Duplicate
            ],
            _database=db,
        ).insert_into("users_duplicates")

        df = db.table("users_duplicates").select()
        checker = QualityChecker()
        checks = [DataQualityCheck.column_unique("email")]
        report = checker.check(df, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed
        assert report.results[0].details["duplicates"] == 1

    def test_check_row_count_passed(self, users_table):
        """Test row_count check that passes."""
        checker = QualityChecker()
        checks = [DataQualityCheck.row_count(min=1, max=10)]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert report.results[0].passed

    def test_check_row_count_failed(self, users_table):
        """Test row_count check that fails."""
        checker = QualityChecker()
        checks = [DataQualityCheck.row_count(min=5, max=10)]  # Only 3 rows
        report = checker.check(users_table, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed

    def test_check_completeness_passed(self, users_table):
        """Test completeness check that passes."""
        checker = QualityChecker()
        checks = [DataQualityCheck.column_completeness("email", threshold=0.9)]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert report.results[0].passed
        assert report.results[0].details["completeness"] == 1.0  # 100% complete

    def test_check_completeness_failed(self, db):
        """Test completeness check that fails."""
        db.create_table(
            "users_incomplete",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": "alice@example.com"},
                {"id": 2, "email": None},  # Missing
                {"id": 3, "email": None},  # Missing
            ],
            _database=db,
        ).insert_into("users_incomplete")

        df = db.table("users_incomplete").select()
        checker = QualityChecker()
        checks = [DataQualityCheck.column_completeness("email", threshold=0.9)]
        report = checker.check(df, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed
        assert report.results[0].details["completeness"] < 0.9

    def test_check_custom_passed(self, users_table):
        """Test custom check that passes."""

        def custom_check(data):
            return len(data) > 0

        checker = QualityChecker()
        checks = [DataQualityCheck.custom(custom_check)]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert report.results[0].passed

    def test_check_custom_failed(self, users_table):
        """Test custom check that fails."""

        def custom_check(data):
            return len(data) > 10  # Only 3 rows

        checker = QualityChecker()
        checks = [DataQualityCheck.custom(custom_check)]
        report = checker.check(users_table, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed

    def test_multiple_checks_all_passed(self, users_table):
        """Test multiple checks all passing."""
        checker = QualityChecker()
        checks = [
            DataQualityCheck.column_not_null("email"),
            DataQualityCheck.column_range("age", min=0, max=150),
            DataQualityCheck.column_unique("email"),
        ]
        report = checker.check(users_table, checks)

        assert report.overall_status == "passed"
        assert len(report.results) == 3
        assert all(r.passed for r in report.results)

    def test_multiple_checks_some_failed(self, db):
        """Test multiple checks with some failures."""
        db.create_table(
            "users_mixed",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": "alice@example.com", "age": 25},
                {"id": 2, "email": None, "age": 200},  # Both invalid
            ],
            _database=db,
        ).insert_into("users_mixed")

        df = db.table("users_mixed").select()
        checker = QualityChecker()
        checks = [
            DataQualityCheck.column_not_null("email"),
            DataQualityCheck.column_range("age", min=0, max=150),
        ]
        report = checker.check(df, checks)

        assert report.overall_status == "failed"
        assert len(report.failed_checks) >= 1
        assert len(report.passed_checks) < len(report.results)

    def test_fail_fast(self, db):
        """Test fail_fast option stops after first failure."""
        db.create_table(
            "users_multi_fail",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": None, "age": 200},  # Both invalid
            ],
            _database=db,
        ).insert_into("users_multi_fail")

        df = db.table("users_multi_fail").select()
        checker = QualityChecker(fail_fast=True)
        checks = [
            DataQualityCheck.column_not_null("email"),  # This will fail
            DataQualityCheck.column_range("age", min=0, max=150),  # This won't run
        ]
        report = checker.check(df, checks)

        assert report.overall_status == "failed"
        # With fail_fast, only the first check should run
        assert len(report.results) == 1
        assert not report.results[0].passed

    def test_execution_time_recorded(self, users_table):
        """Test that execution time is recorded in report."""
        checker = QualityChecker()
        checks = [DataQualityCheck.column_not_null("email")]
        report = checker.check(users_table, checks)

        assert report.execution_time_seconds is not None
        assert report.execution_time_seconds >= 0

    def test_unknown_check_type(self, users_table):
        """Test handling of unknown check type."""
        checker = QualityChecker()
        checks = [{"type": "unknown_check", "name": "test"}]
        report = checker.check(users_table, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed
        assert "Unknown check type" in report.results[0].message

    def test_check_with_exception(self, users_table):
        """Test handling of exception in check execution."""

        def failing_check(data):
            raise ValueError("Check failed")

        checker = QualityChecker()
        checks = [DataQualityCheck.custom(failing_check)]
        report = checker.check(users_table, checks)

        assert report.overall_status == "failed"
        assert not report.results[0].passed
        assert "exception" in report.results[0].message.lower()


class TestQualityReport:
    """Test QualityReport functionality."""

    def test_report_to_dict(self, users_table):
        """Test converting report to dictionary."""
        checker = QualityChecker()
        checks = [
            DataQualityCheck.column_not_null("email"),
            DataQualityCheck.column_range("age", min=0, max=150),
        ]
        report = checker.check(users_table, checks)

        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict["overall_status"] == "passed"
        assert report_dict["total_checks"] == 2
        assert report_dict["passed_checks"] == 2
        assert report_dict["failed_checks"] == 0
        assert "results" in report_dict
        assert len(report_dict["results"]) == 2

    def test_report_passed_property(self, users_table):
        """Test report.passed property."""
        checker = QualityChecker()
        checks = [DataQualityCheck.column_not_null("email")]
        report = checker.check(users_table, checks)

        assert report.passed
        assert report.overall_status == "passed"

    def test_report_failed_property(self, db):
        """Test report.passed property with failures."""
        db.create_table(
            "users_fail",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": None},
            ],
            _database=db,
        ).insert_into("users_fail")

        df = db.table("users_fail").select()
        checker = QualityChecker()
        checks = [DataQualityCheck.column_not_null("email")]
        report = checker.check(df, checks)

        assert not report.passed
        assert report.overall_status == "failed"

    def test_report_failed_checks(self, db):
        """Test report.failed_checks property."""
        db.create_table(
            "users_mixed_results",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        Records(
            _data=[
                {"id": 1, "email": "alice@example.com", "age": 25},
                {"id": 2, "email": None, "age": 200},
            ],
            _database=db,
        ).insert_into("users_mixed_results")

        df = db.table("users_mixed_results").select()
        checker = QualityChecker()
        checks = [
            DataQualityCheck.column_not_null("email"),
            DataQualityCheck.column_range("age", min=0, max=150),
        ]
        report = checker.check(df, checks)

        failed = report.failed_checks
        assert len(failed) >= 1
        assert all(not check.passed for check in failed)

    def test_report_passed_checks(self, users_table):
        """Test report.passed_checks property."""
        checker = QualityChecker()
        checks = [
            DataQualityCheck.column_not_null("email"),
            DataQualityCheck.column_range("age", min=0, max=150),
        ]
        report = checker.check(users_table, checks)

        passed = report.passed_checks
        assert len(passed) == 2
        assert all(check.passed for check in passed)
