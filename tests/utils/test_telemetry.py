"""Tests for structured logging and metrics utilities."""

from __future__ import annotations

import json
import logging
from unittest.mock import Mock, patch

import pytest

from moltres.utils.telemetry import (
    MetricsCollector,
    StructuredLogger,
    create_performance_hook_from_logger,
    get_metrics_collector,
    get_structured_logger,
)


class TestStructuredLogger:
    """Tests for StructuredLogger class."""

    def test_init_with_default_logger(self):
        """Test initialization with default logger."""
        logger = StructuredLogger()
        assert logger.logger is not None
        assert isinstance(logger.logger, logging.Logger)

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        custom_logger = logging.getLogger("custom")
        logger = StructuredLogger(logger_instance=custom_logger)
        assert logger.logger is custom_logger

    def test_log_query_start_basic(self):
        """Test logging query start with basic parameters."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        logger.log_query_start("SELECT * FROM users")
        assert mock_logger.debug.called
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["event"] == "query_start"
        assert "SELECT * FROM users" in event["sql"]
        assert "timestamp" in event

    def test_log_query_start_with_params(self):
        """Test logging query start with parameters."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        params = {"id": 1, "name": "test"}
        logger.log_query_start("SELECT * FROM users WHERE id = :id", params=params)
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["params"] == params

    def test_log_query_start_with_metadata(self):
        """Test logging query start with metadata."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        metadata = {"source": "api", "user_id": 123}
        logger.log_query_start("SELECT * FROM users", metadata=metadata)
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["source"] == "api"
        assert event["user_id"] == 123

    def test_log_query_start_truncates_long_sql(self):
        """Test that long SQL queries are truncated."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        long_sql = "SELECT " + "x" * 1000
        logger.log_query_start(long_sql)
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert len(event["sql"]) == 500

    def test_log_query_end_basic(self):
        """Test logging query end with basic parameters."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        logger.log_query_end("SELECT * FROM users", duration=0.5)
        assert mock_logger.debug.called
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["event"] == "query_end"
        assert event["duration"] == 0.5

    def test_log_query_end_with_rowcount(self):
        """Test logging query end with row count."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        logger.log_query_end("SELECT * FROM users", duration=0.5, rowcount=42)
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["rowcount"] == 42

    def test_log_query_end_logging_levels(self):
        """Test that query end uses appropriate logging levels based on duration."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)

        # Short duration (< 1s) -> debug
        logger.log_query_end("SELECT * FROM users", duration=0.5)
        assert mock_logger.debug.called
        mock_logger.reset_mock()

        # Medium duration (1-5s) -> info
        logger.log_query_end("SELECT * FROM users", duration=2.0)
        assert mock_logger.info.called
        mock_logger.reset_mock()

        # Long duration (> 5s) -> warning
        logger.log_query_end("SELECT * FROM users", duration=6.0)
        assert mock_logger.warning.called

    def test_log_query_error_basic(self):
        """Test logging query error with basic parameters."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        error = ValueError("Test error")
        logger.log_query_error("SELECT * FROM users", error)
        assert mock_logger.error.called
        call_args = mock_logger.error.call_args[0][0]
        event = json.loads(call_args)
        assert event["event"] == "query_error"
        assert event["error_type"] == "ValueError"
        assert event["error_message"] == "Test error"

    def test_log_query_error_with_duration(self):
        """Test logging query error with duration."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        error = ValueError("Test error")
        logger.log_query_error("SELECT * FROM users", error, duration=0.3)
        call_args = mock_logger.error.call_args[0][0]
        event = json.loads(call_args)
        assert event["duration"] == 0.3

    def test_log_query_error_with_exc_info(self):
        """Test that log_query_error includes exception info."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        error = ValueError("Test error")
        logger.log_query_error("SELECT * FROM users", error)
        # Check that exc_info is passed
        assert mock_logger.error.call_args[1]["exc_info"] == error

    def test_log_connection_event_basic(self):
        """Test logging connection event with basic parameters."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        logger.log_connection_event("connect")
        assert mock_logger.debug.called
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["event"] == "connection"
        assert event["connection_event"] == "connect"

    def test_log_connection_event_with_metadata(self):
        """Test logging connection event with metadata."""
        mock_logger = Mock()
        logger = StructuredLogger(logger_instance=mock_logger)
        metadata = {"pool_size": 10, "database": "test"}
        logger.log_connection_event("pool_checkout", metadata=metadata)
        call_args = mock_logger.debug.call_args[0][0]
        event = json.loads(call_args)
        assert event["pool_size"] == 10
        assert event["database"] == "test"


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_init(self):
        """Test initialization."""
        collector = MetricsCollector()
        assert collector._query_count == 0
        assert collector._query_duration_sum == 0.0
        assert collector._query_duration_max == 0.0
        assert collector._error_count == 0
        assert collector._connection_count == 0

    def test_record_query_success(self):
        """Test recording successful query."""
        collector = MetricsCollector()
        collector.record_query(0.5, success=True)
        assert collector._query_count == 1
        assert collector._query_duration_sum == 0.5
        assert collector._query_duration_max == 0.5
        assert collector._error_count == 0

    def test_record_query_failure(self):
        """Test recording failed query."""
        collector = MetricsCollector()
        collector.record_query(0.3, success=False)
        assert collector._query_count == 1
        assert collector._error_count == 1

    def test_record_query_multiple(self):
        """Test recording multiple queries."""
        collector = MetricsCollector()
        collector.record_query(0.5, success=True)
        collector.record_query(1.0, success=True)
        collector.record_query(0.3, success=False)
        assert collector._query_count == 3
        assert collector._query_duration_sum == 1.8
        assert collector._query_duration_max == 1.0
        assert collector._error_count == 1

    def test_record_connection(self):
        """Test recording connection event."""
        collector = MetricsCollector()
        collector.record_connection()
        assert collector._connection_count == 1
        collector.record_connection()
        assert collector._connection_count == 2

    def test_get_metrics_empty(self):
        """Test getting metrics when no queries recorded."""
        collector = MetricsCollector()
        metrics = collector.get_metrics()
        assert metrics["query_count"] == 0
        assert metrics["query_duration_avg"] == 0.0
        assert metrics["query_duration_max"] == 0.0
        assert metrics["error_count"] == 0
        assert metrics["error_rate"] == 0.0
        assert metrics["connection_count"] == 0

    def test_get_metrics_with_data(self):
        """Test getting metrics with recorded data."""
        collector = MetricsCollector()
        collector.record_query(0.5, success=True)
        collector.record_query(1.0, success=True)
        collector.record_query(0.3, success=False)
        collector.record_connection()
        metrics = collector.get_metrics()
        assert metrics["query_count"] == 3
        assert metrics["query_duration_avg"] == pytest.approx(0.6, abs=0.01)
        assert metrics["query_duration_max"] == 1.0
        assert metrics["error_count"] == 1
        assert metrics["error_rate"] == pytest.approx(1.0 / 3.0, abs=0.01)
        assert metrics["connection_count"] == 1

    def test_reset(self):
        """Test resetting metrics."""
        collector = MetricsCollector()
        collector.record_query(0.5, success=True)
        collector.record_connection()
        collector.reset()
        assert collector._query_count == 0
        assert collector._query_duration_sum == 0.0
        assert collector._query_duration_max == 0.0
        assert collector._error_count == 0
        assert collector._connection_count == 0
        metrics = collector.get_metrics()
        assert metrics["query_count"] == 0


class TestGlobalFunctions:
    """Tests for global utility functions."""

    def test_get_structured_logger(self):
        """Test getting global structured logger."""
        logger = get_structured_logger()
        assert isinstance(logger, StructuredLogger)

    def test_get_metrics_collector(self):
        """Test getting global metrics collector."""
        collector = get_metrics_collector()
        assert isinstance(collector, MetricsCollector)

    def test_create_performance_hook_query_start(self):
        """Test performance hook for query_start event."""
        hook = create_performance_hook_from_logger()
        with patch("moltres.utils.telemetry._structured_logger") as mock_logger:
            metadata = {"event": "query_start", "params": {"id": 1}}
            hook("SELECT * FROM users", 0.0, metadata)
            mock_logger.log_query_start.assert_called_once()

    def test_create_performance_hook_query_end(self):
        """Test performance hook for query_end event."""
        hook = create_performance_hook_from_logger()
        with patch("moltres.utils.telemetry._structured_logger") as mock_logger:
            with patch("moltres.utils.telemetry._metrics_collector") as mock_collector:
                metadata = {"event": "query_end", "rowcount": 10}
                hook("SELECT * FROM users", 0.5, metadata)
                mock_logger.log_query_end.assert_called_once()
                mock_collector.record_query.assert_called_once_with(0.5, success=True)

    def test_create_performance_hook_query_error(self):
        """Test performance hook for query_error event."""
        hook = create_performance_hook_from_logger()
        with patch("moltres.utils.telemetry._structured_logger") as mock_logger:
            with patch("moltres.utils.telemetry._metrics_collector") as mock_collector:
                error = ValueError("Test error")
                metadata = {"event": "query_error", "error": error}
                hook("SELECT * FROM users", 0.3, metadata)
                mock_logger.log_query_error.assert_called_once()
                mock_collector.record_query.assert_called_once_with(0.3, success=False)

    def test_create_performance_hook_metadata_filtering(self):
        """Test that performance hook filters metadata correctly."""
        hook = create_performance_hook_from_logger()
        with patch("moltres.utils.telemetry._structured_logger") as mock_logger:
            metadata = {
                "event": "query_start",
                "params": {"id": 1},
                "custom_field": "value",
                "another_field": 42,
            }
            hook("SELECT * FROM users", 0.0, metadata)
            call_kwargs = mock_logger.log_query_start.call_args[1]
            # Should exclude 'event' and 'params' from metadata
            assert "event" not in call_kwargs.get("metadata", {})
            assert "params" not in call_kwargs.get("metadata", {})
            # Should include custom fields
            assert call_kwargs.get("metadata", {}).get("custom_field") == "value"
            assert call_kwargs.get("metadata", {}).get("another_field") == 42
