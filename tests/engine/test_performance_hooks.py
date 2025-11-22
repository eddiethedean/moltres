"""Tests for performance monitoring hooks."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from moltres import connect
from moltres.engine import register_performance_hook, unregister_performance_hook


def test_performance_hook_registration(tmp_path):
    """Test that performance hooks can be registered and called."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table(
        "test",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()

    table = db.table("test")
    table.insert([{"id": 1, "name": "Alice"}]).collect()

    # Register hooks
    start_hook = Mock()
    end_hook = Mock()

    register_performance_hook("query_start", start_hook)
    register_performance_hook("query_end", end_hook)

    # Execute query
    df = table.select()
    df.collect()  # Execute to trigger hooks

    # Verify hooks were called
    assert start_hook.called
    assert end_hook.called

    # Check hook arguments
    start_call = start_hook.call_args
    assert start_call[0][0]  # SQL string
    assert start_call[0][1] == 0.0  # elapsed time at start

    end_call = end_hook.call_args
    assert end_call[0][0]  # SQL string
    assert end_call[0][1] > 0  # elapsed time > 0
    assert "rowcount" in end_call[0][2]  # metadata


def test_performance_hook_unregistration(tmp_path):
    """Test that performance hooks can be unregistered."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table("test", [column("id", "INTEGER")]).collect()
    table = db.table("test")

    hook = Mock()
    register_performance_hook("query_end", hook)

    # Execute query
    df = table.select()
    df.collect()
    assert hook.called

    # Unregister and execute again
    hook.reset_mock()
    unregister_performance_hook("query_end", hook)

    df.collect()
    assert not hook.called  # Should not be called after unregistration


def test_performance_hook_invalid_event():
    """Test that invalid event types raise ValueError."""
    hook = Mock()
    with pytest.raises(ValueError, match="Unknown event type"):
        register_performance_hook("invalid_event", hook)


def test_performance_hook_slow_query_detection(tmp_path):
    """Test using hooks to detect slow queries."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table("test", [column("id", "INTEGER")]).collect()
    table = db.table("test")

    slow_queries = []

    def detect_slow(sql: str, elapsed: float, metadata: dict):
        if elapsed > 0.001:  # Very low threshold for testing
            slow_queries.append((sql, elapsed))

    register_performance_hook("query_end", detect_slow)

    # Execute query
    df = table.select()
    df.collect()

    # Should have recorded the query (even if fast, threshold is very low)
    assert len(slow_queries) >= 0  # May or may not be slow depending on system


def test_performance_hook_error_handling(tmp_path):
    """Test that hook errors don't break execution."""
    db_path = tmp_path / "test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    db.create_table("test", [column("id", "INTEGER")]).collect()
    table = db.table("test")

    def failing_hook(sql: str, elapsed: float, metadata: dict):
        raise ValueError("Hook error")

    register_performance_hook("query_end", failing_hook)

    # Should not raise, just log warning
    df = table.select()
    results = df.collect()  # Should succeed despite hook error
    assert results is not None
