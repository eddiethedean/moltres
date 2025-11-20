"""Tests for IO write operations."""

from __future__ import annotations

import pytest

from moltres.io.write import insert_rows
from moltres.utils.exceptions import UnsupportedOperationError


def test_insert_rows_placeholder():
    """Test that insert_rows raises UnsupportedOperationError with helpful message."""
    with pytest.raises(UnsupportedOperationError) as exc_info:
        insert_rows("table_name", [{"id": 1}])

    error_msg = str(exc_info.value)
    assert "not implemented" in error_msg.lower() or "placeholder" in error_msg.lower()
    assert "TableHandle.insert()" in error_msg or "table().insert()" in error_msg.lower()
