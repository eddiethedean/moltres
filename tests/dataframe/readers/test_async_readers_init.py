"""Tests for async_readers_init module."""

from __future__ import annotations


from moltres.dataframe.io.readers import async_readers_init


def test_async_readers_init_module():
    """Test that async_readers_init module can be imported."""
    # The module is just an empty init file, but we can test it exists
    assert async_readers_init is not None
    assert hasattr(async_readers_init, "__all__")
    assert async_readers_init.__all__ == []
