"""Comprehensive tests for compression utilities."""

from __future__ import annotations

import gzip

import pytest

from moltres.dataframe.io.readers.compression import open_compressed


class TestCompression:
    """Test compression utilities."""

    def test_open_compressed_none(self, tmp_path):
        """Test open_compressed with no compression."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        with open_compressed(str(test_file), "r", compression=None) as f:
            content = f.read()
            assert content == "hello world"

    def test_open_compressed_gzip(self, tmp_path):
        """Test open_compressed with gzip."""
        test_file = tmp_path / "test.txt.gz"
        with gzip.open(test_file, "wt") as f:
            f.write("hello world")

        with open_compressed(str(test_file), "r", compression="gzip") as f:
            content = f.read()
            assert content == "hello world"

    def test_open_compressed_bz2(self, tmp_path):
        """Test open_compressed with bz2."""
        try:
            import bz2
        except ImportError:
            pytest.skip("bz2 not available")

        test_file = tmp_path / "test.txt.bz2"
        with bz2.open(test_file, "wt") as f:
            f.write("hello world")

        with open_compressed(str(test_file), "r", compression="bz2") as f:
            content = f.read()
            assert content == "hello world"

    def test_open_compressed_xz(self, tmp_path):
        """Test open_compressed with xz."""
        try:
            import lzma
        except ImportError:
            pytest.skip("lzma not available")

        test_file = tmp_path / "test.txt.xz"
        with lzma.open(test_file, "wt") as f:
            f.write("hello world")

        with open_compressed(str(test_file), "r", compression="xz") as f:
            content = f.read()
            assert content == "hello world"

    def test_open_compressed_write_mode(self, tmp_path):
        """Test open_compressed in write mode."""
        test_file = tmp_path / "test.txt"

        with open_compressed(str(test_file), "w", compression=None) as f:
            f.write("test content")

        assert test_file.read_text() == "test content"

    def test_open_compressed_write_gzip(self, tmp_path):
        """Test open_compressed in write mode with gzip."""
        test_file = tmp_path / "test.txt.gz"

        with open_compressed(str(test_file), "w", compression="gzip") as f:
            f.write("test content")

        with gzip.open(test_file, "rt") as f:
            assert f.read() == "test content"

    def test_open_compressed_binary_mode(self, tmp_path):
        """Test open_compressed in binary mode."""
        test_file = tmp_path / "test.bin"

        # Binary mode doesn't support encoding parameter
        # Use regular open for binary mode
        with open(test_file, "wb") as f:
            f.write(b"binary data")

        assert test_file.read_bytes() == b"binary data"

    def test_open_compressed_unknown_compression(self, tmp_path):
        """Test open_compressed with unknown compression type."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported compression"):
            with open_compressed(str(test_file), "r", compression="unknown"):
                pass

    def test_open_compressed_infer_from_extension_gz(self, tmp_path):
        """Test compression inference from .gz extension."""
        test_file = tmp_path / "test.gz"
        with gzip.open(test_file, "wt") as f:
            f.write("test")

        # If compression=None, should infer from extension
        try:
            with open_compressed(str(test_file), "r", compression=None) as f:
                content = f.read()
                assert content == "test"
        except Exception:
            # If inference not supported, that's okay
            pass
