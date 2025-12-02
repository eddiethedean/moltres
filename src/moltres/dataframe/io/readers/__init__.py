"""Format-specific readers for :class:`DataFrame` operations."""

from __future__ import annotations

from .csv_reader import read_csv, read_csv_stream
from .json_reader import read_json, read_json_stream, read_jsonl, read_jsonl_stream
from .parquet_reader import read_parquet, read_parquet_stream
from .schema_inference import apply_schema_to_rows, infer_schema_from_rows
from .text_reader import read_text, read_text_stream

__all__ = [
    "apply_schema_to_rows",
    "infer_schema_from_rows",
    "read_csv",
    "read_csv_stream",
    "read_json",
    "read_json_stream",
    "read_jsonl",
    "read_jsonl_stream",
    "read_parquet",
    "read_parquet_stream",
    "read_text",
    "read_text_stream",
]
