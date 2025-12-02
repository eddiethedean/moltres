"""Moltres integrations with popular frameworks."""

from __future__ import annotations

# SQLAlchemy integration (always available)
from . import data_quality
from . import sqlalchemy as sqlalchemy_integration

__all__ = [
    "sqlalchemy",
    "sqlalchemy_integration",
    "data_quality",
    "django",
    "fastapi",
    "streamlit",
    "airflow",
    "prefect",
    "pytest",
    "dbt",
]

# Optional Django integration
try:
    from . import django as django_integration

    __all__.append("django_integration")
except ImportError:
    django_integration = None  # type: ignore[assignment]

# Optional FastAPI integration
try:
    from . import fastapi as fastapi_integration

    __all__.append("fastapi_integration")
except ImportError:
    fastapi_integration = None  # type: ignore[assignment]

# Optional Streamlit integration
try:
    from . import streamlit as streamlit_integration

    __all__.append("streamlit_integration")
except ImportError:
    streamlit_integration = None  # type: ignore[assignment]

# Optional Airflow integration
try:
    from . import airflow as airflow_integration

    __all__.append("airflow_integration")
except ImportError:
    airflow_integration = None  # type: ignore[assignment]

# Optional Prefect integration
try:
    from . import prefect as prefect_integration

    __all__.append("prefect_integration")
except ImportError:
    prefect_integration = None  # type: ignore[assignment]

# Optional Pytest integration
try:
    from . import pytest as pytest_integration

    __all__.append("pytest_integration")
except ImportError:
    pytest_integration = None  # type: ignore[assignment]

# Optional dbt integration
try:
    from . import dbt as dbt_integration

    __all__.append("dbt_integration")
except ImportError:
    dbt_integration = None  # type: ignore[assignment]
