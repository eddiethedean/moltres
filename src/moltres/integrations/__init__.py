"""Moltres integrations with popular frameworks."""

from __future__ import annotations

__all__ = ["fastapi", "django", "data_quality"]

# Optional Django integration - only import if available
try:
    from . import django as django_integration

    __all__.append("django_integration")
except ImportError:
    django_integration = None  # type: ignore[assignment]

# Optional Streamlit integration - only import if available
try:
    from . import streamlit as streamlit_integration

    __all__.append("streamlit_integration")
except ImportError:
    streamlit_integration = None  # type: ignore[assignment]

# Optional Airflow integration - only import if available
try:
    from . import airflow as airflow_integration

    __all__.append("airflow_integration")
except ImportError:
    airflow_integration = None  # type: ignore[assignment]

# Optional Prefect integration - only import if available
try:
    from . import prefect as prefect_integration

    __all__.append("prefect_integration")
except ImportError:
    prefect_integration = None  # type: ignore[assignment]

# Data quality framework (always available)
from . import data_quality

# Optional Pytest integration - only import if available
try:
    from . import pytest as pytest_integration

    __all__.append("pytest_integration")
except ImportError:
    pytest_integration = None  # type: ignore[assignment]

# Optional dbt integration - only import if available
try:
    from . import dbt as dbt_integration

    __all__.append("dbt_integration")
except ImportError:
    dbt_integration = None  # type: ignore[assignment]
