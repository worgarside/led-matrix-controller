"""Helper functions."""

from __future__ import annotations

import re


def to_kebab_case(string: str) -> str:
    """Convert string to kebab case."""
    return _remove_multiple_hyphens(
        re.sub(r"[^a-z]", "-", camel_to_kebab_case(string), flags=re.IGNORECASE).lower()
    )


def camel_to_kebab_case(string: str) -> str:
    """Convert camel case string to kebab case."""
    return _remove_multiple_hyphens(re.sub(r"(?<!^)(?=[A-Z])", "-", string).lower())


def _remove_multiple_hyphens(string: str) -> str:
    """Remove multiple hyphens."""
    return re.sub(r"-+", "-", string).strip("-")
