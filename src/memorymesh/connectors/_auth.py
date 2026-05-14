"""Shared HTTP authentication header helpers for MemoryMesh connectors.

Extracted so connectors do not duplicate ``Bearer`` / ``Basic`` header
construction.  All functions return a plain ``dict`` that can be merged
with other request headers.
"""

from __future__ import annotations

import base64


def bearer_header(token: str) -> dict[str, str]:
    """Return an ``Authorization: Bearer <token>`` header dict.

    Args:
        token: The raw bearer token value.

    Returns:
        A single-entry dict ``{"Authorization": "Bearer <token>"}``.
    """
    return {"Authorization": f"Bearer {token}"}


def basic_header(user: str, password: str) -> dict[str, str]:
    """Return an ``Authorization: Basic <b64>`` header dict.

    Args:
        user: The username or email address.
        password: The password or API token.

    Returns:
        A single-entry dict ``{"Authorization": "Basic <b64(user:password)>"}``.
    """
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}
