"""Shared urllib HTTP helpers for MemoryMesh API connectors.

All functions use only the Python standard library ``urllib`` — no
third-party HTTP libraries are required or imported.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from loguru import logger


def api_get(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 30.0,
) -> Any:
    """HTTP GET and return parsed JSON, or ``None`` on error.

    Args:
        url: Full request URL (including any query parameters).
        headers: HTTP request headers dict.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON value (dict, list, etc.) or ``None`` on HTTP / network
        error.
    """
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        logger.warning(f"HTTP {exc.code} GET {url}: {exc.reason}")
        return None
    except Exception as exc:
        logger.warning(f"GET failed {url}: {exc}")
        return None


def api_post(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float = 30.0,
) -> Any:
    """HTTP POST with JSON body and return parsed JSON, or ``None`` on error.

    Args:
        url: Full request URL.
        headers: HTTP request headers dict.
        payload: JSON-serialisable request body.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON value or ``None`` on HTTP / network error.
    """
    body = json.dumps(payload).encode()
    hdrs = {**headers, "Content-Type": "application/json"}
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        logger.warning(f"HTTP {exc.code} POST {url}: {exc.reason}")
        return None
    except Exception as exc:
        logger.warning(f"POST failed {url}: {exc}")
        return None


def api_get_with_headers(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 30.0,
) -> tuple[Any, dict[str, str]]:
    """HTTP GET returning both parsed JSON and response headers.

    Useful for APIs that communicate state (rate limits, cursors, etc.) via
    response headers.

    Args:
        url: Full request URL.
        headers: HTTP request headers dict.
        timeout: Socket timeout in seconds.

    Returns:
        ``(data, resp_headers)`` where *data* is the parsed JSON value (or
        ``None`` on error) and *resp_headers* maps lowercase header names to
        their string values.
    """
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        return data, resp_headers
    except urllib.error.HTTPError as exc:
        logger.warning(f"HTTP {exc.code} GET {url}: {exc.reason}")
        hdrs: dict[str, str] = {}
        if exc.headers:
            hdrs = {k.lower(): v for k, v in exc.headers.items()}
        return None, hdrs
    except Exception as exc:
        logger.warning(f"GET failed {url}: {exc}")
        return None, {}
