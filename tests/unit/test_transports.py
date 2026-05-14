"""Unit tests for transport utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from memorymesh.server.transports import run


def _make_ctx(tmp_path: Path, transport: str = "stdio") -> MagicMock:
    ctx = MagicMock()
    ctx.config.server.transport = transport
    ctx.config.server.http.host = "127.0.0.1"
    ctx.config.server.http.port = 8765
    ctx.config.storage.base_dir = tmp_path
    return ctx


def test_run_selects_stdio(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, transport="stdio")
    mcp = MagicMock()
    with patch("memorymesh.server.transports.run_stdio") as mock_stdio:
        run(mcp, ctx)
    mock_stdio.assert_called_once_with(mcp)


def test_run_selects_http(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path, transport="streamable-http")
    mcp = MagicMock()
    with patch("memorymesh.server.transports.run_http") as mock_http:
        run(mcp, ctx)
    mock_http.assert_called_once()
    args = mock_http.call_args
    assert args[0][0] is mcp
