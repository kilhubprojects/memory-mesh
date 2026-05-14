"""MemoryMesh — local MCP hub for personal data."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version("memorymesh-mcp")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.8.0"

__all__ = ["__version__"]
