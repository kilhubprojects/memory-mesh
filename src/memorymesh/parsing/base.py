"""Abstract base class for all document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from memorymesh.core.models import ParsedDocument


class Parser(ABC):
    """Extracts plain text from a file of a specific type.

    Each concrete parser handles one or more file extensions.  The daemon
    selects the appropriate parser via :func:`~memorymesh.parsing.registry.get_parser`.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> frozenset[str]:
        """File extensions this parser can handle (e.g. ``frozenset({".txt"})``).

        Extensions must be lowercase and include the leading dot.
        """

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse *path* and return a :class:`~memorymesh.core.models.ParsedDocument`.

        Must never raise — failures are returned as a ``ParsedDocument`` with
        empty ``text`` and the error recorded in ``metadata["error"]``.

        Args:
            path: Absolute path to the file to parse.

        Returns:
            A :class:`~memorymesh.core.models.ParsedDocument` with extracted text.
        """
