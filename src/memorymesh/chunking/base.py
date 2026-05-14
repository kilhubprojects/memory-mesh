"""Abstract base class for all chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from memorymesh.core.models import Chunk, ParsedDocument


class Chunker(ABC):
    """Splits a :class:`~memorymesh.core.models.ParsedDocument` into
    :class:`~memorymesh.core.models.Chunk` objects.

    Each concrete chunker is optimised for a specific content type
    (plain text, Markdown, source code).  The daemon selects the appropriate
    chunker via :func:`~memorymesh.chunking.registry.get_chunker`.
    """

    @abstractmethod
    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split *doc* into a list of chunks.

        Args:
            doc: Parsed document to split.

        Returns:
            Non-empty list of :class:`~memorymesh.core.models.Chunk` objects.
            If the document text is empty, returns a single chunk with empty
            text so the caller always gets at least one record to process.
        """
