"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Converts text strings to dense float vectors.

    All implementations must be safe to call from a single thread.  The daemon
    calls :meth:`embed_documents` in batches; the provider handles internal
    batching and device placement.

    Privacy invariant: providers must never log document text.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique identifier for the underlying model (used in collection metadata)."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output vectors."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document texts.

        Args:
            texts: Non-empty list of strings to embed.  Empty strings are
                allowed and will produce a valid (zero-ish) vector.

        Returns:
            List of float vectors, one per input text, same order.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        May use a different prompt prefix than :meth:`embed_documents` for
        asymmetric models (e.g. E5-family).

        Args:
            text: Query string.

        Returns:
            Float vector of length :attr:`embedding_dim`.
        """
