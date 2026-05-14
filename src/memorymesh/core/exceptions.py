"""MemoryMesh exception hierarchy.

All public exceptions inherit from :class:`MemoryMeshError` so callers can
catch the entire family with a single ``except MemoryMeshError`` clause while
still being able to distinguish specific failure modes when needed.
"""

from __future__ import annotations


class MemoryMeshError(Exception):
    """Base class for all MemoryMesh exceptions."""


class ConfigError(MemoryMeshError):
    """Raised when the configuration file is missing, malformed, or invalid."""


class IndexingError(MemoryMeshError):
    """General failure during the indexing pipeline."""


class ParseError(IndexingError):
    """A file could not be parsed (corrupted, unsupported encoding, etc.)."""


class EmbeddingError(IndexingError):
    """The embedding provider failed to produce vectors."""


class StorageError(MemoryMeshError):
    """A read or write to the vector store or metadata store failed."""


class EmbeddingModelMismatchError(StorageError):
    """The collection was built with a different embedding model.

    The caller must run ``memorymesh reindex --all`` before the server can start.
    """


class SearchError(MemoryMeshError):
    """The search engine failed to execute a query."""


class DocumentNotFoundError(MemoryMeshError):
    """The requested document path is not inside any configured source."""


class DocumentTooLargeError(MemoryMeshError):
    """The document exceeds the ``max_bytes`` limit for ``get_document``."""


class PermissionDeniedError(PermissionError):
    """Raised when a client lacks the required permission for an operation.

    Args:
        client_id: The client that was denied.
        required: The permission level that was needed.
        reason: Human-readable explanation.
    """

    def __init__(self, client_id: str, required: object, reason: str = "") -> None:
        self.client_id = client_id
        self.required = required
        req_val = getattr(required, "value", str(required))
        msg = f"Client {client_id!r} does not have {req_val!r} permission" + (
            f": {reason}" if reason else ""
        )
        super().__init__(msg)


class RateLimitExceededError(Exception):
    """Raised when a client exceeds its token-bucket rate limit.

    Args:
        client_id: The client that was throttled.
        retry_after_s: Approximate seconds until the next token is available.
    """

    def __init__(self, client_id: str, retry_after_s: float) -> None:
        self.client_id = client_id
        self.retry_after_s = retry_after_s
        super().__init__(
            f"Rate limit exceeded for client {client_id!r}. Retry after {retry_after_s:.1f}s."
        )
