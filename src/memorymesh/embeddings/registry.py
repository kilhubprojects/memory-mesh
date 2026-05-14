"""Embedding provider registry.

Usage::

    from memorymesh.embeddings.registry import get_provider

    provider = get_provider(config.embeddings)
    vectors = provider.embed_documents(texts)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from memorymesh.core.exceptions import ConfigError
from memorymesh.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from memorymesh.core.models import EmbeddingsConfig


def get_provider(config: EmbeddingsConfig | None = None) -> EmbeddingProvider:
    """Return an :class:`~memorymesh.embeddings.base.EmbeddingProvider` for *config*.

    Currently only ``"sentence_transformers"`` is supported in the MVP.

    Args:
        config: Embedding settings.  Uses defaults (``all-MiniLM-L6-v2``) when ``None``.

    Returns:
        A ready-to-use :class:`~memorymesh.embeddings.base.EmbeddingProvider`.

    Raises:
        :class:`~memorymesh.core.exceptions.ConfigError`: For unknown provider keys.
    """
    from memorymesh.core.models import EmbeddingsConfig as _Cfg

    cfg = config or _Cfg()

    if cfg.provider == "sentence_transformers":
        from memorymesh.embeddings.sentence_transformers_provider import (
            SentenceTransformersProvider,
        )

        return SentenceTransformersProvider(config=cfg)

    raise ConfigError(
        f"Unknown embedding provider {cfg.provider!r}. "
        "Supported providers: ['sentence_transformers']"
    )
